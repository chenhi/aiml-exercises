import datetime, copy
import torch
from torch import nn
import matplotlib.pyplot as plt

from rlbase import log, smoothing, PrototypeQFunction, DeepRL, TensorMDP


class MCTSQFunction(PrototypeQFunction):
    def __init__(self, mdp: TensorMDP, heuristic_model: nn.Module, model_args={}, device="cpu", dict_device="cpu"):
        self.device = device
        self.dict_device = dict_device
        self.mdp = mdp
        self.heuristic_model = heuristic_model
        self.model_args = model_args

        if heuristic_model != None:
            self.h = heuristic_model(**model_args, **self.mdp.nn_args).to(self.device)
            self.h.eval()
        else:
            self.h = None

        # Keys: hashable states.  Values: tensor with shape of actions
        self.n = {}
        self.w = {}         # Note this is the total rewards, not the total wins
        self.p = {}

    # Helper function for turning dictionary elements into tensors
    def ucb_get_parts(self, state: torch.Tensor, get_p = True):
        
        # Initialize
        q, n = torch.zeros((state.size(0),) + self.mdp.action_shape, device=self.device), torch.zeros((state.size(0),) + self.mdp.action_shape, device=self.device)
        if get_p:
            p = torch.zeros((state.size(0),) + self.mdp.action_shape, device=self.device) 
        
        # Handle degenerate case
        if state.size(0) == 0:
            if get_p:
                return q, n, p
            else:
                return q, n
        
        # Get
        for i in range(state.size(0)):
            statehash = self.mdp.state_to_hashable(state[i])
            if statehash in self.n:
                n[i] = self.n[statehash].to(self.device)
                q[i] = self.w[statehash].to(self.device) / n[i]
                if get_p:
                    p[i] = self.p[statehash].to(self.device)
        n_tot = n.flatten(1, -1).sum(dim=1).reshape((state.size(0),) + self.mdp.action_projshape)
        if get_p:
            return q.nan_to_num(0), torch.sqrt(1 + n_tot) / (1 + n), p
        else:
            return q.nan_to_num(0), torch.sqrt(1 + n_tot) / (1 + n)

    # Gets a probability vector or the maximum
    # Heuristic parameter may also be a tensor
    def ucb_get(self, state: torch.Tensor, heuristic_parameter = 1.):
        q, n_ratio, p = self.ucb_get_parts(state, get_p=True)

        # Exploration vs. exploitation: the second term dominates when unexplored, then the first term dominates when more explored
        return q + heuristic_parameter * p * n_ratio        # TODO q is a value so can be negative... ???? also do we want "Get" to return N or ...?
    

    # Will either:
    # 1) Do some searches, then ???? # TODO default 1024 maybe should be 0??
    # 2) Do no searches, and use only the heuristic function
    def get(self, state: torch.Tensor, action = None, heuristic_parameter= 1., temperature=1., num_searches = 1024):
        #q, n_ratio = self.ucb_get_parts(state, get_p=False)
        #output = q + self.mdp.masked_softmax(self.h(state), state) * heuristic_parameter * n_ratio
        #output = self.mdp.masked_softmax(self.h(state), state)
        if num_searches > 0:
            self.search(state, ucb_parameter=heuristic_parameter, num=num_searches, temperature=temperature)
            q, _, p = self.ucb_get_parts(state)
            # Omit the visit count, we don't want any exploration
            vals = q + heuristic_parameter * p
        else:
            vals = self.h(state)
        if action == None:
            return vals
        else:
            return (action * vals).sum(tuple(range(1, action.dim())))


    def policy(self, state: torch.Tensor, heuristic_parameter = 1., temperature=1., stochastic=False, valid_filter=True, num_searches=1024):
        if stochastic:
            return self.mdp.get_random_action_weighted(self.mdp.masked_softmax(self.get(state, heuristic_parameter=heuristic_parameter, temperature=temperature, num_searches=num_searches), state))
        else:
            return self.mdp.get_max_action(self.get(state, heuristic_parameter=heuristic_parameter, temperature=temperature, num_searches=num_searches) + self.mdp.neginf_kill_actions(state))
        
    # Only save and load the network (Q gets way too big)
    def save(self, fname):
        model_scripted = torch.jit.script(self.h)
        model_scripted.save(fname)
        # with open(fname + ".q", 'wb') as f:
        #     pickle.dump({'n': self.n, 'w': self.w, 'p': self.p}, f)
        
    def load(self, fname, indices=None):
        self.h = torch.jit.load(fname, map_location=torch.device(self.device))
        # with open(fname + ".q", 'rb') as f:
        #     load_dict = pickle.load(f)
        #     self.n = load_dict['n']
        #     self.w = load_dict['w']
        #     self.p = load_dict['p']
        
    def null(self, indices=None):
        if self.heuristic_model != None:
            self.h = self.heuristic_model(**self.model_args, **self.mdp.nn_args).to(self.device)
            self.h.eval()
        else:
            self.h = None
        
        self.null_q()

    def null_q(self):
        self.n, self.w, self.p = {}, {}, {}
        

    # Helper function
    def in_dict_indices(self, state: torch.Tensor, d: dict) -> torch.Tensor:
        in_indices = []
        out_indices = []
        for i in range(state.size(0)):
            if self.mdp.state_to_hashable(state[i]) in d:
                in_indices.append(i)
            else:
                out_indices.append(i)
        return torch.tensor(in_indices, device=self.device).int(), torch.tensor(out_indices, device=self.device).int()

    # Starting at a given state, conducts a fixed number of Monte-Carlo searches, using the Q function in visited states and the heuristic function in new states
    # Updates the Q, N, W, P functions for explored states
    # Returns probability vector for initial state and results of each game
    def search(self, state, num: int, ucb_parameter, temperature=1.0, max_depth=1000):
        
        depth = 0

        # Do num * batch searches from some initial states; we do not need to track which search correpsonds to which batch
        # Shape: (num * batch) + state_shape
        s = (state[None] * torch.ones((num, 1) + self.mdp.state_projshape, device=self.device).float()).flatten(0, 1)
        
        # Records the states and actions taken for each self-play
        history_state = torch.zeros((0, ) + self.mdp.state_shape, device=self.device).float()
        history_action = torch.zeros((0, ) + self.mdp.action_shape, device=self.device).float()
        # Records the self-play index for each entry in the history
        history_index = torch.tensor([], device=self.device)
        
        # For storing self-plays that have entered leaves
        leaf_s = torch.zeros((0,) + self.mdp.state_shape, device=self.device)

        # Number of self-plays is num * batch
        # Tracks indices for s
        index_tracker = torch.arange(num * state.size(0), device=self.device)
        # Tracks indices for leaf_s
        leaf_index_tracker = torch.tensor([], device=self.device)
        

        # Go down tree, using upper confidence bound and dictionary q values to play
        while s.size(0) + leaf_s.size(0) > 0:

            # Check if safeguard exceeded
            if depth >= max_depth:
                print("Max depth reached in search.  Something might have gone wrong.")
                valid_actions = self.mdp.valid_action_filter(s)
                num_valid = valid_actions.sum()
                print(f"Valid: {valid_actions}")
                print(f"Get: {self.ucb_get(s, ucb_parameter * num_valid)}")
                input()
                break
            
            # Check for leaves and initialize dictionary on them
            in_index, out_index = self.in_dict_indices(s, self.n)
            new_leaves = s[out_index]
            self.h.eval()
            with torch.no_grad():           # Fixes memory leak (because ps gets stored later)
                ps = self.mdp.masked_softmax(self.h(new_leaves), new_leaves)
            for i in range(new_leaves.size(0)):
                # Note: device on CPU to save GPU memory, need to convert later
                self.n[self.mdp.state_to_hashable(new_leaves[i])] = torch.zeros(self.mdp.action_shape, device=self.dict_device)
                self.w[self.mdp.state_to_hashable(new_leaves[i])] = torch.zeros(self.mdp.action_shape, device=self.dict_device)
                self.p[self.mdp.state_to_hashable(new_leaves[i])] = ps[i][None].to(device=self.dict_device)

            # Get actions for explored and unexplored states, and add to history, update counter
            valid_actions = self.mdp.valid_action_filter(s)
            num_valid = valid_actions.flatten(1, -1).sum(dim=1).reshape((-1,) + self.mdp.action_projshape)

            # The AlphaGo algorithm calls for argmax, but we will do random, because the node statistics are not updated in parallel for us
            #action = self.mdp.get_max_action(self.ucb_get(s, ucb_parameter * num_valid), mask=valid_actions)
            # Note that ucb_get is in the range [-1, infty)
            action = self.mdp.get_random_action_weighted((1 + self.ucb_get(s, ucb_parameter * num_valid)) * valid_actions)
            for i in range(action.size(0)):
                self.n[self.mdp.state_to_hashable(s[i])] += action[i].to(device=self.dict_device)

            with torch.no_grad():
                leaf_action = self.mdp.get_random_action_weighted(self.mdp.masked_softmax(self.h(leaf_s), leaf_s))
            
            # We don't keep the simulation nodes in history for memory reasons, but do keep the first leaf encountered
            history_state = torch.cat([history_state, s])
            history_action = torch.cat([history_action, action])
            history_index = torch.cat([history_index, index_tracker])

            # Add new leaves to leaf states, remove from non-leaf states, and update indices
            leaf_s = torch.cat([leaf_s, new_leaves])
            leaf_index_tracker = torch.cat([leaf_index_tracker, index_tracker[out_index]])
            leaf_action = torch.cat([leaf_action, action[out_index]])
            s = s[in_index]
            index_tracker = index_tracker[in_index]
            action = action[in_index]

            # Do transition
            s, r = self.mdp.transition(s, action)
            leaf_s, leaf_r = self.mdp.transition(leaf_s, leaf_action)


            # Update statistics for terminal states

            # Collect results from main and leaves, and get indices
            results = torch.cat([r[self.mdp.is_terminal(s).flatten()], leaf_r[self.mdp.is_terminal(leaf_s).flatten()]])
            terminal_index = torch.cat([index_tracker[self.mdp.is_terminal(s).flatten()], leaf_index_tracker[self.mdp.is_terminal(leaf_s).flatten()]])
            
            for i in range(terminal_index.size(0)):
                update_index = (history_index == terminal_index[i])
                update_state = history_state[update_index]
                update_action = history_action[update_index]
                update_result = results[i]
                update_player = self.mdp.get_player(update_state).flatten(1, -1)
                for j in range(update_state.size(0)):
                    self.w[self.mdp.state_to_hashable(update_state[j])] += (update_action[j] * update_result[update_player[j]]).to(device=self.dict_device)

            # Remove terminal states and indices
            index_tracker = index_tracker[~self.mdp.is_terminal(s).flatten()]
            s = s[~self.mdp.is_terminal(s).flatten()]
            leaf_index_tracker = leaf_index_tracker[~self.mdp.is_terminal(leaf_s).flatten()]
            leaf_s = leaf_s[~self.mdp.is_terminal(leaf_s).flatten()]
            

            # Increase depth
            depth += 1

            # Clean up memory (from the dictionary gets)
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Return probability vector for initial state (supposedly less "prone to outliers" to train on this)
        p_vector = torch.zeros((state.size(0), ) + self.mdp.action_shape, device=self.device).float()
        for i in range(state.size(0)):
            p_vector[i] = self.n[self.mdp.state_to_hashable(state[i])].to(self.device) ** (1 / temperature)
            p_vector[i] = p_vector[i] / p_vector[i].flatten().sum()
        return p_vector
        
    


# Assumption: the only reward is given at the terminal state, so is only checked here
class DMCTS(DeepRL):

    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, model_args = {}, device="cpu"):
        super().__init__(mdp, MCTSQFunction(mdp, model, model_args, device=device, dict_device=device), device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def get_statistics(self, state: torch.Tensor) -> str:
        q, n_ratio = self.q.ucb_get_parts(state, get_p=False)
        h = self.q.h(state)
        statehash = self.mdp.state_to_hashable(state)
        output = f"Q values:\n{q}\n"
        output += f"Visit factor:\n{n_ratio}\n"
        #output += f"Visits:\n{self.q.n[statehash] if statehash in self.q.n}\n"
        output += f"Heuristic logit values:\n{h}\n"
        output += f"Heuristic values:\n{self.mdp.masked_softmax(h, state)}\n"
        #output += f"Action values:\n{self.q.get(state, None)}\n"
        return output

    # Returns average rewards TODO need to permute players...
    def compare_models(self, models, num_plays, num_searches, ucb_parameter):
        if self.mdp.num_players > 2:
            raise NotImplementedError
        
        # Evens start on evens, odd start on odd
        s = self.mdp.get_initial_state(num_plays * 2)
        rewards = torch.zeros((s.size(0), 2), device=self.device)

        while torch.prod(self.mdp.is_terminal(s)).item() != 1:
            player_index = (self.mdp.get_player(s).flatten() + torch.arange(s.size(0), device=self.device)) % 2
            
            # Actions
            a = torch.zeros((s.size(0),) + self.mdp.action_shape)
            a[player_index == 0] = models[0].policy(s[player_index == 0], heuristic_parameter=ucb_parameter, temperature=.1, num_searches=num_searches, stochastic=True)
            a[player_index == 1] = models[1].policy(s[player_index == 1], heuristic_parameter=ucb_parameter, temperature=.1, num_searches=num_searches, stochastic=True)
            s, r = self.mdp.transition(s, a)
            rewards += r
            
        # Odd player indices are swapped
        rewards[1::2] = rewards[1::2] * -1
        # Return average rewards
        return rewards.sum(0) / s.size(0)




    def mcts(self, lr: float | list[tuple], wd: float, num_iterations: int, num_episodes: int, num_selfplay: int, num_searches: int, max_steps: int, ucb_parameter: float, temperature: float, train_batch: int, tournament_length: int, tournament_searches: int, train_times = 1, memory_size = 500000, dethrone_threshold = .1, save_path=None, verbose=True, initial_log=""):        
        
        # Initialize logging
        logtext = initial_log
        start_time = datetime.datetime.now()
        logtext += log(f"Started logging.")
        logtext += log(f"Device: {self.device}", verbose)
        logtext += log(f"MDP:\n{self.mdp}", verbose)
        logtext += log(f"Model:\n{self.q.h}", verbose)
        logtext += log(f"Loss function:\n{self.loss_fn}", verbose)
        logtext += log(f"Optimizer:\n{self.optimizer}", verbose)
        logtext += log(f"Learn rate: {lr}\nWeight decay: {wd}\nNumber of iterations: {num_iterations}\nNumber of self-plays per iteration: {num_selfplay}\nNumber of Monte-Carlo searches per play: {num_searches}\nUpper confidence bound parameter: {ucb_parameter}\nTemperature: {temperature}\nTraining batch size: {train_batch}\nMemory size: {memory_size}\n", verbose)


        opt = self.optimizer(self.q.h.parameters(), lr=lr,  weight_decay=wd)

        # Initialize memory and records; memory persists across iterations, losses reset on new model
        memory_inputs = torch.zeros((0,) + self.mdp.state_shape, device=self.device)
        memory_values = torch.zeros((0,) + self.mdp.action_shape, device=self.device)
        losses = []
        sim_changes = []
        generation_model = copy.deepcopy(self.q)

        # Versions with no dictionaries
        temp_gen_model = copy.deepcopy(self.q)
        temp_cur_model = copy.deepcopy(self.q)

        model_num_iterations = 0
        
        # At the end of each iteration, we evaluate the current model against the generation (self-play) model
        for i in range(num_iterations):
            
            logtext += log(f"Iteration {i+1} at time {datetime.datetime.now() - start_time}", verbose)
            model_num_iterations = 1

            # In each episode, we simulate a certain number of self-plays; we train at each step
            for ep in range(num_episodes):

                # Initialize iteration state and loss statistics
                logtext += log(f"Iteration {i+1}, episode {ep+1} at time {datetime.datetime.now() - start_time}", verbose)
                s = self.mdp.get_initial_state(num_selfplay)
                total_loss = 0.
                num_train = 0.
                

                # Generate data via self-plays and the generation model
                for _ in range(max_steps):

                    # Do searches
                    p_vector = generation_model.search(s, ucb_parameter=ucb_parameter, num=num_searches, temperature=temperature)

                    # Add to record
                    memory_inputs = torch.cat([memory_inputs[-memory_size + s.size(0):], s], dim=0)
                    memory_values = torch.cat([memory_values[-memory_size + p_vector.size(0):], p_vector], dim=0)

                    # Select actions and do all the transitions
                    a = self.mdp.get_random_action_weighted(p_vector)
                    s, _ = self.mdp.transition(s, a)

                    # If state is terminal, then discard it
                    s = s[(~self.mdp.is_terminal(s)).flatten()]
                    # If there are no more states, end the iteration
                    if s.size(0) == 0:
                        break

                    # Train the current model
                    for _ in range(train_times):
                        indices = torch.randint(memory_inputs.size(0), (train_batch,), device=self.device)
                        x = memory_inputs[indices]
                        y = memory_values[indices]
                        self.q.h.train()
                        pred = self.q.h(x)

                        loss = self.loss_fn(pred.flatten(1, -1), y.flatten(1, -1))
                        loss.backward()
                        opt.step()
                        opt.zero_grad()

                        total_loss += loss.item()
                        num_train += 1


                # At end of episode, calculate losses
                losses.append(total_loss / num_train)
                logtext += log(f"Input/value memory sizes: {memory_inputs.size(0)}/{memory_values.size(0)}", verbose)
                logtext += log(f"Loss: {total_loss/num_train} on {num_train} batches of size {min(len(indices), train_batch)}.", verbose)


            # Evaluate models at the end of an iteration
            logtext += log(f"\nEvaluating models at end of iteration with {tournament_length} matches and {tournament_searches} searches.", verbose)
            # We want to delete Q data, so we need to take a copy
            temp_gen_model.h = generation_model.h
            temp_cur_model.h = self.q.h
            temp_gen_model.null_q()
            temp_cur_model.null_q()
            win_ratios = self.compare_models([temp_gen_model, temp_cur_model], num_plays=tournament_length, num_searches=tournament_searches, ucb_parameter=ucb_parameter)
            
            if win_ratios[1] - win_ratios[0] > dethrone_threshold:
                logtext += log(f"Replacing old self-play model; rewards {win_ratios}.")
                generation_model.h = copy.deepcopy(self.q.h)
                sim_changes.append(len(losses))
            else:
                logtext += log(f"Keeping old self-play model; rewards {win_ratios}.")


            logtext += log("\n", verbose)

        # End logging and saving
        logtext += log(f"Training completed at time {datetime.datetime.now() - start_time}", verbose)                    

        if save_path != None:
            self.q.save(save_path)
            logtext += log(f"Saved model to {save_path}", verbose)

            # Plot losses
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 1, 1)
            plt.plot(range(len(losses)), losses, label=f'Loss')
            plt.legend(loc='lower left')
            plt.title(f'Training Loss')
            for l in sim_changes:
                plt.axvline(l, linestyle=':', color='r')

            plotpath = save_path + f".losses.png"
            plt.savefig(plotpath)
            logtext += log(f"Saved loss plot to {plotpath}", verbose)

            logpath = save_path + ".log"
            with open(logpath, "w") as f:
                logtext += log(f"Saved logs to {logpath}", verbose)
                f.write(logtext)