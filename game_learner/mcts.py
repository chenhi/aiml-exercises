import random, datetime, pickle
import torch
from torch import nn
import matplotlib.pyplot as plt

from rlbase import log, smoothing, PrototypeQFunction, DeepRL, TensorMDP


class MCTSQFunction(PrototypeQFunction):
    def __init__(self, mdp: TensorMDP, heuristic_model: nn.Module, model_args={}, device="cpu"):
        self.device = device
        self.mdp = mdp
        self.h = heuristic_model(**model_args, **self.mdp.nn_args).to(self.device)
        self.h.eval()

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
                n[i] = self.n[statehash]
                q[i] = self.w[statehash] / n[i]
                if get_p:
                    p[i] = self.p[statehash]
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
    

    # Same as above, but uses the heuristic function instead of p
    def get(self, state: torch.Tensor, action = None, heuristic_parameter= 1.):
        q, n_ratio = self.ucb_get_parts(state, get_p=False)
        output = q + self.mdp.masked_softmax(self.h(state), state) * heuristic_parameter * n_ratio

        if action == None:
            return output         # TODO q is a value so can be negative... ???? also do we want "Get" to return N or ...?
        else:
            return (action * output).sum(tuple(range(1, action.dim())))


    def policy(self, state: torch.Tensor, heuristic_parameter = 1., stochastic=False, valid_filter=True):
        if stochastic:
            return self.mdp.get_random_action_weighted(self.get(state, heuristic_parameter=heuristic_parameter) * self.mdp.valid_action_filter(state))      # TODO q...?
        else:
            return self.mdp.get_max_action(self.get(state, heuristic_parameter=heuristic_parameter) + self.mdp.neginf_kill_actions(state))
        
    def save(self, fname):
        with open(fname + ".q", 'wb') as f:
            pickle.dump({'n': self.n, 'w': self.w, 'p': self.p}, f)
        model_scripted = torch.jit.script(self.h)
        model_scripted.save(fname + ".h")

    def load(self, fname, indices=None):
        with open(fname + ".q", 'rb') as f:
            load_dict = pickle.load(f)
            self.n = load_dict['n']
            self.w = load_dict['w']
            self.p = load_dict['p']
        self.h = torch.jit.load(fname + ".h", map_location=torch.device(self.device))
    
    def null(self, indices = None):
        self.n, self.w, self.p = {}, {}, {}
        #self.h.weight.data.fill_(0.)
        # TODO Heuristic function reset not implemented

        
    


# Assumption: the only reward is given at the terminal state, so is only checked here
class DMCTS(DeepRL):

    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, model_args = {}, device="cpu"):
        super().__init__(mdp, MCTSQFunction(mdp, model, model_args, device), device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def get_statistics(self, state: torch.Tensor) -> str:
        q, n_ratio = self.q.ucb_get_parts(state, get_p=False)
        h = self.mdp.masked_softmax(self.q.h(state), state)
        output = f"Q values:\n{q}\n"
        output += f"Visits:\n{self.q.n[self.mdp.state_to_hashable(state)]}\n"
        output += f"Values:\n{self.q.w[self.mdp.state_to_hashable(state)]}\n"
        output += f"Visit factor:\n{1/n_ratio}\n"
        output += f"Heuristic logit values:\n{self.q.h(state)}\n"
        output += f"Heuristic values:\n{h}\n"
        output += f"Visit-tempered heuristic values:\n{h * n_ratio}\n"
        output += f"Action values, masked, with softmax:\n{torch.softmax((self.q.get(state, None) + self.mdp.neginf_kill_actions(state)).flatten(1,-1), dim=1).reshape((-1,) + self.mdp.action_shape)}\n"
        return output

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
    # Updates the Q, N, W, P functions for explored states, returns probability vector for initial state
    def search(self, state, num: int, ucb_parameter, temperature=1.0, max_depth=1000):
        
        # Do num * batch searches from some initial states; we do not need to track which search correpsonds to which batch
        # Shape: (num * batch) + state_shape
        s = (state[None] * torch.ones((num, 1) + self.mdp.state_projshape, device=self.device).float()).flatten(0, 1)
        history_state = torch.zeros((0, ) + self.mdp.state_shape, device=self.device).float()
        history_action = torch.zeros((0, ) + self.mdp.action_shape, device=self.device).float()
        depth = 0

        leaf_s = torch.zeros((0,) + self.mdp.state_shape, device=self.device)

        # Number of self-plays is num * batch
        index_tracker = torch.arange(num * state.size(0), device=self.device)
        leaf_index_tracker = torch.tensor([], device=self.device)
        history_index = torch.tensor([], device=self.device)

        # Go down tree, using upper confidence bound and dictionary q values to play
        while s.size(0) + leaf_s.size(0) > 0:
            if depth >= max_depth:
                print("Max depth reached in search.  Something might have gone wrong.")
                valid_actions = self.mdp.valid_action_filter(s)
                num_valid = valid_actions.sum()
                print(f"Valid: {valid_actions}")
                print(f"Get: {self.q.ucb_get(s, ucb_parameter * num_valid)}")
                input()
                break
            
            # Check for leaves and initialize dictionary
            in_index, out_index = self.in_dict_indices(s, self.q.n)
            new_leaves = s[out_index]
            ps = self.mdp.masked_softmax(self.q.h(new_leaves), new_leaves)
            for i in range(new_leaves.size(0)):
                self.q.n[self.mdp.state_to_hashable(new_leaves[i])] = torch.zeros(self.mdp.action_shape, device=self.device)
                self.q.w[self.mdp.state_to_hashable(new_leaves[i])] = torch.zeros(self.mdp.action_shape, device=self.device)
                self.q.p[self.mdp.state_to_hashable(new_leaves[i])] = ps[i][None]

            # Get actions for explored and unexplored states, and add to history, update counter
            valid_actions = self.mdp.valid_action_filter(s)
            num_valid = valid_actions.flatten(1, -1).sum(dim=1).reshape((-1,) + self.mdp.action_projshape)

            # The AlphaGo algorithm calls for argmax, but we will do random, because the node statistics are not updated in parallel for us
            #action = self.mdp.get_max_action((1 + self.q.ucb_get(s, ucb_parameter * num_valid)) * valid_actions)
            action = self.mdp.get_random_action_weighted((1 + self.q.ucb_get(s, ucb_parameter * num_valid)) * valid_actions)
            for i in range(action.size(0)):
                self.q.n[self.mdp.state_to_hashable(s[i])] += action[i]

            leaf_action = self.mdp.get_random_action_weighted(self.mdp.masked_softmax(self.q.h(leaf_s), leaf_s))
            
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
            terminal_index = torch.cat([index_tracker[self.mdp.is_terminal(s).flatten()], leaf_index_tracker[self.mdp.is_terminal(leaf_s).flatten()]])
            results = torch.cat([r[self.mdp.is_terminal(s).flatten()], leaf_r[self.mdp.is_terminal(leaf_s).flatten()]])
            for i in range(terminal_index.size(0)):
                update_index = (history_index == terminal_index[i])
                update_state = history_state[update_index]
                update_action = history_action[update_index]
                update_result = results[i]
                update_player = self.mdp.get_player(update_state).flatten(1, -1)
                for j in range(update_state.size(0)):
                    self.q.w[self.mdp.state_to_hashable(update_state[j])] += update_action[j] * update_result[update_player[j]]

            # Remove terminal states and indices
            index_tracker = index_tracker[~self.mdp.is_terminal(s).flatten()]
            s = s[~self.mdp.is_terminal(s).flatten()]
            leaf_index_tracker = leaf_index_tracker[~self.mdp.is_terminal(leaf_s).flatten()]
            leaf_s = leaf_s[~self.mdp.is_terminal(leaf_s).flatten()]
            

            # Increase depth
            depth += 1

        # Return probability vector for initial state
        p_vector = torch.zeros((state.size(0), ) + self.mdp.action_shape, device=self.device).float()
        for i in range(state.size(0)):
            p_vector[i] = self.q.n[self.mdp.state_to_hashable(state[i])] ** (1 / temperature)
            p_vector[i] = p_vector[i] / p_vector[i].flatten().sum()
        return p_vector
    
    
    
    def old_search(self, state, num: int, ucb_parameter, temperature=1.0, max_depth=1000):
        
        # Do a certain number of searches from the initial state
        for i in range(num):
            history = []
            s = state
            depth = 0

            # Go down tree, using upper confidence bound and dictionary q values to play
            while self.mdp.state_to_hashable(s) in self.q.q and self.mdp.is_terminal(s).item() == False:
                if depth >= max_depth:
                    print("Max depth researched in selection.  Something might have gone wrong.")
                    valid_actions = self.mdp.valid_action_filter(s)
                    num_valid = valid_actions.sum()
                    print(f"Valid: {valid_actions}")
                    print(f"Get: {self.q.ucb_get(s, ucb_parameter * num_valid)}")
                    input()
                    break

                # Q has values [-1, 1] generally, so do a shift
                valid_actions = self.mdp.valid_action_filter(s)
                num_valid = valid_actions.sum()
                # Scale the parameter by the number of valid actions
                action = self.mdp.get_max_action((1 + self.q.ucb_get(s, ucb_parameter * num_valid)) * valid_actions)
                history.append((s, action))
                s, r = self.mdp.transition(s, action)
                depth += 1
                
            # Once we reach a leaf, use the heuristic function p to simulate play
            # TODO this cna be parallelized, i.e. the evaluaiton
            if self.mdp.is_terminal(s).item() == False:
                self.q.q[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape, device=self.device)
                self.q.n[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape, device=self.device)
                self.q.n_tot[self.mdp.state_to_hashable(s)] = 0
                self.q.w[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape, device=self.device)

                # Placeholder prior probability given by neural network, do in batches TODO implement
                # evaluation_queue.append(s)
                # if len(evaluation_queue) >= evaluation_batch_size:
                #     heuristic_prob = heuristic(torch.cat(evaluation_queue, dim=0))
                #     for i in range(heuristic_prob.size(0)):
                #         self.p[self.mdp.state_to_hashable(evaluation_queue[i])] = heuristic_prob[i:i+1]

                # Sigmoid so that the values are in (0, 1)
                # These prior probabilities are only assigned once; eventually, with enough visits the quality function will dominate the heuristic
                self.q.p[self.mdp.state_to_hashable(s)] = self.mdp.masked_softmax(self.q.h(s), s)

            while self.mdp.is_terminal(s).item() == False:
                if depth >= max_depth:
                    print("Max depth researched in expansion.  Something may have gone wrong.")
                    print(f"Board: {self.mdp.board_str(s)[0]}")
                    print(f"Heuristic: {self.q.h(s)}")
                    print(f"Probabilities: {torch.exp(self.q.h(s) + self.mdp.neginf_kill_actions(s))}")
                    break
                # Exponential means un-normalized softmax
                action = self.mdp.get_random_action_weighted(self.mdp.masked_softmax(self.q.h(s), s))
                # We don't keep the simulation nodes in history for memory reasons
                s, r = self.mdp.transition(s, action)
                depth += 1
            
            while len(history) > 0:
                s, action = history.pop()
                if self.mdp.state_to_hashable(s) in self.q.p:
                    self.q.n[self.mdp.state_to_hashable(s)] += action[0]
                    self.q.n_tot[self.mdp.state_to_hashable(s)] += action[0].sum().item()
                    self.q.w[self.mdp.state_to_hashable(s)] += action[0] * r[0, self.mdp.get_player(s).flatten(1, -1)[0]]
                    self.q.q[self.mdp.state_to_hashable(s)] = self.q.w[self.mdp.state_to_hashable(s)] / self.q.n[self.mdp.state_to_hashable(s)]

        # Return probability vector for initial state (even if it isn't updated)
        p_vector = self.q.n[self.mdp.state_to_hashable(state)] ** (1 / temperature)
        return (p_vector / p_vector.flatten().sum())[None]
                



    def mcts(self, lr: float, num_iterations: int, num_selfplay: int, num_searches: int, max_steps: int, ucb_parameter: float, temperature: float, train_batch: int, train_iterations = 1, save_path=None, verbose=True, initial_log=""):        
        logtext = initial_log
        
        start_time = datetime.datetime.now()
        logtext += log(f"Started logging.")
        
        logtext += log(f"Learn rate: {lr}\nNumber of iterations: {num_iterations}\nNumber of self-plays per iteration: {num_selfplay}\nNumber of Monte-Carlo searches per play: {num_searches}\nUpper confidence bound parameter: {ucb_parameter}\nTemperature: {temperature}\nTraining batch size: {train_batch}\nTraining iteration interval: {train_iterations}")

        losses = []

        training_inputs = []
        training_values = []

        # In each iteration, we simulate a certain number of self-plays, then we train on the resulting data
        for i in range(num_iterations):
            logtext += log(f"Iteration {i+1}", verbose)
            opt = self.optimizer(self.q.h.parameters(), lr=lr)


            s = self.mdp.get_initial_state(num_selfplay)

            # The probability vectors for each self-play
            record_state = torch.tensor([], device=self.device)
            record_p = torch.tensor([], device=self.device)

            logtext += log(f"Self-plays at time {datetime.datetime.now() - start_time}", verbose)
            # Each step in the self-plays
            for step in range(max_steps):
                
                # Do searches
                p_vector = self.search(s, ucb_parameter=ucb_parameter, num=num_searches, temperature=temperature)

                # Add probability vector to record
                record_state = torch.cat([record_state, s], dim=0)
                record_p = torch.cat([record_p, p_vector], dim=0)
                
                # Select actions and do all the transitions
                a = self.mdp.get_random_action_weighted(p_vector)
                s, _ = self.mdp.transition(s, a)

                # If state is terminal, then discard it
                s = s[(~self.mdp.is_terminal(s)).flatten()]
                # If there are no more states, break
                if s.size(0) == 0:
                    break
                        
            # TODO think about having the bot concede? might save some simulations

            logtext += log(f"Training at {datetime.datetime.now() - start_time}", verbose)

            training_inputs.append(record_state)
            training_values.append(record_p)

            # Do training on recent data
            inputs = torch.cat(training_inputs[-train_iterations:])
            values = torch.cat(training_values[-train_iterations:])
            indices = list(range(inputs.size(0)))
            total_loss = 0.
            num_train = 0.
            self.q.h.train()
            for i in range(inputs.size(0)):
                get_indices = random.sample(indices, min(len(indices), train_batch))
                x = inputs[get_indices]
               
                # Already in range (0, 1)
                y = values[get_indices]
                
                # Before softmax
                pred = self.q.h(x)

                loss = self.loss_fn(pred.flatten(1, -1), y.flatten(1, -1))
                loss.backward()
                opt.step()
                opt.zero_grad()

                total_loss += loss.item()
                num_train += 1

            self.q.h.eval()
            losses.append(total_loss / num_train)
            logtext += log(f"Loss: {total_loss/num_train} on {num_train} batches of size {min(len(indices), train_batch)}.", verbose)

        if save_path != None:
            self.q.save(save_path)
            logtext += log(f"Saved model to {save_path}.h, {save_path}.q", verbose)

            # Plot losses
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 1, 1)
            plt.plot(range(num_iterations), losses, label=f'Loss')
            plt.legend(loc='lower left')
            plt.title('Training Loss')
            plotpath = save_path + f".losses.png"
            plt.savefig(plotpath)
            logtext += log(f"Saved losses plot to {plotpath}", verbose)

            logpath = save_path + ".log"
            with open(logpath, "w") as f:
                logtext += log(f"Saved logs to {logpath}", verbose)
                f.write(logtext)



                

            