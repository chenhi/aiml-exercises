import zipfile, random, warnings, datetime, copy, pickle, tempfile
import torch
from torch import nn
from collections import namedtuple, deque
import matplotlib.pyplot as plt

from rlbase import TensorMDP, DeepRL, log, smoothing, PrototypeQFunction



# I sometimes indicate terminal states by negating all values in the state, so I don't have to implement a method to check terminal conditions, which can be costly.


# Source, action, target, reward
TransitionData = namedtuple('TransitionData', ('s', 'a', 't', 'r'))

# When the deque hits memory limit, it removes the oldest item from the list
class ExperienceReplay():
    def __init__(self, memory: int):
        self.memory = deque([], maxlen=memory)

    def size(self):
        return len(self.memory)

    # Push a single transition (comes in the form of a list)
    def push(self, d: TransitionData):
        self.memory.append(d)

    def push_batch(self, data: TransitionData, transforms = {'s': lambda s: s, 'a': lambda a: a, 'r': lambda r: r}):
        for i in range(data.s.size(0)):
                self.push(TransitionData(transforms['s'](data.s[i]), transforms['a'](data.a[i]), transforms['s'](data.t[i]), transforms['r'](data.r[i])))

            

    def sample(self, num: int, batch=True) -> list:
        if batch:
            data = random.sample(self.memory, num)
            return TransitionData(torch.stack([datum.s for datum in data]), torch.stack([datum.a for datum in data]), torch.stack([datum.t for datum in data]), torch.stack([datum.r for datum in data]))
        else:
            return random.sample(self.memory, num)       
    
    def __len__(self):
        return len(self.memory)


# A Q-function where the inputs and outputs are all tensors, for a single player
class NNQFunction(PrototypeQFunction):
    def __init__(self, mdp: TensorMDP, q_model, model_args={}, device="cpu"):
        if mdp.state_shape == None or mdp.action_shape == None:
            raise Exception("The input MDP must handle tensors.")
        if q_model == None:
            self.q = None
        else:
            self.q = q_model(**model_args, **mdp.nn_args).to(device)
            self.q.eval()
        
        self.mdp = mdp
        self.device=device

    # Make the Q function perform randomly.
    def lobotomize(self):
        self.q = None

    # If input action = None, then return the entire vector of action values of shape (batch, ) + action_shape
    # Otherwise, output shape (batch, )
    def get(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.q == None:
            pred = self.mdp.get_random_action(state)    
        else:
            self.q.eval()
            pred = self.q(state)
        
        if action == None:
            return pred
        else:
            return (pred * action).sum(tuple(range(1,pred.dim())))

    # Input is a tensor of shape (batches, ) + state.shape
    # Output is a tensor of shape (batches, )
    def val(self, state, valid_filter=True) -> torch.Tensor:
        if self.q == None:
            return torch.zeros(state.size(0))
        self.q.eval()
        return (self.q(state) + valid_filter * self.mdp.neginf_kill_actions(state)).flatten(1, -1).max(1).values * ~torch.flatten(self.mdp.is_terminal(state))
    
    # Input is a batch of state vectors
    # Returns the value of an optimal policy at a given state, shape (batches, ) + action_shape
    def policy(self, state, valid_filter=True) -> torch.Tensor:
        if self.q == None:
            return self.mdp.get_random_action(state)
        self.q.eval()
        return self.mdp.get_max_action(self.q(state) + valid_filter * self.mdp.neginf_kill_actions(state))


    # Does a Q-update based on some observed set of data
    # TransitionData entries are tensors
    def update(self, data: TransitionData, learn_rate, optimizer_class: torch.optim.Optimizer, loss_fn, target_q=None, valid_filter=True):
        if not isinstance(self.q, nn.Module):
            Exception("NNQFunction needs to have a class extending nn.Module.")

        if self.q == None:
            return

        if target_q == None:
            target_q = self

        self.q.train()
        opt = optimizer_class(self.q.parameters(), lr=learn_rate)
        pred = self.get(data.s, data.a)
        y = data.r + self.mdp.discount * target_q.val(data.t, valid_filter=valid_filter)
        self.q.train()
        loss = loss_fn(pred, y)

        # Optimize
        loss.backward()
        opt.step()
        opt.zero_grad()

        self.q.eval()

        return loss.item()
    

# A wrapper for Q functions for multiple players
class NNQMultiFunction(PrototypeQFunction):
    def __init__(self, mdp: TensorMDP, q_model, model_args={}, device="cpu"):
        self.qs = [NNQFunction(mdp, q_model, model_args=model_args, device=device) for i in range(mdp.num_players)]
        self.mdp = mdp
        self.device=device

    # If input action = None, then return the entire vector of action values of shape (batch, ) + action_shape
    # Otherwise, output shape (batch, )
    def get(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        players = self.mdp.get_player(state)
        output = torch.zeros((state.size(0), ) + self.mdp.action_shape) if action == None else torch.zeros(state.size(0), device=self.device)
        for p in range(self.mdp.num_players):
            indices = torch.arange(state.size(0), device=self.device)[players.flatten() == p]
            output[indices] = self.qs[p].get(state[indices], None if action == None else action[indices])
        return output

    # Input is a tensor of shape (batches, ) + state.shape
    # Output is a tensor of shape (batches, )
    def val(self, state, valid_filter=True) -> torch.Tensor:
        players = self.mdp.get_player(state)
        output = torch.zeros((state.size(0), ) + self.mdp.state_shape, device=self.device)
        for p in range(self.mdp.num_players):
            indices = torch.arange(state.size(0), device=self.device)[players.flatten() == p]
            output[indices] = self.qs[p].val(state, valid_filter)
        return output
    
    # Input is a batch of state vectors
    # Returns the value of an optimal policy at a given state, shape (batches, ) + action_shape
    def policy(self, state, valid_filter=True) -> torch.Tensor:
        players = self.mdp.get_player(state)
        output = torch.zeros((state.size(0), ) + self.mdp.action_shape, device=self.device)
        for p in range(self.mdp.num_players):
            indices = torch.arange(state.size(0), device=self.device)[players.flatten() == p]
            output[indices] = self.qs[p].policy(state[indices], valid_filter)
        return output
    
    def save(self, fname):
        with zipfile.ZipFile(fname, mode="w") as zf:
            for i in range(self.mdp.num_players):
                model_scripted = torch.jit.script(self.qs[i].q)
                with tempfile.NamedTemporaryFile() as tmp:
                    model_scripted.save(tmp.name)
                    zf.write(tmp.name, f"player.{i}", compress_type=zipfile.ZIP_STORED)
      

    def load(self, fname, indices=None):
        with zipfile.ZipFile(fname, mode="r") as zf:
            for i in range(self.mdp.num_players):
                if indices == None or i in indices:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        load_file = zf.extract(f"player.{i}",  tmpdir)
                        self.qs[i].q = torch.jit.load(load_file, map_location=torch.device(self.device))



    def null(self, indices = None):
        for i in range(self.mdp.num_players):
            if indices == None or i in indices:
                self.qs[i].lobotomize()



# For now, for simplicity, fix a single strategy
# Note that qs is policy_qs
class DQN(DeepRL):
    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, memory_capacity: int, model_args = {}, device="cpu"):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        # self.mdp = mdp
        # self.device = device

        # # For deep Q learning
        # #self.qs = [NNQFunction(mdp, model, model_args=model_args, device=device) for i in range(mdp.num_players)]
        # self.q = NNQMultiFunction(mdp, model, model_args, device)

        super().__init__(mdp, NNQMultiFunction(mdp, model, model_args, device), device)

        self.memories = [ExperienceReplay(memory_capacity) for i in range(mdp.num_players)]
        self.memory_capacity = memory_capacity
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # For classical Q learning
        self.q_dict = {}


    # Keeps the first action; the assumption is the later actions are "passive" (i.e. not performed by the given player)
    # Adds the rewards
    # Returns a tuple: (composition, to_memory)
    def compose_transition_tensor(self, first: TransitionData, second: TransitionData, player_index: int):
        if torch.prod((first.t == second.s) == 0).item():
            # Make this error non-fatal but make a note
            warnings.warn("The source and target states do not match.")

        # Two cases: say (s,a,r,t) compose (s',a',r',t')
        # If player(t) = player_index or t is terminal, then (s',a',r',t'), i.e. replace because loop completed last turn, has been commited to memory
        # Else, (s,a,r+r',t'), i.e. it has not, so compose
        # Filter shape (batch, 1, ...).  For annoying tensor dimension reasons, need different filters that are basically the same
        filter_s = (self.mdp.get_player(first.t) == player_index) | self.mdp.is_terminal(first.t)
        filter_a = filter_s.view((filter_s.size(0), ) + self.mdp.action_projshape)
        filter_r = filter_s.flatten()
        new_s = filter_s * second.s + (filter_s == False) * first.s
        new_a = filter_a * second.a + (filter_a == False) * first.a
        new_t = second.t
        new_r = second.r + (filter_r == False) * first.r
    
        # After the above update, commit to the memory bank rows where: 
        # new_s is not terminal AND (
        # player(new_s) = player(new_t) = player_index, i.e. the cycle completed and came back to the recording player, OR 
        # where new_t is terminal )
        filter = (~self.mdp.is_terminal(new_s) & (((self.mdp.get_player(new_s) == player_index) & (self.mdp.get_player(new_t) == player_index)) | (self.mdp.is_terminal(new_t)))).flatten()
        return (TransitionData(new_s, new_a, new_t, new_r), TransitionData(new_s[filter], new_a[filter], new_t[filter], new_r[filter]))


    def minmax(self, state = None):
        if state == None:
            state = self.mdp.get_initial_state()

        actions = self.mdp.valid_action_filter(state)
        while len(torch.nonzero(actions).tolist()) > 0:
            a = self.mdp.get_random_action_weighted(actions)
            t, r = self.mdp.transition(state, a)
            if self.mdp.is_terminal(t).item():
                self.q_dict[(self.mdp.state_to_hashable(state[0]), self.mdp.action_to_hashable(a[0]))] = r[0].tolist()
            else:
                vals = []
                val = self.minmax(self, t)
                vals.append(val)
                self.q_dict[(self.mdp.state_to_hashable(state[0]), self.mdp.action_to_hashable(a[0]))] = val

        p = self.mdp.get_player(state).int()[0]
        val = max(vals, key=lambda y: y[p])
        return val

    # Handling multiplayer: each player keeps their own "record", separate from memory
    # When any entry in the record has source = target, then the player "banks" it in their memory
    # The next time an action is taken, if the source = target, then it gets overwritten
    def deep_q(self, lr: float, dq_episodes: int, episode_length: int, ramp_start: int, ramp_end: int, greed_start: float, greed_end: float, training_delay: int, sim_batch: int, train_batch: int, copy_interval_eps=1, valid_filter=True, save_interval=100, save_path=None, verbose=False, graph_smoothing=10, initial_log="", save_memory=False, load_memory = None):

        dq_episodes, episode_length, ramp_start, ramp_end, sim_batch, train_batch, copy_interval_eps, training_delay = int(dq_episodes), int(episode_length), int(ramp_start), int(ramp_end), int(sim_batch), int(train_batch), max(1, int(copy_interval_eps)), int(training_delay)
        expl_start, expl_end = 1. - greed_start, 1. - greed_end

        do_logging = True if verbose or save_path != None else False

        if train_batch < 2:
            Exception("Training batch size must be greater than 1 for sampling.")

        # Logging
        logtext = initial_log

        if load_memory != None:
            with open(load_memory, 'rb') as f:
                self.memories = pickle.load(f)
                logtext += log(f"Loaded replay memory from {load_memory}")

        if do_logging:
            logtext += log(f"Device: {self.device}", verbose)
            logtext += log(f"MDP:\n{self.mdp}", verbose)
            logtext += log(f"Model:\n{self.q.qs[0].q}", verbose)
            logtext += log(f"Loss function:\n{self.loss_fn}", verbose)
            logtext += log(f"Optimizer:\n{self.optimizer}", verbose)
            logtext += log(f"Zeroing out invalid moves" if valid_filter else f"Penalizing invalid moves")
            logtext += log(f"Learn rate: {lr}, episodes: {dq_episodes}, start and end exploration: [{expl_start}, {expl_end}], ramp: [{ramp_start}, {ramp_end}], training delay: {training_delay}, episode length: {episode_length}, batch size: {sim_batch}, training batch size: {train_batch}, copy frequency: {copy_interval_eps}, memory capacity: {self.memory_capacity}.", verbose)
            
            
            episode_losses = [[] for i in range(self.mdp.num_players)]
            tests = []                                          # Format: tests[i][j] is the jth iteration of the ith test
            initial_test = self.mdp.tests(self.q.qs)
            for i in range(len(initial_test)):
                tests.append([initial_test[i]])
            start_time = datetime.datetime.now()
            logtext += log(f"Starting training at {start_time}\n", verbose)

                        
        # Initialize target network
        target_qs = [copy.deepcopy(self.q.qs[i]) for i in range(self.mdp.num_players)]
        for i in range(self.mdp.num_players):
            target_qs[i].q.eval()
        
        # Begin training
        for ep_num in range(dq_episodes):
            # Set exploration value
            expl_cur = min(max(((expl_end - expl_start) * ep_num + (ramp_end * expl_start - ramp_start * expl_end))/(ramp_end - ramp_start), expl_end), expl_start)

            # Track losses and memory if we are logging
            if verbose or save_path != None:
                losses = [0.] * self.mdp.num_players
                num_updates = [0] * self.mdp.num_players
                memorylen = [self.memories[i].size() for i in range(self.mdp.num_players)]
                logtext += log(f"Initializing episode {ep_num+1}. Greed {1-expl_cur:>0.5f}. Memory {memorylen}.", verbose)
            
            # Get initial state
            s = self.mdp.get_initial_state(sim_batch)
            # Initialize "records" for each player
            player_record = [None for i in range(self.mdp.num_players)]
            
            for k in range(episode_length):

                # Execute the transition on the "actual" state
                
                # Get the actions using each player's policy using epsilon-greed
                a = torch.zeros((sim_batch, ) + self.mdp.action_shape, device=self.device)
                random_numbers = torch.rand(s.size(0))
                a[random_numbers < expl_cur] = self.mdp.get_random_action(s[random_numbers < expl_cur])
                a[random_numbers >= expl_cur] = self.q.policy(s[random_numbers >= expl_cur], valid_filter=valid_filter)

                # Do the transition 
                t, r = self.mdp.transition(s, a)
                
                # Update player records and memory
                for pi in range(self.mdp.num_players):
                    # If it's the first move, just put the record in.  Note we only care about the recorder's reward.
                    if player_record[pi] == None:
                        player_record[pi] = TransitionData(s, a, t, r[:,pi])
                    else:
                        player_record[pi], to_memory = self.compose_transition_tensor(player_record[pi], TransitionData(s, a, t, r[:,pi]), pi)
                        self.memories[pi].push_batch(to_memory)

                        
                # Train the policy on a random sample in memory (once the memory bank is big enough)
                for i in range(self.mdp.num_players):
                    if len(self.memories[i]) >= train_batch and ep_num >= training_delay:
                        losses[i] += self.q.qs[i].update(self.memories[i].sample(train_batch), lr, optimizer_class=self.optimizer, loss_fn=self.loss_fn, target_q=target_qs[i], valid_filter=valid_filter)
                        num_updates[i] += train_batch

                # Set the state to the next state
                s = t

            
            # Copy the target network to the policy network if it is time
            if ep_num > training_delay and (ep_num+1) % copy_interval_eps == 0:
                for i in range(self.mdp.num_players):
                    target_qs[i].q.load_state_dict(self.q.qs[i].q.state_dict())
                logtext += log("Copied policy to target networks for all players.", verbose)

            # Save a copy of the model if time
            if save_path != None and (ep_num+1) % save_interval == 0:
                self.q.save(f"{save_path}.{ep_num+1}")
                logtext += log(f"Saved {ep_num+1}-iteration model to {save_path}.{ep_num+1}") 

            
            # Record statistics at end of episode
            if do_logging:
                for i in range(len(losses)):
                    if num_updates[i] > 0:
                        losses[i] = losses[i]/num_updates[i]
                        episode_losses[i].append(losses[i])
                    else:
                        losses[i] = float('NaN')
                logtext += log(f"Episode {ep_num+1} average loss: {losses}", verbose)
                test_results = self.mdp.tests(self.q.qs)
                for j in range(len(test_results)):
                    tests[j].append(test_results[j])

            
        # End-of-training logging and saving

        if do_logging:
            end_time = datetime.datetime.now()
            logtext += log(f"\nTraining finished at {end_time}")
            logtext += log(f"Total time: {end_time - start_time}")

        if save_path != None:
            # Save final model
            self.q.save(save_path)
            logtext += log(f"Saved final model to {save_path}")
            
            if save_memory:
                with open(save_path + ".mem", 'wb') as f:
                    pickle.dump(self.memories, f)
                    logtext += log(f"Saved replay memory to {save_path}.mem")

            # Plot losses
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 1, 1)
            for i in range(self.mdp.num_players):
                plt.plot(range(dq_episodes - len(episode_losses[i]), dq_episodes), smoothing(episode_losses[i], graph_smoothing), label=f'Player {i}')
            plt.legend(loc='lower left')
            plt.title('Smoothed Loss')
            plotpath = save_path + f".losses.png"
            plt.savefig(plotpath)
            logtext += log(f"Saved losses plot to {plotpath}", verbose)

            # Plot test results
            for j in range(len(tests)):
                plt.figure(figsize=(12, 12))
                plt.subplot(1, 1, 1)
                for k in tests[j][0].keys():
                    vals = []
                    for i in range(len(tests[j])):
                        vals.append(tests[j][i][k])
                    plt.plot(range(dq_episodes + 1), smoothing(vals, graph_smoothing), label=f'{k}')
                plt.legend(loc='lower left')
                plt.title(f'Test {j} (smoothed)')
                plotpath = save_path + f".test{j}.png"
                plt.savefig(plotpath)
                logtext += log(f"Saved test {j} plot to {plotpath}", verbose)


        if do_logging:
            logtext += log("Benchmarking against random.")
            simulation_results = self.simulate_against_random(self.mdp.num_simulations)
            for i in range(self.mdp.num_players):
                logtext += log(f"In {self.mdp.num_simulations} simulations by player {i}, {simulation_results[i][0]} wins, {simulation_results[i][1]} losses, {simulation_results[i][2]} ties, {simulation_results[i][3]} invalid moves, {simulation_results[i][4]} unknown results.")

        if save_path != None:
            logpath = save_path + ".log"
            with open(logpath, "w") as f:
                logtext += log(f"Saved logs to {logpath}", verbose)
                f.write(logtext)
