import zipfile, os, random, warnings, datetime, copy, math
import torch
from torch import nn
from collections import namedtuple, deque
from qlearn import MDP, QFunction, PrototypeQFunction
from aux import log, smoothing
import matplotlib.pyplot as plt

# The player order matches their names.
# I sometimes indicate terminal states by negating all values in the state.
#Sometimes this means I don't have to implement a method to check terminal conditions globally, which can be costly, and can just do it locally in transitions.


#################### DEEP Q-LEARNING ####################

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
            s = torch.stack([datum[0] for datum in data])
            return TransitionData(torch.stack([datum.s for datum in data]), torch.stack([datum.a for datum in data]), torch.stack([datum.t for datum in data]), torch.stack([datum.r for datum in data]))
        else:
            return random.sample(self.memory, num)       
    
    def __len__(self):
        return len(self.memory)
    


class TensorMDP(MDP):
    def __init__(self, state_shape, action_shape, default_memory: int, discount=1, num_players=1, penalty = -2, num_simulations=1000, default_hyperparameters={}, symb = {}, nn_args={}, input_str = "", batched=False):
        
        super().__init__(None, None, discount, num_players, penalty, symb, input_str, default_hyperparameters, batched)
        self.nn_args = nn_args
        self.default_memory = default_memory

        # Whether we filter out illegal moves in training or not
        #self.filter_illegal = filter_illegal

        self.state_shape = state_shape
        self.action_shape = action_shape

        # Useful for getting things in the right shape
        self.state_projshape = (1, ) * len(self.state_shape)
        self.action_projshape = (1, ) * len(self.action_shape)

        self.state_linshape = 0
        for i in range(len(self.state_shape)):
            self.state_linshape += self.state_shape[i]
        self.action_linshape = 0
        for i in range(len(self.action_shape)):
            self.action_linshape += self.action_shape[i]

        self.num_simulations = num_simulations

    ##### ACTIONS #####

    def valid_action_filter(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_random_action(self, state, max_tries=100) -> torch.Tensor:
        return self.get_random_action_from_filter(self.valid_action_filter(state).float())
    
    def get_random_action_from_filter(self, filter, max_tries=100) -> torch.Tensor:
        while (filter.flatten(1,-1).count_nonzero(dim=1) <= 1).prod().item() != 1:                             # Almost always terminates after one step
            temp = torch.rand((filter.size(0),) + self.action_shape, device=self.device) * filter
            filter = (temp == temp.flatten(1,-1).max(1).values.reshape((-1,) + self.action_projshape)).float()
            max_tries -= 1
            if max_tries == 0:
                break
        return filter * 1.
    
    # Output has state shape
    def is_valid_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return (self.valid_action_filter(state) * action).flatten(1,-1).sum(1, keepdim=True).reshape((-1,) + self.state_projshape) == 1.

    def tests(self, qs: list[PrototypeQFunction]):
        return []
        
    # Not batched
    def state_to_hashable(self, state: torch.Tensor):
        return tuple(state.flatten().tolist())
    
    def hashable_to_state(self, state):
        return torch.tensor(state).reshape(self.state_shape)
    
    def action_to_hashable(self, action: torch.Tensor):
        return tuple(action.flatten().tolist())
    
    def hashable_to_action(self, action):
        return torch.tensor(action).reshape(self.action_shape)
    



# A Q-function where the inputs and outputs are all tensors
class NNQFunction(QFunction):
    def __init__(self, mdp: TensorMDP, q_model, model_args={}, device="cpu"):
        if mdp.state_shape == None or mdp.action_shape == None:
            raise Exception("The input MDP must handle tensors.")
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
            # A little inefficient because we only take the diagonals, 
            return torch.tensordot(pred.flatten(start_dim=1), action.flatten(start_dim=1), dims=([1],[1])).diagonal()

    # Input is a tensor of shape (batches, ) + state.shape
    # Output is a tensor of shape (batches, )
    def val(self, state) -> torch.Tensor:
        if self.q == None:
            return torch.zeros(state.size(0))
        self.q.eval()
        return self.q(state).flatten(1, -1).max(1).values * ~torch.flatten(self.mdp.is_terminal(state))
    
    # Input is a batch of state vectors
    # Returns the value of an optimal policy at a given state, shape (batches, ) + action_shape
    def policy(self, state, max_tries=100) -> torch.Tensor:
        if self.q == None:
            return self.mdp.get_random_action(state)
        
        filter = self.q(state).flatten(1, -1)
        filter = (filter == filter.max(1).values[:,None])
        while (filter.count_nonzero(dim=1) <= 1).prod().item() != 1:                            # Almost always terminates after one step
            filter = torch.rand(filter.shape, device=self.device) * filter
            filter = (filter == filter.max(1).values[:,None])
            max_tries -= 1
            if max_tries == 0:
                break
        return filter.unflatten(1, self.mdp.action_shape) * 1.

    # Does a Q-update based on some observed set of data
    # TransitionData entries are tensors
    def update(self, data: TransitionData, learn_rate, optimizer_class: torch.optim.Optimizer, loss_fn, target_q=None, ):
        if not isinstance(self.q, nn.Module):
            Exception("NNQFunction needs to have a class extending nn.Module.")

        if self.q == None:
            return

        if target_q == None:
            target_q = self

        self.q.train()
        opt = optimizer_class(self.q.parameters(), lr=learn_rate)
        pred = self.get(data.s, data.a)
        y = data.r + target_q.val(data.t)
        self.q.train()
        loss = loss_fn(pred, y)

        # Optimize
        loss.backward()
        opt.step()
        opt.zero_grad()

        self.q.eval()

        return loss.item()






# Input shape (batch_size, ) + state_tensor
# We implement a random tensor of 0's and 1's generating a random tensor of floats in [0, 1), then converting it to a bool, then back to a float.
def greedy_tensor(q: NNQFunction, state, eps = 0.):
    return q.mdp.get_random_action(state) if random.random() < eps else q.policy(state)









class DeepRL():
    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, model_args = {}, device="cpu"):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    def save(self, fname):
        raise NotImplementedError     

    def load(self, fname, indices=None):
        raise NotImplementedError
    
    def null(self, indices = None):
        raise NotImplementedError

    
    def stepthru_game(self):
        s = self.mdp.get_initial_state()
        print(f"Initial state:\n{self.mdp.board_str(s)[0]}")
        turn = 0
        while self.mdp.is_terminal(s) == False:
            turn += 1
            p = int(self.mdp.get_player(s)[0].item())
            print(f"Turn {turn}, player {p+1} ({self.mdp.symb[p]})")
            a = self.qs[p].policy(s)
            print(f"Chosen action: {self.mdp.action_str(a)[0]}")
            s, r = self.mdp.transition(s, a)
            print(f"Next state:\n{self.mdp.board_str(s)[0]}")
            print(f"Rewards for players: {r[0].tolist()}")
            input("Enter to continue.\n")
        input("Terminal state reached.  Enter to end. ")


    def simulate(self):
        s = self.mdp.get_initial_state()
        while self.mdp.is_terminal(s) == False:
            p = int(self.mdp.get_player(s)[0].item())
            a = self.qs[p].policy(s)
            if self.mdp.is_valid_action(s, a):
                s, r = self.mdp.transition(s, a)
            else:
                a = self.mdp.get_random_action(s)
                s, r = self.mdp.transition(s, a)
        return r

    def simulate_against_random(self, num_simulations: int, replay_loss = False, verbose = False):
        output = []
        for i in range(self.mdp.num_players):
            if verbose:
                print(f"Simulating player {i} against random bot for {num_simulations} simulations.")
            wins, losses, ties, invalids, unknowns  = 0, 0, 0, 0, 0
            for j in range(num_simulations):
                s = self.mdp.get_initial_state()
                if replay_loss:
                    history = [s]
                while self.mdp.is_terminal(s).item() == False:
                    p = int(self.mdp.get_player(s).item())
                    if p == i:
                        a = self.qs[i].policy(s)
                        if self.mdp.is_valid_action(s, a).item():
                            s, r = self.mdp.transition(s, a)
                        else:
                            invalids += 1
                            a = self.mdp.get_random_action(s)
                            s, r = self.mdp.transition(s, a)
                    else:
                        a = self.mdp.get_random_action(s)
                        s, r = self.mdp.transition(s, a)
                    if replay_loss:
                        history.append(s)
                if r[0,i].item() == 1.:
                    wins += 1
                elif r[0, i].item() == -1.:
                    losses += 1
                    if replay_loss:
                        for s in history:
                            print(self.mdp.board_str(s)[0])
                            input()
                elif r[0, i].item() == 0.:
                    ties += 1
                else:
                    unknowns += 1
            output.append((wins, losses, ties, invalids, unknowns))
            if verbose:
                print(f"Player {i} {wins} wins, {losses} losses, {ties} ties, {invalids} invalid moves, {unknowns} unknown results.")
        return output
            
 



# For now, for simplicity, fix a single strategy
# Note that qs is policy_qs
class DQN():
    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, memory_capacity: int, model_args = {}, device="cpu"):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.device = device


        # For deep Q learning
        self.qs = [NNQFunction(mdp, model, model_args=model_args, device=device) for i in range(mdp.num_players)]
        self.memories = [ExperienceReplay(memory_capacity) for i in range(mdp.num_players)]
        self.memory_capacity = memory_capacity
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # For classical Q learning
        self.q_dict = {}

    def save_q(self, fname):
        zf = zipfile.ZipFile(fname, mode="w")
        for i in range(self.mdp.num_players):
            model_scripted = torch.jit.script(self.qs[i].q)
            model_scripted.save(f"temp/player.{i}")
            zf.write(f"temp/player.{i}", f"player.{i}", compress_type=zipfile.ZIP_STORED)
            os.remove(f"temp/player.{i}")
        zf.close()

            

    def load_q(self, fname, indices=None):
        zf = zipfile.ZipFile(fname, mode="r")
        for i in range(self.mdp.num_players):
            if indices == None or i in indices:
                zf.extract(f"player.{i}", "temp/")
                self.qs[i].q = torch.jit.load(f"temp/player.{i}")
                os.remove(f"temp/player.{i}")
        zf.close()


    def null_q(self, indices = None):
        for i in range(self.mdp.num_players):
            if indices == None or i in indices:
                self.qs[i].lobotomize()
                
    def stepthru_game(self):
        s = self.mdp.get_initial_state()
        print(f"Initial state:\n{self.mdp.board_str(s)[0]}")
        turn = 0
        while self.mdp.is_terminal(s) == False:
            turn += 1
            p = int(self.mdp.get_player(s)[0].item())
            print(f"Turn {turn}, player {p+1} ({self.mdp.symb[p]})")
            a = self.qs[p].policy(s)
            print(f"Chosen action: {self.mdp.action_str(a)[0]}")
            s, r = self.mdp.transition(s, a)
            print(f"Next state:\n{self.mdp.board_str(s)[0]}")
            print(f"Rewards for players: {r[0].tolist()}")
            input("Enter to continue.\n")
        input("Terminal state reached.  Enter to end. ")

    def simulate(self):
        s = self.mdp.get_initial_state()
        while self.mdp.is_terminal(s) == False:
            p = int(self.mdp.get_player(s)[0].item())
            a = self.qs[p].policy(s)
            if self.mdp.is_valid_action(s, a):
                s, r = self.mdp.transition(s, a)
            else:
                a = self.mdp.get_random_action(s)
                s, r = self.mdp.transition(s, a)
        return r


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


    def simulate_against_random(self, num_simulations: int, replay_loss = False, verbose = False):
        output = []
        for i in range(self.mdp.num_players):
            if verbose:
                print(f"Simulating player {i} against random bot for {num_simulations} simulations.")
            wins, losses, ties, invalids, unknowns  = 0, 0, 0, 0, 0
            for j in range(num_simulations):
                s = self.mdp.get_initial_state()
                if replay_loss:
                    history = [s]
                while self.mdp.is_terminal(s).item() == False:
                    p = int(self.mdp.get_player(s).item())
                    if p == i:
                        a = self.qs[i].policy(s)
                        if self.mdp.is_valid_action(s, a).item():
                            s, r = self.mdp.transition(s, a)
                        else:
                            invalids += 1
                            a = self.mdp.get_random_action(s)
                            s, r = self.mdp.transition(s, a)
                    else:
                        a = self.mdp.get_random_action(s)
                        s, r = self.mdp.transition(s, a)
                    if replay_loss:
                        history.append(s)
                if r[0,i].item() == 1.:
                    wins += 1
                elif r[0, i].item() == -1.:
                    losses += 1
                    if replay_loss:
                        for s in history:
                            print(self.mdp.board_str(s)[0])
                            input()
                elif r[0, i].item() == 0.:
                    ties += 1
                else:
                    unknowns += 1
            output.append((wins, losses, ties, invalids, unknowns))
            if verbose:
                print(f"Player {i} {wins} wins, {losses} losses, {ties} ties, {invalids} invalid moves, {unknowns} unknown results.")
        return output
            
        
    def minmax(self, state = None):
        if state == None:
            state = self.mdp.get_initial_state()

        actions = self.mdp.valid_action_filter(state)
        while len(torch.nonzero(actions).tolist()) > 0:
            a = self.mdp.get_random_action_from_filter(actions)
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
    def deep_q(self, lr: float, dq_episodes: int, episode_length: int, ramp_start: int, ramp_end: int, greed_start: float, greed_end: float, training_delay: int, sim_batch: int, train_batch: int, copy_interval_eps=1, save_interval=100, save_path=None, verbose=False, graph_smoothing=10, initial_log=""):

        dq_episodes, episode_length, ramp_start, ramp_end, sim_batch, train_batch, copy_interval_eps, training_delay = int(dq_episodes), int(episode_length), int(ramp_start), int(ramp_end), int(sim_batch), int(train_batch), max(1, int(copy_interval_eps)), int(training_delay)
        expl_start, expl_end = 1. - greed_start, 1. - greed_end

        do_logging = True if verbose or save_path != None else False

        if train_batch < 2:
            Exception("Training batch size must be greater than 1 for sampling.")

        # Logging
        logtext = initial_log
        if do_logging:
            logtext += log(f"Device: {self.device}", verbose)
            logtext += log(f"MDP:\n{self.mdp}", verbose)
            logtext += log(f"Model:\n{self.qs[0].q}", verbose)
            logtext += log(f"Loss function:\n{self.loss_fn}", verbose)
            logtext += log(f"Optimizer:\n{self.optimizer}", verbose)
            logtext += log(f"Learn rate: {lr}, episodes: {dq_episodes}, start and end exploration: [{expl_start}, {expl_end}], ramp: [{ramp_start}, {ramp_end}], training delay: {training_delay}, episode length: {episode_length}, batch size: {sim_batch}, training batch size: {train_batch}, copy frequency: {copy_interval_eps}, memory capacity: {self.memory_capacity}.", verbose)
            
            episode_losses = [[] for i in range(self.mdp.num_players)]
            tests = []                                          # Format: tests[i][j] is the jth iteration of the ith test
            initial_test = self.mdp.tests(self.qs)
            for i in range(len(initial_test)):
                tests.append([initial_test[i]])
            start_time = datetime.datetime.now()
            logtext += log(f"Starting training at {start_time}\n", verbose)
                        
        # Initialize target network
        target_qs = [copy.deepcopy(self.qs[i]) for i in range(self.mdp.num_players)]
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
                # To do this, we need to iterate over players, because each has a different q function for determining the strategy
                p = self.mdp.get_player(s)

                # Get the actions using each player's policy #TODO can make this more efficient
                a = torch.zeros((sim_batch, ) + self.mdp.action_shape, device=self.device)
                for pi in range(self.mdp.num_players):
                    # Get the indices corresponding to this player's turn
                    indices = torch.arange(sim_batch, device=self.device)[p.flatten() == pi]
                    a[indices] = greedy_tensor(self.qs[pi], s[indices], expl_cur)


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
                        losses[i] += self.qs[i].update(self.memories[i].sample(train_batch), lr, optimizer_class=self.optimizer, loss_fn=self.loss_fn, target_q=target_qs[i])
                        num_updates[i] += train_batch

                # Set the state to the next state
                s = t

            
            # Copy the target network to the policy network if it is time
            if ep_num > training_delay and (ep_num+1) % copy_interval_eps == 0:
                for i in range(self.mdp.num_players):
                    target_qs[i].q.load_state_dict(self.qs[i].q.state_dict())
                logtext += log("Copied policy to target networks for all players.", verbose)

            # Save a copy of the model if time
            if save_path != None and (ep_num+1) % save_interval == 0:
                self.save_q(f"{save_path}.{ep_num+1}")
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
                test_results = self.mdp.tests(self.qs)
                for j in range(len(test_results)):
                    tests[j].append(test_results[j])

            


        if do_logging:
            end_time = datetime.datetime.now()
            logtext += log(f"\nTraining finished at {end_time}")
            logtext += log(f"Total time: {end_time - start_time}")

            simulation_results = self.simulate_against_random(self.mdp.num_simulations)
            for i in range(self.mdp.num_players):
                logtext += log(f"In {self.mdp.num_simulations} simulations by player {i}, {simulation_results[i][0]} wins, {simulation_results[i][1]} losses, {simulation_results[i][2]} ties, {simulation_results[i][3]} invalid moves, {simulation_results[i][4]} unknown results.")

        if save_path != None:
            # Save final model
            self.save_q(save_path)
            logtext += log(f"Saved final model to {save_path}")
            
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

            logpath = save_path + ".log"
            with open(logpath, "w") as f:
                logtext += log(f"Saved logs to {logpath}", verbose)
                f.write(logtext)




class DMCTS(DeepRL):

    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, model_args = {}, device="cpu"):
        super().__init__(mdp, model, loss_fn, optimizer, model_args, device)
        
        self.pv = model()
        self.q = {}
        self.n = {}         # Keys: hashable states.  Values: tensor with shape of actions
        self.w = {}
        self.p = {}



    # Unbatched
    def ucb(self, state, param: float):
        statehash = self.mdp.state_to_hashable(state)
        if statehash not in self.q:
            return 0
        return self.q[statehash] + param * math.sqrt(torch.sum(self.n[statehash])) / (1 + self.n[statehash])

    # Unbatched
    def search(self, state, ucb_parameter: int, p, num: int):
        history = [state]

        # Go down tree
        while self.mdp.state_to_hashable(state) in self.q:
            ucb = self.ucb(state, ucb_parameter)
            action = (ucb == ucb.max()).float()
            state, r = self.mdp.transition(state, action)
            history.append(state)
        
        # Once we reach a leaf
        self.q[self.mdp.state_to_hashable(state)] = torch.zeros(self.mdp.action_shape)
        self.n[self.mdp.state_to_hashable(state)] = torch.zeros(self.mdp.action_shape)
        self.w[self.mdp.state_to_hashable(state)] = torch.zeros(self.mdp.action_shape)
        self.p[self.mdp.state_to_hashable(state)] = p(state)
        while self.mdp.terminal(state) == False:
            state, r = self.mdp.transition(state, p(state))
            history.append(state)

        # Back-update
        while len(history) > 0:
            state = history.pop()
            self.n[self.mdp.state_to_hashable(state)] += 1
            self.w[self.mdp.state_to_hashable(state)] += r[self.mdp.get_player(state)]
            self.q[self.mdp.state_to_hashable(state)] = self.w[self.mdp.state_to_hashable(state)] / self.n[self.mdp.state_to_hashable(state)]
            

    # Unbatched
    def choose_action(self, prob_vector):
        prob_vector.flatten()
        pass    




    def mcts(self, lr: float, num_iterations: int, num_selfplay: int, num_searches: int, max_steps: int, ucb_parameter: float, sim_batch: int, train_batch: int, copy_interval_eps=1, save_interval=100, save_path=None, verbose=False, graph_smoothing=10, initial_log=""):
        prior_p = copy.deepcopy(self.pv)
        for i in range(num_iterations):
            s = self.mdp.get_initial_state(num_selfplay)
            for step in range(max_steps):
                a = torch.tensor([])
                for play in range(num_selfplay):
                    self.search(s[play], ucb_parameter, num=num_searches)
                    a = torch.cat([a, self.choose_action(self.n[self.mdp.state_to_hashable(s)])])
                s, r = self.mdp.transition(s, a)

                

            