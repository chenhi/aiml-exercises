import random, warnings
import numpy as np
import pickle, zipfile, os
import torch
from torch import nn
from collections import namedtuple, deque





# WARNING: states and actions must be hashable if using with QFunction.
class MDP():
    def __init__(self, states, actions, discount=1, num_players=1, state_shape = None, action_shape = None, batched=False):
        self.actions = actions
        self.discount = discount
        self.num_players = num_players
        self.batched = batched

        # The states do not literally have to be tensors for the state_shape to be defined.  This just specifies, when they are turned into tensors, what shape they should be have, ignoring batch dimension
        self.states = states
        self.state_shape = state_shape
        self.actions = actions
        self.action_shape = action_shape
        

    def copy(self):
        raise NotImplementedError

    def is_state(self, state):
        if self.states != None:
            return True if state in self.states else False
        elif self.state_shape != None:
            return True if state[0].shape == self.state_shape else False
        else:
            return True
        
    def is_action(self, action):
        if self.actions != None:
            return True if action in self.actions else False
        elif self.action_shape != None:
            return True if action[0].shape == self.action_shape else False
        else:
            return True

    # Takes in the "full" state and returns the status of the game
    def board_str(self, s) -> str:
        raise NotImplementedError
        
    def get_random_state(self):
        if self.states != None:
            return random.choice(self.states)
        raise NotImplementedError

    # Can re-implement this to take the state into consideration to avoid invalid moves    
    def get_actions(self, s = None):
        if s == None:
            return self.actions
        else:
            raise NotImplementedError

        
    def is_valid_action(self, state, action):
        return action in self.get_actions(state)

    def get_random_action(self, state = None):
        return random.choice(self.get_actions(state))

    # Returns: (next state, reward)
    def transition(self, s, a):
        raise NotImplementedError

    # Returns a valid start state and starting player for the game.
    def get_initial_state(self):
        raise NotImplementedError

    # Tells us if a state is terminal.
    def is_terminal(self, s):
        raise NotImplementedError
    
    # Gets the current player of the given state.
    def get_player(self, state):
        raise NotImplementedError




#################### CLASSICAL Q-LEARNING ####################


# Returns the set of maximum arguments
def argmax(args: list, f: callable):
    maxval = None
    output = []
    for x in args:
        y = f(x)
        if maxval == None:
            output.append(x)
            maxval = y
        elif y == maxval:
            output.append(x)
        elif y > maxval:
            maxval = y
            output = [x]
    return output

def valmax(args: list, f: callable):
    maxval = None
    for x in args:
        y = f(x)
        if maxval == None or y > maxval:
            maxval = y
    return maxval


# Q (Quality) function class
# Given a set of states S and a set of actions A, a value function maps Q: S x A ---> R and reflects the value (in terms of rewards) of performing a given action at the state.
# The sets S and A might be infinite.  Therefore in practice we do not require Q to be defined everywhere.
# We may or may not specify a set from which 
class QFunction():

    def __init__(self, mdp: MDP):
        self.q = {}
        self.mdp = mdp

    def copy(self):
        new_q = QFunction(self.mdp.copy())
        new_q.q = self.q.copy()
    
    def get(self, s, a) -> float:
        return 0 if (s, a) not in self.q else self.q[(s, a)]

    # Returns the value at a given state, i.e. max_a(Q(s, a))
    # Value of terminal state should always be 0
    def val(self, s) -> float:
        if self.mdp.is_terminal(s):
            return 0
        
        # If we have a defined (finite) set of actions, just iterate
        if self.mdp.actions != None:
            return valmax(self.mdp.get_actions(s), lambda a: self.get(s, a))
        else:
            raise NotImplementedError           #If there are infinitely many actions, this needs to be handled explicitly
    
    # Returns a list of optimal policies.  If the state is terminal or there are no valid actions, return empty list.
    # This is probably only ever called internally.  But separating it anyway.
    def policies(self, state):
        if self.mdp.is_terminal(state):
            return []
        
        # If we have a defined set of actions, we can just do an argmax.
        if self.mdp.actions != None:
            return argmax(self.mdp.get_actions(state), lambda a: self.get(state,a))
        else:
            raise NotImplementedError


    # Returns a randomly selected optimal policy.
    def policy(self, s) -> float:
        pols = self.policies(s)
        if len(pols) == 0:
            if self.mdp.actions != None:
                pols = self.mdp.actions
            else:
                raise NotImplementedError
        return random.choice(pols)

    # Does a Q-update based on some observed set of data
    # Data is a list of the form (state, action, reward, next state)
    def update(self, data: list[tuple[any, any, float, any]], learn_rate):
        for d in data:
            s,a,r,t = d[0], d[1], d[2], d[3]
            self.q[(s,a)] = (1 - learn_rate) * self.get(s,a) + learn_rate * (r + self.mdp.discount * self.val(t))

    # Learn based on a given strategy for some number of iterations, updating each time.
    # In practice, this doesn't get used so much, because the "game" has to handle rewards between players (not the Q function itself)
    def learn(self, strategy: callable, learn_rate: float, iterations: int):
        s = self.mdp.get_initial_state()
        for i in range(iterations):
            if self.mdp.is_terminal(s):
                s = self.mdp.get_initial_state()
                continue
            a = strategy(s)
            t, r = self.mdp.transition(s, a)
            self.update([(s,a,r,t)], learn_rate)
            s = t

    # Learn in batches.  An update happens each iteration, on all past experiences (including previous iterations).  A state reset happpens each episode.
    # In practice, this doesn't get used so much, because the "game" has to handle rewards between players (not the Q function itself)
    def batch_learn(self, strategy: callable, learn_rate: float, iterations: int, episodes: int, episode_length: int, remember_experiences = True):
        experiences = []
        for i in range(iterations):
            for j in range(episodes):
                s = self.mdp.get_initial_state()
                for k in range(episode_length):
                    a = strategy(s)
                    t, r = self.mdp.transition(s, a)
                    experiences.append((s, a, r, t))
                    if self.mdp.is_terminal(t):
                        break
                    s = t
            self.update(experiences, learn_rate)
            if not remember_experiences:
                experiences = []
                


# For backwards compatibility
class ValueFunction(QFunction):
    def __init__(self, mdp: MDP):
        super().__init__(mdp)





# Greedy function to use as a strategy.  Default is totally greedy.
# This only works if the set of actions is defined and finite
def greedy(q: QFunction, state, eps = 0.):
    return q.mdp.get_random_action(state) if random.random() < eps else q.policy(state)




def get_greedy(q: QFunction, eps: float) -> callable:
    return lambda s: greedy(q, s, eps)



# To each player, the game will act like a MDP; i.e. they do not distinguish between the opponents and the environment
# We can only batch learn, since there is a lag in rewards updating
# Player order can make a difference, so when training, do not shuffle the order

# Requirements for the MDP:
# The state must be a tuple, whose first entry is the current player
# The rewards are returned as a tuple, corresponding to rewards for each player
class SimpleGame():
    def __init__(self, mdp: MDP):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.qs = [QFunction(self.mdp) for i in range(mdp.num_players)]
        self.state = None

    def set_greed(self, eps):
        if type(eps) == list:
            self.strategies = [get_greedy(self.qs[i], eps[i]) for i in range(self.mdp.num_players)]
        else:
            self.strategies = [get_greedy(self.qs[i], eps) for i in range(self.mdp.num_players)]

    def save_q(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.qs, f)
        
    def load_q(self, fname):
        with open(fname, 'rb') as f:
            self.qs = pickle.load(f)
    
    # Non-batched method
    def current_player(self, s) -> int:
        if s == None:
            return None
        if self.mdp.batched:
            return self.mdp.get_player(s).item()
        else:
            return self.mdp.get_player(s)

    # Player data is (start state, action taken, all reward before next action, starting state for next action)
    def batch_learn(self, learn_rate: float, iterations: int, episodes: int, episode_length: int, verbose=False, savefile=None):
        player_experiences = [[] for i in range(self.mdp.num_players)]
        for i in range(iterations):
            for j in range(episodes):
                if verbose and j % 10 == 9:
                    print(f"Training iteration {i+1}, episode {j+1}", end='\r')
                s = self.mdp.get_initial_state()
                queue =[None for k in range(self.mdp.num_players)]
                for k in range(episode_length):
                    p = self.current_player(s)
                    a = self.strategies[p](s)
                    t, r = self.mdp.transition(s, a)

                    # For this player, bump the queue and add
                    if queue[p] != None:
                        player_experiences[p].append(tuple(queue[p]) + (s,))
                    if self.mdp.is_terminal(t):
                        player_experiences[p].append((s,a,r[p],t))
                    else:
                        queue[p] = [s, a, r[p]]
                    

                    # Update rewards for all other players; if the player hasn't taken an action yet, no reward (but is accounted somewhat by zero sum nature)
                    # If the state is terminal, also append
                    for l in range(self.mdp.num_players):
                        if l != p and queue[l] != None:
                            queue[l][2] += r[l]
                            if self.mdp.is_terminal(t):
                                player_experiences[l].append(tuple(queue[l]) + (t,))

                    # If terminal state, then stop the episode.  Otherwise, update state and continue playing 
                    if self.mdp.is_terminal(t):
                        break
                    s = t
            # Do an update for each player
            for p in range(self.mdp.num_players):
                self.qs[p].update(player_experiences[p], learn_rate)
        if verbose:
            total = 0
            for e in player_experiences:
                total += len(e)
            print(f"Trained on {total} experiences.")
        if savefile != None:
            with open(savefile, 'wb') as f:
                pickle.dump(player_experiences, f)
                if verbose:
                    print(f"Saved experiences to {savefile}")























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
    def push(self, s, a, r, t):
        self.memory.append(TransitionData(s,a,t,r))

    def push_batch(self, data: TransitionData):
        if data.s.size(0) != data.a.size(0) or data.a.size(0) != data.t.size(0) or data.t.size(0) != data.r.size(0):
            raise Exception("Batch sizes do not agree.")
        batch_size = data.s.size(0)
        for i in range(batch_size):
            self.push(TransitionData(data.s[i, ...], data.a[i, ...], data.t[i, ...], data.r[i, ...]))

            

    def sample(self, num: int, batch=True) -> list:
        if batch:
            data = random.sample(self.memory, num)
            s = torch.stack([datum[0] for datum in data])
            return TransitionData(torch.stack([datum.s for datum in data]), torch.stack([datum.a for datum in data]), torch.stack([datum.t for datum in data]), torch.stack([datum.r for datum in data]))
        else:
            return random.sample(self.memory, num)       
    
    def __len__(self):
        return len(self.memory)
    






# A Q-function where the inputs and outputs are all tensors
class NNQFunction(QFunction):
    def __init__(self, mdp: MDP, q_model, loss_fn, optimizer_class: torch.optim.Optimizer):
        if mdp.state_shape == None or mdp.action_shape == None:
            raise Exception("The input MDP must handle tensors.")
        self.q = q_model()
        self.q.eval()
        self.mdp = mdp
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class

    # Input has shape (batches, ) + state_shape and (batches, ) + action_shape
    def get(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The neural network produces a tensor of shape (batches, ) + action_shape
        pred = self.q(state.float())
        # A little inefficient because we only take the diagonals, 
        return torch.tensordot(pred, action.float().T, 1).diagonal()

    # Input is a tensor of shape (batches, ) + state.shape
    # Output is a tensor of shape (batches, )
    def val(self, state) -> torch.Tensor:
        terminal = torch.flatten(self.mdp.is_terminal(state.float()))
        return torch.flatten(self.q(state.float()), 1, -1).max(1).values * ~terminal
    
    # Input is a batch of state vectors
    # Returns the value of an optimal policy at a given state, shape (batches, ) + action_shape
    def policy(self, state) -> torch.Tensor:
        flattened_indices = torch.flatten(self.q(state.float()), 1, -1).max(1).indices
        linear_dim = 0
        for d in self.mdp.action_shape:
            linear_dim += d
        indices_onehot = (torch.eye(linear_dim)[None] * torch.ones((state.size(0), 1, 1)))[torch.arange(state.size(0)), flattened_indices]
        return indices_onehot.reshape((state.size(0), ) + self.mdp.action_shape)


    # Does a Q-update based on some observed set of data
    # TransitionData entries are tensors
    def update(self, data: TransitionData, learn_rate):
        if not isinstance(self.q, nn.Module):
            Exception("NNQFunction needs to have a class extending nn.Module.")

        self.q.train()

        opt = self.optimizer(self.q.parameters(), lr=learn_rate)
        
        pred = self.get(data.s.float(), data.a)
        y = data.r + self.val(data.t.float())

        loss = self.loss_fn(pred, y)

        # Optimize
        loss.backward()
        opt.step()
        opt.zero_grad()

        self.q.eval()






# Input shape (batch_size, ) + state_tensor
# We implement a random tensor of 0's and 1's generating a random tensor of floats in [0, 1), then converting it to a bool, then back to a float.
def greedy_tensor(q: NNQFunction, state, eps = 0.):
    return q.mdp.get_random_action(state) if random.random() < eps else q.policy(state)

def get_greedy_tensor(q: NNQFunction, eps: float) -> callable:
    return lambda s: greedy_tensor(q, s, eps)



# For now, for simplicity, fix a single strategy
# Note that qs is policy_qs
class DQN():
    def __init__(self, mdp: MDP, q_model: nn.Module, loss_fn, optimizer, memory_capacity: int):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.target_qs = [NNQFunction(mdp, q_model, loss_fn, optimizer) for i in range(mdp.num_players)]
        self.qs = [NNQFunction(mdp, q_model, loss_fn, optimizer) for i in range(mdp.num_players)]
        self.memories = [ExperienceReplay(memory_capacity) for i in range(mdp.num_players)]

    # Note that greedy involves values, so it should refer to the target network
    def set_greed(self, eps):
        if type(eps) == list:
            self.strategies = [get_greedy_tensor(self.target_qs[i], eps[i]) for i in range(self.mdp.num_players)]
        else:
            self.strategies = [get_greedy_tensor(self.target_qs[i], eps) for i in range(self.mdp.num_players)]

    def save_q(self, fname):
        zf = zipfile.ZipFile(fname, mode="w")
        for i in range(self.mdp.num_players):
            model_scripted = torch.jit.script(self.qs[i].q)
            model_scripted.save(f"{i}.{fname}")
            zf.write(f"{i}.{fname}", f"{i}.pt", compress_type=zipfile.ZIP_STORED)
            os.remove(f"{i}.{fname}")
        zf.close()

            

    def load_q(self, fname):
        for i in range(self.mdp.num_players):
            self.qs[i].q = torch.jit.load(f"p{i}.{fname}")
            self.copy_policy_to_target()
            self.qs[i].q.eval()					# Set to evaluation mode


    # Keeps the first action; the assumption is the later actions are "passive" (i.e. not performed by the given player)
    # Adds the rewards
    # Returns a tuple: (composition, to_memory)
    def compose_transition_tensor(self, first: TransitionData, second: TransitionData, player_index: int):
        if torch.prod((first.t == second.s) == 0).item():
            # Make this error non-fatal but make a note
            warnings.warn("The source and target states do not match.")

        # Two cases: say (s,a,r,t) composed (s',a',r',t')
        # If player(t) = player_index, then (s',a',r',t') else (s,a,r+r',t')
        # Note once we reach terminal state, 
        filter = self.mdp.get_player(first.t) == player_index 
        new_s = filter * second.s + (filter == False) * first.s
        new_a = filter * second.a + (filter == False) * first.a
        new_r = second.r + (filter == False) * first.r
        new_t = second.t
            
        # After the above update, return rows where player(s) = player(t') = player_index OR where t' is terminal but s' is not (regardless of player)
        filter = (((self.mdp.get_player(first.s) == player_index) & (self.mdp.get_player(second.t) == player_index)) | (self.mdp.is_terminal(second.t) & ~self.mdp.is_terminal(second.s))).flatten()
        return (TransitionData(new_s, new_a, new_t, new_r), TransitionData(new_s[filter], new_a[filter], new_t[filter], new_r[filter]))

    # Copy the weights of one NN to another
    def copy_policy_to_target(self):
        for i in range(self.mdp.num_players):
            self.target_qs[i].q.load_state_dict(self.qs[i].q.state_dict())


    # Handling multiplayer: each player keeps their own "record", separate from memory
    # When any entry in the record has source = target, then the player "banks" it in their memory
    # The next time an action is taken, if the source = target, then it gets overwritten
    def deep_learn(self, learn_rate: float, episodes: int, episode_length: int, batch_size: int, train_batch_size: int, copy_frequency: int, verbose=False):
        for j in range(episodes):
            # Make sure the target network is the same as the policy network
            self.copy_policy_to_target()

            s = self.mdp.get_initial_state(batch_size)
            # "Records" for each player
            player_record = [None for i in range(self.mdp.num_players)]
            
            for k in range(episode_length):  
                # Execute the transition on the "actual" state
                # To do this, we need to iterate over players, because each has a different q function for determining the strategy
                p = self.mdp.get_player(s)                                  # p.shape = (batch, )       s.shape = (batch, 2, 6, 7)

                # To get the actions, apply each player's strategy
                a = torch.zeros((batch_size, ) + self.mdp.action_shape)
                for pi in range(self.mdp.num_players):
                    # Get the indices corresponding to this player's turn
                    indices = torch.arange(batch_size)[p.flatten() == float(pi)].tolist()
                    
                    # Apply thie player's strategy to their turns
                    # Greedy is not parallelized, so there isn't efficiency loss with this
                    player_actions = torch.cat([self.strategies[pi](s[l:l+1]) if l in indices else torch.zeros((1, ) + self.mdp.action_shape) for l in range(batch_size)], 0)
                    a += player_actions
                    #player_actions = self.strategies[i](s[indices, ...])
                    #a += player_actions * (p == float(pi))[:, None]


                # Do the transition                       
                t, r = self.mdp.transition(s, a)
                
                # Update player records and memory
                for pi in range(self.mdp.num_players):
                    # If it's the first move, just put the record in
                    if player_record[pi] == None:
                        player_record[pi] = TransitionData(s, a, t, r)
                    else:
                        player_record[pi], to_memory = self.compose_transition_tensor(player_record[pi], TransitionData(s, a, t, r), pi)
                        self.memories[pi].push_batch(to_memory)
                
                # Train the policy on a random sample in memory (once the memory bank is big enough)
                for i in range(self.mdp.num_players):
                    if self.memories[i].size() >= train_batch_size:
                        self.qs[i].update(self.memories[i].sample(train_batch_size), learn_rate)
                        
                # Copy the target network to the policy network if it is time
                if (k+1) % copy_frequency == 0:
                    self.copy_policy_to_target()
                    


# TODO carefuly about eval vs train
# TODO zip players