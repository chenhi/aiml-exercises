import random, warnings
import numpy as np
import pickle
import torch
from torch import nn
from collections import namedtuple, deque

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

# Source, action, target, reward
TransitionData = namedtuple('TransitionData', ('s', 'a', 't', 'r'))

# Keeps the first action; the assumption is the later actions are "passive" (i.e. not performed by the given player)
# Adds the rewards
def compose_transition_tensor(first: TransitionData, second: TransitionData):
    if first.t != second.t:
        # Make this error non-fatal but make a note
        warnings.warn("The source and target states do not match.")
    return TransitionData(first.s, first.a, second.t, first.r + second.r)



# When the deque hits memory limit, it removes the oldest item from the list
class ExperienceReplay():
    def __init__(self, memory: int):
        self.memory = deque([], maxlen=memory)

    # Push a single transition (comes in the form of a list)
    def push(self, s, a, r, t):
        self.memory.append(TransitionData(s,a,r,t))

    def push_batch(self, source_batch, action_batch, target_batch, reward_batch):
        if source_batch.size(0) != action_batch.size(0) or action_batch.size(0) != target_batch.size(0) or target_batch.size(0) != reward_batch.size(0):
            raise Exception("Batch sizes do not agree.")
        batch_size = source_batch.size(0)
        for i in len(batch_size):
            self.push(source_batch[i, ...], action_batch[i, ...], target_batch[i, ...], reward_batch[i, ...])

            

    def sample(self, num: int, batch=True) -> list:
        if batch:
            return torch.stack(random.sample(self.memory, num))
        else:
            return random.sample(self.memory, num)       
    
    def __len__(self):
        return len(self.memory)
    


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
            return True if state.shape == self.state_shape else False
        else:
            return True
        
    # Only need to define this if you are doing deep learning
    def state_to_tensor(self, state):
        raise NotImplementedError
    
    def action_to_tensor(self, action):
        raise NotImplementedError
    
    # Action first because it's more similar to "batches"
    def to_tensor(self, state, action):
        return np.outer(self.action_to_tensor(action), self.state_to_tensor(state))

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

    def get_random_action(self, s = None):
        return random.choice(self.get_actions(s))

    # Returns: (next state, reward)
    def transition(self, s, a):
        raise NotImplementedError

    # Returns a valid start state and starting player for the game.
    def get_initial_state(self):
        raise NotImplementedError

    # Tells us if a state is terminal.
    def is_terminal(self, s) -> bool:
        raise NotImplementedError
    
    # Gets the current player of the given state.  TODO batch vs. non-batch
    def get_player(self, state) -> int:
        raise NotImplementedError


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





# Essentially instantiates a model and contains some methods to run it.
# Also, everything should be vectorized. TODO The mdp can't be?
class NNQFunction(QFunction):
    def __init__(self, mdp: MDP, q_model, lossfn, optimizer: torch.optim.Optimizer):
        self.q = q_model()
        self.mdp = mdp
        self.lossfn = lossfn
        self.optimizer = optimizer

    # Input has shape (batches, ) + state_shape and (batches, ) + action_shape
    def get(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The neural network produces a tensor of shape (batches, ) + action_shape
        pred = self.q(self.mdp.state_to_tensor(state))
        return pred * action

    # Input is a tensor of shape (batches, ) + state.shape
    # Output is a tensor of shape (batches, )
    def val(self, state) -> torch.Tensor:
        if self.mdp.is_terminal(state):
            return 0
        return torch.flatten(self.q(self.mdp.state_to_tensor(state)), 1, -1).max(1).values
    
    # Input indices shape (batch, linear length)
    def flattened_index_to_shaped(self, indices: torch.Tensor, shape: tuple):
        shapevol = 1
        for s in shape:
            shapevol *= s
        if indices.size(1) != shapevol:
            raise Exception("Linear length does not match shape volume.")
        # Index [a,b,c] in shape (x, y, z) has linear index t = a + bx + cxy, so invert: a = t mod x, b = (t - a)//x mod y, c = (t - a - bx)//xy mod z
        factors = []
        units = []
        shapeprod = 1
        residue = torch.zeros(indices.shape, dtype=float)
        for i in range(len(shape)):
            val = ((indices - residue)//shapeprod) % shape[i]
            # Shape (batch, shape_i)
            factors.append(val)
            residue += val
            shapeprod *= shape[i]
            # Create a tuple (1, 1, ..., s_i, 1, ...)
            l = [1 for j in len(shape)]
            l[i] = s[i]
            units.append(tuple(l))

        # Tensor up, move batch to last index
        output = torch.tensor([], dtype=float)
        for i in range(len(shape)-1, -1, -1):
            output = output * factors[i].reshape(units[i])
        return output

    # Input is a batch of state vectors
    # Returns the value of an optimal policy at a given state, shape (batches, ) + action_shape
    def policy(self, state) -> torch.Tensor:
        if self.mdp.is_terminal(s):
            return self.mdp.get_random_action()
        # The Q function outputs a tensor of shape (batch, ) + 
        return self.flattened_index_to_shaped(torch.flatten(self.q(self.mdp.state_to_tensor(state)), 1, -1).max(1).indices, self.mdp.action_shape)


    # TODO the following needs to be changed
    # Does a Q-update based on some observed set of data
    # Data is a list of the form (state, action, reward, next state)
    def update(self, data: torch.Tensor, learn_rate):
        # TODO ??
        if not isinstance(self.q, nn.Module):
            Exception("NNQFunction needs to have a class extending nn.Module.")
        opt = self.optimizer(self.q.parameters(), lr=learn_rate)
        X, y = data     # Data should be a tuple (tensor shape (batches, state shape, action shape), tensor shape (batches, reward, next state))
        # TODO Actually have to compute y
        pred = self.q(X)
        # Don't call val, it means re-evaluating.  Just do it here.
        #vals = pred.argmax(dim=1)


        loss = self.lossfn(pred, y)

        # Optimize
        loss.backward()
        opt.step()
        opt.zero_grad()



# Copy the weights of one NN to another
def copy(source: NNQFunction, target: NNQFunction):
    target.q.load_state_dict(source.q.state_dict())

# Greedy function to use as a strategy.  Default is totally greedy.
# This only works if the set of actions is defined and finite
def greedy(q: QFunction, s, e = 0.):
    return q.mdp.get_random_action(s) if random.random() < e else q.policy(s)


# Input shape (batch_size, ) + state_tensor
# We implement a random tensor of 0's and 1's generating a random tensor of floats in [0, 1), then converting it to a bool, then back to a float.
def greedy_tensor(q: NNQFunction, state, eps = 0.):
    return (torch.rand(q.mdp.action_shape) < 0.5).to(dtype=float) if random.random() < e else q.policy(state)


def get_greedy(q: QFunction, e: float) -> callable:
    return lambda s: greedy(q, s, e)

def get_greedy_tensor(eps: float) -> callable:
    return lambda q, s: greedy_tensor(q, s, eps)


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




# For now, for simplicity, fix a single strategy
class DQN():
    def __init__(self, mdp: MDP, q_model: nn.Module, memory_capacity: int):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.target_qs = [NNQFunction(mdp, q_model) for i in range(mdp.num_players)]
        self.policy_qs = [NNQFunction(mdp, q_model) for i in range(mdp.num_players)]
        self.memories = [ExperienceReplay(memory_capacity) for i in range(mdp.num_players)]

    def set_greed(self, eps):
        # if type(eps) == list:
        #     self.strategies = [get_greedy_tensor(self.qs[i], eps[i]) for i in range(self.mdp.num_players)]
        # else:
        #     self.strategies = [get_greedy_tensor(self.qs[i], eps) for i in range(self.mdp.num_players)]
        self.strategy = get_greedy_tensor(eps)

    def save_q(self, fname):
        for i in range(self.mdp.num_players):
            model_scripted = torch.jit.script(self.policy_qs[i])
            model_scripted.save(fname + f".p{i}")

    def load_q(self, fname):
        for i in range(self.mdp.num_players):
            self.policy_qs[i] = torch.jit.load(fname + f"p{i}")
            self.target_qs[i] = torch.jit.load(fname + f"p{i}")
            self.policy_qs[i].eval()					# Set to evaluation mode

    # Handling multiplayer: each player keeps their own "record", separate from memory
    # When any entry in the record has source = target, then the player "banks" it in their memory
    # The next time an action is taken, if the source = target, then it gets overwritten
    def deep_learn(self, learn_rate: float, iterations: int, episodes: int, episode_length: int, batch_size: int, verbose=False):
        for i in range(iterations):
            for j in range(episodes):
                s = self.mdp.get_initial_state(batch_size)
                # "Records" for each player
                player_source = [None for i in self.mdp.num_players]
                player_action = [None for i in self.mdp.num_players]
                player_target = [None for i in self.mdp.num_players]
                player_reward = [None for i in self.mdp.num_players]

                for k in range(episode_length):  
                    # Execute the transition on the "actual" state
                    # To do this, we need to iterate over players, because each has a different q function for determining the strategy
                    p = self.mdp.get_player(s)                                  # p.shape = (batch, )       s.shape = (batch, 2, 6, 7)

                    # To get the actions, we need to use each player's strategy # TODO ok it's not hard to make this so each player has its own strategy...
                    a = torch.zeros((batch_size, ) + self.mdp.action_shape)
                    for i in range(self.mdp.num_players):
                        # TODO can we make this more efficient, i.e. just ignore the other ones and put it back in?
                        player_a = self.strategy(s)

                        a += player_a * (p == float(i))[:, None]                    # Zero out actions on other players' turns # TODO how to broadcast to right
                        
                    t, r = self.mdp.transition(s, a)

                    # Update player records
                    # If a current record has player(s) = player(t), replace it
                    
                    # Update memory
                    # If an updated record has player(s) = player(t), put it into memory



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