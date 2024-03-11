import random
import numpy as np
import pickle
import torch
from torch import nn
from collections import namedtuple

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



# WARNING: states and actions must be hashable!
class MDP():
    def __init__(self, states, actions, discount=1):
        self.states = states
        self.actions = actions
        self.discount = discount   

    def copy(self):
        raise NotImplementedError

    def is_state(self, s):
        return True if self.states == None or s in self.states else False
    
    def is_action(self, a):
        return True if self.actions == None or a in self.actions else False
    
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



# Value function Q
# Given a set of states S and a set of actions A, a value function maps Q: S x A ---> R and reflects the value (in terms of rewards) of performing a given action at the state.
# The sets S and A might be infinite.  Therefore in practice we do not require Q to be defined everywhere.
# We may or may not specify a set from which 
class ValueFunction():

    def __init__(self, mdp: MDP):
        self.q = {}
        self.mdp = mdp

    def copy(self):
        new_q = ValueFunction(self.mdp.copy())
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

        # If the set of actions is undefined and infinite, we don't have much control over things.
        # We will assume that there is always some choice that hasn't been explored yet, which therefore has value 0.
        # This is generally not going to be true; it's better to do some kind of regression on the states we know.
        # The result is that states will not have negative value, only zero value.  This can cause the AI to claim an immediate reward but then enter a really bad state.
        values = []
        for t, a in self.q.keys():
            if s == t:
                values.append(self.q[(s,a)])
        #values = [self.q[(s,a)] if s == t else 0 for t, a in self.q.keys()]
        
        if len(values) == 0:
            return 0
        m = max(values)
        if max(values) < 0:
            return 0
        return max(values)
        
    # Returns the value of an optimal policy at a given state
    def policy(self, s) -> float:
        if self.mdp.is_terminal(s):
            return self.mdp.get_random_action()

        # If we have a defined set of actions, we can just do an argmax
        if self.mdp.actions != None:
            return random.choice(argmax(self.mdp.get_actions(s), lambda a: self.get(s,a)))
        
        # Same issue as val() when the actions are not defined.
        # The result is that when enterng a state that only has negative known values, the AI will choose a random action.
        valid = []
        maxim = None
        for t, a in self.q.keys():
            if s == t:
                valid.append(((s, a), self.q[(s,a)]))
                if maxim == None:
                    maxim = valid[-1][1]
                else:
                    maxim = max(maxim, valid[-1][1])            
            
        if len(valid) == 0 or maxim < 0:
            return self.mdp.get_random_action()
        
        best = []
        for k, v in valid:
            if v == maxim:
                best.append(k[1])
        
        return random.choice(best)
        

    # Does a Q-update based on some observed set of data
    # Data is a list of the form (state, action, reward, next state)
    def update(self, data: list[tuple[any, any, float, any]], learn_rate):
        for d in data:
            s,a,r,t = d[0], d[1], d[2], d[3]
            self.q[(s,a)] = (1 - learn_rate) * self.get(s,a) + learn_rate * (r + self.mdp.discount * self.val(t))

    # Does a single Q update using the reward and new state coming from the attached mdp.  Returns the new state. 
    # If the input state is terminal, just return a new state
    def single_update(self, s, a, learn_rate):
        if self.mdp.is_terminal(s):
            return self.mdp.get_initial_state()
        t, r = self.mdp.transition(s, a)
        self.update([(s, a, r, t)], learn_rate)
        return t

    # Learn based on a give nstrategy for some number of iterations
    def learn(self, strategy: callable, learn_rate: float, iterations: int):
        s = self.mdp.get_initial_state()
        for i in range(iterations):
            s = self.single_update(s, strategy(s), learn_rate)
            if self.mdp.is_terminal(s):
                s = self.mdp.get_initial_state()

    def batch_learn(self, strategy: callable, learn_rate: float, iterations: int, episodes: int, episode_length: int):
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
                

class NNValueFunction(ValueFunction):
    def __init__(self, mdp: MDP, nnq: nn.Module):
        self.nnq = nnq
        self.mdp = mdp

    def to_tensor(s, a):
        raise NotImplementedError

    def get(self, s, a) -> float:
        return self.nnq(self.to_tensor(s, a))


# Greedy function to use as a strategy.  Default is totally greedy.
# This only works if the set of actions is defined and finite
def greedy(q: ValueFunction, s, e = 0.):
    return q.mdp.get_random_action(s) if random.random() < e else random.choice(argmax(q.mdp.get_actions(s), lambda a: q.get(s,a)))

def get_greedy(q: ValueFunction, e: float) -> callable:
    return lambda s: greedy(q, s, e)




# To each player, the game will act like a MDP; i.e. they do not distinguish between the opponents and the environment
# We can only batch learn, since there is a lag in rewards updating
# Player order can make a difference, so when training, do not shuffle the order

# Requirements for the MDP:
# The state must be a tuple, whose first entry is the current player
# The rewards are returned as a tuple, corresponding to rewards for each player
class SimpleGame():
    def __init__(self, mdp: MDP, num_players):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.num_players = num_players
        self.qs = [ValueFunction(self.mdp) for i in range(num_players)]
        self.state = None

    def set_greed(self, eps):
        self.strategies = [get_greedy(self.qs[i], eps[i]) for i in range(self.num_players)]

    def save_q(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.qs, f)
        
    def load_q(self, fname):
        with open(fname, 'rb') as f:
            self.qs = pickle.load(f)

    def transition(self, p, s, a) -> tuple[int, object, np.ndarray]:
        return self.mdp.transition((p, s), a)
    
    def current_player(self, s) -> int:
        return None if s == None else s[0]

    # Player data is (start state, action taken, all reward before next action, starting state for next action)
    def batch_learn(self, learn_rate: float, iterations: int, episodes: int, episode_length: int, verbose=False, savefile=None):
        player_experiences = [[] for i in range(self.num_players)]
        for i in range(iterations):
            for j in range(episodes):
                if verbose and j % 10 == 9:
                    print(f"Training iteration {i+1}, episode {j+1}", end='\r')
                s = self.mdp.get_initial_state()
                queue =[None for k in range(self.num_players)]
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
                    for l in range(self.num_players):
                        if l != p and queue[l] != None:
                            queue[l][2] += r[l]
                            if self.mdp.is_terminal(t):
                                player_experiences[l].append(tuple(queue[l]) + (t,))

                    # If terminal state, then stop the episode.  Otherwise, update state and continue playing 
                    if self.mdp.is_terminal(t):
                        break
                    s = t
            # Do an update for each player
            for p in range(self.num_players):
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
