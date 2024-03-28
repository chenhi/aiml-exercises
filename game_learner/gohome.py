import sys
import numpy as np
from qlearn import *
import random

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

def validate_int_pair(x, lower: int, upper: int) -> bool:
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], int) and isinstance(x[1], int):
        if lower == None or (lower != None and x[0] >= lower[0] and x[1] >= lower[1]):
            if upper == None or (upper != None and x[0] <= upper[0] and x[1] <= upper[1]):
                return True
    return False
        

# Create a simple MDP.  It's a 6x6 grid with a robot on it trying to go home.

class GoHomeMDP(MDP):
    def __init__(self, size: list, start = None, home = None, discount=1):
        if validate_int_pair(size, (1,1), None):
            self.size = size
        else:
            raise Exception("Invalid size tuple.")        
        
        if validate_int_pair(start, (0,0), self.size):
            self.start = start
        else:
            self.start = self.get_random_state()

        if validate_int_pair(home, (0,0), self.size):
            self.home = home
        else:
            self.home = [random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)]

        actions = [(1,0), (-1, 0), (0, 1), (0, -1)]
        states = [(i, j) for i in range(self.size[0]) for j in range(self.size[1])]
        super().__init__(states, actions, discount)

    def get_actions(self, s):
        return self.actions
    
    def get_player(self, s):
        return 0

    def is_terminal(self, s):
        return True if s == self.home else False

    def transition(self, s, a):
        if self.is_terminal(s):
            return s, 0
        #if s in self.states and a in self.states:
        t = (s[0] + a[0], s[1] + a[1])
        # If invalid move
        if t[0] < 0 or t[1] < 0 or t[0] >= self.size[0] or t[1] >= self.size[1]:
            return s, 0
        return t, 1 if self.is_terminal(t) else 0
    
    def get_initial_state(self):
        return self.start


    
# get command line options
options = sys.argv[1:]

home = (3,3)
mdp = GoHomeMDP((6,6), (0,0), home, 0.9)


# Test: play the MDP
if 'play' in options:
    s = mdp.get_initial_state()
    reward = 0
    history = [s]
    while not mdp.is_terminal(s):
        #print(s)
        a = mdp.get_random_action()
        s, r = mdp.transition(s, a)
        history.append(s)
        reward += r

    print(history, reward)

# Test: value function
q = ValueFunction(mdp)

# Test single update and val
if 'single' in options:
    s = (home[0] - 1, home[1])
    a = (1,0)
    t,r = mdp.transition(s, a)
    print(q.update([(s,a,r,t)], 0.5))
    print(q.get(s,a))
    print(q.val(s))
    print(q.val((0,0)))

# Test update
if 'update' in options:
    exp = 0.5
    lr = 0.5
    iter = 100000
    q.learn(get_greedy(q, exp), lr, iter)
    output = [[0. for j in range(0, mdp.size[1])] for i in range(0, mdp.size[0])]
    for i in range(0, mdp.size[0]):
        for j in range(0, mdp.size[1]):
            output[i][j] = q.val((i, j))
    print(q.q)
    
    heat = np.round(np.array(output), 3)
    print(heat)

    fig, ax = plt.subplots()
    im = ax.imshow(heat)
    for i in range(mdp.size[0]):
        for j in range(mdp.size[1]):
            text = ax.text(j, i, heat[i, j],
                        ha="center", va="center", color="w")
    ax.set_title(f"Learning rate {lr}, Explore {exp}, Iterations {iter}")
    plt.show()

#Test batch update, policy
if 'batch' in options:
    exp = 0.5
    lr = 0.5
    iter = 100
    eps = 10
    eplen = 15
    q.batch_learn(get_greedy(q, exp), lr, iter, eps, eplen)
    output = [[0. for j in range(0, mdp.size[1])] for i in range(0, mdp.size[0])]
    for i in range(0, mdp.size[0]):
        for j in range(0, mdp.size[1]):
            output[i][j] = q.val((i, j))
    print(q.q)
    
    heat = np.round(np.array(output), 3)
    print(heat)

    fig, ax = plt.subplots()
    im = ax.imshow(heat)
    for i in range(mdp.size[0]):
        for j in range(mdp.size[1]):
            text = ax.text(j, i, heat[i, j],
                        ha="center", va="center", color="w")
    ax.set_title(f"lr {lr}, expl {exp}, its {iter}, eps {eps}, eplen {eplen}")
    plt.show()

    pol = [[0. for j in range(0, mdp.size[1])] for i in range(0, mdp.size[0])]
    dumpol = [[0. for j in range(0, mdp.size[1])] for i in range(0, mdp.size[0])]
    for i in range(0, mdp.size[0]):
        for j in range(0, mdp.size[1]):
            pol[i][j] = q.policy((i, j))
            dumpol[i][j] = pol[i][j][0] + pol[i][j][1]
    print(pol)
    print(np.array(dumpol))
