import sys
import numpy as np
from deepqlearn import *
from qlearn import MDP, QFunction
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
        

# Simple test game of a robot trying to go home on an (n, n) grid.  The edges are cliffs and falling off is a negative reward and terminal state.
# States: shape (n, n) tensor, with a 1 at the robot's position.
# Actions: (4, ) tensor.  Convention: +x, -x, +y, -y
class GoHomeTensorMDP(MDP):
    def __init__(self, size: list, start = None, home = None, discount=1):
        if validate_int_pair(size, (1,1), None):
            self.size = size
        else:
            raise Exception("Invalid size tuple.")        
        
        if validate_int_pair(start, (0,0), self.size):
            self.start = start
        else:
            raise Exception("Invaid start.")

        if validate_int_pair(home, (0,0), self.size):
            self.home = home
        else:
            raise Exception("Invalid home.")

        super().__init__(None, None, discount, num_players=1, state_shape=(3, 3), action_shape=(2, 2), batched=True)

        self.initial_state = torch.zeros(self.size, dtype=int)
        self.initial_state[start] = 1

        self.home_state = torch.zeros(self.size, dtype=int)
        self.home_state[home] = 1

        #self.upper = torch.cat([torch.eye(self.size, dtype=int)[1:,:], torch.zeros(1, self.size, dtype=int)])
        #self.lower = torch.cat([torch.zeros(1, self.size, dtype=int), torch.eye(self.size, dtype=int)[:-1,:]])
        #self.left = torch.eye(2)[0,None,None] * self.upper + torch.eye(2)[1,None,None] * self.lower
        #self.right = torch.eye(2)[0,None,None] * self.lower + torch.eye(2)[1,None,None] * self.upper

    def get_player(self, state: torch.Tensor) -> torch.Tensor:
        return torch.ones(state.size(0))[:, None, None, None]
    
    def get_player_vector(self, state: torch.Tensor) -> torch.Tensor:
        return self.get_player(self, state)

    def get_random_action(self, state):
        indices = (torch.rand(state.size(0)) * 4).int()
        return torch.eye(4, dtype=int)[None].expand(state.size(0), -1, -1)[torch.arange(state.size(0)), indices]
    
    def get_initial_state(self, batch_size=1) -> torch.Tensor:
        return self.initial_state

    def is_valid_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        return torch.zeros(state.shape, dtype=int) == 0
    
    def transition(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass #TODO!!!

    def is_terminal(self, s):
        return (s == self.home_state) | (s == torch.zeros(self.size, dtype=int))

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
        return self.initial_state
    
    def get_random_state(self) -> torch.Tensor:
        return self.get_initial_state()


class GoHomeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding='same'),
            nn.ReLU(),
            #nn.Dropout(p=0.2),             # Dropout introduces randomness into the Q function.  Not sure if this is desirable.
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (5,5), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64*7*6, 64*7*6),
            nn.ReLU(),
            nn.BatchNorm1d(64*7*6),
            nn.Linear(64*7*6, 7),
            nn.BatchNorm1d(7)
        )
    
    def forward(self, x):
        return self.stack(x)


    
# get command line options
options = sys.argv[1:]

home = (3,3)
mdp = GoHomeTensorMDP((6,6), (0,0), home, 0.9)


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
