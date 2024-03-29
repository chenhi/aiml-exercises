import sys
import numpy as np
from deepqlearn import *
from qlearn import MDP, QFunction
import random

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl


# Simple test game of a robot trying to go home on an (n, n) grid.  The edges are cliffs and falling off is a negative reward and terminal state.
# States: shape (n, n) tensor, with a 1 at the robot's position.
# Actions: (2,) tensor
class GoHomeTensorMDP(MDP):
    def __init__(self, size: list, start = None, home = None, discount=1):
        self.size = size
        self.start = start
        self.home = home

        super().__init__(None, None, discount, num_players=1, state_shape=size, action_shape=(3, 3), batched=True)

        self.initial_state = torch.zeros(self.size)
        self.initial_state[start] = 1

        self.home_state = torch.zeros(self.size)
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