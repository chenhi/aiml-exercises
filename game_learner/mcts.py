import zipfile, os, random, warnings, datetime, copy, math
import torch
from torch import nn
from collections import namedtuple, deque
from qlearn import MDP, QFunction, PrototypeQFunction
from aux import log, smoothing
import matplotlib.pyplot as plt
from deepqlearn import TensorMDP, NNQFunction, DeepRL


class DMCTS(DeepRL):

    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, model_args = {}, device="cpu"):
        super().__init__(mdp, model, loss_fn, optimizer, model_args, device)
        
        self.pv = model()
        self.q = {}
        self.n = {}         # Keys: hashable states.  Values: tensor with shape of actions
        self.w = {}
        self.p = {}

    # Unbatched
    # def q_val(self, state):
    #     statehash = self.mdp.state_to_hashable(state)
    #     if statehash not in self.q:
    #         return torch.zeros((1,) + self.mdp.action_shape)
    #     return self.q[statehash]

    # Unbatched
    def ucb(self, state, param: float):
        statehash = self.mdp.state_to_hashable(state)
        if statehash not in self.q:
            return torch.zeros((1,) + self.mdp.action_shape)
        return (self.q[statehash].nan_to_num(0) + param * math.sqrt(torch.sum(self.n[statehash])) / (1 + self.n[statehash]))[None]

    # Unbatched
    def search(self, state, ucb_parameter: int, p: nn.Module, num: int):
        for i in range(num):
            history = []
            s = state

            # Go down tree, using upper confidence bound and dictionary q values to play
            while self.mdp.state_to_hashable(s) in self.q and self.mdp.is_terminal(s).item() == False:
                ucb = self.ucb(s, ucb_parameter)
                action = self.mdp.get_random_action_from_values(ucb * self.mdp.valid_action_filter(s))
                history.append((s, action))
                s, r = self.mdp.transition(s, action)
                
            
            # Once we reach a leaf, use p to simulate play
            if self.mdp.is_terminal(s).item() == False:
                self.q[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape)
                self.n[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape)
                self.w[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape)
                self.p[self.mdp.state_to_hashable(s)] = p(s)
            while self.mdp.is_terminal(s).item() == False:
                action = self.choose_action(torch.sigmoid(p(s)) * self.mdp.valid_action_filter(s))
                history.append((s, action))
                s, r = self.mdp.transition(s, action)
            
            # Back-update
            while len(history) > 0:
                s, action = history.pop()
                if self.mdp.state_to_hashable(s) in self.p:
                    self.n[self.mdp.state_to_hashable(s)] += action[0]
                    self.w[self.mdp.state_to_hashable(s)] += action[0] * r[0,self.mdp.get_player(s)[0]][0]
                    self.q[self.mdp.state_to_hashable(s)] = self.w[self.mdp.state_to_hashable(s)] / self.n[self.mdp.state_to_hashable(s)]
                

    # Batched
    # If divide by zero, get nan, then comparison gives False, so resulting vector is 0
    def choose_action(self, prob_vector):
        return (torch.cumsum(prob_vector.flatten(1, -1) / torch.sum(prob_vector.flatten(1,-1), 1), 1) > torch.rand(prob_vector.size(0),)[:,None]).diff(dim=1, prepend=torch.zeros((prob_vector.size(0),1))).reshape((-1,) + self.mdp.action_shape)

    def mcts(self, lr: float, num_iterations: int, num_selfplay: int, num_searches: int, max_steps: int, ucb_parameter: float, sim_batch: int, train_batch: int, copy_interval_eps=1, save_interval=100, save_path=None, verbose=False, graph_smoothing=10, initial_log=""):
        prior_p = copy.deepcopy(self.pv)
        training_data = []
        for i in range(num_iterations):
            iteration_data = []
            s = self.mdp.get_initial_state(num_selfplay)
            for step in range(max_steps):
                a = torch.tensor([])
                for play in range(num_selfplay):
                    self.search(s[play], ucb_parameter, p=prior_p, num=num_searches)
                    a = torch.cat([a, self.choose_action(self.n[self.mdp.state_to_hashable(s)])])
                s, r = self.mdp.transition(s, a)
                iteration_data.append()

                

            