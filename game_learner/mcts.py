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
        self.n_tot = {}
        self.w = {}
        self.p = {}
                

    # Unbatched
    # def q_val(self, state):
    #     statehash = self.mdp.state_to_hashable(state)
    #     if statehash not in self.q:
    #         return torch.zeros((1,) + self.mdp.action_shape)
    #     return self.q[statehash]

    # Unbatched, return action shape
    def ucb(self, state, param: float):
        statehash = self.mdp.state_to_hashable(state)
        if statehash not in self.q:
            return torch.zeros((1,) + self.mdp.action_shape)
        # Exploration vs. exploitation: the second term dominates when unexplored, then the first term dominates when more explored
        return (self.q[statehash].nan_to_num(0) + self.p[statehash] * (param * math.sqrt(self.n_tot[statehash]) / (1 + self.n[statehash])))[None]

    # Unbatched
    # Starting at a given state, conducts a fixed number of Monte-Carlo searches, using the Q function in visited states and the heuristic function in new states
    # Updates the Q, N, W, P functions
    # Returns probability vector with batched dimension added
    def search(self, state, heuristic: nn.Module, num: int, ucb_parameter = 2.0, temperature=1.0, evaluation_batch_size = 8, p_threshold = 100, max_depth=1000):
        evaluation_queue = []

        # Do a certain number of searches from the initial state
        for i in range(num):
            history = []
            s = state
            depth = 0

            # Go down tree, using upper confidence bound and dictionary q values to play
            while self.mdp.state_to_hashable(s) in self.q and self.mdp.is_terminal(s).item() == False:
                if depth >= max_depth:
                    print("Max depth researched in selection.")
                    break

                # Q has values [-1, 1] generally, so do a shift
                action = self.mdp.get_max_action((1 + self.ucb(s, ucb_parameter)) * self.mdp.valid_action_filter(s))
                history.append((s, action))
                s, r = self.mdp.transition(s, action)
                depth += 1
                
            # Once we reach a leaf, use the heuristic function p to simulate play
            # TODO this cna be parallelized, i.e. the evaluaiton
            if self.mdp.is_terminal(s).item() == False:
                self.q[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape)
                self.n[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape)
                self.n_tot[self.mdp.state_to_hashable(s)] = 0
                self.w[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape)

                # Placeholder prior probability given by neural network, do in batches TODO implement
                # evaluation_queue.append(s)
                # if len(evaluation_queue) >= evaluation_batch_size:
                #     heuristic_prob = heuristic(torch.cat(evaluation_queue, dim=0))
                #     for i in range(heuristic_prob.size(0)):
                #         self.p[self.mdp.state_to_hashable(evaluation_queue[i])] = heuristic_prob[i:i+1]

                # Sigmoid so that the values are in (0, 1)
                self.p[self.mdp.state_to_hashable(s)] = torch.sigmoid(heuristic(s))

            while self.mdp.is_terminal(s).item() == False:
                if depth >= max_depth:
                    print("Max depth researched in expansion.")
                    break
                # TODO do we want to cache these values
                action = self.mdp.get_random_action_weighted(torch.sigmoid(heuristic(s)) * self.mdp.valid_action_filter(s)) 
                # We don't keep the simulation nodes for memory reasons
                #history.append((s, action))
                s, r = self.mdp.transition(s, action)
                depth += 1
            
            while len(history) > 0:
                s, action = history.pop()
                if self.mdp.state_to_hashable(s) in self.p:
                    self.n[self.mdp.state_to_hashable(s)] += action[0]
                    self.n_tot[self.mdp.state_to_hashable(s)] += action[0].sum().item()
                    self.w[self.mdp.state_to_hashable(s)] += action[0] * r[0,self.mdp.get_player(s)[0]][0]
                    self.q[self.mdp.state_to_hashable(s)] = self.w[self.mdp.state_to_hashable(s)] / self.n[self.mdp.state_to_hashable(s)]
                    # Update the prior probability if threshold passed
                    if self.n_tot[self.mdp.state_to_hashable(s)] > p_threshold:
                        p_vector = self.n[self.mdp.state_to_hashable(s)] ** (1 / temperature)
                        self.p[self.mdp.state_to_hashable(s)] = p_vector / p_vector.flatten().sum()

        # Return probability vector for initial state (even if it isn't updated)
        p_vector = self.n[self.mdp.state_to_hashable(state)] ** (1 / temperature)
        return (p_vector / p_vector.flatten().sum())[None]
                



    def mcts(self, lr: float, num_iterations: int, num_selfplay: int, num_searches: int, max_steps: int, ucb_parameter: float, temperature: float, train_batch: int, p_threshold: int, save_path=None, verbose=False, graph_smoothing=10, initial_log=""):        
        logtext = initial_log
        prior_p = copy.deepcopy(self.pv)
        training_inputs = []
        training_values = []

        # In each iteration, we simulate a certain number of self-plays, then we train on the resulting data
        for i in range(num_iterations):
            logtext += log(f"Iteration {i+1}")
            opt = self.optimizer(self.pv.parameters(), lr=lr)
                


            iteration_data = torch.tensor([])
            s = self.mdp.get_initial_state(num_selfplay)        # TODO not suppose to parallelize the self plays?

            # The probability vectors for each self-play
            states = [torch.tensor([]) for j in range(num_selfplay)]
            p_vectors = [torch.tensor([]) for j in range(num_selfplay)]
            results = [None for j in range(num_selfplay)]

            # Each step in the self-plays
            for step in range(max_steps):
                step_p = torch.tensor([])

                # Each self-play
                for play in range(num_selfplay):
                    # If a result has been obtained, then append a trivial action and skip everything
                    if results[play] != None:
                        step_p = torch.cat([step_p, torch.zeros((1,) + self.mdp.action_shape)], dim=0)
                        continue

                    # Do searches
                    p_vector = self.search(s[play:play+1], heuristic=prior_p, ucb_parameter=ucb_parameter, num=num_searches, temperature=temperature, p_threshold=p_threshold)

                    # Add probability vector to record
                    states[play] = torch.cat([states[play], s[play:play+1]], dim=0)
                    p_vectors[play] = torch.cat([p_vectors[play], p_vector], dim=0)
                    step_p = torch.cat([step_p, p_vector], dim=0)

                # Select actions and do all the transitions
                a = self.mdp.get_random_action_weighted(step_p)
                s, r = self.mdp.transition(s, a)

                # If state is terminal, then record the result (i.e. the winner)
                for play in range(num_selfplay):
                    if results[play] == None and self.mdp.is_terminal(s[play:play+1]).item():
                        results[play] = r[play]

            # We do not actually need to bother with the results, since we do not have the bot concede            
            # For unresolved games, set rewards to zero
            # for play in range(num_selfplay):
            #     if results[play] == None:
            #         results[play] = torch.zeros(self.mdp.num_players)

            training_inputs.append(torch.cat(states, dim=0))
            training_values.append(torch.cat(p_vectors, dim=0))

            # Do training on recent data
            indices = list(range(training_values[-1].size(0)))
            total_loss = 0.
            num_train = 0.
            self.pv.train()
            for i in range(training_values[-1].size(0)):
                get_indices = random.sample(indices, min(len(indices), train_batch))
                x = training_inputs[-1][get_indices]
                y = training_values[-1][get_indices]

                pred = self.pv(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                opt.step()
                opt.zero_grad()

                total_loss += loss.item()
                num_train += 1

            self.pv.eval()
            logtext += log(f"Loss: {total_loss/num_train} on {num_train} batches of size {min(len(indices), train_batch)}.")
        


                

            