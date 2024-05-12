import zipfile, os, random, warnings, datetime, copy, math, time, pickle
import torch
from torch import nn
from collections import namedtuple, deque
from qlearn import MDP, QFunction, PrototypeQFunction
from aux import log, smoothing
import matplotlib.pyplot as plt
from deepqlearn import TensorMDP, NNQFunction, DeepRL


class MCTSQFunction(PrototypeQFunction):
    def __init__(self, mdp: TensorMDP, heuristic_model: nn.Module, model_args={}, device="cpu"):
        self.device = device
        self.mdp = mdp
        self.h = heuristic_model(**model_args, **self.mdp.nn_args).to(self.device)
        self.h.eval()

        # Keys: hashable states.  Values: tensor with shape of actions
        self.q = {}
        self.n = {}         
        self.n_tot = {}
        self.w = {}
        self.p = {}

    # Gets a probability vector or the maximum
    # 
    def get(self, state: torch.Tensor, heuristic_parameter: torch.Tensor | float):
        q, p, n_tot, n = torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        for i in range(state.size(0)):
            statehash = self.mdp.state_to_hashable(state)
            if statehash in self.q:
                q = torch.cat([q, self.q[statehash][None]])
                p = torch.cat([p, self.p[statehash][None]])
                n = torch.cat([n, self.n[statehash][None]])
                n_tot = torch.cat([n_tot, torch.tensor([self.n_tot[statehash]], device=self.device)])
            else:
                q = torch.cat([q, torch.zeros((1,) + self.mdp.action_shape, device=self.device)])
                p = torch.cat([p, torch.zeros((1,) + self.mdp.action_shape, device=self.device)])
                n = torch.cat([n, torch.zeros((1,) + self.mdp.action_shape, device=self.device)])
                n_tot = torch.cat([n_tot, torch.zeros(1)])
        n_tot = n_tot.reshape((-1,) + self.mdp.action_projshape)

        # Exploration vs. exploitation: the second term dominates when unexplored, then the first term dominates when more explored
        return q.nan_to_num(0) + p * (heuristic_parameter * math.sqrt(n_tot) / (1 + n))
    

    def policy(self, state: torch.Tensor, heuristic_parameter = 0., stochastic=True, valid_filter=True):
        if stochastic:
            return self.mdp.get_random_action_weighted(self.get(state, heuristic_parameter))
        else:
            return self.mdp.get_max_action(self.ucb(state, heuristic_parameter))
        
    def save(self, fname):
        with open(fname + ".q", 'wb') as f:
            pickle.dump({'q': self.q, 'n': self.n, 'n_tot': self.n_tot, 'w': self.w, 'p': self.p}, f)
        model_scripted = torch.jit.script(self.h)
        model_scripted.save(fname + ".h")

    def load(self, fname, indices=None):
        with open(fname + ".q", 'rb') as f:
            load_dict = pickle.load(f)
            self.q = load_dict['q']
            self.n = load_dict['n']
            self.n_tot = load_dict['n_tot']
            self.w = load_dict['w']
            self.p = load_dict['p']
        self.h = torch.jit.load(fname + ".h")
    
    def null(self, indices = None):
        self.q, self.n, self.n_tot, self.w, self.p = {}, {}, {}, {}, {}
        #self.h.weight.data.fill_(0.)
        # TODO Heuristic function reset not implemented

        
    



class DMCTS(DeepRL):

    def __init__(self, mdp: TensorMDP, model: nn.Module, loss_fn, optimizer, model_args = {}, device="cpu"):
        super().__init__(mdp, MCTSQFunction(mdp, model, model_args, device), device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    # Input batched size 1
    # Starting at a given state, conducts a fixed number of Monte-Carlo searches, using the Q function in visited states and the heuristic function in new states
    # Updates the Q, N, W, P functions for explored states, returns probability vector for initial state
    def search(self, state, heuristic: nn.Module, num: int, ucb_parameter, temperature=1.0, max_depth=1000):
        evaluation_queue = []

        # Do a certain number of searches from the initial state
        for i in range(num):
            history = []
            s = state
            depth = 0

            # Go down tree, using upper confidence bound and dictionary q values to play
            while self.mdp.state_to_hashable(s) in self.q.q and self.mdp.is_terminal(s).item() == False:
                if depth >= max_depth:
                    print("Max depth researched in selection.")
                    break

                # Q has values [-1, 1] generally, so do a shift
                valid_actions = self.mdp.valid_action_filter(s)
                num_valid = valid_actions.sum()
                # Scale the parameter by the number of valid actions
                action = self.mdp.get_max_action((1 + self.q.get(s, ucb_parameter * num_valid)) * valid_actions)
                history.append((s, action))
                s, r = self.mdp.transition(s, action)
                depth += 1
                
            # Once we reach a leaf, use the heuristic function p to simulate play
            # TODO this cna be parallelized, i.e. the evaluaiton
            if self.mdp.is_terminal(s).item() == False:
                self.q.q[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape, device=self.device)
                self.q.n[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape, device=self.device)
                self.q.n_tot[self.mdp.state_to_hashable(s)] = 0
                self.q.w[self.mdp.state_to_hashable(s)] = torch.zeros(self.mdp.action_shape, device=self.device)

                # Placeholder prior probability given by neural network, do in batches TODO implement
                # evaluation_queue.append(s)
                # if len(evaluation_queue) >= evaluation_batch_size:
                #     heuristic_prob = heuristic(torch.cat(evaluation_queue, dim=0))
                #     for i in range(heuristic_prob.size(0)):
                #         self.p[self.mdp.state_to_hashable(evaluation_queue[i])] = heuristic_prob[i:i+1]

                # Sigmoid so that the values are in (0, 1)
                # These prior probabilities are only assigned once; eventually, with enough visits the quality function will dominate the heuristic
                # TODO zero out invalid?
                self.q.p[self.mdp.state_to_hashable(s)] = torch.softmax(heuristic(s).flatten(1, -1), dim=1).reshape((-1, ) + self.mdp.action_shape)

            while self.mdp.is_terminal(s).item() == False:
                if depth >= max_depth:
                    print("Max depth researched in expansion.")
                    break
                # Exponential means un-normalized softmax
                action = self.mdp.get_random_action_weighted(torch.exp(heuristic(s)) * self.mdp.valid_action_filter(s)) 
                # We don't keep the simulation nodes in history for memory reasons
                s, r = self.mdp.transition(s, action)
                depth += 1
            
            while len(history) > 0:
                s, action = history.pop()
                if self.mdp.state_to_hashable(s) in self.q.p:
                    self.q.n[self.mdp.state_to_hashable(s)] += action[0]
                    self.q.n_tot[self.mdp.state_to_hashable(s)] += action[0].sum().item()
                    self.q.w[self.mdp.state_to_hashable(s)] += action[0] * r[0,self.mdp.get_player(s)[0]][0]
                    self.q.q[self.mdp.state_to_hashable(s)] = self.q.w[self.mdp.state_to_hashable(s)] / self.q.n[self.mdp.state_to_hashable(s)]

        # Return probability vector for initial state (even if it isn't updated)
        p_vector = self.q.n[self.mdp.state_to_hashable(state)] ** (1 / temperature)
        return (p_vector / p_vector.flatten().sum())[None]
                



    def mcts(self, lr: float, num_iterations: int, num_selfplay: int, num_searches: int, max_steps: int, ucb_parameter: float, temperature: float, train_batch: int, train_iterations = 1, save_path=None, verbose=True, initial_log=""):        
        logtext = initial_log
        
        start_time = time.perf_counter()
        logtext += log(f"Started logging.")

        # TODO The deep copy is only needed if we parallelize the training
        # prior_p = copy.deepcopy(self.pv)
        prior_p = self.q.h
        
        training_inputs = []
        training_values = []

        start_time = datetime.datetime.now()

        # In each iteration, we simulate a certain number of self-plays, then we train on the resulting data
        for i in range(num_iterations):
            logtext += log(f"Iteration {i+1}", verbose)
            opt = self.optimizer(self.q.h.parameters(), lr=lr)
                


            iteration_data = torch.tensor([], device=self.device)
            s = self.mdp.get_initial_state(num_selfplay)        # TODO not suppose to parallelize the self plays?

            # The probability vectors for each self-play
            states = torch.tensor([], device=self.device)
            p_vectors = torch.tensor([], device=self.device)

            logtext += log(f"Self-plays at {time.perf_counter() - start_time}", verbose)
            # Each step in the self-plays
            for step in range(max_steps):
                step_p = torch.tensor([], device=self.device)

                # Each self-play TODO parallelize search
                for play in range(s.size(0)):
                    # Do searches
                    p_vector = self.search(s[play:play+1], heuristic=prior_p, ucb_parameter=ucb_parameter, num=num_searches, temperature=temperature)

                    # Add probability vector to record TODO these don't need to be sorted by play
                    states = torch.cat([states, s[play:play+1]], dim=0)
                    p_vectors = torch.cat([p_vectors, p_vector], dim=0)
                    step_p = torch.cat([step_p, p_vector], dim=0)

                # Select actions and do all the transitions
                a = self.mdp.get_random_action_weighted(step_p)
                s, _ = self.mdp.transition(s, a)

                # If state is terminal, then discard it
                s = s[(~self.mdp.is_terminal(s)).flatten()]
                # If there are no more states, break
                if s.size(0) == 0:
                    break
                        
            # TODO think about having the bot concede? might save some simulations

            logtext += log(f"Training at {time.perf_counter() - start_time}", verbose)

            training_inputs.append(states)
            training_values.append(p_vectors)

            # Do training on recent data
            inputs = torch.cat(training_inputs[-train_iterations:])
            values = torch.cat(training_values[-train_iterations:])
            indices = list(range(inputs.size(0)))
            total_loss = 0.
            num_train = 0.
            self.q.h.train()
            for i in range(inputs.size(0)):
                get_indices = random.sample(indices, min(len(indices), train_batch))
                x = inputs[get_indices]
                y = values[get_indices]

                # Before softmax
                pred = self.q.h(x)

                loss = self.loss_fn(pred, y)
                loss.backward()
                opt.step()
                opt.zero_grad()

                total_loss += loss.item()
                num_train += 1

            self.q.h.eval()
            logtext += log(f"Loss: {total_loss/num_train} on {num_train} batches of size {min(len(indices), train_batch)}.", verbose)

        if save_path != None:
            self.q.save(save_path)
            logtext += log(f"Saved model to {save_path}.h, {save_path}.q", verbose)

            logpath = save_path + ".log"
            with open(logpath, "w") as f:
                logtext += log(f"Saved logs to {logpath}", verbose)
                f.write(logtext)

            # Saving:
            # 1. dictionaries 2. heuristic 3. data??
        


                

            