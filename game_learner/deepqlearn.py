import random, warnings
import zipfile, os
import torch
from torch import nn
from collections import namedtuple, deque
from qlearn import MDP, QFunction





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
            model_scripted.save(f"{fname}.{i}")
            zf.write(f"{fname}.{i}", f"{fname}.{i}", compress_type=zipfile.ZIP_STORED)
            os.remove(f"{fname}.{i}")
        zf.close()

            

    def load_q(self, fname):
        zf = zipfile.ZipFile(fname, mode="r")
        for i in range(self.mdp.num_players):
            zf.extract(f"{fname}.{i}")
            self.qs[i].q = torch.jit.load(f"{fname}.{i}")
            os.remove(f"{fname}.{i}")
            self.copy_policy_to_target()
            self.qs[i].q.eval()					# Set to evaluation mode
        zf.close()


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
            print(f"Initializing episode {j+1}.  Batch size {batch_size}.")
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
# TODO verbose
