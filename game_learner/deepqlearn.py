import zipfile, os, random, warnings
import torch
from torch import nn
from collections import namedtuple, deque
from qlearn import MDP, QFunction
from aux import log
import matplotlib.pyplot as plt

# The player order matches their names.
# I sometimes indicate terminal states by negating all values in the state.
#Sometimes this means I don't have to implement a method to check terminal conditions globally, which can be costly, and can just do it locally in transitions.


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
    def push(self, d: TransitionData):
        self.memory.append(d)

    def push_batch(self, data: TransitionData, transforms = {'s': lambda s: s, 'a': lambda a: a, 'r': lambda r: r}):
        for i in range(data.s.size(0)):
                self.push(TransitionData(transforms['s'](data.s[i]), transforms['a'](data.a[i]), transforms['s'](data.t[i]), transforms['r'](data.r[i])))

            

    def sample(self, num: int, batch=True) -> list:
        if batch:
            data = random.sample(self.memory, num)
            s = torch.stack([datum[0] for datum in data])
            return TransitionData(torch.stack([datum.s for datum in data]), torch.stack([datum.a for datum in data]), torch.stack([datum.t for datum in data]), torch.stack([datum.r for datum in data]))
        else:
            return random.sample(self.memory, num)       
    
    def __len__(self):
        return len(self.memory)
    


class TensorMDP(MDP):
    def __init__(self, state_shape, action_shape, discount=1, num_players=1, penalty = -2, default_hyperparameters={}, symb = {}, input_str = "", batched=False):
        
        super().__init__(None, None, discount, num_players, penalty, symb, input_str, default_hyperparameters, batched)

        # Whether we filter out illegal moves in training or not
        #self.filter_illegal = filter_illegal

        self.state_shape = state_shape
        self.action_shape = action_shape

        # Useful for getting things in the right shape
        self.state_projshape = (1, ) * len(self.state_shape)
        self.action_projshape = (1, ) * len(self.action_shape)

        self.state_linshape = 0
        for i in range(len(self.state_shape)):
            self.state_linshape += self.state_shape[i]
        self.action_linshape = 0
        for i in range(len(self.action_shape)):
            self.action_linshape += self.action_shape[i]

    def valid_action_filter(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

        

# A Q-function where the inputs and outputs are all tensors
class NNQFunction(QFunction):
    def __init__(self, mdp: TensorMDP, q_model, loss_fn, optimizer_class: torch.optim.Optimizer, device="cpu"):
        if mdp.state_shape == None or mdp.action_shape == None:
            raise Exception("The input MDP must handle tensors.")
        self.q, self.target_q = q_model().to(device), q_model().to(device)
        self.q.eval()
        self.target_q.eval()

        self.mdp = mdp
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class
        self.device=device

    # Make the Q function perform randomly.
    def lobotomize(self):
        self.q = None

    # Copy the weights of one NN to another
    def copy_policy_to_target(self):
        self.target_q.load_state_dict(self.q.state_dict())

    # If input action = None, then return the entire vector of action values of shape (batch, ) + action_shape
    # Otherwise, output shape (batch, )
    def get(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.q.eval()
        if self.q == None:
            pred = self.mdp.get_random_action(state)    
        else:
            pred = self.q(state)
        if action == None:
            return pred
        else:
            # A little inefficient because we only take the diagonals, 
            return torch.tensordot(pred.flatten(start_dim=1), action.flatten(start_dim=1), dims=([1],[1])).diagonal()

    # Input is a tensor of shape (batches, ) + state.shape
    # Output is a tensor of shape (batches, )
    def val(self, state, target=False) -> torch.Tensor:
        q = self.target_q if target else self.q
        if q == None:
            return torch.zeros(state.size(0))
        return q(state).flatten(1, -1).max(1).values * ~torch.flatten(self.mdp.is_terminal(state))
    
    # Input is a batch of state vectors
    # Returns the value of an optimal policy at a given state, shape (batches, ) + action_shape
    def policy(self, state, max_tries=100) -> torch.Tensor:
        if self.q == None:
            return self.mdp.get_random_action(state)
        
        filter = self.q(state).flatten(1, -1)
        filter = (filter == filter.max(1).values[:,None])
        while (filter.count_nonzero(dim=1) <= 1).prod().item() != 1:                            # Almost always terminates after one step
            filter = torch.rand(filter.shape, device=self.device) * filter
            filter = (filter == filter.max(1).values[:,None])
            max_tries -= 1
            if max_tries == 0:
                break
        return filter.unflatten(1, self.mdp.action_shape)

    # Does a Q-update based on some observed set of data
    # TransitionData entries are tensors
    def update(self, data: TransitionData, learn_rate):
        if not isinstance(self.q, nn.Module):
            Exception("NNQFunction needs to have a class extending nn.Module.")

        if self.q == None:
            return

        self.q.train()
        self.target_q.eval()
        opt = self.optimizer(self.q.parameters(), lr=learn_rate)
        pred = self.get(data.s, data.a)
        y = data.r + self.val(data.t, target=True)
        loss = self.loss_fn(pred, y)

        # Optimize
        loss.backward()
        opt.step()
        opt.zero_grad()

        self.q.eval()

        return loss.item()






# Input shape (batch_size, ) + state_tensor
# We implement a random tensor of 0's and 1's generating a random tensor of floats in [0, 1), then converting it to a bool, then back to a float.
def greedy_tensor(q: NNQFunction, state, eps = 0.):
    return q.mdp.get_random_action(state) if random.random() < eps else q.policy(state)


# For now, for simplicity, fix a single strategy
# Note that qs is policy_qs
class DQN():
    def __init__(self, mdp: TensorMDP, q_model: nn.Module, loss_fn, optimizer, memory_capacity: int, device="cpu"):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.qs = [NNQFunction(mdp, q_model, loss_fn, optimizer, device=device) for i in range(mdp.num_players)]
        self.memories = [ExperienceReplay(memory_capacity) for i in range(mdp.num_players)]
        self.memory_capacity = memory_capacity
        self.device = device

    def save_q(self, fname):
        zf = zipfile.ZipFile(fname, mode="w")
        for i in range(self.mdp.num_players):
            model_scripted = torch.jit.script(self.qs[i].q)
            model_scripted.save(f"temp/player.{i}")
            zf.write(f"temp/player.{i}", f"player.{i}", compress_type=zipfile.ZIP_STORED)
            os.remove(f"temp/player.{i}")
        zf.close()

            

    def load_q(self, fname, indices=None):
        zf = zipfile.ZipFile(fname, mode="r")
        for i in range(self.mdp.num_players):
            if indices == None or i in indices:
                zf.extract(f"player.{i}", "temp/")
                self.qs[i].q = torch.jit.load(f"temp/player.{i}")
                self.qs[i].copy_policy_to_target()
                os.remove(f"temp/player.{i}")
        zf.close()


    def null_q(self, indices = None):
        for i in range(self.mdp.num_players):
            if indices == None or i in indices:
                self.qs[i].lobotomize()
                
    def stepthru_game(self):
        s = self.mdp.get_initial_state()
        print(f"Initial state:\n{self.mdp.board_str(s)[0]}")
        turn = 0
        while self.mdp.is_terminal(s) == False:
            turn += 1
            p = self.mdp.get_player(s)[0].item()
            print(f"Turn {turn}, player {p+1} ({self.mdp.symb[p]})")
            a = self.qs[p].policy(s)
            print(f"Chosen action: {self.mdp.action_str(a)[0]}")
            s, r = self.mdp.transition(s, a)
            print(f"Next state:\n{self.mdp.board_str(s)[0]}")
            print(f"Rewards for players: {r[0].tolist()}")
            input("Enter to continue.\n")
        input("Terminal state reached.  Enter to end. ")


    # Keeps the first action; the assumption is the later actions are "passive" (i.e. not performed by the given player)
    # Adds the rewards
    # Returns a tuple: (composition, to_memory)
    def compose_transition_tensor(self, first: TransitionData, second: TransitionData, player_index: int):
        if torch.prod((first.t == second.s) == 0).item():
            # Make this error non-fatal but make a note
            warnings.warn("The source and target states do not match.")

        # Two cases: say (s,a,r,t) compose (s',a',r',t')
        # If player(t) = player_index or t is terminal, then (s',a',r',t'), i.e. replace because loop completed last turn, has been commited to memory
        # Else, (s,a,r+r',t'), i.e. it has not, so compose
        # Filter shape (batch, 1, ...).  For annoying tensor dimension reasons, need different filters that are basically the same
        filter_s = (self.mdp.get_player(first.t) == player_index) | self.mdp.is_terminal(first.t)
        filter_a = filter_s.view((filter_s.size(0), ) + self.mdp.action_projshape)
        filter_r = filter_s.flatten()
        new_s = filter_s * second.s + (filter_s == False) * first.s
        new_a = filter_a * second.a + (filter_a == False) * first.a
        new_t = second.t
        new_r = second.r + (filter_r == False) * first.r
    
        # After the above update, commit to the memory bank rows where: 
        # new_s is not terminal AND (
        # player(new_s) = player(new_t) = player_index, i.e. the cycle completed and came back to the recording player, OR 
        # where new_t is terminal )
        filter = (~self.mdp.is_terminal(new_s) & (((self.mdp.get_player(new_s) == player_index) & (self.mdp.get_player(new_t) == player_index)) | (self.mdp.is_terminal(new_t)))).flatten()
        return (TransitionData(new_s, new_a, new_t, new_r), TransitionData(new_s[filter], new_a[filter], new_t[filter], new_r[filter]))



    # Handling multiplayer: each player keeps their own "record", separate from memory
    # When any entry in the record has source = target, then the player "banks" it in their memory
    # The next time an action is taken, if the source = target, then it gets overwritten
    def deep_learn(self, lr: float, dq_episodes: int, episode_length: int, anneal_eps: int, expl_start: float, expl_end: float, sim_batch: int, train_batch: int, copy_interval_eps: int, savelog=None, verbose=False):

        dq_episodes, episode_length, anneal_eps, sim_batch, train_batch, copy_interval_eps = int(dq_episodes), int(episode_length), int(anneal_eps), int(sim_batch), int(train_batch), int(copy_interval_eps)
        if train_batch < 2:
            Exception("Training batch size must be greater than 1 for sampling.")

        logtext = ""
        if verbose or savelog != None:
            logtext += log(f"Device: {self.device}", verbose)
            logtext += log(f"MDP:\n{self.mdp}", verbose)
            logtext += log(f"Model:\n{self.qs[0].q}", verbose)
            logtext += log(f"Loss function:\n{self.qs[0].loss_fn}", verbose)
            logtext += log(f"Optimizer:\n{self.qs[0].optimizer}", verbose)
            logtext += log(f"Learn rate: {lr}, start and end exploration: {expl_start}->{expl_end}, episodes: {dq_episodes}, episode length: {episode_length}, batch size: {sim_batch}, training batch size: {train_batch}, initial greed annealing time: {anneal_eps}, copy frequency: {copy_interval_eps}, memory capacity: {self.memory_capacity}.", verbose)
            wins = [0] * self.mdp.num_players
            penalties = [0] * self.mdp.num_players
            episode_losses = [[] for i in range(self.mdp.num_players)]

        frame = 0
        for j in range(dq_episodes):
            # Set greed
            greed_cur = ((anneal_eps - j) * expl_start + j * expl_end)/anneal_eps if j <= anneal_eps else expl_end
            if verbose or savelog != None:
                losses = [0.] * self.mdp.num_players
                updates = 0

            # Make sure the target network is the same as the policy network
            for i in range(self.mdp.num_players):
                self.qs[i].copy_policy_to_target()
            
            
            if verbose or savelog != None:
                memorylen = [self.memories[i].size() for i in range(self.mdp.num_players)]
                logtext += log(f"Initializing episode {j+1}. Greed {1-greed_cur:>0.5f}. Batch size {sim_batch}. Memory {memorylen}. Player wins {wins}, penalties {penalties}.", verbose)
            
            s = self.mdp.get_initial_state(sim_batch)
            # "Records" for each player
            player_record = [None for i in range(self.mdp.num_players)]
            
            for k in range(episode_length):
                frame += 1

                # Execute the transition on the "actual" state
                # To do this, we need to iterate over players, because each has a different q function for determining the strategy
                p = self.mdp.get_player(s)                                  # p.shape = (batch, )       s.shape = (batch, 2, 6, 7)

                # To get the actions, apply each player's strategy
                a = torch.zeros((sim_batch, ) + self.mdp.action_shape, device=self.device)
                for pi in range(self.mdp.num_players):
                    # Get the indices corresponding to this player's turn
                    indices = torch.arange(sim_batch, device=self.device)[p.flatten() == pi].tolist()
                    
                    # Apply thie player's strategy to their turns
                    # Greedy is not parallelized, so there isn't efficiency loss with this
                    player_actions = torch.cat([greedy_tensor(self.qs[pi], s[l:l+1], greed_cur) if l in indices else torch.zeros((1, ) + self.mdp.action_shape, device=self.device) for l in range(sim_batch)], 0)
                    a += player_actions

                # Do the transition 
                t, r = self.mdp.transition(s, a)
                
                # Tracking wins and penalities
                if verbose or savelog != None:
                    for pi in range(self.mdp.num_players):
                        wins[pi] += torch.sum(r[:,pi] > 0).item()
                        penalties[pi] += torch.sum(r[:,pi] <= self.mdp.penalty).item()

                # Update player records and memory
                for pi in range(self.mdp.num_players):
                    # If it's the first move, just put the record in.  Note we only care about the recorder's reward.
                    if player_record[pi] == None:
                        player_record[pi] = TransitionData(s, a, t, r[:,pi])
                    else:
                        player_record[pi], to_memory = self.compose_transition_tensor(player_record[pi], TransitionData(s, a, t, r[:,pi]), pi)
                        self.memories[pi].push_batch(to_memory)

                        
                # Train the policy on a random sample in memory (once the memory bank is big enough)
                for i in range(self.mdp.num_players):
                    if len(self.memories[i]) >= train_batch:
                        losses[i] += self.qs[i].update(self.memories[i].sample(train_batch), lr)
                        updates += train_batch

                # Restart terminal states TODO for now, just check if the whole thing is done and end the episode to hasten things up
                # if torch.prod(self.mdp.is_terminal(s)).item() == 1:
                #     break

                # Don't forget to set the state to the next state
                s = t
            
            # Copy the target network to the policy network if it is time
            if (j+1) % copy_interval_eps == 0:
                for i in range(self.mdp.num_players):
                    self.qs[i].copy_policy_to_target()
                logtext += log("Copied policy to target networks for all players.", verbose)
            
            
            if (verbose or savelog != None) and updates > 0:
                for i in range(len(losses)):
                    losses[i] = losses[i]/updates
                    episode_losses[i].append(losses[i])
                logtext += log(f"Episode {j+1} average loss: {losses}", verbose)
                

        if savelog != None:
            # First, do an averaging of the losses
            convolved_losses = [[] for i in range(self.mdp.num_players)]
            for i in range(self.mdp.num_players):
                for j in range(len(episode_losses[i])):
                    total = 0.
                    num = 0
                    for k in range(-5, 6):
                        if j + k >= 0 and j + k < len(episode_losses[i]):
                            total += episode_losses[i][j+k]
                            num += 1
                    convolved_losses[i].append(total/num)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 1, 1)
            for i in range(self.mdp.num_players):
                plt.plot(range(dq_episodes - len(convolved_losses[i]), dq_episodes), convolved_losses[i], label=f'Player {i} smoothed losses')
            plt.legend(loc='lower right')
            plt.title('Losses with convolution kernel size 11')

            plotpath = savelog + ".png"
            plt.savefig(plotpath)
            logtext += log(f"Saved accuracy/loss plot to {plotpath}", verbose)

            with open(savelog, "w") as f:
                logtext += log(f"Saved logs to {savelog}", verbose)
                f.write(logtext)