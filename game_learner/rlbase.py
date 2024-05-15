import random
import torch
import matplotlib.pyplot as plt



def log(l: str, print_out=True) -> str:
    if print_out:
        print(l)
    return str(l) + "\n"

# span n means length 2n+1
def smoothing(vals: list, span: int):
    out = []
    for j in range(len(vals)):
        total = 0.
        num = 0
        for k in range(-span, span+1):
            if j + k >= 0 and j + k < len(vals):
                total += vals[j+k]
                num += 1
        out.append(total/num)
    return out

class PrototypeQFunction():
    def __init__():
        raise NotImplementedError
    
    def get(self, state, action) -> float:
        raise NotImplementedError
    
    def save(self, fname):
        raise NotImplementedError     

    def load(self, fname, indices=None):
        raise NotImplementedError
    
    def null(self, indices = None):
        raise NotImplementedError
    




# WARNING: states and actions must be hashable if using with QFunction.
class MDP():
    def __init__(self, states, actions, discount=1, num_players=1, penalty = -1000, symb = {}, input_str = "", default_hyperparameters = {}, batched=False):
        self.actions = actions
        self.discount = discount
        self.num_players = num_players
        self.batched = batched
        self.penalty=penalty
        self.symb = symb
        self.input_str = input_str
        self.default_hyperparameters = default_hyperparameters

        # The states do not literally have to be tensors for the state_shape to be defined.  This just specifies, when they are turned into tensors, what shape they should be have, ignoring batch dimension
        self.states = states

        self.actions = actions
        
    def str_to_action(self, input: str):
        raise NotImplementedError
    
    def board_str(self, state) -> str:
        raise NotImplementedError
    




    def is_state(self, state):
        if self.states != None:
            return True if state in self.states else False
        elif self.state_shape != None:
            return True if state[0].shape == self.state_shape else False
        else:
            return True
        
    def is_action(self, action):
        if self.actions != None:
            return True if action in self.actions else False
        elif self.action_shape != None:
            return True if action[0].shape == self.action_shape else False
        else:
            return True

    def action_str(self, action) -> str:
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

        
    def is_valid_action(self, state, action):
        return action in self.get_actions(state)

    def get_random_action(self, state = None):
        return random.choice(self.get_actions(state))

    # Returns: (next state, reward)
    def transition(self, s, a):
        raise NotImplementedError

    # Returns a valid start state and starting player for the game.
    def get_initial_state(self):
        raise NotImplementedError

    # Tells us if a state is terminal.
    def is_terminal(self, s):
        raise NotImplementedError
    
    # Gets the current player of the given state.
    def get_player(self, state):
        raise NotImplementedError




class TensorMDP(MDP):
    def __init__(self, state_shape, action_shape, default_memory: int, discount=1, num_players=1, penalty = -2, num_simulations=1000, default_hyperparameters={}, symb = {}, nn_args={}, input_str = "", batched=False, device="cpu"):
        
        super().__init__(None, None, discount, num_players, penalty, symb, input_str, default_hyperparameters, batched)
        self.nn_args = nn_args
        self.default_memory = default_memory
        self.device=device


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

        self.num_simulations = num_simulations

    ##### ACTIONS #####

    def valid_action_filter(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    # Inserts negative infinity at invalid actions (logits version of multilication by masking filter)
    # The reason this is needed is that the operation of zeroing out entries in filters corresponds, when taking maximums, to replacing those entries with negative infinity
    # The return value of this function should be added to an action tensor
    def neginf_kill_actions(self, state: torch.Tensor) -> torch.Tensor:
        return (-torch.inf * (1 - self.valid_action_filter(state).float())).nan_to_num(0)
    
    # Helper function
    def masked_softmax(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return (input + self.neginf_kill_actions(state)).flatten(1, -1).softmax(dim=1).reshape((-1,) + self.action_shape)
    
    # Chooses a valid action unifoirmly at random
    def get_random_action(self, state) -> torch.Tensor:
        return self.get_random_action_weighted(self.valid_action_filter(state).float())
    
    # Chooses a action from the indices with maximum value (uniformly at random if more than one)
    def get_max_action(self, values) -> torch.Tensor:         # TODO what if all values are 0
        return self.get_random_action_weighted(values == values.flatten(1,-1).max(1).values.reshape((-1,) + self.action_projshape).float())
        
    # Input action shape, return an action with probability weighted by the entries
    def get_random_action_weighted(self, weights) -> torch.Tensor:
        # Zero out negative entries and flatten
        weights = ((weights > 0) * weights).flatten(1, -1)
        return (torch.cumsum(weights / torch.sum(weights, 1)[:, None], 1) > torch.rand(weights.size(0), device=self.device)[:,None]).diff(dim=1, prepend=torch.zeros((weights.size(0),1), device=self.device)).reshape((-1,) + self.action_shape)
    
    # Output has state shape
    def is_valid_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return (self.valid_action_filter(state) * action).flatten(1,-1).sum(1, keepdim=True).reshape((-1,) + self.state_projshape) == 1.

    def tests(self, qs: list[PrototypeQFunction]):
        return []
        
    # Not batched
    def state_to_hashable(self, state: torch.Tensor):
        return tuple(state.flatten().tolist())
    
    def hashable_to_state(self, state):
        return torch.tensor(state).reshape(self.state_shape)
    
    def action_to_hashable(self, action: torch.Tensor):
        return tuple(action.flatten().tolist())
    
    def hashable_to_action(self, action):
        return torch.tensor(action).reshape(self.action_shape)
    
    # Return an action uniformly from non-zero entries
    # Deprecated for get_random_action_weighted; more efficient
    # def get_random_action_from_filter(self, filter, max_tries=100) -> torch.Tensor:
    #     filter = filter != 0.
    #     while (filter.flatten(1,-1).count_nonzero(dim=1) <= 1).prod().item() != 1:                             # Almost always terminates after one step
    #         temp = torch.rand((filter.size(0),) + self.action_shape, device=self.device) * filter
    #         filter = (temp == temp.flatten(1,-1).max(1).values.reshape((-1,) + self.action_projshape)).float()
    #         max_tries -= 1
    #         if max_tries == 0:
    #             break
    #     return filter * 1.




# Base class for implementing reinforcement learning algorithms
# Contains some common methods, e.g. simulating games, etc.
class DeepRL():
    def __init__(self, mdp: TensorMDP, q: PrototypeQFunction, device="cpu"):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.device = device
        self.q = q

    def get_statistics(self, state: torch.Tensor) -> str:
        output = f"Action values:\n{self.q.get(state, None)}\n"
        output += f"Action values, masked with softmax:\n{torch.softmax((self.q.get(state, None) + self.mdp.neginf_kill_actions(s)).flatten(1,-1), dim=1).reshape((-1,) + self.mdp.action_shape)}\n"
        return output
    
    def play(self, human_players: list):
        s = self.mdp.get_initial_state()
        total_rewards = torch.zeros(1, self.mdp.num_players)
        while self.mdp.is_terminal(s).item() == False:
            p = int(self.mdp.get_player(s).item())
            print(f"\n{self.mdp.board_str(s)[0]}")

            if p in human_players:
                res = input(self.mdp.input_str)
                a = self.mdp.str_to_action(res)
                if a == None:
                    print("Did not understand input.")
                    continue
                s, r = self.mdp.transition(s,a)
            else:
                print(self.get_statistics(s))
                a = self.q.policy(s)
                print(f"Chosen action: {self.mdp.action_str(a)}\n")
                if self.mdp.is_valid_action(s, a):
                    s, r = self.mdp.transition(s, a)
                else:
                    print("Bot tried to make an illegal move.  Playing randomly.")
                    a = self.mdp.get_random_action(s)
                    print(f"Randomly chosen action: \n{item(a, mdp)}.\n")
                    s, r = self.mdp.transition(s, a)
            total_rewards += r
            print(f"Rewards: {r.tolist()[0]}.")
            print(f"Aggregate rewards: {total_rewards.tolist()[0]}.")
        if r[0,p].item() == 1.:
            winnerstr = f"Player {p + 1} ({self.mdp.symb[p]}), {'a person' if p in human_players else 'a bot'}, won."
        elif r[0,p].item() == 0.:
            winnerstr = 'The game is a tie.'
        else:
            winnerstr = "Somehow I'm not sure who won."
        
        print(f"\n{self.mdp.board_str(s)[0]}\n\n{winnerstr}\nTotal rewards: {total_rewards.tolist()[0]}.\n\n")



    def stepthru_game(self, verbose=False):
        s = self.mdp.get_initial_state()
        print(f"Initial state:\n{self.mdp.board_str(s)[0]}")
        turn = 0
        while self.mdp.is_terminal(s) == False:
            turn += 1
            p = int(self.mdp.get_player(s)[0].item())
            print(f"Turn {turn}, player {p+1} ({self.mdp.symb[p]})")
            if verbose:
                print(f"Values: {self.q.get(s)}")
            a = self.q.policy(s)
            print(f"Chosen action: {self.mdp.action_str(a)[0]}")
            s, r = self.mdp.transition(s, a)
            print(f"Next state:\n{self.mdp.board_str(s)[0]}")
            print(f"Rewards for players: {r[0].tolist()}")
            input("Enter to continue.\n")
        input("Terminal state reached.  Enter to end. ")


    def simulate(self):
        s = self.mdp.get_initial_state()
        while self.mdp.is_terminal(s) == False:
            # p = int(self.mdp.get_player(s)[0].item())
            a = self.q.policy(s)
            if self.mdp.is_valid_action(s, a):
                s, r = self.mdp.transition(s, a)
            else:
                a = self.mdp.get_random_action(s)
                s, r = self.mdp.transition(s, a)
        return r

    def simulate_against_random(self, num_simulations: int, valid_filter=True, replay_loss = False, verbose = True):
        output = []
        for i in range(self.mdp.num_players):
            if verbose:
                print(f"Simulating player {i} against random bot for {num_simulations} simulations.")
            wins, losses, ties, invalids, unknowns  = 0, 0, 0, 0, 0
            for j in range(num_simulations):
                s = self.mdp.get_initial_state()
                if replay_loss:
                    history = [s]
                while self.mdp.is_terminal(s).item() == False:
                    p = int(self.mdp.get_player(s).item())
                    if p == i:
                        a = self.q.policy(s, valid_filter=valid_filter)
                        if self.mdp.is_valid_action(s, a).item():
                            s, r = self.mdp.transition(s, a)
                        else:
                            invalids += 1
                            a = self.mdp.get_random_action(s)
                            s, r = self.mdp.transition(s, a)
                    else:
                        a = self.mdp.get_random_action(s)
                        s, r = self.mdp.transition(s, a)
                    if replay_loss:
                        history.append(s)
                if r[0,i].item() == 1.:
                    wins += 1
                elif r[0, i].item() == -1.:
                    losses += 1
                    if replay_loss:
                        for s in history:
                            print(self.mdp.board_str(s)[0])
                            input()
                elif r[0, i].item() == 0.:
                    ties += 1
                else:
                    unknowns += 1
            output.append((wins, losses, ties, invalids, unknowns))
            if verbose:
                print(f"Player {i} {wins} wins, {losses} losses, {ties} ties, {invalids} invalid moves, {unknowns} unknown results.")
        return output
            
