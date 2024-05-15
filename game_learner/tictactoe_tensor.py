from rlbase import TensorMDP, PrototypeQFunction
import torch
from torch import nn


class TTTTensorMDP(TensorMDP):

    def __init__(self, discount=1, penalty=-1, device = "cpu"):
        defaults1 = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.5, 
            'dq_episodes': 2000, 
            'episode_length': 15, 
            'sim_batch': 16, 
            'train_batch': 128, 
            'ramp_start': 0,
            'ramp_end': 500,
            'training_delay': 0, 
            'copy_interval_eps': 10
            }
        defaults2 = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.9, 
            'dq_episodes': 1500, 
            'ramp_start': 100,
            'ramp_end': 1400,
            'training_delay': 100,
            'episode_length': 15, 
            'sim_batch': 16, 
            'train_batch': 128,
            'copy_interval_eps': 5
            }
        defaults3 = {
            'lr': 0.001, 
            'greed_start': 0.0, 
            'greed_end': 0.8, 
            'dq_episodes': 1200, 
            'ramp_start': 100,
            'ramp_end': 1100,
            'training_delay': 100,
            'episode_length': 15, 
            'sim_batch': 32, 
            'train_batch': 256,
            'copy_interval_eps': 5
            }
        defaults4 = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.65, 
            'dq_episodes': 1500, 
            'ramp_start': 100,
            'ramp_end': 1400,
            'training_delay': 100,
            'episode_length': 15, 
            'sim_batch': 32, 
            'train_batch': 256,
            'copy_interval_eps': 5
            }
        defaults5 = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.60, 
            'dq_episodes': 2500, 
            'ramp_start': 100,
            'ramp_end': 2500,
            'training_delay': 100,
            'episode_length': 15, 
            'sim_batch': 32, 
            'train_batch': 256,
            'copy_interval_eps': 5
            }
        defaults6 = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.60, 
            'dq_episodes': 4000, 
            'ramp_start': 100,
            'ramp_end': 2500,
            'training_delay': 100,
            'episode_length': 15, 
            'sim_batch': 32, 
            'train_batch': 256,
            'copy_interval_eps': 1
            }
        defaults7 = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.60, 
            'dq_episodes': 800, 
            'ramp_start': 20,
            'ramp_end': 500,
            'training_delay': 20,
            'episode_length': 15, 
            'sim_batch': 128, 
            'train_batch': 256,
            'copy_interval_eps': 1,
            }
        super().__init__(state_shape=(2,3,3), action_shape=(3,3), default_memory=100000, discount=discount, num_players=2, batched=True, default_hyperparameters=defaults7, \
                         symb = {0: "X", 1: "O", None: "-"}, input_str = "Input position to play, e.g. '1, 3' for the 1st row and 3rd column: ", penalty=penalty, num_simulations=10000, device=device)
        
    def __str__(self):
        return f"TTTTensorMDP: discount={self.discount}, penalty={self.penalty}, device={self.device}"


    ##### UI RELATED METHODS #####
    # Non-batch and not used in internal code, efficiency not as important

    def action_str(self, action: torch.Tensor) -> list[str]:
        outs = []
        for b in range(action.shape[0]):
            flat_pos = action[b].flatten().max(0).indices.item()
            outs.append(f"position {flat_pos//3+1, flat_pos % 3+1}")
        return outs


    def str_to_action(self, input: str) -> torch.Tensor:
        coords = input.split(',')
        try:
            i = int(coords[0]) - 1
            j = int(coords[1]) - 1
        except:
            return None
        out = torch.zeros((1,3,3), device=self.device)
        out[0,i,j] = 1
        return out

    def board_str(self, state):
        outs = []
        for i in range(state.size(0)):
            s = abs(state[i])
            board_str = ""
            for i in range(3):
                for j in range(3):
                    if s[0,i,j].item() == 1:
                        board_str += self.symb[0]
                    elif s[1,i,j].item() == 1:
                        board_str += self.symb[1]
                    else:
                        board_str += self.symb[None]
                board_str += "\n"
            outs.append(board_str)
        return outs



    ##### INTERNAL LOGIC #####


    ### PLAYERS ### 

    def get_player(self, state: torch.Tensor) -> torch.Tensor:
        summed = abs(state.sum((2,3)))
        return (1 * (summed[:,0] > summed[:,1]))[:, None, None, None]

    def get_player_vector(self, state: torch.Tensor) -> torch.Tensor:
        return torch.eye(2, device=self.device)[self.get_player(state)[:, 0, 0, 0]][:,:,None, None]

    # Return shape (batch, 2, 1, 1)
    def swap_player(self, player_vector: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(player_vector, torch.eye(2, device=self.device)[[1,0]], dims=([1], [0])).swapaxes(1, -1)


    ### ACTIONS ###

    def valid_action_filter(self, state: torch.Tensor) -> torch.Tensor:
        return state.sum((1)) == 0
    


    ### STATES ###


    def get_initial_state(self, batch_size=1) -> torch.Tensor:
        return torch.zeros((batch_size, 2, 3, 3), device=self.device)

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    def transition(self, state, action):
        action = (self.is_terminal(state) == False)[:, 0, :, :] * action
        p = self.get_player(state)
        p_tensor = self.get_player_vector(state)

        new_state = state + (p_tensor * action[:,None,:,:]) * self.is_valid_action(state, action)

        winner = self.is_winner(new_state, p)
        full = self.is_full(new_state)

        new_state = (1 - 2 * (~self.is_terminal(state) & (winner | full))) * new_state

        reward = (winner * p_tensor - winner * self.swap_player(p_tensor))[:,:,0,0]
        reward += (torch.logical_and(self.is_valid_action(state, action) == False, self.is_terminal(state).reshape((-1,) + self.state_projshape) == False) * self.penalty * p_tensor)[:,:,0,0]

        return new_state, reward

    def is_winner(self, state, player):
        s = state[torch.arange(state.size(0), device=self.device), player[:,0,0,0]]
        return ((s.sum(1) == 3).sum(1) + (s.sum(2) == 3).sum(1) + (s[:,0,0] + s[:,1,1] + s[:,2,2] == 3) + (s[:,0,2] + s[:,1,1] + s[:,2,0] == 3) > 0)[:,None,None,None]

    def get_random_state(self):
        return self.get_initial_state()

    def is_terminal(self, state: torch.Tensor) -> torch.Tensor:
        return ((state * -1).flatten(1,3).max(1).values > 0)[:, None, None, None]
    
    def is_full(self, state: torch.Tensor) -> torch.Tensor:
        return (abs(state.sum((1,2,3))) == 9)[:, None, None, None]
    
    ##### TESTING FUNCTIONS #####

    def orbit(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.cat([y, y.flip(-1)])
        return torch.cat([y, y.rot90(k=1, dims=[-1,-2]), y.rot90(k=2, dims=[-1,-2]), y.rot90(k=3, dims=[-1,-2])])

    def tests(self, qs: list[PrototypeQFunction]):
        
        # Vs. random test
        


        # Test 0: the AI is playing as player 0 (first player, X).  We are interested in the 8 boards in the orbit of the first board:
        # XO.    +O.    +OO
        # +X.    XXO    -X-
        # +.O    +..    X--
        # Here, + means a winning move, . a tie move, and - a losing move.
        # The other two board to the right are also possibilities.  We only track the first one.  Note that it is possible for the bot to select other branches, so
        # the pass score might be negative with a still-optimal AI.

        s = torch.tensor([[[[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]], [[0.,1.,0.],[0.,0.,0.],[0.,0.,1.]]]]).to(device=self.device)                                      
        win_s = torch.cat([s,s])
        tie_s = torch.cat([s,s,s])
        win_a = torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,0.],[1.,0.,0.]]]).to(device=self.device)            
        tie_a = torch.tensor([[[0.,0.,0.],[0.,0.,0.],[0.,1.,0.]],[[0.,0.,1.],[0.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,1.],[0.,0.,0.]]]) .to(device=self.device) 
        
        win_values = qs[0].get(self.orbit(win_s), self.orbit(win_a))
        tie_values = qs[0].get(self.orbit(tie_s), self.orbit(tie_a))

        win_stdev = win_values.std().item()
        tie_stdev = tie_values.std().item()
        mean_diff = win_values.mean().item() - tie_values.mean().item()
        min_diff = win_values.min().item() - tie_values.max().item()
        
        separated_win_max = win_values.reshape(8,-1).flatten(1,-1).max(1).values
        separated_tie_max = tie_values.reshape(8,-1).flatten(1,-1).max(1).values
        pass_score = (separated_win_max - separated_tie_max).min().item()

        test0 = {'winning play stdev': win_stdev, 'losing stdev': tie_stdev, 'diff of means': mean_diff, 'min distance': min_diff, 'test pass score': pass_score}
        
        
        # Test 1: the AI is playing as player 1 (second player, O). We are interested in the boards:
        # X.-   -.X
        # .O.   .O.
        # -.X   X.-
        # We want to see clustering amongst the Q-values for playing on the sides vs. on the corners, with the sides being higher.
        s = torch.tensor([[[[1.,0.,0.],[0.,0.,0.],[0.,0.,1.]],[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]], [[[0.,0.,1.],[0.,0.,0.],[1.,0.,0.]],[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]]]).to(device=self.device) 
        side_a = torch.tensor([[[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[1.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,1.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.],[0.,1.,0.]]]).to(device=self.device) 
        side_values = qs[1].get(s[[0,0,0,0,1,1,1,1]], side_a[[0,1,2,3,0,1,2,3]]).to(device=self.device) 
        corner_a = torch.tensor([[[0.,0.,1.],[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.],[1.,0.,0.]],[[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]]).to(device=self.device) 
        corner_values = qs[1].get(s[[0,0,1,1]], corner_a).to(device=self.device) 

        side_stdev = side_values.std().item()
        corner_stdev = corner_values.std().item()
        mean_diff = side_values.mean().item() - corner_values.mean().item()
        min_diff = side_values.min().item() - corner_values.max().item()

        separated_side_max = side_values.reshape(2,-1).flatten(1,-1).max(1).values
        separated_corner_max = corner_values.reshape(2,-1).flatten(1,-1).max(1).values
        pass_score = (separated_side_max - separated_corner_max).min().item()

        test1 = {'side play (tie) stdev': side_stdev, 'corner play (lose) stdev': corner_stdev, 'diff of means': mean_diff, 'min distance': min_diff, 'test pass score': pass_score}

        return [test0, test1]

    






# Some prototyping the neural network
# Input tensors have shape (batch, 2, 3, 3)
class TTTNN(nn.Module):


    def __init__(self, channels=32, num_hiddens=3):
        super().__init__()
        self.stack = {}

        # Tests
        self.stack = nn.Sequential(
            nn.Conv2d(2, channels, (3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            )
        for i in range(num_hiddens):
            self.stack.append(nn.Conv2d(channels, channels, (3,3), padding='same'))
            self.stack.append(nn.LeakyReLU())
            self.stack.append(nn.BatchNorm2d(channels))
        self.stack.append(nn.Conv2d(channels, 1, (3,3), padding='same'))


    # Output of the stack is shape (batch, 1, 3, 3), so we do a simple reshaping.
    def forward(self, x):
        return self.stack(x)[:,0]


class TTTResNN(nn.Module):
    def __init__(self, num_hiddens = 4, hidden_depth=1, hidden_width = 32):
        super().__init__()
        self.head_stack = nn.Sequential(
            nn.Conv2d(2, hidden_width, (3,3), padding='same'),
            nn.BatchNorm2d(hidden_width),
            nn.ReLU(),
        )
        hlays = []
        for i in range(num_hiddens):
            lay = nn.Sequential(nn.Conv2d(hidden_width, hidden_width, (3,3), padding='same'))
            for j in range(hidden_depth-1):
                lay.append(nn.BatchNorm2d(hidden_width))
                lay.append(nn.ReLU())
                lay.append(nn.Conv2d(hidden_width, hidden_width, (3,3), padding='same'))
            lay.append(nn.BatchNorm2d(hidden_width))
            hlays.append(lay)
        
        self.hidden_layers = nn.ModuleList(hlays)
        self.tail = nn.Conv2d(hidden_width, 1, (3,3), padding='same')
        self.relu = nn.ReLU()                                               # This is necessary for saving?

    def forward(self, x):
        x = self.head_stack(x)
        for h in self.hidden_layers:
            x = self.relu(h(x) + x)
        return self.tail(x)[:,0]

class TTTCatNN(nn.Module):
    def __init__(self, num_hiddens = 3, hidden_width = 32):
        super().__init__()
        self.head_stack = nn.Sequential(
            nn.Conv2d(2, hidden_width, (3,3), padding='same'),
            nn.BatchNorm2d(hidden_width),
            nn.ReLU(),
        )
        hlays = []
        for i in range(num_hiddens):
            lay = nn.Sequential(nn.Conv2d(hidden_width * (i+1) + 2, hidden_width, (3,3), padding='same'))
            lay.append(nn.BatchNorm2d(hidden_width))
            lay.append(nn.ReLU())
            hlays.append(lay)
        
        self.hidden_layers = nn.ModuleList(hlays)
        self.tail = nn.Conv2d(hidden_width, 1, (3,3), padding='same')
        self.relu = nn.ReLU()                                               # This is necessary for saving?

    def forward(self, x):
        outputs = [x, self.head_stack(x)]
        for h in self.hidden_layers:
            outputs.append(h(torch.cat(outputs, dim=1)))
        return self.tail(outputs[-1])[:,0]
