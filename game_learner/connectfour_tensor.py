from qlearn import MDP
from deepqlearn import *
import numpy as np
import sys, warnings
import torch


# A shift function with truncation, like torch.roll except without the "rolling over"
def shift(x: torch.Tensor, shift: int, axis: int, device="cpu") -> torch.Tensor:
    if shift == 0:
        return x
    if abs(shift) >= x.shape[axis]:
        return torch.zeros(x.shape, device=device)
    
    zero_shape = list(x.shape)
    if shift > 0:
        zero_shape[axis] = shift
        return torch.cat((torch.zeros(zero_shape, device=device), torch.index_select(x, axis, torch.arange(0, x.shape[axis] - shift, device=device))), axis)
    else:
        zero_shape[axis] = -shift
        return torch.cat((torch.index_select(x, axis, torch.arange(-shift, x.shape[axis], device=device)), torch.zeros(zero_shape, device=device)), axis)




# State tensor shape (batches, player_channel = 2, row = 6, column = 7)
# Action tensor (batches, column = 7)
# Reward tensor (batches, )
class C4TensorMDP(TensorMDP):

    def __init__(self, device="cpu"):
        hyperpar = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.6,
            'dq_episodes': 3000, 
            'ramp_start': 100,
            'ramp_end': 1900,
            'training_delay': 100,
            'episode_length': 45,
            'sim_batch': 128, 
            'train_batch': 256, 
            'copy_interval_eps': 3
            }
        super().__init__(state_shape=(2,6,7), action_shape=(7,), discount=1, num_players=2, batched=True, default_memory = 1000000, default_hyperparameters=hyperpar, \
                         symb = {0: "O", 1: "X", None: "-"}, input_str = "Input column to play (1-7). ", penalty=-2)
        self.device=device
        
        # Generate kernels for detecting winner
        # Shape: (6*7, 16, 6, 7)
        # The first dimension corresponds to the flattened index of the center (i, j) <--> 7i + j
        stacks = []
        for i in range(6):
            for j in range(7):
                filters = []
                center = torch.zeros((6, 7), device=device)
                center[i, j] = 1
                # Make the vertical filters
                u1 = shift(center, 1, 0, device=device)
                u2 = shift(center, 2, 0, device=device)
                u3 = shift(center, 3, 0, device=device)
                d1 = shift(center, -1, 0, device=device)
                d2 = shift(center, -2, 0, device=device)
                d3 = shift(center, -3, 0, device=device)
                filters.append(center + u1 + u2 + u3)
                filters.append(d1 + center + u1 + u2)
                filters.append(d2 + d1 + center + u1)
                filters.append(d3 + d2 + d1 + center)

                # Make the horizontal filters
                r1 = shift(center, 1, 1, device=device)
                r2 = shift(center, 2, 1, device=device)
                r3 = shift(center, 3, 1, device=device)
                l1 = shift(center, -1, 1, device=device)
                l2 = shift(center, -2, 1, device=device)
                l3 = shift(center, -3, 1, device=device)
                filters.append(center + r1 + r2 + r3)
                filters.append(l1 + center + r1 + r2)
                filters.append(l2 + l1 + center + r1)
                filters.append(l3 + l2 + l1 + center)
                
                # Make the diagonal filters
                ur1 = shift(shift(center, 1, 0, device=device), 1, 1, device=device)
                ur2 = shift(shift(center, 2, 0, device=device), 2, 1, device=device)
                ur3 = shift(shift(center, 3, 0, device=device), 3, 1, device=device)
                dl1 = shift(shift(center, -1, 0, device=device), -1, 1, device=device)
                dl2 = shift(shift(center, -2, 0, device=device), -2, 1, device=device)
                dl3 = shift(shift(center, -3, 0, device=device), -3, 1, device=device)
                filters.append(center + ur1 + ur2 + ur3)
                filters.append(dl1 + center + ur1 + ur2)
                filters.append(dl2 + dl1 + center + ur1)
                filters.append(dl3 + dl2 + dl1 + center)

                # Make the diagonal filters
                ul1 = shift(shift(center, 1, 0, device=device), -1, 1, device=device)
                ul2 = shift(shift(center, 2, 0, device=device), -2, 1, device=device)
                ul3 = shift(shift(center, 3, 0, device=device), -3, 1, device=device)
                dr1 = shift(shift(center, -1, 0, device=device), 1, 1, device=device)
                dr2 = shift(shift(center, -2, 0, device=device), 2, 1, device=device)
                dr3 = shift(shift(center, -3, 0, device=device), 3, 1, device=device)
                filters.append(center + ul1 + ul2 + ul3)
                filters.append(dr1 + center + ul1 + ul2)
                filters.append(dr2 + dr1 + center + ul1)
                filters.append(dr3 + dr2 + dr1 + center)

                # Shape (16, 6, 7)
                stacks.append(torch.stack(filters))
        # Shape (42, 16, 6, 7)
        self.filter_stack = torch.stack(stacks)

    def __str__(self):
        return f"C4TensorMDP: discount={self.discount}, penalty={self.penalty}"

    ##### UI RELATED METHODS #####
    # Non-batch and not used in internal code, efficiency not as important
        
    # Return shape (1, 7)
    def int_to_action(self, index: int):
        if index < 0 or index > 7:
            warnings.warn("Index out of range.")
            return np.zeros((1, 7), device=self.device)
        return torch.eye(7, device=self.device)[index:index+1,:]
    
    def str_to_action(self, input: str) -> torch.Tensor:
        try:
            i = int(input) - 1
            if i < 0 or i > 7:
                raise Exception()
        except:
            return None
        return self.int_to_action(i)
    
    def action_str(self, action: torch.Tensor) -> list[str]:
        outs = []
        for i in range(action.shape[0]):
            a = action[i, :].max(0).indices.item()
            outs.append(f"column {a+1}")
        return outs

    # Returns a pretty looking board in a string.
    def board_str(self, state: torch.Tensor) -> list[str]:
        # Iterate through the batches and return a list.
        outs = []
        players = self.get_player(state)
        term = self.is_terminal(state)
        for i in range(state.shape[0]):
            s = state[i]
            # We indicate a terminal state by negating everything, so if the state is terminal we need to undo it
            if term[i].item():
                s = s * -1
            out = ""
            out += f"Current player: {self.symb[players[i].item()]}\n"
            pos0 = s[0] > s[1]
            pos1 = s[0] < s[1]

            # Start from the top row
            for row in range(5, -1, -1):
                rowtext = "|"
                for col in range(7):
                    if pos0[row, col].item():
                        rowtext += self.symb[0]
                    elif pos1[row, col].item():
                        rowtext += self.symb[1]
                    else:
                        rowtext += self.symb[None]
                    
                out += rowtext + "|\n"
            out += "|=======|\n|1234567|"
            outs.append(out)
        return outs

    ##### INTERNAL LOGIC #####


    ### PLAYERS ### 

    # Return shape (batch, 1, 1, 1)
    def get_player(self, state: torch.Tensor) -> torch.Tensor:
        return (state.sum((1,2,3)) % 2)[:,None,None,None]

    # Return shape (batch, 2, 1, 1)
    def get_player_vector(self, state: torch.Tensor) -> torch.Tensor:
        return torch.eye(2, device=self.device)[None].expand(state.size(0),-1,-1)[torch.arange(state.size(0)),self.get_player(state).int()[:, 0, 0, 0]][:,:,None, None]
    
    # Return shape (batch, 2, 1, 1)
    def swap_player(self, player_vector: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(player_vector, torch.eye(2, device=self.device)[[1,0]], dims=([1], [0])).swapaxes(1, -1)



    ### ACTIONS ###

    # Return shape (batch, 7), boolean type
    def valid_action_filter(self, state: torch.Tensor) -> torch.Tensor:
        return state.sum((1,2)) < 6


    ### STATES ###

    # Return shape (batch_size, 2, 6, 7)
    def get_initial_state(self, batch_size=1) -> torch.Tensor:
        return torch.zeros((batch_size, 2, 6, 7), device=self.device) 
    
    # Checks for a winner centered at a given position
    # Inputs: state has stape (batch, 2, 6, 7), player has shape (batch, ), and center has shape (batch, 6, 7)
    # Returns a shape (batch, 1, 1, 1) boolean saying whether the indicated player is a winner.
    def is_winner(self, state: torch.Tensor, player: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return ((torch.tensordot((state[torch.arange(state.shape[0]), player.int()[:,0,0,0]][:,None,None,:,:] * center[:,:,:,None,None]).flatten(1, 2), self.filter_stack, ([1,2,3], [0,2,3])) == 4).sum(1) > 0)[:,None, None, None]
        # Unwound version
        # Get the channel for the indicated player, shape (batch, 6, 7)
        #player_board = state[torch.arange(state.shape[0]), player[:,0,0,0]]
        # Put in the position of the last move and flatten it, shape (batch, 42, 6, 7)
        #player_board_center = (player_board[:,None,None,:,:] * center[:,:,:,None,None]).flatten(1, 2)
        # Put in the stack of filters, shape (batch, 16), i.e. contract along the size 42 dimension, then take the dot product between the board and filters, i.e. contract again
        #dotted = torch.tensordot(player_board_center, self.filter_stack, ([1,2,3], [0,2,3]))
        # If any 4's appear, then this is a winner
        #return ((dotted == 4).sum(1) > 0)[:,None, None, None]

    # The game is short enough that maybe we don't care about intermediate states
    def get_random_state(self) -> torch.Tensor:
        return self.get_initial_state()

    # Indicate a terminal state by negating everything.  So, if the maximum value of the negation is positive, then it is a terminal state.
    def is_terminal(self, state: torch.Tensor) -> torch.Tensor:
        return ((state * -1).flatten(1,3).max(1).values > 0)[:, None, None, None]
    
    # Sum the channels, and board factors.  The result should be 1 * 1 * 6 * 7 = 42.  If it is greater, something went wrong and we won't account for it.
    def is_full(self, state: torch.Tensor) -> torch.Tensor:
        return (abs(state.sum((1,2,3))) == 42)[:, None, None, None]


    ### TRANSITION ###

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    # Return tuple of tensors with shape (batch, ) + state_shape for the next state and (batch, 2) for the reward
    def transition(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # If the state is terminal, don't do anything.  Zero out the corresponding actions.
        action = (self.is_terminal(state) == False)[:, :, 0, 0] * action
        p = self.get_player(state)
        p_tensor = self.get_player_vector(state)

        # Make the move!  If the move is invalid, no move will be made.
        # First, get the position we want to add by summing the channels and columns: state.sum((1,2))
        # This has shape (batch, 7); we extend it by ones to shape (batch, 6, 7): torch.tensordot(state.sum((1,2)), torch.ones(6), 0)
        #coltotals = state.sum((1,2))[:,None,:].expand(-1,6,-1)
        # Then, we compare it to tensor of shape (7, 6) to get a tensor of shape (batch, 6, 7).  The result is a board with True along the correct row
        #rowcounter = torch.arange(6)[:,None].expand(-1,7)
        # Then, we multiply it with action to isolate the right column, shape (batch, 6, 7)
        #newpos = action[:,None,:].expand(-1,6,-1)  * (coltotals == rowcounter)
        newpos = action[:,None,:].expand(-1,6,-1)  * (state.sum((1,2))[:,None,:].expand(-1,6,-1) == torch.arange(6, device=self.device)[:,None].expand(-1,7))
        
        # Then, we put the player channel back in to get a shape (batch, 2, 7, 6)
        #newpiece = p_tensor * newpos[:,None,:,:].expand(-1,2,-1,-1)
        # Then, we add it to the state
        #newstate = state + newpiece
        newstate = state + p_tensor * newpos[:,None,:,:].expand(-1,2,-1,-1)

        # Check whether the current player is a winner after playing at position newpos
        winner = self.is_winner(newstate, p, newpos)
        # Also check if the board is full, meaning a draw
        full = self.is_full(newstate)
        # Check if the original state was terminal
        was_terminal = self.is_terminal(state)

        # Make the state terminal if there is a winner or if the board is full, and if it wasn't terminal to begin with
        newstate = (1 - 2 * (~was_terminal & (winner | full))) * newstate
        
        # Give out a reward if there is a winner.
        reward = (winner.float() * p_tensor - winner * self.swap_player(p_tensor))[:,:,0,0]
        # Invalid moves don't result in a change in the board.  However, we want to impose a penalty (except when it's a terminal state)
        reward += (torch.logical_and(self.is_valid_action(state, action) == False, self.is_terminal(state).reshape((-1,) + self.state_projshape) == False).float() * self.penalty * p_tensor)[:,:,0,0]
        
        return newstate, reward
    



    

class C4ResNN(nn.Module):
    def __init__(self, num_hidden_conv = 5, hidden_conv_depth=1, hidden_conv_layers = 32, num_hidden_linear = 3, hidden_linear_depth=2, hidden_linear_width=16):
        super().__init__()
        self.head_stack = nn.Sequential(
            nn.Conv2d(2, hidden_conv_layers, (3,3), padding='same'),
            nn.BatchNorm2d(hidden_conv_layers),
            nn.ReLU(),
        )
        hlays = []
        for i in range(num_hidden_conv - 1):
            lay = nn.Sequential(nn.Conv2d(hidden_conv_layers, hidden_conv_layers, (5,5), padding='same'))
            for j in range(hidden_conv_depth):
                lay.append(nn.BatchNorm2d(hidden_conv_layers))
                lay.append(nn.ReLU())
                lay.append(nn.Conv2d(hidden_conv_layers, hidden_conv_layers, (5,5), padding='same'))
            lay.append(nn.BatchNorm2d(hidden_conv_layers))
            hlays.append(lay)
        
        self.hidden_conv_layers = nn.ModuleList(hlays)

        self.conv_to_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_conv_layers * 7 * 6, hidden_linear_width * 7 * 6)
        )

        hlays = []
        for i in range(num_hidden_linear):
            lay = nn.Sequential(nn.Linear(hidden_linear_width * 42, hidden_linear_width * 42))
            for j in range(hidden_linear_depth - 1):
                lay.append(nn.BatchNorm1d(hidden_linear_width*42))
                lay.append(nn.ReLU())
                lay.append(nn.Linear(hidden_linear_width * 42, hidden_linear_width * 42))
            lay.append(nn.BatchNorm1d(hidden_linear_width * 42))
            hlays.append(lay)
        
        self.hidden_linear_layers = nn.ModuleList(hlays)


        self.tail = nn.Sequential(
                nn.Linear(hidden_linear_width * 42, 7)    
        )
        self.relu = nn.ReLU()                                               # This is necessary for saving?

    def forward(self, x):
        x = self.head_stack(x)
        for h in self.hidden_conv_layers:
            x = self.relu(h(x) + x)
        x = self.conv_to_linear(x)
        for h in self.hidden_linear_layers:
            x = self.relu(h(x) + x)
        return self.tail(x)
    




class C4NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(2, 32, (5,5), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, (3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Flatten(),
            nn.Linear(16*7*6, 8*7*6),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8*7*6),
            nn.Linear(8*7*6, 4*7*6),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4*7*6),
            nn.Linear(4*7*6, 2*7*6),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2*7*6),
            nn.Linear(2*7*6, 7*6),
            nn.LeakyReLU(),
            nn.BatchNorm1d(7*6),
            nn.Linear(7*6, 7*3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(7*3),
            nn.Linear(7*3, 7)
        )

    def forward(self, x):
        return self.stack(x)
