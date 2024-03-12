from qlearn import *
import numpy as np
import random

# Set of states: (player, 7-tuple of strings of 0 and 1 of length in [0, 6], winner or None if not terminal)
# Set of actions: integers [0, 6]

class C4MDP(MDP):
    def __init__(self):
        super().__init__(None, range(0, 7), discount=1, num_players=2, state_shape=(2,7,6), action_shape=(7,))
        self.symb = {'0': "O", '1': "X", '': "-"}

    def board_str(self, s):
        rows = ["|" for i in range(6)]
        p, cols, _ = s
        out = ""
        out += f"Current player: {self.symb[str(p)]}\n"
        for col in cols:
            for c in range(6):
                rows[5 - c] += self.symb[''] if c >= len(col) else self.symb[col[c]]
        for row in rows:
            out += f"{row}|\n"
        out += "|1234567|"
        return out

    def get_actions(self, state):
        notfull = []
        for i in range(7):
            if len(state[1][i]) < 6:
                notfull.append(i)
        if len(notfull) > 0:
            return notfull
        else:
            return self.actions
    
    def state_to_tensor(self, state):
        # Ignore player and winner
        tensor = np.zeros((2,7,6), dtype=float)
        for i in range(7):
            for j in range(len(state[1][i])):
                tensor[int(state[1][i][j])][i][j] = 1.
        return tensor
    
    # Input shape (2, 7, 6), i.e. no batch.  Any non-zero value in a channel is interpreted as 1.
    def tensor_to_state(self, state_tensor: torch.Tensor):
        pass

    # Input shape (batch_size, 7) or (7, )
    # Returns torch.Tensor in first case, scalar in second case
    def tensor_to_action(self, action_tensor: torch.Tensor):
        if len(action_tensor.shape) == 1:
            return action_tensor.max().indices.item()
        elif len(action_tensor.shape) == 2:
            return action_tensor.max(1).indices


    def action_to_tensor(self, action):
        tensor = np.zeros((7,))
        tensor[action] = 1.
        return tensor

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    def transition(self, state, a):
        # Copy it
        p, s, _ = state
            
        # Check if move is valid, i.e. if the column is full and if a is in the set of actions
        # If it's not a valid move, give a penalty (a large penality; it should learn to never make these moves)
        if a not in self.actions or len(s[a]) >= 6:
            if p == 0:
                penalty = (-1000, 0,)
            else:
                penalty = (0, -1000)
            return state, penalty
        
        # If the move is valid, make it
        b = list(s)
        b[a] += str(p)
        b = tuple(b)

        # Check for a new winner, i.e. a winner made by the current move, which can only be the current player
        # w = (1, -1) if p == 0 else (-1, 1)
        # w = (1 - 2*p, -1 + 2*p)
        if self.four(p, b, a):
            return (1 - p, b, p), (1 - 2*p, -1 + 2*p)
        else:
            return (1 - p, b, None), (0, 0)
    
    def get_initial_state(self, start_player = 0):
        if start_player != 0 and start_player != 1:
            start_player = random.choice((0,1))
        return (start_player, ("","","","","","",""), None)
    
    # The game is short enough that maybe we don't care about intermediate states
    def get_random_state(self):
        return self.get_initial_state()

    def is_terminal(self, state):
        return False if state[2] == None else True
    
    # '0' or '1' for a player, '' for empty
    def get_by_coords(self, s, row, col):
        return '' if (row >= len(s[col]) <= row or row < 0) else s[col][row]

    # Determine if the move in the indicate column created new four
    def four(self, p, s, a):
        # Check the column; it has to be at the top, i.e. end
        if len(s[a]) >= 4 and s[a][-4:] == str(p) * 4:
            return True

        # Check the row
        r = len(s[a]) - 1
        length = 1
        # To the right
        for i in range(a+1, 7):
            if self.get_by_coords(s, r, i) != str(p):
                break
            length += 1
        # To the left
        for i in range(a-1, -1, -1):
            if self.get_by_coords(s, r, i) != str(p):
                break
            length += 1
        if length >= 4:
            return True
        
        # Check the up-diagonal
        length = 1
        # To the right
        for i in range(a+1, 7):
            if self.get_by_coords(s, r+i-a, i) != str(p):
                break
            length += 1
        # To the left
        for i in range(a-1, -1, -1):
            if self.get_by_coords(s, r+i-a, i) != str(p):
                break
            length += 1
        if length >= 4:
            return True
        
        # Check the down-diagonal
        length = 1
        # To the right
        for i in range(a+1, 7):
            if self.get_by_coords(s, r-i+a, i) != str(p):
                break
            length += 1
        # To the left
        for i in range(a-1, -1, -1):
            if self.get_by_coords(s, r-i+a, i) != str(p):
                break
            length += 1
        if length >= 4:
            return True
        
        return False
        

    def is_full(self, state):
        s = state[1]
        for row in s:
            if len(row) < 6:
                return False
        return True
    


class C4TensorMDP(MDP):
    def __init__(self):
        # State tensor (which_player, player_channel, board_width, board_height)
        # Action tensor (num_columns, )
        super().__init__(None, None, discount=1, num_players=2, state_shape=(2,2,7,6), action_shape=(7,))
        self.symb = {'0': "O", '1': "X", '': "-"}
        self.swap = torch.tensor([[0,1],[1,0]], dtype=float)

    def swap_player(self, state):
        return torch.tensordot(self.swap, state, ([1], [0]))

    # Assumes the state is valid.
    def get_player(self, state):
        return state.sum((1,2,3)).max(0).indices.item()
    

    def get_initial_state(self, start_player = 0):
        if start_player != 0 and start_player != 1:
            start_player = random.choice((0,1))
        if start_player == 0:
            return torch.tensordot(torch.tensor([1,0], dtype=float), torch.zeros((2,7,6), dtype=float), 0)
        else:
            return torch.tensordot(torch.tensor([0,1], dtype=float), torch.zeros((2,7,6), dtype=float), 0)
    

    def board_str(self, state):
        rows = ["|" for i in range(6)]
        p = self.get_player(state)
        out += f"Current player: {self.symb[str(p)]}\n"
        board = state[p,...]
        pos0 = board[0, ...] > board[1, ...]
        pos1 = board[0, ...] < board[1, ...]
        #posvoid = board[0, ...] == board[1, ...]

        # Start from the top row
        out = ""
        for row in range(5, -1, -1):
            rowtext = "|"
            for col in range(7):
                if pos0[row, col].item():
                    c = self.symb["0"]
                elif pos1[row, col].item():
                    c = self.symb["1"]
                else:
                    c = self.symb['']
            out += rowtext + "|"
        out += "|1234567|"
        return out

    def get_actions(self, state):
        notfull = []
        for i in range(7):
            if len(state[1][i]) < 6:
                notfull.append(i)
        if len(notfull) > 0:
            return notfull
        else:
            return self.actions
    
    def state_to_tensor(self, state):
        # Ignore player and winner
        tensor = np.zeros((2,7,6), dtype=float)
        for i in range(7):
            for j in range(len(state[1][i])):
                tensor[int(state[1][i][j])][i][j] = 1.
        return tensor
    
    # Input shape (2, 7, 6), i.e. no batch.  Any non-zero value in a channel is interpreted as 1.
    def tensor_to_state(self, state_tensor: torch.Tensor):
        pass

    # Input shape (batch_size, 7) or (7, )
    # Returns torch.Tensor in first case, scalar in second case
    def tensor_to_action(self, action_tensor: torch.Tensor):
        if len(action_tensor.shape) == 1:
            return action_tensor.max().indices.item()
        elif len(action_tensor.shape) == 2:
            return action_tensor.max(1).indices
        else:
            raise Exception(f"Received tensor of shape {action_tensor.shape}, don't know how to interpret as action." )


    def action_to_tensor(self, action):
        tensor = np.zeros((7,))
        tensor[action] = 1.
        return tensor

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    def transition(self, state, a):
        # Copy it
        p, s, _ = state
            
        # Check if move is valid, i.e. if the column is full and if a is in the set of actions
        # If it's not a valid move, give a penalty (a large penality; it should learn to never make these moves)
        if a not in self.actions or len(s[a]) >= 6:
            if p == 0:
                penalty = (-1000, 0,)
            else:
                penalty = (0, -1000)
            return state, penalty
        
        # If the move is valid, make it
        b = list(s)
        b[a] += str(p)
        b = tuple(b)

        # Check for a new winner, i.e. a winner made by the current move, which can only be the current player
        # w = (1, -1) if p == 0 else (-1, 1)
        # w = (1 - 2*p, -1 + 2*p)
        if self.four(p, b, a):
            return (1 - p, b, p), (1 - 2*p, -1 + 2*p)
        else:
            return (1 - p, b, None), (0, 0)
    

    
    # The game is short enough that maybe we don't care about intermediate states
    def get_random_state(self):
        return self.get_initial_state()

    def is_terminal(self, state):
        return False if state[2] == None else True
    
    # '0' or '1' for a player, '' for empty
    def get_by_coords(self, s, row, col):
        return '' if (row >= len(s[col]) <= row or row < 0) else s[col][row]

    # Determine if the move in the indicate column created new four
    def four(self, p, s, a):
        # Check the column; it has to be at the top, i.e. end
        if len(s[a]) >= 4 and s[a][-4:] == str(p) * 4:
            return True

        # Check the row
        r = len(s[a]) - 1
        length = 1
        # To the right
        for i in range(a+1, 7):
            if self.get_by_coords(s, r, i) != str(p):
                break
            length += 1
        # To the left
        for i in range(a-1, -1, -1):
            if self.get_by_coords(s, r, i) != str(p):
                break
            length += 1
        if length >= 4:
            return True
        
        # Check the up-diagonal
        length = 1
        # To the right
        for i in range(a+1, 7):
            if self.get_by_coords(s, r+i-a, i) != str(p):
                break
            length += 1
        # To the left
        for i in range(a-1, -1, -1):
            if self.get_by_coords(s, r+i-a, i) != str(p):
                break
            length += 1
        if length >= 4:
            return True
        
        # Check the down-diagonal
        length = 1
        # To the right
        for i in range(a+1, 7):
            if self.get_by_coords(s, r-i+a, i) != str(p):
                break
            length += 1
        # To the left
        for i in range(a-1, -1, -1):
            if self.get_by_coords(s, r-i+a, i) != str(p):
                break
            length += 1
        if length >= 4:
            return True
        
        return False
        

    def is_full(self, state):
        s = state[1]
        for row in s:
            if len(row) < 6:
                return False
        return True
    






# Some prototyping the neural network
# Input tensors have shape (batch, 7, 2, 7, 6)
class C4NN(nn.Module):
    def __init__(self):
        self.stack = nn.Sequential(
            nn.Flatten(0, 1),
            nn.Conv2d(2, 64, (4,4), padding='same'),
            nn.Unflatten(0, (-1, 7)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(7*2*7*6*64, 7*2*7*6*64),
            nn.ReLU(),
            nn.Linear(7*2*7*6*64, 7) # ok im still confused about whether we are learning q or learning p
        )

