from qlearn import *
import numpy as np
import random, sys

options = sys.argv[1:]

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
        # State tensor (num_batches, player_channel, board_height, board_width)
        # Mark the game as terminal by turning both players "on"
        # Action tensor (num_batches, num_columns, )
        super().__init__(None, None, discount=1, num_players=2, state_shape=(2,6,7), action_shape=(7,))
        self.symb = {'0': "O", '1': "X", '': "-"}
        self.penalty = -10000000.

    # If player 0 has placed more than player 1, then it's player 1's turn.  Otherwise, it's player 0's turn.
    # Returns a tensor of shape (batch, )
    def get_player(self, state):
        summed = state.sum((2,3))
        return 1 * (summed[:,0] > summed[:,1])
    

    # Returns a tensor of shape (batch, 2)
    def get_player_vector(self, state):
        return torch.eye(2, dtype=float)[self.get_player(state)]

    # Player 0 is always the first player.
    def get_initial_state(self, batch_size=1):
        return torch.tensordot(torch.ones((batch_size, ), dtype=float), torch.zeros((2,6,7), dtype=float), 0)    

    def board_str(self, state):
        # Iterate through the batches and return a list.
        outs = []
        players = self.get_player(state)
        term = self.is_terminal(state)
        for i in range(state.shape[0]):
            s = state[i, ...]
            # We indicate a terminal state by negating everything, so if the state is terminal we need to undo it
            if term[i].item():
                s = s * -1
            out = ""
            out += f"Current player: {self.symb[str(players[i].item())]}\n"
            pos0 = s[0, ...] > s[1, ...]
            pos1 = s[0, ...] < s[1, ...]
            #posvoid = board[0, ...] == board[1, ...]

            # Start from the top row
            for row in range(5, -1, -1):
                rowtext = "|"
                for col in range(7):
                    if pos0[row, col].item():
                        rowtext += self.symb["0"]
                    elif pos1[row, col].item():
                        rowtext += self.symb["1"]
                    else:
                        rowtext += self.symb['']
                    
                out += rowtext + "|\n"
            out += "|1234567|"
            outs.append(out)
        return outs

    # Sum the channels and columns, add the action, should be <= 6.  Then, do an "and" along the rows.
    # Return shape (batch, )
    def is_valid_move(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        return torch.prod(state.sum((1,2)) + action <= torch.ones(action.shape, dtype=float) * 6, 1)

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    # Return tuple of tensors with shape (batch, ) + state_shape for the next state and (batch, 2) for the reward
    # TODO maybe later -- when given an action vector (which might not be standard basis), maybe can select the top valid action?
    def transition(self, state: torch.Tensor, action: torch.Tensor):
        
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")

        #p = self.get_player(state)
        #p_tensor = torch.eye(2, dtype=float)[p]     # Implemented by get_player_vector but more efficient to compute directly
        p_tensor = self.get_player_vector(state)

        # Make the move!  Note this operation is safe: if the move is invalid, no move will be made.
        # First, get the position we want to add by summing the channels and columns: state.sum((1,2))
        # This has shape (batch, 7); we extend it by ones to shape (batch, 6, 7): torch.tensordot(state.sum((1,2)), torch.ones(6, dtype=float), 0)
        coltotals = state.sum((1,2))[:,None,:].expand(-1,6,-1)
        # Then, we compare it to tensor of shape (7, 6) to get a tensor of shape (batch, 6, 7).  The result is a board with True along the correct row
        rowcounter = torch.arange(6, dtype=float)[:,None].expand(-1,7)
        # Then, we multiply it with action to isolate the right column, shape (batch, 6, 7)
        newpos = action[:,None,:].expand(-1,6,-1)  * (coltotals == rowcounter)
        # Then, we put the player channel back in to get a shape (batch, 2, 7, 6)
        newpiece = p_tensor[:,:,None, None].expand(-1,-1,6,7) * newpos[:,None,:,:].expand(-1,2,-1,-1)
        # Then, we add it to the state
        newstate = state + newpiece

        # Check for a winner
        #winner = self.has_winner(newstate, newpiece)
        # Make the state terminal if there is a winner
        #return (-1 * winner) * newstate, # TODO

        # Invalid moves don't result in a change in the board.  However, we want to impose a penalty.
        # Shape (batch, 2)
        reward = (self.is_valid_move(state, action) == False).to(dtype=float) * self.penalty * p_tensor
        

        return newstate, reward

    # TODO: returns a shape (batch, 2) tensor saying who the winner is.  0 if no winner
    def has_winner(self, newstate, newpiece):
        pass
    
    # The game is short enough that maybe we don't care about intermediate states
    def get_random_state(self):
        return self.get_initial_state()

    # Indicate a terminal state by negating everything.  So, if the maximum value of the negation is positive, then it is a terminal state.
    def is_terminal(self, state: torch.Tensor):
        return (state * -1).flatten(1,3).max(1).values > 0
    
    # Sum the channels, and board factors.  The result should be 1 * 1 * 6 * 7 = 42.  If it is greater, something went wrong and we won't account for it.
    def is_full(self, state: torch.Tensor):
        return state.sum((1,2,3)) == 42 * torch.ones(state.shape[0], dtype=float)
    


# Testing
if "test" in options:
    mdp = C4TensorMDP()

    s = mdp.get_initial_state(batch_size=2)
    print("Testing get_player and get_player_vector.  Player 0 taking a turn on board 0.")
    s[0][0][0][0] = 1.
    print(mdp.get_player(s))
    print(mdp.get_player_vector(s))
    print("Player 1 taking a turn on board 0.")
    s[0][1][1][0] = 1.
    print(mdp.get_player(s))
    print(mdp.get_player_vector(s))
    print("Player 0 taking a turn on board 1.")
    s[1][0][0][3] = 1.
    print(mdp.get_player(s))
    print(mdp.get_player_vector(s))

    print("\nTesting printing board.")    
    print(mdp.board_str(s)[0])
    print(mdp.board_str(s)[1])

    print("\nTesting transition.  Placing two pieces in column 1 on board 0, and a piece in column 6, then column 5 on board 1.")
    t = mdp.get_initial_state(batch_size=2)
    a = torch.zeros((2,7), dtype=float)
    a[0, 1] = 1.
    a[1, 6] = 1.
    b = torch.zeros((2,7), dtype=float)
    b[0, 1] = 1.
    b[1, 5] = 1.
    t1, _ = mdp.transition(t, a)
    print(mdp.board_str(t1)[0] + "\n" + mdp.board_str(t1)[1])
    t2, _ = mdp.transition(t1, b)
    print(mdp.board_str(t2)[0] + "\n" + mdp.board_str(t2)[1])

    print("\nTesting invalid move handling.  Players each play 4 times in column 0.  Print output of is_valid_move each time.")
    v = mdp.get_initial_state()
    a = torch.zeros((1,7), dtype=float)
    a[0,0] = 1.
    for i in range(6):
        print("is_valid_move", mdp.is_valid_move(v, a))
        v, pen = mdp.transition(v, a)
        print(mdp.board_str(v)[0])
        print("reward", pen)
    print("Now player 0 will attempt to play in column 0, then column 1.  Then player 1 will attempt to play in column 0.")
    print("is_valid_move", mdp.is_valid_move(v, a))
    v, pen = mdp.transition(v, a)
    print(mdp.board_str(v)[0])
    print("reward", pen)
    b = torch.zeros((1,7), dtype=float)
    b[0,1] = 1
    print("is_valid_move", mdp.is_valid_move(v, b))
    v, pen = mdp.transition(v, b)
    print(mdp.board_str(v)[0])
    print("reward", pen)
    print("is_valid_move", mdp.is_valid_move(v, a))
    v, pen = mdp.transition(v, a)
    print(mdp.board_str(v)[0])
    print("reward", pen)

    print("\nChecking win condition, draw condition and terminal state.")

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

