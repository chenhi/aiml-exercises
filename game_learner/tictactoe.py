from qlearn import *
import numpy as np
import random

# State format: (player, 9-tuple)
# While this is finite, we will just model it as a 10 dimensional vector.
example_state = (1, (1, 0, -1, 1, 1, 0, 0, -1, 0))
#         x |   | o
#         x | x | 
#           | o |
# It's o's turn

# The set of actions is (at most) the set of spaces in the 3x3 grid.  If a player attempts to place in an invalid spot, the state does not change.
ttt_actions = [(i, j) for i in range(3) for j in range(3)]


# The players are 1, and -1 in that order
class TTTMDP(MDP):
    def __init__(self):
        self.symb = {0: "X", 1: "O", -1: "."}
        super().__init__(None, ttt_actions, discount=1, num_players=2)

    # States are tuples (0 or 1, 9-tuples of 1,0,-1). Arrays are tuples (1 or -1, 3x3 array).
    def state_to_array(self, state):
        return [[state[i % 3 + j * 3] for i in range(3)] for j in range(3)]
    
    def array_to_state(self, arr):
        flattened = []
        for row in arr:
            flattened += row
        return tuple(flattened)
    
    def board_str(self, s):
        out = ""
        p, arr = s[0], self.state_to_array(s[1])
        out += f"Current player: {self.symb[p]}\n"
        for row in arr:
            for x in row:
                out += self.symb[x]
            out += "\n"
        return out

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    def transition(self, state, a):
        # Copy it
            
        # Check if move is valid
        # If it's not a valid move, give a penalty (a large penality; it should learn to never make these moves)
        # In practice, this penalty should never appear due to the get_actions method.  If it does, there's a bug that should be fixed.
        if state[1][3*a[0] + a[1]] != -1:
            if state[0] == 0:
                penalty = (-1000, 0)
            elif state[0] == 1:
                penalty = (0, -1000)
            return state, penalty
        
        # If the move is valid, make it
        newboard = []
        for i in range(9):
            if i == 3*a[0] + a[1]:
                newboard.append(state[0])
            else:
                newboard.append(state[1][i])
        newstate = (1 - state[0], tuple(newboard))
        
        # Check for a winner
        w = self.winner(newstate)
        if w == 0 or w == 1:
            return newstate, (1 - 2 * w, 2 * w - 1)
        else:
            return newstate, (0,0)
    
    def get_initial_state(self, start_player = 0):
        if start_player == 0 or start_player == 1:
            return (start_player, (-1,-1,-1,-1,-1,-1,-1,-1,-1))
        else:
            return (random.choice([0,1]), (-1,-1,-1,-1,-1,-1,-1,-1,-1))
    
    # The game is short enough that maybe we don't care about intermediate states
    def get_random_state(self):
        return self.get_initial_state()

    def get_actions(self, s = None):
        if s == None:
            return self.actions
        else:
            empty = []
            for i in range(9):
                if s[1][i] == -1:
                    empty.append((i//3, i % 3))
            return self.actions if len(empty) == 0 else empty

    def is_terminal(self, state):
        return False if self.winner(state) == None else True
    
    # Assume there is only one winner
    # Returns 0,1 for winner, -1 if draw, None if still ongoing
    def winner(self, state):
        for i in range(3):
            # Check rows and columns    
            if state[1][3*i] == state[1][3*i+1] and state[1][3*i+1] == state[1][3*i+2] and state[1][3*i] != -1:
                return state[1][3*i]
            if state[1][i] == state[1][i+3] and state[1][i+3] == state[1][i+6] and state[1][i] != -1:
                return state[1][i]
        # Check diagonals
        if state[1][0] == state[1][4] and state[1][4] == state[1][8] and state[1][0] != -1:
            return state[1][0]
        if state[1][2] == state[1][4] and state[1][4] == state[1][6] and state[1][2] != -1:
            return state[1][2]
        if self.is_full(state):
            return -1
        return None

    def is_full(self, state):
        for i in range(9):
            if state[1][i] == -1:
                return False
        return True
    
    def get_player(self, state):
        return state[0]



