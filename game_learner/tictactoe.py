from qlearn import *
import numpy as np
import random

# The set of states is (at most) the set of ways to label a 3x3 grid with -1, 0, 1, times {-1, +1} to indicate whose turn it is.
# While this is finite, we will just model it as a 10 dimensional vector.
example_state = (1, (1, 0, -1, 1, 1, 0, 0, -1, 0))
example_array = (-1, [[1,0,-1],[1,1,0],[0,-1,0]])
#         x |   | o
#         x | x | 
#           | o |
# It's o's turn

# The set of actions is (at most) the set of spaces in the 3x3 grid.  If a player attempts to place in an invalid spot, the state does not change.
ttt_actions = [(i, j) for i in range(3) for j in range(3)]


# The players are 1, and -1 in that order
class TTTMDP(MDP):
    def __init__(self):
        self.symb = {-1: "X", 1: "O", 0: "*"}
        super().__init__(None, ttt_actions, 1)

    # States are tuples (0 or 1, 9-tuples of 1,0,-1). Arrays are tuples (1 or -1, 3x3 array).
    def state_to_array(self, state):
        return (int(1 - 2 * state[0]), [[state[1][i % 3 + j * 3] for i in range(3)] for j in range(3)])
    
    def array_to_state(self, player, arr):
        flattened = []
        for row in arr:
            flattened += row
        return (int((1 - player)/2), tuple(flattened))
    
    def board_str(self, s):
        out = ""
        p, arr = self.state_to_array(s)
        out += f"Current player: {self.symb[p]}\n"
        for row in arr:
            for x in row:
                out += self.symb[x]
            out += "\n"
        return out

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    def transition(self, state, a):
        # Copy it
        p, s = self.state_to_array(state)
            
        # Check if move is valid
        # If it's not a valid move, give a penalty (a large penality; it should learn to never make these moves)
        if s[a[0]][a[1]] != 0:
            if p == 1:
                penalty = (-1000, 0)
            else:
                penalty = (0, -1000)
            return state, penalty
        
        # If the move is valid, make it
        b = s.copy()
        b[a[0]][a[1]] = p
        newstate = self.array_to_state(p * -1, b)

        # Check for a winner
        w = self.winner(newstate)
        if w == 1 or w == -1:
            return newstate, (w, -w)
        else:
            return newstate, (0,0)
    
    def get_initial_state(self, start_player = 1):
        if start_player == 1 or start_player == -1:
            return (int((1-start_player)/2), (0,0,0,0,0,0,0,0,0))
        else:
            return (random.choice((0,1)), (0,0,0,0,0,0,0,0,0))
    
    # The game is short enough that maybe we don't care about intermediate states
    def get_random_state(self):
        return self.get_initial_state()

    def is_terminal(self, state):
        return False if self.winner(state) == None else True
    
    # Assume there is only one winner
    # Returns 0 if draw, None if still ongoing
    def winner(self, state):
        _, s = self.state_to_array(state)
        for i in range(3):
            # Check rows and columns    
            if s[0][i] == s[1][i] and s[1][i] == s[2][i] and s[0][i] != 0:
                return s[0][i]
            if s[i][0] == s[i][1] and s[i][1] == s[i][2] and s[i][0] != 0:
                return s[i][0]
        # Check diagonals
        if s[0][0] == s[1][1] and s[1][1] == s[2][2] and s[1][1] != 0:
            return s[1][1]
        if s[0][2] == s[1][1] and s[1][1] == s[2][0] and s[1][1] != 0:
            return s[1][1]
        if self.is_full(state):
            return 0
        return None

    def is_full(self, state):
        _, s = self.state_to_array(state)
        for i in range(3):
            for j in range(3):
                if s[i][j] == 0:
                    return False
        return True
    



