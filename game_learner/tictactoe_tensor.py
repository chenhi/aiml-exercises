from qlearn import *
import numpy as np
import random


class TTTTensorMDP(MDP):
    def __init__(self):
        self.symb = {0: "X", 1: "O", -1: "."}
        super().__init__(None, None, discount=1, num_players=2, state_shape=(2,3,3), action_shape=(3,3), batched=True)


    def board_str(self, s):
        out = ""
        p, arr = s[0], self.state_to_array(s[1])
        out += f"Current player: {self.symb[p]}\n"
        for row in arr:
            for x in row:
                out += self.symb[x]
            out += "\n"
        return out

    def is_valid_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        return torch.prod(torch.prod(state.sum(1) + action <= 1, dim=1), dim=1) == 1

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    def transition(self, state, a):


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