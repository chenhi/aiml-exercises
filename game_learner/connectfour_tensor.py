from qlearn import MDP
from deepqlearn import *
import numpy as np
import sys, warnings
import torch

# Testing options
# 'test' to test
# 'verbose' to print more messages
# 'slow' to do a slow simulation of Q-updates
# 'saveload' to test saving and loading

options = sys.argv[1:]


# State tensor shape (batches, player_channel = 2, row = 6, column = 7)
# Action tensor (batches, column = 7)
# Reward tensor (batches, )
class C4TensorMDP(MDP):
    def __init__(self):
        super().__init__(None, None, discount=1, num_players=2, state_shape=(2,6,7), action_shape=(7,), batched=True, \
                         symb = {0: "O", 1: "X", None: "-"}, input_str = "Input column to play (1-7). ")


    # Logic: if player 0 has more pieces than player 1, then it's player 1's turn.  Otherwise, it's player 0's turn.
    # Return shape (batch, 1, 1, 1)
    def get_player(self, state: torch.Tensor) -> torch.Tensor:
        summed = abs(state.sum((2,3)))
        return (1 * (summed[:,0] > summed[:,1])).to(dtype=int)[:, None, None, None]

    # Return shape (batch, 2, 1, 1)
    def get_player_vector(self, state: torch.Tensor) -> torch.Tensor:
        return torch.eye(2, dtype=int)[self.get_player(state)[:, 0, 0, 0]][:,:,None, None] # TODO maybe improve this

    # Non-batch and not used in internal code, but might be useful
    # Return shape (1, 7)
    def int_to_action(self, index: int):
        if index < 0 or index > 7:
            warnings.warn("Index out of range.")
            return np.zeros((1, 7), dtype=int)
        return torch.eye(7, dtype=int)[index:index+1,:]
    
    def str_to_action(self, input: str) -> torch.Tensor:
        try:
            i = int(input)
            if i < 0 or i > 7:
                raise Exception()
        except:
            return None
        return self.int_to_action(i)
    
    # Return shape (batch, 7), boolean type
    def valid_action_filter(self, state: torch.Tensor):
        return state.sum((1,2)) < 6

    # Gets a random valid action; if there are none, then it plays in the 0th column. #TODO make it return 0 instead
    def get_random_action(self, state):
        #indices = (torch.rand(state.size(0)) * 7).int()
        indices = (torch.rand((state.size(0), 7)) * self.valid_action_filter(state).int()).max(1).indices
        return torch.eye(7, dtype=int)[None].expand(state.size(0), -1, -1)[torch.arange(state.size(0)), indices]

    # Return shape (batch, 2, 1, 1)
    def swap_player(self, player_vector: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(player_vector, torch.tensor([[0,1],[1,0]], dtype=int), dims=([1], [0])).swapaxes(1, -1)

    # Player 0 is always the first player.
    # Return shape (batch_size, 2, 6, 7)
    def get_initial_state(self, batch_size=1) -> torch.Tensor:
        return torch.zeros((batch_size, 2, 6, 7), dtype=int)
#        return torch.tensordot(torch.ones((batch_size, ), dtype=int), torch.zeros((2,6,7), dtype=int), 0)    

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

    # Sum the channels and columns, add the action, should be <= 6.  Then, do an "and" along the rows.
    # Input shape (batch, 2, 6, 7) and (batch, 7)
    # Return shape (batch, 1)
    def is_valid_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        return torch.prod(state.sum((1,2)) + action <= 6, 1)

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    # Return tuple of tensors with shape (batch, ) + state_shape for the next state and (batch, 2) for the reward
    def transition(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # If the batches don't match, this can cause weird behavior.
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        
        # If the state is terminal, don't do anything.  Zero out the corresponding actions.
        action = (self.is_terminal(state) == False)[:, :, 0, 0] * action
        p = self.get_player(state)
        p_tensor = self.get_player_vector(state)

        # Make the move!  Note this operation is safe: if the move is invalid, no move will be made.
        # First, get the position we want to add by summing the channels and columns: state.sum((1,2))
        # This has shape (batch, 7); we extend it by ones to shape (batch, 6, 7): torch.tensordot(state.sum((1,2)), torch.ones(6, dtype=int), 0)
        coltotals = state.sum((1,2))[:,None,:].expand(-1,6,-1)
        # Then, we compare it to tensor of shape (7, 6) to get a tensor of shape (batch, 6, 7).  The result is a board with True along the correct row
        rowcounter = torch.arange(6, dtype=int)[:,None].expand(-1,7)
        # Then, we multiply it with action to isolate the right column, shape (batch, 6, 7)
        newpos = action[:,None,:].expand(-1,6,-1)  * (coltotals == rowcounter)
        # Then, we put the player channel back in to get a shape (batch, 2, 7, 6)
        newpiece = p_tensor * newpos[:,None,:,:].expand(-1,2,-1,-1)
        # Then, we add it to the state
        newstate = state + newpiece

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
        # Invalid moves don't result in a change in the board.  However, we want to impose a penalty.
        reward += ((self.is_valid_action(state, action) == False).float() * self.penalty * p_tensor)[:,:,0,0]
        
        return newstate, reward

    # A shift function with truncation, like torch.roll except without the "rolling over"
    def shift(self, x: torch.Tensor, shift: int, axis: int) -> torch.Tensor:
        if shift == 0:
            return x
        if abs(shift) >= x.shape[axis]:
            return torch.zeros(x.shape, dtype=int)
        
        zero_shape = list(x.shape)
        if shift > 0:
            zero_shape[axis] = shift
            return torch.cat((torch.zeros(zero_shape, dtype=int), torch.index_select(x, axis, torch.arange(0, x.shape[axis] - shift))), axis)
        else:
            zero_shape[axis] = -shift
            return torch.cat((torch.index_select(x, axis, torch.arange(-shift, x.shape[axis])), torch.zeros(zero_shape, dtype=int)), axis)

    # Inputs: state has stape (batch, 2, 6, 7), player has shape (batch, ), and lastmove has shape (batch, 6, 7)
    # Returns a shape (batch, ) boolean saying whether the indicated player is a winner.
    def is_winner(self, state: torch.Tensor, player: torch.Tensor, lastmove: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != player.shape[0] or player.shape[0] != lastmove.shape[0]:
            raise Exception("Batch sizes must agree.")
        batches = state.shape[0]

        # Get the channel for the indicated player, shape (batch, 6, 7)
        player_board = state[torch.arange(batches), player[:,0,0,0]]

        filters = []
        # Make the horizontal filters
        u1 = self.shift(lastmove, 1, 1)
        u2 = self.shift(lastmove, 2, 1)
        u3 = self.shift(lastmove, 3, 1)
        d1 = self.shift(lastmove, -1, 1)
        d2 = self.shift(lastmove, -2, 1)
        d3 = self.shift(lastmove, -3, 1)
        filters.append(lastmove + u1 + u2 + u3)
        filters.append(d1 + lastmove + u1 + u2)
        filters.append(d2 + d1 + lastmove + u1)
        filters.append(d3 + d2 + d1 + lastmove)

        # Make the horizontal filters
        r1 = self.shift(lastmove, 1, 2)
        r2 = self.shift(lastmove, 2, 2)
        r3 = self.shift(lastmove, 3, 2)
        l1 = self.shift(lastmove, -1, 2)
        l2 = self.shift(lastmove, -2, 2)
        l3 = self.shift(lastmove, -3, 2)
        filters.append(lastmove + r1 + r2 + r3)
        filters.append(l1 + lastmove + r1 + r2)
        filters.append(l2 + l1 + lastmove + r1)
        filters.append(l3 + l2 + l1 + lastmove)
        
        # Make the diagonal filters
        ur1 = self.shift(self.shift(lastmove, 1, 1), 1, 2)
        ur2 = self.shift(self.shift(lastmove, 2, 1), 2, 2)
        ur3 = self.shift(self.shift(lastmove, 3, 1), 3, 2)
        dl1 = self.shift(self.shift(lastmove, -1, 1), -1, 2)
        dl2 = self.shift(self.shift(lastmove, -2, 1), -2, 2)
        dl3 = self.shift(self.shift(lastmove, -3, 1), -3, 2)
        filters.append(lastmove + ur1 + ur2 + ur3)
        filters.append(dl1 + lastmove + ur1 + ur2)
        filters.append(dl2 + dl1 + lastmove + ur1)
        filters.append(dl3 + dl2 + dl1 + lastmove)

        # Make the diagonal filters
        ul1 = self.shift(self.shift(lastmove, 1, 1), -1, 2)
        ul2 = self.shift(self.shift(lastmove, 2, 1), -2, 2)
        ul3 = self.shift(self.shift(lastmove, 3, 1), -3, 2)
        dr1 = self.shift(self.shift(lastmove, -1, 1), 1, 2)
        dr2 = self.shift(self.shift(lastmove, -2, 1), 2, 2)
        dr3 = self.shift(self.shift(lastmove, -3, 1), 3, 2)
        filters.append(lastmove + ul1 + ul2 + ul3)
        filters.append(dr1 + lastmove + ul1 + ul2)
        filters.append(dr2 + dr1 + lastmove + ul1)
        filters.append(dr3 + dr2 + dr1 + lastmove)

        # Shape (16, batch, 6, 7)
        filter_tensor = torch.stack(filters)

        # Sum along the board axes, check how many filters give rise to 4, then sum that number, then check if it is positive
        return (((player_board[None,:,:,:].expand(16, -1, -1, -1) * filter_tensor).sum((2,3)) == (torch.ones(16, dtype=int) * 4)[:,None].expand(-1, batches)).sum(0) > 0)[:,None, None, None]
    
    # The game is short enough that maybe we don't care about intermediate states
    def get_random_state(self) -> torch.Tensor:
        return self.get_initial_state()

    # Indicate a terminal state by negating everything.  So, if the maximum value of the negation is positive, then it is a terminal state.
    def is_terminal(self, state: torch.Tensor) -> torch.Tensor:
        return ((state * -1).flatten(1,3).max(1).values > 0)[:, None, None, None]
    
    # Sum the channels, and board factors.  The result should be 1 * 1 * 6 * 7 = 42.  If it is greater, something went wrong and we won't account for it.
    def is_full(self, state: torch.Tensor) -> torch.Tensor:
        return (abs(state.sum((1,2,3))) == 42 * torch.ones(state.shape[0], dtype=int))[:, None, None, None]
    




# Some prototyping the neural network
# Input tensors have shape (batch, 2, 7, 6)
class C4NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(2, 32, (3,3), padding='same'),
            nn.ReLU(),
            #nn.Dropout(p=0.2),             # Dropout introduces randomness into the Q function.  Not sure if this is desirable.
            #nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (5,5), padding='same'),
            nn.ReLU(),
            #nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64*7*6, 64*7*6),
            nn.ReLU(),
            #nn.BatchNorm1d(64*7*6),
            nn.Linear(64*7*6, 7)
        )
    
    def zero_out(self):
        nn.init.zeros_(self.stack[0].weight)
        nn.init.zeros_(self.stack[0].bias)
        nn.init.zeros_(self.stack[2].weight)
        nn.init.zeros_(self.stack[2].bias)
        nn.init.zeros_(self.stack[5].weight)
        nn.init.zeros_(self.stack[5].bias)
        nn.init.zeros_(self.stack[7].weight)
        nn.init.zeros_(self.stack[7].bias)

    def forward(self, x):
        return self.stack(x)





#################### TESTING ####################


# Testing
if "test" in options:
    print("\nRUNNING TESTS:\n")

    # We can print everything or just the pass/fail statements.
    verbose = True if "verbose" in options else False
    mdp = C4TensorMDP()

    s = mdp.get_initial_state(batch_size=2)
    print("Testing batching, get_player and get_player_vector.")
    if verbose:
        print("Creating two boards.  The current player on both boards should be player 0.")
    ok = True
    if verbose:
        print(mdp.get_player(s))
        print(mdp.get_player_vector(s))
    p = mdp.get_player(s)
    pp = mdp.get_player_vector(s)
    if not (p[0,0,0,0].item() == 0 and p[1,0,0,0].item() == 0 and pp[0,:,0,0].tolist() == [1, 0] and pp[1,:,0,0].tolist() == [1, 0]):
        ok = False
    
    if verbose:
        print("\nManually playing on board 0, but not board 1.  The current player on board 0 should be player 1, and on board 1 should be player 0.")
    s[0][0][0][0] = 1.
    if verbose:
        print(mdp.get_player(s))
        print(mdp.get_player_vector(s))
    p = mdp.get_player(s)
    pp = mdp.get_player_vector(s)
    if not (p[0,0,0,0].item() == 1 and p[1,0,0,0].item() == 0 and pp[0,:,0,0].tolist() == [0, 1] and pp[1,:,0,0].tolist() == [1, 0]):
        ok = False

    if ok:
        print ("PASS")
    else:
        print("FAIL!!!")


    print("\nTesting swap player.")   
    if verbose:
        print("initial players:", mdp.get_player_vector(s))
        print("swapped:", mdp.swap_player(mdp.get_player_vector(s)))
    pp = mdp.swap_player(mdp.get_player_vector(s))
    if pp[0,:,0,0].tolist() == [1, 0] and pp[1,:,0,0].tolist() == [0, 1]:
         print("PASS")
    else:
        print("FAIL!!!")

    print("\nTesting printing board and initial state.  The following should be a nicely decorated empty board.")   
    s = mdp.get_initial_state()
    print(mdp.board_str(s)[0])


    print("\nTesting transitions.")
    if verbose:
        print("Board 0: two plays in column 1.  Board 1: a play in column 6, then column 5.")
    t = mdp.get_initial_state(batch_size=2)
    a = torch.zeros((2,7), dtype=int)
    a[0, 1] = 1.
    a[1, 6] = 1.
    b = torch.zeros((2,7), dtype=int)
    b[0, 1] = 1.
    b[1, 5] = 1.
    t1, _ = mdp.transition(t, a)
    if verbose:
        print(mdp.board_str(t1)[0] + "\n" + mdp.board_str(t1)[1])
    t2, _ = mdp.transition(t1, b)
    if verbose:
        print(mdp.board_str(t2)[0] + "\n" + mdp.board_str(t2)[1])
    if t2[0,0,0,1].item() == 1 and t2[0,1,1,1].item() == 1 and t2[1,0,0,6].item() == 1 and t2[1,1,0,5] == 1:
        print("PASS")
    else:
        print("FAIL!!!")

    print("\nTesting invalid move handling and negative rewards.")
    if verbose:
        print("Players each play 4 times in column 0.  The last two should be reported invalid.")
    v = mdp.get_initial_state()
    a = torch.zeros((1,7), dtype=int)
    a[0,0] = 1.
    ok = True
    for i in range(6):
        if verbose:
            print("is_valid_action", mdp.is_valid_action(v, a))
        if mdp.is_valid_action(v, a) == False:
            ok = False
        v, pen = mdp.transition(v, a)
        if verbose:
            print(mdp.board_str(v)[0])
            print("reward", pen)
            print("")
        if pen[0,0].item() != 0 and pen[0,1].item != 0:
            ok = False

    if verbose:
        print("Now player 0 will attempt to play in column 0, then column 1.  Then player 1 will attempt to play in column 0.")
        print("is_valid_action", mdp.is_valid_action(v, a))
    if mdp.is_valid_action(v, a).item() == True:
        ok = False
    v, pen = mdp.transition(v, a)
    if verbose:
        print(mdp.board_str(v)[0])
        print("reward", pen)
        print("")
    if pen[0,0].item() >= 0 and pen[0,1].item() != 0:
        ok = False

    b = torch.zeros((1,7), dtype=int)
    b[0,1] = 1
    if verbose:
        print("is_valid_action", mdp.is_valid_action(v, b))
    if mdp.is_valid_action(v, b).item() == False:
        ok = False
    v, pen = mdp.transition(v, b)
    if verbose:
        print(mdp.board_str(v)[0])
        print("reward", pen)
        print("")
    if pen[0,0].item() != 0 and pen[0,1].item != 0:
        ok = False
    
    if verbose:
        print("is_valid_action", mdp.is_valid_action(v, a))
    if mdp.is_valid_action(v, a).item() == True:
        ok = False
    v, pen = mdp.transition(v, a)
    if verbose:
        print(mdp.board_str(v)[0])
        print("reward", pen)
    if pen[0,0].item() != 0 and pen[0,1].item >= 0:
        ok = False

    if ok:
        print("PASS")
    else:
        print("FAIL!!!")

    print("\nChecking win condition and reward, and terminal state.")
    if verbose:
        print("Player 0 and 1 will play in columns 0 and 1.  The game should enter a terminal state after play 7, and play 8 should not proceed.")
    v = mdp.get_initial_state()
    a = torch.tensor([[1,0,0,0,0,0,0]], dtype=int)
    b = torch.tensor([[0,1,0,0,0,0,0]], dtype=int)
    ok = True
    for i in range(4):
        v, r = mdp.transition(v, a)
        if verbose:
            print(mdp.board_str(v)[0])
            print(f"reward {r}, terminal {mdp.is_terminal(v)}")
        if i == 3 and (mdp.is_terminal(v).item() == False or r[0,0].item() != 1 or r[0,1].item() != -1):
            ok = False
        v, r = mdp.transition(v, b)
        if verbose:
            print(mdp.board_str(v)[0])
            print(f"reward {r}, terminal {mdp.is_terminal(v)}")
            print("")
        if i == 3 and (mdp.is_terminal(v).item() == False or r[0,0].item() != 0 or r[0,1].item() != 0):
            ok = False
    
    if ok:
        print("PASS")
    else:
        print("FAIL!!!")
        
        

    print("\nChecking full board condition and reward.  Also, checks valid_action_filter()")
    ok = True
    if verbose:
        print("Should enter a terminal state with no reward.")
    v = mdp.get_initial_state()
    for i in range(3):
        a = mdp.int_to_action(i)
        for j in range(6):
            v, r = mdp.transition(v, a)
            if verbose:
                print(mdp.board_str(v)[0])
                print(f"reward {r}, terminal {mdp.is_terminal(v)}")
                print("")
    if verbose:
        print("Now avoiding a win...")
    a = mdp.int_to_action(6)
    v, r = mdp.transition(v, a)
    if verbose:
        print(mdp.board_str(v)[0])
        print(f"reward {r}, terminal {mdp.is_terminal(v)}")
        print("")
    for i in range(3, 7):
        a = mdp.int_to_action(i)
        for j in range(6):
            v, r = mdp.transition(v, a)
            if verbose:
                print(mdp.board_str(v)[0])
                print(f"reward {r}, terminal {mdp.is_terminal(v)}")
                print("")
            x = mdp.get_random_action(v)
            if (i, j) == (6, 1) and torch.prod(x == mdp.int_to_action(6)).item() != 1:
                print(f"{x.tolist()} is not an allowable action.  Only {mdp.int_to_action(6).tolist()} is.")
                ok = False
            if (i, j) == (6, 4) and (mdp.is_full(v).item() != True or mdp.is_terminal(v).item() != True or r[0,0].item() != 0 or r[0,1].item() != 0):
                ok = False
            elif (i, j) == (6, 5) and (mdp.is_full(v).item() != True or mdp.is_terminal(v).item() != True or r[0,0].item() != 0 or r[0, 1].item() != 0):
                ok = False
    if verbose:
        print("\nBoard should be full and in a terminal state in the second to last play.  The last play should have proceeded with no penalty.")

    if ok:
        print("PASS")
    else:
        print("FAIL!!!")

    print("\nThe following should be a full board with no winner.")
    print(mdp.board_str(v)[0])




    print("\nTesting get_random_action() and NNQFunction.get().")
    q = NNQFunction(mdp, C4NN, torch.nn.HuberLoss(), torch.optim.SGD)
    s = torch.zeros((64, 2, 6, 7)).float()
    a = mdp.get_random_action(s)
    
    if verbose:
        print("State s, action a, and Q(s,a):")
        print(s, a)
        print(q.get(s, a))
    if q.get(s,a).shape == (64,):
        print("PASS")
    else:
        print("FAIL!!!")

    print("\nTesting basic execution of val() and policy().")
    ok = True
    s = torch.zeros((7, 2, 6, 7)).float()
    a = torch.eye(7).float()
    if verbose:
        print("This list is q(s, a), with a ranging through all actions.")
        print(q.get(s, a))
        print("This is the value of s.  It should be the maximum in the list.")
        print(q.val(s)[0])
    if max(q.get(s, a).tolist()) != q.val(s)[0].item():
        ok = False

    if verbose:
        print("The policy should also match the above.")
        print(q.policy(s)[0])

    if q.get(s, q.policy(s)[0:1]).item() != q.val(s)[0].item():
        ok = False

    if ok:
        print("PASS")
    else:
        print("FAIL!!! (Note can sometimes fail due to rounding error.)")


    print("\nTesting update.")
    s = s
    a = a
    t, r = mdp.transition(s, a)
    d = TransitionData(s, a, t, r[:,0])

    if verbose:
        print("Q(s,a) and Q(t,a):")
        print(q.get(s,a))
        print(q.get(t,a))

    before = q.get(s,a)
    if verbose:
        print("Before Q(s,a):")
        print(before)
    
    q.update(d, learn_rate=0.5)
    after = q.get(s,a)
    if verbose:
        print("After Q(s,a):")
        print(after)
        print("You might have noticed the values are large.") # TODO ???

    if torch.prod(before - after != torch.tensor([0])) == True:
        print("PASS.  A change happened, but maybe not the right one.")
    else:
        print("FAIL!!! No update, very unlikely.")

    q = NNQFunction(mdp, C4NN, torch.nn.MSELoss(), torch.optim.SGD)
    if "slow" in options:
        print("\nCurrent values:")
        print(q.val(t))
        print("\nRunning update 100 times and checking for convergence.")
        for i in range(0, 500):
            q.update(d, learn_rate=0.01)
            print(q.get(s,a))




    print("\nTesting if deep Q-learning algorithm throws errors.")
    dqn = DQN(mdp, C4NN, torch.nn.HuberLoss(), torch.optim.SGD, 1000)
    dqn.set_greed(0.5)
    dqn.deep_learn(0.5, 10, 100, 1, 4, 10, verbose=verbose)
    print("PASS.  Things ran to completion, at least.")

    if "saveload" in options:
        print("\nTesting saving and loading.")
        
        dqn.save_q("temp.pt")
        new_dqn = DQN(mdp, C4NN, torch.nn.HuberLoss(), torch.optim.SGD, 1000)
        new_dqn.load_q("temp.pt")
        os.remove("temp.pt")

        if verbose:
            print("Old:")
            print(dqn.qs[0].get(s, a))
            print(list(dqn.qs[0].q.parameters())[1])
            print("Saved and loaded:")
            print(new_dqn.qs[0].get(s,a))
            print(list(new_dqn.qs[0].q.parameters())[1])
        if torch.sum(dqn.qs[0].get(s, a) - new_dqn.qs[0].get(s, a)) == 0:
            print("PASS")
        else:
            print("FAIL!!!")

    

