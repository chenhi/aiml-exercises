from qlearn import MDP
from deepqlearn import *
import numpy as np
import sys, warnings
import torch

from connectfour_tensor import *

# Testing options
# 'test' to test
# 'verbose' to print more messages
# 'convergence' to do a simulation of Q-updates for testing convergence
# 'dqn' to run the DQN
# 'saveload' to test saving and loading
options = sys.argv[1:]


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("\nRUNNING TESTS:\n")

# We can print everything or just the pass/fail statements.
verbose = True if "verbose" in options else False
mdp = C4TensorMDP(device=device)

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
a = torch.zeros((2,7), device=device)
a[0, 1] = 1.
a[1, 6] = 1.
b = torch.zeros((2,7), device=device)
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
a = torch.zeros((1,7), device=device)
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

b = torch.zeros((1,7), device=device)
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
a = torch.tensor([[1,0,0,0,0,0,0]], device=device).float()
b = torch.tensor([[0,1,0,0,0,0,0]], device=device).float()
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
s = torch.zeros((64, 2, 6, 7))
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
s = torch.zeros((7, 2, 6, 7))
a = torch.eye(7)
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

q = NNQFunction(mdp, C4NN, torch.nn.MSELoss(), torch.optim.Adam)
if "convergence" in options:
    #print("Should converge to:")
    #print(d.r + q.val(d.t.float()))
    lr = float(input("learn rate? rec. 0.00025 or 0.0005 or 0.001: "))
    print("\nRunning update 100 times and checking for convergence.")
    for i in range(0, 500):
        q.update(d, learn_rate=lr)
        print(q.get(s,a))
    print("Did it converge to:")
    print(d.r + q.val(d.t.float()))


if "dqn" in options:
    print("\nTesting if deep Q-learning algorithm throws errors.")
    dqn = DQN(mdp, C4NN, torch.nn.HuberLoss(), torch.optim.SGD, 1000)
    dqn.deep_learn(0.5, 0.5, 0.5, 10, 100, 1, 4, 5, 10, verbose=verbose)
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



