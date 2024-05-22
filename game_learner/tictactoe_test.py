from qlearn import *
from deepqlearn import *
import torch, sys, random
from tictactoe_tensor import *

options = sys.argv[1:]


print("RUNNING TESTS:\n")

verbose = True if "verbose" in options else False

mdp = TTTTensorMDP()
s = mdp.get_initial_state(2)
print(mdp.board_str(s)[0])
print(mdp.board_str(s)[1])

print("Fixed action to check:")
b = mdp.get_random_action(s)
print(b)

for i in range(12):
    print("play")
    a = mdp.get_random_action(s)
    print(a.tolist())
    t, r = mdp.transition(s, a)
    print(mdp.board_str(t)[0])
    print(mdp.board_str(t)[1])
    print(r.tolist())
    s = t
    print("is the original action still valid")
    print(mdp.is_valid_action(s, b))
