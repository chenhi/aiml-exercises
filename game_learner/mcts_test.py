from mcts import DMCTS
from tictactoe_tensor import TTTTensorMDP, TTTNN, TTTResNN
import torch

mdp = TTTTensorMDP()
game = DMCTS(mdp, TTTResNN, torch.nn.HuberLoss(), torch.optim.Adam)

s = game.mdp.get_initial_state()
game.search(s, 1., game.pv, 5000)
print("Q1", game.q[mdp.state_to_hashable(s)])
t, _ = mdp.transition(s, mdp.str_to_action("2,2"))
game.search(t, 1., game.pv, 4000)
print("Q2", game.q[mdp.state_to_hashable(t)])
#print("N", game.n)
#print("W", game.w)
#print("P", game.p)


# TODO still sometimes get stuck in infinte loop