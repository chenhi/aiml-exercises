from mcts import DMCTS
from tictactoe_tensor import TTTTensorMDP, TTTNN, TTTResNN
import torch

mdp = TTTTensorMDP()
game = DMCTS(mdp, TTTResNN, torch.nn.HuberLoss(), torch.optim.Adam)

s = game.mdp.get_initial_state()
game.search(s, 2., game.pv, 1000)
print("Q", game.q[mdp.state_to_hashable(s)])
#print("N", game.n)
#print("W", game.w)
#print("P", game.p)


# TODO still sometimes get stuck in infinte loop