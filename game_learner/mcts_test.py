from mcts import DMCTS
from tictactoe_tensor import TTTTensorMDP, TTTNN, TTTResNN
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


mdp = TTTTensorMDP(device=device)
game = DMCTS(mdp, TTTResNN, torch.nn.CrossEntropyLoss(), torch.optim.Adam, device=device)


game.mcts(lr = 0.01, num_iterations=50, num_selfplay=10, num_searches=100, max_steps=100, ucb_parameter=2, temperature=1, train_batch=8, p_threshold_multiplier=10)




s = game.mdp.get_initial_state()





#game.search(s, 0, game.pv, 5000)
#print("Q1", game.q[mdp.state_to_hashable(s)])
#print("N", game.n[mdp.state_to_hashable(s)])
#print("W", game.w[mdp.state_to_hashable(s)])
#print("P", game.p[mdp.state_to_hashable(s)])
t, _ = mdp.transition(s, mdp.str_to_action("2,2"))
#game.search(t, 0, game.pv, 4000)
#print("Q2", game.q[mdp.state_to_hashable(t)])
#print("N", game.n[mdp.state_to_hashable(t)])
#print("W", game.w[mdp.state_to_hashable(t)])
#print("P", game.p[mdp.state_to_hashable(t)])


u = game.mdp.get_initial_state()
u[0,0,0,0] = 1.
u[0,1,1,1] = 1.
u[0,0,2,2] = 1.
print(game.mdp.board_str(u)[0])

b = 2

# print(game.pv(u))
# print(torch.sigmoid(game.pv(u)))
# game.search(u, game.pv, 5000, ucb_parameter = b, temperature = 1)
# print("Q", game.q[mdp.state_to_hashable(u)])
# print("N", game.n[mdp.state_to_hashable(u)])
# print("W", game.w[mdp.state_to_hashable(u)])
# print("P", game.p[mdp.state_to_hashable(u)])
# print("ucb", game.ucb(u, b))
# print("???", (game.ucb(u, b)) * game.mdp.valid_action_filter(u))



print("Q", game.q[mdp.state_to_hashable(u)])
print("N", game.n[mdp.state_to_hashable(u)])
print("W", game.w[mdp.state_to_hashable(u)])
print("P", game.p[mdp.state_to_hashable(u)])