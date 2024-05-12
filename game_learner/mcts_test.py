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


# TODO regularize using weight_decay=lr/10?

mdp = TTTTensorMDP(device=device)
game = DMCTS(mdp, TTTResNN, torch.nn.CrossEntropyLoss(), torch.optim.Adam, device=device)



game.mcts(lr = 0.01, num_iterations=20, num_selfplay=10, num_searches=100, max_steps=100, ucb_parameter=10, temperature=1, train_batch=8, save_path="testmcts_save")

game.q.load("testmcts_save")

game.simulate_against_random(1000)


b = 10



s = game.mdp.get_initial_state()


# print(torch.softmax(game.pv(s).flatten(1, -1), dim=1))
# game.search(s, game.pv, 100, ucb_parameter = 10, temperature = 1)
# print("Q", game.q[mdp.state_to_hashable(s)])
# print("N", game.n[mdp.state_to_hashable(s)])
# print("W", game.w[mdp.state_to_hashable(s)])
# print("P", game.p[mdp.state_to_hashable(s)])






t, _ = mdp.transition(s, mdp.str_to_action("2,2"))
#game.search(t, 0, game.pv, 4000)
#print("Q2", game.q[mdp.state_to_hashable(t)])
#print("N", game.n[mdp.state_to_hashable(t)])
#print("W", game.w[mdp.state_to_hashable(t)])
#print("P", game.p[mdp.state_to_hashable(t)])


# u = game.mdp.get_initial_state()
# u[0,0,0,0] = 1.
# u[0,1,1,1] = 1.
# u[0,0,2,2] = 1.
# print(game.mdp.board_str(u)[0])



# print(torch.softmax(game.pv(u).flatten(1, -1), dim=1))
# game.search(u, game.pv, 5000, ucb_parameter = b, temperature = 1)
# print("Q", game.q[mdp.state_to_hashable(u)])
# print("N", game.n[mdp.state_to_hashable(u)])
# print("W", game.w[mdp.state_to_hashable(u)])
# print("P", game.p[mdp.state_to_hashable(u)])

