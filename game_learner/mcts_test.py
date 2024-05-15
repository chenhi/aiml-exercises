from mcts import DMCTS
from tictactoe_tensor import TTTTensorMDP, TTTNN, TTTResNN
from connectfour_tensor import C4TensorMDP, C4ResNN
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


mdp = TTTTensorMDP(penalty=-0.1, device=device)
game = DMCTS(mdp, TTTResNN, torch.nn.CrossEntropyLoss(), torch.optim.Adam, device=device)

# s = mdp.get_initial_state(2)
# print(game.search(s, 100, ucb_parameter=1.)) # TODO bug, always selecting the same...

# print(game.search(s, 100, ucb_parameter=1.)) # TODO bug, always selecting the same...

# print(game.q.n[mdp.state_to_hashable(mdp.get_initial_state())])
# print(game.q.w[mdp.state_to_hashable(mdp.get_initial_state())])


game.mcts(lr = 0.001, num_iterations=50, num_selfplay=30, num_searches=20, max_steps=100, ucb_parameter=10, temperature=1, train_batch=8, train_iterations=5, save_path="ttt/bots/30sp.50its.ttt.mcts")

#game.q.load("ttt/bots/50its.ttt.mcts")
game.simulate_against_random(1000, replay_loss=False)
game.play([1])


# TODO regularize using weight_decay=lr/10?

#mdp = C4TensorMDP(device=device)
#game = DMCTS(mdp, C4ResNN, torch.nn.CrossEntropyLoss(), torch.optim.Adam, device=device)

#game.q.load("c4/bots/10its.c4.mcts")

#game.mcts(lr = 0.01, num_iterations=10, num_selfplay=10, num_searches=10, max_steps=100, ucb_parameter=10, temperature=1, train_batch=8, train_iterations=5, save_path="c4/bots/10moreits.c4.mcts")

#game.q.load("c4/bots/10its.c4.mcts")

#game.stepthru_game(verbose=True)


#game.simulate_against_random(1000)




#b = 10



#s = game.mdp.get_initial_state()


# print(torch.softmax(game.pv(s).flatten(1, -1), dim=1))
# game.search(s, game.pv, 100, ucb_parameter = 10, temperature = 1)
# print("Q", game.q[mdp.state_to_hashable(s)])
# print("N", game.n[mdp.state_to_hashable(s)])
# print("W", game.w[mdp.state_to_hashable(s)])
# print("P", game.p[mdp.state_to_hashable(s)])






#t, _ = mdp.transition(s, mdp.str_to_action("2,2"))
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

