from connectfour import C4MDP, C4ResNN
from mcts import DMCTS
import datetime, torch


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


save = f"{prefix}-vickyjwang.c4.mcts"


mdp = C4MDP(device=device)
game = DMCTS(mdp, C4ResNN, torch.nn.CrossEntropyLoss(), torch.optim.Adam, device=device)
game.mcts(lr = 0.00025, wd = 0.0025, num_iterations=32, num_episodes=32, num_selfplay=32, num_searches=32, max_steps=100, tournament_length = 64, tournament_searches=32, ucb_parameter=5, temperature_start=0.5, temperature_end = 0.1, train_batch=128, memory_size=80000, save_path=save)
