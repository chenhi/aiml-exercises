import connectfour as c4
import tictactoe as ttt
import connectfour_tensor as c4t
import tictactoe_tensor as dttt
from qlearn import *
from deepqlearn import *
import os, datetime, re, sys, torch

open_str = '\nCommand-line options: python play.py <game> <play/train/simulate>\n\
    If play (default), play against a model.\n\
    If train, continue to play after training.\n\
    If simulate, watch two bots in a model play each other.\n\n'


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if len(sys.argv) == 1:
    print(open_str)

#==================== GAME DEFINITION AND SELECTION ====================#

tttmdp = ttt.TTTMDP()
c4mdp = c4.C4MDP()
c4tmdp = c4t.C4TensorMDP()
dtttmdp = dttt.TTTTensorMDP(device=device)

names = ["Tic-Tac-Toe", "Connect Four", "Deep Tic-Tac-Toe", "Deep Connect Four"]
shortnames = ["ttt", "c4", "dttt", "dc4"]
mdps = [tttmdp, c4mdp, dtttmdp, c4tmdp]
games = [QLearn(tttmdp), QLearn(c4mdp), DQN(dtttmdp, dttt.TTTNN, torch.nn.HuberLoss(), torch.optim.Adam, 100000, device=device), DQN(c4tmdp, c4t.C4NN, torch.nn.HuberLoss(), torch.optim.Adam, 1000000, device=device)]
file_exts = ['.ttt.pkl', '.c4.pkl', '.dttt.pt', '.dc4.pt']
types = ["qlearn", "qlearn", "dqn", "dqn"]


# If the game was specified, choose it
if len(sys.argv) > 1:
    for i in range(len(shortnames)):
        if sys.argv[1] == shortnames[i]:
            res = i
else:
    print("Games:")
    for i in range(len(names)):
        print(f"[{i}] {names[i]} ({shortnames[i]})")
    res = input("\nSelect game from list (input number): ")

try:
    game_index = int(res)
    name = names[game_index]
    mdp = mdps[game_index]
    game = games[game_index]
    file_ext = file_exts[game_index]
    shortname = shortnames[game_index]
    type = types[game_index]
except:
    games_str = ""
    for i in range(len(names)):
        games_str += names[i] + ' (' + shortnames[i] + '), '
    games_str = games_str[:-2]
    print(f"Didn't recognize that game.  The games are {games_str}.  Exiting.")
    exit()


print(f"\nPlaying {name}.\n")

#==================== BOT TRAINING ====================#

debug = True if 'debug' in sys.argv else False

train_new = False
simulate = False
if len(sys.argv) > 2:
    if sys.argv[2] == "train":
        train_new = True
        simulate = False
    elif sys.argv[2] == "play":
        train_new = False
        simulate = False
    elif sys.argv[2] == "simulate":
        train_new = False
        simulate = True
    else:
        res = input("Enter 't' to train a new bot, or anything else to skip. ")
        train_new = True if res == 't' else False
        simulate = False


prompts = {
    'lr': 'Learn rate in [0, 1]',
    'expl': 'The ungreed/exploration rate in [0, 1]',
    'expl_start': 'The starting exploration rate in [0, 1]',
    'expl_end': 'The ending exporation rate in [0, 1]',
    'anneal_eps': 'Number of initial episodes to anneal exploration', 
    'dq_episodes': 'The number of episodes (iterations of the game)',
    'q_episodes': 'The number of episodes (experience generations)',
    'episode_length': 'The length of each episode',
    'iterations': 'The number of training iterations',
    'sim_batch': 'The size of each simulation batch',
    'train_batch': 'The size of each training batch',
    'copy_interval_eps': 'Interval between copying policy to target in episodes',
}

# Train AI
if train_new:
    
    hpar = mdp.default_hyperparameters    
    
    res = input("Name of file (alphanumeric only, max length 64, w/o extension): ")
    fname_end = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_" + re.sub(r'\W+', '', res)[0:64] + f"{file_ext}"
    fname = f'bots/{shortname}/' + fname_end
    logpath = fname + ".log"
        
    
    for k, v in hpar.items():
        res = input(f"{prompts[k] if k in prompts else k} (default {v}): ")
        try:
            hpar[k] = float(res)
        except:
            print(f"Not a valid value.  Setting greed to {v}.")

    if type == "qlearn":
        #game.set_greed(expl)
        game.batch_learn(**hpar, verbose=True, savefile=fname + ".exp")
        game.save_q(fname)

    if type == "dqn":
        game.deep_learn(**hpar, savelog=logpath, verbose=True, debug=debug)
        game.save_q(fname)



#==================== BOT SELECTION ====================#

save_files = [each for each in os.listdir(f'bots/{shortname}/') if each.endswith(file_ext)]
save_files.sort()
saves = ['RANDOMBOT'] + save_files
load_indices = []
load_index = -1

if train_new:
    load_index = saves.index(fname_end)
    print(f"Loaded the newly trained bot: {fname_end}")
else:
    savestr = ""
    for i in range(len(saves)):
        savestr += f"[{i}] {saves[i]}\n"
    res = input(f"\nThere are some saved bots:\n{savestr}\n\nIf you want to load them, enter the number.  Otherwise, enter anything else.\n")
    try:
        i = int(res)
        if i >= 0 and i < len(saves):
            load_index = int(res)        
            print(f"Loaded {saves[int(res)]}\n")
    except:
        print("No bot loaded.")


# Load the AI; special case is index = 0 which is random AI
if load_index == 0:
    game.null_q()
elif load_index != -1:
    game.load_q(f'bots/{shortname}/' + saves[load_index])




#==================== PLAY THE GAME ====================#

if simulate:
    game.stepthru_game()
    exit()


# TODO Load two computers vs two humans?

bot_list = [False for i in range(mdp.num_players)]

# Play the AI
# Assumptions: all AI play from a single model, and only one human player.
def item(obj, mdp: MDP, is_list=False):
    if mdp.batched:
        if is_list:
            return obj[0]
        elif torch.numel(obj) == 1:
            return obj[0].item()
        else:
            return obj[0].tolist()
    return obj

while True:
    if load_index >= 0:
        res = input(f"Which player to play as?  An integer from 1 to {mdp.num_players}, or 'q' to quit. ")
        if res == 'q':
            exit()
        try:
            player_index = int(res) - 1
            if player_index < 0 or player_index >= mdp.num_players:
                raise Exception()
        except:
            print("Unrecognized response.")
            continue
        
    
    s = game.mdp.get_initial_state()
    while item(game.mdp.is_terminal(s), mdp) == False:
        p = int(item(game.mdp.get_player(s), mdp))
        print(f"\n{item(game.mdp.board_str(s), mdp, is_list=True)}")
        if p == player_index:
            res = input(mdp.input_str)
            a = mdp.str_to_action(res)
            if a == None:
                print("Did not understand input.")
                continue
            s, r = game.mdp.transition(s,a)
        else:
            print("Action values:")
            print(item(game.qs[p].get(s, None), mdp))
            a = game.qs[p].policy(s)
            print(f"Chosen action: \n{item(a, mdp)}.\n")
            if mdp.is_valid_action(s, a):
                s, r = game.mdp.transition(s, a)
            else:
                print("Bot tried to make an illegal move.  Playing randomly.")
                a = game.mdp.get_random_action(s)
                print(f"Randomly chosen action: \n{item(a, mdp)}.\n")
                s, r = game.mdp.transition(s, a)
    if item(r, mdp)[p] == 1.:
        winnerstr = f"Player {p + 1} ({game.mdp.symb[p]}), {'a person' if p == player_index else 'a bot'}, won."
    elif item(r, mdp)[p] == 0.:
        winnerstr = 'The game is a tie.'
    else:
        winnerstr = "Somehow I'm not sure who won."
    
    print(f"\n{item(game.mdp.board_str(s), mdp, is_list=True)}\n\n{winnerstr}\n\n")
