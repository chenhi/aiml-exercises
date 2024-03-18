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

# Train AI
if train_new:
    if type == "qlearn":
        default=0.5
        res = input(f"How greedy should it be?  A number in [0, 1] (default {default}): ")
        try:
            expl = 1. - float(res)
            if expl < 0 or expl > 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting greed to {default}.")
            expl = default


    if type == "dqn":
        default = 1.0
        res = input(f"Start (un-)greed?  A number in [0, 1] (default {default}): ")
        try:
            expl_start = 1. - float(res)
            if expl_start < 0 or expl_start > 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting greed to {default}.")
            expl_start = default


        default = 0.1
        res = input(f"End (un-)greed?  A number in [0, 1] (default {default}): ")
        try:
            expl_end = 1. - float(res)
            if expl_end < 0 or expl_end > 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting greed to {default}.")
            expl_end = default
        
    if type == "qlearn":
        default=0.1
    elif type == "dqn":
        default = 0.00025
    res = input(f"Learning rate? A number in [0, 1] (default {default}): ")
    try:
        lr = float(res)
        if lr < 0 or lr > 1:
            raise Exception
    except:
        print(f"Not a valid value.  Setting learning rate to {default}.")
        lr = default

    if type == "qlearn":
        default = 64
        res = input(f"How many game runs between AI updates (default {default}): ")
        try:
            eps = int(res)
            if eps < 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting episodes to {default}.")
            eps = default

        default = 100
        res = input(f"How many AI updates (default {default}): ")
        try:
            its = int(res)
            if its < 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting iterations to {default}.")
            its = default
        
    if type == "dqn":
        default = 2000
        res = input(f"How many episodes (default {default}): ")
        try:
            episodes = int(res)
            if episodes < 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting episodes to {default}.")
            episodes = default

        default = 50
        res = input(f"How many turns in an episode (default {default}): ")
        try:
            turns = int(res)
            if turns < 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting turns per episode to {default}.")
            turns = default

        default = 500
        res = input(f"Delay before training (default {default}): ")
        try:
            delay = int(res)
            if delay < 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting training delay to {default}.")
            delay = default



    res = input("Name of file (alphanumeric only, max length 64, w/o extension): ")
    fname_end = re.sub(r'\W+', '', res)[0:64] + f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}"
    fname = 'bots/' + fname_end
    logpath = fname + ".log"
        
    if type == "qlearn":
        game.set_greed(expl)
        game.batch_learn(lr, its, eps, 1000, verbose=True, savefile=fname + ".exp")

        # Save the AI
        game.save_q(fname)

    if type == "dqn":
        
        
        #game.deep_learn(learn_rate=0.0001, greed_start = 1, greed_end = 0.3, episodes=5000, episode_length=20, batch_size=4, episodes_before_train=500, train_batch_size=32, copy_frequency=250, savelog=logpath, verbose=True)
        
        # This one gave good results for tic tac toe
        #game.deep_learn(learn_rate=0.0001, greed_start =1.0, greed_end = 0.1, episodes=2000, episode_length=20, batch_size=4, episodes_before_train=200, train_batch_size=32, copy_frequency=250, savelog=logpath, verbose=True)
        
        # Trying C4
        #game.deep_learn(learn_rate=0.0001, greed_start = 1, greed_end = 0.1, episodes=2000, episode_length=50, batch_size=4, episodes_before_train=500, train_batch_size=32, copy_frequency=250, savelog=logpath, verbose=True)
        
        

        #game.deep_learn(learn_rate=0.0001, greed_start = 1, greed_end = 0.3, episodes=500, episode_length=20, batch_size=4, episodes_before_train=50, train_batch_size=32, copy_frequency=25, savelog=logpath, verbose=True)
        
        game.deep_learn(learn_rate=lr, greed_start=expl_start, greed_end=expl_end, episodes=episodes, episode_length=turns, batch_size=4, episodes_before_train=delay, train_batch_size=32, copy_frequency=250, savelog=logpath, verbose=True)
        
        game.save_q(fname)



#==================== BOT SELECTION ====================#


saves = ['RANDOMBOT'] + [each for each in os.listdir('bots/') if each.endswith(file_ext)]
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
    game.load_q('bots/' + saves[load_index])




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
        p = item(game.mdp.get_player(s), mdp)
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
