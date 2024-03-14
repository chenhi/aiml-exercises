import connectfour as c4
import tictactoe as ttt
from qlearn import *
import os, datetime, re, sys, torch


tttmdp = ttt.TTTMDP()
c4mdp = c4.C4MDP()
c4tmdp = c4.C4TensorMDP()

names = ["Tic-Tac-Toe", "Connect Four", "Tensor Connect Four"]
shortnames = ["ttt", "c4", "c4t"]
mdps = [tttmdp, c4mdp, c4tmdp]
games = [SimpleGame(tttmdp), SimpleGame(c4mdp), DQN(c4tmdp, c4.C4NN, torch.nn.HuberLoss, torch.optim.Adam, 10000)]
file_exts = ['.ttt.pkl', '.c4.pkl', '.c4t.pkl']
types = ["classical", "classical", "deep"]

option = sys.argv[1]
res = None
for i in range(len(shortnames)):
    if option == shortnames[i]:
        res = i

if res == None:
    print("Games:")
    for i in range(len(names)):
        print(f"[{i}] {names[i]}")
    res = input("Select game from list (input number): ")

try:
    game_index = int(res)
    name = names[game_index]
    mdp = mdps[game_index]
    game = games[game_index]
    file_ext = file_exts[game_index]
    shortname = shortnames[game_index]
    type = types[game_index]
except:
    print("Don't know what you mean.  Exiting.")
    exit()


print(f"\nPlaying {name}.\n")

saves = [each for each in os.listdir('bots/') if each.endswith(file_ext)]
if len(saves) == 0:
    res = input(f"There are no bots.  Enter 'n' to train a new one, or 'p' to play without bots. ")
    if res == 'n':
        saveindex = -1
    else:
        saveindex = -2
else:
    savestr = ""
    for i in range(len(saves)):
        savestr += f"[{i}] {saves[i]}\n"
    res = input(f"There are some saved bots:\n{savestr}\n\nWhich one do you want to load?  Enter number, or 'n' to train a new one, or 'p' to play without bots. ")
    try:
        saveindex = int(res)
    except:
        if res == 'n':
            saveindex = -1
        else:
            saveindex = -2


if saveindex >= 0:
    # Load the AI
    game.load_q('bots/' + saves[saveindex])
elif saveindex == -1:
    # Train AI

    if type == "clasical":
        default=0.5
        res = input(f"How greedy should it be?  A number in [0, 1] (default {default}): ")
        try:
            expl = 1. - float(res)
            if expl < 0 or expl > 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting greed to {default}.")
            expl = default

        default=0.1
        res = input(f"Learning rate? A number in [0, 1] (default {default}): ")
        try:
            lr = float(res)
            if lr < 0 or lr > 1:
                raise Exception
        except:
            print(f"Not a valid value.  Setting learning rate to {default}.")
            lr = default

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
        
        res = input("Name of file (alphanumeric only, max length 64, w/o extension): ")
        fname = 'bots/' + re.sub(r'\W+', '', res)[0:64] + f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}"
        
        game.set_greed(expl)
        game.batch_learn(lr, its, eps, 1000, verbose=True, savefile=fname + ".exp")

        # Save the AI
        game.save_q(fname)
    elif type == "deep":
        game.set_greed(0.5)
        game.memory_capacity = 10000
        game.deep_learn(learn_rate=0.1, iterations=100, episodes=100, episode_length=1000, batch_size=64, train_batch_size=64, copy_frequency=100)




# Play the AI
if shortname == "ttt":
    while True:
        if saveindex >= -1:
            res = input("Play as first (X) or second (O) player?  Enter '1' or '2' or 'q' to quit: ")
            if res == 'q':
                exit()
            if res != '1' and res != '2':
                continue
            if res == '1':
                comp = 1
            else:
                comp = 0
        else:
            comp = -1

        s = game.mdp.get_initial_state()
        while game.mdp.is_terminal(s) == False:
            p = s[0]
            print('\n' + game.mdp.board_str(s))
            if p != comp:
                re = input(f"Input position to play e.g. '1,3' for row 1, column 3. ")
                try:
                    res = re.split(",")
                    x, y = int(res[0].strip()), int(res[1].strip())
                    if (x-1,y-1) in game.mdp.get_actions(s):
                        s, _ = game.mdp.transition(s, (x-1,y-1))
                    else:
                        raise Exception()
                except:
                    print("ERROR: Invalid action, try again.")
            else:
                for a in game.mdp.get_actions(s):
                    print(f"Value of action {a} is {game.qs[comp].get(s, a)}.")
                a = game.qs[comp].policy(s)
                print(f"Chosen action: {a}.\n")
                s, _ = game.mdp.transition(s, a)

        winner = game.mdp.winner(s)
        if winner == -1:
            winnerstr = 'The game is a tie.'
        elif winner == comp:
            winnerstr = f"The computer ({game.mdp.symb[winner]}) won."
        else:
            winnerstr = f"You ({game.mdp.symb[winner]}) won."
        print(f"{game.mdp.board_str(s)}\n{winnerstr}\n\n")
elif shortname == "c4":
    while True:
        if saveindex >= -1:
            res = input("Play as first or second player?  Enter '1' or '2' or 'q' to quit: ")
            if res == 'q':
                exit()
            if res != '1' and res != '2':
                continue
            if res == '1':
                comp = 1
            else:
                comp = 0
        else:
            comp = -1

        s = game.mdp.get_initial_state()
        while game.mdp.is_terminal(s) == False:
            p = s[0]
            print(game.mdp.board_str(s))
            if p != comp:
                re = input(f"Input column to play (1-7). ")
                s, _ = game.mdp.transition(s, int(re) - 1)
            else:
                for a in game.mdp.get_actions(s):
                    print(f"Value of action {a} is {game.qs[comp].get(s, a)}.")
                a = game.qs[comp].policy(s)
                print(f"Chosen action: {a}.\n")
                s, _ = game.mdp.transition(s, a)
                print(game.mdp.state_to_tensor(s), game.mdp.action_to_tensor(a))
            

        print(f"{game.mdp.board_str(s)}The winner is {game.mdp.symb[str(s[2])]}.\n\n")
elif shortname == "c4t":
    while True:
        if saveindex >= -1:
            exit()
            res = input("Play as first or second player?  Enter '1' or '2' or 'q' to quit: ")
            if res == 'q':
                exit()
            if res != '1' and res != '2':
                continue
            if res == '1':
                comp = 1
            else:
                comp = 0
        else:
            comp = -1

        s = game.mdp.get_initial_state()
        while game.mdp.is_terminal(s).item() == False:
            p = game.mdp.get_player(s).item()
            print(game.mdp.board_str(s)[0])
            if p != comp:
                re = input(f"Input column to play (1-7). ")
                s, r = game.mdp.transition(s, game.mdp.get_single_action_vector(int(re) - 1))
                if r[0,p].item() == 1.:
                    print(f"\nThe winner is player {p} ({game.mdp.symb[p]}).\n")
            else:
                for a in game.mdp.get_actions(s):
                    print(f"Value of action {a} is {game.qs[comp].get(s, a)}.")
                a = game.qs[comp].policy(s)
                print(f"Chosen action: {a}.\n")
                s, _ = game.mdp.transition(s, a)
                print(game.mdp.state_to_tensor(s), game.mdp.action_to_tensor(a))
        


