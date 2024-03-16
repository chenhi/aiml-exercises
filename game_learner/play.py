import connectfour as c4
import tictactoe as ttt
import connectfour_tensor as c4t
import tictactoe_tensor as dttt
from qlearn import *
from deepqlearn import *
import os, datetime, re, sys, torch

open_str = '\nCommand-line options: python play.py <game> <play/train/simulate>\n\
    If play (default), only play against one model.\n\
    If train, continue to play after training.\n\
    If simulate, allow choosing different models.\n\n'


if len(sys.argv) == 1:
    print(open_str)

#==================== GAME DEFINITION AND SELECTION ====================#

tttmdp = ttt.TTTMDP()
c4mdp = c4.C4MDP()
c4tmdp = c4t.C4TensorMDP()
dtttmdp = dttt.TTTTensorMDP()

names = ["Tic-Tac-Toe", "Connect Four", "Deep Tic-Tac-Toe", "Deep Connect Four"]
shortnames = ["ttt", "c4", "dttt", "dc4"]
mdps = [tttmdp, c4mdp, dtttmdp, c4tmdp]
games = [QLearn(tttmdp), QLearn(c4mdp), DQN(dtttmdp, dttt.TTTNN, torch.nn.HuberLoss(), torch.optim.SGD, 100000), DQN(c4tmdp, c4t.C4NN, torch.nn.HuberLoss(), torch.optim.SGD, 100000)]
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
    print("Don't know what you mean.  Exiting.")
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
        fname_end = re.sub(r'\W+', '', res)[0:64] + f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}"
        fname = 'bots/' + fname_end
        
        game.set_greed(expl)
        game.batch_learn(lr, its, eps, 1000, verbose=True, savefile=fname + ".exp")

        # Save the AI
        game.save_q(fname)

    elif type == "dqn":
        game.set_greed(0.5)
        logtext = game.deep_learn(learn_rate=0.01, episodes=10000, episode_length=20, batch_size=16, train_batch_size=64, copy_frequency=5, verbose=True)
        res = "tempname"

        fname_end = re.sub(r'\W+', '', res)[0:64] + f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}"
        fname = 'bots/' + fname_end
        game.save_q(fname)
        logpath = fname + ".log"
        with open(logpath, "w") as f:
            logtext += log(f"Saved logs to {logpath}")
            f.write(logtext)



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


#TODO zeroing out doesn't help
# Load the AI; special case is index = 0 which is random AI
if load_index == 0:
    #game.zero_out()
    pass
elif load_index != -1:
    game.load_q('bots/' + saves[load_index])



# if simulate:
#     # Get the subset of bots we want to play against.
#     while True:
#         if len(saves) == len(load_indices):
#             print("All bots already loaded.\n")
#             break
#         savestr = ""
#         loadstr = ""
#         for i in range(len(saves)):
#             if i not in load_indices:
#                 savestr += f"[{i}] {saves[i]}\n"
#             else:
#                 loadstr += f"{saves[i]}, "
#         res = input(f"Already loaded: {loadstr[0:-2]}\n\nThere are some saved bots:\n{savestr}\n\nIf you want to load them, enter the number.  Otherwise, enter anything else.\n")
#         try:
#             i = int(res)
#             if i < 0 or i >= len(saves):
#                 break
#             if i in load_indices:
#                 print("Was already loaded.")
#                 continue
#             load_indices.append(int(res))
#             print(f"Loaded {saves[int(res)]}\n")
#         except:
#             break

# # If only one bot loaded, we don't need to ask which turn it plays.
# if len(load_indices) == 1:
#     # Load the AI; special case is index = 0 which is random AI
#     if load_indices[0] > 0:
#         game.load_q('bots/' + saves[load_indices[0]])


# TODO individual loading




#==================== PLAY THE GAME ====================#

# Simulation:
if simulate:
    # if len(sys.argv) >= 4:
    #     res = sys.argv[3]
    # else:
    #     res = input(f"How many simulations?  Enter -1 for step-through mode. ")
    # try:
    #     num_sim = int(res)
    #     if num_sim < 0:
    #         raise Exception()
    # except:
    #     print("Couldn't interpret the passed argument for number of simulations.  Defaulting to step-through simulations.")
    #     num_sim = -1

    # if num_sim > 0:
    #     game.simulate_game(num_sim, 1000)
    #     exit()
    # elif num_sim < 0:
    #while True:
    game.stepthru_game()
    #res = input("\nAnother round? (y/n) ")
    #    if res == 'n':
    input("Any input to exit. ") 
    exit()



bot_list = [False for i in range(mdp.num_players)]

# Play the AI
# Assumptions: all AI play from a single model, and only one human player.
# TODO: more flexible play formats via command
while True:
    break
    if load_index >= 0:
        res = input(f"Which player to play as?  An integer from 1 to {mdp.num_players}, or 'q' to quit. ")
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
            res = input(mdp.input_str)
            a = mdp.str_to_action(res)
            if a == None:
                print("Did not understand input.")
                continue
            s, r = game.mdp.transition(s,a)
        else:
            print("Action values:")
            print(game.qs[comp].q(s.float())[0].tolist())
            a = game.qs[comp].policy(s.float())
            print(f"Chosen action: {a}.\n")
            t, r = game.mdp.transition(s, a)
            while torch.sum(t).item() == torch.sum(s).item():
                print("Bot tried to make an illegal move.  Playing randomly.")
                a = game.mdp.get_random_action(s)
                t, r = game.mdp.transition(s, a)
            s = t
        if r[0,p].item() == 1.:
            print(game.mdp.board_str(s)[0])
            print(f"\nThe winner is player {p} ({game.mdp.symb[p]}).\n")
            


while True:
    # for i in range(mdp.num_players):
    #     res = input(f"Enter 'b' to make bot player {i+1} ({mdp.symb[i]}) and 'p' for player. ")
    #     if res == 'b':
    #         bot_list[i] = True

    if shortname == "ttt":
        while True:
            s = game.mdp.get_initial_state()
            while game.mdp.is_terminal(s) == False:
                p = s[0]
                print('\n' + game.mdp.board_str(s))
                if not bot_list[p]:
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
                        print(f"Value of action {a} is {game.qs[p].get(s, a)}.")
                    a = game.qs[p].policy(s)
                    print(f"Chosen action: {a}.\n")
                    s, _ = game.mdp.transition(s, a)

            winner = game.mdp.winner(s)
            if winner == -1:
                winnerstr = 'The game is a tie.'
            else:
                winnerstr = f"Player {winner + 1} ({game.mdp.symb[winner]}), {'a computer' if bot_list[winner] else 'a person'}, won."
            print(f"{game.mdp.board_str(s)}\n{winnerstr}\n\n")
    elif shortname == "c4":
        while True:
            if len(load_indices) > 0:
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
                

            print(f"{game.mdp.board_str(s)}The winner is {game.mdp.symb[str(s[2])]}.\n\n")
    elif shortname == "dc4":
        while True:
            if load_index >= 0:
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
                    res = input(f"Input column to play (1-7). ")
                    s, r = game.mdp.transition(s, game.mdp.action_index_to_tensor(int(res) - 1))
                else:
                    print("Action values:")
                    print(game.qs[comp].q(s.float()).tolist())
                    a = game.qs[comp].policy(s.float())
                    print(f"Chosen action: {a}.\n")
                    t, r = game.mdp.transition(s, a)
                    while torch.sum(t).item() == torch.sum(s).item():
                        print("Bot tried to make an illegal move.  Playing randomly.")
                        a = game.mdp.get_random_action(s)
                        t, r = game.mdp.transition(s, a)
                    s = t
                if r[0,p].item() == 1.:
                    print(game.mdp.board_str(s)[0])
                    print(f"\nThe winner is player {p} ({game.mdp.symb[p]}).\n")
    elif shortname == "dttt":
        while True:
            if load_index >= 0:
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
                    res = input(mdp.input_str)
                    a = mdp.str_to_action(res)
                    if a == None:
                        print("Did not understand input.")
                        continue
                    s, r = game.mdp.transition(s,a)
                else:
                    print("Action values:")
                    print(game.qs[comp].q(s.float())[0].tolist())
                    a = game.qs[comp].policy(s.float())
                    print(f"Chosen action: {a}.\n")
                    t, r = game.mdp.transition(s, a)
                    while torch.sum(t).item() == torch.sum(s).item():
                        print("Bot tried to make an illegal move.  Playing randomly.")
                        a = game.mdp.get_random_action(s)
                        t, r = game.mdp.transition(s, a)
                    s = t
                if r[0,p].item() == 1.:
                    print(game.mdp.board_str(s)[0])
                    print(f"\nThe winner is player {p} ({game.mdp.symb[p]}).\n")


