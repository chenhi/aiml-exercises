import tictactoe
from qlearn import *
import os, datetime, re

ttt = tictactoe.TTTMDP()
game = SimpleGame(ttt, 2)

saves = [each for each in os.listdir() if each.endswith('.pkl')]
savestr = ""
for i in range(len(saves)):
    savestr += f"[{i}] {saves[i]}\n"
res = input(f"There are some saved bots:\n{savestr}\n\nWhich one do you want to load?  Enter number or 'n' to train a new one. ")
try:
    saveindex = int(res)
except:
    saveindex = None


if saveindex != None:
    # Load the AI
    game.load_q(saves[saveindex])
else:
    # Train AI
    res = input("How greedy should it be?  A number in [0, 1]: ")
    try:
        expl = 1. - float(res)
        if expl < 0 or expl > 1:
            raise Exception
    except:
        print("Not a valid value.  Setting greed to 0.5.")
        expl = 0.5


    res = input("Learning rate? A number in [0, 1]: ")
    try:
        lr = float(res)
        if lr < 0 or lr > 1:
            raise Exception
    except:
        print("Not a valid value.  Setting learning rate to 0.5.")
        lr = 0.5

    res = input("How many game runs between AI updates? E.g. 10: ")
    try:
        eps = int(res)
        if eps < 1:
            raise Exception
    except:
        print("Not a valid value.  Setting episodes to 10.")
        eps = 10

    res = input("How many AI updates? E.g. 100: ")
    try:
        its = int(res)
        if its < 1:
            raise Exception
    except:
        print("Not a valid value.  Setting iterations to 100.")
        its = 100
    
    res = input("Name of file (alphanumeric only, max length 64, w/o extension): ")
    fname = re.sub(r'\W+', '', res)[0:64] + f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    
    game.set_greed([expl, expl])
    game.batch_learn(lr, its, eps, 1000, verbose=True)
    
    # Setting the maximum episode length to be very high is safe, since the game is very short
    # This will run the game 10 times before retraining, and do this 1000 times.  Good for an initial training
    #game.batch_learn(0.5, 1000, 10, 1000)

    # This will barely train it, mostly for testing
    #game.batch_learn(0.5, 10, 10, 1000)

    # Save the AI
    game.save_q(fname)


# Play the AI
while True:
    res = input("Play as first or second player?  Enter '1' or '2' or 'q' to quit: ")
    if res == 'q':
        exit()
    if res != '1' and res != '2':
        continue
    if res == '1':
        player, comp = 1, 1
    else:
        player, comp = -1, 0

    s = game.mdp.get_initial_state()
    while game.mdp.is_terminal(s) == False:
        p, arr = game.mdp.state_to_array(s)
        print(game.mdp.board_str(s))
        if p == player:
            re = input(f"Input position to play e.g. 1,3 for row 1, column 3. ")
            res = re.split(",")
            x = int(res[0].strip())
            y = int(res[1].strip())
            s, _ = game.mdp.transition(s, (x-1,y-1))
        else:
            for a in game.mdp.actions:
                print(f"Value of action {a} is {game.qs[comp].get(s, a)}.")
            a = game.qs[comp].policy(s)
            print(f"Chosen action: {a}.\n")
            s, _ = game.mdp.transition(s, a)

    print(f"{game.mdp.board_str(s)}The winner is {game.mdp.symb[game.mdp.winner(s)]}.\n\n")
    


