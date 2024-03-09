import tictactoe
from interface import *

ttt = tictactoe.TTTMDP()

# Train AI
game = SimpleGame(ttt, 2)
game.set_greed([0.8, 0.8])
game.batch_learn(0.5, 1000, 10, 1000)

# Play the AI
while True:
    s = game.mdp.get_initial_state()
    while game.mdp.is_terminal(s) == False:
        p, arr = game.mdp.state_to_array(s)
        print(game.mdp.board_str(s))
        if p == 1:
            re = input(f"Input position to play e.g. 1,3 for row 1, column 3. ")
            res = re.split(",")
            x = int(res[0].strip())
            y = int(res[1].strip())
            s, _ = game.mdp.transition(s, (x-1,y-1))
        else:
            for a in game.mdp.actions:
                print(f"Value of action {a} is {game.qs[1].get(s, a)}.")
            a = game.qs[1].policy(s)
            print(f"Chosen action: {a}.\n")
            s, _ = game.mdp.transition(s, a)

    print(f"{game.mdp.board_str(s)}The winner is {game.mdp.symb[game.mdp.winner(s)]}.\n\n")
    res = input("Keep playing?  y/n ")
    if res.lower() == "n":
        exit()
    


