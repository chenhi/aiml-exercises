import tictactoe as ttt


game = ttt.TTTMDP()
s = game.get_initial_state()


# Convention: 1 is X, -1 is O

while game.is_terminal(s) == False:
    re = input(f"{game.board_str(s)}Input position to play e.g. 1,3 for row 1, column 3. ")
    res = re.split(",")
    x = int(res[0].strip())
    y = int(res[1].strip())
    s, _ = game.transition(s, (x-1,y-1))

print(f"{game.board_str(s)}The winner is {game.symb[game.winner(s)]}.")