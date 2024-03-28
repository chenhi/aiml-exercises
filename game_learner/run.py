import connectfour as c4
import tictactoe as ttt
import connectfour_tensor as c4t
import tictactoe_tensor as dttt
import gohome as gh
from qlearn import *
from deepqlearn import *
import os, datetime, re, sys, torch

open_str = '\nCommand-line options: python play.py <game> <play/train/simulate/benchmark/tournament>\n\
    If play (default), play against a model.\n\
    If train, continue to play after training.\n\
    If simulate, watch two bots in a model play each other.\n\
    If benchmark, simulates games between a bot and a random bot.\n\
    If tournament, simulates a tournament between all trained bots.\n\n'


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if len(sys.argv) == 1:
    print(open_str)

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

#==================== GAMEMODE LOGIC ====================#

mode = "play"

if len(sys.argv) > 2:
    mode = sys.argv[2]

#==================== GAME DEFINITION AND SELECTION ====================#

tttmdp = ttt.TTTMDP()
c4mdp = c4.C4MDP()
c4tmdp = c4t.C4TensorMDP()
dtttmdp = dttt.TTTTensorMDP(device=device)
ghmdp = gh.GoHomeMDP((6,6), (0,0), (3,3), 0.9)

names = ["Tic-Tac-Toe", "Connect Four", "Deep Tic-Tac-Toe", "Deep Connect Four", "Robot Go Home"]
shortnames = ["ttt", "c4", "dttt", "dc4", "home"]
mdps = [tttmdp, c4mdp, dtttmdp, c4tmdp, ghmdp]
games = [QLearn(tttmdp), QLearn(c4mdp), DQN(dtttmdp, dttt.TTTNN, torch.nn.HuberLoss(), torch.optim.Adam, 100000, device=device), DQN(c4tmdp, c4t.C4NN, torch.nn.HuberLoss(), torch.optim.Adam, 1000000, device=device), QLearn(ghmdp)]
file_exts = ['.ttt.pkl', '.c4.pkl', '.dttt.pt', '.dc4.pt', '.home.pkl']
types = ["qlearn", "qlearn", "dqn", "dqn", 'qlearn']


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

#==================== LOAD SAVES ====================#


save_files = [each for each in os.listdir(f'bots/{shortname}/') if each.endswith(file_ext)]
save_files.sort()


#==================== BOT TRAINING ====================#

debug = True if 'debug' in sys.argv else False


prompts = {
    'lr': 'Learn rate in [0, 1]',
    'expl': 'The ungreed/exploration rate in [0, 1]',
    'greed_start': 'The starting greed in [0, 1] (0 means random)',
    'greed_end': 'The ending greed in [0, 1] (0 means random)',
    'ramp_start': 'Which episode to start ramping up greed',
    'ramp_end': 'Which episode to end greed ramp', 
    'dq_episodes': 'The number of episodes (iterations of the game)',
    'q_episodes': 'The number of episodes (experience generations)',
    'episode_length': 'The length of each episode',
    'iterations': 'The number of training iterations',
    'sim_batch': 'The size of each simulation batch',
    'train_batch': 'The size of each training batch',
    'copy_interval_eps': 'Interval between copying policy to target in episodes',
    'training_delay': 'What episode to start training'
}

# Train AI

if mode == "train":

    res = input("Load existing model (Y/n)? ")
    if res.lower() == 'y':
        saves = save_files
        load_indices = []
        savestr = ""
        for i in range(len(saves)):
            savestr += f"[{i}] {saves[i]}\n"
        print(f"\nSaved bots:\n{savestr}\n")
        res = input(f"Select initial bot: ")
        try:
            res = int(res)
            if res >= 0 and res < len(saves):
                game.load_q(f'bots/{shortname}/' + saves[res])
                print(f"Loaded {saves[int(res)]}\n")
        except:
            print("Didn't understand.")

    
    hpar = mdp.default_hyperparameters

    res = input("Name of file (alphanumeric only, max length 64, w/o extension): ")
    fname_end = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_" + re.sub(r'\W+', '', res)[0:64] + f"{file_ext}"
    fname = f'bots/{shortname}/' + fname_end

        
    
    for k, v in hpar.items():
        res = input(f"{prompts[k] if k in prompts else k} (default {v}): ")
        try:
            hpar[k] = float(res)
        except:
            print(f"Not a valid value.  Setting {k} to {v}.")

    if type == "qlearn":
        #game.set_greed(expl)
        game.batch_learn(**hpar, verbose=True, savefile=fname + ".exp")
        game.save_q(fname)

    if type == "dqn":
        game.deep_learn(**hpar, verbose=True, debug=debug, save_path=fname)



#==================== BOT TOURNAMENT OR SIMULATION ====================#
    
if mode == "tournament":
    saves = save_files
    
    logtext = "Bot Tournament\n\nList of bots:\n"
    for i in range(len(saves)):
        logtext += log(f"[{i}] {saves[i]}")
    

    # Generate all n-tuples of k numbers
    k = len(saves)
    n = mdp.num_players
    matches = []
    for i in range(k**n):
        total = i
        modulus = total % k
        match = (modulus)
        for i in range(n):
            modulus = total % k
            match = match + (modulus, ) if i > 0 else (modulus, )
            total = (total - modulus)//k
        matches.append(match)

    # Bot records
    records = [{'win': 0, 'loss': 0, 'tie': 0} for i in range(k)]
    logtext += log("")

    for match in matches:
        for i in range(n):
            game.load_q(f'bots/{shortname}/' + saves[match[i]], [i])
        r = game.simulate()
        logtext += log(f"Result of match {match}: {r[0].int().tolist()}")
        for i in range(n):
            if r[0,i].item() > 0:
                records[match[i]]['win'] += 1
            elif r[0,i].item() < 0:
                records[match[i]]['loss'] += 1
            else:
                records[match[i]]['tie'] += 1
    
    logtext += log("\n\nFinal standings:")
    # First log the raw standings
    for i in range(k):
        logtext += log(f"{saves[i]}: {records[i]}")
    
    # Sort by wins, loss, and score = win - loss
    indices = list(range(k))
    by_win = sorted(indices, key=lambda t: records[t]['win'], reverse=True)
    by_loss = sorted(indices, key=lambda t: records[t]['loss'])
    by_score = sorted(indices, key=lambda t: records[t]['loss'] - records[t]['win'])
    
    logtext += log("\nBy wins:")
    for i in range(k):
        logtext += log(f"{saves[by_win[i]]}: {records[by_win[i]]}")
    
    logtext += log("\nBy losses:")
    for i in range(k):
        logtext += log(f"{saves[by_loss[i]]}: {records[by_loss[i]]}")

    logtext += log("\nBy score:")
    for i in range(k):
        logtext += log(f"{saves[by_score[i]]}: {records[by_score[i]]}")

    with open(f"logs/tournament.{shortname}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log", "w") as f:
        f.write(logtext)
    
    exit()


if mode == "simulate":
    saves = save_files
    load_indices = []
    savestr = ""
    for i in range(len(saves)):
        savestr += f"[{i}] {saves[i]}\n"
    print(f"Saved bots:\n{savestr}\n\n")
    for i in range(mdp.num_players):
        res = input(f"Bot for player {i+1}/{mdp.num_players}: ")
        try:
            res = int(res)
            if res >= 0 and res < len(saves):
                game.load_q(f'bots/{shortname}/' + saves[res], [i])
                print(f"Loaded {saves[int(res)]}\n")
        except:
            print("Didn't understand.")
            i -= 1

    game.stepthru_game()
    exit()



#==================== BOT SELECTION ====================#


# Add random bot
saves = ['RANDOMBOT'] + save_files
load_index = -1

if mode == "play":
    savestr = ""
    for i in range(len(saves)):
        savestr += f"[{i}] {saves[i]}\n"
    res = input(f"\nThere are some saved bots:\n{savestr}\n\nIf you want to load them, enter either a number.  Otherwise, enter anything else: ")
    try:
        res = int(res)
        if res >= 0 and res < len(saves):
            load_index = res        
            print(f"Loaded {saves[res]}\n")
    except:
        print("No bot loaded.")

    # Load the AI; special case is index = 0 which is random AI
    if load_index == 0:
        game.null_q()
    elif load_index != -1:
        game.load_q(f'bots/{shortname}/' + saves[load_index])


    
#==================== BENCHMARK AGAINST RANDOM ====================#

if mode == "benchmark":

    do_all = False
    savestr = ""
    for i in range(len(saves)):
        savestr += f"[{i}] {saves[i]}\n"
    res = input(f"\nSaved bots:\n{savestr}\n\nEnter number to load or anything else to benchmark them all: ")
    try:
        load_index = int(res)
        if load_index == 0:
            game.null_q()
        elif load_index > 0:
            game.load_q(f'bots/{shortname}/' + saves[load_index])
        else:
            raise Exception
        print(f"Loaded {saves[load_index]}\n")
    except:
        print(f"Benchmarking all.")
        do_all = True

    # Load the AI; special case is index = 0 which is random AI


    sims = int(input("How many iterations against random bot? "))
    if do_all == False:
        replay = True if input("Enter 'y' to replay losses: ").lower() == 'y' else False
        game.simulate_against_random(sims, replay_loss=replay, verbose=True)
    else:
        logtext = ""
        logtext += log("Simulating {name} random bot for {sims} simulations.\n\n")
        for i in range(1, len(saves)):
            game.load_q(f'bots/{shortname}/' + saves[i])
            result = game.simulate_against_random(sims, replay_loss=False, verbose=False)
            logtext += log(f"\n")
            for j in range(len(result)):
                logtext += log(f"Bot {saves[i]} as player {j}: {result[j][0]} wins, {result[j][1]} losses, {result[j][2]} ties, {result[j][3]} invalid moves, {result[j][4]} unknown results.\n")

        with open(f"logs/benchmark.{shortname}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log", "w") as f:
            f.write(logtext)

    exit()
    




#==================== PLAY THE GAME ====================#

while True:
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
