import os, datetime, re, sys, torch

from connectfour import C4MDP, C4NN, C4ResNN
from tictactoe import TTTMDP, TTTNN, TTTResNN, TTTCatNN
from nothanks import NoThanksMDP, NoThanksNN

from qlearn import *
from deepqlearn import *
from rlbase import log


# Alter for colab
base_path = ""
# base_path = "/content/gdrive/My Drive/colab/"

open_str = '\nCommand-line options: python play.py <game> <play/train/simulate/benchmark/tournament>\n\
    If play (default), play against a model, against other humans, or watch bots play each other.\n\
    If train, continue to play after training.\n\
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

# Auxillary function used to handle the fact that for MDPs we tend to get back scalars or arrays, and for TensorMDPs we tend to get back batched tensors with batch size 1
def item(obj, mdp: MDP, is_list=False):
    if mdp.batched:
        if is_list:
            return obj[0]
        elif torch.numel(obj) == 1:
            return obj[0].item()
        else:
            return obj[0].tolist()
    return obj

def load_bots(qgame, saves) -> str:
    logtext = ""
    savestr = ""
    for i in range(len(saves)):
        savestr += f"[{i}] {saves[i]}\n"
    res = input(f"\nThere are some saved bots:\n{savestr}\n\nEnter a number, a comma-separated list of numbers of length {qgame.mdp.num_players}, or empty for a bot that plays randomly: ").split(',')

    if len(res) == 1:
        try:
            index = int(res[0])
            if index >= 0 and index < len(saves):
                print(base_path + f'{shortname}/bots/dqn/' + saves[index])
                qgame.q.load(base_path + f'{shortname}/bots/dqn/' + saves[index])
                logtext += log(f"Loaded {saves[index]} for all players.")
            else:
                raise Exception
        except:
            qgame.q.null()
            logtext += log("Loaded RANDOMBOT for all players.")
    else:
        qgame.null()
        for i in range(min(len(res), game.mdp.num_players)):
            try:
                index = int(res[i])
                if index < 0 or index >= len(saves):
                    logtext += log(f"{i} is not a bot on the list.  Loading RANDOMBOT as player {i}.")
                else:
                    game.q.load(base_path + f'{shortname}/bots/dqn/' + saves[index], [i])
            except:
                logtext += log(f"Didn't understand {s}.  Loading RANDOMBOT as player {i}.")
        if len(res) > game.mdp.num_players:
            logtext += log(f"Extra bots in the list in excess of number of players ignored.")
        elif len(res) < game.mdp.num_players:
            logtext += log(f"Loaded RANDOMBOT for unspecified bots.")

    return logtext


#==================== GAMEMODE LOGIC ====================#

mode = "play"

if len(sys.argv) > 2:
    mode = sys.argv[2]

#==================== GAME DEFINITION AND SELECTION ====================#


c4mdp = C4MDP(device=device)
tttmdp = TTTMDP(device=device)
ntmdp = NoThanksMDP(num_players=5, device=device)

names = {
    "ttt": "Tic-Tac-Toe",
    "c4": "Connect Four",
    "nothanks": "No Thanks!",
}

mdps = {
    'ttt': tttmdp,
    'c4': c4mdp,
    'nothanks': ntmdp,
}

file_exts = {
    'ttt': '.ttt.pt', 
    'c4': '.c4.pt', 
    'nothanks': 'nt.pt', 
}

nnarchss = {
    'ttt': [TTTResNN, TTTNN, TTTCatNN], 
    'c4': [C4ResNN, C4NN], 
    'nothanks': [NoThanksNN], 
}

nnargss = {
    'ttt': [{'num_hiddens': 4, 'hidden_depth': 1, 'hidden_width': 32}, {'num_hiddens': 3, 'channels': 32}, {'num_hiddens': 5, 'hidden_width': 16}], 
    'c4': [{'num_hidden_conv': 7, 'hidden_conv_depth': 1, 'hidden_conv_layers': 32, 'num_hidden_linear': 3, 'hidden_linear_depth': 1, 'hidden_linear_width': 32}, {}], 
    'nothanks': [{}], 
}


# If the game was specified, choose it
shortname = ""
if len(sys.argv) > 1:
    if sys.argv[1] in names:
        shortname = sys.argv[1]

else:
    print("Games:")
    for shortname in names:
        print(f"[{shortname}] {names[shortname]}")
    shortname = input("\nSelect game from list (input short name): ").strip().lower()

try:
    name = names[shortname]
    mdp = mdps[shortname]
    file_ext = file_exts[shortname]
    nnarchs = nnarchss[shortname]
    nnargs = nnargss[shortname]
except:
    print(f"Didn't recognize that game.  Exiting.")
    exit()


print(f"\nPlaying {name}.\n")

#==================== LOAD SAVES ====================#


save_files = [each for each in os.listdir(base_path + f'{shortname}/bots/dqn/') if each.endswith(file_ext)]
save_files.sort()


#==================== BOT TRAINING ====================#


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
    'training_delay': 'What episode to start training',
    'num_hiddens': 'Number of residual layers', 
    'hidden_depth': 'Linear layers per residual layer', 
    'hidden_width': 'Dimension of each residual layer divided by 18',
    'num_hidden_conv': 'Number of hidden convolutional residual layers',
    'hidden_conv_depth': 'Number of convolutional layers in each residual layer',
    'hidden_conv_layers': 'Number of layers in each convolutional layer',
    'num_hidden_linear': 'Number of hidden linear residual layers',
    'hidden_linear_depth': 'Number of linear layers in each residual layer',
    'hidden_linear_width': 'Width of each linear layer',
    }

# Train AI

if mode == "train":

    logtext = ""

    res = input("Name of file (alphanumeric only, max length 64, w/o extension): ")
    fname_end = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_" + re.sub(r'\W+', '', res)[0:64] + f"{file_ext}"
    fname = base_path + f'{shortname}/bots/dqn/' + fname_end

    print(f"Will save model to {fname}\n")

    res = input(f"Replay memory size (default: {mdp.default_memory}): ")
    try:
        memory = int(res)
        if memory < 0:
            raise Exception
        print(f"Set replay memory to {memory}.")
    except:
        memory = mdp.default_memory
        print(f"Did not understand.  Setting replay memory to default {memory}.")


    res = input("Load existing model (y/n)? ")
    if res.lower() == 'y':
        game = DQN(mdp, nn.Module, torch.nn.HuberLoss(), torch.optim.Adam, memory, device=device)
        logtext += load_bots(game, save_files)
    else:
        print("No model pre-loaded.")
        if len(nnarchs) == 0:
            print("No available neural network architectures to train on.  Aborting.")
            exit()
        print("Available neural network architectures:")
        for i in range(len(nnarchs)):
            print(f"[{i}] {nnarchs[i]}")
        res = input("Select archtecture (default is [0]): ")
        try:
            arch_index = int(res)
            print(f"Choosing {nnarchs[arch_index]}.")
            model = nnarchs[arch_index]
        except:
            print(f"Didn't understand, choosing default {nnarchs[0]}.")
            arch_index = 0
            model = nnarchs[arch_index]
        model_args = nnargs[arch_index]
            
        for k, v in nnargs[arch_index].items():
            res = input(f"{prompts[k] if k in prompts else k} (default {v}): ")
            try:
                model_args[k] = int(res)
            except:
                print(f"Not a valid value.  Setting {k} to {v}.")

        game = DQN(mdp, model, torch.nn.HuberLoss(), torch.optim.Adam, memory_capacity=memory, model_args=model_args, device=device)
    
    res = input("Penalize (pen) or prohibit (pro) invalid moves? (defualt prohbit) ")
    if res.lower() == "pen":
        valid_filter = False
        print("Chose penalty mode.")
    else:
        valid_filter = True
        print("Chose prohibition mode.")

    hpar = mdp.default_hyperparameters
    for k, v in hpar.items():
        res = input(f"{prompts[k] if k in prompts else k} (default {v}): ")
        try:
            hpar[k] = float(res)
        except:
            print(f"Not a valid value.  Setting {k} to {v}.")

    # if type == "qlearn":
    #     #game.set_greed(expl)
    #     game.batch_learn(**hpar, memory=100, verbose=True, savefile=fname + ".exp")
    #     game.save(fname)


    game.deep_q(**hpar, valid_filter=valid_filter, verbose=True, save_path=fname, initial_log = logtext)


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
            game.load(base_path + f'{shortname}/bots/dqn/' + saves[match[i]], [i])
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

    with open(base_path + f"logs/tournament.{shortname}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log", "w") as f:
        f.write(logtext)
    
    exit()


#==================== BOT SELECTION ====================#

if mode == "play":
    game = DQN(mdp, None, torch.nn.HuberLoss(), torch.optim.Adam, 0, device=device)
    load_bots(game, save_files)

#==================== BENCHMARK AGAINST RANDOM ====================#

if mode == "benchmark":

    sims = int(input("How many iterations against random bot? "))
    res = input("Benchmark all? (y/n): ")
    if res.lower() == 'y':
        do_all = True
    else:
        do_all = False

    if do_all == False:
        game = DQN(mdp, None, torch.nn.HuberLoss(), torch.optim.Adam, 0, device=device)
        load_bots(game, save_files)
        replay = True if input("Enter 'y' to replay losses: ").lower() == 'y' else False
        game.simulate_against_random(sims, replay_loss=replay, verbose=True)
    else:
        logtext = ""
        logtext += log(f"Simulating {name} against RANDOMBOT for {sims} simulations.")
        for i in range(1, len(save_files)):
            game.load(base_path + f'{shortname}/bots/dqn/' + save_files[i])
            result = game.simulate_against_random(sims, replay_loss=False, verbose=False)
            logtext += log(f"")
            for j in range(len(result)):
                logtext += log(f"Bot {save_files[i]} as player {j}: {result[j][0]} wins, {result[j][1]} losses, {result[j][2]} ties, {result[j][3]} invalid moves, {result[j][4]} unknown results.")

        with open(base_path + f"logs/benchmark.{shortname}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log", "w") as f:
            f.write(logtext)

    exit()
    




#==================== PLAY THE GAME ====================#

while True:
    players = []
    res = input(f"Which players are human?  A comma-separated list of numbers from 1 to {mdp.num_players}, empty to watch the bots play, and 'q' to quit. ").split(',')
    if res[0].lower() == 'q':
        exit()
    for r in res:
        try:
            player_index = int(r) - 1
            if player_index < 0 or player_index >= mdp.num_players:
                raise Exception()
            else:
                players.append(player_index)
        except:
            print(f"Didn't recognize {r.strip()}.  Using bot.")
    game.play(players, verbose=True)
    