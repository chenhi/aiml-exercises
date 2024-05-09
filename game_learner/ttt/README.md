# Experiments on a toy example: Tic-Tac-Toe

The goal of this note is to record the results of some experiments conducted while attempting to train bots using reinforcement learning techniques guided by neural networks, for example tuning hyperparameters or playing with different model architectures.  Though we intend to train bots to play different games, we restrict our attention in this note to Tic-Tac-Toe, for which bots can be quickly trained on essentially any modern computer without special equipment.

Ultimately, our goal is to train bots to play a game from scratch, i.e. without human knowledge like AlphaZero[^SSS17] as opposed to AlphaGo[^SH16].

## Deep Q-networks (DQN)

In a 2015 paper[^MKS15], Minh, Kavukcuoglu, Silver *et. al.* outline an algorithm for using neural networks within a Q-learning algorithm (DQN) to train bots to play various player vs. environment (PvE) games on the Atari video game platform.  The same algorithms may be readily adapted to PvP games by giving each player their own Q-function, with each player considering the other players as part of the environment.  Unlike the PvE setting, the "environment" is now no longer given by a static and fixed (stochastic) Markov decision process, but one which evolves along with the player.  We do not investigate whether the theoretical assumptions justifying Q-learning algorithms remain valid in this set-up, but experimentally we find that they are still able to train good bots.

I made a few minor design modifications to the DQN algorithm (as well as ignoring issues that obviously do not arise in our setting).  I simulate games in batches, which is not difficult due to the very discrete nature of the games we consider, which should give a speed boost; this introduces the size of the simulation batches as an additional hyperparameter.

We also note that convergence alone (i.e. vanishing of loss) does not necessarily indicate a good bot, since its performance *in natura* depends on the quality of the generated training data.  Ensuring good data often means not letting the behavior policy get too greedy.  Due to the random play, this means the loss typically has a non-zero lower bound.  I personally found it helpful to conceptually separate the simulation of data and the learning on that simulated data.

Finally, we note that all loss curves have undergone a smoothing with kernel width 11 for readability.

### Dealing with illegal moves


There is an immediate design choice that needs to be made: how to handle illegal moves by the bot.  Because the bot plays deterministically, if left unhandled it is possible for the bot to get stuck in infinite loops attempting illegal moves.  In classical Q-learning, actions are just a set and it is straightforward to restrict the set of moves to legal ones.  In deep Q-learning, actions are represented by basis vectors in a fixed vector space, and it is possible for the neural network to select an illegal move.  I saw three solutions:
+ **Randomness:** choose a random action when an illegal action is chosen.
+ **Prohibition:** prohibit illegal moves by zeroing out their corresponding components.
+ **Penalty:** teach the bot to avoid illegal moves by assigning a penalty to illegal moves.

Randomness appeared to me strictly inferior to prohibition, so I didn't experiment with it.  Below are the loss curves comparing prohibition and penalty.
<p align="center">
<img src="graphs/20240413213956_zeroout2.dttt.pt.losses.png" width="24%"><img src="graphs/20240413220118_zerooutrnn.dttt.pt.losses.png" width="24%"><img src="graphs/20240413212415_penalty2.dttt.pt.losses.png" width="24%"><img src="graphs/20240413222418_resnnpenalty.dttt.pt.losses.png" width="24%"></p>
<p align="center">
Prohibition (left two; linear and residual architectures) vs. penalty (right two; linear and residual architectures).
</p>

The prohibition curves appear more noisy, likely since it is common for the bot to choose an illegal move during exploration and incur a penalty.  In terms of final performance against a random player, they seemed to be fairly similar, sometimes one perfoming better than the other.  The loss curves seem to exhibit more convergence for prohibition; intuitively this makes sense as there is less to learn.  Moreover, some specific case metrics seem to indicate that prohibition should perform better in general:
<p align="center">
<img src="graphs/20240413203510_test_zeroout.dttt.pt.test0.png" width="24%"><img src="graphs/20240413213956_zeroout2.dttt.pt.test0.png" width="24%"><img src="graphs/20240413210030_test_penalty.dttt.pt.test0.png" width="24%"><img src="graphs/20240413212415_penalty2.dttt.pt.test0.png" width="24%">
<img src="graphs/20240413203510_test_zeroout.dttt.pt.test1.png" width="24%"><img src="graphs/20240413213956_zeroout2.dttt.pt.test1.png" width="24%"><img src="graphs/20240413210030_test_penalty.dttt.pt.test1.png" width="24%"><img src="graphs/20240413212415_penalty2.dttt.pt.test1.png" width="24%">
</p><p align="center">
Performance metrics for prohibition (left two) vs. penalty (right two).
</p>

In particular, for Test 1, we see that we tend to see much more separation between the green curve and the blue/orange curves using prohibition, an indication that the bot is learning to distinguish a particular group of losing moves vs. tying moves.

I implemented penalty first since it was more straightforward, but later switched to prohibition.  Many experiments in the remainder of the document use penalty; one unintended benefit of this is that the number of illegal moves attempted by the bot can be used as a metric for how well the bot has learned the basic rules of the game.


### Magnitude of penalty

The magnitude of the penalty has an effect on neural network training where it does not in classical Q-learning.  For classical Q-learning, the function is an arbitrary function on a discrete set of states.  For deep Q-learning, the function is ``built from'' linear functions defined on a vector space continuum (but only evaluated on a discrete subset).  In particular, for deep Q-learning, large values can skew the weights during training.  In the beginning, I had set the penalty to -1000, which worked classically but caused divergence when training neural networks.  I tested this in an experiment comparing penalties of -2 vs. -1000.  I also tested a penalty of -1, which was not significantly different from a penalty of -2.

<p align="center">
<img src="graphs/20240325030438_badpenalty.dttt.pt.log.losses.png" width="40%"><img src="graphs/20240324030517_4000its.dttt.pt.log.losses.png" width="40%">
<p>
<p align="center">Loss curve over 4000 iterations: -1000 penalty (left) vs. -2 penalty (right).</p>

<table align="center">
<tr><td>iterations</td><td colspan="2">1000</td><td colspan="2">1500</td><td colspan="2">4000</td></tr>
<tr><td>penalty</td><td>2</td><td>-1000</td><td>-2</td><td>-1000</td><td>2</td><td>-1000</td></tr>
<tr><td>player 1 losses vs. random</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td></tr>
<tr><td>player 2 losses vs. random</td><td>1.27%</td><td>1.25%</td><td>1.60%</td><td>2.81%</td><td>0.00%</td><td>0.10%</td></tr>
<tr><td>player 1 invalid moves</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
<tr><td>player 2 invalid moves</td><td>6</td><td>63</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
</table>

With a large penalty the model much longer to begin to converge.  A large penalty negatively impacts the performance of the bot measured in losses as well as, perhaps counterintuitively, its ability to avoid illegal moves.  In the long term, the bot appears to be able to adjust its weights to account for the large penalty, but in general it seems best to avoid it.

### Greed and convergence

To generate good gameplay data, the bots must strike a balance between exploration and exploitation.  For Q-learning we will use a simple greed parameter to control the probability that the bot plays according to what it thinks is optimal (exploitation) versus randomly (exploration).  It is an annoying convention that the so-called greed parameter measures how much the bot explores; we will use the term *exploration parameter* instead, which is complementary to the *greed parameter*, i.e. a greed parameter of 0 means it always plays randomly.  This greed parameter may change over time.

In PvE games, it is typically recommended to start the greed low, around 0.0, and then end high, around 0.9.  The reasoning is that the bot should explore a lot in the beginning, then hone in on a winning strategy.  In PvP games, experimentally, it appears better to keep the ending greed lower.  We postulate the following reason: in PvE situations, the player has, ignoring randomness, total control over which branch of the game tree to go down.  Therefore, it is okay for the player to forget branches of the tree that it does not like.  On the other hand, in PvP situations, the opposing player has an equal share of control.  Setting the greed parameter too high can causes the neural network to forget some branches of the game tree.

To visualize the effect of greed on convergence and performance, I trained a bot under the following schemes for 4000 iterations.
+ **No greed:** the greed stayed at 0.0 throughout, i.e. the training data was completely randomly generated.
+ **Middle greed:** the greed ramped from 0.0 to 0.6 linearly in the interval $[100, 2500]$.
+ **Max greed:** the greed ramped from 0.0 to the maximum 1.0 linearly in the interval $[100, 3900]$.

<p align="center">
<img src="graphs/20240328201353_nogreed.dttt.pt.losses.png" width="33%"><img src="graphs/20240328101230_postbug_baseline.dttt.pt.losses.png" width="33%"><img src="graphs/20240328191907_greedto100.dttt.pt.losses.png" width="33%">
<p>
<p align="center">Loss curves over 4000 iterations, no vs. middle vs. max greed.</p>


<table align="center">
<tr><td></td><td colspan="4">player 1</td><td colspan="4">player 2</td></tr>
<tr><td>greed</td><td>win</td><td>loss</td><td>tie</td><td>invalid</td><td>win</td><td>loss</td><td>tie</td><td>invalid</td></tr>
<tr><td>no</td><td>69.37%</td><td>19.43%</td><td>11.20%</td><td>31168</td><td>47.47%</td><td>40.07%</td><td>12.46%</td><td>21536</td></tr>
<tr><td>middle</td><td>98.96%</td><td>0.00%</td><td>1.04%</td><td>0</td><td>90.73%</td><td>0.00%</td><td>9.27%</td><td>0</td></tr>
<tr><td>max</td><td>95.54%</td><td>0.63%</td><td>3.83%</td><td>284</td><td>90.44%</td><td>0.00%</td><td>9.56%</td><td>0</td></tr>
</table>

We observe that higher greed can result in converging to a value with lower loss, but does not necessarily result in a better bot.  We also observe divergence for the first player and convergence for the second player using the no-greed policy.  My guess for why is that if the opponent plays randomly, this can result in higher variance in outcomes.



### TODO

We will see later, e.g. in Figure \ref{relu or batch}, that sometimes a player 0 bot can be trained to never lose but still occasionally attempt illegal moves, reflecting that it's generally easier for player 0 to win.

[^MKS15]: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, *et. al.*, Human-level control through deep reinforcement learning, *Nature* volume 518, pages 529–533 (2015).

[^SH16]: David Silver Aja Huang, *et. al.*, Mastering the game of Go with deep neural networks and tree search, *Nature* volume 529, pages 484-489 (2016).
		
[^SSS17]: David Silver, Julian Schrittwieser, Karen Simonyan, *et. al.*, Mastering the game of Go without human knowledge, *Nature* volume 550, pages 354-359 (2017).

