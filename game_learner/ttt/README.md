# Experiments on a toy example: Tic-Tac-Toe

The goal of this note is to record the results of some experiments conducted while attempting to train bots using reinforcement learning techniques guided by neural networks, for example tuning hyperparameters or playing with different model architectures.  Though we intend to train bots to play different games, we restrict our attention in this note to Tic-Tac-Toe, for which bots can be quickly trained on essentially any modern computer without special equipment.

In a 2015 paper[^MKS15], Minh, Kavukcuoglu, Silver *et. al.* outline an algorithm for using neural networks within a Q-learning algorithm (DQN) to train bots to play various player vs. environment (PvE) games on the Atari video game platform.  The same algorithms may be readily adapted to PvP games by giving each player their own Q-function, with each player considering the other players as part of the environment.  Unlike the PvE setting, the ``environment'' is now no longer given by a static and fixed (stochastic) Markov decision process, but one which evolves along with the player.  We do not investigate whether the theoretical assumptions justifying Q-learning algorithms remain valid in this set-up, but experimentally we find that they are still able to train good bots.

Before discussing DQN, we give a brief review of reinforcement learning.  The goal of reinforcement learning is to learn an optimal *policy* for a bot operating in some stochastic environment, i.e. an assignment $\pi: S \rightarrow A$ where $S$ is the set of game states and $A$ is the set of actions, or more generally a "stochastic" function, i.e. non-deterministic function with underlying probability distribution.  Alternatively, we can learn a *Q-function* (quality function) $Q: S \times A \rightarrow \bR$, which assigns a numerical value to the act of taking a given action at a given state.  A Q-function induces a (stochastic) policy by taking the actions with maximum value with equal probability.


[^MKS15]: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, *et. al.*, Human-level control through deep reinforcement learning, *Nature* volume 518, pages 529–533 (2015).

[^SH16]: David Silver Aja Huang, *et. al.*, Mastering the game of Go with deep neural networks and tree search, *Nature* volume 529, pages 484-489 (2016).
		
[^SSS17]: David Silver, Julian Schrittwieser, Karen Simonyan, *et. al.*, Mastering the game of Go without human knowledge, *Nature* volume 550, pages 354-359 (2017).

