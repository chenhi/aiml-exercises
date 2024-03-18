import random, pickle


# WARNING: states and actions must be hashable if using with QFunction.
class MDP():
    def __init__(self, states, actions, discount=1, num_players=1, penalty = -1000, symb = {}, input_str = "", batched=False):
        self.actions = actions
        self.discount = discount
        self.num_players = num_players
        self.batched = batched
        self.penalty=penalty
        self.symb = symb
        self.input_str = input_str

        # The states do not literally have to be tensors for the state_shape to be defined.  This just specifies, when they are turned into tensors, what shape they should be have, ignoring batch dimension
        self.states = states

        self.actions = actions
        
    def str_to_action(self, input: str):
        raise NotImplementedError
    
    def board_str(self, state) -> str:
        raise NotImplementedError
    




    def is_state(self, state):
        if self.states != None:
            return True if state in self.states else False
        elif self.state_shape != None:
            return True if state[0].shape == self.state_shape else False
        else:
            return True
        
    def is_action(self, action):
        if self.actions != None:
            return True if action in self.actions else False
        elif self.action_shape != None:
            return True if action[0].shape == self.action_shape else False
        else:
            return True

    def action_str(self, action) -> str:
        raise NotImplementedError


        
    def get_random_state(self):
        if self.states != None:
            return random.choice(self.states)
        raise NotImplementedError

    # Can re-implement this to take the state into consideration to avoid invalid moves    
    def get_actions(self, s = None):
        if s == None:
            return self.actions
        else:
            raise NotImplementedError

        
    def is_valid_action(self, state, action):
        return action in self.get_actions(state)

    def get_random_action(self, state = None):
        return random.choice(self.get_actions(state))

    # Returns: (next state, reward)
    def transition(self, s, a):
        raise NotImplementedError

    # Returns a valid start state and starting player for the game.
    def get_initial_state(self):
        raise NotImplementedError

    # Tells us if a state is terminal.
    def is_terminal(self, s):
        raise NotImplementedError
    
    # Gets the current player of the given state.
    def get_player(self, state):
        raise NotImplementedError




#################### CLASSICAL Q-LEARNING ####################


# Returns the set of maximum arguments
def argmax(args: list, f: callable):
    maxval = None
    output = []
    for x in args:
        y = f(x)
        if maxval == None:
            output.append(x)
            maxval = y
        elif y == maxval:
            output.append(x)
        elif y > maxval:
            maxval = y
            output = [x]
    return output

def valmax(args: list, f: callable):
    maxval = None
    for x in args:
        y = f(x)
        if maxval == None or y > maxval:
            maxval = y
    return maxval


# Q (Quality) function class
# Given a set of states S and a set of actions A, a value function maps Q: S x A ---> R and reflects the value (in terms of rewards) of performing a given action at the state.
# The sets S and A might be infinite.  Therefore in practice we do not require Q to be defined everywhere.
# We may or may not specify a set from which 
class QFunction():

    def __init__(self, mdp: MDP):
        self.q = {}
        self.mdp = mdp

    def copy(self):
        new_q = QFunction(self.mdp.copy())
        new_q.q = self.q.copy()
    
    def get(self, s, a) -> float:
        return 0 if (s, a) not in self.q else self.q[(s, a)]

    # Returns the value at a given state, i.e. max_a(Q(s, a))
    # Value of terminal state should always be 0
    def val(self, s) -> float:
        if self.mdp.is_terminal(s):
            return 0
        
        # If we have a defined (finite) set of actions, just iterate
        if self.mdp.actions != None:
            return valmax(self.mdp.get_actions(s), lambda a: self.get(s, a))
        else:
            raise NotImplementedError           #If there are infinitely many actions, this needs to be handled explicitly
    
    # Returns a list of optimal policies.  If the state is terminal or there are no valid actions, return empty list.
    # This is probably only ever called internally.  But separating it anyway.
    def policies(self, state):
        if self.mdp.is_terminal(state):
            return []
        
        # If we have a defined set of actions, we can just do an argmax.
        if self.mdp.actions != None:
            return argmax(self.mdp.get_actions(state), lambda a: self.get(state,a))
        else:
            raise NotImplementedError


    # Returns a randomly selected optimal policy.
    def policy(self, s) -> float:
        pols = self.policies(s)
        if len(pols) == 0:
            if self.mdp.actions != None:
                pols = self.mdp.actions
            else:
                raise NotImplementedError
        return random.choice(pols)

    # Does a Q-update based on some observed set of data
    # Data is a list of the form (state, action, reward, next state)
    def update(self, data: list[tuple[any, any, float, any]], learn_rate):
        for d in data:
            s,a,r,t = d[0], d[1], d[2], d[3]
            self.q[(s,a)] = (1 - learn_rate) * self.get(s,a) + learn_rate * (r + self.mdp.discount * self.val(t))

    # Learn based on a given strategy for some number of iterations, updating each time.
    # In practice, this doesn't get used so much, because the "game" has to handle rewards between players (not the Q function itself)
    def learn(self, strategy: callable, learn_rate: float, iterations: int):
        s = self.mdp.get_initial_state()
        for i in range(iterations):
            if self.mdp.is_terminal(s):
                s = self.mdp.get_initial_state()
                continue
            a = strategy(s)
            t, r = self.mdp.transition(s, a)
            self.update([(s,a,r,t)], learn_rate)
            s = t

    # Learn in batches.  An update happens each iteration, on all past experiences (including previous iterations).  A state reset happpens each episode.
    # In practice, this doesn't get used so much, because the "game" has to handle rewards between players (not the Q function itself)
    def batch_learn(self, strategy: callable, learn_rate: float, iterations: int, episodes: int, episode_length: int, remember_experiences = True):
        experiences = []
        for i in range(iterations):
            for j in range(episodes):
                s = self.mdp.get_initial_state()
                for k in range(episode_length):
                    a = strategy(s)
                    t, r = self.mdp.transition(s, a)
                    experiences.append((s, a, r, t))
                    if self.mdp.is_terminal(t):
                        break
                    s = t
            self.update(experiences, learn_rate)
            if not remember_experiences:
                experiences = []
                


# For backwards compatibility
class ValueFunction(QFunction):
    def __init__(self, mdp: MDP):
        super().__init__(mdp)





# Greedy function to use as a strategy.  Default is totally greedy.
# This only works if the set of actions is defined and finite
def greedy(q: QFunction, state, eps = 0.):
    return q.mdp.get_random_action(state) if random.random() < eps else q.policy(state)




def get_greedy(q: QFunction, eps: float) -> callable:
    return lambda s: greedy(q, s, eps)



# To each player, the game will act like a MDP; i.e. they do not distinguish between the opponents and the environment
# We can only batch learn, since there is a lag in rewards updating
# Player order can make a difference, so when training, do not shuffle the order

# Requirements for the MDP:
# The state must be a tuple, whose first entry is the current player
# The rewards are returned as a tuple, corresponding to rewards for each player
class QLearn():
    def __init__(self, mdp: MDP):
        # An mdp that encodes the rules of the game.  The current player is part of the state, which is of the form (current player, actual state)
        self.mdp = mdp
        self.qs = [QFunction(self.mdp) for i in range(mdp.num_players)]
        self.state = None

    def set_greed(self, eps):
        if type(eps) == list:
            self.strategies = [get_greedy(self.qs[i], eps[i]) for i in range(self.mdp.num_players)]
        else:
            self.strategies = [get_greedy(self.qs[i], eps) for i in range(self.mdp.num_players)]

    def save_q(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.qs, f)
        
    def load_q(self, fname, indices=None):
        with open(fname, 'rb') as f:
            temp_qs = pickle.load(f)
        if indices == None:
            self.qs = temp_qs
        else:
            for i in indices:
                self.qs[i] = temp_qs[i]

    def null_q(self, indices = None):
        if indices == None:
            self.qs = [QFunction(self.mdp) for i in range(self.mdp.num_players)]
        else:
            for i in indices:
                self.qs[i] = QFunction(self.mdp)
    
    def stepthru_game(self):
        s = self.mdp.get_initial_state()
        print(f"Initial state:\n{self.mdp.board_str(s)}")
        turn = 0
        while self.mdp.is_terminal(s) == False:
            turn += 1
            p = self.mdp.get_player(s)
            print(f"Turn {turn}, player {p+1} ({self.mdp.symb[p]})")
            a = self.qs[p].policy(s)
            print(f"Chosen action: {self.mdp.action_str(a)}")
            s, r = self.mdp.transition(s, a)
            print(f"Next state:\n{self.mdp.board_str(s)}")
            print(f"Rewards for players: {r}")
            input("Enter to continue.\n")
        input("Terminal state reached.  Enter to end. ")

    # Non-batched method
    def current_player(self, s) -> int:
        if s == None:
            return None
        if self.mdp.batched:
            return self.mdp.get_player(s).item()
        else:
            return self.mdp.get_player(s)

    # Player data is (start state, action taken, all reward before next action, starting state for next action)
    def batch_learn(self, learn_rate: float, iterations: int, episodes: int, episode_length: int, verbose=False, savefile=None):
        player_experiences = [[] for i in range(self.mdp.num_players)]
        for i in range(iterations):
            for j in range(episodes):
                if verbose and j % 10 == 9:
                    print(f"Training iteration {i+1}, episode {j+1}", end='\r')
                s = self.mdp.get_initial_state()
                queue =[None for k in range(self.mdp.num_players)]
                for k in range(episode_length):
                    p = self.current_player(s)
                    a = self.strategies[p](s)
                    t, r = self.mdp.transition(s, a)

                    # For this player, bump the queue and add
                    if queue[p] != None:
                        player_experiences[p].append(tuple(queue[p]) + (s,))
                    if self.mdp.is_terminal(t):
                        player_experiences[p].append((s,a,r[p],t))
                    else:
                        queue[p] = [s, a, r[p]]
                    

                    # Update rewards for all other players; if the player hasn't taken an action yet, no reward (but is accounted somewhat by zero sum nature)
                    # If the state is terminal, also append
                    for l in range(self.mdp.num_players):
                        if l != p and queue[l] != None:
                            queue[l][2] += r[l]
                            if self.mdp.is_terminal(t):
                                player_experiences[l].append(tuple(queue[l]) + (t,))

                    # If terminal state, then stop the episode.  Otherwise, update state and continue playing 
                    if self.mdp.is_terminal(t):
                        break
                    s = t
            # Do an update for each player
            for p in range(self.mdp.num_players):
                self.qs[p].update(player_experiences[p], learn_rate)
        if verbose:
            total = 0
            for e in player_experiences:
                total += len(e)
            print(f"Trained on {total} experiences.")
        if savefile != None:
            with open(savefile, 'wb') as f:
                pickle.dump(player_experiences, f)
                if verbose:
                    print(f"Saved experiences to {savefile}")

