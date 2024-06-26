import torch
import torch.nn as nn

from deepqlearn import TensorMDP



class NoThanksMDP(TensorMDP):
    start_chips_table = {3: 11, 4: 11, 5: 11, 6: 9, 7: 7}


    # State: who owns each card (33 cards total), how many chips each player has and the "center" has, who the current player is.  Shape (num_players + 1, num_cards + 2).
    # First factor: -1 is "middle", others are the players
    # Second factor: 0 is current player, 1 is number of chips, and 2-34 are the cards 3-35
    # Action: no = 0, take = 1, shape (2, )
    def __init__(self, num_players: int, num_cards = 33, smallest_card = 3, num_played_cards = 24, device="cpu"):
        num_players = min(max(num_players, 3), 7)

        defaults = {
            'lr': 0.00025, 
            'greed_start': 0.0, 
            'greed_end': 0.50, 
            'dq_episodes': 100, 
            'ramp_start': 25,
            'ramp_end': 100,
            'training_delay': 25,
            'episode_length': 50, 
            'sim_batch': 16, 
            'train_batch': 64,
            }
        super().__init__(state_shape=(num_players + 1, num_cards + 2), action_shape=(2,), default_memory=1000000, discount=0.9, num_players=num_players, batched=True, default_hyperparameters=defaults, \
                         symb = {0: "1", 1: "2", 2: '3', 3: "4", 4: '5', None: "-"}, input_str = "Type 'no' to reject the card and 'ok' to accept it. ", penalty=-1, nn_args={'num_players': num_players, 'num_cards': num_cards}, num_simulations=100, device=device)


        self.num_cards = num_cards
        self.smallest_card = smallest_card
        self.num_played_cards = num_played_cards

        self.start_chips = NoThanksMDP.start_chips_table[num_players]

        # Dot this along axis 0 (note the extra dimension for the "board")
        self.turn_matrix = torch.cat([torch.eye(num_players+1, device=self.device)[1:-1], torch.eye(num_players+1, device=self.device)[0:1], torch.eye(num_players+1, device=self.device)[-1:]])

        self.up_card = torch.cat([torch.eye(self.num_cards, device=self.device)[1:], torch.eye(self.num_cards, device=self.device)[0:1]])
        self.down_card = torch.cat([torch.eye(self.num_cards, device=self.device)[-1:], torch.eye(self.num_cards, device=self.device)[0:-1]])

        self.initial_state = torch.zeros((1,) + self.state_shape, device=self.device)
        # Set current player
        self.initial_state[0,0,0] = 1.
        # Set initial chips
        for i in range(num_players):
            self.initial_state[0,i,1] = self.start_chips



    ##### UI RELATED METHODS #####
    # Non-batch and not used in internal code, efficiency not as important

    def action_str(self, action: torch.Tensor) -> list[str]:
        outs = []
        for b in range(action.size(0)):
            outs.append(f"No thanks!" if action[b,0].item() > action[b,1].item() else "Taking card.")
        return outs

    def str_to_action(self, input: str) -> torch.Tensor:
        c = (input + " ").lower()[0]
        out = torch.zeros((1,2), device=self.device)
        if c == 'n':
            out[0,0] = 1.
        else:
            out[0,1] = 1.
        return out

    def board_str(self, state):
        outs = []
        for i in range(state.size(0)):
            s = abs(state[i:i+1])
            board_str = ""
            p = self.get_player(s).int().item() + 1
            board_str += f"Current player: {p}\n" if self.is_terminal(s).item() == False else "Game has terminated.\n"
            board_str += f"Current flipped card and chips: {s[0,-1,1].int().item()} chips on card {self.get_center_card(s).int().item()}\n"
            board_str += f"Number of revealed cards so far: {self.num_cards_out(s).int().item()}/{self.num_played_cards}.\n"
            board_str += f"Chips each player owns: {s[0,:-1,1].int().tolist()}.\n"
            for j in range(self.num_players):
                cards = (s[0,j,2:].int() * torch.arange(self.smallest_card, self.smallest_card + self.num_cards, device=self.device)).tolist()
                board_str += f"Player {j+1} has cards {[c for c in cards if c != 0]}.\n"
            outs.append(board_str)
        return outs

    def get_rules(self=None):
        rules = "\n==========RULES==========\n"
        rules += f"The deck consists of 24 integer-valued cards drawn from the range [3, 35].\n"
        if self != None:
            num = len(self.players)
            rules += f"There are currently {num} players, each starting with {NoThanksMDP.startChips[num]} chips.\n"
        rules += f"On your turn, you can either take the card and chips on the table, or reject it by paying one chip if you have one.  The chips you take are playable, the card goes in your bank.\n"
        rules += f"When the cards are exhausted, the game ends, and you score as follows.\n"
        rules += f"For each run (consecutive sequence) in your bank, you add the score of the lowest card in the run.  You then subtract the number of chips.\n"
        rules += f"Lowest score wins.\n"
        rules += f"==========RULES==========\n\n"
        return rules
    

    ##### INTERNAL LOGIC #####


    ### PLAYERS ### 

    # Output shape (batch, 1, 1)
    def get_player(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(self.get_player_vector(state), torch.arange(self.num_players + 1, device=self.device).float(), ([1],[0]))[:,None]

    # Output shape (batch, num_players+1, 1)    
    def get_player_vector(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :, 0:1]

    def next_player(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.tensordot(state[:,:,0:1], self.turn_matrix, ([1],[0])).swapaxes(1,2), state[:,:,1:]], dim=2)

    ### ACTIONS ###

    # You can always take the card, but you cannot say no thanks if you have no chips; also no actions if terminal
    def valid_action_filter(self, state: torch.Tensor) -> torch.Tensor:
        return ~self.is_terminal(state)[:,0] * (torch.ones((state.size(0), 2), device=self.device) - (self.has_no_chips(state)[:,0] * torch.tensor([1.,0.], device=self.device)))

    ### STATES ###

    
    # Returns all zeros except a 1 for the center for a card which is not owned by any player (or in the center)
    def get_random_card(self, state, max_tries=100):    
        filter = (state.sum(1) == 0.).float()                                                    # Won't choose a "chip" or "player token" because those are always nonzero
        while (filter.count_nonzero(dim=1) <= 1).prod().item() != 1:                             # Almost always terminates after one step
            temp = torch.rand((state.size(0), self.num_cards + 2), device=self.device) * filter
            filter = (temp == temp.max(1).values[:,None]).float()
            max_tries -= 1
            if max_tries == 0:
                break
        return torch.cat([torch.zeros((state.size(0), self.num_players, self.num_cards + 2), device=self.device), (filter * 1.)[:,None]], 1)


    def get_initial_state(self, batch_size=1) -> torch.Tensor:
        init = self.initial_state * torch.ones((batch_size,) + self.state_shape, device=self.device)
        return init + self.get_random_card(init)

    # Output shape (batch, num_cards)
    def get_current_player_cards(self, state):
        return state[torch.arange(state.size(0), device=self.device), self.get_player(state)[:,0,0].int(), 2:]
    
    def get_center_card_vector(self, state):
        return state[:,-1,2:]
    
    def get_center_card(self, state):
        return torch.tensordot(self.get_center_card_vector(state), torch.arange(self.smallest_card, self.smallest_card + self.num_cards, device=self.device).float(), ([1],[0]))[:,None,None]

    # Here, players only gain rewards on their turns
    def transition(self, state, action):
        # Zero out actions for invalid actions
        is_valid = self.is_valid_action(state, action)
        action *= is_valid[:, 0]
        reward = torch.zeros((state.size(0), self.num_players), device=self.device)

        # Those who say no thanks, remove a token (only people with tokens can take actions)
        state[:,0:-1,1] -= (action[:,0] == 1.)[:,None] * (state[:,0:-1,0] == 1.).float()
        # Put a token in the middle
        state[:,-1,1] += (action[:,0] == 1.).float()
        # A token is a point, so adjust the reward
        reward[torch.arange(state.size(0), device=self.device),self.get_player(state)[:,0,0].int()] -= (action[:,0] == 1.).float()

        # Those who take the card, reassign card, give tokens, check card count, flip new card or end game
        # Reward logic: take card N, 
        # if has N+1 then reward N+1
        up_condition = self.get_current_player_cards(state) * torch.tensordot(self.get_center_card_vector(state), self.up_card, ([1],[0]))      # Shape (batch, num_cards)
        reward[torch.arange(state.size(0), device=self.device), self.get_player(state)[:,0,0].int()] += (action[:,1] == 1.) * torch.sum(torch.arange(self.smallest_card, self.smallest_card + self.num_cards, device=self.device) * up_condition, 1)
        # if does not have N-1 then reward -N
        down_condition = torch.tensordot(1 - self.get_current_player_cards(state), self.up_card, ([1],[0])) * self.get_center_card_vector(state)
        reward[torch.arange(state.size(0), device=self.device), self.get_player(state)[:,0,0].int()] -= (action[:,1] == 1.) * torch.sum(torch.arange(self.smallest_card, self.smallest_card + self.num_cards, device=self.device) * down_condition, 1)
        # Also a reward for chips
        reward[torch.arange(state.size(0), device=self.device),self.get_player(state)[:,0,0].int()] += (action[:,1] == 1.) * state[:,-1,1]

        # Give card and chips to player
        state += (action[:,1] == 1.).float()[:,None,None] * torch.tensordot(self.get_player_vector(state)[:,:,0], torch.cat([torch.zeros((state.size(0), 1), device=self.device), state[:,-1,1:]], 1), dims=0).diagonal(dim1=0,dim2=2).swapaxes(0,2).swapaxes(1,2)
        
        # Remove card from center (zero out center)
        state[:,-1,:] -= (action[:,1] == 1.).float()[:,None] * state[:,-1,:]

        # Flip a new card if the game has not ended
        state += (action[:,1] == 1.).float()[:,None,None] * (self.num_cards_out(state) < self.num_played_cards) * self.get_random_card(state)

        # Advance to the next player if action was valid
        state = self.next_player(state)* is_valid + state * (~is_valid)

        return state, reward / self.num_cards
    
    # Output shape (batch, 1, 1)
    # Checks whether the current player has no chips
    def has_no_chips(self, state):
        return (state[torch.arange(state.size(0), device=self.device), self.get_player(state)[:,0,0].int(), 1] == 0)[:,None,None]

    def num_cards_out(self, state):
        return state[:,:,2:].sum((1,2))[:,None,None]

    def get_random_state(self):
        return self.get_initial_state()

    # Indicate a terminal state by having no card in the center
    def is_terminal(self, state: torch.Tensor) -> torch.Tensor:
        return state.sum(2)[:,-1:,None] == 0.




# Input shape (batch, )
class NoThanksNN(nn.Module):
    def __init__(self, num_players: int, num_cards: int):
        super().__init__()
        self.num_players = num_players
        self.num_cards = num_cards
        self.stack = nn.Sequential(
            nn.Conv1d(num_players + 1, num_players * 16, 3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_players * 16),
            nn.Flatten(),
            nn.Linear(num_players * 16 * (num_cards + 2), num_players * 16 * (num_cards + 2)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_players * 16 * (num_cards + 2)),
            nn.Linear(num_players * 16 * (num_cards + 2), num_players * 4 * (num_cards + 2)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_players * 4 * (num_cards + 2)),
            nn.Linear(num_players * 4 * (num_cards + 2), num_players * (num_cards + 2)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_players * (num_cards + 2)),
            nn.Linear(num_players * (num_cards + 2), 2)
        )

    def forward(self, x):
        return self.stack(x)
    



class NoThanksResNN(nn.Module):
    def __init__(self, num_players: int, num_cards: int, num_hiddens = 4, hidden_depth=1, hidden_width = 16):
        super().__init__()
        self.num_players = num_players
        self.num_cards = num_cards

        self.head_stack = nn.Sequential(
            nn.Conv1d(num_players + 1, num_players * hidden_width, 3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_players * hidden_width),
            nn.Flatten(),
        )
        hlays = []
        for i in range(num_hiddens):
            lay = nn.Sequential(nn.Linear(num_players * hidden_width * (num_cards + 2), num_players * hidden_width * (num_cards + 2)))
            for j in range(hidden_depth-1):
                lay.append(nn.BatchNorm1d(num_players * hidden_width * (num_cards + 2)))
                lay.append(nn.ReLU())
                lay.append(nn.Linear(num_players * hidden_width * (num_cards + 2), num_players * hidden_width * (num_cards + 2)))
            lay.append(nn.BatchNorm1d(num_players * hidden_width * (num_cards + 2)))
            hlays.append(lay)
        
        self.hidden_layers = nn.ModuleList(hlays)
        self.tail = nn.Sequential(
            nn.Linear(num_players * hidden_width * (num_cards + 2), num_players * (num_cards + 2)),
            nn.ReLU(),
            nn.BatchNorm1d(num_players * (num_cards + 2)),
            nn.Linear(num_players * (num_cards + 2), num_cards + 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cards + 2),
            nn.Linear(num_cards + 2, 2),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.head_stack(x)
        for h in self.hidden_layers:
            x = self.relu(h(x) + x)
        return self.tail(x)

