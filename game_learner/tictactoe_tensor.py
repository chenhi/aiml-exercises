from qlearn import *
from deepqlearn import *
import torch, sys

options = sys.argv[1:]


class TTTTensorMDP(TensorMDP):
    def __init__(self):
        super().__init__(state_shape=(2,3,3), action_shape=(3,3), discount=1, num_players=2, batched=True, \
                         symb = {0: "X", 1: "O", None: "-"}, input_str = "Input position to play, e.g. '1, 3' for the 1st row and 3rd column: ", penalty=-2)

    def __str__(self):
        return f"TTTTensorMDP: discount={self.discount}, penalty={self.penalty}"

    def str_to_action(self, input: str) -> torch.Tensor:
        coords = input.split(',')
        try:
            i = int(coords[0]) - 1
            j = int(coords[1]) - 1
        except:
            return None
        out = torch.zeros((1,3,3), dtype=int)
        out[0,i,j] = 1
        return out



    def board_str(self, state):
        outs = []
        for i in range(state.size(0)):
            s = abs(state[i])
            board_str = ""
            for i in range(3):
                for j in range(3):
                    if s[0,i,j].item() == 1:
                        board_str += self.symb[0]
                    elif s[1,i,j].item() == 1:
                        board_str += self.symb[1]
                    else:
                        board_str += self.symb[None]
                board_str += "\n"
            outs.append(board_str)
        return outs
                        
    
    def get_player(self, state: torch.Tensor) -> torch.Tensor:
        summed = abs(state.sum((2,3)))
        return (1 * (summed[:,0] > summed[:,1])).to(dtype=int)[:, None, None, None]

    def get_player_vector(self, state: torch.Tensor) -> torch.Tensor:
        return torch.eye(2, dtype=int)[self.get_player(state)[:, 0, 0, 0]][:,:,None, None]



    def valid_action_filter(self, state: torch.Tensor) -> torch.Tensor:
        return state.sum((1)) == 0
    
    def get_random_action(self, state, max_tries=100) -> torch.Tensor:
        filter = self.valid_action_filter(state)
        tries = 0
        while (filter.count_nonzero(dim=(1,2)) <= 1).prod().item() != 1:                             # Almost always terminates after one step
            temp = torch.rand(state.size(0), 3, 3) * filter
            filter = temp == temp.max()
            tries += 1
            if tries >= max_tries:
                break
        return filter.int()
    
    def is_valid_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        return torch.prod(torch.prod(state.sum(1) + action <= 1, 1), 1)
    


    # Return shape (batch, 2, 1, 1)
    def swap_player(self, player_vector: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(player_vector, torch.tensor([[0,1],[1,0]], dtype=int), dims=([1], [0])).swapaxes(1, -1)
    
    def get_initial_state(self, batch_size=1) -> torch.Tensor:
        return torch.zeros((batch_size, 2, 3, 3))
    
    def action_str(self, action: torch.Tensor) -> list[str]:
        outs = []
        for b in range(action.shape[0]):
            flat_pos = action[b].flatten().max(0).indices.item()
            outs.append(f"position {flat_pos//3+1, flat_pos % 3+1}")
        return outs


    def is_valid_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        return (torch.prod(torch.prod(state.sum(1) + action <= 1, dim=1), dim=1) == 1)[:,None, None, None]

    # Reward is 1 for winning the game, -1 for losing, and 0 for a tie; awarded upon entering terminal state
    def transition(self, state, action):

        if state.shape[0] != action.shape[0]:
            raise Exception("Batch sizes must agree.")
        
        action = (self.is_terminal(state) == False)[:, 0, :, :] * action
        p = self.get_player(state)
        p_tensor = self.get_player_vector(state)

        new_state = state + (p_tensor * action[:,None,:,:]) * self.is_valid_action(state, action)

        winner = self.is_winner(new_state, p)
        full = self.is_full(new_state)

        new_state = (1 - 2 * (~self.is_terminal(state) & (winner | full))) * new_state

        reward = (winner.float() * p_tensor - winner * self.swap_player(p_tensor))[:,:,0,0]
        reward += ((self.is_valid_action(state, action) == False).float() * self.penalty * p_tensor)[:,:,0,0]

        return new_state, reward

    def is_winner(self, state, player):
        s = state[torch.arange(state.size(0)), player[:,0,0,0]]
        return ((s.sum(1) == 3).sum(1) + (s.sum(2) == 3).sum(1) + (s[:,0,0] + s[:,1,1] + s[:,2,2] == 3).int() + (s[:,0,2] + s[:,1,1] + s[:,2,0] == 3).int() > 0)[:,None,None,None]

    def get_random_state(self):
        return self.get_initial_state()

    def is_terminal(self, state: torch.Tensor) -> torch.Tensor:
        return ((state * -1).flatten(1,3).max(1).values > 0)[:, None, None, None]
    
    def is_full(self, state: torch.Tensor) -> torch.Tensor:
        return (abs(state.sum((1,2,3))) == 9)[:, None, None, None]
    






# Some prototyping the neural network
# Input tensors have shape (batch, 2, 7, 6)
class TTTNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(2, 32, (3,3), padding='same'),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),             # Dropout introduces randomness into the Q function.  Not sure if this is desirable.
            #nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3,3), padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (3,3), padding='same'),
            nn.LeakyReLU(),
            #nn.BatchNorm2d(64),
            #nn.Flatten(),
            #nn.Linear(128*3*3, 64*3*3),
            nn.Conv2d(128, 256, (3,3), padding='same'),
            nn.LeakyReLU(),
            #nn.Linear(64*3*3, 64),
            #nn.ReLU(),
            #nn.BatchNorm1d(64*7*6),
            #nn.Linear(64, 9), 
            nn.Conv2d(256, 1, (3,3), padding='same'),
            #nn.Unflatten(1, (3, 3))
        )
    
    # Output of the stack is shape (batch, 1, 3, 3), so we do a simple reshaping.
    def forward(self, x):
        return self.stack(x)[:,0]




if "test" in options:
    print("RUNNING TESTS:\n")

    verbose = True if "verbose" in options else False

    mdp = TTTTensorMDP()
    s = mdp.get_initial_state()
    print(mdp.board_str(s)[0])

    for i in range(12):
        print("play")
        a = mdp.get_random_action(s)
        print(a.tolist())
        t, r = mdp.transition(s, a)
        print(mdp.board_str(t)[0])
        print(r.tolist())
        s = t

    dqn = DQN(mdp, TTTNN, torch.nn.HuberLoss(), torch.optim.SGD, 100000)
    dqn.deep_learn(0.1, 0.5, 0.5, 50, 10, 16, 16, 16, 4, True)

