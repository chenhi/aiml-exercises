import torch, sys
import torch.nn as nn

from deepqlearn import TensorMDP
from nothanks import *


mdp = NoThanksMDP(num_players=5)
s = mdp.get_initial_state()
print(s, s.shape)
print(mdp.get_player_vector(s), mdp.get_player_vector(s).shape)
print(mdp.get_player(s), mdp.get_player(s).shape)
print(mdp.board_str(s)[0])
# print(s[0,-1,:])
# print(mdp.get_random_card(s).shape)
# print(mdp.valid_action_filter(s))
# print(mdp.get_random_action(s))
#t = mdp.next_player(s)
# print(mdp.board_str(t)[0])
# print(mdp.str_to_action('n'))
# print(mdp.str_to_action('y'))
s,r = mdp.transition(s, mdp.str_to_action('y'))
print(mdp.board_str(s)[0],r)
s,r = mdp.transition(s, mdp.str_to_action('n'))
print(mdp.board_str(s)[0],r)
s,r = mdp.transition(s, mdp.str_to_action('n'))
print(mdp.board_str(s)[0],r)
s,r = mdp.transition(s, mdp.str_to_action('y'))
print(mdp.board_str(s)[0],r)
s,r = mdp.transition(s, mdp.str_to_action('n'))
print(mdp.board_str(s)[0],r)
print(s)
print(mdp.get_current_player_cards(s))
print(torch.tensordot(mdp.get_current_player_cards(s), mdp.down_card, ([1],[0])))

s = mdp.get_initial_state(2)
print(s)

# state = s
# action = mdp.str_to_action('n')
# action *= (mdp.is_terminal(state) == False)
# print(action)
# reward = torch.zeros(state.size(0), mdp.num_players)
# # Those who say no thanks, give a token (only people with tokens can take actions)
# state[:,0:-1,1] -= (action[:,0] == 1.)[:,None] * (state[:,0:-1,0] == 1.).float()
#     # Put a token in the middle
# state[:,-1,1] += (action[:,0] == 1.) * (mdp.is_terminal(state) == False).float()
# reward -= (action[:,0] == 1.) * (state[:,0:-1,0] == 1.).float()
# print(mdp.board_str(state)[0])
# print(state)
# print(reward)

# state = s
# s[0,-1,1] = 100
# action = mdp.str_to_action('y')
# state += (action[:,1] == 1.).float()[:,None,None] * torch.tensordot(mdp.get_player_vector(state)[:,:,0], torch.cat([torch.zeros((state.size(0), 1)), state[:,-1,1:]], 1), dims=0).diagonal(dim1=0,dim2=2).swapaxes(0,2).swapaxes(1,2)
# state[:,-1,:] -= (action[:,1] == 1.).float()[:,None] * state[:,-1,:]
# print(mdp.board_str(state)[0])
# print(state)
# state += (mdp.num_cards_out(state) < mdp.num_played_cards) * mdp.get_random_card(state)
# print(mdp.board_str(state)[0])
# print(state[torch.arange(state.size(0)), mdp.get_player(state)[:,0,0].int(), 1] == 0)