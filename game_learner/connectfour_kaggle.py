from qlearn import MDP
from deepqlearn import *
import numpy as np
import sys, warnings
import torch

from connectfour_tensor import *


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def agent(observation, configuration):
    # Number of Columns on the Board.
    columns = configuration.columns
    # Number of Rows on the Board.
    rows = configuration.rows
    # Number of Checkers "in a row" needed to win.
    inarow = configuration.inarow

    game = C4TensorMDP(rows, columns, inarow, device)

    # The current serialized Board (rows x columns).
    board = observation.board

    

    

    # Which player the agent is playing as (1 or 2).
    mark = observation.mark
    # Isn't this redundant?

    # Return which column to drop a checker (action).
    return 0
