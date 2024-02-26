import torch
import torchvision
from torch import nn
from torchvision import datasets

import numpy as np


# Define classes for the models
class Dense2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.reset_parameters = initWeights                 # Initialization (does this actually work????  i.e. does it apply it to the stuff below>???)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            # batch normalization?
            nn.Linear(256, 256),
            nn.ReLU(),
            # batch noralization?
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )
    

    def rescale(self, x):
        return x / 255.
    
    def unrescale(self, x):
        return x * 255.    

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Conv2(nn.Module):             # Input should be a shape (28, 28) tensor?
    def __init__(self):
        super().__init__()
        # How to do initialization?
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding='same'),                    # Input channel 1, output 32, filter size (3,3)
            nn.ReLU(),
            nn.MaxPool2d((2,2)),    # Image now 14x14
            nn.Dropout(p=0.2),
            # batch norm?
            nn.Conv2d(32, 64, (3,3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),    # Image now 7x7
            nn.Dropout(p=0.2),
            # batch norm?
            nn.Flatten(),
            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # batch norm?
            nn.Linear(128, 10)
        )

    def rescale(self, x):
        return x / 255.
    
    def unrescale(self, x):
        return x * 255.  

    def addChannels(self, x, n=1):
        return torch.reshape(list(x.shape)[:-2] + [n] + list(x.shape)[-2:])
    
    def forward(self, x):
        logits = self.convolution_stack(x)
        return logits