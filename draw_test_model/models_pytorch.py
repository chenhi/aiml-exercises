import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand

import datetime

# Weight initializer
def initWeights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)                # I guess there's not a huge difference between uniform vs. normal, and uniform probably easier to sample
        if self.bias is not None:                               # Set the linear bias (just copied it from default nn.Linear code but can tweak it here)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        #m.bias.data.fill_(0.01)                                # E.g. if I just want to set it to something.



class Dense2(nn.Module):
    def __init__(self):
        super().__init__()
        self.label = "dense2"
        self.flatten = nn.Flatten()
        self.reset_parameters = initWeights                 # Initialization (does this actually work????  i.e. does it apply it to the stuff below>???)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
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



class Conv3(nn.Module):             # Input shape (1, 28, 28) or (BATCH SIZE, 1, 28, 28)
    def __init__(self):
        super().__init__()
        self.label = "conv3"
        # How to do initialization?
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding='same'),                    # Input channel 1, output 32, filter size (3,3)
            nn.MaxPool2d((2,2)),    # Image now 14x14
            nn.ReLU(),              # ReLU after MaxPool more efficient
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3,3), padding='same'),
            nn.MaxPool2d((2,2)),    # Image now 7x7
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.Flatten(),           # Default start_dim=1
            nn.Linear(7*7*64, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        logits = self.convolution_stack(x)
        return logits



class Conv2(nn.Module):             # Input shape (1, 28, 28) or (BATCH SIZE, 1, 28, 28)
    def __init__(self):
        super().__init__()
        self.label = "conv2"
        # How to do initialization?
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding='same'),                    # Input channel 1, output 32, filter size (3,3)
            nn.MaxPool2d((2,2)),    # Image now 14x14
            nn.ReLU(),              # ReLU after MaxPool more efficient
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3,3), padding='same'),
            nn.MaxPool2d((2,2)),    # Image now 7x7
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.Flatten(),           # Default start_dim=1
            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10)
        )

    def rescale(self, x):
        return x / 255.
    
    def unrescale(self, x):
        return x * 255.  

    def addChannels(self, x, n=1):
        return torch.reshape(list(x.shape)[:-2] + [n] + list(x.shape)[-2:])
    
    def randomWiggle(self, x):
        zoom = rand.exponential(scale=1.0)                                              # Probability of no zoom is 1 - e^(-scale) 
        if zoom > 1.:
            x = v2.functional.affine(scale=1./zoom)(x)                                          # Maybe select the zoom from some distribution?
        rot = rand.exponential(scale=0.3) * np.sign(rand.uniform(-1, 1))                # Want 95% of zooms to between [-10, 10], so want 1 - e^(-10 * scale) roughly 95%
        x_trans = rand.exponential(scale=max(zoom, 1.)) (np.sign(rand.uniform(-1,1)))     # Depends on zoom.  With no zoom, 95% of translates up to 3. 
        y_trans = rand.exponential(scale=max(zoom, 1.)) (np.sign(rand.uniform(-1,1)))
        x = v2.functional.affine(degrees=rot, translate=[x_trans, y_trans])(x)
        return x 

    def forward(self, x):
        logits = self.convolution_stack(x)
        return logits