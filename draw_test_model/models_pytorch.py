import torch
from torch import nn
from torchvision.transforms import v2

import numpy as np
import numpy.random as rand
import math

########## AUXILLARY FUNCTIONS ##########

# Weight initializer (basically standard, copied from nn.Linear)
def initWeights(m):
    if isinstance(m, nn.Linear):
        # I guess there's not a huge difference between uniform vs. normal, and uniform probably easier to sample
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)
        #m.bias.data.fill_(0.01)                                # E.g. if I just want to set it to something.


########## MODEL DEFINITIONS ##########

class Dense2(nn.Module):
    def __init__(self):
        super().__init__()
        self.label = "dense2"
        self.flatten = nn.Flatten()
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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ConvBig(nn.Module):             # Input shape (1, 28, 28) or (BATCH SIZE, 1, 28, 28)
    def __init__(self):
        super().__init__()
        self.label = "convbig"
        # How to do initialization?
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(1, 128, (3,3), padding='same'),                    # Input channel 1, output 32, filter size (3,3)
            nn.MaxPool2d((2,2)),    # Image now 14x14
            nn.ReLU(),              # ReLU after MaxPool more efficient
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3,3), padding='same'),
            nn.MaxPool2d((2,2)),    # Image now 7x7
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.Flatten(),           # Default start_dim=1
            nn.Linear(7*7*128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        logits = self.convolution_stack(x)
        return logits



class ConvSkip2(nn.Module):             # Input shape (1, 28, 28) or (BATCH SIZE, 1, 28, 28)
    def __init__(self):
        super().__init__()
        self.label = "convskip"
        # How to do initialization?
        self.stack1 = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), padding='same'),                    # Input channel 1, output 32, filter size (3,3)
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3,3), padding='same'),                    # Input channel 1, output 32, filter size (3,3)
            nn.MaxPool2d((2,2)),    # Image now 14x14
            nn.ReLU(),              # ReLU after MaxPool more efficient
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
        )
        self.stack2 = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), padding='same'),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3,3), padding='same'),  
            nn.MaxPool2d((2,2)),    # Image now 7x7
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
        )

        self.stack3 = nn.Sequential(
            nn.Linear(7*7*64 + 14*14*64, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10)
        )

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x1 = self.stack1(x)
        x2 = self.stack2(x1)
        x12 = torch.cat([self.flatten(x1), self.flatten(x2)], dim=1)
        logits = self.stack3(x12)
        return logits


class ConvSkip(nn.Module):             # Input shape (1, 28, 28) or (BATCH SIZE, 1, 28, 28)
    def __init__(self):
        super().__init__()
        self.label = "convskip"
        # How to do initialization?
        self.stack1 = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), padding='same'),                    # Input channel 1, output 32, filter size (3,3)
            nn.MaxPool2d((2,2)),    # Image now 14x14
            nn.ReLU(),              # ReLU after MaxPool more efficient
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
        )
        self.stack2 = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), padding='same'),
            nn.MaxPool2d((2,2)),    # Image now 7x7
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
        )

        self.stack3 = nn.Sequential(
            nn.Linear(7*7*64 + 14*14*64, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10)
        )

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x1 = self.stack1(x)
        x2 = self.stack2(x1)
        x12 = torch.cat([self.flatten(x1), self.flatten(x2)], dim=1)
        logits = self.stack3(x12)
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

    def forward(self, x):
        logits = self.convolution_stack(x)
        return logits
    
    # The rest of this is not used or needed

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
    
# TODO not done yet
class RandomWiggle(v2.Transform):
    def __init__(self, zoom_probability, rotation_95) -> None:
        self.zoom_probability = zoom_probability
        # We use a truncated exponential distribution for the zoom amount, and will only ever zoom out.
        # This means probability of zoom is e^{-scale}, so the scale is -ln(zoom_probability)
        self.zoom_scale = -1 * math.log(self.zoom_probability)
        # The angle in degrees that 95% of zooms should be in (in either direction)
        self.rotation_95 = rotation_95
        self.rotation_scale = math.log(.05) / (-math.abs(rotation_95))

        nn.utils._log_api_usage_once(self)


    def __call__(self, pic):        
        # Zoom
        zoom = rand.exponential(scale = self.zoom_scale)
        if zoom > 1:
            pic = v2.functional.affine(scale=1./zoom)(x)
        
        # Rotate
        rot = rand.exponential(scale=self.rotation_scale) * rand.choice([-1.,1.])
        x = v2.functional.affine(degrees=rot)(x)        # Is this how to use this function?

        # Translate