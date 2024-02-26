import torch
import torchvision
from torch import nn
from torchvision import datasets

import matplotlib.pyplot as plt
import numpy as np

from models_pytorch import Conv2, Dense2

#####################################################################
# Basic logger

def log(l: str) -> str:
    print(l)
    return l + "\n"


#####################################################################
# WHAT TO TRAIN

# Format: the dataset, the model, the number of epochs, save model/results
trainList = [   
#    ("digits", "dense2", 9, True),
    {'data': "digits", 'model': "conv2", 'opt': 'adam', 'epochs': 1, 'save': True},
#    ("fashion", "dense2", 9, True),
#    ("fashion", "conv2", 5, True),
    ]

#####################################################################
# SOME UNIFORM OPTIONS

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Some standard stuff

# Weight initializer
def initWeights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)                # I guess there's not a huge difference between uniform vs. normal, and uniform probably easier to sample
        if self.bias is not None:                               # Set the linear bias (just copied it from default nn.Linear code but can tweak it here)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        #m.bias.data.fill_(0.01)                                # E.g. if I just want to set it to something.

# Loss function
loss_fn, loss_fn_name = nn.CrossEntropyLoss(), "nll"                   # Inputs are logits (i.e. pre-softmax)
#loss_fn, loss_fn_name = torch.nn.HingeEmbeddingLoss(margin=1.), "hinge"        

# Batch size (i.e. how often back-propagation happens)
batch_size = 64

#####################################################################


# Storage
history = []
results = []





# Get training and test data
# Note: data comes in tensor of the form (BATCH SIZE, 1, 28, 28) and is already scaled

digitsTrainData = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
digitsTestData = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
digitsTrainLoader = torch.utils.data.DataLoader(digitsTrainData, batch_size=batch_size)
digitsTestLoader = torch.utils.data.DataLoader(digitsTestData, batch_size=batch_size)


fashionTrainData = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
fashionTestData = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),            # Check maybe this already has the extra channel???
)
fashionTrainLoader = torch.utils.data.DataLoader(fashionTrainData, batch_size=batch_size)
fashionTestLoader = torch.utils.data.DataLoader(fashionTestData, batch_size=batch_size)





# Now actually train the models and save the results
for d in trainList:
    logtext = ""
    logtext += log(f"Using {device} device")

    # Load the relevant data
    if d['data'] == "digits":
        train, test = digitsTrainLoader, digitsTestLoader
        logtext += log("Using MNIST digits data")
    elif d['data'] == "fashion":
        train, test = fashionTrainLoader, fashionTestLoader
        logtext += log("Using MNIST fashion data")
    else:
        print("Data", d['data'], "not found!  Exiting.")
        exit()
    trainsize = len(train.dataset)
    testsize = len(test.dataset)
    logtext += log("Training dataset size " + str(trainsize))
    logtext += log("Test dataset size " + str(testsize))

    # Load the specified model
    if d['model'] == "dense2":
        m = Dense2()
    elif d['model'] == "conv2":
        m = Conv2()
    else:
        print("Model", d['model'], "not found!  Exiting.")
        exit()
    logtext += log("Loaded model:")
    logtext += log(str(m))

    # Make the name
    name = d['data'] + "." + d['model'] + "." + loss_fn_name + "." + d['opt'] + "." + str(d['epochs'])

    # Define optimizer on the model (m.parameters() is iterator over parameters)
    if d['opt'] == 'sgd':
        optimizer = torch.optim.SGD(m.parameters(), lr=1e-3, dampening=0, momentum=0, weight_decay=0)                    # lr = learning rate
    elif d['opt'] == 'adam':
        optimizer, optimizer_name = torch.optim.Adam(m.parameters(), lr = 1e-3), "adam"
    else:
        print("Optimizer", d['opt'], "not found!  Exiting.")
        exit()
    logtext += log("Loaded optimizer:")
    logtext += log(str(optimizer))


    for ep in range(0, d['epochs']):
        logtext += log("Epoch " + str(ep + 1) + "/" + str(d['epochs']) + ": -------------------------------------")

        logtext += log("Training: " + name)
        m.train()           # Sets in training mode, e.g. vs. m.eval()
        for curBatch, (X, y) in enumerate(train):
            X, y = X.to(device), y.to(device)

            pred = m(X)                     # Output is a tensor with gradient functions (and knows about m)
            loss = loss_fn(pred, y)         # Also a tensor with gradient functions

            loss.backward()                 # Computes the gradients (now stored in the tensors in the model m)
            optimizer.step()                # Performs an optimization step for the tensors in the model m
            optimizer.zero_grad()           # Zeroes out computed gradients in m

            if curBatch % 100 == 0:         # Print some information every 100 batches
                loss, current = loss.item(), (curBatch + 1) * batch_size
                logtext += log(f"loss: {loss:>7f}  [{current:>5d}/{trainsize:>5d}]")
        
        logtext += log("Testing: " + name)
        m.eval()
        test_loss, correct = 0,0
        with torch.no_grad():
            for X, y in test:
                X, y = X.to(device), y.to(device)
                pred = m(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()     # argmax returns a vector whose nth entry is the maximum index in the nth row
            test_loss /= len(test)
            correct /= testsize
            logtext += log(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    logtext += log("Done training and testing!")

    if d['save'] == True:
        torch.save(m.state_dict(), name + ".pth")
        logtext += log("Saved PyTorch model to " + name + ".pth")
        with open(name + ".log", "w") as f:
            f.write(logtext)