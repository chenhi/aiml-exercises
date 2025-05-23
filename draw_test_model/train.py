import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2

import tensorflow as tf

import matplotlib.pyplot as plt
import datetime

import models_pytorch as ptm
import models_tensorflow as tfm
from util import log, tf_summary_string

#################### TRAINING LIST AND OPTIONS ####################


# ptm means a PyTorch model, tfm means a TensorFlow/Keras model
# Note that PyTorch entries require a 'rate', not implemented for TensorFlow
trainList = [
#    {'data': 'digits', 'model': ptm.Dense2, 'opt': 'adam', 'epochs': 1, 'rate': 1e-3, 'save': True},
#     {'data': 'digits', 'model': ptm.Conv2, 'opt': 'adam', 'epochs': 50, 'rate': 1e-2, 'save': True},
#    {'data': 'digits', 'model': ptm.Conv3, 'opt': 'adam', 'epochs': 50, 'rate': 1e-2, 'save': True},
    {'data': 'digits', 'model': ptm.ConvSkip, 'opt': 'adam', 'epochs': 50, 'rate': 1e-3, 'save': True},
#    {'data': 'fashion', 'model': ptm.Dense2, 'opt': 'adam', 'epochs': 5, 'save': True},
#    {'data': 'fashion', 'model': tfm.Conv2, 'opt': 'adam', 'epochs': 2, 'save': True},
    ]


##### PyTorch #####

# Loss function
loss_fn, loss_fn_name = nn.CrossEntropyLoss(), "nll"                   # Inputs are logits (i.e. pre-softmax)
#loss_fn, loss_fn_name = torch.nn.HingeEmbeddingLoss(margin=1.), "hinge"        

# Batch size (i.e. how often back-propagation happens)
batch_size = 64

# Shuffling
shuffle = True

##### TensorFlow #####

# Loss function
#tf_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="nll")              # Use without softmax
tf_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(name="nll")                                 # Use with softmax
#tf_loss_fn = tf.keras.losses.CategoricalHinge(name="hinge")






#################### DATA AND FEATURE TRANSFORMATIONS ####################

# PyTorch
# ToTensor transforms to tensor of the form (BATCH SIZE, 1, 28, 28) with values in range [0, 1]
digitsTrainData = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.RandomAffine(30, translate=(.2,.2), scale=(.6, 1.), shear=(-20, 20, -20, 20)), torchvision.transforms.ToTensor()]),
    #transform=torchvision.transforms.ToTensor(),
)
digitsTestData = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
digitsTrainLoader = torch.utils.data.DataLoader(digitsTrainData, batch_size=batch_size, shuffle=shuffle)
digitsTestLoader = torch.utils.data.DataLoader(digitsTestData, batch_size=batch_size)


fashionTrainData = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.RandomAffine(10, translate=(.3,.3), scale=(.75, 1.)), torchvision.transforms.ToTensor()]),
)
fashionTestData = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),            # Check maybe this already has the extra channel???
)
fashionTrainLoader = torch.utils.data.DataLoader(fashionTrainData, batch_size=batch_size, shuffle=shuffle)
fashionTestLoader = torch.utils.data.DataLoader(fashionTestData, batch_size=batch_size)


# TensorFlow 
tf_digitsTrainData, tf_digitsTestData = tf.keras.datasets.mnist.load_data()
tf_fashionTrainData, tf_fashionTestData = tf.keras.datasets.fashion_mnist.load_data()






#################### TRAIN, TEST, SAVE ####################

# Get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Now actually train the models and save the results
for d in trainList:
    # Load the specified model
    m = d['model']()
    
    # First, figure out if it's PyTorch or TensorFlow (or unrecognized).
    if isinstance(m, nn.Module):
        isTorch = True
    elif isinstance(m, tf.keras.Model):
        isTorch = False
    else:
        print("ERROR: Unrecognized model format (not PyTorch nor TensorFlow).")
        print(m)
        print("Skipping entry in training list.")
        continue
    
    # Start the log
    logtext = ""
    logtext += log(f"PyTorch version {torch.__version__}") if isTorch else log(f"TensorFlow version {tf.__version__}")
    logtext += log(f"Using {device} device")


    # Load the relevant data
    if d['data'] == "digits":
        if isTorch:
            train, test = digitsTrainLoader, digitsTestLoader
        else:
            train, test = tf_digitsTrainData, tf_digitsTestData
    elif d['data'] == "fashion":
        if isTorch:
            train, test = fashionTrainLoader, fashionTestLoader
        else:
            train, test = tf_fashionTrainData, tf_fashionTestData
    else:
        print("ERROR: Data", d['data'], "not recognized.  Skipping entry in training list.")
        continue

    # Some extra inputs for PyTorch
    if isTorch:
        logtext += log(f"Training dataset and transforms: {train.dataset}")
        logtext += log(f"Validation dataset and transforms: {test.dataset}")
        logtext += log(f"Batch size: {batch_size}")
        logtext += log(f"Shuffle: {shuffle}")

    # Log model information and make the name
    logtext += log("Loaded model:")
    if isTorch:
        logtext += log(str(m))
        name = f"{d['data']}.{m.label}.{loss_fn_name}.{d['opt']}.{d['epochs']}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        # For TensorFlow, the summary will come later
        name = f"{d['data']}.{m.name}.{tf_loss_fn.name}.{d['opt']}.{d['epochs']}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"


    # Start training
    if isTorch:
 
         # Define optimizer on the model (m.parameters() is iterator over parameters)
        if d['opt'] == 'sgd':
            optimizer, optimizer_name = torch.optim.SGD(m.parameters(), lr=d['rate'], dampening=0, momentum=0, weight_decay=0)                    # lr = learning rate
        elif d['opt'] == 'adam':
            optimizer, optimizer_name = torch.optim.Adam(m.parameters(), lr = d['rate']), "adam"
        else:
            print("Optimizer", d['opt'], "not found!  Exiting.")
            exit()
        logtext += log("Loaded optimizer:")
        logtext += log(str(optimizer))

        # Record accuracy and loss for testing and training in each epoch
        trainLoss = []
        trainAcc = []
        testLoss = []
        testAcc = []
        for ep in range(0, d['epochs']):
            logtext += log("Epoch " + str(ep + 1) + "/" + str(d['epochs']) + ": -------------------------------------")

            logtext += log("Training: " + name)
            m.train()           # Sets in training mode, e.g. vs. m.eval()
            
            train_loss = 0.
            train_correct = 0.
            for curBatch, (X, y) in enumerate(train):
                X, y = X.to(device), y.to(device)

                pred = m(X)                     # Output is a tensor with gradient functions (and knows about m)
                loss = loss_fn(pred, y)         # Also a tensor with gradient functions

                loss.backward()                 # Computes the gradients (now stored in the tensors in the model m)
                optimizer.step()                # Performs an optimization step for the tensors in the model m
                optimizer.zero_grad()           # Zeroes out computed gradients in m

                train_loss += loss.item()
                train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                trainsize = len(train.dataset)

                if curBatch % 100 == 0:         # Print some information every 100 batches
                    loss, current = loss.item(), (curBatch + 1) * batch_size
                    logtext += log(f"loss: {loss:>7f}  [{current:>5d}/{trainsize:>5d}]")

            # Log the training accuracy and loss
            train_loss /= len(train)
            train_acc = train_correct / trainsize
            trainLoss.append(train_loss)
            trainAcc.append(train_acc)
            
            logtext += log("Testing: " + name)
            m.eval()
            test_loss, correct = 0.,0
            with torch.no_grad():
                for X, y in test:
                    X, y = X.to(device), y.to(device)
                    pred = m(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()     # argmax returns a vector whose nth entry is the maximum index in the nth row
                
                testsize = len(test.dataset)
                test_loss /= len(test)
                test_acc = correct / testsize
                logtext += log(f"Test Error: \n Accuracy: {(test_acc*100):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            testLoss.append(test_loss)
            testAcc.append(test_acc)

        logtext += log("Done training and testing!")
        
        # Log the accuracy and loss across all epochs
        logtext += log("Training accuracy: " + str(trainAcc))
        logtext += log("Training loss: " + str(trainLoss))
        logtext += log("Validation accuracy: " + str(testAcc))
        logtext += log("Validation loss: " + str(testLoss))

    else:
        
        logtext += log(f"Training: {name}")
        m.compile(optimizer=d['opt'], loss=tf_loss_fn, metrics=['accuracy'])
        history = m.fit(x=train[0], y=train[1], validation_data=test, epochs=d['epochs'])
        results = m.evaluate(x=test[0], y=test[1], verbose=2)
        
        logtext += log("Done training!")
        logtext += log(results)
        logtext += log(history)
        logtext += log(tf_summary_string(m))


    # Plot some graphs
    if isTorch:
        epochs_range = range(d['epochs'])
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, trainAcc, label='Training Accuracy')
        plt.plot(epochs_range, testAcc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, trainLoss, label='Training Loss')
        plt.plot(epochs_range, testLoss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        #plt.show()        
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(d['epochs'])

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        #plt.show()

    # Save the model, logs and graph
    if d['save'] == True:
        #torch.save(m.state_dict(), name + ".pth")
        mpath = "models/" + name + (".pth" if isTorch else ".keras")
        plotpath = mpath + ".png"
        logpath = mpath + ".log"

        # Save graph
        plt.savefig(plotpath)
        logtext += log(f"Saved accuracy/loss plot to {plotpath}")
        
        # Save model
        if isTorch:
            model_scripted = torch.jit.script(m)
            model_scripted.save(mpath)
        else:
            m.save(mpath)
        logtext += log(f"Saved model to {mpath}")

        # Save logs
        with open(logpath, "w") as f:
            logtext += log(f"Saved logs to {logpath}")
            f.write(logtext)
        #m.summary(print_fn=lambda x: f.write(x + '\n'))