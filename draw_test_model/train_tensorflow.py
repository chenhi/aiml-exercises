# WARNING: This script is mostly deprecated.  I've been using PyTorch more.

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow version: ", tf.__version__)

#####################################################################
# WHAT TO TRAIN

# Format: the dataset, the model, the number of epochs, save model/results
whatToTrain = [   
#    ("digits", "dense2", 9, True),
    ("digits", "conv2", 100, True),
#    ("fashion", "dense2", 9, True),
#    ("fashion", "conv2", 5, True),
    ]

#####################################################################
# SOME UNIFORM OPTIONS

# Some standard stuff
def getVarScale():              # Call a function to generate a different seed each time
    return tf.keras.initializers.VarianceScaling(scale=0.001, mode='fan_in', distribution='normal')     

                                                                                                    # Sparse means expects one-hot outputs
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="nll")              # Use without softmax
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(name="nll")                                 # Use with softmax
#loss_fn = tf.keras.losses.CategoricalHinge(name="hinge")

#####################################################################

# Storage
models = []
history = []
results = []


# Define the models specified
for d in whatToTrain:
    name = d[0] + "." + d[1] + "." + loss_fn.name + "." + str(d[2])
    if d[1] == "dense2":
        model = tf.keras.models.Sequential(name=name)
        model.add(tf.keras.Input(shape=(28,28)))
        model.add(tf.keras.layers.Rescaling(1./255))
        model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=getVarScale()))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=getVarScale()))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=getVarScale()))
        models.append(model)
    elif d[1] == "conv2":
        model = tf.keras.models.Sequential(name=name)
        model.add(tf.keras.Input(shape=(28,28)))
        model.add(tf.keras.layers.Rescaling(1./255))
        #model.add(tf.keras.layers.RandomRotation((-.1, .1), fill_mode="constant", fill_value=0))
        #model.add(tf.keras.layers.RandomZoom((0., 0.5), fill_mode="constant", fill_value=0))
        #model.add(tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0))
        #model.add(tf.keras.layers.Lambda(lambda x: x[...,None] * tf.ones(tf.concat([tf.shape(x), [3]], axis=0),dtype=x.dtype)))        # Tensors up to add RGB channel
        #model.add(tf.keras.layers.Lambda(lambda x: x[...,None]))                                                                       # Actually it was enough to just add a 1 dimensional channel
        model.add(tf.keras.layers.Reshape((28, 28, 1)))                                                                                 # This is the simpler way to do it
        model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer=getVarScale()))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer=getVarScale()))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),  kernel_initializer=getVarScale()))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        #model.add(tf.keras.layers.Dense(10, activation='relu', kernel_initializer=getVarScale()))
        model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer=getVarScale()))
        models.append(model)
    else:
        print("Model", d[1], "not found, exiting.")
        exit()



# Get training and test data
digitsTrainData, digitsTestData = tf.keras.datasets.mnist.load_data()
fashionTrainData, fashionTestData = tf.keras.datasets.fashion_mnist.load_data()

# Now actually train the models and save the results
for i in range(0, len(whatToTrain)):
    m = models[i]
    epochs = whatToTrain[i][2]
    if whatToTrain[i][0] == "digits":
        train, test = digitsTrainData, digitsTestData
    elif whatToTrain[i][0] == "fashion":
        train, test = fashionTrainData, fashionTestData
    else:
        print("Data", whatToTrain[i][0], "not found!  Exiting.")
        exit()
    
    print("Training:", m.name)
    m.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    history.append(m.fit(x=train[0], y=train[1], validation_data=test, epochs=epochs))
    results.append(m.evaluate(x=test[0], y=test[1], verbose=2))
    
    print(results[i])
    print(history[i].history)

    acc = history[i].history['accuracy']
    val_acc = history[i].history['val_accuracy']

    loss = history[i].history['loss']
    val_loss = history[i].history['val_loss']

    epochs_range = range(epochs)

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
    plt.show()

    if whatToTrain[i][3] == True:
        plt.savefig("models/" + name + ".keras.png")
        m.save("models/" + m.name + ".keras")
        with open("models/" + m.name + ".log", "w") as f:
            models[i].summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("\n\n\n" + str(results[i]) + "\n\n\n" + str(history[i].history))
            f.close()

