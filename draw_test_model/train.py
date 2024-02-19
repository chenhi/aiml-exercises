import tensorflow as tf

print("TensorFlow version: ", tf.__version__)

####################################################################

#class GreyscaleToColor(tf.keras.layers.Layer):
#    trainable = False




#####################################################################

# MNIST data set for digits and fashion
mnist = tf.keras.datasets.mnist
mnistf = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


(x_trainf, y_trainf), (x_testf, y_testf) = mnistf.load_data()


print("MNIST training data", len(x_train), "entries")
print("MNIST test data", len(x_test), "entries")
print("Input shape", x_train[0].shape)
print("Output shape", y_train[0].shape)

print("MNIST Fashion training data", len(x_trainf), "entries")
print("MNIST Fashion test data", len(x_testf), "entries")
print("Input shape", x_trainf[0].shape)
print("Output shape", y_trainf[0].shape)


x_train, x_test = x_train/255.0, x_test/255.0
x_trainf, x_testf = x_trainf/255.0, x_testf/255.0

# Digits models
digitsModels = []

model = tf.keras.models.Sequential(name="digits.dense128")
model.add(tf.keras.Input(shape=(28,28)))
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))

digitsModels.append(model)

model = tf.keras.models.Sequential(name="digits.conv32x64")
model.add(tf.keras.Input(shape=(28,28)))
#model.add(tf.keras.layers.Lambda(lambda x: x[...,None] * tf.ones(tf.concat([tf.shape(x), [3]], axis=0),dtype=x.dtype)))       # Tensors up with a vector of 3 1's
model.add(tf.keras.layers.Lambda(lambda x: x[...,None]))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10))

digitsModels.append(model)

# Fashion model
fashionModels = []

model = tf.keras.models.Sequential(name="fashion.dense128")
model.add(tf.keras.Input(shape=(28,28)))
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))

fashionModels.append(model)


model = tf.keras.models.Sequential(name="fashion.conv32x64")
model.add(tf.keras.Input(shape=(28,28)))
#model.add(tf.keras.layers.Lambda(lambda x: x[...,None] * tf.ones(tf.concat([tf.shape(x), [3]], axis=0),dtype=x.dtype)))       # Tensors up with a vector of 3 1's
model.add(tf.keras.layers.Lambda(lambda x: x[...,None]))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10))

fashionModels.append(model)


# Run the digits model on a the first entry just to mess around
#print("The first entry is", x_train[:1])
#print("The first entry is a", y_train[:1])
#badpredict = model(x_train[:1]).numpy()
#print("Before training, it is predicted to be", badpredict)
#print("Probability that the first entry is a given digit", tf.nn.softmax(badpredict).numpy())

#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)			# The logit function is the inverse to the logistic function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#print("Loss", loss_fn(y_train[:1], badpredict).numpy())
#print("-log(1/10) =", -1.0 * tf.math.log(1/10).numpy())


# Now actually train the models

for x in digitsModels:
    print("Training:", x.name)
    x.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    x.fit(x_train, y_train, epochs=5)
    x.evaluate(x_test, y_test, verbose=2)

for x in fashionModels:
    print("Training:", x.name)
    x.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    x.fit(x_trainf, y_trainf, epochs=5)
    x.evaluate(x_testf, y_testf, verbose=2)


# Add a layer converting the predictions to probabilities, then save the models
for x in digitsModels + fashionModels:
    probmodel = tf.keras.models.Sequential()
    probmodel.add(x)                            # Inherits the weights already chosen from the previous model
    probmodel.add(tf.keras.layers.Softmax())
    probmodel.save(x.name + ".keras")