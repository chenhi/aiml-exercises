import tensorflow as tf



########## AUXILLARY FUNCTIONS ##########

# Weight initailizer
def getVarScale():              # Call a function to generate a different seed each time
    return tf.keras.initializers.VarianceScaling(scale=0.001, mode='fan_in', distribution='normal')   



########## MODEL DEFINITIONS ##########

def Dense2(name="Dense2"):
    model = tf.keras.models.Sequential(name)
    model.add(tf.keras.Input(shape=(28,28)))
    model.add(tf.keras.layers.Rescaling(1./255))
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=getVarScale()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=getVarScale()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=getVarScale()))
    return model


def Conv2(name="conv2"):
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
    return model
    