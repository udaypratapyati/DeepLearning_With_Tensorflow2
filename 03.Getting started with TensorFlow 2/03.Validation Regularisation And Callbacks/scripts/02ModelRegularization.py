import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(64, activation = 'relu',
        kernel_regularizer = tf.keras.regularizers.l2(0.001)),
    Dense(1, activation='sigmoid')
])

# l2 requires one argument coefficient that multiplies with sum of squared weights in this layer
# Dense layer has weights and biases and weight matrix is called kernel

model.compile(optimizer='adadelta',
            loss='binary_crossentropy',
            metrics=['accuracy'])
# model.fit(inputs, targets, validation_split=0.25)

# Weight decay penalty term in automatically added to the loss function when we compile the model
# This penalizes the large weights in the first layer

model = Sequential([
    Dense(64, activation = 'relu',
        kernel_regularizer = tf.keras.regularizers.l1(0.005)),
    Dense(1, activation='sigmoid')
])

# We can also use l1 regularization
# This coefficient multiplies with sum of absolute weight values

model = Sequential([
    Dense(64, activation = 'relu',
        kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.005, l2=0.001),
        bias_regularizer = tf.keras.regularizers.l2(0.001)),
    Dense(1, activation='sigmoid')
])

# We can use both regularization as well as above.
# coefficients can be used independently.
# we can use bias regularizer also as above.

from keras.layers import Dropout

model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# We can use dropout layer also for regularization
# Argument is dropout rate
# Each weight connection between these two layers is set to zero with probability 0.5
# Also referred to as Bernouli Dropout | Weights are effectively multiplied by Bernoulli Random Variable
# Each of the weights are dropped out independently from one another
# Dropout is also applied independently across each element in the batch at training time

model.compile(
    optimizer='adadelta',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# model.fit(inputs, targets, validation_split=0.25)
# model.evaluate(val_inputs, val_targets)
# model.predict(test_inputs)

# When we are using dropouts, we typically have two different modes for how we run the network
# During training time, we randomly dropout the weights and this is training mode
# However when we are evaluating or making predictions from it, we stopped randomly dropping out the weights
# And this is testing mode.
# These two modes are handled behind the scenes by model.fit
# We can get more layer level control later in this course.