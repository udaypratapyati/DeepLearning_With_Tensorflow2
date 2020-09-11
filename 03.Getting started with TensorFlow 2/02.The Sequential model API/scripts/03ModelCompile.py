import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Softmax

model = Sequential([
    Dense(64, activation='elu', input_shape = (32,)), #exponential linear unit activation
    Dense(1, activation='sigmoid') # 'relu', 'linear', 'tanh'
])

model.compile(
    optimizer = 'sgd', # 'adam', 'rmsprop', 'adadelta'
    loss = 'binary_crossentropy', # 'mean_squared_error', 'categorical_crossentropy'
    metrics=['accuracy', 'mae']
)

#Another way for compiling
# We are using tf.keras.optimizers module here
# Why we are using it so that we can have better control over parameters
model.compile(
    optimizer = tf.keras.optimizers.SGD(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.MeanAbsoluteError()]
)

model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7), tf.keras.metrics.MeanAbsoluteError()]
)