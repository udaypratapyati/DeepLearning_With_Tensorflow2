# If datasets are bigger, they will not fit into memory
# A generator is function that returns an object that you can iterate over
# and it yields a series of vaalues but doesn't store those values in memory.
# Instead it saves its own internal state and each time we iterate the generator.
# It yields the next value in series.
# So in this way, we can use generators to feed data into model when data doesn't fit in memory.

# Simple example of python generator
def text_file_reader(filepath):
    with open(filepath, 'r') as f:
        for row in f:
            yield row

text_datagen = text_file_reader('data_file.txt')

next(text_datagen) # 'A Line of text\n'
next(text_datagen) # 'Another line of text\n'

# It takes file path as an argument.
# It opens the file and iterates through the rows of the file.
# But instead of returning a row, it uses the yield statement.     
# It returns a text_datagen generator object, I can now iterate over the text_datagen object.
# Generator doesn't hold every line of file in memory but reads one line and yields that line
# Each time its iterated over. 
# We can manually step one iteration of generator by calling next on the generator

import numpy as np 

def get_data(batch_size):
    while True:
        y_train = np.random.choice([0,1], (batch_size, 1))
        x_train = np.random.randn(batch_size, 1) + (2 * y_train - 1)
        yield x_train, y_train

datagen = get_data(32)

x, y = next(datagen)

# We can also make generators that generate an infinite series of values.
# Above function takes batch size argument and returns a generator that yields
# batch of training inputs and outputs.
# Notice that each output is a binary value, either zero or one 
# and each input is a real value sampled from a Gaussian distribution.
# datagen object will be created as generator and return tuple of batch of inputs and outputs with batch size 32
# Now we have generator that generates batches of data each time we iterate over it.
# As before, we could iterate over this generator by calling next on it.

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(1, activation='sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer='sgd')

model.fit_generator(datagen, steps_per_epoch=1000, epochs=10)

# model.evaluate_generator(datagen_eval, steps=100)

# model.predict(datagen_test, steps=100)

# We are using model.fit_generator to feed data to the model.
# First argument is generator object is passed we just created.
# steps_per_epoch means after 1000 iterations, it should count as 1 epoch.
# Generator is yielding infinite series of data batches, and Total epochs = 10.

for _ in range(10000):
    x_train, y_train = next(datagen)
    model.train_on_batch(x_train, y_train)

# train_on_batch method just performs one optimizer update for single batch of training data.
# So we are specifying the number of iterations for entire training.
# And at each step we get a batch of training data by calling next on our data generator.
# Then we just feed that batch of training data into the model.train_on_batch method.
# This is slightly lower level way of handling model training using data generators.

