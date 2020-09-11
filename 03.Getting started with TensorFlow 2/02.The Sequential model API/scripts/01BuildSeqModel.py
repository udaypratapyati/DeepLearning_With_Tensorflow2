import tensorflow as tf 

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Softmax

#model = Sequential()

# model.add(Dense(16, activation = 'relu', input_shape = (784,)))
# model.add(Dense(16, activation='softmax'))


model = Sequential([
        Flatten(input_shape = (28,28)), #(784, ) #Unrolling two dimensional data into one long vector before sending to dense layer
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer = 'Adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
)

print(model.loss)
print(model.metrics)
print(model.optimizer)
print(model.weights)