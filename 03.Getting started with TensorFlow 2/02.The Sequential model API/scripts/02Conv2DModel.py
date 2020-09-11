import tensorflow as tf 
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Softmax, MaxPooling2D

# This code represents the parameters for conv2D layer (None represents the batch_size)
model = Sequential([
    Conv2D(16,(3,3),activation = 'relu',input_shape=(32,32,3)), # (None, 30,30,16)
    MaxPooling2D((3,3)),                                        # (None, 10,10,16)
    Flatten(),                                                  # (None, 1600)
    Dense(64, activation='relu'),                               # (None, 64)
    Dense(10, activation='softmax')                             # (None, 10) 
])

# If padding is same, no. of parameters become same as input size.
model = Sequential([
    Conv2D(16, kernel_size=(3,3), padding = "SAME", activation = 'relu',input_shape=(32,32,3)), # (None, 32,32,16)
    MaxPooling2D(pool_size=(3,3)),                                                              # (None, 10,10,16)
    Flatten(),                                                                                  # (None, 1600)
    Dense(64, activation='relu'),                                                               # (None, 64)
    Dense(10, activation='softmax')                                                             # (None, 10) 
])

# If kernel_size and pool_size is same, we can use shortcut as below
model = Sequential([
    Conv2D(16, kernel_size=3, padding = "SAME", activation = 'relu',input_shape=(32,32,3)), # (None, 32,32,16)
    MaxPooling2D(pool_size=3),                                                              # (None, 10,10,16)
    Flatten(),                                                                              # (None, 1600)
    Dense(64, activation='relu'),                                                           # (None, 64)
    Dense(10, activation='softmax')                                                         # (None, 10) 
])

model.summary()