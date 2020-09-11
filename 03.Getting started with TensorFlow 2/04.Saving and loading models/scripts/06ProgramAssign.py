import tensorflow as tf 
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import numpy as np 
import pandas as pd 

def get_new_model(input_shape):
    """
    This function should build a Sequential model according to the above specification. Ensure the 
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    Your function should also compile the model with the Adam optimiser, sparse categorical cross
    entropy loss function, and a single accuracy metric.
    """
    model = Sequential([
        Conv2D(16,(3,3),padding="SAME",activation='relu',name='conv_1', input_shape=(input_shape)),
        Conv2D(8,(3,3),padding="SAME",activation='relu',name='conv_2'),
        MaxPooling2D((8,8),name='pool_1'),
        Flatten(name='flatten'),
        Dense(32,activation='relu',name='dense_1'),
        Dense(10,activation='softmax',name='dense_2')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

# checkpoint_path = 'checkpoints_every_epoch/checkpoint_{epoch:03d}'
# checkpoint_dir = os.path.dirname(checkpoint_path)
# !ls {checkpoint_dir}
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# latest