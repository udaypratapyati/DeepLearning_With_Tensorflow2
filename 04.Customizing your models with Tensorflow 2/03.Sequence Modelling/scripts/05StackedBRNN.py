# With a recurrent neural network layers that we've used so far. 
# The layers have only returned the output at the final time step.
# Sometimes what you'd like is for a recurrent layer to return an output at every time step in the sequence. 
# These outputs can then be used for the final model predictions 
# or could be used as an input for another recurrent layer further downstream.

import tensorflow as tf 
from keras.layers import Input, Masking, LSTM, Dense, Bidirectional
from keras.models import Model

inputs = Input(shape=(None,10))                 # (None, None, 10)
h = Masking(mask_value=0)(inputs)               # (None, None, 10)
h = LSTM(64)(h)                                 # (None, 64)
outputs = Dense(5, activation='softmax')(h)     # (None, 5)

model = Model(inputs, outputs)

# Input layer to have shape None, 10
# Which means we have a flexible sequence length and there are 10 features per time step.
# Model is expecting 3 dimensional input with batch size, sequence length and features.
# Masking layer will ensure zero padding ignoring in neural network
# Masking layer does not change the shape of the incoming tensor at all.
# LSTM only emits the output at final time step and has 64 units.
# SO Output is None for batch size and 64 is number of units of features.

inputs = Input(shape=(None,10))                 # (None, None, 10)
h = Masking(mask_value=0)(inputs)               # (None, None, 10)
h = LSTM(32, return_sequences=True)(h)          # (None, None, 32)
h = LSTM(64)(h)                                 # (None, 64)
outputs = Dense(5, activation='softmax')(h)     # (None, 5)

model = Model(inputs, outputs)

# Each RNN Layer have optional keyword argument return sequences.
# This is false by default
# This is why layer only returns output at final time step.
# But if this option is set true, then layer will return output at each time step.
# Return sequence true means output retains the sequence dimension,
# which can be passed into another recurrent layer for further processing.

inputs = Input(shape=(None,10))                         # (None, None, 10)
h = Masking(mask_value=0)(inputs)                       # (None, None, 10)
h = Bidirectional(LSTM(32, return_sequences=True))(h)   # (None, None, 64)
h = Bidirectional(LSTM(64))(h)                          # (None, 128)
outputs = Dense(5, activation='softmax')(h)             # (None, 5)

model = Model(inputs, outputs)

# We can create a bidirectional layer by using bidirectional wrapper
# and calling it on regular recurrent layer.
# We need to take LSTM layer and pass it into bidirectional wrapper.
# In this, return sequences is true so BRNN will also return sequences.

# we can think of there as being two LSTM networks, one running in forwards time 
# and one running in backwards time and the final output of each of 
# these networks is concatenated to make a tensor with shape none by 128.

# We can change the behavior of the bidirectional wrapper.
# By default, this wrapper concatenates the outputs of the forward and backward RNNs.
# We can change this with merge_mode option. The default value is concat.
# But if we set it to sum, then forward and backward RNN outputs will be added together
# instead of concatenated.

# Tensor shape of the output will change and set to None, None, 32

inputs = Input(shape=(None,10))                         # (None, None, 10)
h = Masking(mask_value=0)(inputs)                       # (None, None, 10)
h = Bidirectional(LSTM(32, return_sequences=True),
                    merge_mode='sum')(h)                # (None, None, 32)
h = Bidirectional(LSTM(64))(h)                          # (None, 128)
outputs = Dense(5, activation='softmax')(h)             # (None, 5)

model = Model(inputs, outputs)