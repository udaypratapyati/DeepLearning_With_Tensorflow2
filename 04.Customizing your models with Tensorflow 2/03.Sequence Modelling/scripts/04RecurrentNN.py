import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU

model = Sequential([
    Embedding(1000, 32, input_length=64), # (None, 64, 32)
    SimpleRNN(64, activation='tanh'),     # (None, 64)
    Dense(5, activation='softmax')        # (None, 5)
])

# This model begins with an embedding layer, which expects a sequence of integer tokens per example.
# Embedding layer output will be (None, 64, 32) as discussed in Embedding layer.
# RNN expects a 3 dimensional tensor input with batch_size, sequence length
# and number of features in each dimension.
# Output will be a two dimensional tensor with shape given by batch_size in first dimension
# and number of RNN units in the second dimension. which here is 64.

model2 = Sequential([
    Embedding(1000, 32),                  # (None, None, 32)
    SimpleRNN(64, activation='tanh'),     # (None, 64)
    Dense(5, activation='softmax')        # (None, 5)
])

# One of the strength of RNN is their ability to take flexible length sequences.
# So in above, we can omit input length argument in embedding layer.
# and that will enable network to take a batch of sequences of any length.
# That's possible because RNN is returning its hidden state at final time step.
# So it will just process incoming sequence whatever its length is and return
# final output we passed into Dense layer.

model3 = Sequential([
    Embedding(1000, 32),                  # (None, None, 32)
    LSTM(64, activation='tanh'),          # (None, 64)
    Dense(5, activation='softmax')        # (None, 5)
])

model4 = Sequential([
    Embedding(1000, 32),                  # (None, None, 32)
    GRU(64, activation='tanh'),           # (None, 64)
    Dense(5, activation='softmax')        # (None, 5)
])