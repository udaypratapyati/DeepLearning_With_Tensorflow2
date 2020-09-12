import tensorflow as tf 
from keras.layers import Embedding
import numpy as np 

embedding_layer = Embedding(1000, 32, input_length=64, mask_zero=True)
test_input = np.random.randint(1000, size=(16,64))

embedded_inputs = embedding_layer(test_input) # (16, 64, 32)
embedded_inputs._keras_mask

# Embedding allows the network to learn its own representation of each token in a sequence input
# the embedding layer takes in a tokenized sequence like the ones we've seen already.
# and will map each one of those separate tokens to a point in some high-dimensional embedding space.
# what the embedding layer does is to learn its own embedding from scratch,
# which is very specific to the particular task and dataset that your network is trained on.

# The embedding layer takes two required arguments.
# The first is the input dimension, which you might find easier to think of as the vocabulary size
# It's just the total number of unique tokens or words in the sequence data inputs
# The second argument is the embedding dimension.

# So each one of the 1,000 separate tokens that are in the inputs will be mapped into a 32-dimensional space.
# I've also selected the input length argument and setting that to 64.
# You may or may not have to set these arguments depending on the architecture of your network.

# In above example, we can take numpy array of size 16, 64 between 0 and 999
# First dimension is batch_size as usual.
# So think test_input as being batch of 16 sequence example.
# Each of which is length 64.

# If we call embedding layer on test input, this will embed each token into
# 32 dimensional embedding space.

# This means every token at every time step of every sequence is mapped to some point
# in 32 dimensional space.

# So the resulting tensor would be 16, 64, 32

# Embedding layer is also able to handle padded sequence inputs correctly.
# We can set mask_zero argument to be true.
# What this means is that the embedding layer will interpret any zeros that are in the input as padding values
# So the network will ignore them.
# Resulting tensor has keras mask attribute that captures where input sequences should be ignored.

