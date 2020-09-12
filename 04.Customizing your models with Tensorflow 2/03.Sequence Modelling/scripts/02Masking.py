import tensorflow as tf 

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Masking
import numpy as np 

test_input = [
    [4,12,33,18],
    [63,23,54,30,19,3],
    [43,37,11,33,15]
]

preprocessed_data = pad_sequences(test_input, padding='post')

masking_layer = Masking(mask_value=0)
preprocessed_data = preprocessed_data[...,np.newaxis] # batch_size, seq_length, features
masked_input = masking_layer(preprocessed_data)
masked_input._keras_mask

print(masked_input)


# Create a masking instance layer called as masking_layer and tell the value used.
# It takes three dimensional input as above.
# Our data is just a single integer per time step in third dimension.
# So feature dimension is one in this case.
# However our test input array is just two dimensional.
# So it does not explicitly contain the feature dimension.
# So we are adding dummy feature dimension to process our data.
# Then we are calling this layer on our processed data input.
# Resulting tensor has extra attribute called keras_mask