import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

test_input = [
    [4,12,33,18],
    [63,23,54,30,19,3],
    [43,37,11,33,15]
]

test_input2 = [
    [[2,1], [3,3]],
    [[4,3], [2,4], [1,1]]
]

preprocessed_data = pad_sequences(test_input, 
                                padding = 'pre',
                                maxlen=5,
                                truncating='post',
                                value=-1)

preprocessed_data2 = pad_sequences(test_input2, padding='post')

print(preprocessed_data)
print(preprocessed_data2)

# [[ 0  0  4 12 33 18]
#  [63 23 54 30 19  3]
#  [ 0 43 37 11 33 15]]

# [[-1  4 12 33 18]
#  [63 23 54 30 19]
#  [43 37 11 33 15]]

# [[[2 1]
#   [3 3]
#   [0 0]]

#  [[4 3]
#   [2 4]
#   [1 1]]]

# pad_sequences require one argument list of lists.
# Function will return two dimensional numpy array output.
# First dimension is batch size or number of examples in test_input.
# padding is done at pre as per argument mentioned.
# We can set padding as post also.
# maxlen is another argument. First number is emitted from 6 to 5.
# Truncating can be done at post level also.
# Another argument is value=-1. Default padding value is zero.
# In first example, we had only one integer in the data per time step.

# If the data has number of features per time step. My test input is still list of lists.
# And each element in the sublist has two features per time step.
# If we preprocess this, we will have a 3 dimensional Numpy Array as output.

# When we pass this pre-processed batch of sequences into our model,
# we want them to ignore sections of these sequences that are padded.