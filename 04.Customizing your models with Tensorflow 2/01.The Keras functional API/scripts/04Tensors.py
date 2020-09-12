import tensorflow as tf 

my_var = tf.Variable([-1,2], dtype = tf.float32, name = 'my_var')
h = my_var + [5, 4]
print(h)

# tf.Tensor([4. 6.], shape=(2,), dtype=float32)

from keras.layers import Dense, Input
from keras.models import Model

inputs = Input(shape=(5,))
h = Dense(16, activation='sigmoid')(inputs)
outputs = Dense(10, activation='softmax')(h)

model = Model(inputs = inputs, outputs = outputs)

print(h)
print(outputs)

print(model.input)
print(model.output)

x = tf.constant([[5,2],[1, 3]])

print(x)

x_arr = x.numpy()

print(x_arr)

x = tf.ones(shape=(2,1))
y = tf.zeros(shape=(2,1))

# array([[5, 2]
#        [1, 3]], dtype = int32)

# Tensor("dense_1/Sigmoid:0", shape=(None, 16), dtype=float32)
# Tensor("dense_2/Softmax:0", shape=(None, 10), dtype=float32)

# Tensor("input_1:0", shape=(None, 5), dtype=float32)
# Tensor("dense_2/Softmax:0", shape=(None, 10), dtype=float32)

# tf.Tensor(
# [[5 2]
# [1 3]], shape=(2, 2), dtype=int32)

# In above ex: we are creating input layer and then sending through a dense layer
# with 16 units which produces the output h.
# This output is result of computation represented by dense layer so its also a tensor

# The value of this tensor is still not known yet.
# The value will depend on the input thats fed into the model
# The computational graph that's built when we write these lines of code for the model, 
# contains the information for how this output tensor should be calculated.
# This tensor only actually gets a concrete value when the graph is executed or when the network is run with a given input. 
# None shape shows the batch size and will be showing actual value when input is given.
# Tne value of both tensors are unknown untill we run the network with actual data.

# If we built the model itself then model as a whole has input and output attribute
# These are precisely the input and output tensors for the model
# Notice that output tensor is exactly what we have already seen for output of final dense layer.

# Finally we can convert tensor to numpy array using numpy() method.

# We can create tensors filled with ones or zeros also.