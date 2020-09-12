# variables store values like model parameters that persists during the program 
# but can also be changed during the program

# Tensors are what are passed around in the computational graph. 
# They represent the inputs and the outputs to the various computational blocks inside the model.

# We can easily access these tensors inside a network, 
# and use them to build new models out of old ones.

import tensorflow as tf 

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D

inputs = Input(shape=(32,1), name = 'input_layer')
h = Conv1D(3, 5, activation = 'relu', name = 'conv1d_layer')(inputs)
h = AveragePooling1D(3, name = 'avg_pool1d_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(20, activation='sigmoid', name='dense_layer')(h)

model = Model(inputs=inputs, outputs = outputs)

print(model.get_layer('conv1d_layer').input)
print(model.get_layer('conv1d_layer').output)

flatten_output = model.get_layer('flatten_layer').output

model2 = Model(inputs=model.input, outputs = flatten_output)

model3 = Sequential([
    model2,
    Dense(10, activation='softmax', name = 'new_dense_layer')
])

new_outputs = Dense(10, activation='softmax')(model2.output)
model3 = Model(inputs = model2.input, outputs = new_outputs)

# Tensor("input_layer:0", shape=(None, 32, 1), dtype=float32)
# Tensor("conv1d_layer/Relu:0", shape=(None, 28, 3), dtype=float32)

# Each Layer has an input tensor and output tensor.
# Every model has an input and output tensor.

# In functional API, we are constructing models by passing input tensor
# and output tensor to the corresponding keyword argument.

# In above example, we have retrieved the Flatten layer in our existing model
# I can then build new model model2 which uses same input as original model.
# But output of this new model2 is Flattened layer.
# So what we have done is remove that final dense layer from our model.

# We can take parts of existing model, use them in new model and add extra
# layers in this new model.

# Sequential API allows to include entire model as one of the items in list
# model3 uses same first few layers as original model but with final dense layer
# removes and adds a new dense layer on top.

# It can also be done by using functional API as above.

flatten_output = model.get_layer('flatten_layer').output

new_outputs = Dense(10, activation='softmax')(flatten_output)
model3 = Model(inputs=model.input, outputs=new_outputs)

# In above code, we don't need to use model2 as intermediary model
# In functional API, we can do it as above.