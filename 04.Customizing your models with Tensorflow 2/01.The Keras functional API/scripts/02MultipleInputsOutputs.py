import tensorflow as tf 
from keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D, Concatenate
from keras.models import Model

inputs = Input(shape=(32,1))
h = Conv1D(16, 5, activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
aux_inputs = Input(shape=(12,))
h = Concatenate()([h, aux_inputs])
outputs = Dense(20, activation='sigmoid')(h)
aux_outputs = Dense(1, activation='Linear')(h)

model = Model(inputs = [inputs, aux_inputs], outputs = [outputs, aux_outputs])

# In this, we have multiple inputs and multiple outputs
# In this new model design, the auxiliary input is included in the model as an extra input to the final dense layer.
# Shape of input layer and output layer has to be taken care for concatenation
# Notice that the input is one-dimensional and so it has the right shape to be fed into the dense layer.
# Just before the final dense layer it's the flattened layer. And this outputs an unrolls tensor h. 
# The next line takes this output tensor h, and concatenates it with the auxiliary input to make a single one-dimensional vector.

# model.compile(loss=['binary_crossentropy', 'mse'], 
#                     loss_weights=[1, 0.4], 
#                     metrics = ['accuracy'])

# history = model.fit([X_train, X_aux], [y_train, y_aux], validation_split=0.2, epochs=20)
# Same goes with model.evaluate and model.predict methods.


# These losses are in same order as we define the outputs list order
# If we have two loss functions though, we need to combine them somehow. 
# We can only train our model using a gradient-based optimizer, 
# if there is a single loss value that we're trying to optimize.

# That's what the new loss_weights keyword argument is doing here. 
# These weights tell the model how to combine the loss functions. So here, 
# the final loss is the binary_crossentropy plus 0.4 times the mean squared error.

inputs = Input(shape=(32,1), name='inputs')
h = Conv1D(16, 5, activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
aux_inputs = Input(shape=(12,), name='aux_inputs')
h = Concatenate()([h, aux_inputs])
outputs = Dense(20, activation='sigmoid', name='outputs')(h)
aux_outputs = Dense(1, activation='Linear', name='aux_outputs')(h)

model = Model(inputs = [inputs, aux_inputs], outputs = [outputs, aux_outputs])

# model.compile(loss={'outputs': 'binary_crossentropy', 'aux_outputs': 'mse'},
#                     loss_weights={'outputs': 1, 'aux_outputs': 0.4},
#                     metrics=['accuracy'])

# history = model.fit({'inputs': X_train, 'aux_inputs': X_aux},
#                     {'outputs': y_train, 'aux_outputs': y_aux},
#                     validation_split=0.2, epochs=20)
