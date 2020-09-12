import tensorflow as tf 

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D

inputs = Input(shape=(8,8,1), name = 'input_layer')
h = Conv2D(16,3,activation='relu', name = 'conv2d_layer', trainable=False)(inputs)
h = MaxPooling2D(3, name='max_pool2d_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(10, activation='softmax', name='softmax_layer')(h)

model = Model(inputs = inputs, outputs = outputs)

model.get_layer('conv2d_layer').trainable = False

# model.compile(loss='sparse_categorical_crossentropy')

# Each layer has trainable argument and by default its true.
# This means values of the weights will not change from initialized values.
# Other hand, Dense layer is not frozen and its weights can be change.

# Another way is to freeze the layer after the model is built.
# Make sure this line is before compiling the model.

model = load_model('my_pretrained_model')

model.trainable = False

flatten_output = model.get_layer('flatten_layer').output
new_outputs = Dense(5, activation='softmax', name='new_softmax_layer')(flatten_output)

new_model = Model(inputs = model.input, outputs = new_outputs)

# new_model.compile(loss='sparse_categorical_crossentropy')
# new_model.fit(X_train, y_train, epochs=10)
