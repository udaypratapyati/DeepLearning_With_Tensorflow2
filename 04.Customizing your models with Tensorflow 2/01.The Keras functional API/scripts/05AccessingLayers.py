import tensorflow as tf 

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D

inputs = Input(shape=(32,1), name = 'input_layer')
h = Conv1D(3, 5, activation = 'relu', name = 'conv1d_layer')(inputs)
h = AveragePooling1D(3, name = 'avg_pool1d_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(20, activation='sigmoid', name='dense_layer')(h)

model = Model(inputs=inputs, outputs = outputs)

print(model.layers)
print(model.layers[1])
print(model.layers[1].weights)
print(model.layers[1].get_weights())
print(model.layers[1].kernel)
print(model.layers[1].bias)
print(model.get_layer('conv1d_layer').bias)

# [<keras.engine.input_layer.InputLayer object at 0x000001358D8FEDC8>, 
#  <keras.layers.convolutional.Conv1D object at 0x00000135A4818608>, 
#  <keras.layers.pooling.AveragePooling1D object at 0x00000135A4818508>, 
#  <keras.layers.core.Flatten object at 0x00000135A4876208>, 
#  <keras.layers.core.Dense object at 0x00000135A4876688>]

# Each of these items in the list is an instance of layer object.
# Layer type is included in these object names.

# We can access individual layer by indexing as well.
# We can access layer weights by weights attribute of that layer.

# [<tf.Variable 'conv1d_1/kernel:0' shape=(5, 1, 3) dtype=float32, numpy=
# array([[[-0.06805122, -0.16137919,  0.5132464 ]],
#        [[ 0.06249267, -0.53560185, -0.09236827]],
#        [[ 0.21422273, -0.4562359 , -0.25090587]],
#        [[ 0.3723374 ,  0.37777388,  0.001531  ]],
#        [[-0.50949305,  0.03056866, -0.3766393 ]]], dtype=float32)>, <tf.Variable 'conv1d_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]

# In case, we don't want to access tensorflow variables but instead just 
# numpy arrays for the parameters over layer, then use get_weights method.

# [array([[[-0.29125655, -0.08961287, -0.25453842]],
#    [[-0.46703756,  0.07784969,  0.37955886]],
#    [[ 0.12469214,  0.34330016, -0.53661454]],
#    [[-0.2554281 , -0.05889434, -0.41781923]],
#    [[-0.22899216,  0.18442315, -0.00226831]]], dtype=float32), array([0., 0., 0.], dtype=float32)]

# We can access kernel and bias separately for conv layer.
# If we try to get kernel and bias for pooling or Flatten layer, we will get error.

# If we define layer by name, we can retrieve layer by name as well.

# <tf.Variable 'conv1d_layer/bias:0' shape=(3,) dtype=float32, numpy=
# array([0., 0., 0.], dtype=float32)>