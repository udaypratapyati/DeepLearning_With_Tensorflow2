import tensorflow as tf
from tensorflow.keras.layers import Layer

class linearMap(Layer):
    # Since the class is inheriting the larers class, it will have all the functions of Layers.
    def __init__(self, input_dim, units):
        super(linearMap, self).__init__()
        w_int = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_int(shape=(input_dim,units)))

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

linear_map = linearMap(3, 2)
inputs = tf.ones((1,3))

print(inputs)               # tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
print(linear_map.w.shape)   # (3, 2)
print(linear_map(inputs))   # tf.Tensor([[-0.01325359 -0.01845775]], shape=(1, 2), dtype=float32)
print(linear_map.weights)
# [<tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
# array([[ 0.01618611,  0.05729821],
#        [ 0.07692757,  0.07510909],
#        [-0.00696352,  0.01691794]], dtype=float32)>]


class linearMap1(Layer):
    # Since the class is inheriting the larers class, it will have all the functions of Layers.
    def __init__(self, input_dim, units):
        super(linearMap1, self).__init__()
        self.w = self.add_weight(shape=(input_dim,units),   # this is a shorcut method to above one.
                                initializer='random_normal')

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

linear_map1 = linearMap1(3, 2)
print(inputs)                # tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
print(linear_map1.w.shape)   # (3, 2)
print(linear_map1(inputs))   # tf.Tensor([[-0.01325359 -0.01845775]], shape=(1, 2), dtype=float32)
print(linear_map1.weights)
# [<tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
# array([[ 0.06170653,  0.01048746],
#        [ 0.00295068,  0.04351053],
#        [ 0.07554378, -0.1694311 ]], dtype=float32)>]


class MyModel(Model):
    
    # Instance creation of layer elements
    def __init__(self, hiddenUnits, outputs, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = Dense(hiddenUnits, activation='sigmoid')
        self.linear = LinearMap(hiddenUnits, outputs)
    
    # Forward pass
    def call(self, inputs):
        h = self.dense(inputs)
        h = self.linear(h)
        return h

model = MyModel(64, 12, name='my_custom_model')

import tensorflow as tf
from tensorflow.keras.layers import Layer

class MyCustomLayer(Layer):

    def __init__(self, input_dim, eps):
        super(MyCustomLayer, self).__init__()
        self.c = eps * tf.ones((input_dim,))

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=0) + self.c

layer = MyCustomLayer(3, 0.1)
x = tf.ones((2, 3))
print(layer(x).numpy())     # [2.1 2.1 2.1]    