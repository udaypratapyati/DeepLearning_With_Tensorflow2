import tensorflow as tf 
from keras.layers import Layer

class LinearMap(Layer):

    def __init__(self, input_dim, units):
        super(LinearMap, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units)))

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

linear_layer = LinearMap(3,2)

inputs = tf.ones((1,3))
print(linear_layer(inputs))
print(linear_layer.weights)

# We are importing Layer class from tensorflow.keras.layers
# We are creating layer variables in the initializer.
# Call method contains the layer computation.

# <tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
# array([[ 0.01982623,  0.03747239],
#        [ 0.08603463,  0.06529035],
#        [-0.00209427,  0.02563126]], dtype=float32)> w
# Tensor("linear_map_1/MatMul:0", shape=(1, 2), dtype=float32)