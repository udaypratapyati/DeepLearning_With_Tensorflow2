import tensorflow as tf 
from keras.layers import Layer, Dense
from keras.models import Model

class LinearMap(Layer):

    def __init__(self, input_dim, units):
        super(LinearMap, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                initializer='random_normal')

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

linear_layer = LinearMap(3,2)

inputs = tf.ones((1,3))
print(linear_layer(inputs))
print(linear_layer.weights)

class MyModel(Model):

    def __init__(self, hidden_units, outputs, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = Dense(hidden_units, activation='sigmoid')
        self.linear = LinearMap(hidden_units, outputs)

    def call(self, inputs):
        h = self.dense(inputs)
        return self.linear(h)

my_model = MyModel(64, 12, name='my_custom_model')
