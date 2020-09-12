import tensorflow as tf 
from keras.layers import Layer, Dense
from keras.models import Model

class MyModel(Model):

    def __init__(self, hidden_units, outputs, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = Dense(hidden_units, activation='sigmoid')
        self.linear = MyCustomLayer(hidden_units, outputs)

    def call(self, inputs):
        h = self.dense(inputs)
        return self.linear(h)


class MyCustomLayer(Layer):

    def __init__(self, input_dim, eps):
        super(MyCustomLayer, self).__init__()
        self.c = eps * tf.ones((input_dim,))

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=0) + self.c

mymodel = MyModel(16, 4, name="my_model")

layer = MyCustomLayer(3, 0.1)
x = tf.ones((2,3))
print(layer(x))
print(layer(x).numpy())

# Tensor("my_custom_layer_2/add:0", shape=(3,), dtype=float32)
