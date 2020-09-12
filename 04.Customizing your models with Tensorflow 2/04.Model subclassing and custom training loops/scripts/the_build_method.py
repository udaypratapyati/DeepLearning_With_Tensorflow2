# -*- coding: utf-8 -*-
"""The_build_method.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17WQz0K5eYP7yylIBbD8k7HbxzJnaqriv

# Flexible input shapes for custom layers
In this reading you will learn how to use the build method to allow custom layers to work with flexible sized inputs.
"""

import tensorflow as tf
print(tf.__version__)

"""## Fix the input shape in the custom layer

Previously, you have created custom layers by initialising all variables in the `__init__` method. For instance, you defined a dense layer called `MyLayer` as follows:
"""

# Create a custom layer

from tensorflow.keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, units, input_dim, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        print('inside init')
        self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal')
        self.b = self.add_weight(shape=(units,),
                             initializer='zeros')
        
    def call(self, inputs):
        print('inside call')
        return tf.matmul(inputs, self.w)+self.b

"""Notice that the required arguments for the `__init__` method are the number of units in the dense layer (`units`) and the input size (`input_dim`). This means that you need to fix these two arguments when you instantiate the layer."""

#  Create a custom layer with 3 units and input dimension of 5

dense_layer = MyLayer(3, 5)

"""Since the input size has been fixed to be 5, this custom layer can only take inputs of that size. For example, we can call the layer as follows:"""

# Call the custom layer on a Tensor input of ones

x = tf.ones((1,5))
print(dense_layer(x))

"""However, forcing the input shape (and therefore the shape of the weights) to be fixed when the layer is instantiated is unnecessary, and it may be more convenient to only do this later on, after the model has been defined. 

For example, in some cases you may not know the input shape at the model building time. We have come across this concept before when building models with the Sequential API. If the `input_shape` argument is omitted, the weights will only be created when an input is passed into the model.

## Allow a flexible input shape in the custom layer

You can delay the weight creation by using the `build` method to define the weights. The `build` method is executed when the `__call__` method is called, meaning the weights are only created only the layer is called with a specific input.

The `build` method has a required argument `input_shape`, which can be used to define the shapes of the layer weights.
"""

# Rewrite the custom layer with lazy weight creation

class MyLayer(Layer):

    def __init__(self, units, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        print('init')
        self.units = units
    
    def build(self, input_shape):
        print('build')
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros')
    def call(self, inputs):
        print('call')
        return tf.matmul(inputs, self.w)+self.b

"""Now, when you instantiate the layer, you only need to specify the number of units in the dense layer (`units`), and not the input size (`input_dim`).

### Create a custom layer with flexible input size
"""

#  Create a custom layer with 3 units

dense_layer = MyLayer(3)

"""This layer can now be called on an input of any size, at which point the layer weights will be created and the input size will be fixed."""

# Call the custom layer on a Tensor input of ones of size 5

x = tf.ones((1,4))
print(dense_layer(x))

# Print the layer weights

dense_layer.weights

"""### Create a new custom layer and pass in a different sized input"""

#  Create a new custom layer with 3 units

dense_layer = MyLayer(3)

# Call the custom layer on a Tensor input of ones of size 4

x = tf.ones((1,5))
print(dense_layer(x))

# Print the layer weights

dense_layer.weights

"""Note that the code for creating a custom layer object is identical, but the shape of the weights in the layer depend on the size of the input passed to the layer.

## Flexible input shapes in models

Deferring the weight creation until the layer is called is also useful when using the custom layer as an intermediate layer inside a larger model. In this case you may want to create several custom layer objects in the model, and it is tedious to keep track of the input shape that each of the custom layers needs. 

By deferring the weight creation as above, the input shape can be inferred from the output of the previous layer.
"""

# Create a model using the custom layer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Softmax

class MyModel(Model):

    def __init__(self, units_1, units_2, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.layer_1 = MyLayer(units_1)
        self.layer_2 = MyLayer(units_2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        return Softmax()(x)

"""In the above model definition, the custom layer `MyLayer` is used twice. Notice that each instance of the custom layer object can have a different input size, depending on the arguments used to create the model and the inputs passed into the model"""

# Create a custom model object

model = MyModel(units_1=32, units_2=10)

"""We can create and initialise all of the weights of the model by passing in an example Tensor input."""

# Create and initialize all of the model weights

_ = model(tf.ones((1, 100)))

# Print the model summary

model.summary()

"""## Further reading and resources 
* https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known
"""