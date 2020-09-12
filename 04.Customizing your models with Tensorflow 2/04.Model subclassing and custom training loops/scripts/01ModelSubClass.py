import tensorflow as tf

from keras.models import Model
from keras.layers import Dense

class MyModel(Model):

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = Dense(16)

    def call(self, inputs):
        return self.dense(inputs)

my_model = MyModel(name='my_model')

# We are importing the model class from Tensorflow.keras.models
# Now we are going to subclass the model class directly so I am creating my own class.
# The basic structure to keep in mind when model subclassing is 
# We create our layers in the initializer and define the forward pass in the call method.

# First thing we are doing is calling the initializer for base class
# and then we are creating the layer for the model.

# What we have done is to create a layer object and assign it as class attribute.
# But we have not called the layer yet. That happens in the call method.

# The call method will take one required argument which is input to the model.
# For this model, all I wanted to do is to call the dense layer only inputs
# and return the results.

# Now we have built the class, all we need to do is to create instance of the class.
# I am instantiating a model object and giving it a name.
# The name "keyword argument" is then being passed down to the base class constructor.

# This object inherits from the model-base class, and so it has all the methods
# We can call the compile method to set a loss function and optimizer and train with fit method.