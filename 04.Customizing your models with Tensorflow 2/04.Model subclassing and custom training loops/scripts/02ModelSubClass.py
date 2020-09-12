import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout

# class MyModel(Model):

#     def __init__(self, num_classes, **kwargs):
#         super(MyModel, self).__init__(**kwargs)
#         self.dense1 = Dense(16, activation='sigmoid')
#         self.dropout = Dropout(0.5)
#         self.dense2 = Dense(num_classes, activation='softmax')

#     def call(self, inputs, training=False):
#         h = self.dense1(inputs)
#         h = self.dropout(h, training=training)
#         return self.dense2(h)

# my_model = MyModel(12, name='my_model')

# I've added in another dense layer and in the call method.
# I'm now passing the input through both dense layers.
# I also decided that I want the my_model class to be flexible enough
# that I can create different model instances with different numbers of outputs.
# So I can reuse it when I have datasets with a different number of classes.

# I've added in an extra argument in the constructor so the user can set the number of classes the model is predicting.
# This argument is being used in the second dense layer to determine the number of neurons in this layer.
# I'm creating a model objects from this class that has a 10-way soft-maxing the final layer.

# We can use training keyword argument to determine the behaviour of the model at training and testing time.
# This keyword argument should be a boolean.

# A really common use of this keyword argument is in batch norm and dropout layers.
# In initializer, I am using a dropout layer with rate 0.5

# I am calling this dropout layer in call method in between the two dense layers
# This layer uses the training keyword argument.

# So when this model is being trained using fit method,
# Dropout will randomly zero out its inputs and scale up the remaining activations.

# At test time, dropout layer does nothing.