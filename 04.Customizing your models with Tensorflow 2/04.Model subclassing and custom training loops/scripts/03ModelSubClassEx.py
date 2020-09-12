import tensorflow as tf 

from keras.models import Model
from keras.layers import Dense

class MyModel(Model):
    
    def __init__(self, units, classes, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.d1 = Dense(units, activation='relu')
        self.d2 = Dense(classes, activation='softmax')

    def call(self,x):
        x = self.d1(x)
        return self.d2(x)

mymodel = MyModel(16, 4, name='my_model')

mymodel.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics='[accuracy')

mymodel.summary()

# How many parameters does this model have if it is called on a numpy array input of shape (128, 8)?

# So this model will have 212 parameters.
# 16*8+16 = 144
# 16*4 + 4 = 68
# Total = 144 + 68 = 212