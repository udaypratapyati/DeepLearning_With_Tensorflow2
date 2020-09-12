from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

class MyModel(Model):
    def __init__(self, classes, **kwargs): 
        '''
        in init, only do initialization, instantiations are done in call fun.
        # create all the layers and attributes in the constructor.
        '''
        super(MyModel, self).__init__(**kwargs)
        self.d1 = Dense(16, activation='relu')
        self.dout = Dropout(0.5)
        self.d2 = Dense(classes, activation='softmax')

    def call(self, x, training=False):
        '''
        forward pass is done in call api.
        '''
        h = self.d1(x)
        h = self.dout(h, training=training)
        return self.d2(x)

model = MyModel(12, name='mymodel') 
model.compile(loss='binary_crossentropy')
model.build((128, 8))
print(model.summary())

##############################################################
class MyModel1(Model):
    def __init__(self, units, classes, **kwargs): 
        '''
        in init, only do initialization, instantiations are done in call fun.
        # create all the layers and attributes in the constructor.
        '''
        super(MyModel1, self).__init__(**kwargs)
        self.d1 = Dense(units, activation='relu')
        self.d2 = Dense(classes, activation='softmax')

    def call(self, x):
        '''
        forward pass is done in call api.
        '''
        x = self.d1(x)
        return self.d2(x)
    
mymodel = MyModel1(16, 4, name='my_model')
mymodel.compile(loss='binary_crossentropy')
mymodel.build( (128, 8))

mymodel.summary()
