from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint

model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer='sgd',
            loss=BinaryCrossentropy(from_logits=True))

checkpoint = ModelCheckpoint('my_model', save_weights_only=True)

# model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])

# Creating a ModelCheckpoint object under checkpoint
# First Argument is filepath this is name that callback is using to save the model
# save_weights_only will save weights and not architecture.
# This will save the model weights after every epoch
# Since we are using same name, so weights are over ridden after each epoch is passing
# We can control that also and see in later this week.
# Three files will be created in current working directory.

# checkpoint
# my_model.data-00000-of00001
# my_model.index

# Alternative way of saving
checkpoint = ModelCheckpoint('keras_model.h5', save_weights_only=True)

# Check the filepath
checkpoint_path = 'model_checkpoint/checkpoint'
# checkpoint = ModelCheckpoint(filepath=checkpoint_path, frequency='epoch',
                            #save_weights_only=True, verbose = 1)
#! ls -lh filepath

# keras_model.h5

# Best practice is to save in native tensorflow format.

# How do we load from previously saved weights.
model.load_weights('my_model') #checkpoint_path
model.load_weights('keras_model.h5')

# How do we save weights manually without checkpoint callback

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_mae', patience=2)

# model.fit(X_train, y_train, validation_split=0.2, epochs=50,
        #callbacks=[early_stopping])

model.save_weights('my_model')
