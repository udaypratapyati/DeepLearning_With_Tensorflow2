import tensorflow as tf  
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

model = Sequential([
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='rmsprop',
            loss = 'sparse_categorical_entropy',
            metrics = ['acc','mae'])

checkpoint = ModelCheckpoint('my_model', save_weights_only=False)

# model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])

# save_weights_only set to false if saving entire model.
# by default, its false.

# my_model/assets (This folder is where files can be stored that are used by tensorflow graph, in our case, it will be empty)
# my_model/saved_model.pb (saved_model.pb, is the file that stores the TensorFlow graph itself.)
# my_model/variables/variables.data-00000-of-00001
# my_model/variables/variables.index

# You can think of this as storing the model architecture. So it contains all the information from when we built and compiled the model, 
# including the optimizer state, in case we want to resume training from a saved model.

# keras_model.h5 is saved if checkpoints is saved as keras_model.h5

model.save('my_model') # SavedModel format

from keras.models import load_model

new_model = load_model('my_model')

# load_model will return complete architecture, model weights
# Now we can do anything with this model for further processing.

# new_model.summary()
# new_model.fit(X_train, y_train, validation_data=(X_val, y_val),
# epochs=20, batch_size=16)
# new_model.evaluate(X_test, y_test)
# new_model.predict(X_samples)

new_keras_model = load_model('keras_model.h5')