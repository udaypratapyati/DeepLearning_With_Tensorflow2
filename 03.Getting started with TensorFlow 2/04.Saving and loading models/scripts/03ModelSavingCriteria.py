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

# checkpoint = ModelCheckpoint('training_run_1/my_model', save_weights_only=True,
                    # save_freq=1000, save_best_only=True, monitor='val_loss|val_acc',
                    # mode = 'max')

# Filepath argument in the ModelCheckpoint will be created on the fly if already not given explicitly.

# model.fit(x_train, y_train, validation_split=(X_val, y_val),
        # epochs=10, batch_size=16, callbacks=[checkpoint])

# save_freq will save number of samples of 1000, batch size is 16 so model will save around 62 iterations
# save_best_only will save if val_loss is minimum else will not save

# checkpoint = ModelCheckpoint('training_run_1/my_model.{epoch}.{batch}', 
                    # save_weights_only=True, save_freq=1000)

# We can save model name with epoch number and batch size to avoid overwriting of files.

# checkpoint = ModelCheckpoint('training_run_1/my_model.{epoch}-{val_loss:.4f}', 
                    # save_weights_only=True, save_freq=1000)