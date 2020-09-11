import tensorflow as tf 
from keras.layers import Conv1D, Flatten, Dense, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping

model = Sequential([
    Conv1D(16,5, activation='relu', input_shape=(128,1)),
    MaxPooling1D(4),
    Flatten(),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss',
#                                patience=5, 
#                                min_delta=0.01,
#                                mode='max') # val_loss is default parameter in earlystopping

#model.fit(X_train, y_train, validation_split=0.2,
        #epochs=100, callbacks=[early_stopping])

# Note: We can also use val_accuracy to decide when to terminate the training
# We could also use any other metric if we were using during model compilation
# String used in monitor is the same which history object creates during training of the model
# Another argument is patience, default set at zero. 
# Patience means training stops if performance goes worse from one epoch to another
# Patience = 5 means stopping training if model performance does not improve in 5 epochs in a row
# min_delta = 0.01 means what qualifies the model performance improvement.
# min_delta = 0 by default, that means any improvement in performance would make patience counter reset
# mode = max, by default = auto, but better set explicitly for improving model performance
