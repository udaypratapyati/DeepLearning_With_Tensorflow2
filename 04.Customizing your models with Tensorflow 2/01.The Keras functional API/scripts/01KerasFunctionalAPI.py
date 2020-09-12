import tensorflow as tf 
from keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D, Sequential
from keras.models import Model

#Functional API
inputs = Input(shape=(32,1))
h = Conv1D(16,5, activation='relu')(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
outputs = Dense(20, activation='sigmoid')(h)

model = Model(inputs=inputs, outputs=outputs)

#Sequential API
model = Sequential([
    Conv1D(16,5,activation='relu', input_shape=(32,1)),
    AveragePooling1D(3),
    Flatten(),
    Dense(20, activation='sigmoid')
])

# model.compile(loss='binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

# test_loss, test_acc = model.evaluate(X_test, y_test)
# preds = model.predict(X_sample)