import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(128, activation='tanh'))
model.add(Dense(2))

opt = Adam(learning_rate=0.05)
model.compile(optimizer=opt, loss='mse', metrics=['mape'])

# During training itself, we can have validation set
# history = model.fit(inputs, targets, validation_split=0.2)

# history also records the performance on the validation set

# print(history.history.keys()) #dict_keys(['loss', 'mape', 'val_loss', 'val_mape'])
# these values were stored inside the history attributes of the history object and this attribute is a Python dictionary.

# Second option:
# Sometimes, datasets are already loaded with train and test splits like MNIST
# Instead of making model.fit do validation split for us, we can explicitly define it

# model.fit(X_train, y_train, validation_data=(X_test, y_test))
# This takes the tuple argument and model will record the performance on validation set also

# Third Option:
from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1)
# model.fit(X_train, y_train, validation_data=(X_val, y_val))