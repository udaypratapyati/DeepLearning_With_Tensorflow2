import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(1, activation='sigmoid', input_shape = (12,))
])
model.compile(
    optimizer='sgd', 
    loss='binary_crossentropy',
    metrics=['accuracy', 'mae']
    )
# model.fit(X_train,y_train)

# We can evaluate the performance on test set with evaluate method
# model.evaluate(X_test, y_test)

# We can store the loss, accuracy in different variables like below
# loss, accuracy = model.evaluate(X_test, y_test)

# If there is more metric, we can save them like given below
# loss, accuracy, mae = model.evaluate(X_test,y_test)

# X_sample: (num_sample, 12) #First dimension is number of examples, Second is no. of features

# If we want to predict on new sample, we will use predict method.
# pred = model.predict(X_sample)

# Example:
# X_sample: (1,12) #Just one example having 12 input features
# pred = model.predict(X_sample) #[[0.07713523]]

# X_sample: (2,12) #for two example having 12 input features
# pred = model.predict(X_sample) #[[0.07713523]
#                                 [0.94515101]]

# For multiclass classification: If we have 3 classes.
# Final layer will have 3 neurons in dense layer
# loss would be changed from binary_crossentropy to categorical_crossentropy
# activation in output layer will be changed to softmax rather than sigmoid
# Considering 12 input features and we are predicting for two samples

# pred = model.predict(X_sample) # [[0.93957397, 0.0189931, 0.04143293]
#                                  [0.01211542, 0.0907736, 0.89711098]]

# In this output, first dimension would be number of samples i.e. 2, 
# second dimension would be 3 i.e for the number of classes.
# Each row output is predicted by softmax function for probability
# Sum of each row has to be 1 as per rules of probability



