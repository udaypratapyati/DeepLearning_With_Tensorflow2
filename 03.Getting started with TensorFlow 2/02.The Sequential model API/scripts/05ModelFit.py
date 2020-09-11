import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(64, activation = 'elu', input_shape = (32,)),
    Dense(100, activation = 'softmax')
])

model.compile(
    optimizer = 'rmsprop',
    loss = 'categorical_crossentropy', #sarse_categorical_crossentropy
    metrics = ['accuracy']
)

# .fit(X_train, y_train, epochs = 10, batch_size=32)
# history = model.fit(X_train, y_train, epochs = 10, batch_size=32, verbose=2)

# verbose = 0,1,2 (values)
# 2 - Print one line per epoch
# 1 - is default
# 0 - will silence the print out

#model.fit returns the history object with loss and accuracy matrix in dictionary form which we can store in dataframe to print 
#graph for the loss vs epochs | accuracy vs epochs | metric vs epoch

# X_train: (num_samples, num_features)
# y_train: (num_samples, num_classes) # y_train is one hot encode vector
# y_train: (num_samples,) # y_train is having sparse representation of one-dimensional array only and not one hot encode vector

# num_samples = number of examples of samples in training set
# if each datapoint is one dimensional array, so X_train would be two dimensional array
# with number of samples in first dimension and number of features in second.
# y_train would also be same except number of classes in second dimension.
# Assumption here is labels represented as one hot vector
# or if label is sparse representation, y_train could be one dimensional vector