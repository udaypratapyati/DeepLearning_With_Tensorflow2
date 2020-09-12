import tensorflow as tf 

# my_model = MyModel()

# def loss(y_hat, y):
#     return tf.reduce_mean(tf.square(y_hat - y))

# with tf.GradientTape() as tape:
#     current_loss = loss(my_model(inputs), outputs)
#     grads = tape.gradient(current_loss, my_model.trainable_variables)

from keras.losses import MeanSquaredError
from keras.optimizers import SGD

loss = MeanSquaredError()
optimizer = SGD(learning_rate=0.05, momentum=0.9)

batch_losses = []

# for inputs, outputs in training_dataset:
#     with tf.GradientTape() as tape:
#         current_loss = loss(my_model(inputs), outputs)
#         grads = tape.gradient(current_loss, my_model.trainable_variables)

#     batch_losses.append(current_loss)
#     optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

