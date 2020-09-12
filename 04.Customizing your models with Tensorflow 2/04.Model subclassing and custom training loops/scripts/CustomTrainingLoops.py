import tensorflow as tf

for epoch in range(10):
	for inputs, outputs in training_dataset:
		with tf.GradientTape() as tape:
		    current_loss = loss(my_model(inputs), outputs)
		    grads = tape.gradient(current_loss, my_model.trainable_variables)
		optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

# How many parameter updates does the following training loop make to the model parameters? 
# Assume there are 60,000 examples in the training set, and it has been batched with batch size 32.

# Ans : 18750 --> 1 update per batch, so, 60000/32 * 10 -> batch size = 32

# Below is not compilable, just for understanding.
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizer import SGD
import numpy as np 

# Build a model, let say sequential
model = tf.keras.modles.Sequential([
    tf.keras.layers.Dense()
])

# we will need a loss function and an optimizer
loss = MeanSquaredError()
opt = SGD(learning_rate=.05, momentum=.9)

epochs_loss = []
for epoch in range(10): # running this for 10 epochs
    batch_loss = []
    for inputs, outputs in training_dataset: # for each batch, get the loss
        with tf.GradientTape() as tape:
            curr_loss = loss(model(inputs), outputs)
            grads = tape.gradient(curr_loss, model.trainable_variables) # Getting gradients

        batch_loss.append(curr_loss)
        opt.apply_gradients(zip(grads, model.trainable_variables)) # updating grad for the current batch.

    epochs_loss.append(np.mean(batch_loss)) # save the mean loss for this epoch