import tensorflow as tf 

from keras.datasets import fashion_mnist
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# Build the Sequential convolutional neural network model

model = Sequential([
    Conv2D(16,(3,3), activation = 'relu', input_shape = (28,28,1)),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10, activation = 'softmax')
])

print(model.summary)

#Define the model optimizer, loss function and metrics
#Remember to pass string if directly passing the name of optimizer

opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer = opt,
             loss = 'sparse_categorical_crossentropy',
             metrics = [acc, mae])

# Define the labels

labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

print(train_labels[0])
print(labels[train_labels[0]])

# Rescale the image values so that they lie in between 0 and 1.

train_images = train_images/255.
test_images = test_images/255. 

# Display one of the images

i = 0
img = train_images[i,:,:]
plt.imshow(img)
plt.show()
print(f"label: {labels[train_labels[i]]}")

# Saving the model.fit into history object for 8 epochs with verbose 2

history = model.fit(train_images[...,np.newaxis], train_labels, epochs = 8, batch_size=256, verbose=2)

# Load the history into a pandas DataFrame
# history object contains history attribute of the model for loss functions and matrix after each of epoch in dictionary format.

df = pd.DataFrame(history.history)
df.head()

# Make a plot for the loss

loss_plot = df.plot(y="loss", title = "Loss vs Epochs", legend = False)
loss_plot.set(xlabel="Epochs", ylabel = "Loss")

# Make a plot for the accuracy

loss_plot = df.plot(y="sparse_categorical_accuracy", title = "Accuracy vs Epochs", legend = False)
loss_plot.set(xlabel="Epochs", ylabel = "Accuracy")

# Make a plot for the additional metric

loss_plot = df.plot(y="mean_absolute_error", title = "MAE vs Epochs", legend = False)
loss_plot.set(xlabel="Epochs", ylabel = "MAE")

#Evaluation on test dataset.

# Evaluate the model

model.evaluate(test_images[...,np.newaxis], test_labels, verbose=2)
test_loss, test_accuracy, test_mae = model.evaluate(test_images[...,np.newaxis], test_labels, verbose=2)

# Choose a random test image

random_inx = np.random.choice(test_images.shape[0])

inx = 30
test_image = test_images[inx]
plt.imshow(test_image)
plt.show()
print(f"lable: {labels[test_labels[inx]]}")

# Get the model predictions

model.predict(test_image[np.newaxis,...,np.newaxis])

predictions = model.predict(test_image[np.newaxis,...,np.newaxis])
print(f"Model Prediction: {labels[np.argmax(predictions)]}")

