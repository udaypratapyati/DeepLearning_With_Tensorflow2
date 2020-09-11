#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# # The Sequential model API

#  ## Coding tutorials
#  #### [1. Building a Sequential model](#coding_tutorial_1)
#  #### [2. Convolutional and pooling layers](#coding_tutorial_2)
#  #### [3. The compile method](#coding_tutorial_3)
#  #### [4. The fit method](#coding_tutorial_4)
#  #### [5. The evaluate and predict methods](#coding_tutorial_5)

# ***
# <a id="coding_tutorial_1"></a>
# ## Building a Sequential model

# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax


# #### Build a feedforward neural network model

# In[15]:


# Build the Sequential feedforward neural network model
# model = Sequential()
# model.add(Flatten(input_shape=(28,28)))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# or
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')
])


# In[18]:


# Print the model summary
# model.weights     # prints the weights of each layer (arrays : initialized)
model.summary()


# ***
# <a id="coding_tutorial_2"></a>
# ## Convolutional and pooling layers

# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


# #### Build a convolutional neural network model

# In[45]:


# Build the Sequential convolutional neural network model
model = Sequential([
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
#     Conv2D(filters=16, kernel_size=(3,3), padding='SAME', activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(3,3)),
    Flatten(),
    Dense(units=10, activation='softmax')
])


# In[46]:


# Print the model summary
model.summary()

# conv2d will have 160 parameters 
# [i × (f×f) × o] + o -- i=>ip rgb channels, o=>no. of filters, fxf=>filter_size


# ***
# <a id="coding_tutorial_3"></a>
# ## The compile method

# #### Compile the model

# In[47]:


# Define the model optimizer, loss function and metrics
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
los = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.Accuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

# model.compile(optimizer=opt, loss=los, metrics=[acc, mae])

model.compile(
#     optimizer='adam',
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'mae']
)


# In[41]:


# Print the resulting model attributes
print(model.loss)
print(model.metrics)
print(model.optimizer)
print(model.optimizer.lr)


# ***
# <a id="coding_tutorial_4"></a>
# ## The fit method

# In[42]:


from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# #### Load the data

# In[48]:


# Load the Fashion-MNIST dataset

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()


# In[49]:


# Print the shape of the training data

print(train_images.shape)


# In[50]:


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


# In[51]:


# Rescale the image values so that they lie in between 0 and 1.
train_images = train_images / 255.
test_images = test_images / 255.


# In[53]:


# Display one of the images

i=0
img = train_images[i, :, :]
plt.imshow(img)
plt.show()
print(f'Label : {labels[train_labels[i]]}')


# #### Fit the model

# In[58]:


# Fit the model

# model.fit(train_images[:,:,:, np.newaxis], train_labels, epochs=2, batch_size=256)
history = model.fit(train_images[:,:,:, np.newaxis], train_labels, epochs=8, batch_size=256, verbose=2)


# #### Plot training history

# In[60]:


# Load the history into a pandas Dataframe

df = pd.DataFrame(history.history)
df.columns = ['loss', 'acc', 'mae']
df


# In[69]:


# Make a plot for the loss
loss_plot = df.plot(y='loss', title='Loss Vs Epochs')
loss_plot.set(xlabel='Epochs', ylabel='Loss')
plt.grid()


# In[67]:


# Make a plot for the accuracy
acc_plot = df.plot(y='acc', title='Accuracy Vs Epochs')
acc_plot.set(xlabel='Epochs', ylabel='Accuracy')
plt.grid()


# In[68]:


# Make a plot for the additional metric
mae_plot = df.plot(y='mae', title='Mean Absolute Error Vs Epochs')
mae_plot.set(xlabel='Epochs', ylabel='Mean Absolute Error')
plt.grid()


# ***
# <a id="coding_tutorial_5"></a>
# ## The evaluate and predict methods

# In[70]:


import matplotlib.pyplot as plt
import numpy as np


# #### Evaluate the model on the test set

# In[71]:


# Evaluate the model
testLoss, testAcc, testMAE = model.evaluate(test_images[...,np.newaxis], test_labels, verbose=2)


# #### Make predictions from the model

# In[76]:


# Choose a random test image

random_inx = np.random.choice(test_images.shape[0])

test_image = test_images[random_inx]
plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[random_inx]]}")


# In[77]:


# Get the model predictions

print(test_image.shape)
pred = model.predict(test_image[np.newaxis,...,np.newaxis])
print(f'Model Prediction : {labels[np.argmax(pred)]}')


# In[ ]:




