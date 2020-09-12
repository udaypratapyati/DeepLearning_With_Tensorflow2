#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
print(tf.__version__)


# # Data Pipeline

#  ## Coding tutorials
#  #### [1. Keras datasets](#coding_tutorial_1)
#  #### [2. Dataset generators](#coding_tutorial_2)
#  #### [3. Keras image data augmentation](#coding_tutorial_3)
#  #### [4. The Dataset class](#coding_tutorial_4)
#  #### [5. Training with Datasets](#coding_tutorial_5)

# ***
# <a id="coding_tutorial_1"></a>
# ## Keras datasets
# 
# For a list of Keras datasets and documentation on recommended usage, see [this link](https://keras.io/datasets/).

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# #### Load the CIFAR-100 Dataset

# In[3]:


from tensorflow.keras.datasets import cifar100


# In[4]:


# Load the CIFAR-100 dataset

(train_images, train_lables), (test_images, test_labels) = cifar100.load_data(label_mode='fine')


# In[5]:


get_ipython().system('ls -lrt /home/jovyan/.keras/cifar100')


# In[6]:


# Confirm that reloading the dataset does not require a download

(train_images, train_lables), (test_images, test_labels) = cifar100.load_data(label_mode='fine')


# #### Examine the Dataset

# In[7]:


# Examine the shape of the data.

print(f'train image {train_images.shape}{train_lables.shape}')
print(f'test image {test_images.shape}{test_labels.shape}')


# In[8]:


# Examine one of the images and its corresponding label

plt.imshow(train_images[0])
print(f'train label {train_lables[0]}')


# In[9]:


# Load the list of labels from a JSON file

import json

with open('data/cifar100_fine_labels.json', 'r') as fine_labels:
    cifar100_fine_labels = json.load(fine_labels)


# The list of labels for the CIFAR-100 dataset are available [here](https://www.cs.toronto.edu/~kriz/cifar.html).

# In[10]:


# Print a few of the labels

cifar100_fine_labels[:10]


# In[11]:


# Print the corresponding label for the example above

cifar100_fine_labels[19]


# #### Load the data using different label modes

# In[12]:


# train_images[(train_lables.T == 86)[0]][:3]


# In[13]:


# Display a few examples from category 87 (index 86) and the list of labels

examples = train_images[(train_lables.T == 86)[0]][:3]
fig, ax = plt.subplots(1,3)
ax[0].imshow(examples[0])
ax[1].imshow(examples[1])
ax[2].imshow(examples[2])


# In[14]:


# Reload the data using the 'coarse' label mode

(train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode='coarse')


# In[15]:


# Display three images from the dataset with the label 6 (index 5)

examples = train_images[(train_labels.T == 5)[0]][:3]
fig, ax = plt.subplots(1,3)
ax[0].imshow(examples[0])
ax[1].imshow(examples[1])
ax[2].imshow(examples[2])


# In[16]:


# Load the list of coarse labels from a JSON file

with open('data/cifar100_coarse_labels.json', 'r') as coarse_labels:
    cifar100_coarse_labels = json.load(coarse_labels)


# In[17]:


# Print a few of the labels

cifar100_coarse_labels[:5]


# In[18]:


# Print the corresponding label for the example above

cifar100_coarse_labels[5]


# #### Load the IMDB Dataset

# In[19]:


from tensorflow.keras.datasets import imdb


# In[20]:


# Load the IMDB dataset

(train_data,train_label),(test_data,test_label) = imdb.load_data()


# In[21]:


# Print an example from the training dataset, along with its corresponding label

print(f'{train_data[0]} \n\n{train_label[0]}')


# In[22]:


# Get the lengths of the input sequences

sequence_lengths = [len(seq) for seq in train_data]


# In[23]:


# Determine the maximum and minimum sequence length

print(np.max(sequence_lengths),'\n', np.min(sequence_lengths))


# #### Using Keyword Arguments

# In[24]:


# Load the data ignoring the 50 most frequent words, use oov_char=2 (this is the default)

(train_data,train_label),(test_data,test_label) = imdb.load_data(skip_top=50, oov_char=2)


# In[25]:


# Get the lengths of the input sequences

sequence_lengths = [len(seq) for seq in train_data]


# In[26]:


# Determine the maximum and minimum sequence length

print(np.max(sequence_lengths),'\n', np.min(sequence_lengths))


# In[ ]:





# In[27]:


# Define functions for filtering the sequences

def remove_oov_char(element):
    ''' Filter function for removing the oov_char. '''
    return [word for word in element if word!=2]

def filter_list(lst):
    ''' Run remove_oov_char on elements in a list. '''
    return [remove_oov_char(element) for element in lst]


# In[28]:


# Remove the oov_char from the sequences using the filter_list function

train_data = filter_list(train_data)


# In[29]:


# Get the lengths of the input sequences

sequence_lengths = [len(seq) for seq in train_data]


# In[30]:


# Determine the maximum and minimum sequence length

print(np.max(sequence_lengths),'\n', np.min(sequence_lengths))


# ***
# <a id="coding_tutorial_2"></a>
# ## Dataset generators

# In[31]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[32]:


y = np.random.choice([0,1], (10, 1))
np.random.randn(10, 1) + (2 * y - 1)


# #### Load the UCI Fertility Dataset
# 
# We will be using a dataset available at https://archive.ics.uci.edu/ml/datasets/Fertility from UC Irvine.

# In[33]:


# Load the fertility dataset

headers = ['Season', 'Age', 'Diseases', 'Trauma', 'Surgery', 'Fever', 'Alcohol', 'Smoking', 'Sitting', 'Output']
fertility = pd.read_csv('data/fertility_diagnosis.txt', delimiter=',', header=None, names=headers)


# In[34]:


# Print the shape of the DataFrame

print(f'Data shape : {fertility.shape}')


# In[35]:


# Show the head of the DataFrame

fertility.head()


# #### Process the data

# In[36]:


# Map the 'Output' feature from 'N' to 0 and from 'O' to 1

fertility['Output'] = fertility['Output'].map(lambda x : 0.0 if x=='N' else 1.0)


# In[37]:


# Show the head of the DataFrame

fertility.head()


# In[38]:


# Convert the DataFrame so that the features are mapped to floats

fertility = fertility.astype('float32')


# In[39]:


# Shuffle the DataFrame

fertility = fertility.sample(frac=1).reset_index(drop=True)


# In[40]:


# Show the head of the DataFrame

print(fertility.shape)
fertility.head()


# In[41]:


# Convert the field Season to a one-hot encoded vector

fertility = pd.get_dummies(fertility, prefix='Season', columns=['Season'])


# In[42]:


# Show the head of the DataFrame

fertility.head()


# In[43]:


# Move the Output column such that it is the last column in the DataFrame

fertility.columns = [col for col in fertility.columns if col != 'Output'] + ['Output']


# In[44]:


# Show the head of the DataFrame

fertility.head()


# In[45]:


# Convert the DataFrame to a numpy array.

fertility = fertility.to_numpy()


# #### Split the Data

# In[46]:


# Split the dataset into training and validation set

training = fertility[0:70]
validation = fertility[70:100]


# In[47]:


# Verify the shape of the training data

print(f'train : {training.shape}')
print(f'validation : {validation.shape}')


# In[48]:


# Separate the features and labels for the validation and training data

training_features = training[:,0:-1]
training_labels = training[:,-1]
validation_features = validation[:,0:-1]
validation_labels = validation[:,-1]


# In[49]:


# print(training_features,'\n',training_labels)


# #### Create the Generator

# In[50]:


# Create a function that returns a generator producing inputs and labels

def get_generator(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield (features[n*batch_size: (n+1)*batch_size], labels[n*batch_size: (n+1)*batch_size])


# In[51]:


# Apply the function to our training features and labels with a batch size of 10

train_generator = get_generator(training_features, training_labels, batch_size=10)


# In[52]:


# Test the generator using the next() function

next(train_generator)


# #### Build the model

# In[53]:


# Create a model using Keras with 3 layers

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization

input_shape = (12,)
output_shape = (1,)

model_input = Input(input_shape)
batch_1 = BatchNormalization(momentum=0.8)(model_input)
dense_1 = Dense(100, activation='relu')(batch_1)
batch_2 = BatchNormalization(momentum=0.8)(dense_1)
output = Dense(1, activation='sigmoid')(batch_2)

model = Model([model_input], output)


# In[54]:


# Display the model summary to show the resultant structure

model.summary()


# #### Compile the model

# In[55]:


# Create the optimizer object

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


# In[56]:


# Compile the model with loss function and metric

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# #### Train and evaluate the model using the generator

# In[57]:


# Calculate the number of training steps per epoch for the given batch size.

batch_size = 5
train_steps = len(training) // batch_size


# In[58]:


# Set the epochs to 3

epochs = 3


# In[59]:


# Train the model

for epoch in range(epochs):
    train_generator = get_generator(training_features, training_labels, batch_size)
    validation_generator = get_generator(validation_features, validation_labels, 30)
    model.fit_generator(train_generator, steps_per_epoch=train_steps, 
                        validation_data=validation_generator, validation_steps=1)


# In[60]:


# Try to run the fit_generator function once more; observe what happens

model.fit_generator(train_generator, steps_per_epoch=train_steps)


# #### Make an infinitely looping generator

# In[61]:


# Create a function that returns an infinitely looping generator

def get_generator_cyclic(features, labels, batch_size=1):
    while True:
        for n in range(int(len(features)/batch_size)):
            yield (features[n*batch_size: (n+1)*batch_size], labels[n*batch_size: (n+1)*batch_size])
            
        permuted = np.random.permutation(len(features)) 
        features = features[permuted]
        labels = labels[permuted]


# In[62]:


# Create a generator using this function.

train_generator_cyclic = get_generator_cyclic(training_features, training_labels, batch_size=batch_size)


# In[63]:


# Assert that the new cyclic generator does not raise a StopIteration

for i in range(2*train_steps):
    next(train_generator_cyclic)


# In[64]:


# Generate a cyclic validation generator

validation_generator_cyclic = get_generator_cyclic(validation_features, validation_labels, batch_size=batch_size)


# In[65]:


# Train the model

model.fit_generator(train_generator_cyclic, steps_per_epoch=train_steps, 
                    validation_data=validation_generator_cyclic, validation_steps=1, epochs=5)


# #### Evaluate the model and get predictions

# In[66]:


# Let's obtain a validation data generator.

validation_generator = get_generator_cyclic(validation_features, validation_labels, batch_size=30)


# In[67]:


# Get predictions on the validation data

pred = model.predict_generator(validation_generator, steps=1)
print(f'Predictions : {np.round(pred.T[0])}')


# In[68]:


# Print the corresponding validation labels

print(validation_labels)


# In[69]:


# Obtain a validation data generator

validation_generator = get_generator_cyclic(validation_features, validation_labels, batch_size=30)


# In[70]:


# Evaluate the model

model.evaluate_generator(validation_generator, steps=1, verbose=2)


# ***
# <a id="coding_tutorial_3"></a>
# ## Keras image data augmentation

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# #### Load the CIFAR-10 Dataset

# In[2]:


from tensorflow.keras.datasets import cifar10


# In[3]:


# Load the CIFAR-10 dataset

(training_features, training_labels), (test_features, test_labels) = cifar10.load_data()


# In[6]:


# Convert the labels to a one-hot encoding

num_classes = 10

training_labels = tf.keras.utils.to_categorical(training_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)


# #### Create a generator function

# In[7]:


# Create a function that returns a data generator

def get_generator(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield (features[n*batch_size:(n+1)*batch_size], labels[n*batch_size:(n+1)*batch_size])


# In[8]:


# Use the function we created to get a training data generator with a batch size of 1

training_generator = get_generator(training_features, training_labels)


# In[9]:


# Assess the shape of the items generated by training_generator using the `next` function to yield an item.

image, label = next(training_generator)
print(image.shape)
print(label.shape)


# In[10]:


# Test the training generator by obtaining an image using the `next` generator function, and then using imshow to plot it.
# Print the corresponding label

from matplotlib.pyplot import imshow

image, label = next(training_generator)
image_unbatched = image[0,:,:,:]
imshow(image_unbatched)
print(label)


# In[11]:


# Reset the generator by re-running the `get_generator` function.

train_generator = get_generator(training_features, training_labels)


# #### Create a data augmention generator

# In[12]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[17]:


# Create a function to convert an image to monochrome
def monochrome(x):
    def func_bw(a):
        average_colour = np.mean(a)
        return [average_colour, average_colour, average_colour]
    x = np.apply_along_axis(func_bw, -1, x)
    
    return x


# In[18]:


# Create an ImageDataGenerator object

image_generator = ImageDataGenerator(
                    preprocessing_function=monochrome,
                    rotation_range=180,
                    rescale=1/255.
                    )
image_generator.fit(training_features)


# Check [the documentation](https://keras.io/preprocessing/image/) for the full list of image data augmentation options. 

# In[19]:


# Create an iterable generator using the `flow` function

image_generator_iterable = image_generator.flow(training_features, training_labels, shuffle=False, batch_size=1)


# In[20]:


# Show a sample from the generator and compare with the original

image, label = next(image_generator_iterable)
image_orig, label_orig = next(train_generator)
figs, axes = plt.subplots(1,2)
axes[0].imshow(image[0,:,:,:])
axes[0].set_title('Transformed')
axes[1].imshow(image_orig[0,:,:,:])
axes[1].set_title('Original')
plt.show()


# #### Flow from directory

# In[21]:


get_ipython().system('ls -hl data/flowers-recognition-split/train')


# In[22]:


# Inspect the directory structure

train_path = 'data/flowers-recognition-split/train'
val_path = 'data/flowers-recognition-split/val'


# In[23]:


# Create an ImageDataGenerator object

datagenerator = ImageDataGenerator(rescale=(1/255.0))


# In[24]:


classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


# In[25]:


# Create a training data generator

train_generator = datagenerator.flow_from_directory(train_path, classes=classes, batch_size=64, target_size=(16, 16))


# In[26]:


# Create a validation data generator

val_generator = datagenerator.flow_from_directory(val_path, classes=classes, batch_size=64, target_size=(16, 16))


# In[27]:


# Get and display an image and label from the training generator

x = next(train_generator)
imshow(x[0][4])
print(x[1][4])


# In[28]:


# Reset the training generator

train_generator = datagenerator.flow_from_directory(train_path, classes=classes, batch_size=64, target_size=(16, 16))


# #### Create a model to train

# In[29]:


# Build a CNN model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense

model = tf.keras.Sequential()
model.add(Input((16,16,3)))
model.add(Conv2D(8, (8, 8), padding='same', activation='relu'))
model.add(MaxPooling2D((4,4)))
model.add(Conv2D(8, (8, 8), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(4, (4, 4), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='softmax'))


# In[30]:


# Create an optimizer object

optimizer = tf.keras.optimizers.Adam(1e-3)


# In[31]:


# Compile the model

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[32]:


# Print the model summary

model.summary()


# #### Train the model

# In[35]:


train_generator.n


# In[33]:


# Calculate the training generator and test generator steps per epoch

train_steps_per_epoch = train_generator.n // train_generator.batch_size
val_steps = val_generator.n // val_generator.batch_size
print(train_steps_per_epoch, val_steps)


# In[36]:


# Fit the model

model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=5)


# #### Evaluate the model

# In[38]:


# Evaluate the model

model.evaluate_generator(val_generator, steps=val_steps, verbose=2)


# #### Predict using the generator

# In[42]:


# Predict labels with the model

pred = model.predict(val_generator, steps=1)
print(f'prediction\n {np.round(pred, 2)}')


# ***
# <a id="coding_tutorial_4"></a>
# ## The Dataset Class

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


# #### Create a simple dataset

# In[2]:


x = np.zeros((100,10,2,2))


# In[4]:


# Create a dataset from the tensor x

dataset1 = tf.data.Dataset.from_tensor_slices(x)


# In[5]:


# Inspect the Dataset object

print(dataset1)
print(dataset1.element_spec)


# In[6]:


x2 = [np.zeros((10,2,2)), np.zeros((5,2,2))]


# In[7]:


# Try creating a dataset from the tensor x2

dataset2 = tf.data.Dataset.from_tensor_slices(x2)


# In[8]:


x2 = [np.zeros((10,1)), np.zeros((10,1)), np.zeros((10,1))]


# In[9]:


# Create another dataset from the new x2 and inspect the Dataset object

dataset2 = tf.data.Dataset.from_tensor_slices(x2)


# In[10]:


# Print the element_spec

print(dataset2.element_spec)


# #### Create a zipped dataset

# In[12]:


# Combine the two datasets into one larger dataset

dataset_zipped = tf.data.Dataset.zip((dataset1, dataset2))


# In[13]:


# Print the element_spec

print(dataset_zipped.element_spec)


# In[14]:


# Define a function to find the number of batches in a dataset

def get_batches(dataset):
    iter_dataset = iter(dataset)
    i = 0
    try:
        while next(iter_dataset):
            i = i+1
    except:
        return i


# In[15]:


# Find the number of batches in the zipped Dataset

get_batches(dataset_zipped)


# #### Create a dataset from numpy arrays

# In[16]:


# Load the MNIST dataset

(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()

print(type(train_features), type(train_labels))


# In[17]:


# Create a Dataset from the MNIST data

mnist_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))


# In[18]:


# Inspect the Dataset object

print(mnist_dataset.element_spec)


# In[20]:


# Inspect the length of an element using the take method

element = next(iter(mnist_dataset.take(1)))
print(f'Len : {len(element)}')


# In[21]:


# Examine the shapes of the data

print(element[0].shape)
print(element[1].shape)


# #### Create a dataset from text data

# In[22]:


# Print the list of text files

text_files = sorted([f.path for f in os.scandir('data/shakespeare')])

print(text_files)


# In[23]:


# Load the first file using python and print the first 5 lines.

with open(text_files[0], 'r') as fil:
    contents = [fil.readline() for i in range(5)]
    for line in contents:
        print(line)


# In[24]:


# Load the lines from the files into a dataset using TextLineDataset

shakespeare_dataset = tf.data.TextLineDataset(text_files)


# In[25]:


# Use the take method to get and print the first 5 lines of the dataset

first_5_lines_dataset = iter(shakespeare_dataset.take(5))
lines = [line for line in first_5_lines_dataset]
for line in lines:
    print(line)


# In[26]:


# Compute the number of lines in the first file

lines = []
with open(text_files[0], 'r') as fil:
    line = fil.readline()
    while line:
        lines.append(line)
        line = fil.readline()
    print(len(lines))


# In[27]:


# Compute the number of lines in the shakespeare dataset we created

shakespeare_dataset_iterator = iter(shakespeare_dataset)
lines = [line for line in shakespeare_dataset_iterator]
print(len(lines))


# #### Interleave lines from the text data files

# In[28]:


# Create a dataset of the text file strings

text_files_dataset = tf.data.Dataset.from_tensor_slices(text_files)
files = [file for file in text_files_dataset]
for file in files:
    print(file)


# In[29]:


# Interleave the lines from the text files

interleaved_shakespeare_dataset = text_files_dataset.interleave(tf.data.TextLineDataset, cycle_length=9)
print(interleaved_shakespeare_dataset.element_spec)


# In[30]:


# Print the first 10 elements of the interleaved dataset

lines = [line for line in iter(interleaved_shakespeare_dataset.take(10))]
for line in lines:
    print(line)


# ***
# <a id="coding_tutorial_5"></a>
# ## Training with Datasets

# In[31]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# #### Load the UCI Bank Marketing Dataset

# In[54]:


# Load the CSV file into a pandas DataFrame

bank_dataframe = pd.read_csv('data/bank/bank-full.csv', delimiter=';')


# In[55]:


# Show the head of the DataFrame

bank_dataframe.head()


# In[56]:


# Print the shape of the DataFrame

print(bank_dataframe.shape)


# In[57]:


# Select features from the DataFrame

features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
            'loan', 'contact', 'campaign', 'pdays', 'poutcome']
labels = ['y']

bank_dataframe = bank_dataframe.filter(features + labels)


# In[58]:


# Show the head of the DataFrame

bank_dataframe.head()


# #### Preprocess the data

# In[59]:


# Convert the categorical features in the DataFrame to one-hot encodings

from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
categorical_features = ['default', 'housing', 'job', 'loan', 'education', 'contact', 'poutcome']

for feature in categorical_features:
    bank_dataframe[feature] = tuple(encoder.fit_transform(bank_dataframe[feature]))


# In[60]:


# Show the head of the DataFrame

bank_dataframe.head()


# In[61]:


# Shuffle the DataFrame

bank_dataframe = bank_dataframe.sample(frac=1).reset_index(drop=True)


# #### Create the Dataset object

# In[62]:


# dict(bank_dataframe)


# In[63]:


# Convert the DataFrame to a Dataset

bank_dataset = tf.data.Dataset.from_tensor_slices(dict(bank_dataframe))


# In[64]:


# Inspect the Dataset object

print(bank_dataset.element_spec)


# #### Filter the Dataset

# In[65]:


# First check that there are records in the dataset for non-married individuals

def check_divorced():
    bank_dataset_iterable = iter(bank_dataset)
    for x in bank_dataset_iterable:
        if x['marital'] != 'divorced':
            print('Found a person with marital status: {}'.format(x['marital']))
            return
    print('No non-divorced people were found!')

check_divorced()


# In[66]:


# Filter the Dataset to retain only entries with a 'divorced' marital status

bank_dataset = bank_dataset.filter(lambda x : tf.equal(x['marital'], tf.constant([b'divorced']))[0] )


# In[67]:


# Check the records in the dataset again

check_divorced()


# #### Map a function over the dataset

# In[68]:


# Convert the label ('y') to an integer instead of 'yes' or 'no'

def map_label(x):
#     x['y'] = 0 if (x['y'] == 'no') else 1
    x['y'] = 0 if (x['y'] == tf.constant([b'no'], dtype=tf.string)) else 1
    return x

bank_dataset = bank_dataset.map(map_label)


# In[69]:


# Inspect the Dataset object

bank_dataset.element_spec


# In[70]:


# Remove the 'marital' column

bank_dataset = bank_dataset.map(lambda x: {key:val for key,val in x.items() if key != 'martial'})


# In[71]:


# Inspect the Dataset object

bank_dataset.element_spec


# In[ ]:


# bank_dataset


# #### Create input and output data tuples

# In[72]:


# Create an input and output tuple for the dataset

def map_feature_label(x):
    features = [[x['age']], [x['balance']], [x['campaign']], x['contact'], x['default'],
                x['education'], x['housing'], x['job'], x['loan'], [x['pdays']], x['poutcome']]
    return (tf.concat(features, axis=0), x['y'])


# In[74]:


# Map this function over the dataset

bank_dataset = bank_dataset.map(map_feature_label)


# In[75]:


# Inspect the Dataset object

bank_dataset.element_spec


# In[81]:


# next(iter(bank_dataset.take(3)))


# #### Split into a training and a validation set

# In[82]:


# Determine the length of the Dataset

dataset_length = 0
for _ in bank_dataset:
    dataset_length += 1
print(dataset_length)


# In[83]:


# Make training and validation sets from the dataset

training_elements = int(dataset_length * 0.7)
train_dataset = bank_dataset.take(training_elements)
val_dataset = bank_dataset.skip(training_elements)


# #### Build a classification model
# 
# Now let's build a model to classify the features.

# In[84]:


# Build a classifier model

from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras import Sequential

model = Sequential()
model.add(Input(shape=(30,)))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(1, activation='sigmoid'))


# In[85]:


# Compile the model

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# In[86]:


# Show the model summary

model.summary()


# #### Train the model

# In[87]:


# Create batched training and validation datasets

train_dataset = train_dataset.batch(20, drop_remainder=True)
val_dataset = val_dataset.batch(100)


# In[89]:


# Shuffle the training data

train_dataset = train_dataset.shuffle(1000)


# In[95]:


# Fit the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)


# In[96]:


# Plot the training and validation accuracy

plt.plot(history.epoch, history.history['accuracy'], label='training')
plt.plot(history.epoch, history.history['val_accuracy'], label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')


# In[ ]:




