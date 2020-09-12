#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# # The Keras functional API

#  ## Coding tutorials
#  #### [1. Multiple inputs and outputs](#coding_tutorial_1)
#  #### [2. Tensors and Variables](#coding_tutorial_2)
#  #### [3. Accessing model layers](#coding_tutorial_3)
#  #### [4. Freezing layers](#coding_tutorial_4)

# ***
# <a id="coding_tutorial_1"></a>
# ## Multiple inputs and outputs

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Load the acute inflammations dataset
# 
# The `acute inflammations` was created by a medical expert as a data set to test the expert system, which will perform the presumptive diagnosis of two diseases of the urinary system. You can find out more about the dataset [here](https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations).
# 
# Attribute information:
# 
# Inputs:
# - Temperature of patient : 35C-42C
# - Occurrence of nausea : yes/no
# - Lumbar pain : yes/no
# - Urine pushing (continuous need for urination) : yes/no
# - Micturition pains : yes/no
# - Burning of urethra, itch, swelling of urethra outlet : yes/no
# 
# Outputs:
# - decision 1: Inflammation of urinary bladder : yes/no
# - decision 2: Nephritis of renal pelvis origin : yes/no

# #### Load the data

# In[3]:


# Load the dataset

from sklearn.model_selection import train_test_split

pd_dat = pd.read_csv('data/diagnosis.csv')
dataset = pd_dat.values


# In[4]:


# Build train and test data splits

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,:6], dataset[:,6:], test_size=0.33)


# In[14]:


print(X_train[:5])
print(Y_train[0])


# In[11]:


# Assign training and testing inputs/outputs

temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train = np.transpose(X_train)
temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test = np.transpose(X_test)

inflam_train, nephr_train = Y_train[:, 0], Y_train[:, 1]
inflam_test, nephr_test = Y_test[:, 0], Y_test[:, 1]


# In[12]:


temp_train


# #### Build the model

# In[41]:


# Build the input layers

from tensorflow.keras.layers import Input

ip_shape = (1,)
temperature = Input(shape=ip_shape, name='temp')
nausea_occurence = Input(shape=ip_shape, name='nocc')
lumbar_pain = Input(shape=ip_shape, name='lumbp')
urine_pushing = Input(shape=ip_shape, name='up')
micturition_pains = Input(shape=ip_shape, name='mict')
bis = Input(shape=ip_shape, name='bis')


# In[42]:


# Create a list of all the inputs

list_inputs = [temperature, nausea_occurence, lumbar_pain, urine_pushing, 
               micturition_pains, bis]


# In[43]:


# Merge all input features into a single large vector

x = tf.keras.layers.concatenate(list_inputs)


# In[44]:


# Use a logistic regression classifier for disease prediction

inflammation_pred = tf.keras.layers.Dense(1, activation='sigmoid', name='inflam')(x)
nephritis_pred = tf.keras.layers.Dense(1, activation='sigmoid', name='nephr')(x)


# In[45]:


# Create a list of all the outputs

list_outputs = [inflammation_pred, nephritis_pred]


# In[46]:


# Create the model object

model = tf.keras.Model(inputs=list_inputs, outputs=list_outputs)


# #### Plot the model

# In[47]:


# Display the multiple input/output model

tf.keras.utils.plot_model(model, 'multi_input_output_model.png', show_shapes=True)


# #### Compile the model

# In[48]:


# Compile the model

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={'inflam':tf.keras.losses.BinaryCrossentropy(),
         'nephr': 'binary_crossentropy'},
    metrics={'inflam':['accuracy'],
         'nephr': ['accuracy']},
    loss_weights=[1., .2]
)


# #### Fit the model 

# In[49]:


# Define training inputs and outputs

inputs_train = {'temp': temp_train, 'nocc': nocc_train, 'lumbp': lumbp_train,
                'up': up_train, 'mict': mict_train, 'bis': bis_train}

outputs_train = {'inflam': inflam_train, 'nephr': nephr_train}


# In[55]:


# Train the model

history = model.fit(inputs_train, outputs_train, epochs=1000, batch_size=128, verbose=False)


# In[51]:


model.summary()


# #### Plot the learning curves

# In[52]:


history.history.keys()


# In[56]:


# Plot the training accuracy

acc_keys = [k for k in history.history.keys() if k in ('inflam_accuracy', 'nephr_accuracy')] 
loss_keys = [k for k in history.history.keys() if not k in acc_keys]

print(acc_keys)
print(loss_keys)

for k, v in history.history.items():
    if k in acc_keys:
        plt.figure(1)
        plt.plot(v)
    else:
        plt.figure(2)
        plt.plot(v)

plt.figure(1)
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(acc_keys, loc='upper right')

plt.figure(2)
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loss_keys, loc='upper right')

plt.show()


# In[57]:


# Evaluate the model

model.evaluate([temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test], 
               [inflam_test, nephr_test], verbose=2)


# In[60]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten

inputs = Input(shape=(16, 16, 3))
h = Conv2D(32, 3, activation='relu')(inputs)
h = AveragePooling2D(3)(h)
outputs = Flatten()(h)
Model = Model(inputs=inputs, outputs=outputs)
print(inputs)
print(h)
print(outputs)
print(Model)
print(Model.inputs)
print(Model.outputs)
Model.summary()


# ***
# <a id="coding_tutorial_2"></a>
# ## Tensors and Variables

# In[68]:


import numpy as np


# #### Create Variable objects

# In[69]:


# Create Variable objects of different type with tf.Variable

strings = tf.Variable(["Hello world!"], tf.string)
floats  = tf.Variable([3.14159, 2.71828], tf.float64)
ints = tf.Variable([1, 2, 3], tf.int32)
complexs = tf.Variable([25.9 - 7.39j, 1.23 - 4.91j], tf.complex128)


# In[72]:


# Initialise a Variable value

tf.Variable(tf.constant(5, shape=(5,5)))


# #### Use and modify Variable values

# In[80]:


# Use the value of a Variable

v = tf.Variable(0.0)
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.

print(type(w))
print(v, w)


# In[81]:


# Increment the value of a Variable

# v = v + 1
# print(v)

v.assign_add(1)
print(v)


# In[82]:


# Decrement the value of a Variable

v.assign_sub(1)
print(v)


# #### Create Tensor objects

# Create a constant tensor and print its type as well as its shape:

# In[83]:


# Create a constant Tensor

x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
print("dtype:", x.dtype)
print("shape:", x.shape)


# In[84]:


# Obtain the value as a numpy array

x.numpy()


# In[85]:


# Create a Tensor of type float32

x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
print(x)


# In[86]:


# Create coefficients

coeffs = np.arange(16)


# In[90]:


# Initialise shapes

shape1 = (4,4)
shape2 = (2,8)
shape3 = (2,2,2,2)


# In[91]:


# Create Tensors of different shape

a = tf.constant(coeffs, shape=shape1)
print("\n a:\n ", a)

b = tf.constant(coeffs, shape=shape2)
print("\n b:\n ", b)

c = tf.constant(coeffs, shape=shape3)
print("\n c:\n ", c)


# #### Useful Tensor operations

# In[93]:


# Create a constant Tensor

t = tf.constant(np.arange(80), shape=[5,2,8])
print(t)


# In[97]:


# Get the rank of a Tensor

rank = tf.rank(t)


# In[98]:


# Display the rank

print("rank: ", rank)


# In[103]:


# Reshape a Tensor

t2 = tf.reshape(t, (10,8))


# In[104]:


# Display the new shape

print("t2.shape: ", t2.shape)


# In[110]:


# Create ones, zeros, identity and constant Tensors
ones = tf.ones(shape=(2,4))
zeros = tf.zeros(shape=(3,))
eye = tf.eye(4)
tensor7 = tf.constant(8., shape=[2,2])


# In[111]:


# Display the created tensors

print("\n Ones:\n ", ones)
print("\n Zeros:\n ", zeros)
print("\n Identity:\n ", eye)
print("\n Tensor filled with 7: ", tensor7)


# In[114]:


# Create a ones Tensor and a zeros Tensor

t1 = tf.ones(shape=(2, 2))
t2 = tf.zeros(shape=(2, 2))


# In[116]:


# Concatentate two Tensors

concat0 = tf.concat([t1, t2], 0)
concat1 = tf.concat([t1, t2], 1)


# In[117]:


# Display the concatenated tensors

print(concat0)
print(concat1)


# In[118]:


# Create a constant Tensor

t = tf.constant(np.arange(24), shape=(3, 2, 4))
print("\n t shape: ", t.shape)


# In[119]:


# Expanding the rank of Tensors

t1 = tf.expand_dims(t, 0)
t2 = tf.expand_dims(t, 1)
t3 = tf.expand_dims(t, 3)


# In[120]:


# Display the shapes after tf.expand_dims

print("\n After expanding dims:\n t1 shape: ", t1.shape, "\n t2 shape: ", t2.shape, "\n t3 shape: ", t3.shape)


# In[123]:


# Squeezing redundant dimensions

t1 = tf.squeeze(t1, 0)
t2 = tf.squeeze(t2, 1)
t3 = tf.squeeze(t3, 3)


# In[124]:


# Display the shapes after tf.squeeze

print("\n After squeezing:\n t1 shape: ", t1.shape, "\n t2 shape: ", t2.shape, "\n t3 shape: ", t3.shape)


# In[126]:


# Slicing a Tensor

t = tf.constant([1,2,3,4,5,6])
print(t[1:4])


# #### Doing maths with Tensors

# In[127]:


# Create two constant Tensors

c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])


# In[128]:


# Matrix multiplication

matmul_cd = tf.matmul(c, d)


# In[129]:


# Display the result

print("\n tf.matmul(c,d):\n", matmul_cd)


# In[130]:


# Elementwise operations

c_times_d = c*d
c_plus_d = c+d
c_minus_d = c-d
c_div_c = c/d


# In[131]:


# Display the results

print("\n c*d:\n", c_times_d)
print("\n c+d:\n", c_plus_d)
print("\n c-d:\n", c_minus_d)
print("\n c/c:\n", c_div_c)


# In[138]:


# Create Tensors

a = tf.constant([[2, 3], [3, 3]])
b = tf.constant([[8, 7], [2, 3]])
x = tf.constant([[-6.89 + 1.78j], [-2.54 + 2.15j]])


# In[139]:


# Absolute value of a Tensor

absx = tf.abs(x)


# In[142]:


# Power of a Tensor

powab = tf.pow(a, b)


# In[143]:


# Display the results

print("\n ", absx)
print("\n ", powab)


# #### Randomly sampled constant tensors

# In[145]:


# Create a Tensor with samples from a Normal distribution

tn = tf.random.normal((2,2), mean=.5, stddev=1.)
print(tn)


# In[146]:


# Create a Tensor with samples from a Uniform distribution

tu = tf.random.uniform((2,1), minval=1, maxval=10)
print(tu)


# In[149]:


# Create a Tensor with samples from a Poisson distribution

tp = tf.random.poisson((2,2), 10)
print(tp)


# In[150]:


# More maths operations

d = tf.square(tn)
e = tf.exp(d)
f = tf.cos(c)
print(d)
print(e)
print(f)


# ***
# <a id="coding_tutorial_3"></a>
# ## Accessing model layers

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Load the pre-trained model

# In this section, we aim to demonstrate accessing layer attributes within a model.
# 
# Let's get started by loading the `VGG19` pre-trained model from the `keras.applications` library, which is a very deep network trained on more than a million images from the ImageNet database. The network is trained to classify images into 1000 object categories.

# In[2]:


# Load the VGG19 model

from tensorflow.keras.applications import VGG19
# vgg_model = VGG19()

from tensorflow.keras.models import load_model
vgg_model = load_model('models/Vgg19.h5')


# In[3]:


# Get the inputs, layers and display the summary

vgg_input = vgg_model.input
vgg_layers = vgg_model.layers
vgg_model.summary()


# #### Build a model to access the layer outputs

# In[4]:


from tensorflow.keras.models import Model


# In[14]:


# Build a model that returns the layer outputs

layerOutputs = [layer.output for layer in vgg_layers]
features = Model(inputs=vgg_input, outputs=layerOutputs)


# In[19]:


# Plot the model
import tensorflow as tf
tf.keras.utils.plot_model(features, 'vgg19_model.png', show_shapes=True)


# In[21]:


# Test the model on a random input

img = np.random.random((1,224,224,3)).astype('float32')
extracted_feautures = features(img)
# extracted_feautures


# #### Load the 'cool cat' picture

# In Zambia’s South Luangwa National Park, a photographer had been watching a pride of lions while they slept off a feast from a buffalo kill. When this female walked away, he anticipated that she might be going for a drink and so he positioned his vehicle on the opposite side of the waterhole. The `cool cat` picture is one of the highly commended 2018 Image from Wildlife Photographer of the Year.

# In[22]:


# Display the original image

import IPython.display as display
from PIL import Image

display.display(Image.open('data/cool_cat.jpg'))


# #### Visualise network features from the input image

# In[23]:


# Preprocess the image

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

img_path = 'data/cool_cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# In[68]:


# Extract the features

extracted_feautures = features.predict(x) # features is the name of the model


# In[71]:


# Visualise the input channels
f1 = extracted_feautures[0]
print(f1.shape)

imgs = f1[0,:,:,]
plt.figure(figsize=(15,15))

for n in range(3):
    ax = plt.subplot(1, 3, n+1)
    plt.imshow(imgs[:,:,n])
    plt.axis('off')
plt.subplots_adjust(wspace=.01, hspace=.01)    
    


# In[53]:


def plot_images(extracted_feature):
    imgs = extracted_feature[0,:,:,]
    print(imgs.shape)
    plt.figure(figsize=(15,15))

    for n in range(16):
        ax = plt.subplot(4, 4, n+1)
        plt.imshow(imgs[:,:,n])
        plt.axis('off')
    plt.subplots_adjust(wspace=.01, hspace=.01)    
    
plot_images(extracted_feautures[1])


# In[54]:


# Visualise some features in the first hidden layer

plot_images(extracted_feautures[1])


# In[56]:


# Build a model to extract features by layer name

extracted_feautures_block1_pool = Model(inputs=features.input, outputs=features.get_layer('block1_pool').output)
pred = extracted_feautures_block1_pool.predict(x) # x is the processed cool cat image


# In[65]:


# Visualise some features from the extracted layer output
pred.shape
plot_images(pred)


# In[66]:


# Extract features from a layer deeper in the network

extracted_feautures_block5_conv4 = Model(inputs=features.input, outputs=features.get_layer('block5_conv4').output)
pred = extracted_feautures_block5_conv4.predict(x) # x is the processed cool cat image


# In[67]:


# Visualise some features from the extracted layer output

plot_images(pred)


# ***
# <a id="coding_tutorial_4"></a>
# ## Freezing layers

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Build the model

# In[2]:


# Build a small Sequential model

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential([
    layers.Dense(4, input_shape=(4,), activation='relu', kernel_initializer='random_uniform',
                 bias_initializer='ones'),
    layers.Dense(2, activation='relu', kernel_initializer='lecun_normal', bias_initializer='ones'),
    layers.Dense(4, activation='softmax'),
])


# In[3]:


# Display the model summary

model.summary()


# #### Examine the weight matrix variation over training

# In[23]:


def getWeights(model):
    return [e.weights[0].numpy() for e in model.layers]

def getBiases(model):
    return [e.bias.numpy() for e in model.layers]

def plotDeltaWeights(W0_layers, b0_layers, W1_layers, b1_layers):
    plt.figure(figsize=(8,8))
    for n in range(3):
        delta_l = W1_layers[n] - W0_layers[n]
        print('Layer '+str(n)+': bias variation: ', np.linalg.norm(b1_layers[n] - b0_layers[n]))
        ax = plt.subplot(1,3,n+1)
        plt.imshow(delta_l)
        plt.title('Layer '+str(n))
        plt.axis('off')
    plt.colorbar()
    plt.suptitle('Weight matrices variation');    


# In[ ]:





# In[24]:


# Retrieve the weights and biases

# W0_layers = [e.weights[0].numpy() for e in model.layers]
# b0_layers = [e.bias.numpy() for e in model.layers]
W0_layers = getWeights(model)
b0_layers = getBiases(model)


# In[25]:


# Construct a synthetic dataset

x_train = np.random.random((100, 4))
y_train = x_train

x_test = np.random.random((20, 4))
y_test = x_test


# In[26]:


# Compile and fit the model

model.compile(optimizer='adam',
              loss='mse',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=50, verbose=False);


# In[27]:


# Retrieve weights and biases

# W1_layers = [e.weights[0].numpy() for e in model.layers]
# b1_layers = [e.bias.numpy() for e in model.layers]
W1_layers = getWeights(model)
b1_layers = getBiases(model)


# In[28]:


plotDeltaWeights(W0_layers, b0_layers, W1_layers, b1_layers)


# In[29]:


# # Plot the variation

# plt.figure(figsize=(8,8))
# for n in range(3):
#     delta_l = W1_layers[n] - W0_layers[n]
#     print('Layer '+str(n)+': bias variation: ', np.linalg.norm(b1_layers[n] - b0_layers[n]))
#     ax = plt.subplot(1,3,n+1)
#     plt.imshow(delta_l)
#     plt.title('Layer '+str(n))
#     plt.axis('off')
# plt.colorbar()
# plt.suptitle('Weight matrices variation');


# #### Freeze layers at build time

# In[30]:


# Count the trainable and non trainable variables before the freezing

n_trainable_variables = len(model.trainable_variables)
n_non_trainable_variables = len(model.non_trainable_variables)


# In[31]:


# Display the number of trainable and non trainable variables before the freezing

print("\n Before freezing:\n\t Number of trainable variables: ", n_trainable_variables,
                         "\n\t Number of non trainable variables: ", n_non_trainable_variables)


# In[33]:


# Build the model

model = Sequential([
    layers.Dense(4, input_shape=(4,), activation='relu', kernel_initializer='random_uniform',
                 bias_initializer='ones', trainable=False),
    layers.Dense(2, activation='relu', kernel_initializer='lecun_normal', bias_initializer='ones'),
    layers.Dense(4, activation='softmax'),
])


# In[34]:


# Count the trainable and non trainable variables after the freezing

n_trainable_variables = len(model.trainable_variables)
n_non_trainable_variables = len(model.non_trainable_variables)


# In[35]:


# Display the number of trainable and non trainable variables after the freezing

print("\n After freezing:\n\t Number of trainable variables: ", n_trainable_variables,
                         "\n\t Number of non trainable variables: ", n_non_trainable_variables)


# In[36]:


# Retrieve weights and biases

W0_layers = getWeights(model)
b0_layers = getBiases(model)


# In[37]:


# Compile and fit the model

model.compile(optimizer='adam',
              loss='mse',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=50, verbose=False);


# In[38]:


# Retrieve weights and biases

W1_layers = getWeights(model)
b1_layers = getBiases(model)


# In[40]:


# Plot the variation

plotDeltaWeights(W0_layers, b0_layers, W1_layers, b1_layers)


# #### Freeze layers of a pre-built model

# In[41]:


# Count the trainable and non trainable variables before the freezing

print("\n Before freezing:\n\t Number of trainable variables: ", len(model.trainable_variables),
                         "\n\t Number of non trainable variables: ", len(model.non_trainable_variables))


# In[43]:


# Freeze the second layer

model.layers[1].trainable = False


# In[44]:


# Count the trainable and non trainable variables after the freezing

print("\n After freezing:\n\t Number of trainable variables: ", len(model.trainable_variables),
                        "\n\t Number of non trainable variables: ", len(model.non_trainable_variables))


# In[45]:


# Compile and fit the model

model.compile(optimizer='adam',
              loss='mse',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=50, verbose=False);


# In[46]:


# Retrieve weights and biases

W2_layers = getWeights(model)
b2_layers = getBiases(model)


# In[47]:


# Plot the variation

plotDeltaWeights(W2_layers, b2_layers, W1_layers, b1_layers)

