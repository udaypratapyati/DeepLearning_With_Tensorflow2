import tensorflow as tf 
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

def rescale(image, label):
    return image/255, label

def label_filter(image, label):
    return tf.squeeze(label) != 9

dataset = dataset.map(rescale)
dataset = dataset.filter(label_filter)

dataset = dataset.shuffle(100)
dataset = dataset.batch(16, drop_remainder=True)
dataset = dataset.repeat(10)

# history = model.fit(dataset, steps_per_epoch=x_train.shape[0]//16, epochs=10)

print(dataset.element_spec)

# (TensorSpec(shape=(16, 32, 32, 3), dtype=tf.uint8, name=None), 
# TensorSpec(shape=(16, 1), dtype=tf.uint8, name=None))

# We can batch data examples by using dataset.batch method.
# Now if I iterate over my dataset, I will be extracting a batch of images and labels.
# Drop remainder argument True means that we get batch size of 16 each time.
# Running this model.fit call will train the model for one epoch all one complete 
# pass through the data set object.

# If we want to train for 10 epochs we could do this.
# dataset.repeat method will do the same.
# If I remove argument from repeat, dataset will repeat indefinitely.
# I can set steps_per_epoch argument so training process knows when an epoch has ended.
# Because my batch_size is 16, the number of steps per epoch 
# will be no. of samples in my training set divided by 16.

# Also, I have set the epochs keyword argument to 10,
# So model will train for 10 complete passes through the training set.

# we might also want to shuffle our dataset randomly.
# it requires one integer argument.
# The way it works is the buffer will stay filled with 100 data examples and
# the batch of 16 will be sampled from the buffer.

# We can also apply transformations to dataset objects for pre-processing
# or filter out certain examples in the data that we don't want to use for training.

# For ex: We can define function rescale which normalizes pixel value of image.
# We are then applying this function to every dataset element with map method.

# We can also filter our certain data set examples that we dont want using filter method.
# Function returns a boolean, the label is actually a one dimensional tensor in this dataset.
# So we are using tf.squeeze to convert it into a scalar.