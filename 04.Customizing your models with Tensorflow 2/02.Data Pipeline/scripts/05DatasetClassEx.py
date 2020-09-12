import tensorflow as tf 
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

print(dataset.element_spec)

# (TensorSpec(shape=(32, 32, 3), dtype=tf.uint8, name=None), 
# TensorSpec(shape=(1,), dtype=tf.uint8, name=None))

# The first dimension is 50000 for both x_train and y_train as per requirement of tuple
# input shape is 32x32x3 as we would expect.
# output length is one.
# We could also iterate over this dataset to pull out each of the data examples.

from keras.preprocessing.image import ImageDataGenerator

img_datagen = ImageDataGenerator(width_shift_range=0.2, horizontal_flip=True)

dataset2 = tf.data.Dataset.from_generator(
    img_datagen.flow, args=[x_train, y_train],
    output_types = (tf.float32, tf.int32),
    output_shapes = ([32,32,32,3], [32,1])
)

print(dataset2.element_spec)

# (TensorSpec(shape=(32, 32, 32, 3), dtype=tf.float32, name=None), 
# TensorSpec(shape=(32, 1), dtype=tf.int32, name=None))

# Notice we are using different static method to do this.
# we are using from_generator method because we are going to pass in a generator
# instead of tensors or numpy arrays.

# This method takes a callable object in first argument that returns a generator when called
# args keyword argument gives argument that should be passed into this callable.
# We also need to specify the output types as TensorFlow types. 
# So that's float32 for the images and int32 for the labels. 
# In last line, we are giving expected output shapes for a batch of inputs and outputs.
# ImageDataGenerator flow method setsup batch size of 32 by default.
# So we are specifying 32 as batch size.