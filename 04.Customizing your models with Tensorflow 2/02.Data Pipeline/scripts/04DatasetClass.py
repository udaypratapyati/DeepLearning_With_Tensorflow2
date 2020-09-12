# The tf.data API gives a unified way of handling datasets of anytype
# and data that might come from range of different sources.
# The main object that we're going to be using in the tf.data module is the dataset class. 
# This is the main abstraction that we can use to handle our datasets, 
# whatever form or size they might come in and 
# apply any necessary preprocessing, or filtering that we might want to do. 

import tensorflow as tf 

dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
dataset2 = tf.data.Dataset.from_tensor_slices([[1,2], [3,4], [5,6]])
dataset3 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([128, 5]))

dataset4 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([256, 4], minval = 1, maxval = 10, dtype=tf.int32),
    tf.random.normal([256])))

for elem in dataset2:
    print(elem.numpy())

print(dataset)
print(dataset2)
print(dataset3.element_spec)
print(dataset4.element_spec)

for element in dataset4.take(2):
    print(element)

# <TensorSliceDataset shapes: (), types: tf.int32>
# <TensorSliceDataset shapes: (2,), types: tf.int32>

# tf.Tensor([1 2], shape=(2,), dtype=int32)
# tf.Tensor([3 4], shape=(2,), dtype=int32)
# tf.Tensor([5 6], shape=(2,), dtype=int32)

# [1 2] # Use elem.numpy()
# [3 4]
# [5 6]

# TensorSpec(shape=(5,), dtype=tf.float32, name=None)

# (TensorSpec(shape=(4,), dtype=tf.int32, name=None), 
# TensorSpec(shape=(), dtype=tf.float32, name=None))

# (<tf.Tensor: id=31, shape=(4,), dtype=int32, numpy=array([6, 1, 3, 8])>, 
# <tf.Tensor: id=32, shape=(), dtype=float32, numpy=-0.14862704>)
# (<tf.Tensor: id=33, shape=(4,), dtype=int32, numpy=array([5, 4, 8, 1])>, 
# <tf.Tensor: id=34, shape=(), dtype=float32, numpy=-0.25700518>)

# Above code returns a dataset object by using from_tensor_slices method
# A list is passed as an argument in from_tensor_slices method.
# We can see TensorSliceDataset object as an output.
# Shape tells us the shape of each individual element.
# In this case, each eleement is a scalar.

# If we pass a list of lists like this, then dataset would read each of these sublists as an individual data element.
# Shape of each element is now two.

# When we create a set object, the object is iterable, and that means 
# we can easily access each element in the dataset by writing a simple for loop.

# We can also get numpy array for each data element by calling numpy method on tensor element.

# If we pass a tensor rather than list or list of lists,
# then dataset will always take the first dimension as the dataset size.
# So this dataset will have 128 elements, each of length five, 
# with random values sampled from a uniform distribution.

# So this dataset will have 128 elements each of length 5 with random values
# sampled from uniform distribution. We can inspect using element_spec property.

# If we pass tuple of tensors to create dataset, thats not a problem.
# In fact, its very common whenever we want to create dataset with input and output data.

# Since first dimension is treated as dataset size so each tensor in tuple
# should have same first dimension as 256 in above example. Else error will come.

# The inputs telling four with random integer values from one to nine.
# Outputs are scalars with values randomly sampled from a standard normal distribution.
# We can inspect the same with element_spec property.

# We can use dataset.take method to extract first two elements of the dataset.
# Each example has four input of integers from one to nine and scalar floats output.