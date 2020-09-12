import tensorflow as tf 
from keras.layers import Dense
from keras.models import Sequential

model = Sequential([
    Dense(1, input_shape = (4,))
])

print(model.weights)

# Specify input shapes  and weights of model will be initialized straight away.
# Its a list and it got two elements, one for kernel of the dense layer and one for bias.
# These are model parameters and will be trained when we use model.fit
# Variables is represented as numpy array.

# [<tf.Variable 'dense_1/kernel:0' shape=(4, 1) dtype=float32, numpy=
# array([[ 0.48951805],
#        [-0.3798477 ],
#        [-0.7528247 ],
#        [-0.7871945 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

my_var = tf.Variable([-1,2], dtype = tf.float32, name = 'my_var')
my_var.assign([3.5, -1.])
print(my_var)

# <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([-1,  2])>
# <tf.Variable 'my_var:0' shape=(2,) dtype=float32, numpy=array([-1.,  2.], dtype=float32)>
# Just note the name of the variable if name is properly defined.
# we can assign the variables manually also.
# While assigning the variables, it has to be in right shape.
# <tf.Variable 'my_var:0' shape=(2,) dtype=float32, numpy=array([ 3.5, -1. ], dtype=float32)>
# We can notice variable is assigned new values now.

x = my_var.numpy() # array([3.5, -1.], dtype = float32)
# We can also directly convert the variable to numpy array using numpy method.