import tensorflow as tf 

from keras.models import Model

x = tf.constant(2.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2
    grad = tape.gradient(y,x)

print(grad)

# tf.Tensor(4.0, shape=(), dtype=float32)

x1 = tf.constant([0, 1, 2, 3], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x1)
    y1 = tf.reduce_sum(x1**2)
    z = tf.math.sin(y1)
    dz_dy1, dz_dx1 = tape.gradient(z, [y1,x1])

print(dz_dy1)
print(dz_dx1)

# tf.Tensor(0.13673721, shape=(), dtype=float32)
# tf.Tensor([0.         0.27347443 0.54694885 0.82042325], shape=(4,), dtype=float32)