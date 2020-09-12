import tensorflow as tf

x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2
    grad = tape.gradient(y, x)
print(grad)     # tf.Tensor(4.0, shape=(), dtype=float32)

x = tf.constant([1,2,3,4,5], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x**2)
    z = tf.math.sin(y)
    dz_dy = tape.gradient(z, y)
print("------------> dz_dy",dz_dy)      # dz_dy tf.Tensor(0.022126757, shape=(), dtype=float32)

x = tf.constant([1,2,3,4,5], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x**2)
    z = tf.math.sin(y)
    dz_dy, dz_dx = tape.gradient(z, [y,x])
print("------------> dz_dy",dz_dy)      # dz_dy tf.Tensor(0.022126757, shape=(), dtype=float32)
print("------------> dz_dx",dz_dx)      # dz_dx tf.Tensor([0.04425351 0.08850703 0.13276054 0.17701405 0.22126757], shape=(5,), dtype=float32)

import tensorflow as tf
x = tf.constant([-1, 0, 1], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.math.exp(x)
    z = 2 * tf.reduce_sum(y)
    dz_dx = tape.gradient(z, x)
print("------------> dz_dx",dz_dx)      # dz_dx tf.Tensor([0.7357589 2.000  5.4365635], shape=(3,), dtype=float32)