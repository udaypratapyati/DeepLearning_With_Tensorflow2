import tensorflow as tf

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Dataset will be saved in local folder as below
# ~/.keras/datasets/mnist.npz
# if dataset is already loaded in this folder, it will not load again.

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ~/.keras/datasets/cifar-10-batches.py/
# ~/.keras/datasets/cifar-10-batches.py.tar.gz

from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000,
                                                    maxlen=100)