import tensorflow as tf 

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

image_data_gen = ImageDataGenerator(rescale=1/255., 
                                    horizontal_flip=True,
                                    height_shift_range=0.2,
                                    fill_mode='nearest',
                                    featurewise_center=True)

image_data_gen.fit(x_train)

train_datagen = image_data_gen.flow(x_train, y_train, batch_size=16)

# model.fit_generator(train_datagen, epochs=20)

# ImageDataGenerator is preprocessing technique on the fly
# rescale puts all pixel values between 0 and 1.
# Horizontal_flip doublts the size of the dataset.
# height_shift_range means that each image will be randomly shifted up or down
# upto 20 percent of the total height of the image.
# When you do above step, it means some pixels needs to be filled in.
# fill_mode means method to fill the values and default is nearest.
# featurewise_center means standardize the dataset and mean of each individual feature is zero
# So for an RGB Image, we would have three features and these three features would be
# standardized across the whole dataset.

# To do this, ImageDataGenerator needs to calculate the dataset feature means first
# before it can start generating pre-processed samples.

# So if we are using standardization technique like this, 
# we then need to run the fit method of the ImageDataGenerator on the training dataset
# or at least a sample of it.

# We can then get the generator itself by using the flow method and passing in
# the training images and labels.