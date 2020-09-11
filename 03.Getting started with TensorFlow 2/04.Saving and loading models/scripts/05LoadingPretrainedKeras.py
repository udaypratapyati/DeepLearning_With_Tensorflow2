from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np 

model = ResNet50(weights='imagenet', include_top=True) #/.keras/models/

img_input = image.load_img('my_picture.jpg', target_size=(224,224))
img_input = image.img_to_array(img_input)
img_input = preprocess_input(img_input[np.newaxis,...])

preds = model.predict(img_input)
decode_predictions = decode_predictions(preds, top =3)[0]

# List of (class, description, probability)

# If weights option set to imagenet, weights trained on imagenet will be loaded
# If weights = None, weights would be randomly initialized.

# Fully connected layer at include_top would not be included if set to False
# It can be used to develop applications like transfer learning.

# np.newaxis is used for number of samples in the batch size.
# Though we are supplying one image but still we need np.newaxis argument

# Model.predict will predict the numpy array
# top3 model predictions only.