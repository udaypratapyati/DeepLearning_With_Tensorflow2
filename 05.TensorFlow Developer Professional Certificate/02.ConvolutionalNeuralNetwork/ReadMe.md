# Convolutional Neural Networks in TensorFlow
- Best practices for using TensorFlow, a popular open-source framework for machine learning.
- Advanced techniques to improve the computer vision model. 
- How to work with real-world images in different shapes and sizes, visualize the journey of an image through convolutions to understand how a computer “sees” information, plot loss and accuracy, and explore strategies to prevent overfitting, including augmentation and dropout. 
- Transfer learning and how learned features can be extracted from models. 

## 01.Exploring a Larger Dataset
- Go deeper into using ConvNets will real-world data
- Techniques that can improve ConvNet performance, particularly when doing image classification!
	- The Cats and Dogs dataset which had been a Kaggle Challenge in image classification!
	
## 02.Augmentation: A technique to avoid overfitting
- Overfitting is simply the concept of being over specialized in training -- namely that your model is very good at classifying what it is trained for, but not so good at classifying things that it hasn't seen. 
- In order to generalize your model more effectively, we will of course need a greater breadth of samples to train it on.
- That's not always possible, but a nice potential shortcut to this is Image Augmentation, where you tweak the training set to potentially increase the diversity of subjects it covers. 

## 03.Transfer Learning
- Building models is great, and can be very powerful. But, we can be limited by the data we have on hand. 
- Not everybody has access to massive datasets or the compute power that's needed to train them effectively. 
- Transfer learning can help solve this -- where people with models trained on large datasets train them, so that you can either use them directly, or, we can use the features that they have learned and apply them to your scenario. 

## 04.Multiclass Classifications
When moving beyond binary into Categorical classification there are some coding considerations we need to take into account.