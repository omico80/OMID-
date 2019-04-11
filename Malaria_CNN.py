### Convolutional Neural Networks with "keras"
## Malaria Detection 
# Kaggle Database 

# importing the necessary libraries for wrangling and manipulation 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pd 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

# importing required libraries for the CNN
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dropout 
from keras.layers import BatchNormalization 

# to initialize the CNN by using the sequential 
classifier = Sequential()

#STEP 1A : convolution 
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = "relu"))

# STEP 2A : Maxpooling 
classifier.add(MaxPooling2D(pool_size =(2,2)))

# STEP 2B : Adding another convolution layer 
classifier.add(Convolution2D(32,3,3,activation = "relu"))

# STEP 2B : Adding another Max Pooling layer to increase the accuracy 
classifier.add(MaxPooling2D(pool_size =(2,2)))
# STEP 3 : Flatten 
classifier.add(Flatten())

#STEP 4 : Fully connected network construction 
classifier.add(Dense(output_dim = 128, activation ="relu"))
classifier.add(Dense(output_dim = 1, activation ="sigmoid"))

# STEP 5 : using the "Stochastic Gradient Function" to compile the layers 
classifier.compile(optimizer = "adam", loss = "binary_crossentropy",metrics=["accuracy"])

### STEP 6 : Image Pre_processing 
## To fit all the images to the CNN model built already 
# Image Augmentation &  Image Enrichment  to avoid overfitting 
# Import the library & class to allow for image augmentation 
# The importation of the images is done through the process of the augentation 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=21998, # number of the images in the test set 
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=5560)# number of the images in the training set 




flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, 
                    class_mode='categorical', batch_size=32, shuffle=True, seed=None, 
                    save_to_dir=None,
                    save_prefix='',
                    save_format='png', follow_links=False, subset=None, interpolation='nearest')



