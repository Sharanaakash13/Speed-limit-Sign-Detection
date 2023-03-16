#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import itertools
import io
import glob
import os
from PIL import Image
import tensorflow as tf
import sklearn.metrics
from skimage.util import random_noise
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.models import load_model


# In[ ]:


# The path wich contains all the Images
path = glob.glob(r"Write the path in which all the images are there\*.jpg")

# Defining two lists for saving the images and labels
Image_list = []
Label_list = []
i = 1

# The directory in which the augmented images will be saved
directory =r'Write the images you want to save the augmented images there'

# Reading the images one by one from the path
for img in path:
    # Saving the class of the ROI which is pre-defined in the name of image
    class_char = img[-5]
    # Reading image
    image = cv.imread(img)
    cropped_image = image
    # resizing the image into 100*100
    image_org = tf.image.resize(cropped_image,(100,100))
    # Defining the name for savin the image in directory
    name = f"IMG - {i}_{class_char}.jpg"
    i = i+1
    # Writing the image in the directory file
    cv.imwrite(os.path.join(directory,name),np.asarray(image_org) )
    
    # Augment the image with different brightness
    bright = tf.image.adjust_brightness(cropped_image, -0.2)
    bright = tf.image.resize(bright,(100,100))
    name = f"IMG - {i}_{class_char}.jpg"
    i = i+1
    cv.imwrite(os.path.join(directory,name),np.asarray(bright) )

# After creating all images and have all the images the model read them
path = glob.glob(r"Write the same path in which the augmented images were saved\*.jpg")
# Defining two lists for saving the images and labels
Image_list = []
Label_list = []
i = 1

# Reading all the images one by one
for img in path:
    class_char = img[-5]
    image = cv.imread(img)
    # Saving the resized image in the defined lists
    Image_list.append(image)
    # Add the label of each image to the label list
    Label_list.append(int(class_char))
    
# Create an array of list for images and labels lists
Image_list = np.array(Image_list)
Label_list = np.array(Label_list)

# Split the data to train validatin and test sets
x_train , x_test , y_train , y_test = train_test_split(Image_list, Label_list , test_size= 0.2 , random_state=42, shuffle=True)
x_train , x_val , y_train , y_val = train_test_split(x_train , y_train , test_size=0.2 , random_state=42 , shuffle=True)

# Normalizing the images
images_train = x_train/255.0
images_val = x_val/255.0
images_test = x_test/255.0
labels_train = y_train
labels_val = y_val
labels_test = y_test

# Changing all the classes to categorical classes
labels_train = keras.utils.to_categorical(labels_train,6)
labels_val = keras.utils.to_categorical(labels_val,6)
labels_test = keras.utils.to_categorical(labels_test,6)


# Creating the CNN model using keras
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6 , activation = 'softmax'))

# Compiling the model with adam optimizer and the accuracy metric and CategoricalCrossentropy
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=['acc'])

# Defining early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',    
    min_delta = 0,
    patience = 2,
    verbose = 0, 
    restore_best_weights = True
)

# Training the model using batch size of 32 and 20 epochs using early stopping callback
history = model.fit(images_train, labels_train, epochs=20, batch_size = 32,
                    validation_data=(images_val, labels_val),callbacks = [early_stopping])

# Evaluating the performance of the model using the validation set
test_loss, accuracy = model.evaluate(images_val,labels_val)

# Evaluating the performance of the model using the test set
test_loss, accuracy = model.evaluate(images_test,labels_test)

