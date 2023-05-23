# Speed-limit-Sign-Detection

This is a Python script for a Convolutional Neural Network (CNN) model that uses the TensorFlow and Keras libraries for image classification. The purpose of this script is to train a model to classify Regions of Interest (ROIs) in images into six pre-defined classes.

Here's a brief overview of what the code does:

* Import necessary libraries such as numpy, pandas, tensorflow, matplotlib, cv2, glob, os, sklearn, etc.

* Use the glob module to get the file paths of all images in a directory that contain the Images.

* Define two lists for storing the images and labels. Then, read each image, resize it to 100x100 pixels, and save it in the defined directory. In addition, the script uses the tf.image.adjust_brightness function to create a brighter version of the image, resizes it, and saves it with a new name.

* After creating all the augmented images, the script reads them using glob, and saves the resized image in the defined lists.

* The script splits the data into training, validation, and test sets, where 80% of the data is for training, 20% for validation, and 20% for testing. It also normalizes the image data and converts the class labels to categorical classes.

* The script creates a CNN model using Keras, consisting of three convolutional layers followed by three max-pooling layers, one flatten layer, and two dense layers, and uses the 'softmax' activation function for the output layer.

* The model is compiled with the CategoricalCrossentropy loss function, the Adam optimizer, and the accuracy metric.

* The script defines early stopping to prevent overfitting during training.

* The model is trained using the fit function, with a batch size of 32 and 20 epochs, using early stopping callback.

* Finally, the script evaluates the performance of the model using the validation and test sets.
