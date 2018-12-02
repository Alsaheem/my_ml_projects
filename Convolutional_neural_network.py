# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:31:19 2018

@author: ayo
"""

#Step 1----- building the CNN
#importing the libraries

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(150,150,3), activation="relu"))
#64,64

#max pooling
#reducing the models complexity without reducing its performance
classifier.add(MaxPooling2D(pool_size=(2,2)))

#another second convolutional layer
classifier.add(Convolution2D(32,3,3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#another third convolutional layer
classifier.add(Convolution2D(32,3,3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

#FULLL CONNECTION
#fully connected layer
classifier.add(Dense(128,activation="relu"))
#another fully connected layer
classifier.add(Dense(128,activation="relu"))

#output layer
classifier.add(Dense(1,activation="sigmoid"))

#compiling the CNN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#if more than two outcomes , well choose categorical_crossentropy

#part 2
#fitting the CNN to our images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(150,150),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(150,150),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)


