#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:17:22 2022

@author: troyobernolte

CNN For Cats and Dogs
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image
import numpy as np


train_runs = 30

#Training Set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#Test Set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               input_shape=[64, 64, 3]))

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

def train():
    cnn.fit(x = training_set, validation_data = test_set, epochs = train_runs)

def test():
    test_image_cat = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
    test_image_cat = image.img_to_array(test_image_cat)
    test_image_cat = np.expand_dims(test_image_cat, axis = 0)
    result = cnn.predict(test_image_cat)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print("Actaul value: cat Predicted Value:", prediction)
    
    test_image_dog = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
    test_image_dog = image.img_to_array(test_image_dog)
    test_image_dog = np.expand_dims(test_image_dog, axis = 0)
    result = cnn.predict(test_image_dog)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print("Actaul value: dog Predicted Value:", prediction)

