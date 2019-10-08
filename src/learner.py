#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle


class ImageClassifier:
    """ A classifier for images.
    """

    def __init__(self, input_shape, load_existing_model=False):
        """ Initialization method.
        :param load_existing_model:
        """
        if load_existing_model:
            self.model = self.load()
        else:
            self.model = self.build_ff_model()
            self.construct_model(input_shape)
            self.configure(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    def construct_model(self, input_shape):
        """ Adds layers to the model.
        :param input_shape
        """
        if self.model is not None:

            # Conv layer and max pooling
            self.model.add(Conv2D(filters=256, kernel_size=(3, 3),
                                  data_format="channels_last", input_shape=input_shape))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # Conv layer and max pooling
            self.model.add(Conv2D(filters=256, kernel_size=(3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # this converts our 3D feature maps to 1D feature vectors
            self.model.add(Flatten())

            # fully connected layer
            self.model.add(Dense(units=64))

            # output layer
            self.model.add(Dense(units=1))
            self.model.add(Activation('sigmoid'))

    @staticmethod
    def normalize(data):
        """ Normalize each image pixel color value from [0,256] to [0,1].
        :param data
        """
        if data is None:
            return None
        return tf.keras.utils.normalize(data, axis=1)

    @staticmethod
    def build_ff_model():
        """ Builds a feed forward model.
        :return: model
        """
        return Sequential()

    def add_flatten_layer(self):
        """ Adds a flatten input layer.
        (28x28 image -> 1x784 vector)
        """
        if self.model is not None:
            self.model.add(tf.keras.layers.Flatten())

    def add_dense_layer(self, units=128, activation=tf.nn.relu, input_shape=None):
        """ Adds a hidden layer (dense = fully connected).
        :param units number of units
        :param activation activation function
        :param input_shape shape of the input
        """
        if self.model is not None:
            if input_shape is None:
                self.model.add(tf.keras.layers.Dense(
                    units=units, activation=activation,
                    kernel_constraint=None, bias_constraint=None))
            else:
                self.model.add(tf.keras.layers.Dense(
                    units=units, activation=activation, input_shape=input_shape,
                    kernel_constraint=None, bias_constraint=None))

    def add_cov2d_layer(self, filters=256, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=None):
        """ Adds an 2D convolutional layer.
        :param filters: the dimensionality of the output space
        :param kernel_size
        :param activation
        :param input_shape
        """
        if self.model is not None:
            if input_shape is None:
                self.model.add(Conv2D(
                    filters=filters, kernel_size=kernel_size, activation=activation))
            else:
                self.model.add(Conv2D(
                    filters=filters, kernel_size=kernel_size, activation=activation,
                    input_shape=input_shape))

    def add_pooling_layer(self, pool_size=(2, 2)):
        """ Adds a pooling layer.
        :param pool_size:
        """
        if self.model is not None:
            self.model.add(MaxPooling2D(pool_size=pool_size))

    def configure(self, optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']):
        """
        Configures the model for training.
        :param optimizer:
        :param loss:
        :param metrics:
        """
        if self.model is not None:
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    def train(self, x_train, y_train,
              epochs=3, batch_size=32, validation_split=0.3):
        """ Start training phase.
        :param x_train:
        :param y_train:
        :param epochs: number of epochs
        :param batch_size
        :param validation_split
        """
        if self.model is not None:
            x_train = self.normalize(x_train)
            self.model.fit(x_train, y_train,
                           epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split)

    def evaluate(self, x_test, y_test, output=True):
        """ Evaluates the model.
        :param x_test
        :param y_test
        :param output: Output the result in the console.
        :return: results
        """
        if self.model is None:
            return None
        val_loss, val_acc = self.model.evaluate(x_test, y_test)
        if output:
            print("Evaluation loss: {}".format(val_loss))
            print("Evaluation accuracy: {}".format(val_acc))
        return val_loss, val_acc

    def save(self, name='num_reader.model'):
        """ Saves the model.
        :param name: name of the model
        """
        if self.model is not None:
            self.model.save(name)

    def load(self, filename='num_reader.model'):
        """ Loads a model.
        :param filename: the name of the model
        :return model
        """
        return tf.keras.models.load_model(filename)

    def shape(self):
        """ Returns the model input data shape.
        :return: shape
        """
        if self.model is None:
            return None
        return self.model._build_input_shape[1], self.model._build_input_shape[2]

    def predict(self, img):
        """ Predicts the content of an image.
        :param img:
        :return: predicted label
        """
        if self.model is None:
            return None
        if not isinstance(img, np.ndarray):
            print("ERROR: Unable to predict type {}".format(type(img)))
            return None
        if img.shape != self.shape():
            print("ERROR: Wrong image shape {}".format(img.shape))
        img = tf.keras.utils.normalize(img, axis=1)
        prediction = self.model.predict(np.array([img, ]))
        return np.argmax(prediction[0])
