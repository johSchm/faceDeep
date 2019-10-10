#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:       CALLBACKS: https://keras.io/callbacks/
todos:
------------------------------------------- """

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import pickle
import time
import os
import cv2


VERSION = "001_002"
RES_PATH = "../res"
LOG_PATH = "../logs"
TENSOR_BOARD_NAME = "Model_" + VERSION
TENSOR_BOARD_LOG_DIR = "../logs/{}"


class ImageClassifier:
    """ A classifier for images. (CNN-based)
    """

    def __init__(self, input_shape, model_path=None, layer_size=128,
                 num_conv_layers=3, num_dense_layers=0, activation=tf.nn.relu,
                 dropout_rate=0.3):
        """ Initialization method.
        :param input_shape
        :param model_path:
        :param layer_size
        :param num_conv_layers
        :param num_dense_layers
        :param activation
        :param dropout_rate
        """
        self.callbacks = self.logger_setup()
        self.input_shape = input_shape
        if model_path is not None:
            self.model = self.load(path=model_path)
        else:
            self.dropout_rate = dropout_rate
            self.layer_size = layer_size
            self.num_conv_layers = num_conv_layers
            self.num_dense_layers = num_dense_layers
            self.activation = activation
            self.model = self.build_ff_model()
            self.construct_model()
            self.configure(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    @staticmethod
    def logger_setup():
        """ Sets the callback and the overall logger.
        :return callbacks
        """
        tb = TensorBoard(log_dir=TENSOR_BOARD_LOG_DIR.format(TENSOR_BOARD_NAME))
        BaseLogger(stateful_metrics=None)
        return [tb]

    def log(self):
        """ Adds a log file with the current model configuration.
        """
        # @TODO How to save/log all model properties?
        log = open(os.path.join(LOG_PATH, "model_log_" + VERSION + ".log"), 'w+')
        log.write("...")
        log.close()

    """def add_summary(self, path=LOG_PATH):
        writer = tf.summary.create_file_writer(path)
        with writer.as_default():
            for step in range(100):
                tf.summary.scalar("my_metric", 0.5, step=step)
                writer.flush()"""

    def construct_model(self):
        """ Adds layers to the model.
        """
        self.add_cov2d_layer(
            size=self.layer_size, kernel_size=(3, 3),
            activation=self.activation, input_shape=self.input_shape)
        self.add_pooling_layer(pool_size=(2, 2))
        self.add_dropout(self.dropout_rate)

        for _ in range(self.num_conv_layers - 1):
            self.add_cov2d_layer(
                size=self.layer_size, kernel_size=(3, 3),
                activation=self.activation, input_shape=self.input_shape)
            self.add_pooling_layer(pool_size=(2, 2))
            self.add_dropout(self.dropout_rate)

        self.add_flatten_layer()

        for _ in range(self.num_dense_layers - 1):
            self.add_dense_layer(units=self.layer_size, activation=self.activation)

        self.add_dense_layer(units=1, activation=tf.nn.sigmoid)

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

    def add_dropout(self, rate=0.3):
        """ Applies dropout to the network.
        :param rate:
        """
        if self.model is not None:
            self.model.add(Dropout(rate))

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

    def add_cov2d_layer(self, size=256, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=None):
        """ Adds an 2D convolutional layer.
        :param size: the dimensionality of the output space
        :param kernel_size
        :param activation
        :param input_shape
        """
        if self.model is not None:
            if input_shape is None:
                self.model.add(Conv2D(
                    filters=size, kernel_size=kernel_size, activation=activation))
            else:
                self.model.add(Conv2D(
                    filters=size, kernel_size=kernel_size, activation=activation,
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

    def train(self, x_train, y_train, x_val=None, y_val=None,
              epochs=3, batch_size=32, validation_split=0.3):
        """ Start training phase.
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :param epochs: number of epochs
        :param batch_size
        :param validation_split
        """
        if self.model is not None:
            val_data = (x_val, y_val)
            if x_val is None and y_val is None:
                val_data = None
            x_train = self.normalize(x_train)
            self.model.fit(x_train, y_train,
                           epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split,
                           validation_data=val_data,
                           callbacks=self.callbacks)

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

    def save(self):
        """ Saves the model.
        """
        if self.model is not None:
            self.model.save(os.path.join(RES_PATH, "models", "model_" + VERSION + ".model"))

    @staticmethod
    def load(path='num_reader.model'):
        """ Loads a model.
        :param path: the name of the model
        :return model
        """
        return tf.keras.models.load_model(path)

    def predict(self, img):
        """ Predicts the content of an image.
        :param img:
        :return: predicted label
        """
        if self.model is None:
            raise TypeError("No model found!")
        if not isinstance(img, np.ndarray):
            raise TypeError("Unable to predict type {}".format(type(img)))
        if img.shape != self.input_shape:
            raise TypeError("Wrong image shape! Got {0}, expected {1}".format(img.shape, self.input_shape))
        img = tf.keras.utils.normalize(img, axis=1)
        prediction = self.model.predict(np.array([img, ]))
        return np.argmax(prediction[0])

