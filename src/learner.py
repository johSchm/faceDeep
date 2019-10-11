#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:       CALLBACKS: https://keras.io/callbacks/
todos:
------------------------------------------- """

import preprocessing as pp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import BaseLogger
import os


VERSION = "001_004"
RES_PATH = "../res"
LOG_PATH = "../logs"
TENSOR_BOARD_NAME = "Model_" + VERSION
TENSOR_BOARD_LOG_DIR = "../logs/{}"


class ImageClassifier:
    """ A classifier for images. (CNN-based)
    """

    def __init__(self, input_shape, model_path=None, layer_size=128,
                 num_conv_layers=3, num_dense_layers=0, activation='relu',
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
        log = open(os.path.join(LOG_PATH, "model_log_" + VERSION + ".log"), 'w+')
        log.write("...")
        log.close()

    def construct_model(self):
        """ Adds layers to the model.
        """
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1), input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.2))

        self.model.add(Flatten())
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(rate=0.1))
        self.add_dense_layer(units=1, activation='sigmoid')
        """self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1), input_shape=self.input_shape))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))

        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))

        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))

        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',
                              padding='same', strides=(1, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))

        #self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        #self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        #self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        #self.model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Dropout(rate=0.25))

        self.model.add(Flatten())
        self.model.add(Dense(units=4096, activation='softmax'))
        self.model.add(Dense(units=1000, activation='softmax'))
        self.model.add(Dropout(rate=0.1))
        self.add_dense_layer(units=1, activation='sigmoid')"""

        """self.add_cov2d_layer(
            size=self.layer_size, kernel_size=(3, 3),
            activation=self.activation, input_shape=self.input_shape)
        self.add_pooling_layer(pool_size=(2, 2))
        self.add_dropout(0.25)

        for _ in range(self.num_conv_layers - 1):
            self.add_cov2d_layer(
                size=self.layer_size, kernel_size=(3, 3),
                activation=self.activation, input_shape=self.input_shape)
            self.add_pooling_layer(pool_size=(2, 2))
            self.add_dropout(0.25)

        self.add_flatten_layer()

        for _ in range(self.num_dense_layers - 1):
            self.add_dense_layer(units=128, activation=self.activation)

        self.add_dropout(0.1)
        self.add_dense_layer(units=50, activation=self.activation)
        self.add_dense_layer(units=1, activation='softmax')"""

    @staticmethod
    def normalize(data, basis=255.0):
        """ Normalize each image pixel color value from [0,256] to [0,1].
        :param data
        :param basis
        """
        if data is None:
            return None
        return data / basis

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

    def add_dropout(self, rate=0.1):
        """ Applies dropout to the network.
        :param rate:
        """
        if self.model is not None:
            self.model.add(Dropout(rate))

    def add_dense_layer(self, units=128, activation='relu', input_shape=None):
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

    def add_cov2d_layer(self, size=256, kernel_size=(3, 3), activation='relu', input_shape=None):
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
                    input_shape=input_shape, data_format='channels_last'))

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
            x_train = self.normalize(x_train)
            if x_val is None and y_val is None:
                self.model.fit(x_train, y_train,
                               epochs=epochs, batch_size=batch_size,
                               validation_split=validation_split,
                               callbacks=self.callbacks)
            else:
                self.model.fit(x_train, y_train,
                               epochs=epochs, batch_size=batch_size,
                               validation_data=(x_val, y_val),
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
        img = tf.keras.utils.normalize(img, axis=1)
        prediction = self.model.predict(np.array([img, ]))
        return prediction[0][0]

    def multi_prediction(self, preprocessor, categories, num_img_per_category):
        """ Predicts multiple images and outputs the results in the console.
        :param preprocessor:
        :param categories:
        :param num_img_per_category:
        """
        if type(preprocessor) is not pp.Preprocessor:
            raise TypeError("Preprocessor type have to be of type {}".format(pp.Preprocessor))
        if num_img_per_category <= 0:
            raise ValueError("Invalid number of images per category: {}".format(num_img_per_category))
        if type(categories) is not list:
            raise TypeError("Categories have to be of type {}".format(list))
        if len(categories) <= 0:
            raise ValueError("Category list length is not valid!")
        for category in categories:
            print("{} -----------------------------------------------------------".format(category))
            path = os.path.join(preprocessor.datadir, category)
            images = list(os.listdir(path))
            for img_name in images[:5]:
                img = preprocessor.load_sample(os.path.join(path, img_name))
                img = preprocessor.resize_image(img)
                img = np.expand_dims(img, axis=2)
                prediction = self.predict(img)
                prediction_mapped = preprocessor.categories[int(round(prediction))]
                print("Prediction of {0} in {1}: {2} ({3})".format(
                    img_name, category, prediction_mapped, prediction))

