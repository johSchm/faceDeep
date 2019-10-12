#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:
todos:
------------------------------------------- """


import os
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import pprint

from tensorflow.python import keras
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model
from PIL import Image
from sklearn.metrics import classification_report
from tabulate import tabulate
from tqdm import tqdm
from enum import Enum
from learner import ImageClassifier


class VGGVersion(Enum):
    """ Supported VGG verions.
    """
    VGG_16 = '16'
    VGG_19 = '19'


class VGGAdapter(ImageClassifier):
    """ A VGG Adapter.
    """

    def __init__(self, version=VGGVersion.VGG_19, input_shape=None, output_shape=None):
        """ Initialization method.
        :param version
        :param input_shape
        :param output_shape
        """
        self.version = version
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.get_base_model()
        self.expand_model()
        self.config_model()

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

    def get_base_model(self):
        """ Returns the corresponding Keras VGG model.
        :return: model
        """
        if self.version == VGGVersion.VGG_16:
            return VGG16(input_shape=self.input_shape, include_top=True, weights='imagenet')
        if self.version == VGGVersion.VGG_19:
            return VGG19(input_shape=self.input_shape, include_top=True, weights='imagenet')
        else:
            raise ValueError("VGG Version not found!")

    def expand_model(self):
        """ It is important to freeze the convolutional based
        before you compile and train the model.
        By freezing or setting layer.trainable = False,
        you prevent the weights in a given layer from
        being updated during training.
        """
        self.model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(len(self.output_shape), activation='softmax')
        self.model = tf.keras.Sequential([
            self.model,
            global_average_layer,
            prediction_layer])

    def config_model(self):
        """ Compiles the model.
        """
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=["accuracy"])

    @staticmethod
    def get_imagenet_class(index):
        """ imagenet_class_index.json:
        https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
        :param index
        :return: classes
        """
        classes = json.load(open("imagenet_class_index.json"))
        return classes[str(index)][1]

    def predict_image(self, img):
        """ Predicts the class of an image.
        :param img:
        :return: prediction
        """
        img = np.expand_dims(img, axis=0)
        image_net_index = np.argmax(self.model.predict(img))
        return self.get_imagenet_class(image_net_index)

    def predict_top_image(self, img, top_value=5):
        """ Performs a top n prediction.
        :param img:
        :param top_value:
        :return: prediction
        """
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img)
        class_indexes = np.argpartition(predictions[0], -top_value)[-top_value:]
        pred = np.array(predictions[0][class_indexes])
        ind = pred.argsort()
        return [class_indexes[ind][::-1], pred[ind][::-1] * 100]

