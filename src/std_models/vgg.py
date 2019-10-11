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


VERSIONS = ['16', '19']


class VGGAdapter:
    """ A VGG Adapter.
    """

    def __init__(self, version='19'):
        """ Initialization method.
        """
        self.model = self.get_model(version)

    @staticmethod
    def get_model(version):
        """ Returns the corresponding Keras VGG model.
        :param version:
        :return: model
        """
        if type(version) is not str:
            raise TypeError("VGG Version has to be a {}".format(str))
        if version == '16':
            return VGG16(include_top=True, weights='imagenet')
        if version == '19':
            return VGG19(include_top=True, weights='imagenet')
        else:
            raise ValueError("VGG Version not found!")

    def get_imagenet_class(self, index):
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

    @staticmethod
    def show_image(img, title):
        """ Displays a image with title.
        :param img:
        :param title:
        """
        plt.title("Erkannt : {}\n".format(title))
        plt.axis('off')
        plt.imshow(img, interpolation='none')
        plt.show()

    def preprocessing(self, path):
        if not os.path.isdir(path):
            raise NotADirectoryError("Directory not found: {}".format(path))
        for
        jpgfile = np.array(Image.open(path).convert('RGB').resize((224, 224)))

        print("Top 1 - Prediction: {}".format(predict_image(current_model, jpgfile)))
        predictions_top_image = predict_top_image(current_model, jpgfile, top_value=TOP_VALUE)
        headers = ['Class name', 'index', 'prediction']
        table = []
        # Ausgabe der n ersten erkannten Klassen
        for i in range(0, TOP_VALUE):
            class_index = predictions_top_image[0][i]
            table.append([str(get_imagenet_class(class_index)), class_index, predictions_top_image[1][i]])
        print(tabulate(table, headers=headers, tablefmt='orgtbl'))