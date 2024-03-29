#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:       DB: https://github.com/tkarras/progressive_growing_of_gans
todos:
------------------------------------------- """

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import sklearn.model_selection as sms
import pandas as pd
import itertools
import pickle
from enum import Enum
from PIL import Image


TRAIN_DATA_FILE = "train.pickle"
TEST_DATA_FILE = "test.pickle"
TRAIN_X_DATA_FILE = "train_x.pickle"
TRAIN_Y_DATA_FILE = "train_y.pickle"
TEST_X_DATA_FILE = "test_x.pickle"
TEST_Y_DATA_FILE = "test_y.pickle"


def load_file(path):
    """ Loads a file.
    :param path:
    :return: file
    """
    if type(path) is not str:
        raise TypeError("Path has to be a string!")
    return cv2.imread(path)


def shared_items(dict_1, dict_2):
    """ Returns the shared items of two dictionaries.
    :param dict_1:
    :param dict_2:
    :return: shared items
    """
    if type(dict_1) is not dict or type(dict_2) is not dict:
        return None
    return {k: dict_1[k] for k in dict_1 if k in dict_2 and dict_1[k] == dict_2[k]}


def equal_values(dictionary):
    """ Checks if all values in a dictionary are equal.
    :param dictionary:
    :return: boolean
    """
    if type(dictionary) is not dict:
        return False
    i = 0
    master_value = 0
    for key, value in dictionary.items():
        if i == 0:
            master_value = value
        if value != master_value:
            return False
        i += 1
    return True


def value_redundancy(dictionary):
    """ Returns if a value is already in the dictionary.
    :param dictionary:
    :return: True or False
    """
    if type(dictionary) is not dict:
        return False
    for key_1, value_1 in dictionary.items():
        for key_2, value_2 in dictionary.items():
            if value_1 == value_2 and key_1 != key_2:
                return True
    return False


def extract_vector_from_matrix(matrix, idx):
    """ Extracts a vector from a matrix (2D List).
    :param matrix:
    :param idx of vector
    :return: vector
    """
    if type(matrix) is not list or type(idx) is not int:
        return None
    vector = []
    for v in matrix:
        vector.append(v[idx])
    return vector


class Colormode(Enum):
    """ Supported color modes.
    """
    GRAYSCALE = "L"
    RGB = "RGB"
    CMYK = "CMYK"


class DataPattern(Enum):
    """ Supported return patterns for the data.
    XY_XY:      Train(data + label) and Test(data + label)
    X_X_Y_Y:    Train(data), Test(data), Train(label), and Test(label)
    """
    XY_XY = 0
    X_X_Y_Y = 1


class Preprocessor:
    """ A data preprocessor for preparing the data for the learner.
    """

    def __init__(self, datadir, categories, img_size, colormode=Colormode.GRAYSCALE, data_pattern=DataPattern.X_X_Y_Y):
        """ Initialization method.
        :param datadir:
        :param categories:
        :param img_size:
        :param data_pattern
        """
        self.datadir = datadir
        self.categories = self.category_mapping(categories)
        self.datapattern = data_pattern
        self.img_size = img_size
        self.colormode = colormode
        self.raw_data = {}

    def show(self, img, label=None):
        """ Displays a sample image.
        :param img:
        :param label
        """
        if img is None:
            raise TypeError("Image is None!")
        if self.colormode == Colormode.GRAYSCALE:
            plt.imshow(img, cmap='gray')
        if self.colormode == Colormode.RGB:
            plt.imshow(img)
        if type(label) is str:
            plt.title("Label: {}".format(label))
        plt.show()

    @staticmethod
    def category_mapping(category_list):
        """ Maps the category list to a dictionary with int flags.
        :param category_list:
        :return: dictionary
        """
        if type(category_list) is not list or len(category_list) <= 0:
            return None
        category_dict = {}
        for i, category in enumerate(category_list):
            category_dict[i] = category
        return category_dict

    def run(self, partial_load=1.0):
        """ Starts the pre-processing routine.
        :param partial_load
        """
        self.load_data_links(partial_load=partial_load)
        if not self.balanced(self.class_distribution()):
            self.random_under_sampling()
        data = self.load_data()
        shuffled_list = self.shuffle(data)
        return self.split_train_test(shuffled_list)

    def save(self, path, *data):
        """ Saves the gathered data locally.
        :param path
        """
        if type(data) is not tuple:
            raise TypeError("Data structure has to be a tuple, got {} instead!".format(type(data)))
        if not os.path.isdir(path):
            print("Directory: {} not found. Creating directory ...".format(path))
            os.makedirs(path)
        if self.datapattern.value == DataPattern.XY_XY.value:
            if len(data[0]) != 2:
                raise ValueError("Passed data does not match the data pattern!")
            pickle_out = open(os.path.join(path, TRAIN_DATA_FILE), "wb")
            pickle.dump(data[0][0], pickle_out)
            pickle_out.close()
            pickle_out = open(os.path.join(path, TRAIN_DATA_FILE), "wb")
            pickle.dump(data[0][1], pickle_out)
            pickle_out.close()
        if self.datapattern.value == DataPattern.X_X_Y_Y.value:
            if len(data[0]) != 4:
                raise ValueError("Passed data does not match the data pattern!")
            pickle_out = open(os.path.join(path, TRAIN_X_DATA_FILE), "wb")
            pickle.dump(data[0][0], pickle_out)
            pickle_out.close()
            pickle_out = open(os.path.join(path, TEST_X_DATA_FILE), "wb")
            pickle.dump(data[0][1], pickle_out)
            pickle_out.close()
            pickle_out = open(os.path.join(path, TRAIN_Y_DATA_FILE), "wb")
            pickle.dump(data[0][2], pickle_out)
            pickle_out.close()
            pickle_out = open(os.path.join(path, TEST_Y_DATA_FILE), "wb")
            pickle.dump(data[0][3], pickle_out)
            pickle_out.close()

    def load(self, path):
        """ Loads stored data.
        :param path
        :return: train, test
        """
        if not os.path.isdir(path):
            raise NotADirectoryError("Data directory: {} not found!".format(path))
        if self.datapattern.value == DataPattern.XY_XY.value:
            pickle_in = open(os.path.join(path, TRAIN_DATA_FILE), "rb")
            train = pickle.load(pickle_in)
            pickle_in = open(os.path.join(path, TRAIN_DATA_FILE), "rb")
            test = pickle.load(pickle_in)
            return train, test
        if self.datapattern.value == DataPattern.X_X_Y_Y.value:
            pickle_in = open(os.path.join(path, TRAIN_X_DATA_FILE), "rb")
            train_x = pickle.load(pickle_in)
            pickle_in = open(os.path.join(path, TRAIN_Y_DATA_FILE), "rb")
            train_y = pickle.load(pickle_in)
            pickle_in = open(os.path.join(path, TEST_X_DATA_FILE), "rb")
            test_x = pickle.load(pickle_in)
            pickle_in = open(os.path.join(path, TEST_Y_DATA_FILE), "rb")
            test_y = pickle.load(pickle_in)
            return train_x, test_x, train_y, test_y

    def class_distribution(self):
        """ Returns a dictionary with the number of samples per class.
        :return: dictionary
        """
        if self.datadir is None or self.categories is None:
            return None
        class_distribution = {}
        for category, _set in self.raw_data.items():
            class_distribution[category] = len(_set)
        return class_distribution

    def load_data_links(self, partial_load=1.0):
        """ Loads the data from source as a list of links.
        :param partial_load load only a certain amount of the data.
        :return: data
        """
        for idx, category in self.categories.items():
            category_set = []
            path = os.path.join(self.datadir, category)
            dirs = os.listdir(path)
            dirs = dirs[0:int(len(dirs) * partial_load)]
            for img in tqdm(dirs):
                category_set.append(img)
            self.raw_data[category] = category_set
        return self.raw_data

    def load_data(self):
        """ Loads the data from source.
        Additional filter methods can be applied.
        :return: data
        """
        data = []
        for idx, category in self.categories.items():
            for img in tqdm(self.raw_data[category]):
                path = os.path.join(self.datadir, category, img)
                img_array = self.load_sample(path)
                data.append(np.array([idx, img_array]))
        return np.array(data)

    def load_sample(self, path):
        """ Loads an sample image from the given path.
        :param path
        :return: sample
        """
        if not os.path.isfile(path):
            raise NotADirectoryError("Sample not found at {}".format(path))
        return np.array(Image.open(path).convert(self.colormode.value).resize((self.img_size, self.img_size)))

    @staticmethod
    def balanced(class_distribution):
        """ Checks whether the sets a balanced or not.
        :param class_distribution:
        :return: True or False
        """
        if type(class_distribution) is not dict:
            return False
        return equal_values(class_distribution)

    @staticmethod
    def largest_set(sets):
        """ Returns the largest sets index.
        :param sets
        :return index
        """
        if sets is None or type(sets) is not list or len(sets) <= 0:
            return None
        if len(sets) == 1:
            return 0
        m, max_len = 0, 0
        for i, _set in enumerate(sets):
            if len(_set) > max_len:
                m, max_len = i, len(_set)
        return m

    @staticmethod
    def smallest_set(dictionary):
        """ Returns the smallest sets index.
        :param dictionary
        :return index
        """
        if type(dictionary) is not dict or len(dictionary) <= 0:
            return None
        key, min_len = 0, 0
        for key, _set in dictionary.items():
            if len(_set) < min_len:
                m, min_len = key, len(_set)
        return key

    def random_under_sampling(self):
        """ Randomly selects samples from the larger set, such that the sizes matches.
        :return: new sets
        """
        if self.raw_data is None:
            return None
        target_set_key = self.smallest_set(self.raw_data)
        target_set_len = len(self.raw_data[target_set_key])
        for key, category_set in self.raw_data.items():
            if key != target_set_key:
                self.raw_data[key] = self.downsize_set(category_set, target_set_len)
        return self.raw_data

    @staticmethod
    def downsize_set(_set, target_len):
        """ Downsize a set to a specific target length.
        :param _set:
        :param target_len:
        :return: reduced set
        """
        if _set is None or len(_set) <= 0 or type(target_len) is not int:
            return None
        if target_len <= 0:
            return []
        while len(_set) != target_len:
            idx = random.randint(0, len(_set) - 1)
            del _set[idx]
        return _set

    @staticmethod
    def sample_label_join(sample_array, label_array):
        """ Constructs a column stack.
        :param sample_array:
        :param label_array:
        :return: stack
        """
        if len(sample_array) != len(label_array):
            raise ValueError("Passed array length is not valid!")
        joined_list = []
        for i, label in enumerate(label_array):
            joined_list.append([label, sample_array[i]])
        return np.array(joined_list)

    @staticmethod
    def sample_label_disjoin(array):
        """ Splits the 2D list into two seperate lists.
        :param array
        :return: array, array
        """
        if len(array) <= 0:
            raise ValueError("Array length is not valid!")
        list_1 = []
        list_2 = []
        for sublist in array:
            list_1.append(sublist[0])
            list_2.append(sublist[1])
        return np.array(list_1), np.array(list_2)

    def split_train_test(self, data, test_size=0.2, random_state=42, label_optimization=True):
        """ Splits the data into training and test set.
        :param data
        :param test_size
        :param random_state
        :param label_optimization
        :return: training set, test set
        """
        if len(data) != 2:
            labels, data = self.sample_label_disjoin(data)
        else:
            labels, data = data[1], data[0]
        x_train, x_test, y_train, y_test = sms.train_test_split(
            data, labels, test_size=test_size, random_state=random_state)
        if label_optimization:
            y_train, y_test = self.label_optimization(y_train, y_test)
        if self.colormode == Colormode.GRAYSCALE:
            x_train = np.expand_dims(x_train, axis=3)
            x_test = np.expand_dims(x_test, axis=3)
        if self.datapattern.value == DataPattern.XY_XY.value:
            train_data = self.sample_label_join(x_train, y_train)
            test_data = self.sample_label_join(x_test, y_test)
            return train_data, test_data
        if self.datapattern.value == DataPattern.X_X_Y_Y.value:
            return x_train, x_test, y_train, y_test
        else:
            raise ValueError("Return Pattern is not valid!")

    @staticmethod
    def label_optimization(*labels):
        """ Optimizes the label list for further processing.
        :param labels:
        :return: optimized label arrays
        """
        if len(labels) <= 0:
            raise ValueError("Label length is not valid!")
        if len(labels) == 1:
            return np.expand_dims(np.array(labels, dtype=np.uint8), 1)
        if len(labels) == 2:
            return np.expand_dims(np.array(labels[0], dtype=np.uint8), 1), \
                   np.expand_dims(np.array(labels[1], dtype=np.uint8), 1)

    def resize_image(self, image):
        """ Normalizes an image array based on the predefined image size value.
        :param image:
        :return: resized array
        """
        if image is None:
            raise TypeError("Image(s) is/are None!")
        if len(image) <= 0:
            raise ValueError("Image list is empty!")
        return cv2.resize(image, (self.img_size, self.img_size))

    @staticmethod
    def shuffle(data):
        """ Shuffles a data list randomly.
        :param data
        :return: shuffled list
        """
        if len(data) <= 0:
            raise ValueError("Data list is empty!")
        random.shuffle(data)
        return data
