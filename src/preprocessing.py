#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
refs:       DB: https://github.com/tkarras/progressive_growing_of_gans
todos:      @TODO: Change data structure
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


DIR_TEST = "../res/test.pickle"
DIR_TRAIN = "../res/train.pickle"


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
    GRAYSCALE = "gray"
    RGB = "RGB"


class DataPattern(Enum):
    """ Supported return patterns for the data.
    XY_XY:      Train(data + label) and Test(data + label)
    X_Y_X_Y:    Train(data), Train(label), Test(data), and Test(label)
    """
    XY_XY = 0
    X_Y_X_Y = 1


class Preprocessor:
    """ A data preprocessor for preparing the data for the learner.
    """

    def __init__(self, datadir, categories, img_size, colormode=Colormode.GRAYSCALE, data_pattern=DataPattern.X_Y_X_Y):
        """ Initialization method.
        :param datadir:
        :param categories:
        :param img_size:
        :param data_pattern
        """
        if os.path.isdir(datadir):
            self.datadir = datadir
        else:
            raise NotADirectoryError("Directory not found {}".format(datadir))
        self.categories = self.category_mapping(categories)
        self.datapattern = data_pattern
        self.img_size = img_size
        self.colormode = colormode
        self.raw_data = {}
        self.train_data = []
        self.test_data = []

    def show_sample(self, img):
        """ Displays a sample image.
        :param img:
        """
        if img is None:
            raise TypeError("Image is None!")
        plt.imshow(img, cmap=self.colormode)
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

    def save(self):
        """ Saves the gathered data locally.
        """
        pickle_out = open(DIR_TRAIN, "wb")
        pickle.dump(self.train_data, pickle_out)
        pickle_out.close()

        pickle_out = open(DIR_TEST, "wb")
        pickle.dump(self.test_data, pickle_out)
        pickle_out.close()

    @staticmethod
    def load():
        """ Loads stored data.
        :return: train, test
        """
        pickle_in = open(DIR_TRAIN, "rb")
        train = pickle.load(pickle_in)
        pickle_in = open(DIR_TEST, "rb")
        test = pickle.load(pickle_in)
        return train, test

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

    def split_train_test(self, data, test_size=0.2, random_state=42):
        """ Splits the data into training and test set.
        :param data
        :param test_size
        :param random_state
        :return: training set, test set
        """
        if len(data) != 2:
            labels, data = self.sample_label_disjoin(data)
        else:
            labels, data = data[1], data[0]
        x_train, x_test, y_train, y_test = sms.train_test_split(
            data, labels, test_size=test_size, random_state=random_state)
        if self.datapattern.value == DataPattern.XY_XY.value:
            train_data = self.sample_label_join(x_train, y_train)
            test_data = self.sample_label_join(x_test, y_test)
            return train_data, test_data
        if self.datapattern.value == DataPattern.X_Y_X_Y.value:
            return x_train, x_test, y_train, y_test
        else:
            raise ValueError("Return Pattern is not valid!")

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
