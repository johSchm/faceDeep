#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
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


"""
https://github.com/tkarras/progressive_growing_of_gans
"""

DIR_TEST = "test.pickle"
DIR_TRAIN = "train.pickle"
DATADIR = "/mnt/HDD/Masterthesis/DB"
CATEGORIES = ["Human", "NoHuman"]
IMG_SIZE = 100


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


class Preprocessor:
    """ A data preprocessor for preparing the data for the learner.
    """

    def __init__(self, datadir=DATADIR, categories=CATEGORIES, img_size=IMG_SIZE):
        """ Initialization method.
        :param datadir:
        :param categories:
        :param img_size:
        """
        if self.dir_valid(datadir):
            self.datadir = datadir
        else:
            print("ERROR: Data directory does not exist!")
        self.categories = self.category_mapping(categories)
        self.img_size = img_size
        self.raw_data = {}
        self.train_data = []
        self.test_data = []

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

    @staticmethod
    def dir_valid(_dir):
        """ Checks if a given directory is valid (accessible).
        :param _dir:
        :return: boolean
        """
        if type(_dir) is not str:
            return False
        return os.path.isdir(_dir)

    def run(self):
        """ Starts the pre-processing routine.
        """
        self.load_data_links(partial_load=0.01)
        if not self.balanced(self.class_distribution()):
            self.random_under_sampling()
        data = self.load_data()
        shuffled_list = self.shuffle(data)
        self.train_data, self.test_data = self.split_train_test(shuffled_list)
        return self.train_data, self.test_data

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

    def load_data(self, resize=True, grayscale=True):
        """ Loads the data from source.
        Additional filter methods can be applied.
        :param resize
        :param grayscale
        :return: data
        """
        param = None
        if grayscale:
            param = cv2.IMREAD_GRAYSCALE
        data = []
        for idx, category in self.categories.items():
            for img in tqdm(self.raw_data[category]):
                try:
                    img_array = self.load_sample(img, category, param)
                    if resize:
                        img_array = self.resize_images(img_array)
                    data.append([idx, img_array])
                except Exception as e:
                    print("Exception occurred! Continuing ...")
                    pass
        return data

    def load_sample(self, name, category, param=None):
        """ Loads an sample image from the given path.
        :param name
        :param category
        :param param for image loading
        :return: sample
        """
        if type(name) is not str or type(category) is not str:
            return None
        path = os.path.join(self.datadir, category, name)
        if param is not None:
            return cv2.imread(path, param)
        return cv2.imread(path)

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
    def sample_label_join(sample_list, label_list):
        """ Constructs a column stack.
        :param sample_list:
        :param label_list:
        :return: stack
        """
        if sample_list is None or label_list is None \
                or len(sample_list) != len(label_list):
            return None
        joined_list = []
        for i, label in enumerate(label_list):
            joined_list.append([label, sample_list[i]])
        return joined_list

    @staticmethod
    def sample_label_disjoin(_list):
        """ Splits the 2D list into two seperate lists.
        :param _list
        :return: list_1, list_2
        """
        if type(_list) is not list or len(_list) <= 0:
            return None
        list_1 = []
        list_2 = []
        for sublist in _list:
            list_1.append(sublist[0])
            list_2.append(sublist[1])
        return list_1, list_2

    def split_train_test(self, data, test_size=0.2, random_state=42):
        """ Splits the data into training and test set.
        :param data
        :param test_size
        :param random_state
        :return: training set, test set
        """
        if type(data) is not list:
            return None
        if len(data) != 2:
            labels, data = self.sample_label_disjoin(data)
        else:
            labels, data = data[1], data[0]
        x_train, x_test, y_train, y_test = sms.train_test_split(
            data, labels, test_size=test_size, random_state=random_state)
        train_data = self.sample_label_join(x_train, y_train)
        test_data = self.sample_label_join(x_test, y_test)
        return train_data, test_data

    def resize_images(self, img_array):
        """ Normalizes an image array based on the predefined image size value.
        :param img_array:
        :return: resized array
        """
        if img_array is None or len(img_array) <= 0:
            return None
        return cv2.resize(img_array, (self.img_size, self.img_size))

    @staticmethod
    def shuffle(data):
        """ Shuffles a data list randomly.
        :param data
        :return: shuffled list
        """
        if type(data) is not list or len(data) <= 0:
            return None
        random.shuffle(data)
        return data
