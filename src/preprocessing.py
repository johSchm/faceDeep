import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import sklearn
import pandas as pd
import itertools


"""
https://github.com/tkarras/progressive_growing_of_gans
"""

DATADIR = "/mnt/HDD/Masterthesis/DB"
CATEGORIES = ["Human", "NoHuman"]
IMG_SIZE = 120


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


class Preprocessor:
    """ A data preprocessor for preparing the data for the learner.
    """

    def __init__(self, datadir=DATADIR, categories=CATEGORIES, img_size=IMG_SIZE):
        """ Initialization method.
        :param datadir:
        :param categories:
        :param img_size:
        """
        self.datadir = datadir
        self.categories = categories
        self.img_size = img_size
        self.data = []
        self.train_data = []
        self.test_data = []

    def run(self):
        """ Starts the pre-processing routine.
        """
        class_distro = self.class_distribution()
        if not self.balanced(class_distro):
            self.random_under_sampling()
        self.load_data()
        self.shuffle(self.data)
        self.train_data, self.test_data = self.split_train_test(self.data)

    def class_distribution(self):
        """ Returns a dictionary with the number of samples per class.
        :return: dictionary
        """
        if self.datadir is None or self.categories is None:
            return None
        class_distro = {}
        for category in self.categories:
            path = os.path.join(self.datadir, category)
            class_distro[category] = os.listdir(path)
        return class_distro

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
        for category in self.categories:
            category_set = []
            path = os.path.join(self.datadir, category)
            class_num = self.categories.index(category)
            for img in tqdm(os.listdir(path)):
                try:
                    img_array = self.load_sample(category, img, param)
                    if resize:
                        img_array = self.resize_images(img_array)
                    category_set.append([img_array, class_num])
                except Exception as e:
                    print("Exception occurred! Continuing ...")
                    pass
            self.data.append(category_set)
        return self.data

    def load_sample(self, category, img_name, param=None):
        """ Loads an sample image from the given path.
        :param category
        :param img_name
        :param param for image loading
        :return: sample
        """
        if category is None or img_name is None:
            return None
        path = os.path.join(self.datadir, category, img_name)
        if param is not None:
            return cv2.imread(path, param)
        return cv2.imread(path)

    @staticmethod
    def balanced(*sets):
        """ Checks whether the sets a balanced or not.
        :param sets:
        :return: True or False
        """
        if sets is None or len(sets) <= 1:
            return False
        if type(sets) is list:
            first_set_len = len(sets[0])
            for _set in sets:
                if len(_set) != first_set_len:
                    return False
            return True
        if type(sets) is dict:
            return equal_values(sets)
        return False

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
    def smallest_set(sets):
        """ Returns the smallest sets index.
        :param sets
        :return index
        """
        if sets is None or type(sets) is not list or len(sets) <= 0:
            return None
        if len(sets) == 1:
            return 0
        m, min_len = 0, 0
        for i, _set in enumerate(sets):
            if len(_set) < min_len:
                m, min_len = i, len(_set)
        return m

    def random_under_sampling(self):
        """ Randomly selects samples from the larger set, such that the sizes matches.
        :return: new sets
        """
        if self.data is None:
            return None
        target_set_idx = self.smallest_set(self.data)
        target_set_len = self.data[target_set_idx]
        for i, category_set in enumerate(self.data):
            if i != target_set_idx:
                self.data[i] = self.downsize_set(category_set, target_set_len)
        return self.data

    @staticmethod
    def downsize_set(_set, target_len):
        """ Downsize a set to a specific target length.
        :param _set:
        :param target_len:
        :return: reduced set
        """
        if _set is None or len(_set) <= 0:
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
        if type(sample_list) is not np.array:
            sample_list = np.array(sample_list)
        if type(label_list) is not np.array:
            label_list = np.array(label_list)
        return np.column_stack(sample_list, label_list)

    def split_train_test(self, data, test_size=0.2, random_state=42):
        """ Splits the data into training and test set.
        :param data
        :param test_size
        :param random_state
        :return: training set, test set
        """
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            data[0], data[1], test_size=test_size, random_state=random_state)
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
        :param data:
        :return: shuffled list
        """
        if data is None or type(data) is not list or len(data) <= 0:
            return None
        return random.shuffle(data)
