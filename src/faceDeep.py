#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """

import preprocessing as pp
import learner as learn
import numpy as np

p = pp.Preprocessor()
p.run()
p.save()
y_train, x_train = p.sample_label_disjoin(p.train_data)
y_test, x_test = p.sample_label_disjoin(p.test_data)
#x_train = np.array(x_train)
#y_train = np.array(y_train)
#x_test = np.array(x_test)
#y_test = np.array(y_test)

#x_train = x_train/255.0

model = learn.ImageClassifier(input_shape=None, load_existing_model=False)

x_train = np.expand_dims(np.array(x_train), axis=3)
y_train = np.expand_dims(np.array(y_train), axis=3)
x_test = np.expand_dims(np.array(x_test), axis=3)
y_test = np.expand_dims(np.array(y_test), axis=3)
'''for i, img in enumerate(x):
    for r, row in enumerate(img):
        for c, col in enumerate(row):
            x[i][r][c] = np.array(x[i][r][c],)
            print(x[i][r][c])'''

model.train(x_train, y_train)
model.evaluate(x_test, y_test)
model.save()
