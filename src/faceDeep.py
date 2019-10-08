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
#p.run()
#p.save()
train, test = p.load()
y_train, x_train = p.sample_label_disjoin(train)
y_test, x_test = p.sample_label_disjoin(test)

x_train = np.expand_dims(np.array(x_train), axis=3)
y_train = np.array(y_train)
x_test = np.expand_dims(np.array(x_test), axis=3)
y_test = np.array(y_test)

model = learn.ImageClassifier(input_shape=x_train.shape[1:], load_existing_model=False)

model.train(x_train, y_train, batch_size=16)
model.evaluate(x_test, y_test)
model.log()
model.save()
