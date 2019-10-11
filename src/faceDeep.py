#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """

import preprocessing as pp
import learner as learn
import numpy as np
import std_models.vgg as vgg


IMG_SIZE = 224
CATEGORIES = ["Human", "NoHuman"]
DATADIR = "/mnt/HDD/Masterthesis/DB"
PROCESSED_IMG_DIR = "../res/data"

p = pp.Preprocessor(
    img_size=IMG_SIZE, categories=CATEGORIES,
    colormode=pp.Colormode.RGB, datadir=DATADIR,
    data_pattern=pp.DataPattern.X_X_Y_Y)
x_train, x_test, y_train, y_test = p.run(partial_load=0.01)
p.save(PROCESSED_IMG_DIR, (x_train, x_test, y_train, y_test))
x_train, x_test, y_train, y_test = p.load(PROCESSED_IMG_DIR)

model = learn.ImageClassifier(input_shape=x_train.shape[1:], model_path="../res/models/model_001_004.model")#"../res/models/model_001_004.model")
model.train(x_train, y_train, x_val=None, y_val=None, validation_split=0.2, batch_size=128, epochs=5)
model.evaluate(x_test, y_test)
model.save()

model.multi_prediction(preprocessor=p, num_img_per_category=5,
                       categories=["Human", "NoHuman", "RealHuman", "RealNoHuman"])

"""
# until here ok
for i in range(0, 20):
    plt.imshow(x_train[i][:, :, 0], cmap=plt.cm.binary)
    plt.show()
    print(y_train[i])
    print()"""