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


IMG_SIZE = 150
CATEGORIES = ["Human", "NoHuman"]
DATADIR = "/mnt/HDD/Masterthesis/DB"
PARTIAL_LOAD = 0.1
PARTIAL_LOAD_STR = "01"
PROCESSED_IMG_DIR = "../res/data/p{0}_s{1}".format(PARTIAL_LOAD_STR, str(IMG_SIZE))

p = pp.Preprocessor(
    img_size=IMG_SIZE, categories=CATEGORIES,
    colormode=pp.Colormode.GRAYSCALE, datadir=DATADIR,
    data_pattern=pp.DataPattern.X_X_Y_Y)
#x_train, x_test, y_train, y_test = p.run(partial_load=PARTIAL_LOAD)
#p.save(PROCESSED_IMG_DIR, (x_train, x_test, y_train, y_test))
x_train, x_test, y_train, y_test = p.load(PROCESSED_IMG_DIR)

model = learn.ImageClassifier(input_shape=x_train.shape[1:], model_path="../res/models/model_001_006.model")
#model.train(x_train, y_train, x_val=None, y_val=None, validation_split=0.2, batch_size=132, epochs=5)
#model.evaluate(x_test, y_test)
#model.save()

model.multi_prediction(preprocessor=p, num_img_per_category=10,
                       categories=["RealHuman", "RealNoHuman"])
