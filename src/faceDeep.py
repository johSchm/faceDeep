#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """

import preprocessing as pp
import learner as learn
import numpy as np
import os


p = pp.Preprocessor(img_size=120)
train, test = p.run(partial_load=0.01)
#p.save()
#train, test = p.load()
y_train, x_train = p.sample_label_disjoin(train)
y_test, x_test = p.sample_label_disjoin(test)

x_train = np.expand_dims(np.array(x_train), axis=3)
y_train = np.array(y_train)
x_test = np.expand_dims(np.array(x_test), axis=3)
y_test = np.array(y_test)

# until here ok
for i in range(0, 20):
    p.show_sample(x_train[i])
    print(y_train[i])
    print()

model = learn.ImageClassifier(input_shape=x_train.shape[1:], model_path=None,
                              layer_size=128,
                              num_conv_layers=3, num_dense_layers=2
                              )#"../res/models/model_1-001.model")

model.train(x_train, y_train, batch_size=16, epochs=3)
model.evaluate(x_test, y_test)
model.save()

print("Starting prediction phase ...")
categories = ["RealHuman", "RealNoHuman"]
for category in categories:
    path = os.path.join("/mnt/HDD/Masterthesis/DB", category)
    for img_name in os.listdir(path):
        img = p.load_sample(os.path.join(path, img_name))
        img = p.resize_image(img)
        img = np.expand_dims(img, axis=2)
        prediction = model.predict(img)
        prediction_mapped = p.categories[prediction]
        print("Prediction of {0} in {1}: {2} ({3})".format(
            img_name, category, prediction_mapped, prediction))
