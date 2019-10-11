#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """

import preprocessing as pp
import learner as learn
import numpy as np


IMG_SIZE = 128


p = pp.Preprocessor(img_size=IMG_SIZE)
#train, test = p.run(partial_load=0.5)
#p.save()
train, test = p.load()
y_train, x_train = p.sample_label_disjoin(train)
y_test, x_test = p.sample_label_disjoin(test)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.expand_dims(np.array(y_train, dtype=np.uint8), 1)
y_test = np.expand_dims(np.array(y_test, dtype=np.uint8), 1)

model = learn.ImageClassifier(input_shape=x_train.shape[1:], model_path=None)#"../res/models/model_001_004.model")
#model.train(x_train, y_train, x_val=None, y_val=None, validation_split=0.2, batch_size=128, epochs=5)
#model.evaluate(x_test, y_test)
#model.save()

model.multi_prediction(preprocessor=p, num_img_per_category=5,
                       categories=["Human", "NoHuman", "RealHuman", "RealNoHuman"])

"""
# until here ok
for i in range(0, 20):
    plt.imshow(x_train[i][:, :, 0], cmap=plt.cm.binary)
    plt.show()
    print(y_train[i])
    print()"""