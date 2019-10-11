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

model = learn.ImageClassifier(input_shape=x_train.shape[1:], model_path="../res/models/model_001_004.model",
                              layer_size=16,
                              num_conv_layers=3, num_dense_layers=1
                              )#"../res/models/model_001_002.model"
model.model.summary()
#model.train(x_train, y_train, x_val=None, y_val=None, validation_split=0.2, batch_size=128, epochs=5)
#model.evaluate(x_test, y_test)
#model.save()

print("Starting prediction phase ...")
categories = ["Human", "NoHuman", "RealHuman", "RealNoHuman"]
for category in categories:
    print("{} -----------------------------------------------------------".format(category))
    path = os.path.join("/mnt/HDD/Masterthesis/DB", category)
    images = list(os.listdir(path))
    for img_name in images[:5]:
        img = p.load_sample(os.path.join(path, img_name))
        img = p.resize_image(img)
        img = np.expand_dims(img, axis=2)
        prediction = model.predict(img)
        prediction_mapped = p.categories[int(round(prediction))]
        print("Prediction of {0} in {1}: {2} ({3})".format(
            img_name, category, prediction_mapped, prediction))

"""
# until here ok
for i in range(0, 20):
    plt.imshow(x_train[i][:, :, 0], cmap=plt.cm.binary)
    plt.show()
    print(y_train[i])
    print()"""