#!/bin/bash

# -------------------------------------------
# author:     Johann Schmidt
# date:       October 2019
# -------------------------------------------

#echo "Starting learner ..."
#~/git/faceDeep/venv/bin/python3 ~/git/faceDeep/src/faceDeep.py

echo "Starting Tensor Board in Browser ..."
/usr/bin/tensorboard --logdir=logs/ &
sleep 2s
chromium http://localhost:6006/ -incognito
