#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:34:54 2018

@author: anviol
"""

# https://marco.ccr.buffalo.edu/

# Train
# https://marco.ccr.buffalo.edu/data/train/train-00001-of-00407
# https://marco.ccr.buffalo.edu/data/train/train-00407-of-00407

# Validate
# https://marco.ccr.buffalo.edu/data/test/test-00001-of-00046
# https://marco.ccr.buffalo.edu/data/test/test-00046-of-00046

import requests
import os

os.chdir('/Users/anviol/Desktop/MARCO Project/train/')

for i in range(1, 408):
    url = "https://marco.ccr.buffalo.edu/data/train/train-" + str(i).zfill(5) + "-of-00407"
    r = requests.get(url)
    # you might have to change the extension
    with open("marcoTrainData" + str(i).zfill(5) + ".tfrecords", 'wb') as f:
        f.write(r.content)

os.chdir('/Users/anviol/Desktop/MARCO Project/validation/')

for i in range(1, 47):
    url = "https://marco.ccr.buffalo.edu/data/test/test-" + str(i).zfill(5) + "-of-00046"
    r = requests.get(url)
    # you might have to change the extension
    with open("marcoTestData" + str(i).zfill(5) + ".tfrecords", 'wb') as f:
        f.write(r.content)
