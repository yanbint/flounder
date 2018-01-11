#!/usr/bin/python
# -*- coding: utf-8 -*-

#   following command:
#       ./Flounder.py ./images/*.jpg

from sklearn.cluster import KMeans   
import matplotlib.pyplot as plt      # matplotlib show image
import matplotlib.image as mpimg
import argparse                                                            
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.utils import shuffle
import os
import sys
from PIL import Image
import cv2
from DataClean import data_clean

class SkinModel(object):
    def __init__(self, skin_model = os.path.split(os.path.realpath(__file__))[0] + os.sep + "MODEL.csv"):
        self._skin_model = skin_model
        self._simple = np.array([])
        self.load_model()

    def load_model(self):
        print "load_model"
        models = []
        with open(self._skin_model, 'r') as fin:
            for line in fin:
                line = line[:-1]
                s = line.split(';')
                models.append([int(s[0]), int(s[1]), int(s[2])])
        self._simple = np.array(models)
        self.model = KMeans(n_clusters=3, random_state=0).fit(self._simple)

    def show(self):
        fig = plt.figure()
        color = ("red", "green", "blue")
        ax= fig.add_subplot(111, projection='3d')
        y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(self._simple)
        colors=np.array(color)[y_pred]
        ax.scatter(self._simple[:, 0], self._simple[:, 1], self._simple[:, 2], c=colors)
        plt.show()

class SkinDetect(object):
    def __init__(self, skin_model, simple_num = 100):
        #self._image = image
        self._skin_model = skin_model
        self._simple = np.array([])
        self._simple_num = simple_num
        self._result = []
        self.black = 0 
        self.white = 0
        self.yellow = 0
        self._skin_mode = skin_model

    def skin_detect(self, image):
        skin_data = data_clean(image, "HSV")
        self._simple = shuffle(np.array(skin_data), random_state=0)[:self._simple_num]
        n = 0
        m = 0
        self._result = self._skin_mode.model.predict(self._simple)
        for i in self._result:
            if i == 1:
                n = n + 1
            if i == 2:
                m = m + 1
        self.black = (n) * 100/self._simple_num
        self.white = (self._simple_num-n-m) * 100/self._simple_num
        self.yellow = (m) * 100/self._simple_num
        self._image = image

    def show_result(self):
        fig = plt.figure()
        ax= fig.add_subplot(131, projection='3d')
        ax.set_title("Origin image data:", fontsize = 12, loc = 'left')
        ax.scatter(self._simple[:, 0], self._simple[:, 1], self._simple[:, 2], c="purple", marker='.')

        color = ("red", "green", "blue")
        ax= fig.add_subplot(132, projection='3d')
        ax.set_title("Predict image data:", fontsize = 12, loc = 'left')
        colors = np.array(color)[self._result]
        ax.scatter(self._simple[:, 0], self._simple[:, 1], self._simple[:, 2], c=colors, marker='x')
        
        img = Image.open(self._image)
        ax= fig.add_subplot(133)
        ax.imshow(img)
        ax.set_title("Black: %d%%, White: %d%%, Yellow: %d%%" % (self.black, self.white, self.yellow), fontsize = 12, loc = 'left')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def __del__(self):
        print "delete SkinDetect"

if __name__=="__main__":
    mySkinModel = SkinModel()
    #mySkinModel.show()
    mySkinDetect = SkinDetect(mySkinModel)

    for image in sys.argv[1:]:
        #print "Processing file: {}".format(image)
        mySkinDetect.skin_detect(image)
        mySkinDetect.show_result()
        print "%s: [Black-%02d%%, Yellow-%02d%%, White-%02d%%]" % (image, mySkinDetect.black, mySkinDetect.yellow, mySkinDetect.white)
