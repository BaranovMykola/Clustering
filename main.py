from sklearn.datasets.samples_generator import make_blobs
import utils
import cluster
import hypertools as hyp
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from kmeans import *
import cv2
import test
from cluster import *
import constant

samples = 10
clusters = 2
dim = 2
img = cv2.imread('./Images/cube.jpg')

if __name__ == '__main__':
    # test.test_std(samples=samples, clusters=clusters, dim=dim, space=np.linspace(0.1, 10, 10), funcs=constant.funcs, names=constant.names, cs=constant.cs)


    test.cluster_image(img, cv2.COLOR_BGR2HSV, 4, constant.funcs[0], "KMenas")
