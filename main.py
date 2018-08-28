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


import resource

rsrc = resource.RLIMIT_DATA
soft, hard = resource.getrlimit(rsrc)
# print ('Soft limit starts as  {0}'.format( soft))
#
resource.setrlimit(rsrc, (1024*1024*2*1024, hard)) #limit to one kilobyte
#
# soft, hard = resource.getrlimit(rsrc)
# print ('Soft limit starts as  {0}'.format( soft))


samples = 10000
clusters = 4
dim = 3
img = cv2.imread('./Images/mix.jpg')
std = 2

if __name__ == '__main__':
    # test.test_std(samples=samples, clusters=clusters, dim=dim, space=np.linspace(0.1, 10, 10), funcs=constant.funcs, names=constant.names, cs=constant.cs)
    # test.test_dim(samples=samples, clusters=clusters, std=std, space=range(20,25), funcs=constant.funcs, names=constant.names, cs=constant.cs)
    # test.test_samples(dim=dim, clusters=clusters, std=std, space=range(200,3000, 500), funcs=constant.funcs, names=constant.names, cs=constant.cs)
    # test.test_clusters(dim=dim, samples=samples, std=std, space=range(2, 10, 1), funcs=constant.funcs, names=constant.names, cs=constant.cs)

    # test.single_cluster(dim=dim, samples=samples, clusters=clusters, std=std, cluster_func=constant.funcs[2])

    test.test_image('./Images/', './Processed/', constant.funcs, constant.names)
