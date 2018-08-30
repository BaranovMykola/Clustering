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
resource.setrlimit(rsrc, (1024*1024*8*1024, hard)) #limit to one kilobyte
#
# soft, hard = resource.getrlimit(rsrc)
# print ('Soft limit starts as  {0}'.format( soft))


samples = 4
clusters = 2
dim = 2
std = 10

if __name__ == '__main__':
    # test.test_std(samples=samples, clusters=clusters, dim=dim, space=np.linspace(0.1, 10, 10), funcs=constant.funcs, names=constant.names, cs=constant.cs)
    # test.test_dim(samples=samples, clusters=clusters, std=std, space=range(2,15), funcs=constant.funcs, names=constant.names, cs=constant.cs)
    # test.test_samples(dim=dim, clusters=clusters, std=std, space=range(200,3000, 500), funcs=constant.funcs, names=constant.names, cs=constant.cs)
    # test.test_clusters(dim=dim, samples=samples, std=std, space=range(2, 10, 1), funcs=constant.funcs, names=constant.names, cs=constant.cs)

    # test.single_cluster(dim=dim, samples=samples, clusters=clusters, std=std, cluster_func=AffinityPropapagation)
    # test.test_single(dim=dim, samples=samples, clusters=clusters, std=std, funcs=constant.funcs, names=constant.names, dst='./Blobs/')

    test.test_image('./Images/', './Processed/', constant.funcs, constant.names)
