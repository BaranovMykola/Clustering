from sklearn import cluster
import constant
import numpy as np

def KMeans(x, clusters):
    kmeans = cluster.KMeans(clusters).fit(x)
    return kmeans.labels_


def AffinityPropapagation(x, clusters):
    affinity = cluster.AffinityPropagation(damping=0.9).fit(x)
    return affinity.labels_


def MeanShift(x, clusters):

    meanshift = cluster.MeanShift(bandwidth=constant.mean_shift_bandwidth, n_jobs=-2).fit(x)
    return meanshift.labels_


def get_l(i):

    sum = 0
    for j in i:
        sum += j**2

    return np.sqrt(sum)