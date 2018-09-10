from sklearn import cluster
import constant
import numpy as np

def KMeans(x, clusters):
    kmeans = cluster.KMeans(clusters).fit(x)
    return kmeans.labels_


def AffinityPropapagation(x, clusters):
    affinity = cluster.AffinityPropagation(damping=0.5).fit(x)
    return affinity.labels_


def MeanShift(x, clusters):

    meanshift = cluster.MeanShift(n_jobs=-2).fit(x)
    return meanshift.labels_


def DBSCAN(x, clusters):

    db = cluster.DBSCAN(eps=0.1, min_samples=15).fit(x)
    return db.labels_


def get_l(i):

    sum = 0
    for j in i:
        sum += j**2

    return np.sqrt(sum)