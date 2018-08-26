from sklearn import cluster

def KMeans(x, clusters):
    kmeans = cluster.KMeans(clusters).fit(x)
    return kmeans.labels_


def AffinityPropapagation(x, clusters):
    affinity = cluster.AffinityPropagation().fit(x)
    return affinity.labels_

