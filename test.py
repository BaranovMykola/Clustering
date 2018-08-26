import cluster
import sklearn
import utils
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from scipy.interpolate import interp1d
from scipy.interpolate import spline


def std(samples, clusters, dim, space, cluster_func):
    lst= []
    # space = np.linspace(0.01, 10, 10)
    # samples = 1000
    # centers = 4
    # dim = 2

    for i in space:
        x, y = make_blobs(n_samples=samples, centers=clusters, n_features=dim, random_state=1, cluster_std=i)
        _y = cluster_func(x, clusters)

        acc = sklearn.metrics.homogeneity_score(y,_y)
        lst.append(acc)

    return np.array(lst)


def test_std(samples, clusters, dim, space, funcs, names, cs):
    for i in range(len(funcs)):
        y = std(samples, clusters, dim, space, funcs[i])
        plt.plot(space, y, cs[i] + 'o')
        plt.plot(space, y, cs[i] + '--', label=names[i])
    utils.show_std(samples, clusters, dim)


def cluster_image(img, colorspace, clusters, func, name):
    points = utils.image_to_points(img)

    labels = func(points, clusters)

    utils.visual(points, labels, img.shape)

    pass