import cluster
import sklearn
import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import hypertools as hyp
import shutil

from sklearn.datasets.samples_generator import make_blobs
from scipy.interpolate import interp1d
from scipy.interpolate import spline


def std(samples, clusters, dim, space, cluster_func):
    lst= []

    for i in space:
        x, y = make_blobs(n_samples=samples, centers=clusters, n_features=dim, random_state=1, cluster_std=i)
        _y = cluster_func(x, clusters)

        acc = sklearn.metrics.homogeneity_score(y,_y)
        lst.append(acc)

    return np.array(lst)


def dim(samples, clusters, std, space, cluster_func):
    lst= []

    for i in space:
        x, y = make_blobs(n_samples=samples, centers=clusters, n_features=i, random_state=1, cluster_std=std)
        _y = cluster_func(x, clusters)

        acc = sklearn.metrics.homogeneity_score(y,_y)
        lst.append(acc)

    return np.array(lst)


def samples(dim, clusters, std, space, cluster_func):
    lst= []

    for i in space:
        x, y = make_blobs(n_samples=i, centers=clusters, n_features=dim, random_state=1, cluster_std=std)
        _y = cluster_func(x, clusters)

        acc = sklearn.metrics.homogeneity_score(y,_y)
        lst.append(acc)

    return np.array(lst)


def clusters(dim, samples, std, space, cluster_func):
    lst= []

    for i in space:
        x, y = make_blobs(n_samples=samples, centers=i, n_features=dim, random_state=1, cluster_std=std)
        _y = cluster_func(x, i)

        acc = sklearn.metrics.homogeneity_score(y,_y)
        lst.append(acc)

    return np.array(lst)


def test_std(samples, clusters, dim, space, funcs, names, cs):
    for i in range(len(funcs)):
        y = std(samples, clusters, dim, space, funcs[i])
        plt.plot(space, y, cs[i] + 'o')
        plt.plot(space, y, cs[i] + '--', label=names[i])
    utils.show_std(samples, clusters, dim)


def test_dim(samples, clusters, std, space, funcs, names, cs):
    for i in range(len(funcs)):
        y = dim(samples, clusters, std, space, funcs[i])
        plt.plot(space, y, cs[i] + 'o')
        plt.plot(space, y, cs[i] + '--', label=names[i])
    utils.show_dim(samples, clusters, std)


def test_samples(dim, clusters, std, space, funcs, names, cs):
    for i in range(len(funcs)):
        y = samples(dim, clusters, std, space, funcs[i])
        plt.plot(space, y, cs[i] + 'o')
        plt.plot(space, y, cs[i] + '--', label=names[i])
    utils.show_samples(dim, clusters, std)


def test_clusters(dim, samples, std, space, funcs, names, cs):
    for i in range(len(funcs)):
        y = clusters(dim, samples, std, space, funcs[i])
        plt.plot(space, y, cs[i] + 'o')
        plt.plot(space, y, cs[i] + '--', label=names[i])
    utils.show_clusters(dim, samples, std)



def cluster_image(img, colorspace, clusters, func, name):
    if colorspace is not None:
        img = cv2.cvtColor(img, colorspace)
    points = utils.image_to_points(img)

    labels = func(points, clusters)
    res = utils.visual(points, labels, img.shape, name, colorspace)

    return res, labels


def test_image(folder, dst, func, names, clusters = 3):
    shutil.rmtree(dst)
    os.mkdir(dst)

    for i in os.listdir(folder):
        print('Processing [{0}] image'.format(i))
        img = cv2.imread(os.path.join(folder, i))
        img = cv2.resize(img, (150,150))
        i = os.path.splitext(i)[0]
        for idx, f in enumerate(func):
            try:
                print('\tStarted processing {0} clustering algorithm'.format(names[idx]))
                res, labels = cluster_image(img, None, clusters, f, '')
                print('\t\tClustered. Prepearing visualization')
                l = np.unique(labels)
                cv2.imwrite( os.path.join(dst, '{0}_{1}_{2}_clusters.png'.format(i, names[idx], l.shape[0])), res )
                hyp.plot(utils.image_to_points(img), '.', group=labels, save_path=os.path.join(dst, '{0}_{1}_scatter.png'.format(i, names[idx])), show=False)
            except:
                with open( os.path.join(dst, '{0}_{1}.txt'.format(i, names[idx])), 'w+') as f:
                    f.write('Error while processing')
            print('\t\tProcessed {0} clustering algorithm'.format(names[idx]))


def single_cluster(dim, samples, std, clusters, cluster_func):
    x, y = make_blobs(n_samples=samples, centers=clusters, n_features=dim, random_state=1, cluster_std=std)
    _y = cluster_func(x, clusters)

    acc = sklearn.metrics.homogeneity_score(y,_y)

    hyp.plot(x, '.', group=y)
    hyp.plot(x, '.', group=_y)

    print('Accuracy {0:0.2f}'.format(acc))


def test_single_cluster(dim, samples, std, clusters, funcs, names, dst):
    shutil.rmtree(dst)
    os.mkdir(dst)

    x, y = make_blobs(n_samples=samples, centers=clusters, n_features=dim, random_state=1, cluster_std=std)
    hyp.plot(x, '.', save_path=os.path.join(dst, 'original.png'), show=False)
    hyp.plot(x, '.', group=y, save_path=os.path.join(dst, 'original_clustered.png'), show=False)

    print('Generated points')

    for i in range(len(funcs)):
        _y = funcs[i](x, clusters)
        hyp.plot(x, '.', group=_y, save_path=os.path.join(dst, names[i]+'.png'), show=False)
        print('Clustered using {0} algorith'.format(names[i]))