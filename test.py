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
from sklearn.datasets import samples_generator
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
    res, mask = utils.visual(points, labels, img.shape, name, colorspace)

    return res, labels, mask


def test_image(folder, dst, func, names, clusters = 3):
    if os.path.isdir(dst):
        shutil.rmtree(dst)

    os.mkdir(dst)
    pts = 'pts'
    clustered_pts = 'clustered_pts'
    mask_dir = 'mask'

    os.mkdir(os.path.join(dst, pts))
    os.mkdir(os.path.join(dst, clustered_pts))
    os.mkdir(os.path.join(dst, mask_dir))

    for i in os.listdir(folder):
        print('Processing [{0}] image'.format(i))
        img = cv2.imread(os.path.join(folder, i))
        img = cv2.resize(img, (200,200))
        i = os.path.splitext(i)[0]
        for idx, f in enumerate(func):
            try:
                print('\tStarted processing {0} clustering algorithm'.format(names[idx]))
                res, labels, mask = cluster_image(img, None, clusters, f, '')
                print('\t\tClustered. Prepearing visualization')
                l = np.unique(labels)
                cv2.imwrite( os.path.join(dst, '{0}_{1}_{2}_clusters.png'.format(i, names[idx], l.shape[0])), res )
                hyp.plot(utils.image_to_points(img), '.', group=labels, save_path=os.path.join(dst, '{0}_{1}_scatter.png'.format(i, names[idx])), show=False)

                hyp.plot(utils.image_to_points(img), '.',
                         save_path=os.path.join(dst, pts, '{0}_{1}_{2}.png'.format(i, names[idx], l.shape[0])), show=False)

                hyp.plot(utils.image_to_points(img), '.', group=labels,
                         save_path=os.path.join(dst, clustered_pts, '{0}_{1}_{2}.png'.format(i, names[idx], l.shape[0])), show=False)

                cv2.imwrite( os.path.join(dst, mask_dir, '{0}_{1}_{2}.png'.format(i, names[idx], l.shape[0])), mask )
            except:
                with open( os.path.join(dst, '{0}_{1}.txt'.format(i, names[idx])), 'w+') as f:
                    f.write('Error while processing')
            print('\t\tProcessed {0} clustering algorithm'.format(names[idx]))


def single_cluster(dim, samples, std, clusters, cluster_func, x = None, y = None):
    # x, y = make_blobs(n_samples=samples, centers=clusters, n_features=dim, random_state=0, cluster_std=std)
    if x is None or y is None:
        x, y = samples_generator.make_circles(n_samples=samples, random_state=True, factor=0.3, noise=0.05)

    _y = cluster_func(x, clusters)

    acc = sklearn.metrics.homogeneity_score(y,_y)

    hyp.plot(x, '.', group=y, save_path='or.png')
    hyp.plot(x, '.', group=_y, save_path='grp.png')

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


def test_points(points_folder, funcs, names, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)

    points_names = os.listdir(points_folder)
    for i in points_names:
        x,y = utils.read_points(os.path.join(points_folder, i))
        print('Started processing {0} with {1} points'.format(i, len(y)))
        for j in range(len(funcs)):
            print('\tProcessing {0} algorithm'.format(names[j]))
            _y = funcs[j](x, len(np.unique(y)))
            hyp.plot(x, '.', group=_y, save_path=os.path.join(dst, '{0}_{1}_{2}_clusters.png'.format(os.path.splitext(i)[0], names[j], len(np.unique(_y)))), show=False)
            print('\tProcessed.')
            # print('Clustered using {0} algorith'.format(names[i]))
