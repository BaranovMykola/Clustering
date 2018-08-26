from sklearn.datasets.samples_generator import make_blobs
import utils
import cluster
import hypertools as hyp
import sklearn
import numpy as np
import matplotlib.pyplot as plt

def KMeans_std():
    lst= []
    space = np.linspace(0.01, 10, 10)
    samples = 1000
    centers = 4
    dim = 2

    for i in space:
        x, y = make_blobs(n_samples=samples, centers=centers, n_features=dim, random_state=1, cluster_std=i)
        _y = cluster.KMeans(x, 4)

        acc = sklearn.metrics.homogeneity_score(y,_y)
        lst.append(acc)

    plt.plot(space, np.array(lst), 'r')
    plt.xlabel('Standart deviation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('S1amples: {0} Centers: {1} Dimensional {2}'.format(samples, centers, dim))
    plt.grid(color='#dddddd', linestyle='-', linewidth=1)
    plt.show()


def KMeans_samples():
    lst= []
    centers = 4
    space = range(centers, 1000, 10)
    dim = 2
    std = 2

    for i in space:
        x, y = make_blobs(n_samples=i, centers=centers, n_features=dim, random_state=1, cluster_std=std)
        _y = cluster.KMeans(x, 4)

        acc = sklearn.metrics.homogeneity_score(y,_y)
        lst.append(acc)

    plt.plot(space, np.array(lst), 'r')
    plt.xlabel('Standart deviation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('STD: {0} Centers: {1} Dimensional {2}'.format(std, centers, dim))
    plt.grid(color='#dddddd', linestyle='-', linewidth=1)
    plt.show()

def KMeans_image(img):
    s = img.shape
    img = img.reshape( (img.shape[0]*img.shape[1], 3) )
    std = np.std(img)
    y = cluster.KMeans(img, 5)
    colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,0,255)]
    for i, v in enumerate(y):
        img[i] = colors[v]

    img = img.reshape(s)

    return img