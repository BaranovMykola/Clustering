from sklearn.datasets.samples_generator import make_blobs
import hypertools as hyp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import constant
import copy
# x, y = make_blobs(n_samples=2000, centers=3, n_features=3, random_state=1, cluster_std=2)
#
# hyp.plot(x, '.', hue=y)

def image_to_points(img):
    shape = img.shape
    square = shape[0]*shape[1]
    img = img.reshape((square,img.shape[2]))

    return img

def colors_code(img):
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return

def show_std(samples, clusters, dim):
    plt.xlabel('Standart deviation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Samples: {0} Clusters: {1} Dimensional {2}'.format(samples, clusters, dim))
    plt.grid(color='#dddddd', linestyle='-', linewidth=1)
    plt.title('Accuracy corelation from deviation')
    plt.show()


def visual(points, labels, shape):
    bgra = np.zeros(shape, np.uint8)
    # bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
    color = map_labels(labels)
    bgra = image_to_points(bgra)

    alpha = 127

    for i in range(len(labels)):
        c = None
        c = copy.deepcopy(color[ labels[i] ])
        # c.append(alpha)
        bgra[i] = c
        # c.clear()

    bgra = points_to_img(bgra, shape)
    img = points_to_img(points, shape)

    alpha = 0.6
    imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgg = cv2.cvtColor(imgg, cv2.COLOR_GRAY2BGR)

    res = ((bgra*alpha+imgg*(1-alpha))).astype(np.uint8)

    res = np.hstack((bgra, img, res))

    cv2.imshow('res', res)
    cv2.waitKey()


def points_to_img(points, shape):
    return points.reshape(shape)


def map_labels(labels):
    color = {}
    idx = 0

    for i in labels:
        if i not in color.keys():
            color[i] =  constant.mask_colors[idx%len(constant.mask_colors)]
            idx+=1

    return color