from sklearn.datasets.samples_generator import make_blobs
import hypertools as hyp
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import constant
import copy
import hypertools as hyp


# x, y = make_blobs(n_samples=2000, centers=3, n_features=3, random_state=1, cluster_std=2)
#
# hyp.plot(x, '.', hue=y)

def image_to_points(img):
    shape = img.shape
    square = shape[0] * shape[1]
    img = img.reshape((square, img.shape[2]))

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
    plt.show()

def show_dim(samples, clusters, dim):
    plt.xlabel('Dimensional')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Samples: {0} Clusters: {1} Dimensional: {2}'.format(samples, clusters, dim))
    plt.grid(color='#dddddd', linestyle='-', linewidth=1)
    plt.show()


def show_samples(dim, clusters, std):
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Dimensional: {0} Clusters: {1} Std {2}'.format(dim, clusters, std))
    plt.grid(color='#dddddd', linestyle='-', linewidth=1)
    plt.show()


def show_clusters(dim, samples, std):
    plt.xlabel('Clusters')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Dimensional: {0} Samples: {1} Std {2}'.format(dim, samples, std))
    plt.grid(color='#dddddd', linestyle='-', linewidth=1)
    plt.show()


def visual(points, labels, shape, name, colorspace):
    mask = np.zeros(shape, np.uint8)

    # color = map_labels_advance(labels, points_to_img(points, shape))
    color = map_labels(labels)

    mask = image_to_points(mask)

    for i in range(len(labels)):
        c = copy.deepcopy(color[labels[i]])
        mask[i] = c

    mask = points_to_img(mask, shape)
    img = points_to_img(points, shape)

    mask_2 = ((mask * constant.mask_ratio + img * (1 - constant.mask_ratio))).astype(np.uint8)

    res = np.hstack((img, mask, mask_2))

    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 2
    t = 4
    a = cv2.getTextSize(name, font, fs, t)
    cv2.putText(res, name, (0,a[0][1]), font, fs, (255,255,255), t, cv2.LINE_AA)

    return res, mask

def points_to_img(points, shape):
    return points.reshape(shape)


def map_labels(labels):
    color = {}
    idx = 0
    c = np.unique(labels).shape[0]
    mask_colors = constant.generate_colors( clusters=c )

    for i in labels:
        if i not in color.keys():
            color[i] = mask_colors[idx % len(constant.mask_colors)]
            idx += 1

    return color


def map_labels_advance(labels, img):
    color = {}
    idx = 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = image_to_points(img).astype(np.float32)

    for i in range(len(labels)):
        img[i][3] = labels[i]

    for i in labels:
        if i not in color.keys():
            # color[i] = constant.mask_colors[idx % len(constant.mask_colors)]
            color[i] = mean_if(img, i)
            idx += 1

    return color


def mean_if(p, l):
    sum = 0
    idx = 0
    for i in p:
        if i[3] == l:
            sum += i[0:3]
            idx += 1

    sum /= idx
    return sum.astype(np.uint8)

def read_points(file):
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    x = []
    y = []

    for i in lines:
        j = i.split(' ')
        x.append([float(j[0]), float(j[1])])
        y.append(float(j[2]))

    return np.array(x), np.array(y)


class ImagePointCreator:

    def __init__(self, img):
        self.image = img
        self.x = []
        self.y = []
        self.label = 0
        self.count = 100
        self.range = 50

    # def _click_and_crop(event, x, y, flags, param):

    def click_and_crop(self,event, x, y, flags, param):
        # grab references to the global variables
        print('Label {0}'.format(self.label))
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(self.count):
                _x = random.randint(max(0,x-self.range), min(999, x+self.range))
                _y = random.randint(max(0, y - self.range), min(999, y + self.range))
                self.x.append([_x,1000-_y])
                self.y.append(self.label)
            print ('Points [{0},{1]] written'.format(x,y))

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.label += 1

            # draw a rectangle around the region of interest
        for i in self.x:
            self.image[max(1000-i[1], 0), i[0]] = 255
        # cv2.rectangle(self.image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", self.image)


def create_points(max, file):
    img = np.zeros((1000, 1000), np.uint8)
    ipc = ImagePointCreator(img)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", ipc.click_and_crop)
    cv2.imshow("image", img)
    cv2.waitKey()

    x = np.array(ipc.x)
    y = np.array(ipc.y)

    with open(file, 'w+') as f:
        for i in range(len(x)):
            f.write('{0} {1} {2}\n'.format(x[i][0]*max/1000.0, x[i][1]*max/1000.0, y[i]))

