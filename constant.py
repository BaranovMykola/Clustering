from cluster import *
import random
import math
import numpy as np

def generate_colors(clusters):
        colors = []
        step_q = math.pow(clusters, 1.0/3)
        step_q = math.ceil(step_q)
        step = math.ceil(255.0/step_q)

        for r in np.linspace(0, 255, step_q):
                for g in np.linspace(0, 255, step_q):
                        for b in np.linspace(0, 255, step_q):
                                colors.append([int(b),int(g),int(r)])
        random.shuffle(colors)
        return colors

# funcs = [KMeans, AffinityPropapagation, MeanShift, DBSCAN]
# names = ['K-Means', 'Affinity Propapagation', 'Mean Shift', 'DBSCAN']

funcs = [DBSCAN]
names = ['DBSCAN']

color = '#FFFFFF #C0C0C0 #808080 #000000 #FF0000' \
        ' #800000 #FFFF00 #808000 #00FF00 #008000' \
        ' #00FFFF #008080 #FF00FF #800080 #0000FF ##000080'.split(' ')

cs = 'r g b c'.split(' ')

mask_colors = generate_colors(10)

mask_ratio = 0.5


mean_shift_bandwidth = 59