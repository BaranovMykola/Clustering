from cluster import *

funcs = [KMeans, AffinityPropapagation]
names = ['K-Means', 'Affinity Propapagation']

color = '#FFFFFF #C0C0C0 #808080 #000000 #FF0000' \
        ' #800000 #FFFF00 #808000 #00FF00 #008000' \
        ' #00FFFF #008080 #FF00FF #800080 #0000FF ##000080'.split(' ')
cs = 'r g b'.split(' ')

mask_colors = [[0,0,0], [0,0,255], [0,255,0], [255,0,0], [255,0,255], [255,255,0], [255,255,255]]