from sklearn.datasets.samples_generator import make_blobs
import hypertools as hyp

x, y = make_blobs(n_samples=2000, centers=3, n_features=3, random_state=1, cluster_std=2)

hyp.plot(x, '.', hue=y)