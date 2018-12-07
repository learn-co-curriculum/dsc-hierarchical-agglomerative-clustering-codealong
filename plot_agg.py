# this code is based Müller and Guido, https://github.com/amueller/mglearn/tree/master/mglearn

def plot_agglomerative():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import KernelDensity
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    m = 16
    k = 3
    X, y = make_blobs(n_samples= m, n_features=2, centers=k, cluster_std=1.3, random_state = 2255)
    agg = AgglomerativeClustering(n_clusters=3)

    eps = X.std() / 2.

    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    gridpoints = np.c_[xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)]

    ax = plt.gca()
    for i, x in enumerate(X):
        ax.text(x[0] + .1, x[1], "%d" % i, horizontalalignment='left', verticalalignment='center')

    ax.scatter(X[:, 0], X[:, 1], s=20, c='grey')
    ax.set_xticks(())
    ax.set_yticks(())

    for i in range((m-1)):
        agg.n_clusters = X.shape[0] - i
        agg.fit(X)

        bins = np.bincount(agg.labels_)
        for cluster in range(agg.n_clusters):
            if bins[cluster] > 1:
                points = X[agg.labels_ == cluster]
                other_points = X[agg.labels_ != cluster]

                kde = KernelDensity(bandwidth= 0.9).fit(points)
                scores = kde.score_samples(gridpoints)
                score_inside = np.min(kde.score_samples(points))
                score_outside = np.max(kde.score_samples(other_points))
                levels = .80 * score_inside + .20 * score_outside
                ax.contour(xx, yy, scores.reshape(100, 100), levels=[levels],
                           colors='k', linestyles='solid', linewidths=0.8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)