import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes, M=1e3, alpha=.5):
        kmeans = KMeans(n_clusters=n_nodes).fit(X)

        affinity = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(i):
                segment_ij = kmeans.cluster_centers_[j] - kmeans.cluster_centers_[i]
                segment_ji = -segment_ij
                segment_norm2 = np.linalg.norm(segment_ij)**2

                projs_i = np.dot(X[kmeans.labels_ == i] - kmeans.cluster_centers_[i], segment_ij)
                score_i = projs_i[projs_i > 0].sum()
                projs_j = np.dot(X[kmeans.labels_ == j] - kmeans.cluster_centers_[j], segment_ji)
                score_j = projs_j[projs_j > 0].sum()

                affinity[i, j] = np.power((score_i + score_j) / (projs_i.shape[0] + projs_j.shape[0]) / segment_norm2, alpha)

        affinity += affinity.T

        q1 = np.quantile(affinity, .25)
        q3 = np.quantile(affinity, .75)

        gamma = np.log(M) / (q3 - q1)
        affinity = np.exp(gamma * affinity)

        labels = SpectralClustering(n_clusters=self.n_classes, affinity='precomputed').fit_predict(affinity)
        self.clusters = [kmeans.cluster_centers_[labels == i] for i in range(self.n_classes)]

    def predict(self, x):
        diffs = [x - cluster[:, np.newaxis] for cluster in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        return np.argmin(min_dists, axis=0)
