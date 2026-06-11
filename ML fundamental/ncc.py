import numpy as np
class NearestCentroid:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = np.array([
            X[y == c].mean(axis=0) for c in self.classes])

    def predict(self, X):
        distances = np.array([np.sqrt(((X - c) ** 2).sum(axis=1))
            for c in self.centroids])
        return self.classes[distances.argmin(axis=0)]