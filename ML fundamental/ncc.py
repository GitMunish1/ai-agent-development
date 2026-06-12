import numpy as np

# Simple nearest-centroid classifier implementation.
class NearestCentroid:
    # Compute class centroids from training data X and labels y.
    def fit(self, X, y):
        # `np.unique(y)` returns sorted unique class labels from the label vector `y`.
        self.classes = np.unique(y)
        # For each class label `c`, select rows `X[y == c]` and compute the mean
        # across samples (axis=0) to obtain the centroid coordinates for that class.
        # The list comprehension builds a list of centroids which we convert to a
        # 2D numpy array where each row corresponds to one class centroid.
        self.centroids = np.array([
            X[y == c].mean(axis=0) for c in self.classes])

    # Predict the class label for each row (sample) in X.
    def predict(self, X):
        # Compute the Euclidean distance between every sample in X and each
        # centroid. For each centroid `c` we compute `(X - c) ** 2` which
        # broadcasts `c` across all rows of X, then sum squared differences
        # across features (axis=1) and take the square root to get distances.
        distances = np.array([np.sqrt(((X - c) ** 2).sum(axis=1))
            for c in self.centroids])
        # `distances` has shape (n_centroids, n_samples). `argmin(axis=0)` picks
        # the index of the nearest centroid for each sample. We then map those
        # centroid indices back to the original class labels stored in
        # `self.classes` and return the predicted label vector.
        return self.classes[distances.argmin(axis=0)]


if __name__ == "__main__":
    # Create a random number generator with a fixed seed for reproducibility.
    # Using `RandomState(42)` ensures the same random numbers each run.
    rng = np.random.RandomState(42)

    # Generate 100 samples (rows) with 2 features (columns) for class 0.
    # `rng.randn(100, 2)` draws from a standard normal (mean 0, std 1).
    # Adding `np.array([1.0, 1.0])` shifts the cluster center to (1, 1).
    X_class0 = rng.randn(100, 2) + np.array([1.0, 1.0])

    # Generate 100 samples for class 1, shifted to center (-1, -1).
    X_class1 = rng.randn(100, 2) + np.array([-1.0, -1.0])

    # Stack the two class arrays vertically to produce a dataset of shape
    # (200, 2) where the first 100 rows belong to class 0 and the next 100
    # rows belong to class 1.
    X = np.vstack([X_class0, X_class1])

    # Construct the label array `y` with 100 zeros (class 0) followed by 100
    # ones (class 1). The length of `y` matches the number of rows in `X`.
    y = np.array([0] * 100 + [1] * 100)

    # Instantiate the classifier, fit it to the generated data, and predict on
    # the same data to do a simple sanity-check of training correctness.
    clf = NearestCentroid()
    clf.fit(X, y)
    preds = clf.predict(X)

    # Compute and print the fraction of correctly predicted labels (accuracy).
    # `(preds == y)` produces a boolean array, `.mean()` casts to float and
    # returns the proportion of True entries.
    print("Accuracy:", (preds == y).mean())
