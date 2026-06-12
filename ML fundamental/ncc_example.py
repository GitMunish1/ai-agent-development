import numpy as np

# Create a reproducible random number generator using a fixed seed (42).
# Using a fixed seed makes experiments deterministic so results can be
# reproduced later.
rng = np.random.RandomState(42)

# Generate 100 samples for class 0. `rng.randn(100, 2)` draws from a
# standard normal distribution (mean=0, std=1) and returns an array of shape
# (100, 2). Adding `np.array([1.0, 1.0])` shifts the cluster center to (1, 1).
X_class0 = rng.randn(100, 2) + np.array([1.0, 1.0])

# Generate 100 samples for class 1, shifted to center (-1, -1) so the two
# classes are separable on average.
X_class1 = rng.randn(100, 2) + np.array([-1.0, -1.0])

# Stack the two class arrays vertically to form a single dataset `X` with
# shape (200, 2): first 100 rows are class 0, next 100 rows are class 1.
X = np.vstack([X_class0, X_class1])

# Create the label vector `y` with 100 zeros (class 0) followed by 100 ones
# (class 1). The length of `y` matches the number of rows in `X`.
y = np.array([0] * 100 + [1] * 100)

# show shapes
print(X.shape, y.shape)

# or run the classifier from ncc.py and print accuracy
from ncc import NearestCentroid
clf = NearestCentroid()
clf.fit(X, y)
print("Accuracy:", (clf.predict(X) == y).mean())