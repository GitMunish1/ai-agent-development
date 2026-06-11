import numpy as np  # import numpy for math with arrays

a = np.array([1, 2, 3], dtype=float)  # first vector
b = np.array([4, 5, 6], dtype=float)  # second vector

# Add the two vectors element by element.
print(f"a + b = {a + b}")

# Compute the dot product of a and b.
print(f"a · b = {np.dot(a, b)}")

# Find the length of vector a.
print(f"|a| = {np.linalg.norm(a):.4f}")

# Compute cosine similarity between a and b.
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

# Create a random 2x3 matrix and multiply it by a 3-element vector.
W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")