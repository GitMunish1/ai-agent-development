import random
from vector import Vector
from matric import Matrix

# Fix the random seed so the weights are reproducible on every run.
random.seed(42)

# Create a 2x3 weight matrix for a neural network layer.
# Each weight is sampled from a normal distribution with mean 0 and std 0.1.
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])

# Define a 3-dimensional input vector.
input_vector = Vector([1.0, 0.5, -0.3])

# Perform the layer computation: matrix-vector multiplication.
# This produces a 2-dimensional output.
output = weights @ input_vector

print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")