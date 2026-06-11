using LinearAlgebra  # needed for the dot product operator and vector norm

a = [1.0, 2.0, 3.0]   # first vector
b = [4.0, 5.0, 6.0]   # second vector

# Add the vectors element-wise and print the result.
println("a + b = ", a + b)

# Dot product multiplies matching entries and adds them.
println("a · b = ", a ⋅ b)

# Compute the length (magnitude) of vector a.
println("|a| = ", √(a ⋅ a))

# Cosine measures how similar the two vectors are.
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication example.
# W is a 2x3 matrix and x is a 3-element vector.
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]

# Multiply W by x to get a 2-element output vector.
println("Wx = ", W * x)
println("This is a neural network layer.")