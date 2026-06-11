class Vector:
    def __init__(self, components):
        # Save the list of values in the vector.
        self.components = list(components)
        # Keep the vector dimension.
        self.dim = len(self.components)

    def __add__(self, other):
        # Add two vectors element by element.
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        # Subtract two vectors element by element.
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        # Multiply matching entries and add them.
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        # Return the length of the vector.
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        # Make the vector length 1, keeping the direction the same.
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        # Measure how close two vectors are in direction.
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        # Show the vector contents nicely when printing.
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")