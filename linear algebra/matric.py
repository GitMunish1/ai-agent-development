from vector import Vector


class Matrix:
    def __init__(self, rows):
        # Store matrix rows as a list of lists.
        self.rows = [list(row) for row in rows]
        # Keep the shape as (number of rows, number of columns).
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        # If we multiply by a vector, return a new vector.
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        # Otherwise assume we are multiplying by another matrix.
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        # Flip rows and columns.
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        # Print the matrix in a readable form.
        return f"Matrix({self.rows})"


# A simple 90-degree rotation matrix.
rotation_90 = Matrix([[0, -1], [1, 0]])
# A point in 2D space.
point = Vector([3, 1])

# Rotate the point by 90 degrees.
rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")