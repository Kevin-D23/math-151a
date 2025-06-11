import numpy as np


def nevilles_method(x_points, y_points, x):
    n = len(x_points)
    Q = np.zeros((n, n))
    Q[:, 0] = y_points
    print(f"Initial Q table:")
    print(Q)
    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = (
                (x - x_points[i + j]) * Q[i, j - 1]
                + (x_points[i] - x) * Q[i + 1, j - 1]
            ) / (x_points[i] - x_points[i + j])
        print(f"After computing column {j}:")
        print(Q)
    return Q[0, n - 1]


def inverse_nevilles_method(x_points, y_points, y_target=0):
    # Interpolate x as a function of y
    n = len(y_points)
    Q = np.zeros((n, n))
    Q[:, 0] = x_points
    print(f"Initial Q table (inverse):")
    print(Q)
    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = (
                (y_target - y_points[i + j]) * Q[i, j - 1]
                + (y_points[i] - y_target) * Q[i + 1, j - 1]
            ) / (y_points[i] - y_points[i + j])
        print(f"After computing column {j} (inverse):")
        print(Q)
    return Q[0, n - 1]


if __name__ == "__main__":
    x_points = [0.25, 0.5, 1, 1.25]
    y_points = [25.2, 49.2, 96.4, 119.4]
    x = 0.75
    print(f"Neville's method interpolation for x = {x}:")
    result = nevilles_method(x_points, y_points, x)
    print(f"Interpolated value at x = {x}: {result}")

