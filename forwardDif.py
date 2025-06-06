import numpy as np


def forward_difference(x_values, y_values):
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    h = x_values[1] - x_values[0]  # Assuming uniform spacing
    approximations = np.zeros(n - 1)
    for i in range(n - 1):
        approximations[i] = (y_values[i + 1] - y_values[i]) / h
    print("Forward Difference Approximations:")
    for i in range(n - 1):
        print(f"f'({x_values[i]}) ≈ {approximations[i]}")
    return approximations


def backward_difference(x_values, y_values):
    n = len(x_values)
    h = x_values[1] - x_values[0]  # Assuming uniform spacing
    return (y_values[-1] - y_values[-2]) / h


if __name__ == "__main__":
    # Example usage for Exercise 10b
    x_values = [-3.0, -2.8, -2.6, -2.4, -2.2, -2.0]
    y_values = [16.08554, 12.64465, 9.863738, 7.623176, 5.825013, 4.389056]
    print("Computing forward differences for x values:", x_values)
    print("Computing forward differences for y values:", y_values)
    approximations = forward_difference(x_values, y_values)
    print("Forward difference approximations:", approximations)
    # Backward difference for the last value
    last_backward = backward_difference(x_values, y_values)
    print(f"Backward difference approximation at x = {x_values[-1]}: {last_backward}")
