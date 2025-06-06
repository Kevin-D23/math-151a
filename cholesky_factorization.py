import numpy as np
from typing import Tuple, Optional
import math


def cholesky_factorization(A: np.ndarray) -> np.ndarray:
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Initialize L matrix
    L = np.zeros((n, n))

    # Perform Cholesky factorization
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Diagonal elements
                sum_squares = sum(L[i, k] ** 2 for k in range(j))
                diagonal_value = A[i, i] - sum_squares

                L[i, j] = math.sqrt(diagonal_value)
            else:  # Off-diagonal elements
                sum_products = sum(L[i, k] * L[j, k] for k in range(j))
                L[i, j] = (A[i, j] - sum_products) / L[j, j]

    return L


def verify_cholesky_factorization(
    A: np.ndarray, L: np.ndarray, tolerance=1e-10
) -> bool:
    reconstructed = L @ L.T
    difference = np.abs(A - reconstructed)
    return np.all(difference < tolerance)


def print_cholesky_solution(matrix_name: str, A: np.ndarray):
    print(f"\n{'='*70}")
    print(f"CHOLESKY FACTORIZATION - MATRIX {matrix_name}")
    print(f"{'='*70}")

    print(f"\nOriginal Matrix A:")
    print(A)

    # Check if matrix is symmetric
    if not np.allclose(A, A.T, atol=1e-10):
        print(f"\nWARNING: Matrix is not symmetric!")
        print(f"A^T =")
        print(A.T)
        print(f"Cholesky factorization requires symmetric matrices.")
        return None

    # Check if matrix is positive definite by examining eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    print(f"\nEigenvalues: {eigenvalues}")

    if np.all(eigenvalues > 1e-12):
        print("Matrix appears to be positive definite (all eigenvalues > 0)")
    else:
        print("WARNING: Matrix may not be positive definite (some eigenvalues ≤ 0)")
        print("Cholesky factorization may fail.")

    try:
        L = cholesky_factorization(A)

        print(f"\nL Matrix (Lower triangular):")
        print(L)

        print(f"\nL^T Matrix (Upper triangular):")
        print(L.T)

        # Verification
        reconstructed = L @ L.T
        print(f"\nVerification: L * L^T =")
        print(reconstructed)

        is_correct = verify_cholesky_factorization(A, L)
        print(f"\nFactorization verification: {'PASSED' if is_correct else 'FAILED'}")

        if is_correct:
            residual = np.linalg.norm(A - reconstructed)
            print(f"Residual error: {residual:.2e}")

        # Show diagonal elements of L
        print(f"\nDiagonal elements of L:")
        for i in range(len(L)):
            print(f"L[{i+1},{i+1}] = {L[i, i]:.6f}")

        return L

    except Exception as e:
        print(f"Error in Cholesky factorization: {e}")
        print("This confirms the matrix is not positive definite.")
        return None


def compare_with_numpy_cholesky(A: np.ndarray, matrix_name: str):
    """
    Compare our implementation with NumPy's Cholesky factorization.
    """
    try:
        L_numpy = np.linalg.cholesky(A)
        print(f"\nNumPy Cholesky result for Matrix {matrix_name}:")
        print(L_numpy)

        # Our implementation
        try:
            L_ours = cholesky_factorization(A)
            print(f"\nOur implementation result:")
            print(L_ours)

            # Compare
            if np.allclose(L_numpy, L_ours, atol=1e-10):
                print("✓ Results match NumPy implementation!")
            else:
                print("⚠ Results differ from NumPy implementation")
                print(f"Difference: {np.max(np.abs(L_numpy - L_ours)):.2e}")
        except:
            print(
                "Our implementation failed (as expected for non-positive definite matrices)"
            )

    except np.linalg.LinAlgError:
        print(f"\nNumPy Cholesky also failed for Matrix {matrix_name}")
        print("This confirms the matrix is not positive definite.")


# Define the matrices from Exercise 4
print("CHOLESKY FACTORIZATION ALGORITHM")
print("Solving Exercise 6: Apply Cholesky to matrices from Exercise 4")
print("=" * 70)

# Matrix a from Exercise 4
A_a = np.array([[4, -1, 1], [-1, 3, 0], [1, 0, 2]])

# Matrix b from Exercise 4
A_b = np.array([[4, 2, 2], [2, 6, 2], [2, 2, 5]])

# Matrix c from Exercise 4
A_c = np.array([[4, 0, 2, 1], [0, 3, -1, 1], [2, -1, 6, 3], [1, 1, 3, 8]])

# Matrix d from Exercise 4
A_d = np.array([[4, 1, 1, 1], [1, 3, 0, -1], [1, 0, 2, 1], [1, -1, 1, 4]])

# Solve all matrices
matrices = [("A", A_a), ("B", A_b), ("C", A_c), ("D", A_d)]

results = {}
for name, matrix in matrices:
    L = print_cholesky_solution(name, matrix)
    results[name] = L

    # Compare with NumPy implementation
    compare_with_numpy_cholesky(matrix, name)
