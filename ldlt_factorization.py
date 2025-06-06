import numpy as np
from typing import Tuple


def ldlt_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    # Initialize L and D matrices
    L = np.eye(n)  # Identity matrix (1's on diagonal)
    D = np.zeros((n, n))

    # Perform LDL^T factorization
    for i in range(n):
        # Calculate D[i,i]
        sum_ld = sum(L[i, k] * D[k, k] * L[i, k] for k in range(i))
        D[i, i] = A[i, i] - sum_ld

        if abs(D[i, i]) < 1e-12:
            raise ValueError(
                f"Zero or near-zero diagonal element at position ({i}, {i})"
            )

        # Calculate L[j,i] for j > i
        for j in range(i + 1, n):
            sum_ld = sum(L[j, k] * D[k, k] * L[i, k] for k in range(i))
            L[j, i] = (A[j, i] - sum_ld) / D[i, i]

    return L, D


def verify_ldlt_factorization(
    A: np.ndarray, L: np.ndarray, D: np.ndarray, tolerance=1e-10
) -> bool:
    """
    Verify that L * D * L^T equals the original matrix A within tolerance.
    """
    reconstructed = L @ D @ L.T
    difference = np.abs(A - reconstructed)
    return np.all(difference < tolerance)


def print_ldlt_solution(matrix_name: str, A: np.ndarray):
    """
    Print the LDL^T factorization solution with detailed output.
    """
    print(f"\n{'='*70}")
    print(f"LDL^T FACTORIZATION - MATRIX {matrix_name}")
    print(f"{'='*70}")

    print(f"\nOriginal Matrix A:")
    print(A)

    # Check if matrix is symmetric
    if not np.allclose(A, A.T, atol=1e-10):
        print(f"\nWARNING: Matrix is not symmetric!")
        print(f"A^T =")
        print(A.T)
        print(f"Proceeding with factorization anyway...")

    try:
        L, D = ldlt_factorization(A)

        print(f"\nL Matrix (Lower triangular with 1's on diagonal):")
        print(L)

        print(f"\nD Matrix (Diagonal):")
        print(D)

        print(f"\nL^T Matrix (Upper triangular):")
        print(L.T)

        # Verification
        reconstructed = L @ D @ L.T
        print(f"\nVerification: L * D * L^T =")
        print(reconstructed)

        is_correct = verify_ldlt_factorization(A, L, D)
        print(f"\nFactorization verification: {'PASSED' if is_correct else 'FAILED'}")

        if is_correct:
            residual = np.linalg.norm(A - reconstructed)
            print(f"Residual error: {residual:.2e}")

        # Extract diagonal elements of D
        print(f"\nDiagonal elements of D:")
        for i in range(len(D)):
            print(f"d_{i+1} = {D[i, i]:.6f}")

        return L, D

    except Exception as e:
        print(f"Error in factorization: {e}")
        return None, None


# Define the matrices from the problem
print("LDL^T FACTORIZATION ALGORITHM")
print("=" * 70)

# Matrix a
A_a = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

# Matrix b
A_b = np.array([[4, 1, 1, 1], [1, 3, -1, 1], [1, -1, 2, 0], [1, 1, 0, 2]])

# Matrix c
A_c = np.array([[4, 1, -1, 0], [1, 3, -1, 0], [-1, -1, 5, 2], [0, 0, 2, 4]])

# Matrix d
A_d = np.array([[6, 2, 1, -1], [2, 4, 1, 0], [1, 1, 4, -1], [-1, 0, -1, 3]])

# Solve all matrices
matrices = [("A", A_a), ("B", A_b), ("C", A_c), ("D", A_d)]

results = {}
for name, matrix in matrices:
    L, D = print_ldlt_solution(name, matrix)
    results[name] = (L, D)

