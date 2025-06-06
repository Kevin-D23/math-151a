import numpy as np
from typing import Tuple


def ldlt_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array(A, dtype=float)
    n = A.shape[0]

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


def forward_substitution_ldlt(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y


def diagonal_substitution(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    z = np.zeros(n)

    for i in range(n):
        if abs(D[i, i]) < 1e-12:
            raise ValueError(f"Zero diagonal element at position ({i}, {i})")
        z[i] = y[i] / D[i, i]

    return z


def backward_substitution_ldlt(L: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = z[i] - np.dot(L.T[i, i + 1 :], x[i + 1 :])

    return x


def solve_linear_system_ldlt(
    A: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Perform LDL^T factorization
    L, D = ldlt_factorization(A)

    # Step 1: Solve Ly = b
    y = forward_substitution_ldlt(L, b)

    # Step 2: Solve Dz = y
    z = diagonal_substitution(D, y)

    # Step 3: Solve L^T x = z
    x = backward_substitution_ldlt(L, z)

    return x, L, D


def print_system_solution_ldlt(system_name: str, A: np.ndarray, b: np.ndarray):
    print(f"\n{'='*70}")
    print(f"SOLVING SYSTEM {system_name} USING LDL^T FACTORIZATION")
    print(f"{'='*70}")

    print(f"\nCoefficient Matrix A:")
    print(A)
    print(f"\nRight-hand side vector b:")
    print(b)

    # Check if matrix is symmetric
    if not np.allclose(A, A.T, atol=1e-10):
        print(f"\nWARNING: Matrix is not symmetric!")
        print(f"LDL^T factorization is designed for symmetric matrices.")
        print(f"Proceeding anyway, but results may not be optimal.")

    try:
        x, L, D = solve_linear_system_ldlt(A, b)

        print(f"\nLDL^T Factorization Results:")
        print(f"L Matrix (Lower triangular with 1's on diagonal):")
        print(L)
        print(f"\nD Matrix (Diagonal):")
        print(D)

        print(f"\nSolution Steps:")
        print(f"1. Solve Ly = b:")
        y = forward_substitution_ldlt(L, b)
        print(f"   y = {y}")

        print(f"2. Solve Dz = y:")
        z = diagonal_substitution(D, y)
        print(f"   z = {z}")

        print(f"3. Solve L^T x = z:")
        print(f"   x = {x}")

        print(f"\nFinal Solution:")
        for i, val in enumerate(x):
            print(f"x{i+1} = {val:.6f}")

        # Verification
        verification = A @ x
        print(f"\nVerification (A * x):")
        print(verification)
        print(f"Original b:")
        print(b)
        residual = np.linalg.norm(verification - b)
        print(f"Residual error: {residual:.2e}")

        return x, L, D

    except Exception as e:
        print(f"Error solving system: {e}")
        return None, None, None


# Exercise 8 Systems
print("MODIFIED LDL^T ALGORITHM FOR SOLVING LINEAR SYSTEMS")
print("Exercise 8: Using modified algorithm from Exercise 7")
print("=" * 70)

# System 8a: 4x1 - x2 + x3 = -1, -x1 + 3x2 = 4, x1 + 2x3 = 5
A_8a = np.array([[4, -1, 1], [-1, 3, 0], [1, 0, 2]])
b_8a = np.array([-1, 4, 5])

# System 8b: 4x1 + 2x2 + 2x3 = 0, 2x1 + 6x2 + 2x3 = 1, 2x1 + 2x2 + 5x3 = 0
A_8b = np.array([[4, 2, 2], [2, 6, 2], [2, 2, 5]])
b_8b = np.array([0, 1, 0])

# System 8c: 4x1 + 2x3 + x4 = -2, 3x2 - x3 + x4 = 0, 2x1 - x2 + 6x3 + 3x4 = 7, x1 + x2 + 3x3 + 8x4 = -2
A_8c = np.array([[4, 0, 2, 1], [0, 3, -1, 1], [2, -1, 6, 3], [1, 1, 3, 8]])
b_8c = np.array([-2, 0, 7, -2])

# System 8d: 4x1 + x2 + x3 + x4 = 2, x1 + 3x2 - x4 = 2, x1 + 2x3 + x4 = 1, x1 - x2 + x3 + 4x4 = 1
A_8d = np.array([[4, 1, 1, 1], [1, 3, 0, -1], [1, 0, 2, 1], [1, -1, 1, 4]])
b_8d = np.array([2, 2, 1, 1])

# Solve all systems
systems = [
    ("8A", A_8a, b_8a),
    ("8B", A_8b, b_8b),
    ("8C", A_8c, b_8c),
    ("8D", A_8d, b_8d),
]

results = {}
for name, A, b in systems:
    x, L, D = print_system_solution_ldlt(name, A, b)
    results[name] = (x, L, D)
