import numpy as np
from typing import Tuple, Optional
import warnings


def modified_ldlt_factorization(
    A: np.ndarray, pivot_threshold=1e-12
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, str]:
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        return None, None, False, "Matrix must be square"

    # Check if matrix is symmetric
    if not np.allclose(A, A.T, atol=1e-10):
        return None, None, False, "Matrix is not symmetric"

    # Initialize L and D matrices
    L = np.eye(n)  # Identity matrix (1's on diagonal)
    D = np.zeros((n, n))

    try:
        # Perform LDL^T factorization with pivot checking
        for i in range(n):
            # Calculate D[i,i]
            sum_ld = sum(L[i, k] * D[k, k] * L[i, k] for k in range(i))
            D[i, i] = A[i, i] - sum_ld

            # Check if pivot is too small (near zero)
            if abs(D[i, i]) < pivot_threshold:
                return (
                    None,
                    None,
                    False,
                    f"Zero or near-zero pivot encountered at position ({i+1}, {i+1}). D[{i+1},{i+1}] = {D[i, i]:.2e}",
                )

            # Calculate L[j,i] for j > i
            for j in range(i + 1, n):
                sum_ld = sum(L[j, k] * D[k, k] * L[i, k] for k in range(i))
                L[j, i] = (A[j, i] - sum_ld) / D[i, i]

        return L, D, True, "Factorization successful"

    except Exception as e:
        return None, None, False, f"Factorization failed: {str(e)}"


def analyze_matrix_properties(A: np.ndarray) -> dict:
    properties = {}

    # Basic properties
    properties["size"] = A.shape[0]
    properties["is_symmetric"] = np.allclose(A, A.T, atol=1e-10)

    # Eigenvalue analysis
    try:
        eigenvalues = np.linalg.eigvals(A)
        properties["eigenvalues"] = eigenvalues
        properties["min_eigenvalue"] = np.min(eigenvalues)
        properties["max_eigenvalue"] = np.max(eigenvalues)
        properties["condition_number"] = (
            np.max(eigenvalues) / np.min(eigenvalues)
            if np.min(eigenvalues) != 0
            else np.inf
        )

        # Definiteness
        if np.all(eigenvalues > 1e-12):
            properties["definiteness"] = "Positive Definite"
        elif np.all(eigenvalues < -1e-12):
            properties["definiteness"] = "Negative Definite"
        elif np.all(np.abs(eigenvalues) > 1e-12):
            properties["definiteness"] = "Indefinite"
        else:
            properties["definiteness"] = "Singular (has zero eigenvalues)"

    except Exception as e:
        properties["eigenvalue_error"] = str(e)

    # Determinant
    try:
        properties["determinant"] = np.linalg.det(A)
    except Exception as e:
        properties["determinant_error"] = str(e)

    # Leading principal minors (for Sylvester's criterion)
    properties["leading_principal_minors"] = []
    try:
        for k in range(1, A.shape[0] + 1):
            minor = np.linalg.det(A[:k, :k])
            properties["leading_principal_minors"].append(minor)
    except Exception as e:
        properties["minor_error"] = str(e)

    return properties


def print_matrix_analysis(matrix_name: str, A: np.ndarray):
    print(f"\n{'='*80}")
    print(f"ANALYSIS OF MATRIX {matrix_name}")
    print(f"{'='*80}")

    print(f"\nOriginal Matrix A:")
    print(A)

    # Analyze matrix properties
    props = analyze_matrix_properties(A)

    print(f"\nMatrix Properties:")
    print(f"  Size: {props['size']}×{props['size']}")
    print(f"  Symmetric: {props['is_symmetric']}")

    if "eigenvalues" in props:
        print(f"  Eigenvalues: {props['eigenvalues']}")
        print(f"  Definiteness: {props['definiteness']}")
        print(f"  Condition number: {props['condition_number']:.2e}")

    if "determinant" in props:
        print(f"  Determinant: {props['determinant']:.6f}")

    if "leading_principal_minors" in props:
        print(f"  Leading principal minors: {props['leading_principal_minors']}")

    # Attempt LDL^T factorization
    print(f"\nAttempting LDL^T Factorization:")
    L, D, success, message = modified_ldlt_factorization(A)

    if success:
        print(f"✓ SUCCESS: {message}")
        print(f"\nL Matrix (Lower triangular with 1's on diagonal):")
        print(L)
        print(f"\nD Matrix (Diagonal):")
        print(D)

        # Verification
        reconstructed = L @ D @ L.T
        print(f"\nVerification: L * D * L^T =")
        print(reconstructed)

        residual = np.linalg.norm(A - reconstructed)
        print(f"Reconstruction error: {residual:.2e}")

        # Analyze diagonal elements
        print(f"\nDiagonal elements of D:")
        for i in range(len(D)):
            print(f"  d_{i+1} = {D[i, i]:.6f}")

        # Check signs for definiteness
        diagonal_elements = [D[i, i] for i in range(len(D))]
        if all(d > 0 for d in diagonal_elements):
            d_definiteness = "Positive Definite (all d_i > 0)"
        elif all(d < 0 for d in diagonal_elements):
            d_definiteness = "Negative Definite (all d_i < 0)"
        else:
            d_definiteness = "Indefinite (mixed signs in D)"

        print(f"  Definiteness from D: {d_definiteness}")

    else:
        print(f"✗ FAILURE: {message}")
        print(f"  This matrix cannot be factorized using standard LDL^T")

        # Suggest reasons for failure
        if "Singular" in props.get("definiteness", ""):
            print(f"  Reason: Matrix is singular (determinant ≈ 0)")
        elif not props.get("is_symmetric", True):
            print(f"  Reason: Matrix is not symmetric")
        else:
            print(f"  Reason: Numerical issues or ill-conditioning")

    return L, D, success, props


# Exercise 14 Matrices
print("MODIFIED LDL^T FACTORIZATION ALGORITHM")
print("Exercise 14: Handling matrices where factorization may not be possible")
print("=" * 80)

# Matrix a
A_14a = np.array([[3, -3, 6], [-3, 2, -7], [6, -7, 13]])

# Matrix b
A_14b = np.array([[3, -6, 9], [-6, 14, -20], [9, -20, 29]])

# Matrix c
A_14c = np.array([[-1, 2, 0, 1], [2, -3, 2, -1], [0, 2, 5, 6], [1, -1, 6, 12]])

# Matrix d
A_14d = np.array([[2, -2, 4, -4], [-2, 3, -4, 5], [4, -4, 10, -10], [-4, 5, -10, 14]])

# Analyze all matrices
matrices = [("A", A_14a), ("B", A_14b), ("C", A_14c), ("D", A_14d)]

results = {}
for name, matrix in matrices:
    L, D, success, props = print_matrix_analysis(name, matrix)
    results[name] = (L, D, success, props)
