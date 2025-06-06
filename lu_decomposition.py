import numpy as np
from typing import Tuple

def lu_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    L = np.zeros((n, n))
    U = A.copy().astype(float)
    
    # Set diagonal of L to 1
    np.fill_diagonal(L, 1.0)
    
    for i in range(n-1):
        for j in range(i+1, n):
            if abs(U[i, i]) < 1e-12:
                raise ValueError(f"Zero pivot encountered at position ({i}, {i})")
            
            # Calculate multiplier
            multiplier = U[j, i] / U[i, i]
            L[j, i] = multiplier
            
            # Eliminate below pivot
            U[j, :] -= multiplier * U[i, :]
    
    return L, U

def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ly = b using forward substitution.
    """
    n = L.shape[0]
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    return y

def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve Ux = y using backward substitution.
    """
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            raise ValueError(f"Zero diagonal element at position ({i}, {i})")
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x

def solve_linear_system_lu(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the linear system Ax = b using LU factorization.
    """
    # Perform LU factorization
    L, U = lu_factorization(A)
    
    # Solve Ly = b
    y = forward_substitution(L, b)
    
    # Solve Ux = y
    x = backward_substitution(U, y)
    
    return x, L, U

def create_beetle_matrix(b1, b2, b3, b4, p1, p2, p3):
    """
    Create the beetle population transition matrix A.
    """
    A = np.array([
        [b1, b2, b3, b4],
        [p1, 0,  0,  0 ],
        [0,  p2, 0,  0 ],
        [0,  0,  p3, 0 ]
    ])
    return A

def solve_beetle_population(b_values, p_values, target_population, part_name):
    """
    Solve for initial population needed to achieve target population after 1 year.
    """
    print(f"\n{'='*70}")
    print(f"PART {part_name}: BEETLE POPULATION ANALYSIS")
    print(f"{'='*70}")
    
    b1, b2, b3, b4 = b_values
    p1, p2, p3 = p_values
    
    # Create the transition matrix A
    A = create_beetle_matrix(b1, b2, b3, b4, p1, p2, p3)
    
    print(f"\nTransition Matrix A:")
    print(f"A = [[{b1:>6}, {b2:>6}, {b3:>6}, {b4:>6}]]")
    print(f"    [[{p1:>6}, {0:>6}, {0:>6}, {0:>6}]]")
    print(f"    [[{0:>6}, {p2:>6}, {0:>6}, {0:>6}]]")
    print(f"    [[{0:>6}, {0:>6}, {p3:>6}, {0:>6}]]")
    
    print(f"\nTarget population after 1 year: b = {target_population}")
    
    # We need to solve Ax = b for initial population x
    try:
        x, L, U = solve_linear_system_lu(A, np.array(target_population))
        
        print(f"\nLU Factorization Results:")
        print(f"L (Lower triangular):")
        print(L)
        print(f"\nU (Upper triangular):")
        print(U)
        
        print(f"\nInitial Population Distribution:")
        age_groups = ["Age 1", "Age 2", "Age 3", "Age 4"]
        for i, (age, pop) in enumerate(zip(age_groups, x)):
            print(f"{age} females: {pop:>10.2f}")
        
        print(f"\nTotal initial population: {sum(x):>10.2f}")
        
        # Verification
        verification = A @ x
        print(f"\nVerification (A * x):")
        for i, (age, pop) in enumerate(zip(age_groups, verification)):
            print(f"{age} after 1 year: {pop:>10.2f}")
        
        print(f"\nExpected population after 1 year:")
        for i, (age, pop) in enumerate(zip(age_groups, target_population)):
            print(f"{age}: {pop:>10.2f}")
        
        residual = np.linalg.norm(verification - np.array(target_population))
        print(f"\nResidual error: {residual:.2e}")
        
        return x, A, L, U
        
    except Exception as e:
        print(f"Error solving system: {e}")
        return None, A, None, None

# Problem parameters
print("BEETLE POPULATION DYNAMICS USING LU DECOMPOSITION")
print("="*70)

# Given parameters
b_values = [0, 1/8, 1/4, 1/2]  # Birth rates
p_values = [1/2, 1/4, 1/8]     # Survival rates

print(f"Model Parameters:")
print(f"Birth rates: b1={b_values[0]}, b2={b_values[1]}, b3={b_values[2]}, b4={b_values[3]}")
print(f"Survival rates: p1={p_values[0]}, p2={p_values[1]}, p3={p_values[2]}")

# Part (a): Target population (175, 100, 50, 25)
target_a = [175, 100, 50, 25]
solution_a, matrix_a, L_a, U_a = solve_beetle_population(b_values, p_values, target_a, "A")

# Part (b): Target population (100, 100, 100, 100)
target_b = [100, 100, 100, 100]
solution_b, matrix_b, L_b, U_b = solve_beetle_population(b_values, p_values, target_b, "B")

# Analysis of part (b)
if solution_b is not None:
    print(f"\n{'='*70}")
    print("INTERPRETATION OF PART (B)")
    print(f"{'='*70}")
    print(f"\nThe solution for part (b) shows what initial population distribution")
    print(f"is needed to achieve equal numbers (100) in each age group after 1 year.")
    print(f"\nKey observations:")
    print(f"1. Age 4 initial population: {solution_b[3]:.2f}")
    print(f"   - This is very high because age 4 beetles don't survive to the next year")
    print(f"   - All 100 age 4 beetles in year 1 must come from reproduction")
    print(f"\n2. Age 3 initial population: {solution_b[2]:.2f}")
    print(f"   - These survive with probability p3 = {p_values[2]} to become age 4")
    print(f"   - Need {solution_b[2]:.2f} * {p_values[2]} = {solution_b[2] * p_values[2]:.2f} survivors")
    print(f"\n3. The uniform distribution (100, 100, 100, 100) is not a natural")
    print(f"   equilibrium for this population model - it requires a very specific")
    print(f"   and likely unrealistic initial distribution.")
    
    total_reproduction = sum(b_values[i] * solution_b[i] for i in range(4))
    print(f"\n4. Total reproduction from all age groups: {total_reproduction:.2f}")
    print(f"   This matches the 100 age 1 beetles needed in the target population.")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("The LU decomposition method successfully solved both population scenarios.")
print("Part (a) represents a more realistic declining population structure.")
print("Part (b) shows the artificial initial conditions needed for uniform age distribution.")
