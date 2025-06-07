import numpy as np
import math


def gaussian_quadrature_n2(f, a, b):
    x1, x2 = -1 / math.sqrt(3), 1 / math.sqrt(3)

    # Weights
    w1, w2 = 1.0, 1.0

    t1, t2 = x1, x2  # nodes on [-1,1]
    x1_transformed = ((b - a) * t1 + (b + a)) / 2
    x2_transformed = ((b - a) * t2 + (b + a)) / 2

    # Gaussian quadrature formula
    integral = (b - a) / 2 * (w1 * f(x1_transformed) + w2 * f(x2_transformed))

    return integral


def exact_integral_a():
    a, b = 0, math.pi / 4

    def antiderivative(x):
        return math.exp(3 * x) * (3 * math.sin(2 * x) - 2 * math.cos(2 * x)) / 13

    return antiderivative(b) - antiderivative(a)


def exact_integral_b():
    a, b = 1, 1.6

    def antiderivative(x):
        return math.log(abs(x**2 - 4))

    return antiderivative(b) - antiderivative(a)


def exact_integral_c():
    a, b = 3, 3.5

    def antiderivative(x):
        return math.sqrt(x**2 - 4)

    return antiderivative(b) - antiderivative(a)


def exact_integral_d():
    a, b = 0, math.pi / 4

    def antiderivative(x):
        return x / 2 + math.sin(2 * x) / 4

    return antiderivative(b) - antiderivative(a)


# Define the integrand functions
def f_a(x):
    return math.exp(3 * x) * math.sin(2 * x)


def f_b(x):
    return 2 * x / (x**2 - 4)


def f_c(x):
    return x / math.sqrt(x**2 - 4)


def f_d(x):
    return math.cos(x) ** 2


def solve_problem():
    print("GAUSSIAN QUADRATURE WITH n=2")
    print("=" * 60)
    print("Approximating integrals using 2-point Gauss-Legendre quadrature")
    print("=" * 60)

    # Problem a: ∫[0 to π/4] e^(3x) * sin(2x) dx
    print("\na. ∫[0 to π/4] e^(3x) * sin(2x) dx")
    print("-" * 40)

    a_limits = (0, math.pi / 4)
    approx_a = gaussian_quadrature_n2(f_a, *a_limits)
    exact_a = exact_integral_a()
    error_a = abs(exact_a - approx_a)

    print(f"   Gaussian quadrature (n=2): {approx_a:.8f}")
    print(f"   Exact value:               {exact_a:.8f}")
    print(f"   Absolute error:            {error_a:.2e}")
    print(f"   Relative error:            {error_a/abs(exact_a)*100:.4f}%")

    # Problem b: ∫[1 to 1.6] 2x/(x^2 - 4) dx
    print("\nb. ∫[1 to 1.6] 2x/(x^2 - 4) dx")
    print("-" * 40)

    b_limits = (1, 1.6)
    approx_b = gaussian_quadrature_n2(f_b, *b_limits)
    exact_b = exact_integral_b()
    error_b = abs(exact_b - approx_b)

    print(f"   Gaussian quadrature (n=2): {approx_b:.8f}")
    print(f"   Exact value:               {exact_b:.8f}")
    print(f"   Absolute error:            {error_b:.2e}")
    print(f"   Relative error:            {error_b/abs(exact_b)*100:.4f}%")

    # Problem c: ∫[3 to 3.5] x/sqrt(x^2 - 4) dx
    print("\nc. ∫[3 to 3.5] x/√(x^2 - 4) dx")
    print("-" * 40)

    c_limits = (3, 3.5)
    approx_c = gaussian_quadrature_n2(f_c, *c_limits)
    exact_c = exact_integral_c()
    error_c = abs(exact_c - approx_c)

    print(f"   Gaussian quadrature (n=2): {approx_c:.8f}")
    print(f"   Exact value:               {exact_c:.8f}")
    print(f"   Absolute error:            {error_c:.2e}")
    print(f"   Relative error:            {error_c/abs(exact_c)*100:.4f}%")

    # Problem d: ∫[0 to π/4] (cos x)^2 dx
    print("\nd. ∫[0 to π/4] (cos x)^2 dx")
    print("-" * 40)

    d_limits = (0, math.pi / 4)
    approx_d = gaussian_quadrature_n2(f_d, *d_limits)
    exact_d = exact_integral_d()
    error_d = abs(exact_d - approx_d)

    print(f"   Gaussian quadrature (n=2): {approx_d:.8f}")
    print(f"   Exact value:               {exact_d:.8f}")
    print(f"   Absolute error:            {error_d:.2e}")
    print(f"   Relative error:            {error_d/abs(exact_d)*100:.4f}%")

if __name__ == "__main__":
    solve_problem()
