import sympy as sp

def newtons_method(expr, var, x0, epsilon=1e-7, max_iter=100):
    f = sp.lambdify(var, expr, modules=["math"])
    df_expr = sp.diff(expr, var)
    df = sp.lambdify(var, df_expr, modules=["math"])

    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        print(f"Iteration {i}: x = {x}")

        if abs(dfx) < 1e-12:
            raise ZeroDivisionError(f"Derivative too small at iteration {i}, x = {x}")

        x_new = x - fx / dfx

        if abs(x_new - x) < epsilon:
            print(f"Converged at iteration {i+1}")
            return x_new

        x = x_new

    raise ValueError(f"Newton's method did not converge within {max_iter} iterations")

import sympy as sp

def modified_newtons_method(expr, var, x0, epsilon=1e-7, max_iter=100):
    # f(x), f'(x), f''(x)
    f = sp.lambdify(var, expr, modules=["math"])
    df_expr = sp.diff(expr, var)
    df = sp.lambdify(var, df_expr, modules=["math"])
    d2f_expr = sp.diff(df_expr, var)
    d2f = sp.lambdify(var, d2f_expr, modules=["math"])

    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)

        numerator = fx * dfx
        denominator = dfx**2 - fx * d2fx

        if abs(denominator) < 1e-12:
            raise ZeroDivisionError(f"Denominator too small at iteration {i}, x = {x}")

        x_new = x - numerator / denominator

        print(f"Modified Iteration {i}: x = {x}")

        if abs(x_new - x) < epsilon:
            print(f"Modified method converged at iteration {i+1}")
            return x_new

        x = x_new

    raise ValueError(f"Modified Newton's method did not converge within {max_iter} iterations")


# Example usage
if __name__ == "__main__":
    x = sp.symbols('x')
    expr = sp.exp(6*x) + 1.441*sp.exp(2*x) - 2.079*sp.exp(4*x) - 0.333

    print("=== Standard Newton's Method ===")
    root = newtons_method(expr, x, x0=-0.5, epsilon=10e-5, max_iter=100)
    print(f"Final root approximation: {root}")
    
    print("=== Modified Newton's Method ===")
    root_modified = modified_newtons_method(expr, x, x0=-0.5, epsilon=10e-5, max_iter=100)
    print(f"Modified method root: {root_modified}")



