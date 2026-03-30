"""
Benchmark functions for optimization.

Input: np.ndarray
Output: float or np.ndarray

https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# -----------------------------------------------------------------------------
# Single-objective, scalar output
# -----------------------------------------------------------------------------


def Sphere(x: np.ndarray) -> float:
    """Convex (simple). -inf <= x <= +inf, min=0 at x=0."""
    return float(np.dot(x, x))


def SumSphere(x: np.ndarray) -> float:
    """Convex (simple). -inf <= x <= +inf, min=0 at x=0."""
    n = x.size
    weights = np.arange(1, n + 1, dtype=x.dtype)
    return float(np.sum(weights * x**2))


def Perm(x: np.ndarray, beta: float = 0.5) -> float:
    """Relatively simple. -n <= x <= +n, min=0 at x=[1,2,3,...,n]."""
    n = x.size
    i = np.arange(1, n + 1, dtype=x.dtype)
    j = np.arange(1, n + 1, dtype=x.dtype)
    # a[j] = sum over i of ((i*j + beta) * ((x[i-1]/i)^j - 1))
    # shape: (n,) and (n,) -> outer gives (n,n); x/i is (n,), (x/i)^j is (n,n) with j as powers
    xi = x / i  # (n,)
    xij = np.power(xi[:, np.newaxis], j)  # (n, n): row i, col j -> (x_i/i)^j
    coeff = (i[:, np.newaxis] * j + beta)  # (n, n)
    a = np.sum(coeff * (xij - 1.0), axis=0)  # (n,)
    return float(np.sum(a**2))


def Perm0(x: np.ndarray, beta: float = 0.5) -> float:
    """Relatively simple. -n <= x <= +n, min=0 at x=[1,1/2,1/3,...,1/n]."""
    n = x.size
    i = np.arange(1, n + 1, dtype=x.dtype)
    j = np.arange(1, n + 1, dtype=x.dtype)
    inv_i = 1.0 / i
    # a[j] = sum over i of (i+beta) * (x[i]^(j+1) - (1/i)^(j+1))
    x_pow = np.power(x[:, np.newaxis], j)   # (n, n): x_i^j
    inv_pow = np.power(inv_i[:, np.newaxis], j)  # (n, n)
    coeff = (i + beta)
    a = np.sum(coeff[:, np.newaxis] * (x_pow - inv_pow), axis=0)
    return float(np.sum(a**2))


def StyblinskiTang(x: np.ndarray) -> float:
    """
    Multiple local minima. -5 <= x <= 5.
    Approx min ≈ -39.16617*n at x ≈ -2.903534 (each dim).
    """
    return float(np.sum(x**4 - 16 * x**2 + 5 * x) / 2.0)


def DixonPrice(x: np.ndarray) -> float:
    """Relatively complex. -10 <= x <= 10, min=0 at x=[1, 1/2, 1/4, ..., 1/2^(n-1)]."""
    y = (x[0] - 1.0) ** 2
    if x.size > 1:
        i = np.arange(2, x.size + 1, dtype=x.dtype)
        y += np.sum(i * (2 * x[1:] * x[1:] - x[:-1]) ** 2)
    return float(y)


def Rosenbrock(x: np.ndarray) -> float:
    """Relatively complex. -inf <= x <= +inf, min=0 at x=[1,1,...,1]."""
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    )


def Schwefel(x: np.ndarray) -> float:
    """
    Complex, deceptive. -500 <= x <= 500, min=0 at x=420.9687 (each dim).
    Many local minima; gradient and small populations tend to get stuck.
    """
    n = x.size
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def Rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    """Complex, many local minima. -2 <= x <= 2, min=0 at x=0."""
    n = x.size
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def Griewank(x: np.ndarray, A: float = 4000.0) -> float:
    """Complex, many local minima. -10 <= x <= 10, min=0 at x=0."""
    n = x.size
    g = np.dot(x, x)
    idx = np.sqrt(np.arange(1, n + 1, dtype=x.dtype))
    f = np.prod(np.cos(x / idx))
    return float(g / A - f + 1.0)


def Gaussian(x: np.ndarray) -> float:
    """Gaussian-shaped. -1 < x < 1 (typical)."""
    return float(np.exp(-2.5 * np.dot(x, x)))


# Backward compatibility alias
def Gussian(x: np.ndarray) -> float:
    """Alias for Gaussian (legacy typo)."""
    return Gaussian(x)


def Ackley(x: np.ndarray) -> float:
    """
    Complex. -32.768 <= x <= 32.768, min=0 at x=0.
    Flat outer region with many local minima and a deep central hole.
    """
    n = x.size
    s = np.sqrt(np.dot(x, x) / n)
    c = np.sum(np.cos(2 * np.pi * x)) / n
    return float(-20.0 * np.exp(-0.2 * s) - np.exp(c) + 20.0 + np.e)


# -----------------------------------------------------------------------------
# 2D problems (scalar output)
# -----------------------------------------------------------------------------


def Franke(x: np.ndarray) -> float:
    """
    Franke's function, 2D. 0 < x < 1.
    Two local maxima of different heights, one local minimum.
    """
    if x.size != 2:
        raise ValueError(f"Franke is a 2D problem; got nx={x.size}")
    x1, x2 = x[0], x[1]
    a1 = 0.75 * np.exp(-0.25 * ((9 * x1 - 2) ** 2 + (9 * x2 - 2) ** 2))
    a2 = 0.75 * np.exp(-(9 * x1 + 1) ** 2 / 49.0 - 0.1 * (9 * x2 + 1) ** 2)
    a3 = 0.50 * np.exp(-0.25 * ((9 * x1 - 7) ** 2 + (9 * x2 - 3) ** 2))
    a4 = -0.2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)
    return float(a1 + a2 + a3 + a4)


def Droplet(x: np.ndarray) -> float:
    """Droplet function, 2D. -1 < x < 1. Maximum at center, bowl elsewhere."""
    if x.size != 2:
        raise ValueError(f"Droplet is a 2D problem; got nx={x.size}")
    r2 = np.dot(x, x)
    return float(-4 * np.exp(-25 / 8.0 * r2) + 7 * np.exp(-125 / 4.0 * r2))


def DeJong2nd(x: np.ndarray) -> float:
    """2D De Jong (Rosenbrock-like)."""
    if x.size != 2:
        raise ValueError(f"DeJong2nd is a 2D problem; got nx={x.size}")
    return float(100 * (x[0] ** 2 - x[1]) ** 2 + (1 - x[0]) ** 2)


# -----------------------------------------------------------------------------
# Multi-output (Pareto / vector objective)
# -----------------------------------------------------------------------------


def Circle2D(x: np.ndarray, ratio: float = 0.3) -> np.ndarray:
    """2D design, 2D circle objective. x1,x2 in [0,1] -> circle C=(0,0) r=1."""
    if x.size != 2:
        raise ValueError(f"Circle2D is a 2D problem; got nx={x.size}")
    r = 1.0 - np.power(np.abs(2 * x[0] - 1), ratio)
    t = (2 * x[1] - 1) * np.pi
    return np.array([r * np.cos(t), r * np.sin(t)], dtype=x.dtype)


def Constr(x: np.ndarray) -> np.ndarray:
    """2D design, 2 outputs. x1 in [0.1, 1], x2 in [0, 5]."""
    if x.size != 2:
        raise ValueError(f"Constr is a 2D problem; got nx={x.size}")
    return np.array([x[0], (1 + x[1]) / x[0]], dtype=x.dtype)


def Srn(x: np.ndarray) -> np.ndarray:
    """2D design, 2 outputs. x1,x2 in [-20, 20]."""
    if x.size != 2:
        raise ValueError(f"Srn is a 2D problem; got nx={x.size}")
    y1 = (x[0] - 2) ** 2 + (x[1] - 1) ** 2 + 2
    y2 = 9 * x[0] - (x[1] - 1) ** 2
    return np.array([y1, y2], dtype=x.dtype)


def ZDT1(x: np.ndarray) -> np.ndarray:
    """ZDT1. x in [0,1], nx>=2. y1 in [0,1], y2 in [0,10]."""
    if x.size < 2:
        raise ValueError(f"ZDT1 requires nx >= 2; got nx={x.size}")
    n = x.size
    f1 = x[0]
    g = 1.0 + 9.0 / (n - 1.0) * np.sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2], dtype=x.dtype)


def ZDT2(x: np.ndarray) -> np.ndarray:
    """ZDT2. x in [0,1], nx>=2. y1 in [0,1], y2 in [0,10]."""
    if x.size < 2:
        raise ValueError(f"ZDT2 requires nx >= 2; got nx={x.size}")
    n = x.size
    f1 = x[0]
    g = 1.0 + 9.0 / (n - 1.0) * np.sum(x[1:])
    h = 1.0 - (f1 / g) ** 2
    f2 = g * h
    return np.array([f1, f2], dtype=x.dtype)


def ZDT3(x: np.ndarray) -> np.ndarray:
    """ZDT3. x in [0,1], nx>=2. y1 in [0,1], y2 in [0,10]."""
    if x.size < 2:
        raise ValueError(f"ZDT3 requires nx >= 2; got nx={x.size}")
    n = x.size
    f1 = x[0]
    g = 1.0 + 9.0 / (n - 1.0) * np.sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    return np.array([f1, f2], dtype=x.dtype)


def ZDT4(x: np.ndarray) -> np.ndarray:
    """ZDT4. x1 in [0,1], x2..xn in [-5,5]. y1 in [0,1], y2 in [0,10]."""
    if x.size < 2:
        raise ValueError(f"ZDT4 requires nx >= 2; got nx={x.size}")
    n = x.size
    f1 = x[0]
    u = (x[1:] - 0.5) * 2
    g = 1.0 + 10 * (n - 1) + np.sum(u**2 - 10 * np.cos(4 * np.pi * u))
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2], dtype=x.dtype)


def ZDT6(x: np.ndarray) -> np.ndarray:
    """ZDT6. x in [0,1], nx>=2 (often nx=10). y1 in [0,1], y2 in [0,10]."""
    if x.size < 2:
        raise ValueError(f"ZDT6 requires nx >= 2; got nx={x.size}")
    n = x.size
    f1 = 1.0 - np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6)
    g = 1.0 + 9.0 * np.power(np.sum(x[1:]) / (n - 1.0), 0.25)
    h = 1.0 - (f1 / g) ** 2
    f2 = g * h
    return np.array([f1, f2], dtype=x.dtype)


# -----------------------------------------------------------------------------
# Custom (Sway LI)
# -----------------------------------------------------------------------------


def RastriginSphere(x: np.ndarray) -> np.ndarray:
    """4 inputs, 2 outputs. x in [-1,1], y in [0,1]."""
    y1 = np.clip(Rastrigin(x[:2]) / 40.0, None, 1.0)
    y2 = Sphere(x[2:]) / 2.0
    return np.array([y1, y2], dtype=x.dtype)


def OneDimProblem(x: np.ndarray) -> float:
    """1D. x in [-10, 50], y roughly in [-3, 23]."""
    if x.size != 1:
        raise ValueError(f"OneDimProblem is 1D; got nx={x.size}")
    a = np.pi * x[0] / 5.0
    b = 10.0 * np.sin(a)
    c = a * (np.abs(1 - x[0] / 32.5) + 0.02)
    return float(b / c)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def Gradient(
    x: np.ndarray,
    func: Callable[[np.ndarray], float],
    dx: float = 1.0e-8,
) -> np.ndarray:
    """Numerical gradient of a scalar function. Returns shape (n,) gradient."""
    y0 = np.asarray(func(x))
    if y0.shape != ():
        raise TypeError("Gradient is only for scalar-output functions")
    y0 = float(y0.flat[0])
    n = x.size
    g = np.empty(n, dtype=x.dtype)
    for i in range(n):
        e = np.zeros(n, dtype=x.dtype)
        e[i] = dx
        g[i] = (func(x + e) - y0) / dx
    return g
