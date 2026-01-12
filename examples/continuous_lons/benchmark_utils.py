"""Utilities for reproducing experiments from:

Jason Adair, Gabriela Ochoa, and Katherine M. Malan. 2019.
Local optima networks for continuous fitness landscapes.
In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '19).
Association for Computing Machinery, New York, NY, USA, 1407-1414.
https://doi.org/10.1145/3319619.3326852
"""

import numpy as np

# Step sizes from the paper for each dimensionality and function
# Maps: dimensionality -> {function_name -> step_size}
STEP_SIZES: dict[int, dict[str, float]] = {
    3: {"Ackley": 0.4546, "Rastrigin": 0.4746, "Birastrigin": 0.5156},
    5: {"Ackley": 0.4646, "Rastrigin": 0.4749, "Birastrigin": 0.4946},
}


def ackley(x: np.ndarray) -> float:
    """
    Ackley function.
    Global minimum: f(0, 0, ..., 0) = 0
    Search domain: [-32.768, 32.768]^n
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)

    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)

    return float(term1 + term2 + a + np.e)


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function.
    Global minimum: f(0, 0, ..., 0) = 0
    Search domain: [-5.12, 5.12]^n
    """
    A = 10
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def birastrigin(x: np.ndarray) -> float:
    """
    Birastrigin function (Lunacek's bi-Rastrigin).
    The function is defined as:
        f(x) = 10 * sum(1 - cos(2*pi*(x_i - mu_1)))
             + min(sum((x_i - mu_1)^2), d*n + s*sum((x_i - mu_2)^2))
    With parameters:
        d = 1
        s = 1 - 1/(2*sqrt(n + 20) - 8.2)
        mu_1 = 2.5
        mu_2 = sqrt(|mu_1^2 - d / s|)
    Search domain: [-5.12, 5.12]^n
    """
    n = len(x)

    # Parameters from the paper
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * np.sqrt(n + 20.0) - 8.2)
    mu_1 = 2.5
    mu_2 = np.sqrt(np.abs((mu_1**2 - d) / s))

    # Rastrigin-like component centered at mu_1
    rastrigin_term = 10.0 * np.sum(1.0 - np.cos(2.0 * np.pi * (x - mu_1)))

    # Double-sphere component: min of two quadratic terms
    sphere_1 = np.sum((x - mu_1) ** 2)
    sphere_2 = d * n + s * np.sum((x - mu_2) ** 2)
    sphere_term = min(sphere_1, sphere_2)

    return float(rastrigin_term + sphere_term)


BENCHMARKS = [
    ("Ackley", ackley, (-32.768, 32.768)),
    ("Rastrigin", rastrigin, (-5.12, 5.12)),
    ("Birastrigin", birastrigin, (-5.12, 5.12)),
]
