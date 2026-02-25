import numpy as np
import pytest

from lonpy import LON, BasinHoppingSampler, BasinHoppingSamplerConfig

N_RUNS = 5
SEED = 42
DOMAIN_2D = [(-5.0, 5.0), (-5.0, 5.0)]


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


@pytest.fixture(scope="session")
def sphere_lon() -> LON:
    config = BasinHoppingSamplerConfig(
        n_runs=N_RUNS,
        max_perturbations_without_improvement=50,
        seed=SEED,
    )
    sampler = BasinHoppingSampler(config)
    return sampler.sample_to_lon(sphere, DOMAIN_2D)


@pytest.fixture(scope="session")
def rastrigin_lon() -> LON:
    """LON with multiple nodes, built from rastrigin function."""
    config = BasinHoppingSamplerConfig(
        n_runs=N_RUNS,
        max_perturbations_without_improvement=50,
        seed=SEED,
    )
    sampler = BasinHoppingSampler(config)
    return sampler.sample_to_lon(rastrigin, DOMAIN_2D)
