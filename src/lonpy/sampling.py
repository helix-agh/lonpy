from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lonpy.lon import LON

StepMode = Literal["percentage", "fixed"]


@dataclass
class BasinHoppingSamplerConfig:
    """
    Configuration for Basin-Hopping sampling.

    Default values have been set to match the paper
      Jason Adair, Gabriela Ochoa, and Katherine M. Malan. 2019.
      Local optima networks for continuous fitness landscapes.
      In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '19).
      Association for Computing Machinery, New York, NY, USA, 1407-1414.
      https://doi.org/10.1145/3319619.3326852

    Attributes:
        n_runs: Number of independent Basin-Hopping runs.
        n_iterations: Number of iterations per run.
        step_mode: Perturbation mode - "percentage" (of domain range)
            or "fixed" (absolute step size).
        step_size: Perturbation magnitude (interpretation depends on step_mode).
        opt_digits: Decimal precision for optimization results.
            Use -1 for double precision.
        hash_digits: Decimal precision for solution hashing. Solutions
            rounded to this precision are considered identical.
        bounded: Whether to enforce domain bounds during perturbation.
        minimizer_method: Scipy minimizer method (default: "L-BFGS-B").
        minimizer_options: Options passed to scipy.optimize.minimize.
        seed: Random seed for reproducibility.
    """

    n_runs: int = 100
    n_iterations: int = 1000
    step_mode: StepMode = "fixed"
    step_size: float = 0.01
    opt_digits: int = -1
    hash_digits: int = 5
    bounded: bool = True
    minimizer_method: str = "L-BFGS-B"
    minimizer_options: dict = field(
        default_factory=lambda: {"ftol": 1e-07, "gtol": 0, "maxiter": 15000}
    )
    seed: int | None = None


class BasinHoppingSampler:
    """
    Basin-Hopping sampler for constructing Local Optima Networks.

    Basin-Hopping is a global optimization algorithm that combines random
    perturbations with local minimization. This implementation records
    transitions between local optima for LON construction.

    Example:
        >>> config = BasinHoppingSamplerConfig(n_runs=10, n_iterations=1000)
        >>> sampler = BasinHoppingSampler(config)
        >>> lon = sampler.sample_to_lon(objective_func, domain)
    """

    def __init__(self, config: BasinHoppingSamplerConfig | None = None):
        self.config = config or BasinHoppingSamplerConfig()

    def bounded_perturbation(
        self,
        x: np.ndarray,
        p: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        y = x + np.random.uniform(low=-p, high=p)
        return np.clip(y, bounds[:, 0], bounds[:, 1])

    def unbounded_perturbation(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return x + np.random.uniform(low=-p, high=p)

    def hash_solution(self, x: np.ndarray, fitness: float = 0.0) -> str:  # noqa: ARG002
        """
        Create hash string for a solution.

        Creates a unique identifier for a local optimum based on
        rounded coordinates.

        Args:
            x: Solution coordinates.
            fitness: Fitness value (unused, kept for API compatibility).

        Returns:
            Hash string identifying the local optimum.
        """
        if self.config.hash_digits < 0:
            rounded = x
        else:
            rounded = np.round(x, self.config.hash_digits)

        rounded = rounded + 0.0  # Convert -0.0 to 0.0 for consistent hashing

        hash_str = "_".join(f"{v:.{max(0, self.config.hash_digits)}f}" for v in rounded)
        return hash_str

    def fitness_to_int(self, fitness: float) -> int:
        """
        Convert fitness to integer representation for storage.

        Args:
            fitness: Floating-point fitness value.

        Returns:
            Scaled integer fitness value.
        """
        if self.config.hash_digits < 0:
            return int(fitness * 1e6)
        scale = 10**self.config.hash_digits
        return np.round(fitness * scale)

    def sample(
        self,
        func: Callable[[Sequence], float],
        domain: list[tuple[float, float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Run Basin-Hopping sampling to generate LON data.

        Args:
            func: Objective function to minimize (f: R^n -> R).
            domain: List of (lower, upper) bounds per dimension.
            progress_callback: Optional callback(run, total_runs) for progress.

        Returns:
            Tuple of (trace_df, raw_records):
                - trace_df: DataFrame with columns [run, fit1, node1, fit2, node2]
                  ready for LON construction.
                - raw_records: List of dicts with detailed iteration data.
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        n_var = len(domain)
        domain_array = np.array(domain)

        # Compute step size based on mode
        if self.config.step_mode == "percentage":
            p = self.config.step_size * np.abs(domain_array[:, 1] - domain_array[:, 0])
        else:
            p = self.config.step_size * np.ones(n_var)

        bounds_scipy = domain if self.config.bounded else None
        bounds_array = domain_array if self.config.bounded else None
        trace_records = []
        raw_records = []

        for run in range(1, self.config.n_runs + 1):
            if progress_callback:
                progress_callback(run, self.config.n_runs)

            # Random initial point
            x0 = np.random.uniform(domain_array[:, 0], domain_array[:, 1])

            if self.config.bounded:
                res = minimize(
                    func,
                    x0,
                    method=self.config.minimizer_method,
                    options=self.config.minimizer_options,
                    bounds=bounds_scipy,
                )
            else:
                res = minimize(
                    func,
                    x0,
                    method=self.config.minimizer_method,
                    options=self.config.minimizer_options,
                )

            if self.config.opt_digits < 0:
                current_x = res.x
                current_f = res.fun
            else:
                current_x = np.round(res.x, self.config.opt_digits)
                current_f = np.round(res.fun, self.config.opt_digits)

            runs_without_improvement = 0
            run_index = 0

            while runs_without_improvement < self.config.n_iterations:
                if self.config.bounded and bounds_array is not None:
                    x_perturbed = self.bounded_perturbation(current_x, p, bounds_array)
                    res = minimize(
                        func,
                        x_perturbed,
                        method=self.config.minimizer_method,
                        bounds=bounds_scipy,
                        options=self.config.minimizer_options,
                    )
                else:
                    x_perturbed = self.unbounded_perturbation(current_x, p)
                    res = minimize(
                        func,
                        x_perturbed,
                        method=self.config.minimizer_method,
                        options=self.config.minimizer_options,
                    )

                if self.config.opt_digits < 0:
                    new_x = res.x
                    new_f = res.fun
                else:
                    new_x = np.round(res.x, self.config.opt_digits)
                    new_f = np.round(res.fun, self.config.opt_digits)

                raw_records.append(
                    {
                        "run": run,
                        "iteration": run_index,
                        "current_x": current_x.copy(),
                        "current_f": current_f,
                        "new_x": new_x.copy(),
                        "new_f": new_f,
                        "accepted": new_f <= current_f,
                    }
                )

                if new_f < current_f:
                    runs_without_improvement = 0
                else:
                    runs_without_improvement += 1

                # Acceptance criterion (minimization: accept if better or equal)
                if new_f <= current_f:
                    node1 = self.hash_solution(current_x, current_f)
                    node2 = self.hash_solution(new_x, new_f)
                    fit1 = self.fitness_to_int(current_f)
                    fit2 = self.fitness_to_int(new_f)

                    trace_records.append(
                        {
                            "run": run,
                            "fit1": fit1,
                            "node1": node1,
                            "fit2": fit2,
                            "node2": node2,
                        }
                    )

                    current_x = new_x.copy()
                    current_f = new_f

                run_index += 1

        trace_df = pd.DataFrame(trace_records, columns=["run", "fit1", "node1", "fit2", "node2"])
        return trace_df, raw_records

    def sample_to_lon(
        self,
        func: Callable[[Sequence], float],
        domain: list[tuple[float, float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> LON:
        trace_df, _ = self.sample(func, domain, progress_callback)

        if trace_df.empty:
            return LON()

        return LON.from_trace_data(trace_df)


def compute_lon(
    func: Callable[[Sequence], float],
    dim: int,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    seed: int | None = None,
    step_size: float = 0.01,
    step_mode: StepMode = "percentage",
    n_runs: int = 10,
    n_iterations: int = 1000,
    opt_digits: int = -1,
    hash_digits: int = 4,
    bounded: bool = True,
) -> LON:
    """
    Compute a LON from an objective function.

    This is the simplest way to construct a Local Optima Network.
    For more control, use BasinHoppingSampler directly.

    Args:
        func: Objective function f(x) -> float to minimize.
        dim: Number of dimensions.
        lower_bound: Lower bound (scalar or per-dimension list/array).
        upper_bound: Upper bound (scalar or per-dimension list/array).
        seed: Random seed for reproducibility.
        step_size: Perturbation step size.
        step_mode: "percentage" (of domain) or "fixed".
        n_runs: Number of independent Basin-Hopping runs.
        n_iterations: Iterations per run.
        opt_digits: Decimal precision for optimization (-1 for double).
        hash_digits: Decimal precision for solution hashing.
        bounded: Whether to enforce domain bounds.

    Returns:
        LON instance.

    Example:
        >>> import numpy as np
        >>> def sphere(x):
        ...     return np.sum(x**2)
        >>> lon = compute_lon(sphere, dim=5, lower_bound=-5.0, upper_bound=5.0)
        >>> print(f"Found {lon.n_vertices} local optima")
    """
    # Convert scalars to lists, leave sequences as-is
    lower_bounds: Sequence[float] = (
        [lower_bound] * dim if isinstance(lower_bound, (int, float)) else lower_bound
    )
    upper_bounds: Sequence[float] = (
        [upper_bound] * dim if isinstance(upper_bound, (int, float)) else upper_bound
    )

    domain = list(zip(lower_bounds, upper_bounds, strict=True))

    config = BasinHoppingSamplerConfig(
        n_runs=n_runs,
        n_iterations=n_iterations,
        step_mode=step_mode,
        step_size=step_size,
        opt_digits=opt_digits,
        hash_digits=hash_digits,
        bounded=bounded,
        seed=seed,
    )

    sampler = BasinHoppingSampler(config)
    return sampler.sample_to_lon(func, domain)
