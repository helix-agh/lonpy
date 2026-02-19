from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lonpy.lon import LON, LONConfig

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
        max_perturbations_without_improvement: Number of perturbations without improvement before stopping.
        step_mode: Perturbation mode - "percentage" (of domain range)
            or "fixed" (absolute step size).
        step_size: Perturbation magnitude (interpretation depends on step_mode).
        fitness_precision: Decimal precision for fitness values.
            Use None for full double precision. Passing negative values behaves the same as passing None.
        coordinate_precision: Decimal precision for coordinate rounding and hashing.
            Solutions rounded to this precision are considered identical.
            Use None for full double precision (no rounding). Passing negative values behaves the same as passing None.
        bounded: Whether to enforce domain bounds during perturbation.
        minimizer_method: Scipy minimizer method (default: "L-BFGS-B").
        minimizer_options: Options passed to scipy.optimize.minimize.
        seed: Random seed for reproducibility.
    """

    n_runs: int = 100
    max_perturbations_without_improvement: int = 1000
    step_mode: StepMode = "fixed"
    step_size: float = 0.01
    fitness_precision: int | None = None
    coordinate_precision: int | None = 5
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
        >>> config = BasinHoppingSamplerConfig(n_runs=10, max_perturbations_without_improvement=1000)
        >>> sampler = BasinHoppingSampler(config)
        >>> lon = sampler.sample_to_lon(objective_func, domain)
    """

    def __init__(self, config: BasinHoppingSamplerConfig | None = None):
        self.config = config or BasinHoppingSamplerConfig()

    def _perturbation(
        self,
        x: np.ndarray,
        p: np.ndarray,
        bounds: np.ndarray | None = None,
    ) -> np.ndarray:
        y = x + np.random.uniform(low=-p, high=p)
        if self.config.bounded and bounds is not None:
            return np.clip(y, bounds[:, 0], bounds[:, 1])
        return y

    def _round_value(self, value: np.ndarray, precision: int | None) -> np.ndarray:
        if precision is None or precision < 0:
            return value
        return np.round(value, precision)

    def _hash_solution(self, x: np.ndarray) -> str:
        """
        Create hash string for a solution.

        Creates a unique identifier for a local optimum based on
        rounded coordinates.

        Args:
            x: Solution coordinates.

        Returns:
            Hash string identifying the local optimum.
        """

        x = (
            x + 0.0
        )  # Convert -0.0 to 0.0 for consistent hashing (avoids in-place mutation of input)

        precision = self.config.coordinate_precision
        formatter = str if precision is None or precision < 0 else lambda v: f"{v:.{precision}f}"
        hash_str = "_".join(formatter(v) for v in x)

        return hash_str

    def _fitness_to_int(self, fitness: float) -> int:
        """
        Convert fitness to integer representation for storage.

        Args:
            fitness: Floating-point fitness value.

        Returns:
            Scaled integer fitness value.
        """
        if self.config.fitness_precision is None or self.config.fitness_precision < 0:
            return int(fitness * 1e6)
        scale = 10**self.config.fitness_precision
        return int(round(fitness * scale))

    def _basin_hopping_sampling(
        self,
        func: Callable[[np.ndarray], float],
        domain: list[tuple[float, float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
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

        bounds_array = domain_array if self.config.bounded else None
        raw_records = []

        for run in range(1, self.config.n_runs + 1):
            if progress_callback:
                progress_callback(run, self.config.n_runs)

            # Random initial point
            x0 = np.random.uniform(domain_array[:, 0], domain_array[:, 1])

            res = minimize(
                func,
                x0,
                method=self.config.minimizer_method,
                options=self.config.minimizer_options,
                bounds=bounds_array if self.config.bounded else None,
            )

            current_x = res.x
            current_f = res.fun

            perturbations_without_improvement = 0
            run_index = 0

            while (
                perturbations_without_improvement
                < self.config.max_perturbations_without_improvement
            ):
                x_perturbed = self._perturbation(current_x, p, bounds_array)
                res = minimize(
                    func,
                    x_perturbed,
                    method=self.config.minimizer_method,
                    options=self.config.minimizer_options,
                    bounds=bounds_array if self.config.bounded else None,
                )

                new_x = res.x
                new_f = res.fun

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
                    perturbations_without_improvement = 0
                else:
                    perturbations_without_improvement += 1

                # Acceptance criterion (minimization: accept if better or equal)
                if new_f <= current_f:
                    current_x = new_x.copy()
                    current_f = new_f

                run_index += 1

        return raw_records

    def _construct_trace_data(self, raw_records: list[dict]) -> pd.DataFrame:
        """
        Construct trace data from accepted transitions in raw records.

        Args:
            raw_records: List of raw sampling records from basin hopping.

        Returns:
            DataFrame with columns [run, fit1, node1, fit2, node2] representing
            actual transitions from current_x to new_x for each accepted move.
        """
        trace_records = []

        for rec in raw_records:
            if not rec["accepted"]:
                continue

            from_x = rec["current_x"]
            from_f = rec["current_f"]
            to_x = rec["new_x"]
            to_f = rec["new_f"]

            from_x_rounded = self._round_value(from_x, self.config.coordinate_precision)
            to_x_rounded = self._round_value(to_x, self.config.coordinate_precision)

            node1 = self._hash_solution(from_x_rounded)
            node2 = self._hash_solution(to_x_rounded)

            fit1 = self._round_value(from_f, self.config.fitness_precision)
            fit2 = self._round_value(to_f, self.config.fitness_precision)

            trace_records.append(
                {
                    "run": rec["run"],
                    "fit1": fit1,
                    "node1": node1,
                    "fit2": fit2,
                    "node2": node2,
                }
            )

        trace_df = pd.DataFrame(trace_records, columns=["run", "fit1", "node1", "fit2", "node2"])
        return trace_df

    def sample(
        self,
        func: Callable[[np.ndarray], float],
        domain: list[tuple[float, float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Run Basin-Hopping sampling and construct trace data.

        Args:
            func: Objective function to minimize (f: R^n -> R).
            domain: List of (lower, upper) bounds per dimension.
            progress_callback: Optional callback(run, total_runs) for progress.

        Returns:
            Tuple of (trace_df, raw_records):
                - trace_df: DataFrame with columns [run, fit1, node1, fit2, node2]
                - raw_records: List of dicts with detailed iteration data.
        """
        # Collect all raw sampling data
        raw_records = self._basin_hopping_sampling(func, domain, progress_callback)

        # Construct trace data from accepted transitions
        trace_df = self._construct_trace_data(raw_records)

        return trace_df, raw_records

    def sample_to_lon(
        self,
        func: Callable[[np.ndarray], float],
        domain: list[tuple[float, float]],
        progress_callback: Callable[[int, int], None] | None = None,
        lon_config: LONConfig | None = None,
    ) -> LON:
        trace_df, _ = self.sample(func, domain, progress_callback)

        if trace_df.empty:
            return LON()

        _lon_config = replace(lon_config) if lon_config is not None else LONConfig()

        if _lon_config.eq_atol is None:
            p = self.config.fitness_precision
            if p is not None and p >= 0:
                _lon_config.eq_atol = 10 ** -(p + 1)

        return LON.from_trace_data(trace_df, config=_lon_config)


def compute_lon(
    func: Callable[[np.ndarray], float],
    dim: int,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    seed: int | None = None,
    step_size: float = 0.01,
    step_mode: StepMode = "fixed",
    n_runs: int = 100,
    max_perturbations_without_improvement: int = 1000,
    fitness_precision: int | None = None,
    coordinate_precision: int | None = 5,
    bounded: bool = True,
    lon_config: LONConfig | None = None,
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
        max_perturbations_without_improvement: Maximum number of consecutive non-improving perturbations before stopping each run.
        fitness_precision: Decimal precision for fitness values (None for full double). Passing negative values behaves the same as passing None.
        coordinate_precision: Decimal precision for coordinate hashing (None for no rounding). Passing negative values behaves the same as passing None.
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
        [lower_bound] * dim if isinstance(lower_bound, int | float) else lower_bound
    )
    upper_bounds: Sequence[float] = (
        [upper_bound] * dim if isinstance(upper_bound, int | float) else upper_bound
    )

    domain = list(zip(lower_bounds, upper_bounds, strict=True))

    config = BasinHoppingSamplerConfig(
        n_runs=n_runs,
        max_perturbations_without_improvement=max_perturbations_without_improvement,
        step_mode=step_mode,
        step_size=step_size,
        fitness_precision=fitness_precision,
        coordinate_precision=coordinate_precision,
        bounded=bounded,
        seed=seed,
    )

    sampler = BasinHoppingSampler(config)
    return sampler.sample_to_lon(func, domain, lon_config=lon_config)
