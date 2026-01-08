import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from lonpy.discrete.local_search import hill_climb
from lonpy.discrete.neighborhoods import (
    FlipNeighborhood,
    Neighborhood,
    SwapNeighborhood,
)
from lonpy.discrete.solution import Solution
from lonpy.lon import LON
from lonpy.problems.base import ProblemInstance

RepresentationType = Literal["bitstring", "permutation"]
NeighborhoodType = Literal["flip", "swap"]


@dataclass
class ILSSamplerConfig:
    """
    Configuration for Iterated Local Search (ILS) sampling.

    ILS alternates between local search (hill climbing) and perturbation
    to explore the fitness landscape and construct a Local Optima Network.

    Attributes:
        n_runs: Number of independent ILS runs.
        max_iterations: Maximum iterations per run (0 = unlimited, use non_improvement_iterations).
        non_improvement_iterations: Stop after this many iterations without improvement.
        perturbation_strength: Number of random moves for perturbation (e.g., bits to flip).
        first_improvement: If True, use first improvement hill climbing.
            If False, use best improvement.
        representation: Solution representation type ("bitstring" or "permutation").
        neighborhood: Neighborhood type ("flip" for bitstrings, "swap" for permutations).
        seed: Random seed for reproducibility.
    """

    n_runs: int = 100
    max_iterations: int = 0
    non_improvement_iterations: int = 100
    perturbation_strength: int = 2
    first_improvement: bool = True
    representation: RepresentationType = "bitstring"
    neighborhood: NeighborhoodType = "flip"
    seed: int | None = None


class ILSSampler:
    """
    Iterated Local Search (ILS) sampler for constructing Local Optima Networks (discrete).

    ILS explores the fitness landscape by alternating between:
    1. Local search (hill climbing) to reach a local optimum
    2. Perturbation to escape and discover new basins

    The sampler records transitions between local optima to construct
    a Local Optima Network (LON).

    Example:
        >>> from lonpy.discrete import ILSSampler, ILSSamplerConfig
        >>> from lonpy.problems import OneMax
        >>>
        >>> problem = OneMax(n=20)
        >>> config = ILSSamplerConfig(n_runs=100)
        >>> sampler = ILSSampler(config)
        >>> lon = sampler.sample_to_lon(problem)
    """

    def __init__(self, config: ILSSamplerConfig | None = None):
        """
        Initialize sampler with configuration.

        Args:
            config: Sampler configuration. Uses defaults if None.
        """
        self.config = config or ILSSamplerConfig()
        self._rng: random.Random | None = None

    def _get_neighborhood(self) -> Neighborhood:
        if self.config.neighborhood == "flip":
            return FlipNeighborhood()
        elif self.config.neighborhood == "swap":
            return SwapNeighborhood()
        else:
            raise ValueError(f"Unknown neighborhood type: {self.config.neighborhood}")

    def _create_initial_solution(self, n: int) -> Solution:
        if self.config.representation == "bitstring":
            return Solution.random_bitstring(n, self._rng)
        elif self.config.representation == "permutation":
            return Solution.random_permutation(n, self._rng)
        else:
            raise ValueError(f"Unknown representation: {self.config.representation}")

    def _fitness_to_int(self, fitness: float, scale: float = 1e6) -> int:
        """
        Convert fitness to integer for storage.

        Args:
            fitness: Floating-point fitness value.
            scale: Scaling factor.

        Returns:
            Scaled integer fitness.
        """
        return round(fitness * scale)

    def _run_single_ils(
        self,
        run_number: int,
        problem: ProblemInstance,
        neighborhood: Neighborhood,
        solution_size: int,
    ) -> list[dict]:
        """
        Run a single ILS trajectory.

        Args:
            run_number: Run identifier.
            problem: Problem instance.
            neighborhood: Neighborhood operator.
            solution_size: Size of solutions.

        Returns:
            List of transition records.
        """
        records = []

        current = self._create_initial_solution(solution_size)
        current.fitness = problem.evaluate(current.data)

        current = hill_climb(
            current,
            problem,
            neighborhood,
            first_improvement=self.config.first_improvement,
            rng=self._rng,
        )

        best = current.copy()
        non_improvement_count = 0
        iteration = 0

        while True:
            if self.config.max_iterations > 0 and iteration >= self.config.max_iterations:
                break
            if non_improvement_count >= self.config.non_improvement_iterations:
                break

            iteration += 1

            perturbed = neighborhood.apply_random_perturbation(best, self.config.perturbation_strength, self._rng)
            perturbed.fitness = problem.evaluate(perturbed.data)

            new_optimum = hill_climb(
                perturbed,
                problem,
                neighborhood,
                first_improvement=self.config.first_improvement,
                rng=self._rng,
            )

            # Record transition from best to new_optimum
            # Both best and new_optimum should have fitness set after hill_climb
            best_fit = best.fitness if best.fitness is not None else 0.0
            new_fit = new_optimum.fitness if new_optimum.fitness is not None else 0.0

            records.append(
                {
                    "run": run_number,
                    "fit1": self._fitness_to_int(best_fit),
                    "node1": best.to_hash(),
                    "fit2": self._fitness_to_int(new_fit),
                    "node2": new_optimum.to_hash(),
                }
            )

            # Update best if new optimum is better or equal
            if problem.better_or_equal(new_fit, best_fit):
                if problem.strictly_better(new_fit, best_fit):
                    non_improvement_count = 0
                else:
                    non_improvement_count += 1
                best = new_optimum
            else:
                non_improvement_count += 1

        return records

    def sample(
        self,
        problem: ProblemInstance,
        solution_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Run ILS sampling to generate LON data.

        Args:
            problem: Problem instance to sample.
            solution_size: Size of solutions. If None, tries to get from problem.n.
            progress_callback: Optional callback(run, total_runs) for progress.

        Returns:
            Tuple of (trace_df, raw_records):
                - trace_df: DataFrame with columns [run, fit1, node1, fit2, node2]
                - raw_records: List of all transition dictionaries
        """
        if self.config.seed is not None:
            self._rng = random.Random(self.config.seed)
        else:
            self._rng = random.Random()

        # Determine solution size
        if solution_size is None:
            n = getattr(problem, "n", None)
            if n is not None:
                solution_size = int(n)
            else:
                raise ValueError("solution_size must be provided if problem doesn't have 'n' attribute")
        assert solution_size is not None

        neighborhood = self._get_neighborhood()
        all_records: list[dict] = []

        for run in range(1, self.config.n_runs + 1):
            if progress_callback:
                progress_callback(run, self.config.n_runs)

            run_records = self._run_single_ils(run, problem, neighborhood, solution_size)
            all_records.extend(run_records)

        trace_df = pd.DataFrame(all_records, columns=pd.Index(["run", "fit1", "node1", "fit2", "node2"]))
        return trace_df, all_records

    def sample_to_lon(
        self,
        problem: ProblemInstance,
        solution_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> LON:
        """
        Run ILS sampling and return LON directly.

        Args:
            problem: Problem instance to sample.
            solution_size: Size of solutions.
            progress_callback: Optional progress callback.

        Returns:
            LON instance constructed from sampling.
        """
        trace_df, _ = self.sample(problem, solution_size, progress_callback)

        if trace_df.empty:
            return LON()

        return LON.from_trace_data(trace_df)


def compute_discrete_lon(
    problem: ProblemInstance,
    solution_size: int | None = None,
    n_runs: int = 100,
    non_improvement_iterations: int = 100,
    perturbation_strength: int = 2,
    first_improvement: bool = True,
    seed: int | None = None,
) -> LON:
    """
    Compute a LON from a discrete optimization problem.

    This is the simplest way to construct a discrete LON.
    For more control, use ILSSampler directly.

    Args:
        problem: Problem instance (must have evaluate() method and 'n' attribute).
        solution_size: Size of solutions (uses problem.n if None).
        n_runs: Number of independent ILS runs.
        non_improvement_iterations: Stop after this many non-improving iterations.
        perturbation_strength: Number of random moves per perturbation.
        first_improvement: Use first improvement hill climbing.
        seed: Random seed for reproducibility.

    Returns:
        LON instance.

    Example:
        >>> from lonpy.problems import OneMax
        >>> problem = OneMax(n=20)
        >>> lon = compute_discrete_lon(problem, n_runs=50)
        >>> print(f"Found {lon.n_vertices} local optima")
    """
    config = ILSSamplerConfig(
        n_runs=n_runs,
        non_improvement_iterations=non_improvement_iterations,
        perturbation_strength=perturbation_strength,
        first_improvement=first_improvement,
        seed=seed,
    )

    sampler = ILSSampler(config)
    return sampler.sample_to_lon(problem, solution_size)
