import random
from abc import ABC, abstractmethod
from typing import Any, Protocol, cast

from lonpy.discrete.solution import Solution
from lonpy.problems.base import ProblemInstance


class DeltaEvaluationSupport(Protocol):
    """Protocol for problems that support delta evaluation."""

    def supports_delta_evaluation(self) -> bool: ...
    def flip_delta(self, solution: Any, index: int) -> float: ...


class Neighborhood(ABC):
    """
    Abstract base class for neighborhood operators.

    A neighborhood defines how to explore solutions adjacent to the current one.
    """

    @abstractmethod
    def get_neighbor_indices(self, solution: Solution) -> list:
        """
        Get list of indices that define the neighborhood.

        Args:
            solution: Current solution.

        Returns:
            List of indices representing possible moves.
        """

    @abstractmethod
    def apply_move(self, solution: Solution, index: int | tuple[int, int]) -> Solution:
        """
        Apply a move to create a neighbor solution.

        Args:
            solution: Current solution.
            index: Move index from get_neighbor_indices().

        Returns:
            New neighbor solution.
        """

    @abstractmethod
    def apply_random_perturbation(self, solution: Solution, strength: int, rng: random.Random | None = None) -> Solution:
        """
        Apply random perturbation of given strength.

        Args:
            solution: Current solution.
            strength: Number of random moves to apply.
            rng: Random number generator.

        Returns:
            Perturbed solution.
        """


class FlipNeighborhood(Neighborhood):
    """
    Flip neighborhood for bitstring representations.

    Each neighbor is obtained by flipping exactly one bit.
    The neighborhood size is n (solution length).
    """

    def get_neighbor_indices(self, solution: Solution) -> list[int]:
        """Return indices 0 to n-1 representing each possible flip."""
        return list(range(solution.n))

    def apply_move(self, solution: Solution, index: int | tuple[int, int]) -> Solution:
        """
        Flip bit at given index.

        Args:
            solution: Current solution.
            index: Bit index to flip (int for FlipNeighborhood).

        Returns:
            New solution with flipped bit.
        """
        if isinstance(index, tuple):
            raise TypeError("FlipNeighborhood expects int index, not tuple")
        neighbor = solution.copy()
        neighbor.flip(index)
        return neighbor

    def apply_random_perturbation(self, solution: Solution, strength: int, rng: random.Random | None = None) -> Solution:
        """
        Flip `strength` random bits.

        Args:
            solution: Current solution.
            strength: Number of bits to flip.
            rng: Random number generator.

        Returns:
            Perturbed solution with `strength` flipped bits.
        """
        if rng is None:
            rng = random.Random()

        perturbed = solution.copy()
        indices = rng.sample(range(solution.n), min(strength, solution.n))
        for idx in indices:
            perturbed.flip(idx)
        return perturbed

    def evaluate_neighbor_with_delta(
        self,
        solution: Solution,
        index: int,
        problem: ProblemInstance,
    ) -> float:
        """
        Evaluate neighbor using delta evaluation if supported.

        Args:
            solution: Current solution (must have fitness set).
            index: Bit index to flip.
            problem: Problem instance.

        Returns:
            Fitness of neighbor after flipping bit at index.
        """
        if (
            hasattr(problem, "flip_delta")
            and hasattr(problem, "supports_delta_evaluation")
            and cast(DeltaEvaluationSupport, problem).supports_delta_evaluation()
            and solution.fitness is not None
        ):
            delta = cast(DeltaEvaluationSupport, problem).flip_delta(solution.data, index)
            return float(solution.fitness + delta)

        # Fall back to full evaluation
        neighbor = self.apply_move(solution, index)
        return float(problem.evaluate(neighbor.data))


class SwapNeighborhood(Neighborhood):
    """
    Swap neighborhood for permutation representations.

    Each neighbor is obtained by swapping two elements.
    The neighborhood size is n*(n-1)/2.
    """

    def get_neighbor_indices(self, solution: Solution) -> list[tuple[int, int]]:
        """Return all (i, j) pairs where i < j representing possible swaps."""
        n = solution.n
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def apply_move(self, solution: Solution, index: int | tuple[int, int]) -> Solution:
        """
        Swap elements at given indices.

        Args:
            solution: Current solution.
            index: Tuple (i, j) of indices to swap.

        Returns:
            New solution with swapped elements.
        """
        if not isinstance(index, tuple):
            raise TypeError("SwapNeighborhood expects tuple index, not int")
        neighbor = solution.copy()
        i, j = index
        neighbor.swap(i, j)
        return neighbor

    def apply_random_perturbation(self, solution: Solution, strength: int, rng: random.Random | None = None) -> Solution:
        """
        Perform `strength` random swaps.

        Args:
            solution: Current solution.
            strength: Number of swaps to perform.
            rng: Random number generator.

        Returns:
            Perturbed solution with `strength` random swaps applied.
        """
        if rng is None:
            rng = random.Random()

        perturbed = solution.copy()
        n = solution.n
        for _ in range(strength):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            while j == i:
                j = rng.randint(0, n - 1)
            perturbed.swap(i, j)
        return perturbed
