from abc import ABC, abstractmethod
from typing import Any, TypeVar

SolutionType = TypeVar("SolutionType")


class ProblemInstance(ABC):
    """
    Abstract base class for optimization problems.

    This class defines the interface that all problem instances must implement,
    enabling a unified approach to both discrete and continuous optimization.

    Subclasses must implement:
        - maximize(): Whether to maximize or minimize
        - evaluate(): Compute fitness of a solution
    """

    @abstractmethod
    def maximize(self) -> bool:
        """
        Return True if this is a maximization problem, False for minimization.

        Returns:
            True for maximization, False for minimization.
        """

    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        """
        Evaluate the fitness of a solution.

        Args:
            solution: The solution to evaluate.

        Returns:
            Fitness value (higher is better for maximization, lower for minimization).
        """

    def strictly_better(self, a: float, b: float) -> bool:
        """
        Check if fitness a is strictly better than fitness b.

        Args:
            a: First fitness value.
            b: Second fitness value.

        Returns:
            True if a is strictly better than b.
        """
        return a > b if self.maximize() else a < b

    def better_or_equal(self, a: float, b: float) -> bool:
        """
        Check if fitness a is better than or equal to fitness b.

        Args:
            a: First fitness value.
            b: Second fitness value.

        Returns:
            True if a is better than or equal to b.
        """
        return a >= b if self.maximize() else a <= b

    def compare(self, a: float, b: float) -> int:
        """
        Compare two fitness values.

        Args:
            a: First fitness value.
            b: Second fitness value.

        Returns:
            1 if a is better, -1 if b is better, 0 if equal.
        """
        if self.strictly_better(a, b):
            return 1
        elif self.strictly_better(b, a):
            return -1
        return 0
