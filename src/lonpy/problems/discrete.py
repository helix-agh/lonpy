from dataclasses import dataclass, field

import numpy as np

from lonpy.problems.base import ProblemInstance


@dataclass
class OneMax(ProblemInstance):
    """
    OneMax problem: maximize the number of 1s in a bitstring.

    This is a simple unimodal problem often used as a benchmark.
    The global optimum is the all-ones bitstring with fitness n.

    Attributes:
        n: Length of the bitstring.

    Example:
        >>> problem = OneMax(n=20)
        >>> solution = [1, 0, 1, 1, 0]  # partial example
        >>> problem.evaluate(solution)
        3
    """

    n: int

    def maximize(self) -> bool:
        """OneMax is a maximization problem."""
        return True

    def evaluate(self, solution: list[int]) -> float:
        """
        Evaluate fitness as sum of 1s.

        Args:
            solution: Bitstring as list of 0/1 integers.

        Returns:
            Number of 1s in the solution.
        """
        return float(sum(solution))

    def flip_delta(self, solution: list[int], index: int) -> float:
        """
        Compute fitness change from flipping bit at index.

        This is O(1) compared to O(n) for full evaluation.

        Args:
            solution: Current bitstring.
            index: Index of bit to flip.

        Returns:
            Change in fitness (+1 if flipping 0->1, -1 if flipping 1->0).
        """
        return -1.0 if solution[index] == 1 else 1.0

    def supports_delta_evaluation(self) -> bool:
        """OneMax supports efficient delta evaluation."""
        return True


@dataclass
class Knapsack(ProblemInstance):
    """
    0/1 Knapsack problem: maximize value without exceeding capacity.

    Each item has a value and weight. Select items to maximize total value
    while keeping total weight within capacity. Infeasible solutions
    (exceeding capacity) get fitness 0.

    Attributes:
        values: Value of each item.
        weights: Weight of each item.
        capacity: Maximum total weight allowed.

    Example:
        >>> problem = Knapsack(values=[60, 100, 120], weights=[10, 20, 30], capacity=50)
        >>> solution = [1, 1, 0]  # take items 0 and 1
        >>> problem.evaluate(solution)
        160.0
    """

    values: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    capacity: float = 0.0

    @property
    def n(self) -> int:
        """Number of items."""
        return len(self.values)

    def maximize(self) -> bool:
        """Knapsack is a maximization problem."""
        return True

    def evaluate(self, solution: list[int]) -> float:
        """
        Evaluate fitness as total value if feasible, 0 otherwise.

        Args:
            solution: Bitstring where 1 means item is selected.

        Returns:
            Total value if weight <= capacity, else 0.
        """
        total_weight = sum(w * s for w, s in zip(self.weights, solution, strict=True))
        if total_weight > self.capacity:
            return 0.0
        return float(sum(v * s for v, s in zip(self.values, solution, strict=True)))

    def supports_delta_evaluation(self) -> bool:
        """Knapsack does not support efficient delta evaluation."""
        return False


@dataclass
class NumberPartitioning(ProblemInstance):
    """
    Number Partitioning Problem (NPP): minimize partition imbalance.

    Given n positive integers, partition them into two subsets such that
    the difference between their sums is minimized. The global optimum
    has fitness 0 (perfect partition).

    Attributes:
        n: Number of items.
        k: Threshold parameter controlling item range (items in [1, 2^(n*k)]).
        seed: Random seed for reproducible instance generation.
        items: The generated items (computed from n, k, seed).

    Example:
        >>> problem = NumberPartitioning(n=10, k=0.5, seed=42)
        >>> solution = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
        >>> fitness = problem.evaluate(solution)
    """

    n: int = 20
    k: float = 0.5
    seed: int = 1
    items: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Generate items if not provided."""
        if not self.items:
            rng = np.random.default_rng(self.seed)
            max_val = int(2 ** (self.n * self.k))
            self.items = [int(rng.integers(1, max_val + 1)) for _ in range(self.n)]

    def maximize(self) -> bool:
        """NPP is a minimization problem."""
        return False

    def evaluate(self, solution: list[int]) -> float:
        """
        Evaluate fitness as absolute difference between partition sums.

        Args:
            solution: Bitstring where 0/1 assigns item to partition A/B.

        Returns:
            |sum(A) - sum(B)| where lower is better.
        """
        sum_a = sum(item for item, bit in zip(self.items, solution, strict=True) if bit == 0)
        sum_b = sum(item for item, bit in zip(self.items, solution, strict=True) if bit == 1)
        return float(abs(sum_a - sum_b))

    def supports_delta_evaluation(self) -> bool:
        """NPP does not support efficient delta evaluation."""
        return False


@dataclass
class NKLandscape(ProblemInstance):
    """
    NK Landscape problem: tunable rugged fitness landscape.

    The NK model is a problem-independent model for constructing multimodal
    landscapes that can gradually be tuned from smooth to rugged:
    - N: number of binary genes in the genotype (string length)
    - K: number of genes that influence a particular gene (0 <= K <= N-1)

    By increasing K from 0 to N-1, landscapes can be tuned from smooth to rugged.

    This implementation uses the adjacent neighborhood model, where the K variables
    forming the context of gene s_i are the K variables closest to s_i in a total
    ordering (s_1, s_2, ..., s_n) using periodic boundaries.

    The fitness is the average of N contribution functions, where each
    contribution f_i depends on bit i and its K adjacent neighbors.

    Attributes:
        n: Length of the bitstring (number of genes).
        k: Epistasis parameter (number of neighboring genes influencing each gene).
        seed: Random seed for reproducible instance generation.
        contributions: Lookup tables for each bit's contribution.

    Example:
        >>> problem = NKLandscape(n=18, k=4, seed=42)
        >>> solution = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        >>> fitness = problem.evaluate(solution)
    """

    n: int = 18
    k: int = 4
    seed: int = 1
    contributions: list[dict[tuple[int, ...], float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Generate NK landscape instance if not already initialized."""
        if self.k < 0 or self.k >= self.n:
            raise ValueError(f"k must be in [0, n-1], got k={self.k}, n={self.n}")

        if not self.contributions:
            self._generate_instance()

    def _generate_instance(self) -> None:
        """Generate contribution tables for adjacent neighborhood model."""
        rng = np.random.default_rng(self.seed)

        self.contributions = []

        for _ in range(self.n):
            # Create contribution lookup table
            # Each entry maps (bit_i, neighbor_0, ..., neighbor_k-1) -> contribution
            n_entries = 2 ** (self.k + 1)  # 2^(K+1) possible combinations
            contribution_table: dict[tuple[int, ...], float] = {}

            for entry_idx in range(n_entries):
                # Convert entry index to bit pattern
                bits = tuple((entry_idx >> b) & 1 for b in range(self.k + 1))
                contribution_table[bits] = float(rng.random())

            self.contributions.append(contribution_table)

    def _get_adjacent_neighbors(self, i: int) -> list[int]:
        """
        Get K adjacent neighbors for gene i using periodic boundaries.

        Args:
            i: Index of the gene.

        Returns:
            List of K neighbor indices.
        """
        return [(i + j + 1) % self.n for j in range(self.k)]

    def maximize(self) -> bool:
        """NK Landscape is a maximization problem."""
        return True

    def evaluate(self, solution: list[int]) -> float:
        """
        Evaluate fitness as average of all contribution functions.

        Args:
            solution: Bitstring as list of 0/1 integers.

        Returns:
            Average contribution (value in [0, 1]).
        """
        total = 0.0

        for i in range(self.n):
            # Get the relevant bits: bit i and its K adjacent neighbors
            neighbors = self._get_adjacent_neighbors(i)
            key_bits = [solution[i]] + [solution[j] for j in neighbors]
            key = tuple(key_bits)
            total += self.contributions[i][key]

        return total / self.n

    def supports_delta_evaluation(self) -> bool:
        """NK Landscape does not support efficient delta evaluation."""
        return False
