from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

RepresentationType = Literal["bitstring", "permutation"]


@dataclass
class Solution:
    """
    Solution representation for discrete optimization problems.

    Supports bitstring and permutation representations.

    Attributes:
        data: The solution data (list of integers).
        fitness: Current fitness value (None if not evaluated).
        representation: Type of representation ("bitstring" or "permutation").

    Example:
        >>> sol = Solution.random_bitstring(n=10)
        >>> sol.data
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]  # random
        >>> sol.fitness = 5.0
    """

    data: list[int] = field(default_factory=list)
    fitness: float | None = None
    representation: RepresentationType = "bitstring"

    @property
    def n(self) -> int:
        """Length of the solution."""
        return len(self.data)

    def copy(self) -> Solution:
        """Create a deep copy of this solution."""
        return Solution(
            data=self.data.copy(),
            fitness=self.fitness,
            representation=self.representation,
        )

    def to_hash(self) -> str:
        """
        Create a hash string identifying this solution.

        Returns:
            String representation suitable for use as a node identifier.
        """
        return "_".join(str(x) for x in self.data)

    def flip(self, index: int) -> None:
        """
        Flip bit at index (for bitstring representation).

        Args:
            index: Index of bit to flip.
        """
        self.data[index] = 1 - self.data[index]
        self.fitness = None  # Invalidate fitness

    def swap(self, i: int, j: int) -> None:
        """
        Swap elements at indices i and j (for permutation representation).

        Args:
            i: First index.
            j: Second index.
        """
        self.data[i], self.data[j] = self.data[j], self.data[i]
        self.fitness = None  # Invalidate fitness

    @classmethod
    def random_bitstring(cls, n: int, rng: random.Random | None = None) -> Solution:
        """
        Create a random bitstring solution.

        Args:
            n: Length of bitstring.
            rng: Random number generator (uses global random if None).

        Returns:
            Solution with random 0/1 values.
        """
        if rng is None:
            data = [random.randint(0, 1) for _ in range(n)]
        else:
            data = [rng.randint(0, 1) for _ in range(n)]
        return cls(data=data, representation="bitstring")

    @classmethod
    def random_permutation(cls, n: int, rng: random.Random | None = None) -> Solution:
        """
        Create a random permutation solution.

        Args:
            n: Length of permutation (values 0 to n-1).
            rng: Random number generator (uses global random if None).

        Returns:
            Solution with random permutation of [0, 1, ..., n-1].
        """
        data = list(range(n))
        if rng is None:
            random.shuffle(data)
        else:
            rng.shuffle(data)
        return cls(data=data, representation="permutation")

    @classmethod
    def from_list(
        cls, data: list[int], representation: RepresentationType = "bitstring"
    ) -> Solution:
        """
        Create a solution from a list.

        Args:
            data: Solution data.
            representation: Type of representation.

        Returns:
            Solution instance.
        """
        return cls(data=data.copy(), representation=representation)

    def __eq__(self, other: object) -> bool:
        """Check equality based on data."""
        if not isinstance(other, Solution):
            return False
        return self.data == other.data

    def __hash__(self) -> int:
        """Hash based on data tuple."""
        return hash(tuple(self.data))

    def __repr__(self) -> str:
        """String representation."""
        data_str = "".join(str(x) for x in self.data) if self.n <= 20 else f"[{self.n} elements]"
        fit_str = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Solution({data_str}, fitness={fit_str})"
