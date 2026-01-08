import random

import pytest

from lonpy.discrete.neighborhoods import FlipNeighborhood, SwapNeighborhood
from lonpy.discrete.solution import Solution


class TestFlipNeighborhood:
    def test_get_neighbor_indices(self):
        solution = Solution.from_list([0, 1, 0, 1, 1])
        neighborhood = FlipNeighborhood()

        indices = neighborhood.get_neighbor_indices(solution)

        assert indices == [0, 1, 2, 3, 4]

    def test_get_neighbor_indices_empty_solution(self):
        solution = Solution.from_list([])
        neighborhood = FlipNeighborhood()

        indices = neighborhood.get_neighbor_indices(solution)

        assert indices == []

    def test_apply_move_flips_bit(self):
        solution = Solution.from_list([0, 1, 0, 1, 1])
        neighborhood = FlipNeighborhood()

        neighbor = neighborhood.apply_move(solution, 0)

        assert neighbor.data == [1, 1, 0, 1, 1]
        assert solution.data == [0, 1, 0, 1, 1]  # Original unchanged

    def test_apply_move_flips_one_to_zero(self):
        solution = Solution.from_list([0, 1, 0, 1, 1])
        neighborhood = FlipNeighborhood()

        neighbor = neighborhood.apply_move(solution, 1)

        assert neighbor.data == [0, 0, 0, 1, 1]

    def test_apply_move_invalidates_fitness(self):
        solution = Solution.from_list([0, 1, 0])
        solution.fitness = 5.0
        neighborhood = FlipNeighborhood()

        neighbor = neighborhood.apply_move(solution, 0)

        assert neighbor.fitness is None
        assert solution.fitness == 5.0  # Original unchanged

    def test_apply_move_rejects_tuple_index(self):
        solution = Solution.from_list([0, 1, 0])
        neighborhood = FlipNeighborhood()

        with pytest.raises(TypeError, match="expects int index"):
            neighborhood.apply_move(solution, (0, 1))

    def test_apply_random_perturbation_flips_correct_count(self):
        solution = Solution.from_list([0, 0, 0, 0, 0])
        neighborhood = FlipNeighborhood()
        rng = random.Random(42)

        perturbed = neighborhood.apply_random_perturbation(solution, strength=3, rng=rng)

        # Count how many bits differ
        differences = sum(a != b for a, b in zip(solution.data, perturbed.data, strict=True))
        assert differences == 3

    def test_apply_random_perturbation_caps_at_solution_length(self):
        solution = Solution.from_list([0, 0, 0])
        neighborhood = FlipNeighborhood()
        rng = random.Random(42)

        perturbed = neighborhood.apply_random_perturbation(solution, strength=10, rng=rng)

        differences = sum(a != b for a, b in zip(solution.data, perturbed.data, strict=True))
        assert differences == 3  # Capped at solution length

    def test_apply_random_perturbation_preserves_original(self):
        solution = Solution.from_list([0, 1, 0, 1])
        original_data = solution.data.copy()
        neighborhood = FlipNeighborhood()
        rng = random.Random(42)

        neighborhood.apply_random_perturbation(solution, strength=2, rng=rng)

        assert solution.data == original_data

    def test_apply_random_perturbation_deterministic_with_seed(self):
        solution = Solution.from_list([0, 1, 0, 1, 0])
        neighborhood = FlipNeighborhood()

        rng1 = random.Random(123)
        perturbed1 = neighborhood.apply_random_perturbation(solution, strength=2, rng=rng1)

        rng2 = random.Random(123)
        perturbed2 = neighborhood.apply_random_perturbation(solution, strength=2, rng=rng2)

        assert perturbed1.data == perturbed2.data


class TestSwapNeighborhood:
    def test_get_neighbor_indices(self):
        solution = Solution.from_list([0, 1, 2, 3], representation="permutation")
        neighborhood = SwapNeighborhood()

        indices = neighborhood.get_neighbor_indices(solution)

        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert indices == expected

    def test_get_neighbor_indices_count(self):
        solution = Solution.from_list([0, 1, 2, 3, 4], representation="permutation")
        neighborhood = SwapNeighborhood()

        indices = neighborhood.get_neighbor_indices(solution)

        n = 5
        expected_count = n * (n - 1) // 2
        assert len(indices) == expected_count

    def test_get_neighbor_indices_single_element(self):
        solution = Solution.from_list([0], representation="permutation")
        neighborhood = SwapNeighborhood()

        indices = neighborhood.get_neighbor_indices(solution)

        assert indices == []

    def test_apply_move_swaps_elements(self):
        solution = Solution.from_list([0, 1, 2, 3], representation="permutation")
        neighborhood = SwapNeighborhood()

        neighbor = neighborhood.apply_move(solution, (0, 2))

        assert neighbor.data == [2, 1, 0, 3]
        assert solution.data == [0, 1, 2, 3]  # Original unchanged

    def test_apply_move_invalidates_fitness(self):
        solution = Solution.from_list([0, 1, 2], representation="permutation")
        solution.fitness = 10.0
        neighborhood = SwapNeighborhood()

        neighbor = neighborhood.apply_move(solution, (0, 1))

        assert neighbor.fitness is None
        assert solution.fitness == 10.0  # Original unchanged

    def test_apply_move_rejects_int_index(self):
        solution = Solution.from_list([0, 1, 2], representation="permutation")
        neighborhood = SwapNeighborhood()

        with pytest.raises(TypeError, match="expects tuple index"):
            neighborhood.apply_move(solution, 0)

    def test_apply_random_perturbation_performs_swaps(self):
        solution = Solution.from_list([0, 1, 2, 3, 4], representation="permutation")
        neighborhood = SwapNeighborhood()
        rng = random.Random(42)

        perturbed = neighborhood.apply_random_perturbation(solution, strength=2, rng=rng)

        # Verify it's still a valid permutation
        assert sorted(perturbed.data) == [0, 1, 2, 3, 4]
        # Verify something changed
        assert perturbed.data != solution.data

    def test_apply_random_perturbation_preserves_original(self):
        solution = Solution.from_list([0, 1, 2, 3], representation="permutation")
        original_data = solution.data.copy()
        neighborhood = SwapNeighborhood()
        rng = random.Random(42)

        neighborhood.apply_random_perturbation(solution, strength=2, rng=rng)

        assert solution.data == original_data

    def test_apply_random_perturbation_deterministic_with_seed(self):
        solution = Solution.from_list([0, 1, 2, 3, 4], representation="permutation")
        neighborhood = SwapNeighborhood()

        rng1 = random.Random(999)
        perturbed1 = neighborhood.apply_random_perturbation(solution, strength=3, rng=rng1)

        rng2 = random.Random(999)
        perturbed2 = neighborhood.apply_random_perturbation(solution, strength=3, rng=rng2)

        assert perturbed1.data == perturbed2.data

    def test_apply_random_perturbation_maintains_permutation(self):
        solution = Solution.from_list([4, 2, 0, 1, 3], representation="permutation")
        neighborhood = SwapNeighborhood()
        rng = random.Random(42)

        for _ in range(10):
            perturbed = neighborhood.apply_random_perturbation(solution, strength=5, rng=rng)
            assert sorted(perturbed.data) == [0, 1, 2, 3, 4]
