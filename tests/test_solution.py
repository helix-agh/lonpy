import random

from lonpy.discrete.solution import Solution


class TestSolutionBasics:
    def test_n_property(self):
        solution = Solution(data=[0, 1, 0, 1, 1])

        assert solution.n == 5

    def test_n_property_empty(self):
        solution = Solution(data=[])

        assert solution.n == 0


class TestSolutionCopy:
    def test_copy_preserves_data(self):
        original = Solution(data=[0, 1, 0], fitness=2.0, representation="permutation")

        copied = original.copy()

        assert copied is not original
        assert copied.data == original.data
        assert copied.fitness == original.fitness
        assert copied.representation == original.representation

    def test_copy_is_deep(self):
        original = Solution(data=[0, 1, 0], fitness=2.0)

        copied = original.copy()
        copied.data[0] = 1
        copied.fitness = 5.0

        assert original.data == [0, 1, 0]
        assert original.fitness == 2.0


class TestSolutionToHash:
    def test_to_hash_bitstring(self):
        solution = Solution(data=[0, 1, 0, 1, 1])

        hash_str = solution.to_hash()

        assert hash_str == "0_1_0_1_1"

    def test_to_hash_permutation(self):
        solution = Solution(data=[3, 1, 4, 0, 2], representation="permutation")

        hash_str = solution.to_hash()

        assert hash_str == "3_1_4_0_2"


class TestSolutionFlip:
    def test_flip_zero_to_one(self):
        solution = Solution(data=[0, 0, 0])

        solution.flip(1)

        assert solution.data == [0, 1, 0]

    def test_flip_first_index(self):
        solution = Solution(data=[0, 1, 1])

        solution.flip(0)

        assert solution.data[0] == 1
        assert solution.data == [1, 1, 1]
        assert solution.fitness is None


class TestSolutionSwap:
    def test_swap_elements(self):
        solution = Solution(data=[0, 1, 2, 3], representation="permutation")

        solution.swap(0, 3)

        assert solution.data == [3, 1, 2, 0]

    def test_swap_adjacent_elements(self):
        solution = Solution(data=[0, 1, 2], representation="permutation")

        solution.swap(0, 1)

        assert solution.data == [1, 0, 2]

    def test_swap_invalidates_fitness(self):
        solution = Solution(data=[0, 1, 2], fitness=10.0, representation="permutation")

        solution.swap(0, 2)

        assert solution.fitness is None

    def test_swap_same_index(self):
        solution = Solution(data=[0, 1, 2], representation="permutation")

        solution.swap(1, 1)

        assert solution.data == [0, 1, 2]


class TestSolutionRandomBitstring:
    def test_random_bitstring_length(self):
        solution = Solution.random_bitstring(n=10)

        assert solution.n == 10

    def test_random_bitstring_representation(self):
        solution = Solution.random_bitstring(n=5)

        assert solution.representation == "bitstring"

    def test_random_bitstring_values(self):
        solution = Solution.random_bitstring(n=100)

        assert all(x in [0, 1] for x in solution.data)

    def test_random_bitstring_fitness_is_none(self):
        solution = Solution.random_bitstring(n=10)

        assert solution.fitness is None

    def test_random_bitstring_with_rng(self):
        rng = random.Random(42)
        solution = Solution.random_bitstring(n=10, rng=rng)

        assert solution.n == 10

    def test_random_bitstring_deterministic_with_seed(self):
        rng1 = random.Random(123)
        solution1 = Solution.random_bitstring(n=20, rng=rng1)

        rng2 = random.Random(123)
        solution2 = Solution.random_bitstring(n=20, rng=rng2)

        assert solution1.data == solution2.data

    def test_random_bitstring_empty(self):
        solution = Solution.random_bitstring(n=0)

        assert solution.data == []
        assert solution.n == 0


class TestSolutionRandomPermutation:
    def test_random_permutation_length(self):
        solution = Solution.random_permutation(n=10)

        assert solution.n == 10

    def test_random_permutation_representation(self):
        solution = Solution.random_permutation(n=5)

        assert solution.representation == "permutation"

    def test_random_permutation_values(self):
        solution = Solution.random_permutation(n=10)

        assert sorted(solution.data) == list(range(10))

    def test_random_permutation_fitness_is_none(self):
        solution = Solution.random_permutation(n=10)

        assert solution.fitness is None

    def test_random_permutation_with_rng(self):
        rng = random.Random(42)
        solution = Solution.random_permutation(n=10, rng=rng)

        assert sorted(solution.data) == list(range(10))

    def test_random_permutation_deterministic_with_seed(self):
        rng1 = random.Random(456)
        solution1 = Solution.random_permutation(n=15, rng=rng1)

        rng2 = random.Random(456)
        solution2 = Solution.random_permutation(n=15, rng=rng2)

        assert solution1.data == solution2.data

    def test_random_permutation_empty(self):
        solution = Solution.random_permutation(n=0)

        assert solution.data == []
        assert solution.n == 0


class TestSolutionFromList:
    def test_from_list_bitstring(self):
        solution = Solution.from_list([0, 1, 0, 1])

        assert solution.data == [0, 1, 0, 1]
        assert solution.representation == "bitstring"

    def test_from_list_permutation(self):
        solution = Solution.from_list([3, 0, 2, 1], representation="permutation")

        assert solution.data == [3, 0, 2, 1]
        assert solution.representation == "permutation"

    def test_from_list_copies_data(self):
        original_data = [0, 1, 0]
        solution = Solution.from_list(original_data)

        original_data[0] = 1

        assert solution.data == [0, 1, 0]

    def test_from_list_fitness_is_none(self):
        solution = Solution.from_list([1, 0, 1])

        assert solution.fitness is None

    def test_from_list_empty(self):
        solution = Solution.from_list([])

        assert solution.data == []
        assert solution.n == 0


class TestSolutionEquality:
    def test_equal_solutions(self):
        solution1 = Solution(data=[0, 1, 0])
        solution2 = Solution(data=[0, 1, 0])

        assert solution1 == solution2

    def test_different_data(self):
        solution1 = Solution(data=[0, 1, 0])
        solution2 = Solution(data=[0, 0, 0])

        assert solution1 != solution2

    def test_different_fitness_still_equal(self):
        solution1 = Solution(data=[0, 1, 0], fitness=1.0)
        solution2 = Solution(data=[0, 1, 0], fitness=2.0)

        assert solution1 == solution2

    def test_different_representation_still_equal(self):
        solution1 = Solution(data=[0, 1, 2], representation="bitstring")
        solution2 = Solution(data=[0, 1, 2], representation="permutation")

        assert solution1 == solution2

    def test_not_equal_to_non_solution(self):
        solution = Solution(data=[0, 1, 0])

        assert solution != [0, 1, 0]
        assert solution != "0_1_0"
        assert solution is not None


class TestSolutionHash:
    def test_hash_equal_solutions(self):
        solution1 = Solution(data=[0, 1, 0])
        solution2 = Solution(data=[0, 1, 0])

        assert hash(solution1) == hash(solution2)

    def test_hash_usable_in_set(self):
        solution1 = Solution(data=[0, 1, 0])
        solution2 = Solution(data=[0, 1, 0])
        solution3 = Solution(data=[1, 1, 1])

        solution_set = {solution1, solution2, solution3}

        assert len(solution_set) == 2

    def test_hash_usable_as_dict_key(self):
        solution = Solution(data=[0, 1, 0])

        d = {solution: "test"}

        assert d[solution] == "test"


class TestSolutionRepr:
    def test_repr_short_bitstring(self):
        solution = Solution(data=[0, 1, 0, 1, 1], fitness=3.0)

        repr_str = repr(solution)

        assert "01011" in repr_str
        assert "3.0000" in repr_str

    def test_repr_no_fitness(self):
        solution = Solution(data=[0, 1, 0])

        repr_str = repr(solution)

        assert "fitness=None" in repr_str

    def test_repr_long_solution(self):
        solution = Solution(data=list(range(25)))

        repr_str = repr(solution)

        assert "25 elements" in repr_str

    def test_repr_exactly_20_elements(self):
        solution = Solution(data=[0] * 20)

        repr_str = repr(solution)

        assert "00000000000000000000" in repr_str
        assert "elements" not in repr_str

    def test_repr_permutation(self):
        solution = Solution(data=[3, 1, 4, 0, 2], representation="permutation")

        repr_str = repr(solution)

        assert "31402" in repr_str


class TestSolutionIntegration:
    def test_copy_then_flip(self):
        original = Solution.from_list([0, 0, 0])
        original.fitness = 0.0

        copied = original.copy()
        copied.flip(0)

        assert original.data == [0, 0, 0]
        assert original.fitness == 0.0
        assert copied.data == [1, 0, 0]
        assert copied.fitness is None

    def test_copy_then_swap(self):
        original = Solution.from_list([0, 1, 2], representation="permutation")
        original.fitness = 10.0

        copied = original.copy()
        copied.swap(0, 2)

        assert original.data == [0, 1, 2]
        assert original.fitness == 10.0
        assert copied.data == [2, 1, 0]
        assert copied.fitness is None

    def test_hash_consistent_after_operations(self):
        solution = Solution.from_list([0, 1, 0])
        initial_hash = solution.to_hash()

        solution.flip(0)
        after_flip_hash = solution.to_hash()

        assert initial_hash == "0_1_0"
        assert after_flip_hash == "1_1_0"
