from lonpy.problems.discrete import Knapsack, NumberPartitioning, OneMax


class TestOneMax:
    def test_maximize_returns_true(self):
        problem = OneMax(n=10)

        assert problem.maximize() is True

    def test_evaluate_all_zeros(self):
        problem = OneMax(n=5)
        solution = [0, 0, 0, 0, 0]

        assert problem.evaluate(solution) == 0.0

    def test_evaluate_mixed(self):
        problem = OneMax(n=5)
        solution = [1, 0, 1, 0, 1]

        assert problem.evaluate(solution) == 3.0

    def test_flip_delta_zero_to_one(self):
        problem = OneMax(n=5)
        solution = [0, 0, 0, 0, 0]

        delta = problem.flip_delta(solution, 2)

        assert delta == 1.0

    def test_supports_delta_evaluation(self):
        problem = OneMax(n=10)

        assert problem.supports_delta_evaluation() is True

    def test_strictly_better_maximization(self):
        problem = OneMax(n=10)

        assert problem.strictly_better(5.0, 3.0) is True
        assert problem.strictly_better(3.0, 5.0) is False
        assert problem.strictly_better(5.0, 5.0) is False

    def test_better_or_equal_maximization(self):
        problem = OneMax(n=10)

        assert problem.better_or_equal(5.0, 3.0) is True
        assert problem.better_or_equal(5.0, 5.0) is True
        assert problem.better_or_equal(3.0, 5.0) is False

    def test_compare_maximization(self):
        problem = OneMax(n=10)

        assert problem.compare(5.0, 3.0) == 1
        assert problem.compare(3.0, 5.0) == -1
        assert problem.compare(5.0, 5.0) == 0


class TestKnapsack:
    def test_maximize_returns_true(self):
        problem = Knapsack(values=[10, 20], weights=[5, 10], capacity=15)

        assert problem.maximize() is True

    def test_n_property(self):
        problem = Knapsack(values=[10, 20, 30], weights=[5, 10, 15], capacity=20)

        assert problem.n == 3

    def test_evaluate_feasible_solution(self):
        problem = Knapsack(
            values=[60.0, 100.0, 120.0],
            weights=[10.0, 20.0, 30.0],
            capacity=50.0,
        )
        solution = [1, 1, 0]  # Take items 0 and 1, weight = 30

        assert problem.evaluate(solution) == 160.0

    def test_evaluate_infeasible_solution(self):
        problem = Knapsack(
            values=[60.0, 100.0, 120.0],
            weights=[10.0, 20.0, 30.0],
            capacity=50.0,
        )
        solution = [1, 1, 1]  # Total weight = 60 > capacity

        assert problem.evaluate(solution) == 0.0

    def test_evaluate_empty_selection(self):
        problem = Knapsack(
            values=[60.0, 100.0, 120.0],
            weights=[10.0, 20.0, 30.0],
            capacity=50.0,
        )
        solution = [0, 0, 0]

        assert problem.evaluate(solution) == 0.0

    def test_supports_delta_evaluation(self):
        problem = Knapsack(values=[10], weights=[5], capacity=10)

        assert problem.supports_delta_evaluation() is False


class TestNumberPartitioning:
    def test_maximize_returns_false(self):
        problem = NumberPartitioning(n=10, k=0.5, seed=42)

        assert problem.maximize() is False

    def test_items_count_equals_n(self):
        problem = NumberPartitioning(n=15, k=0.5, seed=1)

        assert len(problem.items) == 15

    def test_items_are_positive(self):
        problem = NumberPartitioning(n=20, k=0.5, seed=42)

        assert all(item >= 1 for item in problem.items)
        assert all(isinstance(item, int) for item in problem.items)

    def test_evaluate_perfect_partition(self):
        # Create problem with known items
        problem = NumberPartitioning(n=4, k=0.5, seed=1)
        problem.items = [10, 10, 5, 5]  # Override items

        # [0,1,0,1] -> A={10,5}, B={10,5} -> |15-15| = 0
        solution = [0, 1, 0, 1]

        assert problem.evaluate(solution) == 0.0

    def test_evaluate_partial_partition(self):
        problem = NumberPartitioning(n=4, k=0.5, seed=1)
        problem.items = [10, 20, 5, 15]  # Override items

        # [1,0,1,0] -> A={20,15}=35, B={10,5}=15 -> |35-15| = 20
        solution = [1, 0, 1, 0]

        assert problem.evaluate(solution) == 20.0

    def test_supports_delta_evaluation(self):
        problem = NumberPartitioning(n=10)

        assert problem.supports_delta_evaluation() is False

    def test_strictly_better_minimization(self):
        problem = NumberPartitioning(n=10)

        assert problem.strictly_better(3.0, 5.0) is True
        assert problem.strictly_better(5.0, 3.0) is False
        assert problem.strictly_better(5.0, 5.0) is False

    def test_better_or_equal_minimization(self):
        problem = NumberPartitioning(n=10)

        assert problem.better_or_equal(3.0, 5.0) is True
        assert problem.better_or_equal(5.0, 5.0) is True
        assert problem.better_or_equal(5.0, 3.0) is False

    def test_compare_minimization(self):
        problem = NumberPartitioning(n=10)

        assert problem.compare(3.0, 5.0) == 1
        assert problem.compare(5.0, 3.0) == -1
        assert problem.compare(5.0, 5.0) == 0
