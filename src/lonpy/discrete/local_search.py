import random

from lonpy.discrete.neighborhoods import Neighborhood
from lonpy.discrete.solution import Solution
from lonpy.problems.base import ProblemInstance


def hill_climb(
    solution: Solution,
    problem: ProblemInstance,
    neighborhood: Neighborhood,
    first_improvement: bool = True,
    rng: random.Random | None = None,
) -> Solution:
    """
    Perform hill climbing from a starting solution until a local optimum.

    Args:
        solution: Starting solution.
        problem: Problem instance providing evaluation.
        neighborhood: Neighborhood operator.
        first_improvement: If True, accept first improving neighbor.
            If False, evaluate all neighbors and select best (best improvement).
        rng: Random number generator for shuffling neighbor order.

    Returns:
        Local optimum solution.
    """
    if rng is None:
        rng = random.Random()

    current = solution.copy()
    if current.fitness is None:
        current.fitness = problem.evaluate(current.data)

    improved = True
    while improved:
        improved = False
        best_neighbor: Solution | None = None
        best_fitness = current.fitness

        indices = neighborhood.get_neighbor_indices(current)
        if first_improvement:
            rng.shuffle(indices)

        for idx in indices:
            # Use delta evaluation if available (for FlipNeighborhood)
            if hasattr(neighborhood, "evaluate_neighbor_with_delta"):
                neighbor_fitness = neighborhood.evaluate_neighbor_with_delta(current, idx, problem)
                neighbor = None  # Lazy creation
            else:
                neighbor = neighborhood.apply_move(current, idx)
                neighbor.fitness = problem.evaluate(neighbor.data)
                neighbor_fitness = neighbor.fitness

            if problem.strictly_better(neighbor_fitness, best_fitness):
                if first_improvement:
                    if neighbor is None:
                        neighbor = neighborhood.apply_move(current, idx)
                        neighbor.fitness = neighbor_fitness
                    current = neighbor
                    improved = True
                    break
                else:
                    if neighbor is None:
                        neighbor = neighborhood.apply_move(current, idx)
                        neighbor.fitness = neighbor_fitness
                    best_neighbor = neighbor
                    best_fitness = neighbor_fitness

        if not first_improvement and best_neighbor is not None:
            current = best_neighbor
            improved = True

    return current
