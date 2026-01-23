"""Reproduce Figure 4 from the paper:

LON visualizations for the three benchmark functions with n = 5 variables.

Jason Adair, Gabriela Ochoa, and Katherine M. Malan. 2019.
Local optima networks for continuous fitness landscapes.
In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '19).
Association for Computing Machinery, New York, NY, USA, 1407-1414.
https://doi.org/10.1145/3319619.3326852
"""

from collections.abc import Callable

from benchmark_utils import BENCHMARKS, STEP_SIZES

from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig, LONVisualizer

# Experiment parameters from the paper
DIM = 5
N_RUNS = 100
N_ITERATIONS = 1000
STEP_MULTIPLIER = 2  # Use 2 * STEP_SIZE


def create_lon_for_benchmark(
    name: str,
    func: Callable,
    bounds: tuple[float, float],
    seed: int = 42,
):
    step_size = STEP_MULTIPLIER * STEP_SIZES[DIM][name]
    domain = [(bounds[0], bounds[1])] * DIM

    config = BasinHoppingSamplerConfig(
        n_runs=N_RUNS,
        n_iterations=N_ITERATIONS,
        step_mode="fixed",
        step_size=step_size,
        hash_digits=5,
        seed=seed,
        minimizer_options={
            "options": {"ftol": 1e-7, "gtol": 0, "maxiter": 15000},
        },
    )

    sampler = BasinHoppingSampler(config)

    def progress(run: int, total: int) -> None:
        print(f"  {name}: Run {run}/{total}", end="\r")

    lon = sampler.sample_to_lon(func, domain, progress_callback=progress)
    print(f"  {name}: Completed {N_RUNS} runs")

    return lon


def run_experiment(seed: int = 42, save_path: str | None = None) -> dict:
    """Run the full experiment and create LON visualizations."""
    results = {}

    print(f"Running experiment with dim={DIM}, runs={N_RUNS}, iterations={N_ITERATIONS}")
    print(f"Step multiplier: {STEP_MULTIPLIER}x")

    for name, func, bounds in BENCHMARKS:
        step_size = STEP_MULTIPLIER * STEP_SIZES[DIM][name]
        print(f"Processing {name} (step_size={step_size:.4f}):")

        lon = create_lon_for_benchmark(name, func, bounds, seed=seed)
        cmlon = lon.to_cmlon()

        lon_metrics = lon.compute_metrics()
        cmlon_metrics = cmlon.compute_metrics()

        results[name] = {
            "lon": lon,
            "cmlon": cmlon,
            "lon_metrics": lon_metrics,
            "cmlon_metrics": cmlon_metrics,
        }

        print(f"  LON:   {lon_metrics['n_optima']} optima, {lon_metrics['n_funnels']} funnels")
        print(f"  CMLON: {cmlon_metrics['n_optima']} optima, {cmlon_metrics['n_funnels']} funnels")
        print()

    # Create visualizations
    if save_path:
        print("Creating visualizations...")
        visualizer = LONVisualizer()

        for name, data in results.items():
            cmlon = data["cmlon"]
            output_file = f"{save_path}_{name.lower()}_cmlon_3d.png"
            visualizer.plot_3d(cmlon, output_path=output_file, seed=seed)
            print(f"  Saved: {output_file}")

        print()

    return results


def print_summary(results: dict) -> None:
    print("=" * 70)
    print("Summary of LON Metrics")
    print("=" * 70)
    print(f"{'Function':<15} {'n_optima':>10} {'n_funnels':>10} {'n_global':>10} {'strength':>10}")
    print("-" * 70)

    for name in ["Ackley", "Rastrigin", "Birastrigin"]:
        m = results[name]["cmlon_metrics"]
        print(
            f"{name:<15} {m['n_optima']:>10} {m['n_funnels']:>10} "
            f"{m['n_global_funnels']:>10} {m['strength']:>10.4f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    results = run_experiment(seed=42, save_path="figure_4")
    print_summary(results)
