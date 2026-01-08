# lonpy

**Local Optima Networks for Continuous and Discrete Optimization**

![lonpy](assets/icon.png){ width="100%" }

lonpy is a Python library for constructing, analyzing, and visualizing Local Optima Networks (LONs) for both continuous and discrete optimization problems.

## What are Local Optima Networks?

Local Optima Networks (LONs) are graph-based models that capture the global structure of fitness landscapes. They help researchers and practitioners understand:

- **Landscape topology**: How local optima are distributed and connected
- **Search difficulty**: Whether the landscape has a single funnel or multiple competing basins
- **Algorithm behavior**: How optimization algorithms navigate between local optima

## Key Features

<div class="grid cards" markdown>

- **Continuous Optimization**

    ---

    Basin-Hopping sampling for continuous fitness landscapes with configurable perturbation strategies

- **Discrete Optimization**

    ---

    Iterated Local Search (ILS) sampling for combinatorial problems like OneMax, Knapsack, and Number Partitioning

- **Built-in Problems**

    ---

    Ready-to-use problem instances: OneMax, Knapsack, Number Partitioning, and support for custom problems

- **LON & CMLON Support**

    ---

    Both standard LON and Compressed Monotonic LON representations for landscape analysis

- **Rich Metrics**

    ---

    Compute landscape metrics including funnel analysis, neutrality measures, and global optima strength

- **Beautiful Visualizations**

    ---

    2D and 3D plots with support for animated GIFs showing the landscape structure

</div>

## Quick Example

=== "Continuous"

    ```python
    import numpy as np
    from lonpy import compute_lon, LONVisualizer

    # Define the Rastrigin function
    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    # Build the LON
    lon = compute_lon(
        rastrigin,
        dim=2,
        lower_bound=-5.12,
        upper_bound=5.12,
        n_runs=20,
        seed=42
    )

    # Analyze
    metrics = lon.compute_metrics()
    print(f"Found {lon.n_vertices} local optima")
    print(f"Landscape has {metrics['n_funnels']} funnels")

    # Visualize
    viz = LONVisualizer()
    viz.plot_3d(lon, output_path="landscape.png")
    ```

=== "Discrete"

    ```python
    from lonpy import compute_discrete_lon, OneMax, Knapsack

    # OneMax problem
    problem = OneMax(n=20)
    lon = compute_discrete_lon(problem, n_runs=100, seed=42)

    # Analyze
    metrics = lon.compute_metrics()
    print(f"Found {lon.n_vertices} local optima")
    print(f"Landscape has {metrics['n_funnels']} funnels")

    # Knapsack problem
    knapsack = Knapsack(
        values=[60, 100, 120, 80, 90],
        weights=[10, 20, 30, 15, 25],
        capacity=50
    )
    lon = compute_discrete_lon(knapsack, n_runs=100, seed=42)
    ```

## Installation

```bash
pip install lonpy
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start](getting-started/quickstart.md) - Get up and running in 5 minutes
- [Core Concepts](getting-started/concepts.md) - Understand LONs and fitness landscapes
- [API Reference](api/index.md) - Complete API documentation
