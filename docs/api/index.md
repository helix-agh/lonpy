# API Reference

Complete API documentation for lonpy.

## Modules

lonpy is organized into modules for continuous and discrete optimization:

### [LON Module](lon.md)

Data structures for Local Optima Networks.

- [`LON`](lon.md#lonpy.lon.LON) - Local Optima Network representation
- [`CMLON`](lon.md#lonpy.lon.CMLON) - Compressed Monotonic LON

### [Sampling Module](sampling.md)

Sampling algorithms for LON construction.

**Continuous Optimization:**

- [`compute_lon()`](sampling.md#lonpy.continuous.sampling.compute_lon) - High-level convenience function
- [`BasinHoppingSampler`](sampling.md#lonpy.continuous.sampling.BasinHoppingSampler) - Basin-Hopping sampler
- [`BasinHoppingSamplerConfig`](sampling.md#lonpy.continuous.sampling.BasinHoppingSamplerConfig) - Configuration

**Discrete Optimization:**

- [`compute_discrete_lon()`](sampling.md#lonpy.discrete.sampling.compute_discrete_lon) - High-level convenience function
- [`ILSSampler`](sampling.md#lonpy.discrete.sampling.ILSSampler) - Iterated Local Search sampler
- [`ILSSamplerConfig`](sampling.md#lonpy.discrete.sampling.ILSSamplerConfig) - Configuration

**Built-in Problems:**

- [`OneMax`](sampling.md#lonpy.problems.discrete.OneMax) - Maximize 1s in bitstring
- [`Knapsack`](sampling.md#lonpy.problems.discrete.Knapsack) - 0/1 Knapsack problem
- [`NumberPartitioning`](sampling.md#lonpy.problems.discrete.NumberPartitioning) - Number partitioning problem

### [Visualization Module](visualization.md)

Plotting and animation tools.

- [`LONVisualizer`](visualization.md#lonpy.visualization.LONVisualizer) - Visualization class

## Quick Reference

### Creating a LON (Continuous)

```python
from lonpy import compute_lon

lon = compute_lon(
    func=objective_function,
    dim=2,
    lower_bound=-5.0,
    upper_bound=5.0,
    n_runs=20,
    seed=42
)
```

### Creating a LON (Discrete)

```python
from lonpy import compute_discrete_lon, OneMax

problem = OneMax(n=20)
lon = compute_discrete_lon(problem, n_runs=100, seed=42)
```

### Analyzing a LON

```python
# Basic properties
print(f"Optima: {lon.n_vertices}")
print(f"Edges: {lon.n_edges}")
print(f"Best: {lon.best_fitness}")

# Compute metrics
metrics = lon.compute_metrics()

# Convert to CMLON
cmlon = lon.to_cmlon()
cmlon_metrics = cmlon.compute_metrics()
```

### Visualizing a LON

```python
from lonpy import LONVisualizer

viz = LONVisualizer()

# 2D plot
viz.plot_2d(lon, output_path="lon.png")

# 3D plot
viz.plot_3d(lon, output_path="lon_3d.png")

# Animation
viz.create_rotation_gif(lon, output_path="lon.gif")

# All visualizations
viz.visualize_all(lon, output_folder="./output")
```

## Dependencies

lonpy depends on:

- `numpy` - Numerical computations
- `scipy` - Optimization (continuous)
- `pandas` - Data handling
- `igraph` - Graph operations
- `matplotlib` - 2D plotting
- `plotly` - 3D plotting
- `imageio` - GIF creation
- `kaleido` - Static image export
