# Quick Start

This guide will get you up and running with lonpy in just a few minutes.

## Your First LON

Let's create a Local Optima Network for the classic Rastrigin function:

```python
import numpy as np
from lonpy import compute_lon, LONVisualizer

# 1. Define your objective function
def rastrigin(x):
    """The Rastrigin function - highly multimodal."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 2. Build the LON
lon = compute_lon(
    func=rastrigin,      # Your objective function
    dim=2,               # Number of dimensions
    lower_bound=-5.12,   # Search space lower bound
    upper_bound=5.12,    # Search space upper bound
    n_runs=20,           # Number of Basin-Hopping runs
    max_perturbations_without_improvement=500,  # Stop after this many consecutive non-improving perturbations
    seed=42              # For reproducibility
)

# 3. Explore the results
print(f"Local optima found: {lon.n_vertices}")
print(f"Transitions recorded: {lon.n_edges}")
```

## Analyzing the Landscape

lonpy computes useful metrics about your fitness landscape:

```python
# Get landscape metrics
metrics = lon.compute_metrics()

print(f"Number of optima: {metrics['n_optima']}")
print(f"Number of funnels: {metrics['n_funnels']}")
print(f"Global funnels: {metrics['n_global_funnels']}")
print(f"Neutrality: {metrics['neutral']:.1%}")
print(f"Global strength: {metrics['global_strength']:.1%}")
print(f"Sink strength: {metrics['sink_strength']:.1%}")
```

**What do these metrics mean?**

| Metric | Description |
|--------|-------------|
| `n_optima` | Total number of local optima discovered |
| `n_funnels` | Number of sink nodes (basins of attraction) |
| `n_global_funnels` | Funnels leading to the global optimum |
| `neutral` | Proportion of nodes with equal-fitness neighbors |
| `global_strength` | Proportion of global optima incoming strength to total incoming strength |
| `sink_strength` | Proportion of global sinks incoming strength to all sinks incoming strength |
| `success` | Proportion of runs that reached the global optimum |
| `deviation` | Mean absolute deviation from the global optimum |

## Visualizing the Network

### 2D Network Plot

```python
viz = LONVisualizer()

# Create a 2D visualization
fig = viz.plot_2d(lon, output_path="lon_2d.png")
```

In the visualization:

- **Red nodes**: Global optima
- **Pink nodes**: Local optima
- **Edges**: Transitions between optima (thicker = more frequent)

### 3D Landscape View

The 3D view shows fitness on the Z-axis:

```python
fig = viz.plot_3d(lon, output_path="lon_3d.png")
```

### Animated Rotation

Create a rotating GIF to explore the landscape:

```python
viz.create_rotation_gif(
    lon,
    output_path="lon_rotation.gif",
    duration=10,  # seconds
    fps=30
)
```

## Using CMLON

Compressed Monotonic LONs provide a cleaner view by merging equal-fitness nodes:

```python
# Convert to CMLON
cmlon = lon.to_cmlon()

# CMLON has additional metrics
cmlon_metrics = cmlon.compute_metrics()
print(f"Compression: {cmlon_metrics['neutral']:.1%} of nodes merged")
print(f"Global funnel proportion: {cmlon_metrics['global_funnel_proportion']:.1%}")

# Visualize CMLON
viz.plot_2d(cmlon, output_path="cmlon.png")
```

In CMLON visualizations:

- **Red nodes**: Global optima
- **Blue nodes**: Local (suboptimal) sinks
- **Pink nodes**: In global funnel
- **Light blue nodes**: In local funnels

## Complete Example

Here's a full script that generates all visualizations:

```python
import numpy as np
from lonpy import compute_lon, LONVisualizer

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Build LON
lon = compute_lon(
    rastrigin,
    dim=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    n_runs=30,
    max_perturbations_without_improvement=500,
    seed=42
)

# Print analysis
print("=== LON Analysis ===")
print(f"Vertices: {lon.n_vertices}")
print(f"Edges: {lon.n_edges}")

metrics = lon.compute_metrics()
for key, value in metrics.items():
    print(f"{key}: {value}")

# Generate all visualizations
viz = LONVisualizer()
outputs = viz.visualize_all(
    lon,
    output_folder="./output",
    seed=42
)

print("\n=== Generated Files ===")
for name, path in outputs.items():
    print(f"{name}: {path}")
```

## Next Steps

- [Core Concepts](concepts.md) - Understand LON theory
- [Sampling Guide](../user-guide/sampling.md) - Configure Basin-Hopping
- [API Reference](../api/index.md) - Full API documentation
