# Examples

Complete examples demonstrating lonpy's capabilities.

## Continuous Optimization

### Basic LON Analysis

```python
import numpy as np
from lonpy import compute_lon, LONVisualizer

# Define the Rastrigin function
def rastrigin(x):
    """Highly multimodal test function."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Build the LON
lon = compute_lon(
    func=rastrigin,
    dim=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    n_runs=30,
    n_iterations=500,
    seed=42
)

# Analyze
print("=== LON Analysis ===")
print(f"Local optima: {lon.n_vertices}")
print(f"Transitions: {lon.n_edges}")
print(f"Best fitness: {lon.best_fitness}")

metrics = lon.compute_metrics()
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Visualize
viz = LONVisualizer()
viz.plot_2d(lon, output_path="rastrigin_lon.png", seed=42)
viz.plot_3d(lon, output_path="rastrigin_3d.png", seed=42)
```

### Comparing Multiple Functions

```python
import numpy as np
import pandas as pd
from lonpy import compute_lon

# Test functions
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

# Configuration
functions = {
    "Sphere": (sphere, -5.0, 5.0, 0.0),
    "Rastrigin": (rastrigin, -5.12, 5.12, 0.0),
    "Ackley": (ackley, -5.0, 5.0, 0.0),
    "Rosenbrock": (rosenbrock, -5.0, 10.0, 0.0),
}

# Analyze each
results = []
for name, (func, lb, ub, optimal) in functions.items():
    print(f"Analyzing {name}...")

    lon = compute_lon(
        func=func,
        dim=2,
        lower_bound=lb,
        upper_bound=ub,
        n_runs=30,
        n_iterations=500,
        seed=42
    )

    metrics = lon.compute_metrics(known_best=optimal * 10**4)  # scaled
    cmlon = lon.to_cmlon()
    cmlon_metrics = cmlon.compute_metrics(known_best=optimal * 10**4)

    results.append({
        "Function": name,
        "Optima": lon.n_vertices,
        "Funnels": metrics['n_funnels'],
        "Global Funnels": metrics['n_global_funnels'],
        "Strength": f"{metrics['strength']:.1%}",
        "Global Funnel %": f"{cmlon_metrics['global_funnel_proportion']:.1%}",
    })

# Display results
df = pd.DataFrame(results)
print("\n=== Comparison ===")
print(df.to_string(index=False))
```

### Custom Sampling Configuration

```python
import numpy as np
from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig, LONVisualizer

def schwefel(x):
    """Schwefel function - deceptive with distant global optimum."""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Custom configuration for challenging function
config = BasinHoppingSamplerConfig(
    n_runs=100,              # More runs for coverage
    n_iterations=300,        # Moderate depth
    step_mode="percentage",
    step_size=0.15,          # Larger steps for this landscape
    hash_digits=3,           # Coarser grouping
    bounded=True,
    minimizer_method="L-BFGS-B",
    minimizer_options={
        "ftol": 1e-08,
        "gtol": 1e-06,
    },
    seed=42
)

# Sample
domain = [(-500.0, 500.0), (-500.0, 500.0)]
sampler = BasinHoppingSampler(config)

def progress(run, total):
    if run % 10 == 0:
        print(f"Progress: {run}/{total}")

lon = sampler.sample_to_lon(schwefel, domain, progress_callback=progress)

# Analyze
print(f"\nFound {lon.n_vertices} local optima")
print(f"Best fitness: {lon.best_fitness}")

# Known optimum at x = (420.9687, ...) with f(x) ≈ 0
metrics = lon.compute_metrics()
print(f"Funnels: {metrics['n_funnels']}")
print(f"Strength: {metrics['strength']:.1%}")
```

## Discrete Optimization

### OneMax Analysis

```python
from lonpy import compute_discrete_lon, OneMax, LONVisualizer

# Build LON for OneMax
problem = OneMax(n=20)
lon = compute_discrete_lon(
    problem,
    n_runs=100,
    non_improvement_iterations=100,
    seed=42
)

# Analyze
print("=== OneMax LON Analysis ===")
print(f"Local optima: {lon.n_vertices}")
print(f"Transitions: {lon.n_edges}")
print(f"Best fitness: {lon.best_fitness}")

metrics = lon.compute_metrics()
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Convert to CMLON
cmlon = lon.to_cmlon()
cmlon_metrics = cmlon.compute_metrics()
print(f"\nCMLON vertices: {cmlon.n_vertices}")
print(f"Global funnel proportion: {cmlon_metrics['global_funnel_proportion']:.1%}")
```

### Knapsack Problem

```python
from lonpy import compute_discrete_lon, Knapsack, LONVisualizer

# Define a Knapsack instance
problem = Knapsack(
    values=[60, 100, 120, 80, 90, 70, 110, 95, 85, 75],
    weights=[10, 20, 30, 15, 25, 12, 28, 22, 18, 14],
    capacity=80
)

# Build LON
lon = compute_discrete_lon(
    problem,
    n_runs=200,
    non_improvement_iterations=150,
    perturbation_strength=2,
    seed=42
)

# Analyze
print("=== Knapsack LON Analysis ===")
print(f"Number of items: {problem.n}")
print(f"Local optima: {lon.n_vertices}")
print(f"Transitions: {lon.n_edges}")
print(f"Best fitness: {lon.best_fitness}")

metrics = lon.compute_metrics()
print(f"\nFunnels: {metrics['n_funnels']}")
print(f"Global funnels: {metrics['n_global_funnels']}")
print(f"Strength: {metrics['strength']:.1%}")

# CMLON analysis
cmlon = lon.to_cmlon()
cmlon_metrics = cmlon.compute_metrics()
print(f"\nCMLON vertices: {cmlon.n_vertices}")
print(f"Global funnel proportion: {cmlon_metrics['global_funnel_proportion']:.1%}")
```

### Number Partitioning

```python
from lonpy import compute_discrete_lon, NumberPartitioning

# Generate NPP instance
problem = NumberPartitioning(n=20, k=0.5, seed=42)

print(f"NPP instance with {problem.n} items")
print(f"Item values: {problem.items[:5]}... (first 5)")

# Build LON
lon = compute_discrete_lon(
    problem,
    n_runs=150,
    non_improvement_iterations=100,
    perturbation_strength=2,
    seed=42
)

# Analyze
print("\n=== NPP LON Analysis ===")
print(f"Local optima: {lon.n_vertices}")
print(f"Best fitness: {lon.best_fitness}")  # 0 = perfect partition

metrics = lon.compute_metrics()
print(f"Funnels: {metrics['n_funnels']}")
print(f"Neutrality: {metrics['neutral']:.1%}")

# Check if perfect partition exists
if lon.best_fitness == 0:
    print("\nPerfect partition found!")
else:
    print(f"\nBest partition difference: {lon.best_fitness}")
```

### Comparing Discrete Problems

```python
import pandas as pd
from lonpy import compute_discrete_lon, OneMax, Knapsack, NumberPartitioning

# Define problems
problems = {
    "OneMax-20": OneMax(n=20),
    "OneMax-30": OneMax(n=30),
    "Knapsack-10": Knapsack(
        values=[60, 100, 120, 80, 90, 70, 110, 95, 85, 75],
        weights=[10, 20, 30, 15, 25, 12, 28, 22, 18, 14],
        capacity=80
    ),
    "NPP-15": NumberPartitioning(n=15, k=0.5, seed=42),
    "NPP-20": NumberPartitioning(n=20, k=0.5, seed=42),
}

# Analyze each
results = []
for name, problem in problems.items():
    print(f"Analyzing {name}...")

    lon = compute_discrete_lon(
        problem,
        n_runs=100,
        non_improvement_iterations=100,
        seed=42
    )

    metrics = lon.compute_metrics()
    cmlon = lon.to_cmlon()
    cmlon_metrics = cmlon.compute_metrics()

    results.append({
        "Problem": name,
        "n": problem.n,
        "Optima": lon.n_vertices,
        "Funnels": metrics['n_funnels'],
        "Neutrality": f"{metrics['neutral']:.1%}",
        "Global Funnel %": f"{cmlon_metrics['global_funnel_proportion']:.1%}",
    })

# Display results
df = pd.DataFrame(results)
print("\n=== Comparison ===")
print(df.to_string(index=False))
```

### Custom ILS Configuration

```python
from lonpy import ILSSampler, ILSSamplerConfig, OneMax

# Custom configuration
config = ILSSamplerConfig(
    n_runs=200,
    max_iterations=0,               # No limit
    non_improvement_iterations=150, # Stop condition
    perturbation_strength=3,        # Larger perturbation
    first_improvement=True,         # Faster exploration
    representation="bitstring",
    neighborhood="flip",
    seed=42
)

sampler = ILSSampler(config)
problem = OneMax(n=30)

def progress(run, total):
    if run % 20 == 0:
        print(f"Progress: {run}/{total}")

lon = sampler.sample_to_lon(problem, progress_callback=progress)

print(f"\nFound {lon.n_vertices} local optima")
metrics = lon.compute_metrics()
print(f"Funnels: {metrics['n_funnels']}")
```

## CMLON Analysis

### Detailed CMLON Comparison

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
    n_runs=50,
    seed=42
)

# Convert to CMLON
cmlon = lon.to_cmlon()

print("=== LON vs CMLON ===")
print(f"LON vertices:   {lon.n_vertices}")
print(f"CMLON vertices: {cmlon.n_vertices}")
print(f"Compression:    {1 - cmlon.n_vertices/lon.n_vertices:.1%}")

print("\n=== CMLON Structure ===")
print(f"Total sinks:  {len(cmlon.get_sinks())}")
print(f"Global sinks: {len(cmlon.get_global_sinks())}")
print(f"Local sinks:  {len(cmlon.get_local_sinks())}")

# Metrics
cmlon_metrics = cmlon.compute_metrics()
print("\n=== CMLON Metrics ===")
print(f"Global funnel proportion: {cmlon_metrics['global_funnel_proportion']:.1%}")
print(f"Strength to global: {cmlon_metrics['strength']:.1%}")

# Visualize
viz = LONVisualizer()
viz.plot_2d(cmlon, output_path="cmlon_2d.png", seed=42)
viz.plot_3d(cmlon, output_path="cmlon_3d.png", seed=42)
```

## Working with the Graph

### Accessing Graph Properties

```python
import numpy as np
from lonpy import compute_lon

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

lon = compute_lon(rastrigin, dim=2, lower_bound=-5.12, upper_bound=5.12, n_runs=30, seed=42)

# Access igraph object
g = lon.graph

print("=== Graph Properties ===")
print(f"Vertices: {g.vcount()}")
print(f"Edges: {g.ecount()}")
print(f"Density: {g.density():.4f}")

# Find most visited nodes
counts = g.vs["Count"]
top_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:5]

print("\n=== Most Visited Optima ===")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. Node {g.vs[idx]['name']}: {counts[idx]} visits, fitness={g.vs[idx]['Fitness']}")

# Analyze edge weights
if g.ecount() > 0:
    edge_weights = g.es["Count"]
    print(f"\n=== Edge Statistics ===")
    print(f"Total edges: {len(edge_weights)}")
    print(f"Total transitions: {sum(edge_weights)}")
    print(f"Max edge weight: {max(edge_weights)}")
    print(f"Mean edge weight: {np.mean(edge_weights):.2f}")
```

## Batch Analysis

### Analyzing Multiple Instances

```python
import numpy as np
import json
from pathlib import Path
from lonpy import compute_lon, LONVisualizer

def analyze_function(name, func, bounds, output_dir):
    """Analyze a function and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build LON
    lon = compute_lon(
        func=func,
        dim=2,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        n_runs=30,
        n_iterations=500,
        seed=42
    )

    # Compute metrics
    lon_metrics = lon.compute_metrics()
    cmlon = lon.to_cmlon()
    cmlon_metrics = cmlon.compute_metrics()

    # Save metrics
    results = {
        "function": name,
        "lon": {
            "n_vertices": lon.n_vertices,
            "n_edges": lon.n_edges,
            "best_fitness": lon.best_fitness,
            **lon_metrics
        },
        "cmlon": {
            "n_vertices": cmlon.n_vertices,
            "n_edges": cmlon.n_edges,
            **cmlon_metrics
        }
    }

    with open(output_dir / f"{name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    viz = LONVisualizer()
    viz.visualize_all(lon, output_dir / name, seed=42)

    return results

# Example usage
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

results = analyze_function("rastrigin", rastrigin, (-5.12, 5.12), "./analysis")
print(json.dumps(results, indent=2))
```

## Next Steps

- [API Reference](../api/index.md) - Complete API documentation
- [Core Concepts](../getting-started/concepts.md) - Understand LON theory
