# Sampling Guide

This guide covers how to configure sampling algorithms for LON construction.

## Continuous Optimization

### Quick Start

The simplest way to create a LON for continuous problems:

```python
from lonpy import compute_lon

lon = compute_lon(
    func=my_objective,
    dim=2,
    lower_bound=-5.0,
    upper_bound=5.0,
    n_runs=20,
    seed=42
)
```

### Configuration Options

For more control, use `BasinHoppingSamplerConfig`:

```python
from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig

config = BasinHoppingSamplerConfig(
    n_runs=30,              # Number of independent runs
    n_iterations=500,       # Iterations per run
    step_mode="percentage", # "percentage" or "fixed"
    step_size=0.1,          # Perturbation magnitude
    hash_digits=4,          # Precision for node identification
    opt_digits=-1,          # Precision for optimization (-1 = full)
    bounded=True,           # Enforce domain bounds
    minimizer_method="L-BFGS-B",
    seed=42
)

sampler = BasinHoppingSampler(config)
lon = sampler.sample_to_lon(my_objective, domain)
```

### Parameters Explained

#### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_runs` | 10 | Number of independent Basin-Hopping runs |
| `n_iterations` | 1000 | Iterations per run |
| `seed` | None | Random seed for reproducibility |

**Choosing n_runs and n_iterations:**

- More runs = better coverage of the landscape
- More iterations = deeper exploitation from each starting point
- Trade-off: `n_runs × n_iterations` determines total evaluations

```python
# Wide coverage (recommended for unknown landscapes)
config = BasinHoppingSamplerConfig(n_runs=50, n_iterations=200)

# Deep exploration (for complex local structure)
config = BasinHoppingSamplerConfig(n_runs=10, n_iterations=1000)
```

#### Perturbation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_mode` | "fixed" | How to interpret step_size |
| `step_size` | 0.01 | Perturbation magnitude |
| `bounded` | True | Keep perturbations within domain |

**Step modes:**

```python
# Fixed: step_size is absolute distance
config = BasinHoppingSamplerConfig(
    step_mode="fixed",
    step_size=0.5  # Always perturb by ±0.5 in each dimension
)

# Percentage: step_size is fraction of domain range
config = BasinHoppingSamplerConfig(
    step_mode="percentage",
    step_size=0.1  # Perturb by ±10% of (upper - lower)
)
```

**Choosing step size:**

- Too small: Stays in same basin, misses transitions
- Too large: Jumps randomly, misses local structure
- Good starting point: 5-10% of domain range

#### Precision Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hash_digits` | 4 | Decimal places for node identification |
| `opt_digits` | -1 | Decimal places for optimization (-1 = full) |

**hash_digits** determines when two solutions are considered the same optimum:

```python
# High precision: More distinct nodes
config = BasinHoppingSamplerConfig(hash_digits=6)

# Low precision: More merging, fewer nodes
config = BasinHoppingSamplerConfig(hash_digits=2)
```

#### Local Minimizer Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minimizer_method` | "L-BFGS-B" | Scipy minimizer algorithm |
| `minimizer_options` | `{"ftol": 1e-07, "gtol": 1e-05}` | Minimizer options |

```python
# Custom minimizer settings
config = BasinHoppingSamplerConfig(
    minimizer_method="L-BFGS-B",
    minimizer_options={
        "ftol": 1e-10,  # Tighter function tolerance
        "gtol": 1e-08,  # Tighter gradient tolerance
        "maxiter": 1000 # More iterations allowed
    }
)
```

### Domain Specification

The domain is specified as a list of (lower, upper) tuples:

```python
# Same bounds for all dimensions
lon = compute_lon(func, dim=5, lower_bound=-5.0, upper_bound=5.0)

# Different bounds per dimension
domain = [
    (-5.0, 5.0),    # x1
    (0.0, 10.0),    # x2
    (-1.0, 1.0)     # x3
]
sampler = BasinHoppingSampler()
lon = sampler.sample_to_lon(func, domain)
```

## Discrete Optimization

### Quick Start

The simplest way to create a LON for discrete problems:

```python
from lonpy import compute_discrete_lon, OneMax

problem = OneMax(n=20)
lon = compute_discrete_lon(problem, n_runs=100, seed=42)
```

### Configuration Options

For more control, use `ILSSamplerConfig`:

```python
from lonpy import ILSSampler, ILSSamplerConfig, OneMax

config = ILSSamplerConfig(
    n_runs=100,                     # Number of independent ILS runs
    max_iterations=0,               # Max iterations (0 = unlimited)
    non_improvement_iterations=100, # Stop after no improvement
    perturbation_strength=2,        # Number of random moves
    first_improvement=True,         # Hill climbing strategy
    representation="bitstring",     # "bitstring" or "permutation"
    neighborhood="flip",            # "flip" or "swap"
    seed=42
)

sampler = ILSSampler(config)
problem = OneMax(n=20)
lon = sampler.sample_to_lon(problem)
```

### Parameters Explained

#### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_runs` | 100 | Number of independent ILS runs |
| `max_iterations` | 0 | Max iterations per run (0 = unlimited) |
| `non_improvement_iterations` | 100 | Stop after this many non-improving iterations |
| `seed` | None | Random seed for reproducibility |

**Stopping conditions:**

- If `max_iterations > 0`: Stop after that many iterations
- If `non_improvement_iterations > 0`: Stop after that many iterations without improvement
- Both conditions are checked; whichever is reached first

```python
# Run until no improvement for 100 iterations
config = ILSSamplerConfig(
    n_runs=100,
    non_improvement_iterations=100
)

# Run exactly 500 iterations per run
config = ILSSamplerConfig(
    n_runs=50,
    max_iterations=500,
    non_improvement_iterations=500  # Effectively disabled
)
```

#### Perturbation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `perturbation_strength` | 2 | Number of random moves per perturbation |

**Choosing perturbation strength:**

- Too small (1): May not escape current basin
- Too large: Jumps randomly, may miss structure
- Good starting point: 2-3 for small problems, scale with problem size

```python
# For small problems (n < 30)
config = ILSSamplerConfig(perturbation_strength=2)

# For larger problems (n > 50)
config = ILSSamplerConfig(perturbation_strength=3)
```

#### Hill Climbing Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `first_improvement` | True | Use first improvement hill climbing |

**First improvement vs Best improvement:**

- **First improvement** (True): Faster, more diverse exploration
- **Best improvement** (False): More thorough, deterministic convergence

```python
# Fast exploration (recommended)
config = ILSSamplerConfig(first_improvement=True)

# Thorough convergence
config = ILSSamplerConfig(first_improvement=False)
```

#### Representation and Neighborhood

| Parameter | Default | Description |
|-----------|---------|-------------|
| `representation` | "bitstring" | Solution representation type |
| `neighborhood` | "flip" | Neighborhood operator |

**Bitstring representation:**

- Use for binary problems (OneMax, Knapsack, Number Partitioning)
- Use `neighborhood="flip"` (flip one bit)

```python
config = ILSSamplerConfig(
    representation="bitstring",
    neighborhood="flip"
)
```

**Permutation representation:**

- Use for ordering problems (TSP, scheduling)
- Use `neighborhood="swap"` (swap two positions)

```python
config = ILSSamplerConfig(
    representation="permutation",
    neighborhood="swap"
)
```

### Built-in Problems

#### OneMax

```python
from lonpy import OneMax

# Maximize number of 1s in bitstring
problem = OneMax(n=20)
lon = compute_discrete_lon(problem, n_runs=100)
```

#### Knapsack

```python
from lonpy import Knapsack

# Define items and capacity
problem = Knapsack(
    values=[60, 100, 120, 80, 90],
    weights=[10, 20, 30, 15, 25],
    capacity=50
)
lon = compute_discrete_lon(problem, n_runs=100)

# Or load from file
problem = Knapsack.from_file("instance.txt")
```

#### Number Partitioning

```python
from lonpy import NumberPartitioning

# Generate random instance
problem = NumberPartitioning(n=20, k=0.5, seed=42)
lon = compute_discrete_lon(problem, n_runs=100)
```

## Accessing Raw Data

### Continuous

```python
sampler = BasinHoppingSampler(config)
trace_df, raw_records = sampler.sample(func, domain)

# trace_df columns: [run, fit1, node1, fit2, node2]
print(trace_df.head())

# raw_records contains detailed iteration data
for record in raw_records[:5]:
    print(f"Run {record['run']}, Iter {record['iteration']}")
    print(f"  Current: {record['current_f']:.4f}")
    print(f"  New: {record['new_f']:.4f}")
    print(f"  Accepted: {record['accepted']}")
```

### Discrete

```python
sampler = ILSSampler(config)
trace_df, raw_records = sampler.sample(problem)

# trace_df columns: [run, fit1, node1, fit2, node2]
print(trace_df.head())

# raw_records contains transition data
for record in raw_records[:5]:
    print(f"Run {record['run']}")
    print(f"  From fitness: {record['fit1']}")
    print(f"  To fitness: {record['fit2']}")
```

## Progress Monitoring

Track sampling progress with a callback:

```python
def progress(run, total):
    print(f"Run {run}/{total}")

# Continuous
sampler = BasinHoppingSampler(config)
lon = sampler.sample_to_lon(func, domain, progress_callback=progress)

# Discrete
sampler = ILSSampler(config)
lon = sampler.sample_to_lon(problem, progress_callback=progress)
```

## Best Practices

### For Standard Test Functions (Continuous)

```python
# Rastrigin, Ackley, etc. with known bounds
config = BasinHoppingSamplerConfig(
    n_runs=30,
    n_iterations=500,
    step_mode="percentage",
    step_size=0.1,
    hash_digits=4,
    seed=42
)
```

### For Unknown Functions (Continuous)

```python
# Start with wider exploration
config = BasinHoppingSamplerConfig(
    n_runs=50,
    n_iterations=200,
    step_mode="percentage",
    step_size=0.15,  # Larger steps initially
    hash_digits=3,   # Coarser grouping
)

# Refine based on initial results
```

### For High-Dimensional Problems (Continuous)

```python
# More runs needed for coverage
config = BasinHoppingSamplerConfig(
    n_runs=100,
    n_iterations=500,
    step_mode="percentage",
    step_size=0.05,  # Smaller relative steps
)
```

### For Small Discrete Problems

```python
# OneMax, small Knapsack (n < 30)
config = ILSSamplerConfig(
    n_runs=100,
    non_improvement_iterations=100,
    perturbation_strength=2,
    first_improvement=True
)
```

### For Large Discrete Problems

```python
# Large instances (n > 50)
config = ILSSamplerConfig(
    n_runs=200,
    non_improvement_iterations=200,
    perturbation_strength=3,
    first_improvement=True
)
```

## Next Steps

- [Analysis Guide](analysis.md) - Interpret your LON metrics
- [Visualization Guide](visualization.md) - Create plots
- [API Reference](../api/sampling.md) - Full API documentation
