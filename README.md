<div align="center">
    <img src="docs/assets/icon.png" alt="lonpy" width="800">
</div>

[![PyPI version](https://badge.fury.io/py/lonpy.svg)](https://pypi.org/project/lonpy/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Local Optima Networks for Continuous Optimization**

lonpy is a Python library for constructing, analyzing, and visualizing Local Optima Networks (LONs) for continuous optimization problems. LONs provide a powerful way to understand the structure of fitness landscapes, revealing how local optima are connected and how difficult it may be to find global optima.

## Features

- **Basin-Hopping Sampling**: Efficient exploration of fitness landscapes using configurable Basin-Hopping
- **LON Construction**: Automatic construction of Local Optima Networks from sampling data
- **CMLON Support**: Compressed Monotonic LONs for cleaner landscape analysis
- **Rich Metrics**: Compute landscape metrics including funnel analysis and neutrality
- **Beautiful Visualizations**: 2D and 3D plots with support for animated GIFs

## Installation

```bash
pip install lonpy
```

Or install from source:

```bash
git clone https://github.com/agh-a2s/lonpy.git
cd lonpy
pip install -e .
```

## Quick Start

```python
import numpy as np
from lonpy import compute_lon, LONVisualizer

# Define an objective function
def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Construct the LON
lon = compute_lon(
    rastrigin,
    dim=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    n_runs=20,
    n_iterations=500,
    seed=42
)

metrics = lon.compute_metrics()
print(f"Number of funnels: {metrics['n_funnels']}")
print(f"Global funnels: {metrics['n_global_funnels']}")

# Visualize
viz = LONVisualizer()
viz.plot_2d(lon, output_path="lon_2d.png")
viz.plot_3d(lon, output_path="lon_3d.png")
```

### Compressed Monotonic LONs (CMLONs)

CMLONs are a compressed representation where nodes with equal fitness that are connected get merged. This provides a cleaner view of the landscape's funnel structure.

```python
# Convert LON to CMLON
cmlon = lon.to_cmlon()

# Analyze CMLON-specific metrics
cmlon_metrics = cmlon.compute_metrics()
```

### Custom Sampling Configuration

```python
from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig

config = BasinHoppingSamplerConfig(
    n_runs=50,              # Number of independent runs
    n_iterations=1000,      # Iterations per run
    step_size=0.05,         # Perturbation size
    step_mode="per",        # "per" (percentage) or "fix" (fixed)
    hash_digits=4,          # Precision for identifying optima
    seed=42                 # For reproducibility
)

sampler = BasinHoppingSampler(config)

# Define search domain
domain = [(-5.12, 5.12), (-5.12, 5.12)]

# Run sampling
lon = sampler.sample_to_lon(rastrigin, domain)
```

## Documentation

For full documentation, visit: [https://agh-a2s.github.io/lonpy](https://agh-a2s.github.io/lonpy)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
