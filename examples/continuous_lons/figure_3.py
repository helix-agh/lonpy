"""Reproduce Figure 3 from the paper:

3D visualisations for the three benchmark functions with two variables n = 2.

Jason Adair, Gabriela Ochoa, and Katherine M. Malan. 2019.
Local optima networks for continuous fitness landscapes.
In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '19).
Association for Computing Machinery, New York, NY, USA, 1407-1414.
https://doi.org/10.1145/3319619.3326852
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from benchmark_utils import ackley, birastrigin, rastrigin


def create_surface_data(
    func: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create meshgrid data for 3D surface plot."""
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    return X, Y, Z


def plot_benchmark_surfaces(save_path: str | None = None) -> None:
    """Create Figure 3: 3D surface plots of Ackley, Rastrigin, and Birastrigin."""
    fig = plt.figure(figsize=(15, 5))

    # Define benchmark functions with their domains
    benchmarks = [
        ("Ackley", ackley, (-32.768, 32.768)),
        ("Rastrigin", rastrigin, (-5.12, 5.12)),
        ("Birastrigin", birastrigin, (-5.12, 5.12)),
    ]

    for idx, (name, func, bounds) in enumerate(benchmarks, 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")

        X, Y, Z = create_surface_data(func, bounds, resolution=100)

        # Plot surface with viridis colormap
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
            antialiased=True,
        )

        # Set viewing angle to match paper
        ax.view_init(elev=15, azim=45)
        ax.set_proj_type("ortho")
        # Remove axis labels and ticks for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Set subplot label
        ax.set_xlabel(f"({chr(96 + idx)}) {name}", fontsize=12, labelpad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_benchmark_surfaces(save_path="figure_3.png")
