from pathlib import Path

import matplotlib.pyplot as plt
from problems import ackley4, griewank, schwefel2_26
from utils import IMAGES_DIR, FunctionConfig, build_cmlon

from lonpy import LONVisualizer

FUNCTIONS = {
    "Ackley 4": FunctionConfig(
        func=ackley4,
        bounds=(-35, 35),
        step_size=1.631,
        n_iterations=300,
    ),
    "Griewank": FunctionConfig(
        func=griewank,
        bounds=(-200, 200),
        step_size=3.6,
        n_iterations=200,
    ),
    "Schwefel 2.26": FunctionConfig(
        func=schwefel2_26,
        bounds=(-500, 500),
        step_size=151.0,
        n_iterations=4000,
    ),
}

DIMENSIONS = [3, 5, 8]
LABELS = "abcdefghi"


def main() -> None:
    images_dir = Path(IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)

    viz = LONVisualizer()
    func_names = list(FUNCTIONS.keys())

    # Build all CMLONs and save individual figures
    results = {}  # (func_name, dim) -> (cmlon, metrics)
    for func_name in func_names:
        for n_var in DIMENSIONS:
            print(f"Sampling {func_name} n={n_var} ...")
            cmlon = build_cmlon(FUNCTIONS[func_name], n_var)
            metrics = cmlon.compute_metrics()
            results[(func_name, n_var)] = (cmlon, metrics)

            # Save individual figure
            individual_path = images_dir / f"fig3_{func_name}_dim{n_var}.png"
            fig = viz.plot_2d(cmlon, output_path=str(individual_path))
            plt.close(fig)
            print(f"  Saved {individual_path}")

    # Create combined 3x3 grid figure
    n_rows = len(func_names)
    n_cols = len(DIMENSIONS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        dpi=150,
    )

    label_idx = 0
    for row, func_name in enumerate(func_names):
        for col, n_var in enumerate(DIMENSIONS):
            ax = axes[row, col]
            cmlon, metrics = results[(func_name, n_var)]
            success = metrics["global_funnel_proportion"]

            ax.set_aspect("equal")
            ax.axis("off")

            # Render edges and nodes directly onto the grid axes
            graph = cmlon.graph
            edge_widths = viz.compute_edge_widths(graph)
            node_sizes = viz.compute_node_sizes(graph)
            node_colors = viz.compute_cmlon_colors(cmlon)
            layout = viz.get_layout(graph, seed=None)

            if graph.ecount() > 0:
                for i, edge in enumerate(graph.es):
                    src_idx = edge.source
                    tgt_idx = edge.target
                    x0, y0 = layout[src_idx]
                    x1, y1 = layout[tgt_idx]
                    ax.annotate(
                        "",
                        xy=(x1, y1),
                        xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle=f"->,head_length={viz.arrow_size},head_width={viz.arrow_size}",
                            color="dimgray",
                            lw=edge_widths[i],
                            shrinkA=node_sizes[src_idx] * 2,
                            shrinkB=node_sizes[tgt_idx] * 2,
                        ),
                    )

            scatter_sizes = [s**2 * 10 for s in node_sizes]
            ax.scatter(
                layout[:, 0],
                layout[:, 1],
                s=scatter_sizes,
                c=node_colors,
                edgecolors="black",
                linewidths=0.5,
                zorder=10,
            )

            # Add subplot label
            label = LABELS[label_idx]
            ax.set_title(
                f"({label}) {func_name}, $n$ = {n_var}, success = {success:.2f}",
                fontsize=10,
                pad=6,
            )
            label_idx += 1

    plt.tight_layout()

    combined_path = images_dir / "fig3.png"
    fig.savefig(str(combined_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
