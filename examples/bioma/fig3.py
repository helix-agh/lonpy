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
        max_perturbations_without_improvement=300,
    ),
    "Griewank": FunctionConfig(
        func=griewank,
        bounds=(-200, 200),
        step_size=3.6,
        max_perturbations_without_improvement=200,
    ),
    "Schwefel 2.26": FunctionConfig(
        func=schwefel2_26,
        bounds=(-500, 500),
        step_size=151.0,
        max_perturbations_without_improvement=4000,
    ),
}

DIMENSIONS = [3, 5, 8]
LABELS = "abcdefghi"

METRIC_PANELS = [
    ("n_optima", "Nodes"),
    ("n_funnels", "Funnels"),
    ("neutral", "Neutral"),
    ("strength", "Strength"),
    ("success", "Success"),
    ("deviation", "Deviation"),
]

FUNC_STYLES = {
    "Ackley 4": {"color": "tab:blue", "marker": "o"},
    "Griewank": {"color": "tab:purple", "marker": "P"},
    "Schwefel 2.26": {"color": "tab:green", "marker": "^"},
}


def build_all(func_names: list[str]) -> dict:
    """Build all CMLONs and collect metrics."""
    results = {}
    for func_name in func_names:
        for n_var in DIMENSIONS:
            print(f"Sampling {func_name} n={n_var} ...")
            cmlon = build_cmlon(FUNCTIONS[func_name], n_var)
            metrics = cmlon.compute_metrics()
            results[(func_name, n_var)] = (cmlon, metrics)
    return results


def save_network_figures(results: dict, func_names: list[str], images_dir: Path) -> None:
    """Save individual and combined 3x3 CMLON network plots."""
    viz = LONVisualizer()

    for func_name in func_names:
        for n_var in DIMENSIONS:
            cmlon, _metrics = results[(func_name, n_var)]
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
        figsize=(8 * n_cols, 8 * n_rows),
        dpi=150,
    )

    label_idx = 0
    for row, func_name in enumerate(func_names):
        for col, n_var in enumerate(DIMENSIONS):
            ax = axes[row, col]
            cmlon, metrics = results[(func_name, n_var)]
            success = metrics["success"]

            ax.set_aspect("equal")
            ax.axis("off")

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


def save_metrics_figure(results: dict, func_names: list[str], images_dir: Path) -> None:
    """Create a 2x3 grid comparing metrics across dimensions for all functions."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)

    for panel_idx, (metric_key, metric_label) in enumerate(METRIC_PANELS):
        row, col = divmod(panel_idx, 3)
        ax = axes[row, col]

        for func_name in func_names:
            style = FUNC_STYLES[func_name]
            values = [results[(func_name, d)][1][metric_key] for d in DIMENSIONS]
            ax.plot(
                DIMENSIONS,
                values,
                color=style["color"],
                marker=style["marker"],
                label=func_name,
                linewidth=1.5,
                markersize=7,
            )

        ax.set_xlabel("Dimension")
        ax.set_ylabel(metric_label)
        ax.set_xticks(DIMENSIONS)

    # Single shared legend below the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(func_names),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    metrics_path = images_dir / "fig2.png"
    fig.savefig(str(metrics_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {metrics_path}")


def main() -> None:
    images_dir = Path(IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)

    func_names = list(FUNCTIONS.keys())
    results = build_all(func_names)

    save_network_figures(results, func_names, images_dir)
    save_metrics_figure(results, func_names, images_dir)


if __name__ == "__main__":
    main()
