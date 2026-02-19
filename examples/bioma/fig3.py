from problems import ackley4, griewank, schwefel2_26

from lonpy import BasinHoppingSampler, BasinHoppingSamplerConfig, LONVisualizer

FUNCTIONS = {
    "Ackley4": {
        "func": ackley4,
        "bounds": (-35, 35),
        "step_size": 1.631,
        "n_iterations": 300,
    },
    "Griewank": {
        "func": griewank,
        "bounds": (-200, 200),
        "step_size": 3.6,
        "n_iterations": 200,
    },
    "Schwefel 2.26": {
        "func": schwefel2_26,
        "bounds": (-500, 500),
        "step_size": 151.0,
        "n_iterations": 4000,
    },
}

DIMENSIONS = [3, 5, 8]
N_RUNS = 100
HASH_DIGITS = 2
OPT_DIGITS = -1
SEED = 42


def build_cmlon(func_cfg: dict, n_var: int):
    lb, ub = func_cfg["bounds"]
    domain = [(lb, ub)] * n_var

    config = BasinHoppingSamplerConfig(
        n_runs=N_RUNS,
        n_iterations=func_cfg["n_iterations"],
        step_mode="fixed",
        step_size=func_cfg["step_size"],
        opt_digits=OPT_DIGITS,
        hash_digits=HASH_DIGITS,
        bounded=True,
        seed=SEED,
    )

    sampler = BasinHoppingSampler(config)
    lon = sampler.sample_to_lon(
        func_cfg["func"],
        domain,
        progress_callback=lambda r, t: print(f"  run {r}/{t}", end="\r"),
    )
    cmlon = lon.to_cmlon()
    metrics = cmlon.compute_metrics()
    print(f"  nodes={cmlon.n_vertices}, edges={cmlon.n_edges}, metrics={metrics}")
    return cmlon


def main() -> None:
    viz = LONVisualizer()

    for func_name in FUNCTIONS:
        for n_var in DIMENSIONS:
            print(f"Sampling {func_name} n={n_var} ...")
            cmlon = build_cmlon(FUNCTIONS[func_name], n_var)

            viz.plot_2d(cmlon, output_path=f"fig3_{func_name}_dim{n_var}.png")


if __name__ == "__main__":
    main()
