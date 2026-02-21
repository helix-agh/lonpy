from pathlib import Path

import numpy as np
from problems import spread_spectrum_radar_polly_phase, ssc_ruspini
from utils import IMAGES_DIR, FunctionConfig, build_cmlon

from lonpy import LONVisualizer

FUNCTIONS = {
    "SpreadSpectrumRadarPollyPhase": FunctionConfig(
        func=spread_spectrum_radar_polly_phase,
        bounds=(0, 2 * np.pi),
        step_size=1.2566,
        n_iterations=100,
        hash_digits=1,
        dimensions=[3, 5, 8, 13],
        best=0.5,
    ),
    "SSCRuspini": FunctionConfig(
        func=ssc_ruspini,
        bounds=(0, 200),
        step_size=48.64,
        n_iterations=100,
        hash_digits=0,
        dimensions=[6, 8, 10, 12],
        best=None,  # 51064, 12882, 10127, 8576 depending on dim
    ),
}


def main() -> None:
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    viz = LONVisualizer()

    for func_name, func_cfg in FUNCTIONS.items():
        for n_var in func_cfg.dimensions:
            print(f"Sampling {func_name} n={n_var} ...")
            cmlon = build_cmlon(func_cfg, n_var)

            viz.plot_2d(cmlon, output_path=f"{IMAGES_DIR}/fig6_{func_name}_dim{n_var}.png")


if __name__ == "__main__":
    main()
