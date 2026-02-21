from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

from lonpy import CMLON, BasinHoppingSampler, BasinHoppingSamplerConfig

DEFAULT_N_RUNS = 100
DEFAULT_OPT_DIGITS = -1
DEFAULT_HASH_DIGITS = 2
DEFAULT_SEED = 42
IMAGES_DIR = "images"


@dataclass
class FunctionConfig:
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]
    step_size: float
    n_iterations: int
    hash_digits: int = DEFAULT_HASH_DIGITS
    dimensions: list[int] = field(default_factory=lambda: [3, 5, 8])
    best: float | None = None


def build_cmlon(
    func_cfg: FunctionConfig,
    n_var: int,
    *,
    n_runs: int = DEFAULT_N_RUNS,
    opt_digits: int = DEFAULT_OPT_DIGITS,
    seed: int = DEFAULT_SEED,
) -> CMLON:
    lb, ub = func_cfg.bounds
    domain = [(lb, ub)] * n_var

    config = BasinHoppingSamplerConfig(
        n_runs=n_runs,
        n_iterations=func_cfg.n_iterations,
        step_mode="fixed",
        step_size=func_cfg.step_size,
        opt_digits=opt_digits,
        hash_digits=func_cfg.hash_digits,
        bounded=True,
        seed=seed,
    )

    sampler = BasinHoppingSampler(config)
    lon = sampler.sample_to_lon(
        func_cfg.func,
        domain,
        progress_callback=lambda r, t: print(f"  run {r}/{t}", end="\r"),
    )
    return lon.to_cmlon()
