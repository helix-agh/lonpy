from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

import pytest

from lonpy import LON, LONVisualizer
from tests.conftest import SEED


@pytest.fixture
def visualizer() -> LONVisualizer:
    return LONVisualizer()


class TestLayoutReproducibility:
    def test_layout_same_seed_gives_same_coordinates(
        self, rastrigin_lon: LON, visualizer: LONVisualizer
    ) -> None:
        coords1 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        coords2 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        np.testing.assert_array_equal(coords1, coords2)

    def test_layout_different_seeds_give_different_coordinates(
        self, rastrigin_lon: LON, visualizer: LONVisualizer
    ) -> None:
        coords1 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED)
        coords2 = visualizer.get_layout(rastrigin_lon.graph, seed=SEED + 1)
        assert not np.array_equal(coords1, coords2)


class TestPlot2DReproducibility:
    def test_plot_2d_same_seed_same_figure(
        self, sphere_lon: LON, visualizer: LONVisualizer, tmp_path: Path
    ) -> None:
        path1 = tmp_path / "plot1.png"
        path2 = tmp_path / "plot2.png"

        visualizer.plot_2d(sphere_lon, output_path=path1, seed=SEED)
        plt.close("all")
        visualizer.plot_2d(sphere_lon, output_path=path2, seed=SEED)
        plt.close("all")

        assert path1.read_bytes() == path2.read_bytes()
