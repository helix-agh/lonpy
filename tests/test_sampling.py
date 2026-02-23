import pytest

from lonpy.sampling import BasinHoppingSampler, BasinHoppingSamplerConfig


def test_generate_initial_points_invalid_init_mode():
    config = BasinHoppingSamplerConfig(init_mode="sobol")  # type: ignore[arg-type]
    sampler = BasinHoppingSampler(config)
    domain = [(-5.0, 5.0), (-5.0, 5.0)]
    with pytest.raises(ValueError, match="Unknown init_mode 'sobol'"):
        sampler.sample_to_lon(lambda x: sum(x**2), domain)
