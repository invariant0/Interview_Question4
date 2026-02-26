"""Unit tests for entropy scheduler."""

import pytest
from econ_models.config.training_config import EntropyConfig
from econ_models.dl.components.entropy_scheduler import compute_entropy_coeff


class TestComputeEntropyCoeff:
    """Tests for the compute_entropy_coeff pure function."""

    def test_before_start_epoch_returns_initial(self):
        config = EntropyConfig(initial_coeff=1.0, final_coeff=0.0, start_epoch=10, decay_epochs=50)
        assert compute_entropy_coeff(0, config) == 1.0
        assert compute_entropy_coeff(9, config) == 1.0

    def test_after_completion_returns_final(self):
        config = EntropyConfig(initial_coeff=1.0, final_coeff=0.0, start_epoch=10, decay_epochs=50)
        assert compute_entropy_coeff(60, config) == 0.0
        assert compute_entropy_coeff(100, config) == 0.0

    def test_midpoint_returns_linear_interpolation(self):
        config = EntropyConfig(initial_coeff=1.0, final_coeff=0.0, start_epoch=0, decay_epochs=10)
        result = compute_entropy_coeff(5, config)
        assert abs(result - 0.5) < 1e-10

    def test_start_of_decay(self):
        config = EntropyConfig(initial_coeff=2.0, final_coeff=0.0, start_epoch=5, decay_epochs=10)
        assert compute_entropy_coeff(5, config) == 2.0

    def test_end_of_decay(self):
        config = EntropyConfig(initial_coeff=2.0, final_coeff=0.0, start_epoch=5, decay_epochs=10)
        assert compute_entropy_coeff(15, config) == 0.0

    def test_zero_decay_epochs_returns_final(self):
        config = EntropyConfig(initial_coeff=1.0, final_coeff=0.0, start_epoch=0, decay_epochs=0)
        assert compute_entropy_coeff(0, config) == 0.0
