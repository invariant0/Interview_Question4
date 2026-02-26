"""Unit tests for InputPreparer."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from unittest.mock import MagicMock
from econ_models.config.training_config import DistributionalConfig


class TestInputPreparer:
    """Tests for InputPreparer normalization and concatenation."""

    @pytest.fixture
    def mock_state_normalizer(self):
        """Create a mock normalizer that returns input unchanged."""
        norm = MagicMock()
        norm.normalize_capital = lambda x: x
        norm.normalize_productivity = lambda x: x
        norm.normalize_debt = lambda x: x
        return norm

    @pytest.fixture
    def basic_preparer(self, mock_state_normalizer):
        from econ_models.dl.components.input_preparer import InputPreparer
        return InputPreparer(
            state_normalizer=mock_state_normalizer,
            include_debt=False,
            distributional_config=DistributionalConfig(),
        )

    @pytest.fixture
    def risky_preparer(self, mock_state_normalizer):
        from econ_models.dl.components.input_preparer import InputPreparer
        return InputPreparer(
            state_normalizer=mock_state_normalizer,
            include_debt=True,
            distributional_config=DistributionalConfig(),
        )

    def test_input_dim_basic(self, basic_preparer):
        assert basic_preparer.input_dim == 2  # k, z

    def test_input_dim_risky(self, risky_preparer):
        assert risky_preparer.input_dim == 3  # k, b, z

    def test_input_dim_distributional(self, mock_state_normalizer):
        from econ_models.dl.components.input_preparer import InputPreparer
        dist_config = DistributionalConfig(
            is_distributional=True,
            param_names=('rho', 'std', 'convex', 'fixed'),
        )
        preparer = InputPreparer(
            state_normalizer=mock_state_normalizer,
            include_debt=False,
            distributional_config=dist_config,
        )
        assert preparer.input_dim == 6  # k, z + 4 params

    def test_prepare_shape_basic(self, basic_preparer):
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[0.5], [0.6]])
        result = basic_preparer.prepare(k=k, z=z)
        assert result.shape == (2, 2)

    def test_prepare_shape_risky(self, risky_preparer):
        k = tf.constant([[1.0], [2.0]])
        z = tf.constant([[0.5], [0.6]])
        b = tf.constant([[0.1], [0.2]])
        result = risky_preparer.prepare(k=k, z=z, b=b)
        assert result.shape == (2, 3)

    def test_prepare_next_state_same_as_prepare(self, basic_preparer):
        k = tf.constant([[1.0]])
        z = tf.constant([[0.5]])
        r1 = basic_preparer.prepare(k=k, z=z)
        r2 = basic_preparer.prepare_next_state(k_prime=k, z_prime=z)
        tf.debugging.assert_near(r1, r2)
