"""Integration smoke tests for BasicInvestmentModel and RiskyModel.

These tests construct tiny model instances and verify that a single
training step executes successfully with finite outputs.  They require
TensorFlow and all project dependencies to be importable.
"""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.config.training_config import (
    DistributionalConfig,
    EntropyConfig,
    OptimizationConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_params():
    """Minimal EconomicParams."""
    return EconomicParams(
        discount_factor=0.95,
        capital_share=0.3,
        depreciation_rate=0.1,
        productivity_persistence=0.9,
        productivity_std_dev=0.05,
        adjustment_cost_convex=0.5,
        adjustment_cost_fixed=0.01,
        equity_issuance_cost_fixed=0.0,
        equity_issuance_cost_linear=0.0,
        default_cost_proportional=0.0,
        corporate_tax_rate=0.0,
        risk_free_rate=0.02,
        collateral_recovery_fraction=0.5,
    )


def _tiny_config(**overrides):
    """Minimal DeepLearningConfig."""
    cfg = DeepLearningConfig()
    cfg.batch_size = 8
    cfg.learning_rate = 1e-3
    cfg.verbose = False
    cfg.epochs = 2
    cfg.steps_per_epoch = 2
    cfg.hidden_layers = (16, 16)
    cfg.activation_function = 'tanh'
    cfg.value_scale_factor = 1.0
    cfg.use_layer_norm = False
    cfg.capital_min = 0.5
    cfg.capital_max = 2.0
    cfg.productivity_min = 0.8
    cfg.productivity_max = 1.2
    cfg.debt_min = 0.0
    cfg.debt_max = 1.0
    cfg.polyak_averaging_decay = 0.995
    cfg.gradient_clip_norm = 1.0
    cfg.mc_sample_number_bond_priceing = 4
    cfg.mc_sample_number_continuation_value = 4
    cfg.mc_next_candidate_sample = 4
    cfg.min_q_price = 0.01
    cfg.epsilon_value_default = 0.0
    cfg.epsilon_debt = 0.001
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _basic_bounds():
    return {
        'k_min': 0.5, 'k_max': 2.0,
        'z_min': 0.8, 'z_max': 1.2,
    }


def _risky_bounds():
    return {
        'k_min': 0.5, 'k_max': 2.0,
        'z_min': 0.8, 'z_max': 1.2,
        'b_min': 0.0, 'b_max': 1.0,
    }


# ---------------------------------------------------------------------------
# BasicInvestmentModel
# ---------------------------------------------------------------------------

class TestBasicInvestmentModelSmoke:
    """Smoke tests for BasicInvestmentModel construction and single step."""

    def test_construction(self):
        """Model constructs without error."""
        from econ_models.dl.models.basic_investment_model import BasicInvestmentModel
        model = BasicInvestmentModel(
            params=_tiny_params(),
            config=_tiny_config(),
            state_bounds=_basic_bounds(),
            entropy_config=EntropyConfig(
                initial_coeff=0.01, final_coeff=0.0,
                start_epoch=0, decay_epochs=1,
            ),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        assert model is not None

    def test_single_train_step_finite(self):
        """A single training step produces finite metrics."""
        from econ_models.dl.models.basic_investment_model import BasicInvestmentModel
        model = BasicInvestmentModel(
            params=_tiny_params(),
            config=_tiny_config(),
            state_bounds=_basic_bounds(),
            entropy_config=EntropyConfig(
                initial_coeff=0.01, final_coeff=0.0,
                start_epoch=0, decay_epochs=1,
            ),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        # Create small batch
        k = tf.random.uniform((8, 1), 0.5, 2.0)
        z = tf.random.uniform((8, 1), 0.8, 1.2)
        metrics = model._train_step_impl(k, z)
        assert isinstance(metrics, dict)
        for name, value in metrics.items():
            assert tf.math.is_finite(value), f"Metric {name} is not finite: {value}"

    def test_get_optimal_actions(self):
        """get_optimal_actions returns finite values with correct shape."""
        from econ_models.dl.models.basic_investment_model import BasicInvestmentModel
        model = BasicInvestmentModel(
            params=_tiny_params(),
            config=_tiny_config(),
            state_bounds=_basic_bounds(),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        k = tf.random.uniform((4, 1), 0.5, 2.0)
        z = tf.random.uniform((4, 1), 0.8, 1.2)
        inv_rate, invest_prob, v_opt = model.get_optimal_actions(k, z)
        assert inv_rate.shape == (4, 1)
        assert invest_prob.shape == (4, 1)
        assert v_opt.shape == (4, 1)
        assert tf.reduce_all(tf.math.is_finite(inv_rate))
        assert tf.reduce_all(tf.math.is_finite(invest_prob))
        assert tf.reduce_all(tf.math.is_finite(v_opt))


# ---------------------------------------------------------------------------
# RiskyModel
# ---------------------------------------------------------------------------

class TestRiskyModelSmoke:
    """Smoke tests for RiskyModel construction and single step."""

    def test_construction(self):
        """Model constructs without error."""
        from econ_models.dl.models.risky_model import RiskyModel
        model = RiskyModel(
            params=_tiny_params(),
            config=_tiny_config(),
            state_bounds=_risky_bounds(),
            entropy_config=EntropyConfig(
                initial_coeff=0.01, final_coeff=0.0,
                start_epoch=0, decay_epochs=1,
            ),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        assert model is not None

    def test_single_train_step_finite(self):
        """A single training step produces finite metrics."""
        from econ_models.dl.models.risky_model import RiskyModel
        model = RiskyModel(
            params=_tiny_params(),
            config=_tiny_config(),
            state_bounds=_risky_bounds(),
            entropy_config=EntropyConfig(
                initial_coeff=0.01, final_coeff=0.0,
                start_epoch=0, decay_epochs=1,
            ),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        k = tf.random.uniform((8, 1), 0.5, 2.0)
        b = tf.random.uniform((8, 1), 0.0, 1.0)
        z = tf.random.uniform((8, 1), 0.8, 1.2)
        metrics = model._train_step_impl(k, b, z)
        assert isinstance(metrics, dict)
        for name, value in metrics.items():
            assert tf.math.is_finite(value), f"Metric {name} is not finite: {value}"

    def test_get_optimal_actions(self):
        """get_optimal_actions returns finite values with correct shape."""
        from econ_models.dl.models.risky_model import RiskyModel
        model = RiskyModel(
            params=_tiny_params(),
            config=_tiny_config(),
            state_bounds=_risky_bounds(),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        k = tf.random.uniform((4, 1), 0.5, 2.0)
        b = tf.random.uniform((4, 1), 0.0, 1.0)
        z = tf.random.uniform((4, 1), 0.8, 1.2)
        k_opt, b_opt, q_opt, v_opt, invest_prob = model.get_optimal_actions(k, b, z)
        for name, tensor in [("k_opt", k_opt), ("b_opt", b_opt), ("q_opt", q_opt),
                              ("v_opt", v_opt), ("invest_prob", invest_prob)]:
            assert tensor.shape == (4, 1), f"{name} shape: {tensor.shape}"
            assert tf.reduce_all(tf.math.is_finite(tensor)), f"{name} not finite"
