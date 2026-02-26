"""Integration smoke tests for BasicModelDL and RiskFreeModelDL.

These tests construct tiny model instances and verify that a single
training step executes successfully with finite outputs.  They require
TensorFlow and all project dependencies to be importable.
"""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams


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
    """Minimal DeepLearningConfig with all fields required by both models."""
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
    # LR scheduling fields (required by both BasicModelDL and RiskFreeModelDL)
    cfg.lr_decay_rate = 0.99
    cfg.lr_decay_steps = 100
    # Euler residual weight (required by BasicModelDL)
    cfg.euler_residual_weight = 1.0
    # Polyak scheduling (required by RiskFreeModelDL)
    cfg.polyak_decay_end = 0.999
    cfg.polyak_decay_epochs = 10
    cfg.target_update_freq = 1
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _basic_bounds():
    return {
        'k_min': 0.5, 'k_max': 2.0,
        'z_min': 0.8, 'z_max': 1.2,
    }


def _risk_free_bounds():
    return {
        'k_min': 0.5, 'k_max': 2.0,
        'z_min': 0.8, 'z_max': 1.2,
        'b_min': 0.0, 'b_max': 1.0,
    }


# ---------------------------------------------------------------------------
# BasicModelDL
# ---------------------------------------------------------------------------

class TestBasicModelDLSmoke:
    """Smoke tests for BasicModelDL construction and single step."""

    def test_construction(self):
        """Model constructs without error."""
        from econ_models.dl.basic import BasicModelDL
        model = BasicModelDL(
            params=_tiny_params(),
            config=_tiny_config(),
            bounds=_basic_bounds(),
        )
        assert model is not None

    def test_single_train_step_finite(self):
        """A single training step produces finite metrics."""
        from econ_models.dl.basic import BasicModelDL
        model = BasicModelDL(
            params=_tiny_params(),
            config=_tiny_config(),
            bounds=_basic_bounds(),
        )
        k = tf.random.uniform((8, 1), 0.5, 2.0)
        z = tf.random.uniform((8, 1), 0.8, 1.2)
        eps1 = tf.random.normal((8, 1)) * 0.05
        eps2 = tf.random.normal((8, 1)) * 0.05
        metrics = model.train_step(k, z, eps1, eps2)
        assert isinstance(metrics, dict)
        for name, value in metrics.items():
            assert tf.math.is_finite(value), f"Metric {name} is not finite: {value}"

    def test_compute_loss_finite(self):
        """compute_loss returns finite loss values with correct shape."""
        from econ_models.dl.basic import BasicModelDL
        model = BasicModelDL(
            params=_tiny_params(),
            config=_tiny_config(),
            bounds=_basic_bounds(),
        )
        k = tf.random.uniform((4, 1), 0.5, 2.0)
        z = tf.random.uniform((4, 1), 0.8, 1.2)
        eps1 = tf.random.normal((4, 1)) * 0.05
        eps2 = tf.random.normal((4, 1)) * 0.05
        total_loss, bellman_loss, euler_loss, k_prime = model.compute_loss(
            k, z, eps1, eps2,
        )
        assert tf.math.is_finite(total_loss)
        assert tf.math.is_finite(bellman_loss)
        assert tf.math.is_finite(euler_loss)
        assert tf.math.is_finite(k_prime)


# ---------------------------------------------------------------------------
# RiskFreeModelDL
# ---------------------------------------------------------------------------

class TestRiskFreeModelDLSmoke:
    """Smoke tests for RiskFreeModelDL construction and single step."""

    def test_construction(self):
        """Model constructs without error."""
        from econ_models.dl.risk_free import RiskFreeModelDL, OptimizationConfig
        model = RiskFreeModelDL(
            params=_tiny_params(),
            config=_tiny_config(),
            bounds=_risk_free_bounds(),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        assert model is not None

    def test_single_train_step_finite(self):
        """A single training step produces finite metrics."""
        from econ_models.dl.risk_free import RiskFreeModelDL, OptimizationConfig
        model = RiskFreeModelDL(
            params=_tiny_params(),
            config=_tiny_config(),
            bounds=_risk_free_bounds(),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        k = tf.random.uniform((8, 1), 0.5, 2.0)
        b = tf.random.uniform((8, 1), 0.0, 1.0)
        z = tf.random.uniform((8, 1), 0.8, 1.2)
        metrics = model._train_step_impl(k, b, z)
        assert isinstance(metrics, dict)
        for name, value in metrics.items():
            assert tf.math.is_finite(value), f"Metric {name} is not finite: {value}"

    def test_networks_output_correct_shapes(self):
        """Network forward passes return tensors with correct shapes."""
        from econ_models.dl.risk_free import RiskFreeModelDL, OptimizationConfig
        model = RiskFreeModelDL(
            params=_tiny_params(),
            config=_tiny_config(),
            bounds=_risk_free_bounds(),
            optimization_config=OptimizationConfig(use_xla=False),
        )
        # Prepare normalized inputs (3-D: capital, debt, productivity)
        k = tf.random.uniform((4, 1), 0.5, 2.0)
        b = tf.random.uniform((4, 1), 0.0, 1.0)
        z = tf.random.uniform((4, 1), 0.8, 1.2)
        k_norm = model.normalizer.normalize_capital(k)
        z_norm = model.normalizer.normalize_productivity(z)
        b_norm = model.normalizer.normalize_debt(b)
        inputs = tf.concat([k_norm, b_norm, z_norm], axis=1)

        v = model.value_net(inputs)
        k_pol = model.capital_policy_net(inputs)
        b_pol = model.debt_policy_net(inputs)
        mu = model.multiplier_net(inputs)

        assert v.shape == (4, 1)
        assert k_pol.shape == (4, 1)
        assert b_pol.shape == (4, 1)
        assert mu.shape == (4, 1)
        for name, tensor in [("v", v), ("k_pol", k_pol),
                              ("b_pol", b_pol), ("mu", mu)]:
            assert tf.reduce_all(tf.math.is_finite(tensor)), f"{name} not finite"
