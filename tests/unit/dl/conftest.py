"""Shared fixtures for DL tests."""

import pytest
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.training_config import EntropyConfig, OptimizationConfig


@pytest.fixture
def tiny_params():
    return EconomicParams(
        discount_factor=0.9, capital_share=0.3, depreciation_rate=0.1,
        productivity_persistence=0.9, productivity_std_dev=0.05,
        adjustment_cost_convex=1.0, adjustment_cost_fixed=0.0,
        equity_issuance_cost_fixed=0.0, equity_issuance_cost_linear=0.0,
        default_cost_proportional=0.3, corporate_tax_rate=0.2,
        risk_free_rate=0.04, collateral_recovery_fraction=0.5,
    )


@pytest.fixture
def tiny_dl_config():
    return DeepLearningConfig(
        batch_size=8, epochs=2, steps_per_epoch=3,
        hidden_layers=[16, 16], activation_function='relu',
        learning_rate=1e-3, mc_sample_number_bond_priceing=4,
        polyak_averaging_decay=0.995,
    )


@pytest.fixture
def tiny_bounds():
    return {
        'k_min': 0.1, 'k_max': 2.0,
        'z_min': 0.5, 'z_max': 1.5,
        'b_min': 0.0, 'b_max': 1.0,
        'rho_min': 0.8, 'rho_max': 0.95,
        'std_min': 0.01, 'std_max': 0.1,
        'convex_min': 0.5, 'convex_max': 2.0,
        'fixed_min': 0.0, 'fixed_max': 0.1,
        'eta0_min': 0.0, 'eta0_max': 0.05,
        'eta1_min': 0.0, 'eta1_max': 0.05,
    }


@pytest.fixture
def entropy_config():
    return EntropyConfig(
        initial_coeff=1.0, final_coeff=0.0,
        start_epoch=5, decay_epochs=10, capital_noise_std=0.05,
    )


@pytest.fixture
def optimization_config():
    return OptimizationConfig(use_xla=False)
