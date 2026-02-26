"""Shared test fixtures and helper utilities for VFI unit tests."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

# Force CPU for CI — must be called before any TF ops
tf.config.set_visible_devices([], 'GPU')

from econ_models.config.economic_params import EconomicParams


def make_test_params(**overrides) -> EconomicParams:
    """Return a minimal EconomicParams with sensible defaults for testing."""
    defaults = dict(
        discount_factor=0.96,
        capital_share=0.3,
        depreciation_rate=0.1,
        productivity_persistence=0.9,
        productivity_std_dev=0.05,
        adjustment_cost_convex=1.0,
        adjustment_cost_fixed=0.0,
        equity_issuance_cost_fixed=0.0,
        equity_issuance_cost_linear=0.0,
        default_cost_proportional=0.3,
        corporate_tax_rate=0.2,
        risk_free_rate=0.04,
        collateral_recovery_fraction=0.5,
    )
    defaults.update(overrides)
    return EconomicParams(**defaults)
