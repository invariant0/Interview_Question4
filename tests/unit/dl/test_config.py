"""Unit tests for EconomicParams and DeepLearningConfig."""

from __future__ import annotations

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig


# ─── EconomicParams ───


class TestEconomicParams:
    """Tests for EconomicParams frozen dataclass."""

    def test_valid_construction(self):
        p = EconomicParams(
            discount_factor=0.96,
            capital_share=0.3,
            depreciation_rate=0.1,
            productivity_persistence=0.9,
            productivity_std_dev=0.05,
        )
        assert p.discount_factor == 0.96
        assert p.capital_share == 0.3

    def test_discount_factor_too_low(self):
        with pytest.raises(ValueError, match="Discount factor must be in"):
            EconomicParams(discount_factor=0.0)

    def test_discount_factor_too_high(self):
        with pytest.raises(ValueError, match="Discount factor must be in"):
            EconomicParams(discount_factor=1.0)

    def test_discount_factor_negative(self):
        with pytest.raises(ValueError, match="Discount factor must be in"):
            EconomicParams(discount_factor=-0.5)

    def test_frozen_immutability(self):
        p = EconomicParams(discount_factor=0.96)
        with pytest.raises(AttributeError):
            p.discount_factor = 0.99

    def test_optional_fields_default_none(self):
        p = EconomicParams(discount_factor=0.96)
        assert p.capital_share is None
        assert p.corporate_tax_rate is None


# ─── DeepLearningConfig ───


class TestDeepLearningConfig:
    """Tests for DeepLearningConfig mutable dataclass."""

    def test_construction_defaults_none(self):
        cfg = DeepLearningConfig()
        assert cfg.batch_size is None
        assert cfg.learning_rate is None
        assert cfg.epochs is None

    def test_mutation_allowed(self):
        cfg = DeepLearningConfig()
        cfg.batch_size = 256
        assert cfg.batch_size == 256

    def test_update_value_scale(self):
        params = EconomicParams(
            discount_factor=0.96,
            capital_share=0.3,
            depreciation_rate=0.1,
            corporate_tax_rate=0.2,
        )
        cfg = DeepLearningConfig()
        cfg.update_value_scale(params)
        assert cfg.value_scale_factor is not None
        assert cfg.value_scale_factor > 0

    def test_domain_boundaries_settable(self):
        cfg = DeepLearningConfig()
        cfg.capital_min = 0.1
        cfg.capital_max = 10.0
        cfg.productivity_min = 0.3
        cfg.productivity_max = 2.0
        assert cfg.capital_max > cfg.capital_min
        assert cfg.productivity_max > cfg.productivity_min
