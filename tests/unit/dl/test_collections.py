"""Unit tests for NetworkCollection and OptimizerCollection."""

import pytest
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from econ_models.dl.components.network_collection import (
    NetworkCollectionBuilder,
    BASIC_INVESTMENT_SPEC,
    RISKY_SPEC,
)
from econ_models.dl.components.optimizer_collection import (
    OptimizerCollectionBuilder,
    BASIC_INVESTMENT_OPTIMIZER_SPEC,
    RISKY_OPTIMIZER_SPEC,
)


def _make_dl_config(**overrides):
    """Create a minimal DeepLearningConfig for testing."""
    from econ_models.config.dl_config import DeepLearningConfig
    cfg = DeepLearningConfig()
    cfg.hidden_layers = (32, 32)
    cfg.activation_function = 'tanh'
    cfg.value_scale_factor = 1.0
    cfg.learning_rate = 1e-3
    cfg.epochs = 10
    cfg.steps_per_epoch = 5
    cfg.use_layer_norm = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestNetworkCollection:
    """Tests for spec-driven network building."""

    def test_basic_spec_builds(self):
        config = _make_dl_config()
        coll = NetworkCollectionBuilder.build(BASIC_INVESTMENT_SPEC, input_dim=2, config=config)
        # NetworkCollection uses __getattr__ — access nets directly
        assert coll.value_net is not None
        assert coll.capital_policy_net is not None
        assert coll.investment_policy_net is not None

    def test_basic_spec_has_target(self):
        config = _make_dl_config()
        coll = NetworkCollectionBuilder.build(BASIC_INVESTMENT_SPEC, input_dim=2, config=config)
        pairs = coll.target_pairs
        # value_net has a target
        assert len(pairs) >= 1

    def test_risky_spec_builds(self):
        config = _make_dl_config()
        coll = NetworkCollectionBuilder.build(RISKY_SPEC, input_dim=3, config=config)
        expected_names = [
            'value_net', 'continuous_net', 'capital_policy_net',
            'debt_policy_net',
            'investment_decision_net', 'default_policy_net',
        ]
        all_nets = coll.all_networks
        for name in expected_names:
            assert name in all_nets, f"Missing network: {name}"

    def test_risky_spec_target_pairs(self):
        config = _make_dl_config()
        coll = NetworkCollectionBuilder.build(RISKY_SPEC, input_dim=3, config=config)
        pairs = coll.target_pairs
        # value_net and continuous_net have targets
        assert len(pairs) >= 2

    def test_trainable_variables_by_group(self):
        config = _make_dl_config()
        coll = NetworkCollectionBuilder.build(BASIC_INVESTMENT_SPEC, input_dim=2, config=config)
        # Force build by calling each network
        dummy = tf.zeros((1, 2))
        for name, net in coll.all_networks.items():
            net(dummy)
        # Now check variable groups
        for group_spec in BASIC_INVESTMENT_SPEC.variable_groups:
            variables = coll.trainable_variables(group_spec.name)
            assert len(variables) > 0, f"No variables for group {group_spec.name}"


class TestOptimizerCollection:
    """Tests for spec-driven optimizer building."""

    def test_basic_spec_builds(self):
        config = _make_dl_config()
        coll = OptimizerCollectionBuilder.build(
            BASIC_INVESTMENT_OPTIMIZER_SPEC, config=config,
        )
        # Should have 'value' and 'policy' optimizer groups
        all_opts = coll.all_optimizers
        assert 'value' in all_opts
        assert 'policy' in all_opts

    def test_risky_spec_builds(self):
        config = _make_dl_config()
        coll = OptimizerCollectionBuilder.build(
            RISKY_OPTIMIZER_SPEC, config=config,
        )
        expected = ['value', 'continuous', 'capital_debt', 'invest', 'default']
        all_opts = coll.all_optimizers
        for name in expected:
            assert name in all_opts, f"Missing optimizer group: {name}"
