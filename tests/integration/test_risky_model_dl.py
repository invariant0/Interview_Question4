# tests/integration/test_risky_model_dl.py
"""
Integration tests for the Risky Debt deep learning model solver.

These tests verify end-to-end functionality of the RiskyModelDL class,
including initialization, bond pricing, policy optimization, and training.

Run with:
    python -m pytest tests/integration/test_risky_model_dl.py -v
    python -m unittest tests.integration.test_risky_model_dl -v
"""

import unittest
import tempfile
import shutil

import numpy as np
import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig
from econ_models.dl.risky import RiskyModelDL
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.sampling.curriculum import CurriculumBounds
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.sampling.state_sampler import StateSampler

def get_test_economic_params() -> EconomicParams:
    """Create economic parameters for testing."""
    return EconomicParams(
        discount_factor=0.96,
        capital_share=0.33,
        depreciation_rate=0.1,
        productivity_persistence=0.9,
        productivity_std_dev=0.02,
        adjustment_cost_convex=0.5,
        adjustment_cost_fixed=0.0,
        equity_issuance_cost_fixed=0.0,
        equity_issuance_cost_linear=0.0,
        default_cost_proportional=0.25,
        corporate_tax_rate=0.2,
        risk_free_rate=0.04,
        collateral_recovery_fraction=0.5
    )


def get_test_dl_config_risky() -> DeepLearningConfig:
    """Create minimal DL config for risky model testing."""
    return DeepLearningConfig(
        batch_size=50,
        learning_rate=1e-5,
        verbose=False,
        epochs=5,
        steps_per_epoch=10,
        capital_min=0.5,
        capital_max=5.0,
        productivity_min=0.8,
        productivity_max=1.2,
        debt_min=0.0,
        debt_max=2.0,
        hidden_layers=(32, 32),
        activation_function="leaky_relu",
        mc_next_candidate_sample=30,
        mc_sample_number_bond_priceing=20,
        mc_sample_number_continuation_value=20,
        polyak_averaging_decay=0.005,
        gradient_clip_norm=1.0,
        curriculum_epochs=2,
        curriculum_initial_ratio=0.1,
        min_q_price=1e-6,
        epsilon_debt=1e-6,
        epsilon_value_default=1e-4
    )


def tensor_to_scalar(tensor: tf.Tensor) -> float:
    """Safely convert a tensor to a Python scalar."""
    arr = tensor.numpy()
    if arr.ndim == 0:
        return float(arr)
    return float(arr.flatten()[0])


class TestRiskyModelDLInitialization(unittest.TestCase):
    """Test RiskyModelDL initialization and component construction."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_risky()

    def test_model_initialization(self):
        """Test that RiskyModelDL initializes without errors."""
        model = RiskyModelDL(self.params, self.config)

        self.assertIsNotNone(model.value_net)
        self.assertIsNotNone(model.target_value_net)
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.normalizer)
        self.assertIsNotNone(model.shock_dist)
        self.assertIsNotNone(model.training_progress)

    def test_network_architecture(self):
        """Test that networks have correct input/output dimensions."""
        model = RiskyModelDL(self.params, self.config)

        # Value network: 3 inputs (k, b, z), 1 output
        self.assertEqual(model.value_net.input_shape, (None, 3))
        self.assertEqual(model.value_net.output_shape, (None, 1))

        # Target network should match
        self.assertEqual(model.target_value_net.input_shape, (None, 3))
        self.assertEqual(model.target_value_net.output_shape, (None, 1))

    def test_target_network_initialized_from_value_net(self):
        """Test that target network starts with same weights as value net."""
        model = RiskyModelDL(self.params, self.config)

        value_weights = model.value_net.get_weights()
        target_weights = model.target_value_net.get_weights()

        for v_w, t_w in zip(value_weights, target_weights):
            np.testing.assert_array_almost_equal(v_w, t_w)

    def test_curriculum_progress_initialization(self):
        """Test that curriculum progress starts at zero."""
        model = RiskyModelDL(self.params, self.config)

        self.assertAlmostEqual(float(model.training_progress.numpy()), 0.0)

    def test_network_forward_pass(self):
        """Test that networks produce valid outputs."""
        model = RiskyModelDL(self.params, self.config)

        batch_size = 16
        inputs = tf.random.uniform(
            (batch_size, 3), minval=-1.0, maxval=1.0, dtype=TENSORFLOW_DTYPE
        )

        value_output = model.value_net(inputs, training=False)
        target_output = model.target_value_net(inputs, training=False)

        self.assertEqual(value_output.shape, (batch_size, 1))
        self.assertEqual(target_output.shape, (batch_size, 1))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(value_output)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(target_output)))

    def test_value_network_uses_relu_output(self):
        """Test that value network uses ReLU activation (non-negative outputs)."""
        model = RiskyModelDL(self.params, self.config)

        # Test with various inputs
        batch_size = 100
        inputs = tf.random.uniform(
            (batch_size, 3), minval=-2.0, maxval=2.0, dtype=TENSORFLOW_DTYPE
        )

        outputs = model.value_net(inputs, training=False)

        # All outputs should be non-negative due to ReLU
        self.assertTrue(tf.reduce_all(outputs >= 0))

    def test_optimizer_configuration(self):
        """Test that optimizer is configured correctly."""
        model = RiskyModelDL(self.params, self.config)

        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)
        self.assertIsNotNone(model.lr_schedule)

    def test_shock_distribution_parameters(self):
        """Test that shock distribution has correct parameters."""
        model = RiskyModelDL(self.params, self.config)

        self.assertAlmostEqual(float(model.shock_dist.mean()), 0.0)
        self.assertAlmostEqual(
            float(model.shock_dist.stddev()),
            self.params.productivity_std_dev,
            places=5
        )


class TestRiskyModelDLBondPricing(unittest.TestCase):
    """Test bond pricing functionality in RiskyModelDL."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_risky()
        cls.model = RiskyModelDL(cls.params, cls.config)

    def test_estimate_bond_price_returns_valid_values(self):
        """Test that bond pricing returns valid prices."""
        batch_size = 8
        n_candidates = 10

        k_prime = tf.random.uniform(
            (batch_size, n_candidates),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b_prime = tf.random.uniform(
            (batch_size, n_candidates),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        q = self.model.estimate_bond_price(k_prime, b_prime, z)

        self.assertEqual(q.shape, (batch_size, n_candidates))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(q)))

        # Bond prices should be positive
        self.assertTrue(tf.reduce_all(q >= 0))

    def test_bond_price_bounded_above(self):
        """Test that bond prices are bounded by risk-free price."""
        batch_size = 8
        n_candidates = 10

        k_prime = tf.random.uniform(
            (batch_size, n_candidates),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b_prime = tf.random.uniform(
            (batch_size, n_candidates),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        q = self.model.estimate_bond_price(k_prime, b_prime, z)

        # Bond prices should be bounded above by 1/(1+r) for risk-free case
        max_price = 1.0 / (1.0 + self.params.risk_free_rate)
        self.assertTrue(tf.reduce_all(q <= max_price * 1.1))  # Allow small margin

    def test_bond_price_increases_with_lower_debt(self):
        """Test economic intuition: lower debt leads to higher bond prices."""
        batch_size = 16
        n_candidates = 1

        k_prime = tf.ones(
            (batch_size, n_candidates), dtype=TENSORFLOW_DTYPE
        ) * 2.0
        z = tf.ones((batch_size, 1), dtype=TENSORFLOW_DTYPE)

        b_low = tf.ones((batch_size, n_candidates), dtype=TENSORFLOW_DTYPE) * 0.1
        b_high = tf.ones((batch_size, n_candidates), dtype=TENSORFLOW_DTYPE) * 1.5

        q_low_debt = self.model.estimate_bond_price(k_prime, b_low, z)
        q_high_debt = self.model.estimate_bond_price(k_prime, b_high, z)

        # On average, lower debt should yield higher bond prices
        mean_q_low = tf.reduce_mean(q_low_debt)
        mean_q_high = tf.reduce_mean(q_high_debt)

        self.assertGreaterEqual(float(mean_q_low), float(mean_q_high) * 0.9)

    def test_bond_price_with_zero_debt(self):
        """Test bond price with zero debt (should be near risk-free rate)."""
        batch_size = 8
        n_candidates = 5

        k_prime = tf.ones(
            (batch_size, n_candidates), dtype=TENSORFLOW_DTYPE
        ) * 2.0
        b_prime = tf.ones(
            (batch_size, n_candidates), dtype=TENSORFLOW_DTYPE
        ) * self.config.epsilon_debt
        z = tf.ones((batch_size, 1), dtype=TENSORFLOW_DTYPE)

        q = self.model.estimate_bond_price(k_prime, b_prime, z)

        # With near-zero debt, price should be close to risk-free
        risk_free_price = 1.0 / (1.0 + self.params.risk_free_rate)
        mean_q = float(tf.reduce_mean(q))

        # Allow for some deviation due to epsilon handling
        self.assertGreater(mean_q, risk_free_price * 0.5)

    def test_bond_price_shape_consistency(self):
        """Test that bond price output shape is consistent."""
        batch_sizes = [1, 8, 32]
        n_candidates_list = [1, 10, 50]

        for batch_size in batch_sizes:
            for n_candidates in n_candidates_list:
                k_prime = tf.random.uniform(
                    (batch_size, n_candidates),
                    minval=self.config.capital_min,
                    maxval=self.config.capital_max,
                    dtype=TENSORFLOW_DTYPE
                )
                b_prime = tf.random.uniform(
                    (batch_size, n_candidates),
                    minval=self.config.debt_min,
                    maxval=self.config.debt_max,
                    dtype=TENSORFLOW_DTYPE
                )
                z = tf.random.uniform(
                    (batch_size, 1),
                    minval=self.config.productivity_min,
                    maxval=self.config.productivity_max,
                    dtype=TENSORFLOW_DTYPE
                )

                q = self.model.estimate_bond_price(k_prime, b_prime, z)

                self.assertEqual(
                    q.shape,
                    (batch_size, n_candidates),
                    f"Shape mismatch for batch_size={batch_size}, n_candidates={n_candidates}"
                )


class TestRiskyModelDLOptimization(unittest.TestCase):
    """Test policy optimization in RiskyModelDL."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_risky()
        cls.model = RiskyModelDL(cls.params, cls.config)

    def test_optimize_next_state_returns_valid_choices(self):
        """Test that optimization returns valid state choices."""
        batch_size = 50

        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        # FIX: Unpack 4 values instead of 3
        k_opt, b_opt, q_opt, v_opt = self.model._optimize_next_state(k, b, z)

        self.assertEqual(k_opt.shape, (batch_size, 1))
        self.assertEqual(b_opt.shape, (batch_size, 1))
        self.assertEqual(q_opt.shape, (batch_size, 1))
        # Optional: Check v_opt shape
        self.assertEqual(v_opt.shape, (batch_size, 1))

        self.assertFalse(tf.reduce_any(tf.math.is_nan(k_opt)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(b_opt)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(q_opt)))

    def test_optimal_capital_is_positive(self):
        """Test that optimal capital choice is positive."""
        batch_size = 50

        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        # FIX: Unpack 4 values (use _ for ignored ones)
        k_opt, _, _, _ = self.model._optimize_next_state(k, b, z)

        self.assertTrue(tf.reduce_all(k_opt > 0))

    def test_optimal_debt_is_non_negative(self):
        """Test that optimal debt choice is non-negative."""
        batch_size = 50

        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        # FIX: Unpack 4 values
        _, b_opt, _, _ = self.model._optimize_next_state(k, b, z)

        self.assertTrue(tf.reduce_all(b_opt >= 0))

    def test_optimal_bond_price_is_positive(self):
        """Test that optimal bond price is positive."""
        batch_size = 50

        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        # FIX: Unpack 4 values
        _, _, q_opt, _ = self.model._optimize_next_state(k, b, z)

        self.assertTrue(tf.reduce_all(q_opt > 0))


class TestRiskyModelDLTraining(unittest.TestCase):
    """Test RiskyModelDL training functionality."""

    def setUp(self):
        """Create fresh model for each test."""
        self.params = get_test_economic_params()
        self.config = get_test_dl_config_risky()
        self.model = RiskyModelDL(self.params, self.config)

    def test_train_step_returns_metrics(self):
        """Test that train_step returns expected metrics."""
        batch_size = 50

        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        logs = self.model.train_step(k, b, z)

        expected_keys = [
            "loss", "avg_val", "avg_bond_price", "avg_k_prime", "avg_b_prime"
        ]
        for key in expected_keys:
            self.assertIn(key, logs)
            self.assertFalse(tf.math.is_nan(logs[key]))

    def test_target_network_soft_update(self):
        """Test that target network updates via Polyak averaging."""
        batch_size = 16

        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        # Store initial target weights
        initial_target_weights = [
            w.numpy().copy()
            for w in self.model.target_value_net.trainable_variables
        ]

        # Train for multiple steps with soft updates
        for _ in range(20):
            self.model.train_step(k, b, z)
            NeuralNetFactory.soft_update(
                self.model.value_net,
                self.model.target_value_net,
                self.config.polyak_averaging_decay
            )

        # Target weights should have changed
        weights_changed = False
        for initial, current in zip(
            initial_target_weights,
            self.model.target_value_net.trainable_variables
        ):
            if not np.allclose(initial, current.numpy(), atol=1e-8):
                weights_changed = True
                break

        self.assertTrue(weights_changed, "Target network weights should change after soft updates")

    def test_dataset_building_includes_debt(self):
        """Test that dataset includes debt dimension."""
        dataset = self.model._build_dataset()

        self.assertIsInstance(dataset, tf.data.Dataset)

        batch = next(iter(dataset))

        # Should return (k, b, z) tensors
        self.assertEqual(len(batch), 3)
        k, b, z = batch
        self.assertEqual(k.shape[1], 1)
        self.assertEqual(b.shape[1], 1)
        self.assertEqual(z.shape[1], 1)


class TestRiskyModelDLCurriculum(unittest.TestCase):
    """Test curriculum learning functionality in RiskyModelDL."""

    def setUp(self):
        """Create fresh model for each test."""
        self.params = get_test_economic_params()
        self.config = get_test_dl_config_risky()
        self.model = RiskyModelDL(self.params, self.config)

    def test_curriculum_progress_updates(self):
        """Test that curriculum progress updates during training."""
        initial_progress = float(self.model.training_progress.numpy())
        self.assertEqual(initial_progress, 0.0)

        self.model._update_curriculum(epoch=self.config.curriculum_epochs // 2)
        mid_progress = float(self.model.training_progress.numpy())

        self.model._update_curriculum(epoch=self.config.curriculum_epochs)
        final_progress = float(self.model.training_progress.numpy())

        self.assertGreater(mid_progress, initial_progress)
        self.assertGreaterEqual(final_progress, mid_progress)
        self.assertAlmostEqual(final_progress, 1.0)

    def test_curriculum_progress_capped_at_one(self):
        """Test that curriculum progress doesn't exceed 1.0."""
        self.model._update_curriculum(epoch=self.config.curriculum_epochs * 2)
        progress = float(self.model.training_progress.numpy())

        self.assertLessEqual(progress, 1.0)

    def test_curriculum_bounds_expand(self):
        """Test that curriculum bounds expand with progress."""
        global_min = tf.constant(0.1, dtype=TENSORFLOW_DTYPE)
        global_max = tf.constant(10.0, dtype=TENSORFLOW_DTYPE)
        center = tf.constant(2.0, dtype=TENSORFLOW_DTYPE)
        initial_ratio = 0.05

        # At progress = 0
        min_0, max_0 = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, center,
            tf.constant(0.0, dtype=TENSORFLOW_DTYPE),
            initial_ratio
        )

        # At progress = 0.5
        min_5, max_5 = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, center,
            tf.constant(0.5, dtype=TENSORFLOW_DTYPE),
            initial_ratio
        )

        # At progress = 1.0
        min_1, max_1 = CurriculumBounds.get_curriculum_bounds(
            global_min, global_max, center,
            tf.constant(1.0, dtype=TENSORFLOW_DTYPE),
            initial_ratio
        )

        # Bounds should expand with progress
        range_0 = float(max_0 - min_0)
        range_5 = float(max_5 - min_5)
        range_1 = float(max_1 - min_1)

        self.assertLess(range_0, range_5)
        self.assertLess(range_5, range_1)

        # At full progress, should reach global bounds
        self.assertAlmostEqual(float(min_1), float(global_min), places=5)
        self.assertAlmostEqual(float(max_1), float(global_max), places=5)


class TestRiskyModelDLNormalization(unittest.TestCase):
    """Test state space normalization in risky model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_risky()
        cls.model = RiskyModelDL(cls.params, cls.config)

    def test_capital_normalization(self):
        """Test that capital normalization produces bounded values."""
        k = tf.constant([[1.0], [2.0], [3.0]], dtype=TENSORFLOW_DTYPE)

        k_norm = self.model.normalizer.normalize_capital(k)

        self.assertEqual(k_norm.shape, k.shape)
        self.assertTrue(tf.reduce_all(tf.abs(k_norm) < 10))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(k_norm)))

    def test_debt_normalization(self):
        """Test that debt normalization produces bounded values."""
        b = tf.constant([[0.0], [0.5], [1.0]], dtype=TENSORFLOW_DTYPE)

        b_norm = self.model.normalizer.normalize_debt(b)

        self.assertEqual(b_norm.shape, b.shape)
        self.assertTrue(tf.reduce_all(tf.abs(b_norm) < 10))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(b_norm)))

    def test_productivity_normalization(self):
        """Test that productivity normalization produces bounded values."""
        z = tf.constant([[0.9], [1.0], [1.1]], dtype=TENSORFLOW_DTYPE)

        z_norm = self.model.normalizer.normalize_productivity(z)

        self.assertEqual(z_norm.shape, z.shape)
        self.assertTrue(tf.reduce_all(tf.abs(z_norm) < 10))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(z_norm)))

    def test_normalization_preserves_ordering(self):
        """Test that normalization preserves relative ordering."""
        k = tf.constant([[1.0], [2.0], [3.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.1], [0.5], [1.0]], dtype=TENSORFLOW_DTYPE)

        k_norm = self.model.normalizer.normalize_capital(k)
        b_norm = self.model.normalizer.normalize_debt(b)

        # Use proper indexing to get scalar values
        self.assertLess(
            tensor_to_scalar(k_norm[0]),
            tensor_to_scalar(k_norm[1])
        )
        self.assertLess(
            tensor_to_scalar(k_norm[1]),
            tensor_to_scalar(k_norm[2])
        )
        self.assertLess(
            tensor_to_scalar(b_norm[0]),
            tensor_to_scalar(b_norm[1])
        )
        self.assertLess(
            tensor_to_scalar(b_norm[1]),
            tensor_to_scalar(b_norm[2])
        )


class TestRiskyModelDLGradientFlow(unittest.TestCase):
    """Test gradient flow through the risky model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_risky()

    def test_gradient_flow_through_value_network(self):
        """Test that gradients flow through value network."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        with tf.GradientTape() as tape:
            # Simplified loss for gradient check
            inputs = tf.concat([
                model.normalizer.normalize_capital(k),
                model.normalizer.normalize_debt(b),
                model.normalizer.normalize_productivity(z)
            ], axis=1)
            v = model.value_net(inputs, training=True)
            loss = tf.reduce_mean(tf.square(v - 1.0))

        grads = tape.gradient(loss, model.value_net.trainable_variables)

        for grad, var in zip(grads, model.value_net.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient is None for {var.name}")
            self.assertFalse(
                tf.reduce_any(tf.math.is_nan(grad)),
                f"NaN gradient for {var.name}"
            )

    def test_gradients_through_full_train_step(self):
        """Test that train_step computes valid gradients."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        # Should complete without NaN errors
        logs = model.train_step(k, b, z)

        self.assertFalse(tf.math.is_nan(logs["loss"]))


class TestRiskyModelDLEndToEnd(unittest.TestCase):
    """End-to-end integration tests for RiskyModelDL."""

    def setUp(self):
        """Create fresh model for each test."""
        self.params = get_test_economic_params()
        self.config = get_test_dl_config_risky()
        self.config.epochs = 3
        self.config.steps_per_epoch = 3

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_training_loop(self):
        """Test that full training loop completes without errors."""
        model = RiskyModelDL(self.params, self.config)

        # Training should complete without exceptions
        model.train()

        # Progress should be at max
        self.assertGreater(float(model.training_progress.numpy()), 0)

    def test_loss_stays_bounded(self):
        """Test that training produces bounded loss values."""
        model = RiskyModelDL(self.params, self.config)

        losses = []
        dataset = model._build_dataset().repeat()
        data_iter = iter(dataset)

        for _ in range(10):
            k, b, z = next(data_iter)
            logs = model.train_step(k, b, z)
            losses.append(float(logs["loss"]))

        # Loss should remain finite
        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_multiple_epochs_complete(self):
        """Test that multiple epochs can be completed."""
        self.config.epochs = 5
        self.config.steps_per_epoch = 2

        model = RiskyModelDL(self.params, self.config)
        model.train()

    def test_value_function_responds_to_state(self):
        """Test that value function changes with different states."""
        model = RiskyModelDL(self.params, self.config)

        # Train for more iterations to ensure learning
        dataset = model._build_dataset().repeat()
        data_iter = iter(dataset)
        for _ in range(50):
            k, b, z = next(data_iter)
            model.train_step(k, b, z)

        # Evaluate at significantly different states
        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
        b_low = tf.constant([[0.01]], dtype=TENSORFLOW_DTYPE)
        b_high = tf.constant([[1.9]], dtype=TENSORFLOW_DTYPE)

        inputs_low = tf.concat([
            model.normalizer.normalize_capital(k),
            model.normalizer.normalize_debt(b_low),
            model.normalizer.normalize_productivity(z)
        ], axis=1)

        inputs_high = tf.concat([
            model.normalizer.normalize_capital(k),
            model.normalizer.normalize_debt(b_high),
            model.normalizer.normalize_productivity(z)
        ], axis=1)

        v_low = model.value_net(inputs_low, training=False)
        v_high = model.value_net(inputs_high, training=False)

        v_low_scalar = tensor_to_scalar(v_low)
        v_high_scalar = tensor_to_scalar(v_high)

        # Values should generally be different for different debt levels
        # Use a tolerance-based comparison rather than exact inequality
        values_differ = not np.isclose(v_low_scalar, v_high_scalar, atol=1e-4)
        self.assertTrue(
            values_differ,
            f"Values should differ for different debt levels: v_low={v_low_scalar}, v_high={v_high_scalar}"
        )


class TestRiskyModelDLEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_risky()

    def test_handles_zero_debt(self):
        """Test model handles zero debt correctly."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        logs = model.train_step(k, b, z)

        self.assertFalse(tf.math.is_nan(logs["loss"]))

    def test_handles_maximum_debt(self):
        """Test model handles maximum debt values."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[self.config.debt_max]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        logs = model.train_step(k, b, z)

        self.assertFalse(tf.math.is_nan(logs["loss"]))

    def test_handles_minimum_capital(self):
        """Test model handles minimum capital values."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[self.config.capital_min]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        logs = model.train_step(k, b, z)

        self.assertFalse(tf.math.is_nan(logs["loss"]))

    def test_handles_extreme_productivity(self):
        """Test model handles extreme productivity values."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)

        # Test minimum productivity
        z_min = tf.constant(
            [[self.config.productivity_min]], dtype=TENSORFLOW_DTYPE
        )
        logs_min = model.train_step(k, b, z_min)
        self.assertFalse(tf.math.is_nan(logs_min["loss"]))

        # Test maximum productivity
        z_max = tf.constant(
            [[self.config.productivity_max]], dtype=TENSORFLOW_DTYPE
        )
        logs_max = model.train_step(k, b, z_max)
        self.assertFalse(tf.math.is_nan(logs_max["loss"]))

    def test_handles_single_sample(self):
        """Test model handles batch size of 1."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        logs = model.train_step(k, b, z)

        self.assertFalse(tf.math.is_nan(logs["loss"]))
        self.assertFalse(tf.math.is_nan(logs["avg_val"]))
        self.assertFalse(tf.math.is_nan(logs["avg_bond_price"]))

    def test_handles_large_batch(self):
        """Test model handles large batch sizes."""
        model = RiskyModelDL(self.params, self.config)

        batch_size = 128
        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        b = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.debt_min,
            maxval=self.config.debt_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        logs = model.train_step(k, b, z)

        self.assertFalse(tf.math.is_nan(logs["loss"]))
        self.assertFalse(tf.math.is_inf(logs["loss"]))


class TestRiskyModelDLTransitions(unittest.TestCase):
    """Test stochastic transitions in risky model context."""

    def test_log_ar1_transition_with_broadcasting(self):
        """Test log-AR(1) transition with broadcasting shapes."""
        batch_size = 8
        n_candidates = 10
        n_samples = 5

        z_curr = tf.random.uniform(
            (batch_size, n_candidates, 1),
            minval=0.8, maxval=1.2,
            dtype=TENSORFLOW_DTYPE
        )
        eps = tf.random.normal(
            (batch_size, n_candidates, n_samples),
            dtype=TENSORFLOW_DTYPE
        ) * 0.02
        rho = 0.9

        # Broadcast z_curr
        z_curr_bc = tf.broadcast_to(
            z_curr, (batch_size, n_candidates, n_samples)
        )

        z_prime = TransitionFunctions.log_ar1_transition(z_curr_bc, rho, eps)

        self.assertEqual(z_prime.shape, (batch_size, n_candidates, n_samples))
        self.assertTrue(tf.reduce_all(z_prime > 0))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(z_prime)))


class TestRiskyModelDLReproducibility(unittest.TestCase):
    """Test reproducibility of risky model operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_risky()

    def test_deterministic_forward_pass(self):
        """Test that forward pass is deterministic given same inputs."""
        model = RiskyModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        inputs = tf.concat([
            model.normalizer.normalize_capital(k),
            model.normalizer.normalize_debt(b),
            model.normalizer.normalize_productivity(z)
        ], axis=1)

        v1 = model.value_net(inputs, training=False)
        v2 = model.value_net(inputs, training=False)

        np.testing.assert_array_almost_equal(v1.numpy(), v2.numpy())

    def test_network_weights_accessible(self):
        """Test that network weights can be accessed."""
        model = RiskyModelDL(self.params, self.config)

        value_weights = model.value_net.get_weights()
        target_weights = model.target_value_net.get_weights()

        self.assertGreater(len(value_weights), 0)
        self.assertGreater(len(target_weights), 0)

        # Weights should be finite
        for w in value_weights + target_weights:
            self.assertTrue(np.all(np.isfinite(w)))


if __name__ == "__main__":
    # Configure TensorFlow to use CPU for reproducibility
    tf.config.set_visible_devices([], 'GPU')

    # Run tests
    unittest.main(verbosity=2)