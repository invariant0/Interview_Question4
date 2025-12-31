# tests/integration/test_basic_model_dl.py
"""
Integration tests for the Basic RBC deep learning model solver.

These tests verify end-to-end functionality of the BasicModelDL class,
including initialization, forward passes, training steps, and convergence.

Run with:
    python -m pytest tests/integration/test_basic_model_dl.py -v
    python -m unittest tests.integration.test_basic_model_dl -v
"""

import unittest
import tempfile
import shutil

import numpy as np
import tensorflow as tf

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig
from econ_models.dl.basic import BasicModelDL
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.types import TENSORFLOW_DTYPE


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


def get_test_dl_config_basic() -> DeepLearningConfig:
    """Create minimal DL config for basic model testing."""
    return DeepLearningConfig(
        batch_size=32,
        learning_rate=1e-3,
        verbose=False,
        epochs=5,
        steps_per_epoch=10,
        capital_min=0.5,
        capital_max=5.0,
        productivity_min=0.8,
        productivity_max=1.2,
        debt_min=0.0,
        debt_max=0.0,
        hidden_layers=(32, 32),
        activation_function="swish",
        euler_residual_weight=100.0,
        curriculum_epochs=2,
        curriculum_initial_ratio=0.1
    )


def tensor_to_scalar(tensor: tf.Tensor) -> float:
    """Safely convert a tensor to a Python scalar."""
    arr = tensor.numpy()
    if arr.ndim == 0:
        return float(arr)
    return float(arr.flatten()[0])


class TestBasicModelDLInitialization(unittest.TestCase):
    """Test BasicModelDL initialization and component construction."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_basic()

    def test_model_initialization(self):
        """Test that BasicModelDL initializes without errors."""
        model = BasicModelDL(self.params, self.config)

        self.assertIsNotNone(model.value_net)
        self.assertIsNotNone(model.policy_net)
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.normalizer)
        self.assertIsNotNone(model.shock_dist)

    def test_network_architecture(self):
        """Test that networks have correct input/output dimensions."""
        model = BasicModelDL(self.params, self.config)

        # Value network: 2 inputs (k, z), 1 output
        self.assertEqual(model.value_net.input_shape, (None, 2))
        self.assertEqual(model.value_net.output_shape, (None, 1))

        # Policy network: 2 inputs (k, z), 1 output
        self.assertEqual(model.policy_net.input_shape, (None, 2))
        self.assertEqual(model.policy_net.output_shape, (None, 1))

    def test_network_forward_pass(self):
        """Test that networks produce valid outputs."""
        model = BasicModelDL(self.params, self.config)

        batch_size = 16
        inputs = tf.random.uniform(
            (batch_size, 2), minval=-1.0, maxval=1.0, dtype=TENSORFLOW_DTYPE
        )

        value_output = model.value_net(inputs, training=False)
        policy_output = model.policy_net(inputs, training=False)

        self.assertEqual(value_output.shape, (batch_size, 1))
        self.assertEqual(policy_output.shape, (batch_size, 1))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(value_output)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(policy_output)))

    def test_value_scale_factor_updated(self):
        """Test that value scale factor is computed during init."""
        model = BasicModelDL(self.params, self.config)

        self.assertIsNotNone(self.config.value_scale_factor)
        self.assertGreater(self.config.value_scale_factor, 0)
        self.assertIsNotNone(self.config.capital_steady_state)
        self.assertGreater(self.config.capital_steady_state, 0)

    def test_optimizer_has_learning_rate_schedule(self):
        """Test that optimizer uses learning rate schedule."""
        model = BasicModelDL(self.params, self.config)

        self.assertIsNotNone(model.lr_schedule)
        self.assertIsInstance(
            model.lr_schedule,
            tf.keras.optimizers.schedules.ExponentialDecay
        )

    def test_shock_distribution_parameters(self):
        """Test that shock distribution has correct parameters."""
        model = BasicModelDL(self.params, self.config)

        # Check distribution mean is 0
        self.assertAlmostEqual(float(model.shock_dist.mean()), 0.0)

        # Check distribution std dev matches params
        self.assertAlmostEqual(
            float(model.shock_dist.stddev()),
            self.params.productivity_std_dev,
            places=5
        )


class TestBasicModelDLTraining(unittest.TestCase):
    """Test BasicModelDL training functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_basic()
        cls.model = BasicModelDL(cls.params, cls.config)

    def test_compute_loss_returns_valid_tensors(self):
        """Test that compute_loss returns finite loss values."""
        batch_size = 32
        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        total_loss, bellman_loss, euler_loss = self.model.compute_loss(k, z)

        self.assertEqual(total_loss.shape, ())
        self.assertFalse(tf.math.is_nan(total_loss))
        self.assertFalse(tf.math.is_inf(total_loss))
        self.assertFalse(tf.math.is_nan(bellman_loss))
        self.assertFalse(tf.math.is_nan(euler_loss))

    def test_compute_loss_returns_three_components(self):
        """Test that compute_loss returns total, Bellman, and Euler losses."""
        batch_size = 16
        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        result = self.model.compute_loss(k, z)

        self.assertEqual(len(result), 3)
        total_loss, bellman_loss, euler_loss = result

        # Total loss should combine Bellman and Euler
        expected_total = bellman_loss + self.config.euler_residual_weight * euler_loss
        self.assertAlmostEqual(
            float(total_loss),
            float(expected_total),
            places=4
        )

    def test_train_step_updates_weights(self):
        """Test that train_step modifies network weights."""
        # Create fresh model for this test
        model = BasicModelDL(self.params, self.config)

        batch_size = 32
        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        # Store initial weights
        initial_value_weights = [
            w.numpy().copy() for w in model.value_net.trainable_variables
        ]
        initial_policy_weights = [
            w.numpy().copy() for w in model.policy_net.trainable_variables
        ]

        # Perform multiple training steps to ensure weight updates
        for _ in range(5):
            logs = model.train_step(k, z)

        # Verify logs contain expected keys
        self.assertIn("loss", logs)
        self.assertIn("bellman_loss", logs)
        self.assertIn("euler_loss", logs)

        # Verify weights changed (at least some)
        value_weights_changed = False
        for initial, current in zip(
            initial_value_weights, model.value_net.trainable_variables
        ):
            if not np.allclose(initial, current.numpy(), atol=1e-10):
                value_weights_changed = True
                break

        policy_weights_changed = False
        for initial, current in zip(
            initial_policy_weights, model.policy_net.trainable_variables
        ):
            if not np.allclose(initial, current.numpy(), atol=1e-10):
                policy_weights_changed = True
                break

        self.assertTrue(value_weights_changed or policy_weights_changed)

    def test_train_step_returns_dict(self):
        """Test that train_step returns a dictionary of metrics."""
        batch_size = 16
        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        logs = self.model.train_step(k, z)

        self.assertIsInstance(logs, dict)
        self.assertIn("loss", logs)
        self.assertIn("bellman_loss", logs)
        self.assertIn("euler_loss", logs)

    def test_dataset_building(self):
        """Test that dataset is built correctly."""
        dataset = self.model._build_dataset()

        self.assertIsInstance(dataset, tf.data.Dataset)

        # Get one batch
        batch = next(iter(dataset))

        # Should return (k, z) tensors
        self.assertEqual(len(batch), 2)
        k, z = batch
        self.assertEqual(k.shape[1], 1)
        self.assertEqual(z.shape[1], 1)

    def test_dataset_values_in_bounds(self):
        """Test that dataset produces values within configured bounds."""
        dataset = self.model._build_dataset()

        for k, z in dataset.take(5):
            self.assertTrue(
                tf.reduce_all(k >= self.config.capital_min * 0.9)
            )
            self.assertTrue(
                tf.reduce_all(k <= self.config.capital_max * 1.1)
            )
            self.assertTrue(
                tf.reduce_all(z >= self.config.productivity_min * 0.9)
            )
            self.assertTrue(
                tf.reduce_all(z <= self.config.productivity_max * 1.1)
            )


class TestBasicModelDLGradientFlow(unittest.TestCase):
    """Test gradient flow through the basic model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_basic()

    def test_gradient_flow_through_value_network(self):
        """Test that gradients flow through value network."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        with tf.GradientTape() as tape:
            loss, _, _ = model.compute_loss(k, z)

        grads = tape.gradient(loss, model.value_net.trainable_variables)

        for grad, var in zip(grads, model.value_net.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient is None for {var.name}")
            self.assertFalse(
                tf.reduce_any(tf.math.is_nan(grad)),
                f"NaN gradient for {var.name}"
            )

    def test_gradient_flow_through_policy_network(self):
        """Test that gradients flow through policy network."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        with tf.GradientTape() as tape:
            loss, _, _ = model.compute_loss(k, z)

        grads = tape.gradient(loss, model.policy_net.trainable_variables)

        for grad, var in zip(grads, model.policy_net.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient is None for {var.name}")
            self.assertFalse(
                tf.reduce_any(tf.math.is_nan(grad)),
                f"NaN gradient for {var.name}"
            )

    def test_combined_gradient_flow(self):
        """Test that gradients flow through both networks."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        with tf.GradientTape() as tape:
            loss, _, _ = model.compute_loss(k, z)

        all_vars = (
            model.value_net.trainable_variables
            + model.policy_net.trainable_variables
        )
        grads = tape.gradient(loss, all_vars)

        # All gradients should be non-None
        for grad, var in zip(grads, all_vars):
            self.assertIsNotNone(grad, f"Gradient is None for {var.name}")


class TestBasicModelDLNormalization(unittest.TestCase):
    """Test state space normalization in basic model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_basic()
        cls.model = BasicModelDL(cls.params, cls.config)

    def test_capital_normalization(self):
        """Test that capital normalization produces bounded values."""
        k = tf.constant([[1.0], [2.0], [3.0]], dtype=TENSORFLOW_DTYPE)

        k_norm = self.model.normalizer.normalize_capital(k)

        self.assertEqual(k_norm.shape, k.shape)
        self.assertTrue(tf.reduce_all(tf.abs(k_norm) < 10))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(k_norm)))

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

        k_norm = self.model.normalizer.normalize_capital(k)

        # Use proper indexing to get scalar values
        self.assertLess(
            tensor_to_scalar(k_norm[0]),
            tensor_to_scalar(k_norm[1])
        )
        self.assertLess(
            tensor_to_scalar(k_norm[1]),
            tensor_to_scalar(k_norm[2])
        )


class TestBasicModelDLEndToEnd(unittest.TestCase):
    """End-to-end integration tests for BasicModelDL."""

    def setUp(self):
        """Create fresh model for each test."""
        self.params = get_test_economic_params()
        self.config = get_test_dl_config_basic()
        self.config.epochs = 3
        self.config.steps_per_epoch = 5

        # Create temp directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_training_loop(self):
        """Test that full training loop completes without errors."""
        model = BasicModelDL(self.params, self.config)

        # Training should complete without raising exceptions
        model.train()

    def test_loss_decreases_or_stays_bounded(self):
        """Test that training produces bounded loss values."""
        model = BasicModelDL(self.params, self.config)

        losses = []
        dataset = model._build_dataset().repeat()
        data_iter = iter(dataset)

        for _ in range(20):
            k, z = next(data_iter)
            logs = model.train_step(k, z)
            losses.append(float(logs["loss"]))

        self.assertTrue(all(np.isfinite(loss) for loss in losses))
        self.assertLess(max(losses), 1e10)

    def test_multiple_epochs_complete(self):
        """Test that multiple epochs can be completed."""
        self.config.epochs = 5
        self.config.steps_per_epoch = 3

        model = BasicModelDL(self.params, self.config)
        model.train()

    def test_value_function_responds_to_state(self):
        """Test that value function changes with different states."""
        model = BasicModelDL(self.params, self.config)

        # Train for more iterations to ensure learning
        dataset = model._build_dataset().repeat()
        data_iter = iter(dataset)
        for _ in range(50):
            k, z = next(data_iter)
            model.train_step(k, z)

        # Evaluate at significantly different states
        k_low = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)
        k_high = tf.constant([[4.5]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        inputs_low = tf.concat([
            model.normalizer.normalize_capital(k_low),
            model.normalizer.normalize_productivity(z)
        ], axis=1)

        inputs_high = tf.concat([
            model.normalizer.normalize_capital(k_high),
            model.normalizer.normalize_productivity(z)
        ], axis=1)

        v_low = model.value_net(inputs_low, training=False)
        v_high = model.value_net(inputs_high, training=False)

        v_low_scalar = tensor_to_scalar(v_low)
        v_high_scalar = tensor_to_scalar(v_high)

        # Values should be different for different capital levels
        # Allow for small numerical tolerance
        self.assertFalse(
            np.isclose(v_low_scalar, v_high_scalar, atol=1e-4),
            f"Values should differ: v_low={v_low_scalar}, v_high={v_high_scalar}"
        )


class TestBasicModelDLTransitions(unittest.TestCase):
    """Test stochastic transitions in basic model context."""

    def test_log_ar1_transition_shape(self):
        """Test that log-AR(1) transition preserves shape."""
        batch_size = 32
        z = tf.random.uniform(
            (batch_size, 1), minval=0.8, maxval=1.2, dtype=TENSORFLOW_DTYPE
        )
        eps = tf.random.normal((batch_size, 1), dtype=TENSORFLOW_DTYPE) * 0.02
        rho = 0.9

        z_prime = TransitionFunctions.log_ar1_transition(z, rho, eps)

        self.assertEqual(z_prime.shape, z.shape)

    def test_log_ar1_transition_positive(self):
        """Test that log-AR(1) transition produces positive values."""
        batch_size = 100
        z = tf.random.uniform(
            (batch_size, 1), minval=0.5, maxval=1.5, dtype=TENSORFLOW_DTYPE
        )
        eps = tf.random.normal((batch_size, 1), dtype=TENSORFLOW_DTYPE) * 0.05
        rho = 0.9

        z_prime = TransitionFunctions.log_ar1_transition(z, rho, eps)

        self.assertTrue(tf.reduce_all(z_prime > 0))

    def test_log_ar1_transition_mean_reversion(self):
        """Test that log-AR(1) with rho < 1 shows mean reversion."""
        z_high = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        z_low = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)
        eps = tf.constant([[0.0]], dtype=TENSORFLOW_DTYPE)
        rho = 0.9

        z_high_prime = TransitionFunctions.log_ar1_transition(z_high, rho, eps)
        z_low_prime = TransitionFunctions.log_ar1_transition(z_low, rho, eps)

        z_high_val = tensor_to_scalar(z_high)
        z_high_prime_val = tensor_to_scalar(z_high_prime)
        z_low_val = tensor_to_scalar(z_low)
        z_low_prime_val = tensor_to_scalar(z_low_prime)

        # High values should move toward 1
        self.assertLess(z_high_prime_val, z_high_val)
        # Low values should move toward 1
        self.assertGreater(z_low_prime_val, z_low_val)

    def test_transition_with_zero_shock(self):
        """Test transition behavior with zero shock."""
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
        eps = tf.constant([[0.0]], dtype=TENSORFLOW_DTYPE)
        rho = 0.9

        z_prime = TransitionFunctions.log_ar1_transition(z, rho, eps)

        z_prime_val = tensor_to_scalar(z_prime)

        # With z=1 and eps=0, ln(z)=0, so z_prime should be exp(0)=1
        self.assertAlmostEqual(z_prime_val, 1.0, places=5)


class TestBasicModelDLEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_basic()

    def test_handles_minimum_capital(self):
        """Test model handles minimum capital values."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[self.config.capital_min]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        loss, _, _ = model.compute_loss(k, z)

        self.assertFalse(tf.math.is_nan(loss))
        self.assertFalse(tf.math.is_inf(loss))

    def test_handles_maximum_capital(self):
        """Test model handles maximum capital values."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[self.config.capital_max]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        loss, _, _ = model.compute_loss(k, z)

        self.assertFalse(tf.math.is_nan(loss))
        self.assertFalse(tf.math.is_inf(loss))

    def test_handles_extreme_productivity(self):
        """Test model handles extreme productivity values."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)

        # Test minimum productivity
        z_min = tf.constant([[self.config.productivity_min]], dtype=TENSORFLOW_DTYPE)
        loss_min, _, _ = model.compute_loss(k, z_min)
        self.assertFalse(tf.math.is_nan(loss_min))

        # Test maximum productivity
        z_max = tf.constant([[self.config.productivity_max]], dtype=TENSORFLOW_DTYPE)
        loss_max, _, _ = model.compute_loss(k, z_max)
        self.assertFalse(tf.math.is_nan(loss_max))

    def test_handles_single_sample(self):
        """Test model handles batch size of 1."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        loss, bellman_loss, euler_loss = model.compute_loss(k, z)

        self.assertFalse(tf.math.is_nan(loss))
        self.assertFalse(tf.math.is_nan(bellman_loss))
        self.assertFalse(tf.math.is_nan(euler_loss))

    def test_handles_large_batch(self):
        """Test model handles large batch sizes."""
        model = BasicModelDL(self.params, self.config)

        batch_size = 256
        k = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.capital_min,
            maxval=self.config.capital_max,
            dtype=TENSORFLOW_DTYPE
        )
        z = tf.random.uniform(
            (batch_size, 1),
            minval=self.config.productivity_min,
            maxval=self.config.productivity_max,
            dtype=TENSORFLOW_DTYPE
        )

        loss, _, _ = model.compute_loss(k, z)

        self.assertFalse(tf.math.is_nan(loss))
        self.assertFalse(tf.math.is_inf(loss))


class TestBasicModelDLReproducibility(unittest.TestCase):
    """Test reproducibility of basic model training."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.params = get_test_economic_params()
        cls.config = get_test_dl_config_basic()

    def test_deterministic_forward_pass(self):
        """Test that forward pass is deterministic given same inputs."""
        model = BasicModelDL(self.params, self.config)

        k = tf.constant([[2.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        inputs = tf.concat([
            model.normalizer.normalize_capital(k),
            model.normalizer.normalize_productivity(z)
        ], axis=1)

        v1 = model.value_net(inputs, training=False)
        v2 = model.value_net(inputs, training=False)

        np.testing.assert_array_almost_equal(v1.numpy(), v2.numpy())

    def test_network_weights_accessible(self):
        """Test that network weights can be accessed and saved."""
        model = BasicModelDL(self.params, self.config)

        value_weights = model.value_net.get_weights()
        policy_weights = model.policy_net.get_weights()

        self.assertGreater(len(value_weights), 0)
        self.assertGreater(len(policy_weights), 0)

        # Weights should be finite
        for w in value_weights + policy_weights:
            self.assertTrue(np.all(np.isfinite(w)))


if __name__ == "__main__":
    # Configure TensorFlow to use CPU for reproducibility
    tf.config.set_visible_devices([], 'GPU')

    # Run tests
    unittest.main(verbosity=2)