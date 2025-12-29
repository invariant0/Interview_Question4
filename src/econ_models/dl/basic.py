# econ_models/dl/basic.py
"""
Deep Learning solver for the Basic RBC Model.

This module implements a neural network approach to solving the
basic RBC model using value and policy function approximation.

Example:
    >>> from econ_models.dl.basic import BasicModelDL
    >>> model = BasicModelDL(params, config)
    >>> model.train()
"""

from typing import Tuple, Dict

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CashFlowCalculator
)
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.io.checkpoints import save_checkpoint_basic

tfd = tfp.distributions


class BasicModelDL:
    """
    Deep learning solver for the Basic RBC model.

    This class uses neural networks to approximate both the value function
    and optimal investment policy, training via Bellman and Euler residual
    minimization.

    Attributes:
        params: Economic parameters.
        config: Deep learning configuration.
        normalizer: State space normalizer.
        policy_net: Policy function neural network.
        value_net: Value function neural network.
        optimizer: Adam optimizer with learning rate schedule.
        shock_dist: Normal distribution for productivity shocks.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig
    ) -> None:
        """
        Initialize the basic deep learning model.

        Args:
            params: Economic parameters.
            config: Deep learning configuration.
        """
        self.params = params
        self.config = config

        self.config.update_value_scale(self.params)
        self.normalizer = StateSpaceNormalizer(self.config)

        self._build_networks()
        self._build_optimizer()
        self._build_shock_distribution()

    def _build_networks(self) -> None:
        """Construct policy and value neural networks."""
        self.policy_net = NeuralNetFactory.build_mlp(
            input_dim=2,
            output_dim=1,
            config=self.config,
            output_activation='linear',
            name="PolicyNet"
        )

        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=2,
            output_dim=1,
            config=self.config,
            output_activation='linear',
            scale_factor=self.config.value_scale_factor,
            name="ValueNet"
        )

    def _build_optimizer(self) -> None:
        """Construct optimizer with learning rate schedule."""
        decay_epochs = 5
        decay_steps = self.config.steps_per_epoch * decay_epochs

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=decay_steps,
            decay_rate=0.95,
            staircase=False
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

    def _build_shock_distribution(self) -> None:
        """Construct productivity shock distribution."""
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE)
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Create training data pipeline."""
        return DatasetBuilder.build_dataset(self.config, include_debt=False)

    @tf.function
    def compute_loss(
        self,
        k_curr: tf.Tensor,
        z_curr: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute training loss from Bellman and Euler residuals.

        Args:
            k_curr: Current capital values.
            z_curr: Current productivity values.

        Returns:
            Tuple of (total_loss, bellman_loss, euler_loss).
        """
        # Current state processing
        k_norm = self.normalizer.normalize_capital(k_curr)
        z_norm = self.normalizer.normalize_productivity(z_curr)
        inputs_curr = tf.concat([k_norm, z_norm], axis=1)

        investment_rate = self.policy_net(inputs_curr)
        v_curr = self.value_net(inputs_curr)

        # Calculate next period capital and cash flow
        investment = investment_rate * k_curr
        k_prime = investment + (1.0 - self.params.depreciation_rate) * k_curr

        profit = ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)
        adj_cost, marginal_adj_cost = AdjustmentCostCalculator.calculate(
            investment, k_curr, self.params
        )
        cash_flow = CashFlowCalculator.basic_cash_flow(profit, investment, adj_cost)

        # Stochastic transitions with two independent samples
        eps1 = self.shock_dist.sample(sample_shape=tf.shape(z_curr))
        eps2 = self.shock_dist.sample(sample_shape=tf.shape(z_curr))

        z_prime_1 = TransitionFunctions.log_ar1_transition(
            z_curr, self.params.productivity_persistence, eps1
        )
        z_prime_2 = TransitionFunctions.log_ar1_transition(
            z_curr, self.params.productivity_persistence, eps2
        )

        # Compute continuation values and gradients
        k_prime_norm = self.normalizer.normalize_capital(k_prime)
        z_prime_1_norm = self.normalizer.normalize_productivity(z_prime_1)
        z_prime_2_norm = self.normalizer.normalize_productivity(z_prime_2)

        k_prime_norm_combined = tf.concat([k_prime_norm, k_prime_norm], axis=0)
        z_prime_norm_combined = tf.concat([z_prime_1_norm, z_prime_2_norm], axis=0)

        # Envelope condition computation
        with tf.GradientTape() as tape_v:
            tape_v.watch(k_prime_norm_combined)
            inputs_next_combined = tf.concat(
                [k_prime_norm_combined, z_prime_norm_combined], axis=1
            )
            v_prime_combined = self.value_net(inputs_next_combined)

        dv_dk_prime_norm_combined = tape_v.gradient(
            v_prime_combined, k_prime_norm_combined
        )
        dv_dk_prime_combined = dv_dk_prime_norm_combined / self.normalizer.k_range

        v_prime_1, v_prime_2 = tf.split(v_prime_combined, num_or_size_splits=2, axis=0)
        dv_dk_prime_1, dv_dk_prime_2 = tf.split(
            dv_dk_prime_combined, num_or_size_splits=2, axis=0
        )

        # Calculate residuals
        bellman_target_1 = cash_flow + self.params.discount_factor * v_prime_1
        bellman_target_2 = cash_flow + self.params.discount_factor * v_prime_2

        bellman_resid_1 = v_curr - bellman_target_1
        bellman_resid_2 = v_curr - bellman_target_2

        safe_denom = 1.0 + marginal_adj_cost + 1e-8
        foc_resid_1 = 1.0 - (self.params.discount_factor * dv_dk_prime_1) / safe_denom
        foc_resid_2 = 1.0 - (self.params.discount_factor * dv_dk_prime_2) / safe_denom

        # Compute losses
        loss_bellman = tf.reduce_mean(bellman_resid_1 * bellman_resid_2)
        loss_foc = tf.reduce_mean(foc_resid_1 * foc_resid_2)

        total_loss = loss_bellman + self.config.euler_residual_weight * loss_foc

        return total_loss, loss_bellman, loss_foc

    @tf.function
    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one gradient descent step.

        Args:
            k: Capital values.
            z: Productivity values.

        Returns:
            Dictionary of loss values.
        """
        vars_to_train = (
            self.policy_net.trainable_variables
            + self.value_net.trainable_variables
        )

        with tf.GradientTape() as tape:
            loss, loss_b, loss_f = self.compute_loss(k, z)

        grads = tape.gradient(loss, vars_to_train)
        self.optimizer.apply_gradients(zip(grads, vars_to_train))

        return {"loss": loss, "bellman_loss": loss_b, "euler_loss": loss_f}

    def train(self) -> None:
        """Execute the main training loop."""
        print(f"Starting Basic Model Training for {self.config.epochs} epochs...")

        dataset = self._build_dataset()
        data_iter = iter(dataset)

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_loss_bellman = 0.0
            epoch_loss_euler = 0.0

            for _ in range(self.config.steps_per_epoch):
                k, z = next(data_iter)
                logs = self.train_step(k, z)
                epoch_loss += float(logs["loss"])
                epoch_loss_bellman += float(logs["bellman_loss"])
                epoch_loss_euler += float(logs["euler_loss"])

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_loss, epoch_loss_bellman, epoch_loss_euler)
                save_checkpoint_basic(self.value_net, self.policy_net, epoch)

    def _log_epoch(
        self,
        epoch: int,
        total_loss: float,
        bellman_loss: float,
        euler_loss: float
    ) -> None:
        """Log training progress for an epoch."""
        avg_loss = total_loss / self.config.steps_per_epoch
        avg_loss_b = bellman_loss / self.config.steps_per_epoch
        avg_loss_e = euler_loss / self.config.steps_per_epoch

        if isinstance(
            self.optimizer.learning_rate,
            tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            current_lr = self.optimizer.learning_rate(self.optimizer.iterations)
        else:
            current_lr = self.optimizer.learning_rate

        print(
            f"Epoch {epoch:4d} | Loss: {avg_loss:.4e} | "
            f"Bellman: {avg_loss_b:.4e} | Euler: {avg_loss_e:.4e} | "
            f"LR: {current_lr:.2e}"
        )