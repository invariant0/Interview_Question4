# econ_models/dl/risky.py
"""
Deep Learning solver for the Risky Debt Model.

This module implements a neural network approach to solving the
risky debt model with endogenous default and bond pricing.

Example:
    >>> from econ_models.dl.risky import RiskyModelDL
    >>> model = RiskyModelDL(params, config)
    >>> model.train()
"""

from typing import Tuple, Dict

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.sampling.candidate_sampler import CandidateSampler
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CashFlowCalculator,
    BondPricingCalculator
)
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.io.checkpoints import save_checkpoint_risky

tfd = tfp.distributions

class RiskyModelDL:
    """
    Deep learning solver for the Risky Debt model.

    This class uses a value function neural network with a target network
    for stability, similar to Deep Q-Learning approaches. Bond prices are
    computed via Monte Carlo integration over default probabilities.

    Attributes:
        params: Economic parameters.
        config: Deep learning configuration.
        normalizer: State space normalizer.
        value_net: Value function neural network.
        target_value_net: Target network for stable bootstrapping.
        optimizer: Adam optimizer with learning rate schedule.
        shock_dist: Normal distribution for productivity shocks.
        training_progress: Curriculum learning progress variable.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig
    ) -> None:
        """
        Initialize the risky debt deep learning model.

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
        self._initialize_curriculum()

    def _build_networks(self) -> None:
        """Construct value and target neural networks."""
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="relu",
            scale_factor=self.config.value_scale_factor,
            name="RiskyValueNet"
        )

        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="relu",
            scale_factor=self.config.value_scale_factor,
            name="RiskyTargetValueNet"
        )

        # Initialize target weights to match source
        self.target_value_net.set_weights(self.value_net.get_weights())

    def _build_optimizer(self) -> None:
        """Construct optimizer with learning rate schedule."""
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.steps_per_epoch * 20,
            decay_rate=0.9,
            staircase=False
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

    def _build_shock_distribution(self) -> None:
        """Construct productivity shock distribution."""
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE)
        )

    def _initialize_curriculum(self) -> None:
        """Initialize curriculum learning state."""
        self.training_progress = tf.Variable(
            0.0, dtype=TENSORFLOW_DTYPE, trainable=False
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Create training data pipeline with curriculum support."""
        return DatasetBuilder.build_dataset(
            self.config,
            include_debt=True,
            progress_variable=self.training_progress
        )

    @tf.function
    def estimate_bond_price(
        self,
        k_prime_cand: tf.Tensor,
        b_prime_cand: tf.Tensor,
        z_curr: tf.Tensor
    ) -> tf.Tensor:
        """
        Estimate bond price q(k', b', z) via Monte Carlo.

        Args:
            k_prime_cand: Candidate next capital values.
            b_prime_cand: Candidate next debt values.
            z_curr: Current productivity values.

        Returns:
            Estimated bond prices for each candidate.
        """
        batch_size = tf.shape(z_curr)[0]
        n_cand = tf.shape(k_prime_cand)[1]
        n_samples = self.config.mc_sample_number_bond_priceing

        # Sample future shocks
        eps = self.shock_dist.sample(sample_shape=(batch_size, n_cand, n_samples))
        z_curr_bc = tf.expand_dims(
            tf.broadcast_to(z_curr, (batch_size, n_cand)), -1
        )
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )

        # Broadcast candidates
        k_prime_bc = tf.broadcast_to(
            tf.expand_dims(k_prime_cand, -1), (batch_size, n_cand, n_samples)
        )
        b_prime_bc = tf.broadcast_to(
            tf.expand_dims(b_prime_cand, -1), (batch_size, n_cand, n_samples)
        )

        # Evaluate value at (k', b', z') for default check
        flat_shape = (batch_size * n_cand * n_samples, 1)
        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(tf.reshape(k_prime_bc, flat_shape)),
            self.normalizer.normalize_debt(tf.reshape(b_prime_bc, flat_shape)),
            self.normalizer.normalize_productivity(tf.reshape(z_prime, flat_shape))
        ], axis=1)

        v_prime_eval = self.target_value_net(inputs_eval, training=False)
        is_default = tf.cast(
            v_prime_eval <= self.config.epsilon_value_default,
            TENSORFLOW_DTYPE
        )

        # Calculate payoff
        profit = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(
                tf.reshape(k_prime_bc, flat_shape),
                tf.reshape(z_prime, flat_shape),
                self.params
            )
        )
        recovery = BondPricingCalculator.recovery_value(
            profit, tf.reshape(k_prime_bc, flat_shape), self.params
        )
        payoff = BondPricingCalculator.bond_payoff(
            recovery, tf.reshape(b_prime_bc, flat_shape), is_default
        )

        # Average over samples
        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, n_cand, n_samples)), axis=2
        )

        return BondPricingCalculator.risk_neutral_price(
            expected_payoff,
            b_prime_cand,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price
        )

    @tf.function
    def _optimize_next_state(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Find optimal (k', b') by evaluating candidate grid.

        Args:
            k_curr: Current capital values.
            b_curr: Current debt values.
            z_curr: Current productivity values.

        Returns:
            Tuple of (optimal k', optimal b', optimal bond price).
        """
        batch_size = tf.shape(k_curr)[0]
        n_cand = self.config.mc_next_candidate_sample
        n_mc_samples = self.config.mc_sample_number_continuation_value

        # Generate candidates
        k_cand, b_cand = CandidateSampler.sample_candidate(
            batch_size, n_cand, k_curr, b_curr,
            self.config, progress=self.training_progress
        )
        n_actual = tf.shape(k_cand)[1]

        # Estimate bond prices
        q_cand = self.estimate_bond_price(k_cand, b_cand, z_curr)

        # Calculate current cash flows
        revenue = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)
        )

        k_curr_bc = tf.broadcast_to(k_curr, (batch_size, n_actual))
        b_curr_bc = tf.broadcast_to(b_curr, (batch_size, n_actual))

        investment = ProductionFunctions.calculate_investment(
            k_curr_bc, k_cand, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr_bc, self.params
        )

        dividend = CashFlowCalculator.risky_cash_flow(
            tf.broadcast_to(revenue, (batch_size, n_actual)),
            investment, adj_cost,
            b_curr_bc, b_cand, q_cand, self.params
        )

        # Estimate expected continuation value
        eps_samples = self.shock_dist.sample(
            sample_shape=(batch_size, n_actual, n_mc_samples)
        )

        z_curr_expanded = tf.expand_dims(
            tf.broadcast_to(z_curr, (batch_size, n_actual)), -1
        )
        z_prime_samples = TransitionFunctions.log_ar1_transition(
            z_curr_expanded, self.params.productivity_persistence, eps_samples
        )

        k_cand_expanded = tf.broadcast_to(
            tf.expand_dims(k_cand, -1), (batch_size, n_actual, n_mc_samples)
        )
        b_cand_expanded = tf.broadcast_to(
            tf.expand_dims(b_cand, -1), (batch_size, n_actual, n_mc_samples)
        )

        flat_dim = batch_size * n_actual * n_mc_samples
        inputs_next = tf.concat([
            self.normalizer.normalize_capital(
                tf.reshape(k_cand_expanded, (flat_dim, 1))
            ),
            self.normalizer.normalize_debt(
                tf.reshape(b_cand_expanded, (flat_dim, 1))
            ),
            self.normalizer.normalize_productivity(
                tf.reshape(z_prime_samples, (flat_dim, 1))
            )
        ], axis=1)

        v_prime_flat = self.target_value_net(inputs_next, training=False)
        v_prime_samples = tf.reshape(
            v_prime_flat, (batch_size, n_actual, n_mc_samples)
        )
        expected_v_prime = tf.reduce_mean(v_prime_samples, axis=2)

        # Maximize total value
        beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
        rhs_cand = dividend + beta * expected_v_prime
        best_indices = tf.argmax(rhs_cand, axis=1)

        gather_indices = tf.stack(
            [tf.range(batch_size, dtype=tf.int64), best_indices], axis=1
        )

        k_opt = tf.expand_dims(tf.gather_nd(k_cand, gather_indices), 1)
        b_opt = tf.expand_dims(tf.gather_nd(b_cand, gather_indices), 1)
        q_opt = tf.expand_dims(tf.gather_nd(q_cand, gather_indices), 1)

        return k_opt, b_opt, q_opt

    @tf.function
    def train_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one gradient descent step on the value network.

        Args:
            k: Capital values.
            b: Debt values.
            z: Productivity values.

        Returns:
            Dictionary of training metrics.
        """
        with tf.GradientTape() as tape:
            # Find optimal policy
            k_opt, b_opt, q_opt = self._optimize_next_state(k, b, z)

            # Calculate implied value target
            revenue = (
                (1.0 - self.params.corporate_tax_rate)
                * ProductionFunctions.cobb_douglas(k, z, self.params)
            )
            investment = ProductionFunctions.calculate_investment(k, k_opt, self.params)
            adj_cost, _ = AdjustmentCostCalculator.calculate(investment, k, self.params)
            dividend = CashFlowCalculator.risky_cash_flow(
                revenue, investment, adj_cost, b, b_opt, q_opt, self.params
            )

            # Calculate continuation values with two independent samples
            eps1 = self.shock_dist.sample(sample_shape=tf.shape(z))
            eps2 = self.shock_dist.sample(sample_shape=tf.shape(z))
            z_prime_1 = TransitionFunctions.log_ar1_transition(
                z, self.params.productivity_persistence, eps1
            )
            z_prime_2 = TransitionFunctions.log_ar1_transition(
                z, self.params.productivity_persistence, eps2
            )

            k_opt_norm = self.normalizer.normalize_capital(k_opt)
            b_opt_norm = self.normalizer.normalize_debt(b_opt)

            v_prime_1 = self.target_value_net(
                tf.concat([
                    k_opt_norm, b_opt_norm,
                    self.normalizer.normalize_productivity(z_prime_1)
                ], 1),
                training=False
            )
            v_prime_2 = self.target_value_net(
                tf.concat([
                    k_opt_norm, b_opt_norm,
                    self.normalizer.normalize_productivity(z_prime_2)
                ], 1),
                training=False
            )

            # Current value
            v_curr = self.value_net(
                tf.concat([
                    self.normalizer.normalize_capital(k),
                    self.normalizer.normalize_debt(b),
                    self.normalizer.normalize_productivity(z)
                ], 1),
                training=True
            )

            # Calculate AiO loss
            beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
            target_1 = tf.keras.activations.relu(dividend + beta * v_prime_1)
            target_2 = tf.keras.activations.relu(dividend + beta * v_prime_2)

            bellman_res1 = v_curr - target_1
            bellman_res2 = v_curr - target_2
            loss = tf.reduce_mean(bellman_res1 * bellman_res2)

        # Apply gradients
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        if self.config.gradient_clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip_norm)
        self.optimizer.apply_gradients(
            zip(grads, self.value_net.trainable_variables)
        )

        return {
            "loss": loss,
            "avg_val": tf.reduce_mean(v_curr),
            "avg_bond_price": tf.reduce_mean(q_opt),
            "avg_k_prime": tf.reduce_mean(k_opt),
            "avg_b_prime": tf.reduce_mean(b_opt),
        }

    def train(self) -> None:
        """Execute the main training loop."""
        print(f"Starting Risky Model Training for {self.config.epochs} epochs...")

        dataset = self._build_dataset()
        data_iter = iter(dataset)

        for epoch in range(self.config.epochs):
            # Update curriculum progress
            self._update_curriculum(epoch)

            epoch_logs = self._run_epoch(data_iter)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs)

            if epoch % 10 == 0:
                save_checkpoint_risky(self.value_net, epoch)

    def _update_curriculum(self, epoch: int) -> None:
        """Update curriculum learning progress."""
        if self.config.curriculum_epochs > 0:
            progress_val = min(1.0, epoch / self.config.curriculum_epochs)
        else:
            progress_val = 1.0
        self.training_progress.assign(tf.cast(progress_val, TENSORFLOW_DTYPE))

    def _run_epoch(self, data_iter) -> Dict[str, float]:
        """Run one training epoch."""
        epoch_logs = {}

        for step in range(self.config.steps_per_epoch):
            k, b, z = next(data_iter)
            logs = self.train_step(k, b, z)

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            # Soft update target network every step
            NeuralNetFactory.soft_update(
                self.value_net,
                self.target_value_net,
                self.config.polyak_averaging_decay
            )

        return epoch_logs

    def _log_epoch(self, epoch: int, epoch_logs: Dict[str, float]) -> None:
        """Log training progress for an epoch."""
        steps = self.config.steps_per_epoch
        progress_val = float(self.training_progress.numpy())

        avg_loss = epoch_logs['loss'] / steps
        avg_val = epoch_logs['avg_val'] / steps
        avg_q = epoch_logs['avg_bond_price'] / steps
        avg_k_prime = epoch_logs['avg_k_prime'] / steps
        avg_b_prime = epoch_logs['avg_b_prime'] / steps

        print(
            f"Epoch {epoch:4d} | "
            f"Prg: {progress_val:.2f} | "
            f"Loss: {avg_loss:.4e} | "
            f"Val: {avg_val:.2f} | "
            f"Q: {avg_q:.3f} | "
            f"K': {avg_k_prime:.3f} | "
            f"B': {avg_b_prime:.3f}"
        )