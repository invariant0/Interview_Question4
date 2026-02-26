# econ_models/dl/basic.py
"""Deep Learning solver for the Basic RBC Model.

This module implements a neural network approach to solving the basic RBC
model using value and policy function approximation.  Training proceeds
by minimising the Bellman residual (double-sampling for unbiased gradients)
and the first-order-condition (Euler) residual jointly.

Example
-------
>>> from econ_models.dl.basic import BasicModelDL
>>> model = BasicModelDL(params, config, bounds)
>>> model.train()
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.econ import (
    AdjustmentCostCalculator,
    CashFlowCalculator,
    ProductionFunctions,
)
from econ_models.io.checkpoints import save_checkpoint_basic

tfd = tfp.distributions


class BasicModelDL:
    """Deep learning solver for the Basic RBC model.

    Uses neural networks to approximate the value function and the optimal
    investment policy, trained via joint Bellman and Euler residual
    minimisation.

    Attributes:
        params: Economic parameters.
        config: Deep learning configuration.
        normalizer: State-space normalizer.
        policy_net: Policy function neural network.
        value_net: Value function neural network.
        optimizer: Adam optimizer with learning-rate schedule.
        shock_dist: Normal distribution for productivity shocks.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bounds: Dict[str, float],
    ) -> None:
        """Initialise the basic deep learning model.

        Args:
            params: Economic parameters.
            config: Deep learning configuration.
            bounds: State-space bounds (``k_min``, ``k_max``, etc.).
        """
        self.params = params
        self.config = config
        self.bounds = bounds
        self.config.update_value_scale(self.params)
        self.normalizer = StateSpaceNormalizer(self.config)

        self._build_networks()
        self._build_optimizer()
        self._build_shock_distribution()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        """Construct policy and value neural networks."""
        self.policy_net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=self.config,
            output_activation="sigmoid", name="PolicyNet",
        )
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueNet",
        )

    def _build_optimizer(self) -> None:
        """Construct Adam optimizer with exponential-decay learning-rate schedule.

        ``lr_decay_rate`` and ``lr_decay_steps`` are read from
        ``self.config`` (populated from the JSON config file).
        Raises ``ValueError`` if either is missing.
        """
        self.config.validate_scheduling_fields(
            ['lr_decay_rate', 'lr_decay_steps'], 'BasicModelDL'
        )
        decay_rate = self.config.lr_decay_rate
        decay_steps = self.config.lr_decay_steps

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule,
        )
        if decay_rate < 1.0:
            print(
                f"[LR] ExponentialDecay: {self.config.learning_rate:.2e} "
                f"× {decay_rate} every {decay_steps} steps"
            )

    def _build_shock_distribution(self) -> None:
        """Construct the productivity-shock distribution."""
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE),
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Create the training data pipeline with prefetching."""
        dataset = DatasetBuilder.build_dataset(
            self.config, self.bounds, include_debt=False,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        k_curr: tf.Tensor,
        z_curr: tf.Tensor,
        eps1: tf.Tensor,
        eps2: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute training loss from Bellman and Euler residuals.

        Uses double-sampling (two independent shock draws) for unbiased
        gradient estimation of both residuals.

        Args:
            k_curr: Current capital values, shape ``(batch, 1)``.
            z_curr: Current productivity values, shape ``(batch, 1)``.
            eps1: Pre-sampled shock draws (sample 1), shape ``(batch, 1)``.
            eps2: Pre-sampled shock draws (sample 2), shape ``(batch, 1)``.

        Returns:
            Tuple of ``(total_loss, bellman_loss, euler_loss, mean_k_prime)``.
        """
        # Normalise current state
        k_norm = self.normalizer.normalize_capital(k_curr)
        z_norm = self.normalizer.normalize_productivity(z_curr)
        inputs_curr = tf.concat([k_norm, z_norm], axis=1)

        # Policy and value predictions
        k_prime_norm = self.policy_net(inputs_curr)
        k_prime = self.normalizer.denormalize_capital(k_prime_norm)
        investment = k_prime - (1.0 - self.params.depreciation_rate) * k_curr
        v_curr = self.value_net(inputs_curr)

        # --- Cash flow  D = ZKᶿ − I − Φ(I,K) ---
        profit = ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)
        # Φ(I,K) = (ψ₀/2)(I²/K) + ψ₁·K·𝟙{I≠0}
        # marginal_adj_cost = ∂Φ/∂I = ψ₀·(I/K)
        adj_cost, marginal_adj_cost = AdjustmentCostCalculator.calculate(
            investment, k_curr, self.params,
        )
        cash_flow = CashFlowCalculator.basic_cash_flow(
            profit, investment, adj_cost,
        )

        # Future productivity (AR(1) transition)
        z_prime_1 = TransitionFunctions.log_ar1_transition(
            z_curr, self.params.productivity_persistence, eps1,
        )
        z_prime_2 = TransitionFunctions.log_ar1_transition(
            z_curr, self.params.productivity_persistence, eps2,
        )

        # Continuation values and envelope-condition gradients
        z_prime_1_norm = self.normalizer.normalize_productivity(z_prime_1)
        z_prime_2_norm = self.normalizer.normalize_productivity(z_prime_2)

        k_prime_combined = tf.concat([k_prime, k_prime], axis=0)
        z_prime_norm_combined = tf.concat(
            [z_prime_1_norm, z_prime_2_norm], axis=0,
        )

        with tf.GradientTape() as tape_v:
            tape_v.watch(k_prime_combined)
            k_prime_norm_combined = self.normalizer.normalize_capital(
                k_prime_combined,
            )
            inputs_next_combined = tf.concat(
                [k_prime_norm_combined, z_prime_norm_combined], axis=1,
            )
            v_prime_combined = self.value_net(inputs_next_combined)

        dv_dk_prime_combined = tape_v.gradient(
            v_prime_combined, k_prime_combined,
        )

        v_prime_1, v_prime_2 = tf.split(
            v_prime_combined, num_or_size_splits=2, axis=0,
        )
        dv_dk_prime_1, dv_dk_prime_2 = tf.split(
            dv_dk_prime_combined, num_or_size_splits=2, axis=0,
        )

        # --------------------------------------------------
        # A. Bellman Residuals (double-sampling)
        #
        #   V(K,Z) = D + β·𝔼[V(K',Z') | Z]
        #
        #   D^Bell = V(K,Z) − [D + β·V(K',Z')]
        #
        # Product D₁^Bell·D₂^Bell with independent ε₁, ε₂
        # gives unbiased estimate of 𝔼[D²].
        # --------------------------------------------------
        beta = self.params.discount_factor
        bellman_target_1 = cash_flow + beta * v_prime_1
        bellman_target_2 = cash_flow + beta * v_prime_2
        bellman_resid_1 = v_curr - bellman_target_1
        bellman_resid_2 = v_curr - bellman_target_2

        # --------------------------------------------------
        # B. Capital Euler (FOC) Residuals  (ratio form)
        #
        #   Standard FOC w.r.t. K':
        #     (1 + ψ₀·I/K) = β·∂V/∂K'
        #
        #   Ratio form (dividing both sides by 1 + ψ₀·I/K):
        #     D^Euler = 1 − β·∂V/∂K' / (1 + ψ₀·I/K) = 0
        #
        #   This normalises the residual to be dimensionless
        #   and O(1), improving loss balancing with the
        #   Bellman term.  safe_denom adds ε = 1e-8 for
        #   numerical stability.
        # --------------------------------------------------
        safe_denom = 1.0 + marginal_adj_cost + 1e-8
        foc_resid_1 = 1.0 - (beta * dv_dk_prime_1) / safe_denom
        foc_resid_2 = 1.0 - (beta * dv_dk_prime_2) / safe_denom

        # --------------------------------------------------
        # Total AiO Loss (unbiased double-sampling)
        #   Ξ = 𝔼[D₁^Bell·D₂^Bell + ν_E·D₁^Euler·D₂^Euler]
        # --------------------------------------------------
        loss_bellman = tf.reduce_mean(bellman_resid_1 * bellman_resid_2)
        loss_foc = tf.reduce_mean(foc_resid_1 * foc_resid_2)

        total_loss = loss_bellman + self.config.euler_residual_weight * loss_foc

        return (
            total_loss,
            loss_bellman,
            loss_foc,
            tf.reduce_mean(tf.stop_gradient(k_prime)),
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    @tf.function(jit_compile=True)
    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        eps1: tf.Tensor,
        eps2: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Perform one gradient-descent step.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.
            eps1: Pre-sampled shock draws (sample 1).
            eps2: Pre-sampled shock draws (sample 2).

        Returns:
            Dictionary of loss values.
        """
        trainable_vars = (
            self.policy_net.trainable_variables
            + self.value_net.trainable_variables
        )

        with tf.GradientTape() as tape:
            loss, loss_b, loss_f, k_prime = self.compute_loss(
                k, z, eps1, eps2,
            )

        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return {
            "loss": loss,
            "bellman_loss": loss_b,
            "euler_loss": loss_f,
            "k_prime": k_prime,
        }

    def _sample_shocks(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Pre-sample shock draws outside XLA for compatibility.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            Tuple of two independent shock tensors, each ``(batch, 1)``.
        """
        eps = tf.random.normal(
            shape=(batch_size, 2), dtype=TENSORFLOW_DTYPE,
        ) * self.params.productivity_std_dev
        return eps[:, 0:1], eps[:, 1:2]

    # ------------------------------------------------------------------
    # Epoch execution
    # ------------------------------------------------------------------

    @tf.function
    def _run_epoch_steps(
        self,
        dataset_iter: tf.data.Iterator,
        steps: int,
    ) -> Dict[str, tf.Tensor]:
        """Run all steps in one epoch within a single ``tf.function``.

        Keeps the entire epoch in graph mode to reduce Python dispatch
        overhead.

        Args:
            dataset_iter: Iterator over ``(k, z)`` training batches.
            steps: Number of training steps per epoch.

        Returns:
            Accumulated loss metrics across the epoch.
        """
        acc_loss = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_bellman = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_euler = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_k_prime = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)

        for _ in tf.range(steps):
            k, z = next(dataset_iter)
            eps1, eps2 = self._sample_shocks(tf.shape(k)[0])
            logs = self.train_step(k, z, eps1, eps2)
            acc_loss += logs["loss"]
            acc_bellman += logs["bellman_loss"]
            acc_euler += logs["euler_loss"]
            acc_k_prime += logs["k_prime"]

        return {
            "loss": acc_loss,
            "bellman_loss": acc_bellman,
            "euler_loss": acc_euler,
            "k_prime": acc_k_prime,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Execute the main training loop."""
        print(f"Starting Basic Model Training for {self.config.epochs} epochs...")

        dataset = self._build_dataset()
        data_iter = iter(dataset)

        for epoch in range(self.config.epochs):
            logs = self._run_epoch_steps(data_iter, self.config.steps_per_epoch)

            epoch_loss = float(logs["loss"])
            epoch_bellman = float(logs["bellman_loss"])
            epoch_euler = float(logs["euler_loss"])
            k_prime = float(logs["k_prime"])

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(
                    epoch, epoch_loss, epoch_bellman, epoch_euler, k_prime,
                )

            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                save_checkpoint_basic(self.value_net, self.policy_net, epoch)

    def _log_epoch(
        self,
        epoch: int,
        total_loss: float,
        bellman_loss: float,
        euler_loss: float,
        k_prime: float,
    ) -> None:
        """Log training progress for an epoch.

        Args:
            epoch: Current epoch number.
            total_loss: Accumulated total loss over the epoch.
            bellman_loss: Accumulated Bellman loss over the epoch.
            euler_loss: Accumulated Euler loss over the epoch.
            k_prime: Accumulated mean next-period capital over the epoch.
        """
        steps = self.config.steps_per_epoch
        avg_loss = total_loss / steps
        avg_bellman = bellman_loss / steps
        avg_euler = euler_loss / steps
        avg_k_prime = k_prime / steps

        if isinstance(
            self.optimizer.learning_rate,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            current_lr = self.optimizer.learning_rate(self.optimizer.iterations)
        else:
            current_lr = self.optimizer.learning_rate

        print(
            f"Epoch {epoch:4d} | Loss: {avg_loss:.4e} | "
            f"Bellman: {avg_bellman:.4e} | Euler: {avg_euler:.4e} | "
            f"LR: {current_lr:.2e} | K_prime: {avg_k_prime:.4e}"
        )
