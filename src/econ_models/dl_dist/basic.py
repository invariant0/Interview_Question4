# econ_models/dl_dist/basic.py
"""Deep Learning solver for the Basic RBC Model (distributional variant).

This module implements a neural network approach to solving the basic RBC
model using value and policy function approximation.  Unlike the
single-parameter ``dl.basic`` module, this variant accepts economic
parameters ``(rho, std, convex, fixed)`` as additional network inputs so
that a *single* trained model generalises over the parameter distribution.

Equation System (Basic RBC — no debt)
--------------------------------------

Eq. 1 — Cash flow (dividend):

  D = ZKᶿ − I − Φ(I,K)

  where  I = K' − (1−δ)K
         Φ(I,K) = (ψ₀/2)(I²/K) + ψ₁·K·𝟙{I≠0}

Eq. 2 — Bellman equation:

  V(K,Z) = D + β·𝔼[V(K',Z') | Z]

Eq. 4 — Capital Euler (FOC w.r.t. K'):

  Standard form:   (1 + ψ₀·I/K) = β·∂V/∂K'
  Ratio form:      1 − β·∂V/∂K' / (1 + ψ₀·I/K) = 0

  This module uses the ratio form for the Euler residual, which
  normalises the residual to be dimensionless and O(1).

AiO loss (double-sampling for unbiased gradient estimation):

  Ξ(Θ) = 𝔼[D₁^Bell·D₂^Bell + ν_E·D₁^Euler·D₂^Euler]

  where ν_E = ``config.euler_residual_weight``.

Architecture
------------
- **Policy Net** ``π(s, p; θ_π)`` — sigmoid output ∈ (0,1), then
  ``denormalize_capital`` maps to K' ∈ [k_min, k_max].
- **Value Net** ``V(s, p; θ_V)`` — linear output, scaled by
  ``value_scale_factor``.
- Inputs: ``(K, Z, ρ, σ, ψ₀, ψ₁)`` — 6 dimensions, all normalised.

Example
-------
>>> from econ_models.dl_dist.basic import BasicModelDL
>>> model = BasicModelDL(params, config, bounds)
>>> model.train()
"""

from __future__ import annotations

import atexit
import datetime
import os
import shutil
from typing import Dict, Tuple

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import ParamSpaceNormalizer, StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl_dist.training.dataset_builder import DatasetBuilder
from econ_models.dl_dist.basic_final import OptimizationConfig
from econ_models.econ import (
    AdjustmentCostCalculator,
    CashFlowCalculator,
    ProductionFunctions,
)
from econ_models.io.checkpoints import save_checkpoint_basic_dist


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BasicModelDL:
    """Deep learning solver for the Basic RBC model (distributional).

    Minimises the double-sampling AiO loss over a distribution of economic
    parameters:

      Ξ(Θ) = 𝔼[ D₁^Bell·D₂^Bell  +  ν_E·D₁^Euler·D₂^Euler ]

    where:
      D^Bell  = V(K,Z) − [D + β·V(K',Z')]              (Bellman residual)
      D^Euler = 1 − β·∂V/∂K' / (1 + ψ₀·I/K + ε)    (Euler, ratio form)

    The product D₁·D₂ with two independent shock draws provides unbiased
    gradient estimation of 𝔼[D²] (Maliar, Maliar & Winant 2021).

    Policy parameterisation:
      - Policy Net: sigmoid ∈ (0,1) → K' = denormalize_capital(σ(x))
      - Value Net: linear output, scaled by ``value_scale_factor``

    Args:
        params: Economic parameters.
        config: Deep learning configuration.
        bounds: State-space and parameter bounds with keys ``k_min``,
            ``k_max``, ``z_min``, ``z_max``, etc.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bounds: dict,
        optimization_config: OptimizationConfig | None = None,
        checkpoint_dir: str = "checkpoints_pretrain_dist/basic",
        log_dir_prefix: str = "logs/basic_dist",
    ) -> None:
        self.params = params
        self.config = config
        self.bounds = bounds
        self.optimization_config = optimization_config or OptimizationConfig()
        self.checkpoint_dir = checkpoint_dir
        self.log_dir_prefix = log_dir_prefix

        self.config.update_value_scale(self.params)
        self.normalizer_states = StateSpaceNormalizer(self.config)
        self.normalizer_params = ParamSpaceNormalizer(self.bounds)

        self._build_networks()
        self._build_optimizer()
        self._build_summary_writer()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        """Construct policy and value neural networks."""
        self.policy_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="sigmoid",
            name="PolicyNet",
        )
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="linear",
            name="ValueNet",
        )

    def _build_optimizer(self) -> None:
        """Construct Adam optimiser with exponential-decay learning-rate schedule.

        ``lr_decay_rate`` and ``lr_decay_steps`` are read exclusively from
        ``self.config`` (populated from the JSON config file).
        Raises ``ValueError`` if either is missing.
        """
        self.config.validate_scheduling_fields(
            ['lr_decay_rate', 'lr_decay_steps'], 'BasicModelDL_Dist'
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

    def _build_summary_writer(self) -> None:
        """Create TensorBoard summary writer.

        Logs are written to ``<log_dir_prefix>/<timestamp>/``.
        Launch TensorBoard with::

            tensorboard --logdir <log_dir_prefix>
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir_prefix, timestamp)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoard logs → {log_dir}")

    def _sample_shocks(
        self, batch_size: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Pre-sample unit-normal shock draws outside XLA for compatibility.

        Returns:
            Two tensors of shape ``(batch_size, 1)``, each containing
            independent standard-normal draws.  The caller scales by
            per-sample ``std``.
        """
        eps = tf.random.normal(
            shape=(batch_size, 2), dtype=TENSORFLOW_DTYPE,
        )
        return eps[:, 0:1], eps[:, 1:2]

    def _build_dataset(self) -> tf.data.Dataset:
        """Build the training data pipeline with prefetching."""
        dataset = DatasetBuilder.build_dataset(
            self.config,
            self.bounds,
            include_debt=False,
            cross_product=self.optimization_config.cross_product_sampling,
            n_states=self.optimization_config.batch_size_states,
            n_params=self.optimization_config.batch_size_params,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        k_curr: tf.Tensor,
        z_curr: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eps1: tf.Tensor,
        eps2: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute training loss from Bellman and Euler residuals.

        Args:
            k_curr: Current capital, shape ``(batch, 1)``.
            z_curr: Current productivity, shape ``(batch, 1)``.
            rho: Productivity persistence per sample, shape ``(batch, 1)``.
            std: Productivity std-dev per sample, shape ``(batch, 1)``.
            convex: Convex adjustment cost per sample, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost per sample, shape ``(batch, 1)``.
            eps1: Unit-normal shock draw 1, shape ``(batch, 1)``.
            eps2: Unit-normal shock draw 2, shape ``(batch, 1)``.

        Returns:
            Tuple of ``(total_loss, bellman_loss, euler_loss, mean_k_prime,
            mean_dv_dk_prime, std_dv_dk_prime)``.
        """
        # --- Normalise inputs ---
        k_norm = self.normalizer_states.normalize_capital(k_curr)
        z_norm = self.normalizer_states.normalize_productivity(z_curr)
        rho_norm = self.normalizer_params.normalize_rho(rho)
        std_norm = self.normalizer_params.normalize_std(std)
        convex_norm = self.normalizer_params.normalize_convex(convex)
        fixed_norm = self.normalizer_params.normalize_fixed(fixed)
        inputs = tf.concat(
            [k_norm, z_norm, rho_norm, std_norm, convex_norm, fixed_norm],
            axis=1,
        )

        # --- Policy forward pass ---
        k_prime_norm = self.policy_net(inputs)
        k_prime = self.normalizer_states.denormalize_capital(k_prime_norm)
        investment = k_prime - (1.0 - self.params.depreciation_rate) * k_curr
        v_curr = self.value_net(inputs)

        # --- Cash flow  D = ZKᶿ − I − Φ(I,K)  (Eq. 1) ---
        profit = ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)
        # Φ(I,K) = (ψ₀/2)(I²/K) + ψ₁·K·𝟙{I≠0}
        # marginal_adj_cost = ∂Φ/∂I = ψ₀·(I/K)
        adj_cost, marginal_adj_cost = (
            AdjustmentCostCalculator.calculate_dist(
                investment, k_curr, convex, fixed,
            )
        )
        cash_flow = CashFlowCalculator.basic_cash_flow(
            profit, investment, adj_cost,
        )

        # --- Stochastic transitions ---
        z_prime_1 = TransitionFunctions.log_ar1_transition(
            z_curr, rho, eps1 * std,
        )
        z_prime_2 = TransitionFunctions.log_ar1_transition(
            z_curr, rho, eps2 * std,
        )
        # Clamp z' to prevent extreme cash-flows from tail shocks
        z_lo = tf.constant(1e-4, dtype=TENSORFLOW_DTYPE)
        z_hi = tf.constant(1e4, dtype=TENSORFLOW_DTYPE)
        z_prime_1 = tf.clip_by_value(z_prime_1, z_lo, z_hi)
        z_prime_2 = tf.clip_by_value(z_prime_2, z_lo, z_hi)

        # --- Continuation values + envelope-theorem gradients ---
        z_prime_1_norm = self.normalizer_states.normalize_productivity(
            z_prime_1,
        )
        z_prime_2_norm = self.normalizer_states.normalize_productivity(
            z_prime_2,
        )

        k_prime_combined = tf.concat([k_prime, k_prime], axis=0)
        z_prime_norm_combined = tf.concat(
            [z_prime_1_norm, z_prime_2_norm], axis=0,
        )
        rho_norm_combined = tf.concat([rho_norm, rho_norm], axis=0)
        std_norm_combined = tf.concat([std_norm, std_norm], axis=0)
        convex_norm_combined = tf.concat(
            [convex_norm, convex_norm], axis=0,
        )
        fixed_norm_combined = tf.concat(
            [fixed_norm, fixed_norm], axis=0,
        )

        with tf.GradientTape() as tape_v:
            tape_v.watch(k_prime_combined)
            k_prime_norm_combined = self.normalizer_states.normalize_capital(
                k_prime_combined,
            )
            inputs_next_combined = tf.concat(
                [
                    k_prime_norm_combined, z_prime_norm_combined,
                    rho_norm_combined, std_norm_combined,
                    convex_norm_combined, fixed_norm_combined,
                ],
                axis=1,
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
        # A. Bellman Residuals (Eq. 2, double-sampling)
        #
        #   D^Bell = V(K,Z) − [D + β·V(K',Z')]
        #
        # Product D₁^Bell·D₂^Bell with independent ε₁, ε₂
        # gives unbiased estimate of 𝔼[D²].
        # --------------------------------------------------
        bellman_target_1 = cash_flow + self.params.discount_factor * v_prime_1
        bellman_target_2 = cash_flow + self.params.discount_factor * v_prime_2

        bellman_resid_1 = v_curr - bellman_target_1
        bellman_resid_2 = v_curr - bellman_target_2
        # Clamp residuals to prevent overflow in the D₁·D₂ product
        _RESID_MAX = tf.constant(1e4, dtype=TENSORFLOW_DTYPE)
        bellman_resid_1 = tf.clip_by_value(
            bellman_resid_1, -_RESID_MAX, _RESID_MAX,
        )
        bellman_resid_2 = tf.clip_by_value(
            bellman_resid_2, -_RESID_MAX, _RESID_MAX,
        )
        loss_bellman = tf.reduce_mean(bellman_resid_1 * bellman_resid_2)

        # --------------------------------------------------
        # B. Euler (FOC) Residuals  (Eq. 4, ratio form)
        #
        #   Standard FOC:  (1 + ψ₀·I/K) = β·∂V/∂K'
        #
        #   Ratio form (dividing both sides by 1 + ψ₀·I/K):
        #
        #     D^Euler = 1 − β·∂V/∂K' / (1 + ψ₀·I/K) = 0
        #
        #   This normalises the residual to be dimensionless
        #   and O(1), which improves loss balancing with the
        #   Bellman term.  Mathematically equivalent to the
        #   difference form used in the risk-free debt model.
        #
        #   safe_denom adds ε = 1e-8 for numerical stability.
        # --------------------------------------------------
        safe_denom = 1.0 + marginal_adj_cost + 1e-8
        foc_resid_1 = 1.0 - (
            self.params.discount_factor * dv_dk_prime_1
        ) / safe_denom
        foc_resid_2 = 1.0 - (
            self.params.discount_factor * dv_dk_prime_2
        ) / safe_denom
        # Clamp Euler residuals (ratio form should be O(1))
        _EULER_MAX = tf.constant(10.0, dtype=TENSORFLOW_DTYPE)
        foc_resid_1 = tf.clip_by_value(
            foc_resid_1, -_EULER_MAX, _EULER_MAX,
        )
        foc_resid_2 = tf.clip_by_value(
            foc_resid_2, -_EULER_MAX, _EULER_MAX,
        )
        loss_foc = tf.reduce_mean(foc_resid_1 * foc_resid_2)

        # --------------------------------------------------
        # Total AiO Loss:  Ξ = 𝔼[D₁^Bell·D₂^Bell + ν_E·D₁^Euler·D₂^Euler]
        # --------------------------------------------------
        total_loss = (
            loss_bellman + self.config.euler_residual_weight * loss_foc
        )
        # NaN / Inf safety net – replace with zero so one bad batch
        # does not poison the weights permanently.
        total_loss = tf.where(
            tf.math.is_finite(total_loss), total_loss,
            tf.constant(0.0, dtype=TENSORFLOW_DTYPE),
        )

        # --- dV/dK' diagnostics (average over both shock draws) ---
        mean_dv = tf.reduce_mean(tf.stop_gradient(dv_dk_prime_combined))
        std_dv = tf.math.reduce_std(tf.stop_gradient(dv_dk_prime_combined))

        return (
            total_loss,
            loss_bellman,
            loss_foc,
            tf.reduce_mean(tf.stop_gradient(k_prime)),
            mean_dv,
            std_dv,
        )

    @tf.function(jit_compile=True)
    def train_step(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eps1: tf.Tensor,
        eps2: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Perform one gradient-descent step.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.
            rho: Productivity persistence, shape ``(batch, 1)``.
            std: Productivity std-dev, shape ``(batch, 1)``.
            convex: Convex adjustment cost, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)``.
            eps1: Unit-normal shock draw 1, shape ``(batch, 1)``.
            eps2: Unit-normal shock draw 2, shape ``(batch, 1)``.

        Returns:
            Dictionary of scalar loss metrics.
        """
        vars_to_train = (
            self.policy_net.trainable_variables
            + self.value_net.trainable_variables
        )

        with tf.GradientTape() as tape:
            loss, loss_b, loss_f, k_prime, mean_dv, std_dv = self.compute_loss(
                k, z, rho, std, convex, fixed, eps1, eps2,
            )

        grads = tape.gradient(loss, vars_to_train)

        # --- Gradient clipping (prevents NaN from large residuals) ---
        clip_norm = self.config.gradient_clip_norm
        if clip_norm is not None and clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)

        # --- Gradient diagnostics ---
        grad_global_norm = tf.linalg.global_norm(grads)
        grad_max = tf.reduce_max(
            [tf.reduce_max(tf.abs(g)) for g in grads if g is not None]
        )

        self.optimizer.apply_gradients(zip(grads, vars_to_train))

        return {
            "loss": loss,
            "bellman_loss": loss_b,
            "euler_loss": loss_f,
            "k_prime": k_prime,
            "grad_norm": grad_global_norm,
            "grad_max": grad_max,
            "dv_dk_prime_mean": mean_dv,
            "dv_dk_prime_std": std_dv,
        }

    @tf.function
    def _run_epoch_steps(
        self,
        dataset_iter: tf.data.Iterator,
        steps: int,
    ) -> Dict[str, tf.Tensor]:
        """Run all steps in one epoch inside ``tf.function`` to reduce dispatch overhead."""
        acc_loss = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_bellman = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_euler = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_k_prime = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_grad_norm = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_grad_max = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_dv_mean = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
        acc_dv_std = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)

        for _ in tf.range(steps):
            k, z, rho, std, convex, fixed = next(dataset_iter)
            eps1, eps2 = self._sample_shocks(tf.shape(k)[0])
            logs = self.train_step(
                k, z, rho, std, convex, fixed, eps1, eps2,
            )
            acc_loss += logs["loss"]
            acc_bellman += logs["bellman_loss"]
            acc_euler += logs["euler_loss"]
            acc_k_prime += logs["k_prime"]
            acc_grad_norm += logs["grad_norm"]
            acc_grad_max = tf.maximum(acc_grad_max, logs["grad_max"])
            acc_dv_mean += logs["dv_dk_prime_mean"]
            acc_dv_std += logs["dv_dk_prime_std"]

        return {
            "loss": acc_loss,
            "bellman_loss": acc_bellman,
            "euler_loss": acc_euler,
            "k_prime": acc_k_prime,
            "grad_norm": acc_grad_norm,
            "grad_max": acc_grad_max,
            "dv_dk_prime_mean": acc_dv_mean,
            "dv_dk_prime_std": acc_dv_std,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Execute the main training loop."""
        print(
            f"Starting Basic Dist Model Training "
            f"for {self.config.epochs} epochs..."
        )
        if self.optimization_config.cross_product_sampling:
            import math
            ns = self.optimization_config.batch_size_states or int(math.isqrt(self.config.batch_size))
            np_ = self.optimization_config.batch_size_params or (self.config.batch_size // ns)
            print(
                f"Sampling: CROSS-PRODUCT  batch_size_states={ns} × "
                f"batch_size_params={np_} = {ns * np_} effective batch"
            )
        else:
            print(f"Sampling: INDEPENDENT  batch_size={self.config.batch_size}")

        dataset = self._build_dataset()
        data_iter = iter(dataset)

        for epoch in range(self.config.epochs):
            logs = self._run_epoch_steps(
                data_iter, self.config.steps_per_epoch,
            )
            steps = self.config.steps_per_epoch
            avg_k_prime = float(logs["k_prime"]) / steps

            # --- TensorBoard logging (every epoch) ---
            self._log_tensorboard(epoch, logs)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(
                    epoch,
                    float(logs["loss"]),
                    float(logs["bellman_loss"]),
                    float(logs["euler_loss"]),
                    avg_k_prime,
                )

            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                save_checkpoint_basic_dist(
                    self.value_net, self.policy_net, epoch,
                    save_dir=self.checkpoint_dir,
                )

        self.summary_writer.flush()

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def _get_current_lr(self) -> float:
        """Return the current learning rate as a Python float."""
        if isinstance(
            self.optimizer.learning_rate,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            return float(
                self.optimizer.learning_rate(self.optimizer.iterations)
            )
        return float(self.optimizer.learning_rate)

    def _log_tensorboard(
        self,
        epoch: int,
        logs: Dict[str, tf.Tensor],
    ) -> None:
        """Write essential training-health metrics to TensorBoard.

        Metrics are grouped into panels for easy navigation:

        - **loss/** — total, bellman, euler (per-step averages)
        - **gradients/** — global norm, max element (gradient health)
        - **policy/** — mean K' output (policy behaviour)
        - **optimizer/** — learning rate schedule
        - **weights/** — L2 norms of policy & value net parameters
        """
        steps = self.config.steps_per_epoch
        step = epoch  # scalar step for x-axis

        with self.summary_writer.as_default(step=step):
            # ---- Losses (per-step averages) ----
            avg_loss = float(logs["loss"]) / steps
            avg_bellman = float(logs["bellman_loss"]) / steps
            avg_euler = float(logs["euler_loss"]) / steps

            tf.summary.scalar("loss/total", avg_loss)
            tf.summary.scalar("loss/bellman", avg_bellman)
            tf.summary.scalar("loss/euler", avg_euler)
            # Ratio helps detect if one term dominates the other
            if abs(avg_loss) > 1e-12:
                tf.summary.scalar(
                    "loss/bellman_fraction",
                    avg_bellman / avg_loss,
                )

            # ---- Gradient health ----
            avg_grad_norm = float(logs["grad_norm"]) / steps
            tf.summary.scalar("gradients/global_norm_avg", avg_grad_norm)
            tf.summary.scalar(
                "gradients/max_element", float(logs["grad_max"]),
            )

            # ---- Policy / value diagnostics ----
            avg_k_prime = float(logs["k_prime"]) / steps
            tf.summary.scalar("policy/mean_k_prime", avg_k_prime)

            avg_dv_mean = float(logs["dv_dk_prime_mean"]) / steps
            avg_dv_std = float(logs["dv_dk_prime_std"]) / steps
            tf.summary.scalar("value/dv_dk_prime_mean", avg_dv_mean)
            tf.summary.scalar("value/dv_dk_prime_std", avg_dv_std)

            # ---- Optimizer ----
            tf.summary.scalar("optimizer/learning_rate", self._get_current_lr())

            # ---- Weight norms (detect explosion / collapse) ----
            policy_norm = tf.linalg.global_norm(
                self.policy_net.trainable_variables,
            )
            value_norm = tf.linalg.global_norm(
                self.value_net.trainable_variables,
            )
            tf.summary.scalar("weights/policy_net_norm", policy_norm)
            tf.summary.scalar("weights/value_net_norm", value_norm)
            tf.summary.scalar(
                "weights/total_trainable_params",
                sum(
                    tf.size(v).numpy()
                    for v in self.policy_net.trainable_variables
                    + self.value_net.trainable_variables
                ),
            )

    def _log_epoch(
        self,
        epoch: int,
        total_loss: float,
        bellman_loss: float,
        euler_loss: float,
        avg_k_prime: float,
    ) -> None:
        """Log training progress for an epoch."""
        steps = self.config.steps_per_epoch
        avg_loss = total_loss / steps
        avg_loss_b = bellman_loss / steps
        avg_loss_e = euler_loss / steps
        current_lr = self._get_current_lr()

        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {avg_loss:.4e} | "
            f"Bellman: {avg_loss_b:.4e} | "
            f"Euler: {avg_loss_e:.4e} | "
            f"LR: {float(current_lr):.2e} | "
            f"Avg K': {avg_k_prime:.4f}"
        )
