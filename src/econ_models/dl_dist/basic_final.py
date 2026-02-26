# econ_models/dl_dist/basic_final.py
"""Deep Learning solver for the Basic RBC Model with investment decision (distributional).

This module implements a neural network approach to solving the basic RBC
model using value and policy function approximation.  Unlike the single-
parameter ``dl.basic_final`` module, this variant accepts economic
parameters ``(rho, std, convex, fixed)`` as additional network inputs so
that a *single* trained model generalises over the parameter distribution.

Architecture
------------
- **Value Net** ``V(s, p; theta_V)`` -- approximates the value function.
- **Capital Policy** ``pi_k(s, p; theta_k)`` -- outputs normalised ``k'``
  via sigmoid, then ``denormalize_capital`` maps to physical units.
- **Investment Policy** ``pi_i(s, p; theta_i)`` -- outputs ``P(invest)``.

Both policy networks are trained jointly:

* The capital policy maximises expected continuation value.
* The investment policy is trained via binary cross-entropy against
  advantage-based labels derived from the target value network.

The value network is trained to minimise the unbiased double-sampling
Bellman residual with a Polyak-averaged target network for stability.

Optimisations applied
---------------------
- Fused training step with persistent ``GradientTape``.
- XLA compilation for graph optimisation.
- Polyak-averaged target network updates.
- Dataset prefetching.

Example
-------
>>> from econ_models.dl_dist.basic_final import BasicModelDL_FINAL
>>> model = BasicModelDL_FINAL(params, config, bounds)
>>> model.train()
"""

from __future__ import annotations

import atexit
import datetime
import os
import shutil
import time
from typing import Dict, NamedTuple, Tuple

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import ParamSpaceNormalizer, StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl_dist.training.dataset_builder import DatasetBuilder
from econ_models.econ import AdjustmentCostCalculator, ProductionFunctions
from econ_models.io.checkpoints import save_checkpoint_basic_final_dist


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class PolicyOutputs(NamedTuple):
    """Container for policy network outputs.

    Attributes:
        capital_rate: Normalised ``k'`` from the capital policy network.
        invest_prob: Investment probability from the investment policy network.
        final_inv_rate: Investment as a fraction of current capital.
        k_prime: Next-period capital stock (physical units).
        ste_invest: Straight-Through Estimator of the discrete invest decision.
    """

    capital_rate: tf.Tensor
    invest_prob: tf.Tensor
    final_inv_rate: tf.Tensor
    k_prime: tf.Tensor
    ste_invest: tf.Tensor


class OptimizationConfig(NamedTuple):
    """Configuration for training optimisations.

    Attributes:
        use_xla: Enable XLA compilation for faster execution.
        use_mixed_precision: Enable mixed-precision (float16) training.
        prefetch_buffer: ``tf.data`` prefetch buffer size.
        cross_product_sampling: Use cross-product state×param sampling.
            When True, ``batch_size_states`` state points are paired with
            every one of ``batch_size_params`` parameter combos, producing
            a batch of size ``batch_size_states × batch_size_params``.
            This ensures every parameter region sees the full sampled
            state space, stabilising gradients across the parameter
            distribution.
        batch_size_states: Number of distinct state points per batch
            (only used when ``cross_product_sampling=True``).
            *None* → ``isqrt(batch_size)``.
        batch_size_params: Number of distinct parameter combos per batch
            (only used when ``cross_product_sampling=True``).
            *None* → ``batch_size // batch_size_states``.
    """

    use_xla: bool = True
    use_mixed_precision: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE
    cross_product_sampling: bool = False
    batch_size_states: int | None = None
    batch_size_params: int | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BasicModelDL_FINAL:
    """Deep learning solver for the Basic RBC model with investment decision (distributional).

    Both policy networks are jointly optimised to maximise expected value.
    The investment policy uses binary cross-entropy against advantage-based
    labels.  Adjustment cost uses ``invest_prob`` for the fixed-cost
    component.

    Args:
        params: Economic parameters (discount factor, depreciation, etc.).
        config: Deep learning configuration (layers, learning rate, etc.).
        bounds: State-space and parameter bounds with keys ``k_min``,
            ``k_max``, ``z_min``, ``z_max``, etc.
        optimization_config: Optional training-optimisation settings.
        pretrained_checkpoint_dir: Directory containing ``basic_dist``
            checkpoint weights for warm-start initialisation.
        pretrained_epoch: Epoch number of the basic-dist checkpoint to load.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bounds: dict,
        optimization_config: OptimizationConfig | None = None,
        pretrained_checkpoint_dir: str = "checkpoints_pretrain_dist/basic",
        pretrained_epoch: int = 6000,
        checkpoint_dir: str = "checkpoints_final_dist/basic",
        log_dir_prefix: str = "logs/basic_final_dist",
    ) -> None:
        self.params = params
        self.config = config
        self.bounds = bounds
        self.pretrained_checkpoint_dir = pretrained_checkpoint_dir
        self.pretrained_epoch = pretrained_epoch
        self.checkpoint_dir = checkpoint_dir
        self.log_dir_prefix = log_dir_prefix
        self.optimization_config = optimization_config or OptimizationConfig()

        if self.optimization_config.use_mixed_precision:
            self._enable_mixed_precision()

        self.config.update_value_scale(self.params)
        self.normalizer_states = StateSpaceNormalizer(self.config)
        self.normalizer_params = ParamSpaceNormalizer(self.bounds)

        self._cache_constants()
        self._build_networks()
        self._build_optimizers()
        self._build_summary_writer()
        self._compile_train_functions()
        self._load_pretrained_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights from ``basic_dist`` checkpoints.

        Loads the value-net and capital-policy-net weights produced by the
        basic distributional model pre-training stage (saved in
        checkpoints_pretrain_dist/basic) and synchronises the target network.

        Raises:
            FileNotFoundError: If any required weight file is missing.
        """
        from pathlib import Path

        ckpt_dir = Path(self.pretrained_checkpoint_dir)
        epoch = self.pretrained_epoch

        value_file = ckpt_dir / f"basic_value_net_{epoch}.weights.h5"
        policy_file = ckpt_dir / f"basic_policy_net_{epoch}.weights.h5"

        if not value_file.exists():
            raise FileNotFoundError(
                f"Pretrained value-net weights not found: {value_file}"
            )
        self.value_net.load_weights(str(value_file))
        self.target_value_net.set_weights(self.value_net.get_weights())

        if policy_file.exists():
            # self.capital_policy_net.load_weights(str(policy_file))
            print(f"[pretrained] Loaded capital policy net from {policy_file}")

        print(f"[pretrained] Loaded value net from {value_file}")
        print("[pretrained] Target value net synchronised.")

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

    def _enable_mixed_precision(self) -> None:
        """Activate mixed-precision (float16) training policy."""
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(
            f"Mixed precision enabled: compute={policy.compute_dtype}, "
            f"variable={policy.variable_dtype}"
        )

    def _cache_constants(self) -> None:
        """Pre-cast economic parameters as TensorFlow constants."""
        self.beta = tf.constant(
            self.params.discount_factor, dtype=TENSORFLOW_DTYPE,
        )
        self.one_minus_delta = tf.constant(
            1.0 - self.params.depreciation_rate, dtype=TENSORFLOW_DTYPE,
        )
        self.k_min = tf.constant(
            self.bounds["k_min"], dtype=TENSORFLOW_DTYPE,
        )
        self.k_max = tf.constant(
            self.bounds["k_max"], dtype=TENSORFLOW_DTYPE,
        )
        self.half = tf.constant(0.5, dtype=TENSORFLOW_DTYPE)
        self.two = tf.constant(2.0, dtype=TENSORFLOW_DTYPE)

    def _build_networks(self) -> None:
        """Construct value, target-value, and policy neural networks."""
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueNet",
        )
        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="TargetValueNet",
        )
        self.target_value_net.set_weights(self.value_net.get_weights())

        # Capital policy outputs normalised k' (sigmoid activation)
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="sigmoid",
            name="CapitalPolicyNet",
        )

        # Investment policy outputs probability [0, 1]
        self.investment_policy_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="sigmoid", scale_factor=1.0,
            name="InvestmentPolicyNet",
        )

        # Cache variable lists for gradient computation.
        self._policy_variables = (
            self.capital_policy_net.trainable_variables
            + self.investment_policy_net.trainable_variables
        )
        self._value_variables = self.value_net.trainable_variables

    def _build_optimizers(self) -> None:
        """Build Adam optimizers with exponential-decay learning-rate schedules.

        ``lr_decay_rate``, ``lr_decay_steps``, and ``lr_policy_scale``
        are read from ``self.config`` first, falling back to
        ``self.optimization_config`` defaults.
        """
        self.config.validate_scheduling_fields(
            ['lr_decay_rate', 'lr_decay_steps', 'lr_policy_scale',
             'polyak_averaging_decay', 'polyak_decay_end',
             'polyak_decay_epochs', 'target_update_freq'],
            'BasicModelDL_FINAL_Dist'
        )
        decay_rate = self.config.lr_decay_rate
        decay_steps = self.config.lr_decay_steps
        policy_scale = self.config.lr_policy_scale

        lr_schedule_value = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False,
        )
        lr_schedule_policy = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate * policy_scale,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False,
        )

        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule_value,
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule_policy,
        )

        if decay_rate < 1.0:
            print(
                f"[LR] ExponentialDecay: critic={self.config.learning_rate:.2e}, "
                f"policy={self.config.learning_rate * policy_scale:.2e}, "
                f"× {decay_rate} every {decay_steps} steps"
            )

        if self.optimization_config.use_mixed_precision:
            self.value_optimizer = (
                tf.keras.mixed_precision.LossScaleOptimizer(
                    self.value_optimizer,
                )
            )
            self.policy_optimizer = (
                tf.keras.mixed_precision.LossScaleOptimizer(
                    self.policy_optimizer,
                )
            )

    def _sample_shocks(
        self, batch_size: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Pre-sample unit-normal shock draws outside XLA for compatibility.

        Returns:
            Two tensors of shape ``(batch_size, 1)``, each containing
            independent standard-normal draws.
        """
        eps = tf.random.normal(
            shape=(batch_size, 2), dtype=TENSORFLOW_DTYPE,
        )
        return eps[:, 0:1], eps[:, 1:2]

    def _build_dataset(self) -> tf.data.Dataset:
        """Build the training dataset with prefetching.

        If ``cross_product_sampling`` is enabled in the optimisation
        config, the dataset will use cross-product sampling so that every
        sampled economic-parameter combo is paired with the full set of
        sampled states, eliminating parameter-space gradient imbalance.
        """
        dataset = DatasetBuilder.build_dataset(
            self.config,
            self.bounds,
            include_debt=False,
            cross_product=self.optimization_config.cross_product_sampling,
            n_states=self.optimization_config.batch_size_states,
            n_params=self.optimization_config.batch_size_params,
        )
        return dataset.prefetch(self.optimization_config.prefetch_buffer)

    def _compile_train_functions(self) -> None:
        """Wrap training functions with ``tf.function`` and optional XLA."""
        use_xla = self.optimization_config.use_xla
        self._compiled_train_step = tf.function(
            self._train_step_impl, jit_compile=use_xla,
        )
        self._compiled_soft_update = tf.function(
            self._soft_update_target_impl, jit_compile=use_xla,
        )

    # ------------------------------------------------------------------
    # Normalised-input helper
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
    ) -> tf.Tensor:
        """Normalise and concatenate state + parameter inputs.

        Returns:
            Tensor of shape ``(batch, 6)``.
        """
        return tf.concat([
            self.normalizer_states.normalize_capital(k),
            self.normalizer_states.normalize_productivity(z),
            self.normalizer_params.normalize_rho(rho),
            self.normalizer_params.normalize_std(std),
            self.normalizer_params.normalize_convex(convex),
            self.normalizer_params.normalize_fixed(fixed),
        ], axis=1)

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def _soft_update_target_impl(self, decay: tf.Tensor) -> None:
        """Polyak-average update of target-network weights.

        Args:
            decay: Averaging coefficient in ``[0, 1]``.
        """
        for target_var, source_var in zip(
            self.target_value_net.trainable_variables,
            self.value_net.trainable_variables,
        ):
            target_var.assign(
                decay * target_var + (1.0 - decay) * source_var
            )

    def _train_step_impl(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eps1_normal: tf.Tensor,
        eps2_normal: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Execute one fused training step.

        Args:
            k: Current capital, shape ``(batch, 1)``.
            z: Current productivity, shape ``(batch, 1)``.
            rho: Productivity persistence, shape ``(batch, 1)``.
            std: Productivity std-dev, shape ``(batch, 1)``.
            convex: Convex adjustment cost, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)``.
            eps1_normal: Unit-normal shock draw 1, shape ``(batch, 1)``.
            eps2_normal: Unit-normal shock draw 2, shape ``(batch, 1)``.

        Returns:
            Dictionary of scalar training metrics.
        """
        # --- Scale shocks and compute future productivity ---
        eps1 = std * eps1_normal
        eps2 = std * eps2_normal
        z_prime_1 = TransitionFunctions.log_ar1_transition(z, rho, eps1)
        z_prime_2 = TransitionFunctions.log_ar1_transition(z, rho, eps2)

        # --- Normalise state inputs ---
        inputs = self._build_inputs(k, z, rho, std, convex, fixed)

        with tf.GradientTape(persistent=True) as tape:
            # ---- Policy forward pass ----
            invest_prob = self.investment_policy_net(inputs, training=True)
            hard_invest = tf.cast(invest_prob > self.half, TENSORFLOW_DTYPE)
            ste_invest = invest_prob + tf.stop_gradient(
                hard_invest - invest_prob
            )

            k_prime_norm = self.capital_policy_net(inputs, training=True)
            k_prime = self.normalizer_states.denormalize_capital(k_prime_norm)
            investment = k_prime - self.one_minus_delta * k
            k_prime_no_invest = self.one_minus_delta * k
            k_prime_no_invest = tf.clip_by_value(
                k_prime_no_invest, self.k_min, self.k_max,
            )
            final_inv_rate = investment / (k + 1e-8)

            # ---- Next-state inputs ----
            inputs_next_1 = self._build_inputs(
                k_prime, z_prime_1, rho, std, convex, fixed,
            )
            inputs_next_2 = self._build_inputs(
                k_prime, z_prime_2, rho, std, convex, fixed,
            )
            inputs_no_invest_1 = self._build_inputs(
                k_prime_no_invest, z_prime_1, rho, std, convex, fixed,
            )
            inputs_no_invest_2 = self._build_inputs(
                k_prime_no_invest, z_prime_2, rho, std, convex, fixed,
            )

            # ---- Continuation values (online network, for policy grad) ----
            v_prime_1 = self.value_net(inputs_next_1, training=False)
            v_prime_2 = self.value_net(inputs_next_2, training=False)
            v_no_invest_1 = self.value_net(
                inputs_no_invest_1, training=False,
            )
            v_no_invest_2 = self.value_net(
                inputs_no_invest_2, training=False,
            )

            # ---- Cash flow ----
            profit = ProductionFunctions.cobb_douglas(k, z, self.params)
            adj_cost, _ = AdjustmentCostCalculator.calculate_dist(
                investment, k, convex, fixed,
            )
            cash_flow = profit - adj_cost - investment

            # ---- Policy loss ----
            rhs_1 = cash_flow + self.beta * v_prime_1
            rhs_2 = cash_flow + self.beta * v_prime_2
            rhs_1_no_invest = profit + self.beta * v_no_invest_1
            rhs_2_no_invest = profit + self.beta * v_no_invest_2

            value_objective = tf.reduce_mean(
                (rhs_1 + rhs_2) / self.two
            )
            # Investment advantage labels
            avg_advantage = (
                (rhs_1 - rhs_1_no_invest) + (rhs_2 - rhs_2_no_invest)
            )
            investment_label = tf.stop_gradient(
                tf.cast(avg_advantage > 0.0, TENSORFLOW_DTYPE)
            )

            investment_bce = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    investment_label, invest_prob, from_logits=False,
                )
            )

            policy_loss = -value_objective + investment_bce

            # ---- Value (Bellman) loss ----
            v_target_1 = self.target_value_net(
                inputs_next_1, training=False,
            )
            v_target_2 = self.target_value_net(
                inputs_next_2, training=False,
            )
            v_target_no_1 = self.target_value_net(
                inputs_no_invest_1, training=False,
            )
            v_target_no_2 = self.target_value_net(
                inputs_no_invest_2, training=False,
            )

            cash_flow_sg = tf.stop_gradient(cash_flow)
            rhs_bellman_invest_1 = cash_flow_sg + self.beta * v_target_1
            rhs_bellman_invest_2 = cash_flow_sg + self.beta * v_target_2
            rhs_bellman_no_1 = profit + self.beta * v_target_no_1
            rhs_bellman_no_2 = profit + self.beta * v_target_no_2

            bellman_target_1 = tf.maximum(
                rhs_bellman_invest_1, rhs_bellman_no_1,
            )
            bellman_target_2 = tf.maximum(
                rhs_bellman_invest_2, rhs_bellman_no_2,
            )

            v_curr = self.value_net(inputs, training=True)
            bellman_resid_1 = v_curr - tf.stop_gradient(bellman_target_1)
            bellman_resid_2 = v_curr - tf.stop_gradient(bellman_target_2)

            # Unbiased double-sampling Bellman loss
            bellman_loss = tf.reduce_mean(
                bellman_resid_1 * bellman_resid_2
            )

            # Mixed-precision loss scaling
            if self.optimization_config.use_mixed_precision:
                scaled_policy_loss = (
                    self.policy_optimizer.get_scaled_loss(policy_loss)
                )
                scaled_bellman_loss = (
                    self.value_optimizer.get_scaled_loss(bellman_loss)
                )
            else:
                scaled_policy_loss = policy_loss
                scaled_bellman_loss = bellman_loss

        # --- Compute and apply gradients ---
        policy_grads = tape.gradient(
            scaled_policy_loss, self._policy_variables,
        )
        value_grads = tape.gradient(
            scaled_bellman_loss, self._value_variables,
        )
        del tape

        if self.optimization_config.use_mixed_precision:
            policy_grads = self.policy_optimizer.get_unscaled_gradients(
                policy_grads,
            )
            value_grads = self.value_optimizer.get_unscaled_gradients(
                value_grads,
            )

        # --- Gradient clipping (prevents NaN from large residuals) ---
        clip_norm = self.config.gradient_clip_norm
        if clip_norm is not None and clip_norm > 0:
            policy_grads, _ = tf.clip_by_global_norm(policy_grads, clip_norm)
            value_grads, _ = tf.clip_by_global_norm(value_grads, clip_norm)

        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self._policy_variables),
        )
        self.value_optimizer.apply_gradients(
            zip(value_grads, self._value_variables),
        )

        # --- Gradient diagnostics ---
        policy_grad_norm = tf.linalg.global_norm(policy_grads)
        value_grad_norm = tf.linalg.global_norm(value_grads)
        policy_grad_max = tf.reduce_max(
            [tf.reduce_max(tf.abs(g)) for g in policy_grads if g is not None]
        )
        value_grad_max = tf.reduce_max(
            [tf.reduce_max(tf.abs(g)) for g in value_grads if g is not None]
        )

        # --- Investment-decision diagnostics ---
        invest_entropy = tf.reduce_mean(
            -invest_prob * tf.math.log(invest_prob + 1e-8)
            - (1.0 - invest_prob) * tf.math.log(1.0 - invest_prob + 1e-8)
        )
        invest_advantage_mean = tf.reduce_mean(avg_advantage)

        return {
            "policy_loss": policy_loss,
            "bellman_loss": bellman_loss,
            "investment_bce": investment_bce,
            "value_objective": value_objective,
            "mean_capital_rate": tf.reduce_mean(k_prime_norm),
            "mean_invest_ste": tf.reduce_mean(ste_invest),
            "mean_invest_prob": tf.reduce_mean(invest_prob),
            "mean_final_inv_rate": tf.reduce_mean(final_inv_rate),
            "avg_k_prime": tf.reduce_mean(k_prime),
            "avg_v_value": tf.reduce_mean(v_curr),
            "avg_cash_flow": tf.reduce_mean(cash_flow),
            "avg_profit": tf.reduce_mean(profit),
            "avg_adj_cost": tf.reduce_mean(adj_cost),
            "invest_entropy": invest_entropy,
            "invest_advantage_mean": invest_advantage_mean,
            "policy_grad_norm": policy_grad_norm,
            "value_grad_norm": value_grad_norm,
            "policy_grad_max": policy_grad_max,
            "value_grad_max": value_grad_max,
        }

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
        """Public interface for a single training step.

        Args:
            k: Current capital values, shape ``(batch, 1)``.
            z: Current productivity values, shape ``(batch, 1)``.
            rho: Productivity persistence, shape ``(batch, 1)``.
            std: Productivity std-dev, shape ``(batch, 1)``.
            convex: Convex adjustment cost, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)``.
            eps1: Unit-normal shock draw 1, shape ``(batch, 1)``.
            eps2: Unit-normal shock draw 2, shape ``(batch, 1)``.

        Returns:
            Dictionary of scalar training metrics.
        """
        return self._compiled_train_step(
            k, z, rho, std, convex, fixed, eps1, eps2,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run_epoch(
        self, data_iter, decay: float,
    ) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            data_iter: Iterator over ``(k, z, rho, std, convex, fixed)``
                training batches.
            decay: Polyak-averaging decay for target-network updates.

        Returns:
            Accumulated metrics over all steps in the epoch.
        """
        epoch_logs: Dict[str, float] = {}
        decay_tensor = tf.constant(decay, dtype=TENSORFLOW_DTYPE)
        target_update_freq = self.config.target_update_freq

        for step in range(self.config.steps_per_epoch):
            k, z, rho, std, convex, fixed = next(data_iter)
            eps1, eps2 = self._sample_shocks(tf.shape(k)[0])
            logs = self.train_step(
                k, z, rho, std, convex, fixed, eps1, eps2,
            )

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            if step % target_update_freq == 0:
                self._compiled_soft_update(decay_tensor)

        return epoch_logs

    def train(self) -> None:
        """Run the full training loop."""

        print(
            f"Starting Basic Dist Final Model Training "
            f"for {self.config.epochs} epochs"
        )
        print(
            "Architecture: ValueNet + CapitalPolicyNet "
            "+ InvestmentPolicyNet (6-dim input)"
        )
        print(
            f"XLA: {self.optimization_config.use_xla} | "
            f"Mixed Precision: "
            f"{self.optimization_config.use_mixed_precision} | "
            f"Target Update: every "
            f"{self.config.target_update_freq} steps | "
            f"LR decay: {self.config.lr_decay_rate}"
        )
        if self.optimization_config.cross_product_sampling:
            import math
            ns = self.optimization_config.batch_size_states or int(math.isqrt(self.config.batch_size))
            np_ = self.optimization_config.batch_size_params or (self.config.batch_size // ns)
            print(
                f"Sampling: CROSS-PRODUCT  batch_size_states={ns} × batch_size_params={np_} "
                f"= {ns * np_} effective batch"
            )
        else:
            print(f"Sampling: INDEPENDENT  batch_size={self.config.batch_size}")
        print("=" * 60)

        dataset = self._build_dataset()

        # Warm up XLA compilation with one step.
        print("Warming up XLA compilation...")
        data_iter = iter(dataset)
        k_w, z_w, rho_w, std_w, conv_w, fix_w = next(data_iter)
        print(
            f"[Train] First batch shape: k={k_w.shape}, z={z_w.shape}, "
            f"rho={rho_w.shape}, std={std_w.shape}, "
            f"conv={conv_w.shape}, fix={fix_w.shape}"
        )
        eps1_w, eps2_w = self._sample_shocks(tf.shape(k_w)[0])
        _ = self.train_step(
            k_w, z_w, rho_w, std_w, conv_w, fix_w, eps1_w, eps2_w,
        )
        print("Warm-up complete.\n")

        polyak_start = self.config.polyak_averaging_decay
        polyak_end = self.config.polyak_decay_end
        polyak_ramp_epochs = self.config.polyak_decay_epochs
        print(
            f"Polyak decay: {polyak_start} → {polyak_end} "
            f"over {polyak_ramp_epochs} epochs"
        )

        for epoch in range(self.config.epochs):
            # Linear ramp of Polyak-averaging decay
            if polyak_ramp_epochs > 1 and epoch < polyak_ramp_epochs:
                t = epoch / (polyak_ramp_epochs - 1)
                decay = polyak_start + (polyak_end - polyak_start) * t
            else:
                decay = polyak_end

            epoch_start = time.time()
            epoch_logs = self._run_epoch(data_iter, decay)
            epoch_time = time.time() - epoch_start

            # --- TensorBoard logging (every epoch) ---
            self._log_tensorboard(epoch, epoch_logs, polyak_decay=decay)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs, epoch_time, polyak_decay=decay)

            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                save_checkpoint_basic_final_dist(
                    self.value_net,
                    self.capital_policy_net,
                    self.investment_policy_net,
                    epoch,
                    save_dir=self.checkpoint_dir,
                )

        self.summary_writer.flush()

    def _log_epoch(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
        epoch_time: float | None = None,
        polyak_decay: float | None = None,
    ) -> None:
        """Log epoch training metrics to stdout.

        Args:
            epoch: Current epoch number.
            epoch_logs: Accumulated metrics from the epoch.
            epoch_time: Wall-clock time for the epoch in seconds.
            polyak_decay: Current Polyak averaging decay coefficient.
        """
        steps = self.config.steps_per_epoch
        bellman = epoch_logs.get("bellman_loss", 0) / steps
        policy = epoch_logs.get("policy_loss", 0) / steps
        cap_rate = epoch_logs.get("mean_capital_rate", 0) / steps
        inv_ste = epoch_logs.get("mean_invest_ste", 0) / steps
        inv_prob = epoch_logs.get("mean_invest_prob", 0) / steps
        inv_rate = epoch_logs.get("mean_final_inv_rate", 0) / steps
        v_value = epoch_logs.get("avg_v_value", 0) / steps
        k_prime = epoch_logs.get("avg_k_prime", 0) / steps

        current_lr = self._get_current_lr(self.value_optimizer)

        time_str = f" | Time: {epoch_time:.2f}s" if epoch_time else ""
        polyak_str = f" | Polyak: {polyak_decay:.6f}" if polyak_decay is not None else ""

        print(
            f"Epoch {epoch:4d} | "
            f"Bellman: {bellman:.4e} | "
            f"Policy: {policy:.4e} | "
            f"CapRate: {cap_rate:.4f} | "
            f"InvSTE: {inv_ste:.2%} | "
            f"InvProb: {inv_prob:.2%} | "
            f"InvRate: {inv_rate:.4f} | "
            f"V: {v_value:.2f} | "
            f"K': {k_prime:.4f} | "
            f"LR: {current_lr:.2e}"
            f"{polyak_str}"
            f"{time_str}"
        )

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def _get_current_lr(self, optimizer: tf.keras.optimizers.Optimizer) -> float:
        """Return the current learning rate of *optimizer* as a Python float."""
        lr = optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return float(lr(optimizer.iterations))
        return float(lr)

    def _log_tensorboard(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
        polyak_decay: float | None = None,
    ) -> None:
        """Write training-health metrics to TensorBoard.

        Metrics are grouped into panels for easy navigation:

        - **loss/** — policy, bellman, investment BCE, value objective
        - **gradients/** — per-optimizer norms and max elements
        - **policy/** — capital rate, K', investment rate, investment
          probability, STE, entropy, advantage
        - **economics/** — cash flow, profit, adjustment cost
        - **value/** — mean value estimate
        - **optimizer/** — learning-rate schedules for both optimizers
        - **weights/** — L2 norms of all four network parameter sets
        """
        steps = self.config.steps_per_epoch
        step = epoch  # scalar step for x-axis

        with self.summary_writer.as_default(step=step):
            # ---- Losses (per-step averages) ----
            avg_policy = epoch_logs.get("policy_loss", 0) / steps
            avg_bellman = epoch_logs.get("bellman_loss", 0) / steps
            avg_bce = epoch_logs.get("investment_bce", 0) / steps
            avg_val_obj = epoch_logs.get("value_objective", 0) / steps

            tf.summary.scalar("loss/policy_total", avg_policy)
            tf.summary.scalar("loss/bellman", avg_bellman)
            tf.summary.scalar("loss/investment_bce", avg_bce)
            tf.summary.scalar("loss/value_objective", avg_val_obj)
            # Ratio: which component dominates the policy loss?
            if abs(avg_policy) > 1e-12:
                tf.summary.scalar(
                    "loss/bce_fraction",
                    avg_bce / (abs(avg_policy) + 1e-12),
                )

            # ---- Gradient health ----
            tf.summary.scalar(
                "gradients/policy_norm_avg",
                epoch_logs.get("policy_grad_norm", 0) / steps,
            )
            tf.summary.scalar(
                "gradients/value_norm_avg",
                epoch_logs.get("value_grad_norm", 0) / steps,
            )
            tf.summary.scalar(
                "gradients/policy_max",
                epoch_logs.get("policy_grad_max", 0),
            )
            tf.summary.scalar(
                "gradients/value_max",
                epoch_logs.get("value_grad_max", 0),
            )

            # ---- Policy / investment diagnostics ----
            tf.summary.scalar(
                "policy/mean_capital_rate",
                epoch_logs.get("mean_capital_rate", 0) / steps,
            )
            tf.summary.scalar(
                "policy/mean_k_prime",
                epoch_logs.get("avg_k_prime", 0) / steps,
            )
            tf.summary.scalar(
                "policy/mean_invest_prob",
                epoch_logs.get("mean_invest_prob", 0) / steps,
            )
            tf.summary.scalar(
                "policy/mean_invest_ste",
                epoch_logs.get("mean_invest_ste", 0) / steps,
            )
            tf.summary.scalar(
                "policy/mean_inv_rate",
                epoch_logs.get("mean_final_inv_rate", 0) / steps,
            )
            tf.summary.scalar(
                "policy/invest_entropy",
                epoch_logs.get("invest_entropy", 0) / steps,
            )
            tf.summary.scalar(
                "policy/invest_advantage_mean",
                epoch_logs.get("invest_advantage_mean", 0) / steps,
            )

            # ---- Economics ----
            tf.summary.scalar(
                "economics/avg_cash_flow",
                epoch_logs.get("avg_cash_flow", 0) / steps,
            )
            tf.summary.scalar(
                "economics/avg_profit",
                epoch_logs.get("avg_profit", 0) / steps,
            )
            tf.summary.scalar(
                "economics/avg_adj_cost",
                epoch_logs.get("avg_adj_cost", 0) / steps,
            )

            # ---- Value diagnostics ----
            tf.summary.scalar(
                "value/avg_v",
                epoch_logs.get("avg_v_value", 0) / steps,
            )

            # ---- Optimizer ----
            tf.summary.scalar(
                "optimizer/policy_lr",
                self._get_current_lr(self.policy_optimizer),
            )
            tf.summary.scalar(
                "optimizer/value_lr",
                self._get_current_lr(self.value_optimizer),
            )
            if polyak_decay is not None:
                tf.summary.scalar(
                    "optimizer/polyak_decay", polyak_decay,
                )

            # ---- Weight norms (detect explosion / collapse) ----
            capital_norm = tf.linalg.global_norm(
                self.capital_policy_net.trainable_variables,
            )
            invest_norm = tf.linalg.global_norm(
                self.investment_policy_net.trainable_variables,
            )
            value_norm = tf.linalg.global_norm(
                self.value_net.trainable_variables,
            )
            target_norm = tf.linalg.global_norm(
                self.target_value_net.trainable_variables,
            )
            tf.summary.scalar("weights/capital_policy_net_norm", capital_norm)
            tf.summary.scalar("weights/investment_policy_net_norm", invest_norm)
            tf.summary.scalar("weights/value_net_norm", value_norm)
            tf.summary.scalar("weights/target_value_net_norm", target_norm)

            # Value-net vs target-net drift
            tf.summary.scalar(
                "weights/value_target_drift",
                tf.abs(value_norm - target_norm),
            )

            tf.summary.scalar(
                "weights/total_trainable_params",
                sum(
                    tf.size(v).numpy()
                    for v in self.capital_policy_net.trainable_variables
                    + self.investment_policy_net.trainable_variables
                    + self.value_net.trainable_variables
                ),
            )

    # ------------------------------------------------------------------
    # Public API -- inference / simulation
    # ------------------------------------------------------------------

    @tf.function
    def _prepare_inputs(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
    ) -> tf.Tensor:
        """Normalise and concatenate state + parameter variables for network input."""
        return self._build_inputs(k, z, rho, std, convex, fixed)

    @tf.function
    def _get_combined_policy_outputs(
        self,
        inputs: tf.Tensor,
        k_curr: tf.Tensor,
        training: bool = False,
    ) -> PolicyOutputs:
        """Compute combined policy outputs from both policy networks.

        Args:
            inputs: Normalised state+param inputs, shape ``(batch, 6)``.
            k_curr: Current capital, shape ``(batch, 1)``.
            training: Whether networks run in training mode.

        Returns:
            ``PolicyOutputs`` with investment rate, probability, and ``k'``.
        """
        k_prime_norm = self.capital_policy_net(inputs, training=training)
        invest_prob = self.investment_policy_net(inputs, training=training)

        hard_invest = tf.cast(invest_prob > 0.5, TENSORFLOW_DTYPE)
        ste_invest = invest_prob + tf.stop_gradient(
            hard_invest - invest_prob
        )

        k_prime = self.normalizer_states.denormalize_capital(k_prime_norm)
        investment = k_prime - self.one_minus_delta * k_curr
        final_inv_rate = investment / (k_curr + 1e-8)
        k_prime = tf.clip_by_value(k_prime, self.k_min, self.k_max)

        return PolicyOutputs(
            capital_rate=k_prime_norm,
            invest_prob=invest_prob,
            final_inv_rate=final_inv_rate,
            k_prime=k_prime,
            ste_invest=ste_invest,
        )

    @tf.function
    def _get_optimal_actions_for_simulation(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get optimal actions for model simulation.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.
            rho: Productivity persistence, shape ``(batch, 1)``.
            std: Productivity std-dev, shape ``(batch, 1)``.
            convex: Convex adjustment cost, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)``.

        Returns:
            Tuple of ``(final_inv_rate, invest_prob, value_estimate)``.
        """
        inputs = self._prepare_inputs(k, z, rho, std, convex, fixed)
        outputs = self._get_combined_policy_outputs(
            inputs, k_curr=k, training=False,
        )
        v_opt = self.value_net(inputs, training=False)
        return outputs.final_inv_rate, outputs.invest_prob, v_opt

    @tf.function
    def get_policy_outputs(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
    ) -> PolicyOutputs:
        """Get full policy outputs for analysis.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.
            rho: Productivity persistence, shape ``(batch, 1)``.
            std: Productivity std-dev, shape ``(batch, 1)``.
            convex: Convex adjustment cost, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)``.

        Returns:
            ``PolicyOutputs`` containing all policy-network results.
        """
        inputs = self._prepare_inputs(k, z, rho, std, convex, fixed)
        return self._get_combined_policy_outputs(
            inputs, k_curr=k, training=False,
        )

    @tf.function
    def _compute_k_prime_no_invest(self, k: tf.Tensor) -> tf.Tensor:
        """Compute next-period capital with no investment (depreciation only)."""
        return self.one_minus_delta * k

    @tf.function
    def _compute_profit(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Compute Cobb-Douglas production profit."""
        return ProductionFunctions.cobb_douglas(k, z, self.params)

    @tf.function
    def _compute_adjustment_cost(
        self,
        investment: tf.Tensor,
        k: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute investment adjustment cost (distributional variant).

        Args:
            investment: Investment amount.
            k: Current capital stock.
            convex: Convex adjustment cost parameter.
            fixed: Fixed adjustment cost parameter.

        Returns:
            Tuple of ``(adjustment_cost, marginal_adjustment_cost)``.
        """
        return AdjustmentCostCalculator.calculate_dist(
            investment, k, convex, fixed,
        )

    @tf.function
    def compute_cash_flow_for_simulation(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute cash-flow components for simulation and analysis.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.
            rho: Productivity persistence, shape ``(batch, 1)``.
            std: Productivity std-dev, shape ``(batch, 1)``.
            convex: Convex adjustment cost, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)``.

        Returns:
            Tuple of ``(cash_flow, profit, investment, adjustment_cost)``.
        """
        policy_outputs = self.get_policy_outputs(
            k, z, rho, std, convex, fixed,
        )
        depreciated_k = self._compute_k_prime_no_invest(k)
        investment = policy_outputs.k_prime - depreciated_k

        profit = self._compute_profit(k, z)
        adj_cost, _ = self._compute_adjustment_cost(
            investment, k, convex, fixed,
        )
        cash_flow = profit - adj_cost - investment

        return cash_flow, profit, investment, adj_cost
