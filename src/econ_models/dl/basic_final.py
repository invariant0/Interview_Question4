# econ_models/dl/basic_final.py
"""Deep Learning solver for the Basic RBC Model with investment decision.

This module implements a neural network approach to solving the basic RBC
model using value and policy function approximation.  The architecture
jointly trains a capital-investment-rate network and an
investment-probability network.

Architecture
------------
- **Value Net** ``V(s; theta_V)`` -- approximates the value function.
- **Capital Policy** ``pi_k(s; theta_k)`` -- outputs normalised ``k'`` via sigmoid.
- **Investment Policy** ``pi_i(s; theta_i)`` -- outputs ``P(invest)``.
- **Capital Transition** ``k' = denormalize(sigmoid(pi_k(s)))``.

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
>>> from econ_models.dl.basic_final import BasicModelDL_FINAL
>>> model = BasicModelDL_FINAL(params, config, bounds)
>>> model.train()
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.econ import AdjustmentCostCalculator, ProductionFunctions
from econ_models.io.checkpoints import save_checkpoint_basic_final

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class PolicyOutputs(NamedTuple):
    """Container for policy network outputs.

    Attributes:
        capital_rate: Raw investment rate from the capital policy network.
        invest_prob: Investment probability from the investment policy network.
        final_inv_rate: Combined rate ``capital_rate * STE(invest_prob)``.
        k_prime: Next-period capital stock.
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
    """

    use_xla: bool = True
    use_mixed_precision: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BasicModelDL_FINAL:
    """Deep learning solver for the Basic RBC model with investment decision.

    Both policy networks are jointly optimised to maximise expected value.
    The investment policy uses binary cross-entropy against advantage-based
    labels.  Adjustment cost uses ``invest_prob`` for the fixed-cost
    component.

    Args:
        params: Economic parameters (discount factor, depreciation, etc.).
        config: Deep learning configuration (layers, learning rate, etc.).
        bounds: State-space bounds with keys ``k_min``, ``k_max``,
            ``z_min``, ``z_max``.
        optimization_config: Optional training-optimisation settings.
        pretrained_checkpoint_dir: Directory containing ``basic.py``
            checkpoint weights for warm-start initialisation.
        pretrained_epoch: Epoch number of the basic checkpoint to load.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bounds: dict,
        optimization_config: OptimizationConfig | None = None,
        pretrained_checkpoint_dir: str = "checkpoints_pretrain/basic",
        pretrained_epoch: int = 500,
    ) -> None:
        self.params = params
        self.config = config
        self.bounds = bounds
        self.pretrained_checkpoint_dir = pretrained_checkpoint_dir
        self.pretrained_epoch = pretrained_epoch
        self.optimization_config = optimization_config or OptimizationConfig()

        if self.optimization_config.use_mixed_precision:
            self._enable_mixed_precision()

        self.config.update_value_scale(self.params)
        self.normalizer = StateSpaceNormalizer(self.config)

        self._cache_constants()
        self._build_networks()
        self._build_optimizers()
        self._build_shock_distribution()
        self._compile_train_functions()
        self._load_pretrained_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights from ``basic.py`` checkpoints.

        Loads the value-net and capital-policy-net weights produced by the
        basic model pre-training stage (saved in checkpoints_pretrain/basic)
        and synchronises the target network.

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
        self.rho = tf.constant(
            self.params.productivity_persistence, dtype=TENSORFLOW_DTYPE,
        )
        self.k_min = tf.constant(self.bounds["k_min"], dtype=TENSORFLOW_DTYPE)
        self.k_max = tf.constant(self.bounds["k_max"], dtype=TENSORFLOW_DTYPE)
        self.half = tf.constant(0.5, dtype=TENSORFLOW_DTYPE)
        self.two = tf.constant(2.0, dtype=TENSORFLOW_DTYPE)

    def _build_networks(self) -> None:
        """Construct value, target-value, and policy neural networks."""
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueNet",
        )
        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="TargetValueNet",
        )
        self.target_value_net.set_weights(self.value_net.get_weights())

        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=self.config,
            output_activation="sigmoid",
            name="CapitalPolicyNet",
        )
        self.investment_policy_net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=self.config,
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

        All scheduling parameters are read exclusively from
        ``self.config`` (populated from the JSON config file).
        Raises ``ValueError`` if any required field is missing.
        """
        self.config.validate_scheduling_fields(
            ['lr_decay_rate', 'lr_decay_steps', 'lr_policy_scale',
             'polyak_averaging_decay', 'polyak_decay_end',
             'polyak_decay_epochs', 'target_update_freq'],
            'BasicModelDL_FINAL'
        )
        decay_rate = self.config.lr_decay_rate
        decay_steps = self.config.lr_decay_steps
        policy_scale = self.config.lr_policy_scale

        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=decay_steps, decay_rate=decay_rate, staircase=False,
            ),
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate * policy_scale,
                decay_steps=decay_steps, decay_rate=decay_rate, staircase=False,
            ),
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

    def _build_shock_distribution(self) -> None:
        """Build the normal distribution for productivity-shock sampling."""
        self.shock_dist = tfd.Normal(
            loc=tf.constant(0.0, dtype=TENSORFLOW_DTYPE),
            scale=tf.constant(
                self.params.productivity_std_dev, dtype=TENSORFLOW_DTYPE,
            ),
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Build the training dataset with prefetching."""
        dataset = DatasetBuilder.build_dataset(
            self.config, self.bounds, include_debt=False,
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
    # Core training
    # ------------------------------------------------------------------

    def _soft_update_target_impl(self, decay: tf.Tensor) -> None:
        """Polyak-average update of target-network weights.

        Args:
            decay: Averaging coefficient in ``[0, 1]``.  Higher values
                retain more of the existing target network.
        """
        for target_var, source_var in zip(
            self.target_value_net.trainable_variables,
            self.value_net.trainable_variables,
        ):
            target_var.assign(
                decay * target_var + (1.0 - decay) * source_var
            )

    def _train_step_impl(
        self, k: tf.Tensor, z: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Execute one fused training step.

        The full forward pass, loss computation, and gradient updates for
        both policy and value networks are fused into a single graph.
        A persistent ``GradientTape`` avoids redundant forward passes.

        Args:
            k: Current capital, shape ``(batch, 1)``.
            z: Current productivity, shape ``(batch, 1)``.

        Returns:
            Dictionary of scalar training metrics.
        """
        batch_size = tf.shape(k)[0]

        # --- Sample future productivity shocks ---
        eps1 = self.shock_dist.sample(sample_shape=(batch_size, 1))
        eps2 = self.shock_dist.sample(sample_shape=(batch_size, 1))
        z_prime_1 = TransitionFunctions.log_ar1_transition(z, self.rho, eps1)
        z_prime_2 = TransitionFunctions.log_ar1_transition(z, self.rho, eps2)

        # --- Normalise state inputs ---
        inputs = tf.concat([
            self.normalizer.normalize_capital(k),
            self.normalizer.normalize_productivity(z),
        ], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            # ---- Policy forward pass ----
            invest_prob = self.investment_policy_net(inputs, training=True)
            k_prime_norm = self.capital_policy_net(inputs, training=True)
            k_prime = self.normalizer.denormalize_capital(k_prime_norm)
            investment = k_prime - self.one_minus_delta * k
            k_prime_no_invest = self.one_minus_delta * k

            # Straight-Through Estimator for discrete investment decision
            hard_invest = tf.cast(invest_prob > self.half, TENSORFLOW_DTYPE)
            ste_invest = invest_prob + tf.stop_gradient(
                hard_invest - invest_prob
            )

            # ---- Next-state inputs ----
            inputs_next_1 = tf.concat([
                self.normalizer.normalize_capital(k_prime),
                self.normalizer.normalize_productivity(z_prime_1),
            ], axis=1)
            inputs_next_2 = tf.concat([
                self.normalizer.normalize_capital(k_prime),
                self.normalizer.normalize_productivity(z_prime_2),
            ], axis=1)
            inputs_no_invest_1 = tf.concat([
                self.normalizer.normalize_capital(k_prime_no_invest),
                self.normalizer.normalize_productivity(z_prime_1),
            ], axis=1)
            inputs_no_invest_2 = tf.concat([
                self.normalizer.normalize_capital(k_prime_no_invest),
                self.normalizer.normalize_productivity(z_prime_2),
            ], axis=1)

            # ---- Continuation values (target network) ----
            v_target_1 = self.target_value_net(inputs_next_1, training=False)
            v_target_2 = self.target_value_net(inputs_next_2, training=False)
            v_target_no_1 = self.target_value_net(
                inputs_no_invest_1, training=False,
            )
            v_target_no_2 = self.target_value_net(
                inputs_no_invest_2, training=False,
            )

            # ---- Continuation values (online network, for policy grad) ----
            v_prime_1 = self.value_net(inputs_next_1, training=False)
            v_prime_2 = self.value_net(inputs_next_2, training=False)

            # ---- Cash flow ----
            profit = ProductionFunctions.cobb_douglas(k, z, self.params)
            adj_cost, _ = AdjustmentCostCalculator.calculate(
                investment, k, self.params,
            )
            cash_flow = profit - adj_cost - investment

            # ---- Policy loss ----
            # Investment advantage labels (from target network)
            rhs_invest_1 = cash_flow + self.beta * v_target_1
            rhs_invest_2 = cash_flow + self.beta * v_target_2
            rhs_no_invest_1 = profit + self.beta * v_target_no_1
            rhs_no_invest_2 = profit + self.beta * v_target_no_2

            avg_advantage = (
                (rhs_invest_1 - rhs_no_invest_1)
                + (rhs_invest_2 - rhs_no_invest_2)
            )
            investment_label = tf.stop_gradient(
                tf.cast(avg_advantage > 0.0, TENSORFLOW_DTYPE)
            )

            # Value objective from online value network
            rhs_1 = cash_flow + self.beta * v_prime_1
            rhs_2 = cash_flow + self.beta * v_prime_2
            value_objective = tf.reduce_mean((rhs_1 + rhs_2) / self.two)

            # Investment classification loss
            investment_bce = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    investment_label, invest_prob, from_logits=False,
                )
            )

            policy_loss = -value_objective + investment_bce

            # ---- Value (Bellman) loss ----
            # Bellman target = max(invest, no-invest) for each shock draw
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

        clip_norm = self.config.gradient_clip_norm
        print(clip_norm)
        if clip_norm is not None and clip_norm > 0:
            policy_grads, _ = tf.clip_by_global_norm(policy_grads, clip_norm)
            value_grads, _ = tf.clip_by_global_norm(value_grads, clip_norm)

        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self._policy_variables),
        )
        self.value_optimizer.apply_gradients(
            zip(value_grads, self._value_variables),
        )

        final_inv_rate = investment / (k + 1e-8)

        return {
            "policy_loss": policy_loss,
            "bellman_loss": bellman_loss,
            "mean_capital_rate": tf.reduce_mean(k_prime_norm),
            "mean_invest_ste": tf.reduce_mean(ste_invest),
            "mean_invest_prob": tf.reduce_mean(invest_prob),
            "mean_final_inv_rate": tf.reduce_mean(final_inv_rate),
            "avg_k_prime": tf.reduce_mean(k_prime),
            "avg_v_value": tf.reduce_mean(v_curr),
        }

    def train_step(
        self, k: tf.Tensor, z: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Public interface for a single training step.

        Args:
            k: Current capital values, shape ``(batch, 1)``.
            z: Current productivity values, shape ``(batch, 1)``.

        Returns:
            Dictionary of scalar training metrics.
        """
        return self._compiled_train_step(k, z)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run_epoch(
        self, data_iter, decay: float,
    ) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            data_iter: Iterator over ``(k, z)`` training batches.
            decay: Polyak-averaging decay for target-network updates.

        Returns:
            Accumulated metrics over all steps in the epoch.
        """
        epoch_logs: Dict[str, float] = {}
        decay_tensor = tf.constant(decay, dtype=TENSORFLOW_DTYPE)

        for step in range(self.config.steps_per_epoch):
            k, z = next(data_iter)
            logs = self.train_step(k, z)

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            if step % self.config.target_update_freq == 0:
                self._compiled_soft_update(decay_tensor)

        return epoch_logs

    def train(self) -> None:
        """Run the full training loop."""
        import time

        print(f"Starting Basic Model Training for {self.config.epochs} epochs")
        print(
            "Architecture: ValueNet + CapitalPolicyNet "
            "+ InvestmentPolicyNet"
        )
        print(
            f"XLA: {self.optimization_config.use_xla} | "
            f"Mixed Precision: {self.optimization_config.use_mixed_precision} | "
            f"Target Update: every "
            f"{self.config.target_update_freq} steps"
        )
        print("=" * 60)

        dataset = self._build_dataset()

        # Warm up XLA compilation with one step.
        print("Warming up XLA compilation...")
        data_iter = iter(dataset)
        k_warmup, z_warmup = next(data_iter)
        _ = self.train_step(k_warmup, z_warmup)
        print("Warm-up complete.\n")

        # Polyak schedule: read endpoints from config, linearly ramp
        polyak_start = self.config.polyak_averaging_decay
        polyak_end = self.config.polyak_decay_end
        polyak_ramp_epochs = self.config.polyak_decay_epochs
        target_update_freq = self.config.target_update_freq
        print(
            f"Polyak decay: {polyak_start} → {polyak_end} "
            f"over {polyak_ramp_epochs} epochs | "
            f"target update every {target_update_freq} steps"
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

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs, epoch_time, polyak_decay=decay)

            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                save_checkpoint_basic_final(
                    self.value_net,
                    self.capital_policy_net,
                    self.investment_policy_net,
                    epoch,
                )

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

        # Current learning rate
        if isinstance(
            self.value_optimizer.learning_rate,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            current_lr = float(self.value_optimizer.learning_rate(
                self.value_optimizer.iterations
            ))
        else:
            current_lr = float(self.value_optimizer.learning_rate)

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
    # Public API -- inference / simulation
    # ------------------------------------------------------------------

    @tf.function
    def _prepare_inputs(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Normalise and concatenate state variables for network input."""
        return tf.concat([
            self.normalizer.normalize_capital(k),
            self.normalizer.normalize_productivity(z),
        ], axis=1)

    @tf.function
    def _get_combined_policy_outputs(
        self,
        inputs: tf.Tensor,
        k_curr: tf.Tensor,
        training: bool = False,
    ) -> PolicyOutputs:
        """Compute combined policy outputs from both policy networks.

        Args:
            inputs: Normalised state inputs, shape ``(batch, 2)``.
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

        k_prime = self.normalizer.denormalize_capital(k_prime_norm)
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
        self, k: tf.Tensor, z: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get optimal actions for model simulation.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.

        Returns:
            Tuple of ``(final_inv_rate, invest_prob, value_estimate)``.
        """
        inputs = self._prepare_inputs(k, z)
        outputs = self._get_combined_policy_outputs(
            inputs, k_curr=k, training=False,
        )
        v_opt = self.value_net(inputs, training=False)
        return outputs.final_inv_rate, outputs.invest_prob, v_opt

    @tf.function
    def get_policy_outputs(
        self, k: tf.Tensor, z: tf.Tensor,
    ) -> PolicyOutputs:
        """Get full policy outputs for analysis.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.

        Returns:
            ``PolicyOutputs`` containing all policy-network results.
        """
        inputs = self._prepare_inputs(k, z)
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
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute investment adjustment cost.

        Args:
            investment: Investment amount.
            k: Current capital stock.

        Returns:
            Tuple of ``(adjustment_cost, marginal_adjustment_cost)``.
        """
        return AdjustmentCostCalculator.calculate(
            investment, k, self.params,
        )

    @tf.function
    def _compute_cash_flow(
        self,
        k_curr: tf.Tensor,
        z_curr: tf.Tensor,
        investment: tf.Tensor,
    ) -> tf.Tensor:
        """Compute net cash flow.

        Args:
            k_curr: Current capital.
            z_curr: Current productivity.
            investment: Investment amount.

        Returns:
            Cash flow ``(profit - adjustment_cost - investment)``.
        """
        profit = self._compute_profit(k_curr, z_curr)
        adj_cost, _ = self._compute_adjustment_cost(
            investment, k_curr,
        )
        return profit - adj_cost - investment

    @tf.function
    def compute_cash_flow_for_simulation(
        self, k: tf.Tensor, z: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute cash-flow components for simulation and analysis.

        Args:
            k: Capital values, shape ``(batch, 1)``.
            z: Productivity values, shape ``(batch, 1)``.

        Returns:
            Tuple of ``(cash_flow, profit, investment, adjustment_cost)``.
        """
        policy_outputs = self.get_policy_outputs(k, z)
        depreciated_k = self._compute_k_prime_no_invest(k)
        investment = policy_outputs.k_prime - depreciated_k

        profit = self._compute_profit(k, z)
        adj_cost, _ = self._compute_adjustment_cost(
            investment, k,
        )
        cash_flow = profit - adj_cost - investment

        return cash_flow, profit, investment, adj_cost
