"""Deep Learning solver for the Risky Debt Model — Distributional Version.

Faithful distributional extension of ``dl.risky_final.RiskyModelDL_FINAL``.
Networks take economic parameters ``(rho, std, convex, fixed, eta0, eta1)``
as additional inputs, enabling a single trained model to generalise across
a range of economic parameterisations.  All networks have ``input_dim = 9``:
``(k, b, z, rho, std, convex, fixed, eta0, eta1)``.

Architecture
------------
- **Value Net** ``V(s, p)`` with Polyak-averaged target network.
- **Continuous Net** ``V_cont(s, p)`` for continuation value, with target.
- **Capital Policy** ``pi_k(s, p)`` — deterministic ``k'`` (sigmoid → denorm).
- **Debt Policy** ``pi_b(s, p)`` — deterministic ``b'`` (sigmoid → denorm).
- **Investment Policy** ``pi_i(s, p)`` — ``P(invest)`` with STE.
- **Default Policy** ``pi_d(s, p)`` — ``P(default)`` via Direct Value Comparison,
  with Polyak-averaged target for stable bond pricing.
- **Equity Issuance Nets** (invest / no-invest) — STE-gated issuance amount.

Training order (per step, shared shocks):
  0. Default policy — BCE against ``V_cont_target <= 0`` labels.
  1. Unified policy — invest + wait paths, equity FB, entropy.
  1b. Investment decision — BCE supervision.
  2. Continuous critic — double-sampling Bellman residual.
  3. Value critic — Fischer-Burmeister loss.

Optimisations
-------------
- XLA compilation for graph optimisation.
- Polyak-averaged target-network updates (value, continuous, default).
- Pre-computed bond prices reused across gradient tapes.
- ``ParamSpaceNormalizer`` normalises parameters to ``[0, 1]``.
- TensorBoard logging for all training metrics.
- Gradient clipping for training stability.
- Cosine-decay LR with optional policy warm-up.
- Epoch-scheduled entropy and equity-FB weight annealing.

All networks are initialised from pretrained risk-free distributional
checkpoint weights.
"""

from __future__ import annotations

import datetime
import os
from typing import Tuple, Dict, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.lr_schedules import WarmupCosineDecay
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import StateSpaceNormalizer, ParamSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.math import FischerBurmeisterLoss
from econ_models.dl_dist.training.dataset_builder import DatasetBuilder
from econ_models.io.checkpoints import save_checkpoint_risky_final_dist
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    BondPricingCalculator,
)
from econ_models.econ.debt_flow import DebtFlowCalculator

tfd = tfp.distributions

# Number of state dimensions: k, b, z = 3
# Number of param dimensions: rho, std, convex, fixed, eta0, eta1 = 6
# Total input_dim = 9
INPUT_DIM = 9


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class PolicyOutputs(NamedTuple):
    """Container for policy network outputs.

    Capital and debt policies use sigmoid output in [0, 1],
    denormalized to physical units via the state-space normalizer:
    K' = denorm_k(sigmoid_k) ∈ [k_min, k_max],
    B' = denorm_b(sigmoid_b) ∈ [b_min, b_max].

    Shared debt policy network across invest / no-invest paths;
    separate equity-issuance networks per path (invest / no-invest).

    Attributes:
        k_prime: Next-period capital = denorm_k(sigmoid_k).
        b_prime: Next-period debt from shared debt policy network.
        default_prob: Default probability from the default policy network.
        invest_prob: Investment probability from the investment network.
        k_prime_no_invest: Next-period capital if not investing ``(1-δ)k``.
        equity_issuance_invest: Effective equity issuance for invest path.
        issuance_gate_prob_invest: Issuance gate probability for invest path.
        equity_issuance_noinvest: Effective equity issuance for no-invest path.
        issuance_gate_prob_noinvest: Issuance gate probability for no-invest path.
    """

    k_prime: tf.Tensor
    b_prime: tf.Tensor
    default_prob: tf.Tensor
    invest_prob: tf.Tensor
    k_prime_no_invest: tf.Tensor
    equity_issuance_invest: tf.Tensor
    issuance_gate_prob_invest: tf.Tensor
    equity_issuance_noinvest: tf.Tensor
    issuance_gate_prob_noinvest: tf.Tensor


class OptimizationConfig(NamedTuple):
    """Configuration for training optimisations.

    Attributes:
        use_xla: Enable XLA compilation for faster execution.
        prefetch_buffer: ``tf.data`` prefetch buffer size.
        cross_product_sampling: Use cross-product state×param sampling.
            When True, ``batch_size_states`` state points are paired with
            every one of ``batch_size_params`` parameter combos, producing
            a batch of size ``batch_size_states × batch_size_params``.
        batch_size_states: Number of distinct state points per batch
            (only used when ``cross_product_sampling=True``).
            *None* → ``isqrt(batch_size)``.
        batch_size_params: Number of distinct parameter combos per batch
            (only used when ``cross_product_sampling=True``).
            *None* → ``batch_size // batch_size_states``.
    """

    use_xla: bool = True
    prefetch_buffer: int = tf.data.AUTOTUNE
    cross_product_sampling: bool = False
    batch_size_states: int | None = None
    batch_size_params: int | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RiskyModelDL_FINAL:
    """Deep learning solver for the Risky Debt model — Distributional Version.

    Faithful distributional extension of ``dl.risky_final.RiskyModelDL_FINAL``.

    Ten neural networks are jointly optimised:

    * Value and continuous critics are trained via double-sampling Bellman
      residuals and Fischer-Burmeister complementarity, respectively.
    * Capital, debt, equity-invest, and equity-noinvest policies are
      updated via a unified gradient tape with a single policy optimiser.
    * Investment decision is trained via BCE supervision.
    * Default policy is trained via binary cross-entropy against
      advantage-based labels derived from the target continuous network.

    All networks accept 9-dimensional input
    ``(k, b, z, rho, std, convex, fixed, eta0, eta1)``.

    Args:
        params: Economic parameters (discount factor, depreciation, etc.).
        config: Deep learning configuration (layers, learning rate, etc.).
        bonds_config: State-space and parameter bounds dictionary.
        optimization_config: Optional training-optimisation settings.
        pretrained_checkpoint_dir: Directory containing risk-free checkpoint
            weight files used for warm-start initialisation.
        pretrained_epoch: Epoch number of the risk-free checkpoint to load.
        log_dir_prefix: Directory prefix for TensorBoard logs.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bonds_config: dict,
        optimization_config: OptimizationConfig | None = None,
        pretrained_checkpoint_dir: str = "checkpoints_pretrain_dist/risk_free",
        pretrained_epoch: int = 500,
        log_dir_prefix: str = "logs/risky_final_dist",
    ) -> None:
        self.params = params
        self.config = config
        self.bonds_config = bonds_config
        self.pretrained_checkpoint_dir = pretrained_checkpoint_dir
        self.pretrained_epoch = pretrained_epoch
        self.log_dir_prefix = log_dir_prefix
        self.optimization_config = optimization_config or OptimizationConfig()

        self.config.update_value_scale(self.params)
        self.normalizer_states = StateSpaceNormalizer(self.config)
        self.normalizer_params = ParamSpaceNormalizer(self.bonds_config)
        self.epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.training_progress = tf.Variable(
            1.0, dtype=TENSORFLOW_DTYPE, trainable=False,
        )

        self._cache_constants()
        self._build_networks()
        self._build_optimizers()
        self._build_shock_distribution()
        self._compile_train_functions()
        self._build_summary_writer()
        self._load_pretrained_weights()

    # =========================================================================
    # Initialization Methods
    # =========================================================================

    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights from risk-free distributional checkpoints.

        Loads value, capital-policy, debt-policy, and default-policy
        weights and synchronises all target networks.  Validates that
        every required weight file exists before loading any.

        Raises:
            FileNotFoundError: If any required weight file is missing.
        """
        from pathlib import Path

        ckpt_dir = Path(self.pretrained_checkpoint_dir)
        epoch = self.pretrained_epoch

        weight_paths = {
            "value": ckpt_dir / f"risk_free_value_net_{epoch}.weights.h5",
            "capital": ckpt_dir / f"risk_free_capital_policy_net_{epoch}.weights.h5",
            "debt": ckpt_dir / f"risk_free_debt_policy_net_{epoch}.weights.h5",
            "default": ckpt_dir / f"risk_free_default_policy_net_{epoch}.weights.h5",
        }

        missing = [
            name for name, path in weight_paths.items() if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing pretrained weight files for: {missing}. "
                f"Searched in {ckpt_dir}"
            )

        self.value_net.load_weights(str(weight_paths["value"]))
        self.target_value_net.set_weights(self.value_net.get_weights())
        self.continuous_net.load_weights(str(weight_paths["value"]))
        self.target_continuous_net.set_weights(self.continuous_net.get_weights())
        self.default_policy_net.load_weights(str(weight_paths["default"]))
        self.target_default_policy_net.set_weights(
            self.default_policy_net.get_weights()
        )
        # self.capital_policy_net.load_weights(str(weight_paths["capital"]))
        # self.debt_policy_net.load_weights(str(weight_paths["debt"]))

        print(f"[pretrained] Loaded from {ckpt_dir} (epoch {epoch}), targets synced.")

    def _cache_constants(self) -> None:
        """Pre-cast economic parameters as TensorFlow constants.

        Only constants that appear directly in loss expressions, bond
        pricing orchestration, or the training loop are cached here.
        Economic function internals handle their own casting.
        """
        self.beta = tf.constant(self.params.discount_factor, dtype=TENSORFLOW_DTYPE)
        self.one_minus_delta = tf.constant(
            1.0 - self.params.depreciation_rate, dtype=TENSORFLOW_DTYPE,
        )
        self.half = tf.constant(0.5, dtype=TENSORFLOW_DTYPE)
        self.two = tf.constant(2.0, dtype=TENSORFLOW_DTYPE)
        self.entropy_eps = tf.constant(1e-7, dtype=TENSORFLOW_DTYPE)

        # Learning-rate / Polyak scheduling parameters (from config JSON)
        self.config.validate_scheduling_fields(
            ['lr_decay_rate', 'lr_decay_steps', 'lr_policy_scale',
             'polyak_averaging_decay', 'polyak_decay_end',
             'polyak_decay_epochs', 'target_update_freq'],
            'RiskyModelDL_FINAL_Dist',
        )
        self._lr_decay_rate = self.config.lr_decay_rate
        self._lr_decay_steps = self.config.lr_decay_steps
        self._lr_policy_scale = self.config.lr_policy_scale

        # Policy warm-up: linear LR ramp over the first N epochs
        self._policy_warmup_epochs = (
            self.config.policy_warmup_epochs
            if self.config.policy_warmup_epochs is not None else 0
        )

        # Polyak (target-network) decay annealing: start fast, stabilise later
        self._polyak_start = self.config.polyak_averaging_decay
        self._polyak_end = self.config.polyak_decay_end
        self._polyak_anneal_epochs = self.config.polyak_decay_epochs

        # Bond pricing orchestration constants
        self.one_minus_tax = tf.constant(
            1.0 - self.params.corporate_tax_rate, dtype=TENSORFLOW_DTYPE,
        )

        # NOTE: eta_0 and eta_1 are NOT cached as constants because they
        # are per-sample tensors in the distributional version.

        # --- Epoch-scheduled annealing parameters ---
        # Each schedule is a tuple: (w_start, w_end, epoch_start, epoch_end)
        def _sched(base_name: str, default_val: float):
            """Read annealing schedule from config with defaults."""
            w_s = getattr(self.config, f"{base_name}_weight_start", None)
            w_e = getattr(self.config, f"{base_name}_weight_end", None)
            return (
                w_s if w_s is not None else default_val,
                w_e if w_e is not None else default_val,
                getattr(self.config, f"{base_name}_start_epoch", None) or 0,
                getattr(self.config, f"{base_name}_end_epoch", None) or 1,
            )

        _eq_fb_w = (
            self.config.equity_fb_weight
            if self.config.equity_fb_weight is not None
            else 1.0
        )

        self._sched_eq_fb = _sched('equity_fb', _eq_fb_w)
        self._sched_cont_leak = _sched('continuous_leak', 1.0)
        self._sched_ent_cap = _sched('entropy_capital', 0.0)
        self._sched_ent_debt = _sched('entropy_debt', 0.0)
        self._sched_ent_def = _sched('entropy_default', 0.0)

        # tf.Variables so XLA-compiled _train_step_impl sees updates each epoch
        self.equity_fb_weight = tf.Variable(
            self._sched_eq_fb[0], dtype=TENSORFLOW_DTYPE, trainable=False)
        self.continuous_leak_weight = tf.Variable(
            self._sched_cont_leak[0], dtype=TENSORFLOW_DTYPE, trainable=False)
        self.entropy_capital_weight = tf.Variable(
            self._sched_ent_cap[0], dtype=TENSORFLOW_DTYPE, trainable=False)
        self.entropy_debt_weight = tf.Variable(
            self._sched_ent_debt[0], dtype=TENSORFLOW_DTYPE, trainable=False)
        self.entropy_default_weight = tf.Variable(
            self._sched_ent_def[0], dtype=TENSORFLOW_DTYPE, trainable=False)

    def _build_networks(self) -> None:
        """Construct neural networks with parameter-aware inputs (dim=9)."""
        # Value Network V(s, p; θV)
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueNet",
        )

        # Target Value Network Vtarg(s, p; θV-)
        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="TargetValueNet",
        )
        self.target_value_net.set_weights(self.value_net.get_weights())

        # Continuous Network Vcont(s, p; θC)
        self.continuous_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ContinuousNet",
        )
        self.target_continuous_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="TargetContinuousNet",
        )
        self.target_continuous_net.set_weights(self.continuous_net.get_weights())

        # Investment decision network
        self.investment_decision_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="InvestmentNet",
        )

        # Capital Policy Network πk(s, p; θπ_k) -> sigmoid ∈ (0, 1)
        #   K' = denorm_k(sigmoid_k) ∈ [k_min, k_max]
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="CapitalPolicyNet",
        )

        # Debt Policy Network (shared across invest / no-invest paths)
        # πb(s, p; θ) -> sigmoid ∈ (0, 1)
        #   B' = denorm_b(sigmoid_b) ∈ [b_min, b_max]
        self.debt_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="DebtPolicyNet",
        )

        # Default Policy Network πd(s, p; θπ_d) -> prob(default)
        self.default_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="DefaultPolicyNet",
        )

        # Target Default Policy Network πd_targ(s, p; θπ_d-)
        # Used in bond pricing to decouple default predictions from
        # debt policy gradients, preventing the default net from
        # artificially depressing bond prices and suppressing borrowing.
        self.target_default_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="TargetDefaultPolicyNet",
        )
        self.target_default_policy_net.set_weights(
            self.default_policy_net.get_weights()
        )

        # Equity Issuance Network (INVEST path) -> linear output x
        #   issuance_value = relu(x), gate = sigmoid(x), computed inline
        self.equity_issuance_net_invest = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="EquityIssuanceNetInvest",
        )

        # Equity Issuance Network (NO-INVEST path) -> linear output x
        #   issuance_value = relu(x), gate = sigmoid(x), computed inline
        self.equity_issuance_net_noinvest = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="EquityIssuanceNetNoinvest",
        )

        # Cache trainable variables for faster access
        # Unified policy variables (single optimizer, single gradient clip)
        self._policy_variables = (
            self.capital_policy_net.trainable_variables
            + self.debt_policy_net.trainable_variables
            + self.equity_issuance_net_invest.trainable_variables
            + self.equity_issuance_net_noinvest.trainable_variables
        )
        self._investment_variables = self.investment_decision_net.trainable_variables
        self._value_variables = self.value_net.trainable_variables
        self._continuous_variables = self.continuous_net.trainable_variables
        self._default_variables = self.default_policy_net.trainable_variables

    def _build_optimizers(self) -> None:
        """Build Adam optimizers with cosine-annealing learning-rate schedules.

        The LR decays from ``initial_learning_rate`` to
        ``alpha * initial_learning_rate`` over the full training run
        (``epochs * steps_per_epoch`` steps) following a half-cosine curve.
        ``alpha`` is derived from ``lr_decay_rate`` to keep the config
        interface unchanged.

        Policy and default optimizers use a ``WarmupCosineDecay`` schedule
        when ``policy_warmup_epochs > 0``: the LR linearly ramps from
        ~0 to full policy LR over the warm-up window, then cosine-decays.
        Critics always start at full LR (no warm-up).
        """
        total_steps = self.config.epochs * self.config.steps_per_epoch
        # Use lr_decay_rate as the minimum LR fraction (alpha)
        alpha = self._lr_decay_rate

        warmup_steps = self._policy_warmup_epochs * self.config.steps_per_epoch

        def _make_critic_lr(lr: float) -> tf.keras.optimizers.schedules.CosineDecay:
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=total_steps,
                alpha=alpha,
            )

        def _make_policy_lr(lr: float):
            if warmup_steps > 0:
                return WarmupCosineDecay(
                    initial_learning_rate=lr,
                    warmup_steps=warmup_steps,
                    decay_steps=total_steps - warmup_steps,
                    alpha=alpha,
                )
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=total_steps,
                alpha=alpha,
            )

        critic_lr = _make_critic_lr(self.config.learning_rate)
        policy_lr = _make_policy_lr(self.config.learning_rate * self._lr_policy_scale)
        investment_lr = _make_policy_lr(self.config.learning_rate * self._lr_policy_scale)
        default_lr = _make_policy_lr(self.config.learning_rate * self._lr_policy_scale)

        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.continuous_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_lr)
        self.investment_optimizer = tf.keras.optimizers.Adam(learning_rate=investment_lr)
        self.default_optimizer = tf.keras.optimizers.Adam(learning_rate=default_lr)

    def _compile_train_functions(self) -> None:
        """Wrap training functions with ``tf.function`` and optional XLA."""
        use_xla = self.optimization_config.use_xla
        self._optimized_train_step = tf.function(
            self._train_step_impl, jit_compile=use_xla,
        )
        self._soft_update_targets = tf.function(
            self._soft_update_targets_impl, jit_compile=use_xla,
        )

    def _soft_update_targets_impl(self, decay: tf.Tensor) -> None:
        """Polyak-average update of all target-network weights.

        Args:
            decay: Averaging coefficient in ``[0, 1]``.  Higher values
                retain more of the existing target network.
        """
        pairs = [
            (self.target_value_net, self.value_net),
            (self.target_continuous_net, self.continuous_net),
            (self.target_default_policy_net, self.default_policy_net),
        ]
        for target_net, source_net in pairs:
            for target_var, source_var in zip(
                target_net.trainable_variables,
                source_net.trainable_variables,
            ):
                target_var.assign(
                    decay * target_var + (1.0 - decay) * source_var
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

    def _build_shock_distribution(self) -> None:
        """Build standard normal distribution; actual std comes from dataset."""
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(1.0, TENSORFLOW_DTYPE),
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Build the training dataset with prefetching.

        If ``cross_product_sampling`` is enabled in the optimisation
        config, the dataset will use cross-product sampling so that every
        sampled economic-parameter combo is paired with the full set of
        sampled states, eliminating parameter-space gradient imbalance.
        """
        dataset = DatasetBuilder.build_dataset(
            self.config,
            self.bonds_config,
            include_debt=True,
            cross_product=self.optimization_config.cross_product_sampling,
            n_states=self.optimization_config.batch_size_states,
            n_params=self.optimization_config.batch_size_params,
        )
        dataset = dataset.prefetch(self.optimization_config.prefetch_buffer)
        return dataset

    # =========================================================================
    # Input Preparation
    # =========================================================================

    def _prepare_inputs(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
    ) -> tf.Tensor:
        """Normalise and concatenate state + param inputs (dim=9)."""
        return tf.concat([
            self.normalizer_states.normalize_capital(k),
            self.normalizer_states.normalize_debt(b),
            self.normalizer_states.normalize_productivity(z),
            self.normalizer_params.normalize_rho(rho),
            self.normalizer_params.normalize_std(std),
            self.normalizer_params.normalize_convex(convex),
            self.normalizer_params.normalize_fixed(fixed),
            self.normalizer_params.normalize_eta0(eta0),
            self.normalizer_params.normalize_eta1(eta1),
        ], axis=1)

    # =========================================================================
    # Policy Actions
    # =========================================================================

    def _get_policy_actions(
        self,
        inputs: tf.Tensor,
        k: tf.Tensor,
        b: tf.Tensor,
        training: bool = False,
    ) -> PolicyOutputs:
        """Compute all policy network outputs from online networks.

        Shared debt policy network across invest / no-invest paths;
        separate equity-issuance networks per path.
        Capital policy determines k' if investing; k'_no_invest = (1-δ)k otherwise.

        Args:
            inputs: Normalised state+param inputs, shape ``(batch, 9)``.
            k: Current capital, shape ``(batch, 1)``.
            b: Current debt, shape ``(batch, 1)``.
            training: Whether networks run in training mode.

        Returns:
            ``PolicyOutputs`` with shared debt, per-path equity, and shared
            capital/investment/default outputs.
        """
        k_prime = self.normalizer_states.denormalize_capital(
            self.capital_policy_net(inputs, training=training)
        )
        b_prime = self.normalizer_states.denormalize_debt(
            self.debt_policy_net(inputs, training=training)
        )

        default_prob = self.default_policy_net(inputs, training=training)
        invest_prob = self.investment_decision_net(inputs, training=training)
        k_prime_no_invest = self.one_minus_delta * k

        # Equity issuance: separate invest / noinvest nets
        issuance_logit_invest = self.equity_issuance_net_invest(inputs, training=training)
        equity_issuance_invest = tf.nn.relu(issuance_logit_invest)
        issuance_gate_soft_invest = tf.math.sigmoid(issuance_logit_invest)
        issuance_gate_hard_invest = tf.cast(issuance_gate_soft_invest > self.half, TENSORFLOW_DTYPE)
        issuance_gate_prob_invest = issuance_gate_soft_invest + tf.stop_gradient(issuance_gate_hard_invest - issuance_gate_soft_invest)

        issuance_logit_noinvest = self.equity_issuance_net_noinvest(inputs, training=training)
        equity_issuance_noinvest = tf.nn.relu(issuance_logit_noinvest)
        issuance_gate_soft_noinvest = tf.math.sigmoid(issuance_logit_noinvest)
        issuance_gate_hard_noinvest = tf.cast(issuance_gate_soft_noinvest > self.half, TENSORFLOW_DTYPE)
        issuance_gate_prob_noinvest = issuance_gate_soft_noinvest + tf.stop_gradient(issuance_gate_hard_noinvest - issuance_gate_soft_noinvest)

        return PolicyOutputs(
            k_prime=k_prime,
            b_prime=b_prime,
            default_prob=default_prob,
            invest_prob=invest_prob,
            k_prime_no_invest=k_prime_no_invest,
            equity_issuance_invest=equity_issuance_invest,
            issuance_gate_prob_invest=issuance_gate_prob_invest,
            equity_issuance_noinvest=equity_issuance_noinvest,
            issuance_gate_prob_noinvest=issuance_gate_prob_noinvest,
        )

    # =========================================================================
    # Bond Pricing (MC Estimation)
    # =========================================================================

    def estimate_bond_price(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_curr: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
        eps: tf.Tensor = None,
        use_target: bool = True,
    ) -> tf.Tensor:
        """Estimate bond price using default policy network.

        Uses vectorized MC sampling with parameter-aware inputs.

        Args:
            k_prime: Next period capital, shape ``(batch, 1)``.
            b_prime: Next period debt, shape ``(batch, 1)``.
            z_curr: Current productivity, shape ``(batch, 1)``.
            rho, std, convex, fixed, eta0, eta1: Per-sample parameters.
            eps: Pre-sampled shocks of shape ``(batch, n_samples)``.
                If None, samples fresh shocks.
            use_target: If True, use ``target_default_policy_net`` (stable,
                for label/target computation).  If False, use the online
                ``default_policy_net`` (allows gradient flow for policy).

        Returns:
            bond_price: Estimated bond price q, shape ``(batch, 1)``.
        """
        batch_size = tf.shape(z_curr)[0]
        n_samples = self.config.mc_sample_number_bond_priceing

        # 1. Sample future shocks and transition Z (vectorized)
        if eps is None:
            eps_normal = self.shock_dist.sample(
                sample_shape=(batch_size, n_samples),
            )
            eps = std * eps_normal

        z_curr_bc = tf.broadcast_to(z_curr, (batch_size, n_samples))
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, rho, eps,
        )

        # 2. Prepare Broadcasted Inputs — single reshape for all
        k_prime_bc = tf.broadcast_to(k_prime, (batch_size, n_samples))
        b_prime_bc = tf.broadcast_to(b_prime, (batch_size, n_samples))
        rho_bc = tf.broadcast_to(rho, (batch_size, n_samples))
        std_bc = tf.broadcast_to(std, (batch_size, n_samples))
        convex_bc = tf.broadcast_to(convex, (batch_size, n_samples))
        fixed_bc = tf.broadcast_to(fixed, (batch_size, n_samples))
        eta0_bc = tf.broadcast_to(eta0, (batch_size, n_samples))
        eta1_bc = tf.broadcast_to(eta1, (batch_size, n_samples))

        flat_shape = (batch_size * n_samples, 1)

        k_flat = tf.reshape(k_prime_bc, flat_shape)
        b_flat = tf.reshape(b_prime_bc, flat_shape)
        z_flat = tf.reshape(z_prime, flat_shape)
        rho_flat = tf.reshape(rho_bc, flat_shape)
        std_flat = tf.reshape(std_bc, flat_shape)
        convex_flat = tf.reshape(convex_bc, flat_shape)
        fixed_flat = tf.reshape(fixed_bc, flat_shape)
        eta0_flat = tf.reshape(eta0_bc, flat_shape)
        eta1_flat = tf.reshape(eta1_bc, flat_shape)

        inputs_eval = self._prepare_inputs(
            k_flat, b_flat, z_flat,
            rho_flat, std_flat, convex_flat, fixed_flat, eta0_flat, eta1_flat,
        )

        # 3. Default indicator via default policy network.
        #    use_target=True  → target_default_policy_net (stable, for labels)
        #    use_target=False → online default_policy_net (gradient flow)
        if use_target:
            is_default = self.target_default_policy_net(inputs_eval, training=False)
        else:
            is_default = self.default_policy_net(inputs_eval, training=False)

        # 4. Recovery and payoff via centralized econ modules
        profit_next = self.one_minus_tax * ProductionFunctions.cobb_douglas(
            k_flat, z_flat, self.params,
        )
        recovery = BondPricingCalculator.recovery_value(
            profit_next, k_flat, self.params,
        )
        payoff = BondPricingCalculator.bond_payoff(
            recovery, b_flat, is_default,
        )

        # 5. Average over MC samples
        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, n_samples)),
            axis=1,
            keepdims=True,
        )

        # 6. Risk-neutral pricing via centralized module
        bond_price = BondPricingCalculator.risk_neutral_price(
            expected_payoff,
            b_prime,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price,
            risk_free_price_val=1.0 / (1.0 + self.params.risk_free_rate),
        )

        return bond_price

    # =========================================================================
    # Dividend Computation
    # =========================================================================

    def _compute_dividend_common(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        b_prime: tf.Tensor,
        q: tf.Tensor,
        equity_issuance: tf.Tensor,
        issuance_gate_prob: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
        investment: tf.Tensor = None,
        adj_cost: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute dividend and raw payout using shared logic.

        Args:
            k: Current capital.
            b: Current debt.
            z: Current productivity.
            b_prime: Next period debt.
            q: Bond price.
            equity_issuance: Effective equity issuance.
            issuance_gate_prob: Gate probability.
            eta0: Issuance cost parameter.
            eta1: Issuance cost parameter.
            investment: Investment amount (optional, defaults to 0).
            adj_cost: Adjustment cost (optional, defaults to 0).

        Returns:
            Tuple of (dividend, payout).
        """
        if investment is None:
            investment = tf.zeros_like(k)
        if adj_cost is None:
            adj_cost = tf.zeros_like(k)

        revenue = self.one_minus_tax * ProductionFunctions.cobb_douglas(
            k, z, self.params
        )
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(
            b_prime, q, self.params
        )

        payout = revenue + debt_inflow + tax_shield - adj_cost - investment - b
        issuance_cost = issuance_gate_prob * (eta0 + eta1 * equity_issuance)
        dividend = payout - issuance_cost
        return dividend, payout

    def _compute_dividend(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        q: tf.Tensor,
        equity_issuance: tf.Tensor,
        issuance_gate_prob: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute dividend (cash flow) for the risky model (invest branch).

        Args:
            k: Current capital.
            b: Current debt.
            z: Productivity shock.
            k_prime: Next period capital.
            b_prime: Next period debt.
            q: Bond price.
            equity_issuance: Effective equity issuance (raw * gate_prob).
            issuance_gate_prob: Issuance gate probability ∈ [0, 1].
            convex: Per-sample adjustment cost parameter.
            fixed: Per-sample adjustment cost parameter.
            eta0: Per-sample issuance cost parameter.
            eta1: Per-sample issuance cost parameter.

        Returns:
            Tuple of (dividend, payout).
        """
        investment = ProductionFunctions.calculate_investment(
            k, k_prime, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate_dist(
            investment, k, convex, fixed,
        )
        return self._compute_dividend_common(
            k, b, z, b_prime, q, equity_issuance, issuance_gate_prob,
            eta0, eta1, investment, adj_cost
        )

    def _compute_dividend_no_invest(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        b_prime: tf.Tensor,
        q: tf.Tensor,
        equity_issuance: tf.Tensor,
        issuance_gate_prob: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute dividend when NOT investing.

        Args:
           k: Current capital.
           b: Current debt.
           z: Productivity shock.
           b_prime: Next period debt.
           q: Bond price.
           equity_issuance: Effective equity issuance.
           issuance_gate_prob: Issuance gate probability.
           eta0: Issuance cost parameter.
           eta1: Issuance cost parameter.

        Returns:
            Tuple of (dividend, payout).
        """
        return self._compute_dividend_common(
            k, b, z, b_prime, q, equity_issuance, issuance_gate_prob,
            eta0, eta1
        )

    # =========================================================================
    # Training Step
    # =========================================================================


    def _compute_equity_issuance_outputs(
        self,
        network: tf.keras.Model,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute equity issuance values and gate probabilities.

        Args:
            network: Equity issuance network model.
            inputs: Network inputs.
            training: Training flag.

        Returns:
            Tuple of (equity_issuance, issuance_gate_prob).
        """
        logits = network(inputs, training=training)
        equity_issuance = tf.nn.relu(logits)
        gate_soft = tf.math.sigmoid(logits)
        gate_hard = tf.cast(gate_soft > self.half, TENSORFLOW_DTYPE)
        gate_prob = gate_soft + tf.stop_gradient(gate_hard - gate_soft)
        return equity_issuance, gate_prob

    def _train_step_impl(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Optimized fused training step implementation (XLA-compatible).

        Training order:
        0.  Default policy update — uses target RHS for supervision.
            Runs FIRST so that bond pricing inside the policy tape
            sees a freshly calibrated default net.
        1.  Unified policy update — single tape covering invest + wait
            paths.  Loss = invest_value_obj + wait_value_obj + equity_FB
            + entropy (capital, debt, default).  One optimizer step.
        1b. Investment decision update (BCE supervision).
        2.  Continuous critic update — AiO loss (target = max over 2 paths)
        3.  Value critic update — FB loss

        ARCHITECTURE: Single unified policy optimizer with one
        gradient-clipping step.  All policy nets (capital, debt,
        equity_invest, equity_noinvest) updated jointly.

        PERFORMANCE DESIGN:
        - Bond prices computed ONCE outside tape for labels.
        - Policy nets re-run inside tape for gradient flow (cheap).
        - 2 MC bond price calls inside tape (invest/wait, shared debt).
        """
        batch_size = tf.shape(k)[0]

        # Sample all shocks once (shared across all updates within this step)
        # Standard normal, scaled by per-sample std
        eps1_normal = self.shock_dist.sample(sample_shape=(batch_size, 1))
        eps2_normal = self.shock_dist.sample(sample_shape=(batch_size, 1))
        eps1 = std * eps1_normal
        eps2 = std * eps2_normal
        eps_bond_normal = self.shock_dist.sample(
            sample_shape=(batch_size, self.config.mc_sample_number_bond_priceing),
        )
        eps_bond = std * eps_bond_normal

        inputs = self._prepare_inputs(k, b, z, rho, std, convex, fixed, eta0, eta1)
        k_prime_no_invest = self.one_minus_delta * k

        # z_prime (no policy-net dependency, shared across all updates)
        z_prime_1 = TransitionFunctions.log_ar1_transition(z, rho, eps1)
        z_prime_2 = TransitionFunctions.log_ar1_transition(z, rho, eps2)

        # ========== Pre-compute: bond prices + labels (outside tape) ==========
        k_prime_inf = self.normalizer_states.denormalize_capital(
            self.capital_policy_net(inputs, training=False)
        )
        # Shared debt net
        b_prime_inf = self.normalizer_states.denormalize_debt(
            self.debt_policy_net(inputs, training=False)
        )

        # Equity issuance for label computation (frozen, per-path, STE gate)
        equity_issuance_inf_invest, issuance_gate_prob_inf_invest = (
            self._compute_equity_issuance_outputs(
                self.equity_issuance_net_invest, inputs, training=False
            )
        )
        equity_issuance_inf_noinvest, issuance_gate_prob_inf_noinvest = (
            self._compute_equity_issuance_outputs(
                self.equity_issuance_net_noinvest, inputs, training=False
            )
        )

        # Bond prices (expensive MC — computed ONCE, shared debt)
        q_invest = self.estimate_bond_price(
            k_prime_inf, b_prime_inf, z, rho, std, convex, fixed, eta0, eta1, eps_bond,
        )
        q_no_invest = self.estimate_bond_price(
            k_prime_no_invest, b_prime_inf, z, rho, std, convex, fixed, eta0, eta1, eps_bond,
        )

        # Helper: build next-state inputs (params don't change)
        def _next_inputs(k_n, b_n, z_n):
            return self._prepare_inputs(
                k_n, b_n, z_n, rho, std, convex, fixed, eta0, eta1,
            )

        # Next-state inputs for 2 paths x 2 shocks (shared debt)
        inp_inv_1 = _next_inputs(k_prime_inf, b_prime_inf, z_prime_1)
        inp_inv_2 = _next_inputs(k_prime_inf, b_prime_inf, z_prime_2)
        inp_noinv_1 = _next_inputs(k_prime_no_invest, b_prime_inf, z_prime_1)
        inp_noinv_2 = _next_inputs(k_prime_no_invest, b_prime_inf, z_prime_2)

        # Target V for all 4 evaluations
        v_inv_1_tgt = self.target_value_net(inp_inv_1, training=False)
        v_inv_2_tgt = self.target_value_net(inp_inv_2, training=False)
        v_noinv_1_tgt = self.target_value_net(inp_noinv_1, training=False)
        v_noinv_2_tgt = self.target_value_net(inp_noinv_2, training=False)

        # Dividends for label computation (shared debt, per-path equity)
        div_inv_tgt, _ = self._compute_dividend(
            k, b, z, k_prime_inf, b_prime_inf, q_invest,
            equity_issuance_inf_invest, issuance_gate_prob_inf_invest,
            convex, fixed, eta0, eta1,
        )
        div_noinv_tgt, _ = self._compute_dividend_no_invest(
            k, b, z, b_prime_inf, q_no_invest,
            equity_issuance_inf_noinvest, issuance_gate_prob_inf_noinvest,
            eta0, eta1,
        )

        # RHS targets for 2 paths
        rhs_inv_1 = div_inv_tgt + self.beta * v_inv_1_tgt
        rhs_inv_2 = div_inv_tgt + self.beta * v_inv_2_tgt
        rhs_noinv_1 = div_noinv_tgt + self.beta * v_noinv_1_tgt
        rhs_noinv_2 = div_noinv_tgt + self.beta * v_noinv_2_tgt

        # Investment labels: invest vs no-invest
        investment_advantage_1 = rhs_inv_1 - rhs_noinv_1
        investment_advantage_2 = rhs_inv_2 - rhs_noinv_2
        investment_label = tf.stop_gradient(
            tf.cast((investment_advantage_1 + investment_advantage_2) > 0.0, TENSORFLOW_DTYPE)
        )

        # Bernoulli entropy helper
        def _bernoulli_entropy(p):
            p_c = tf.clip_by_value(p, self.entropy_eps, 1.0 - self.entropy_eps)
            return tf.reduce_mean(
                -p_c * tf.math.log(p_c)
                - (1.0 - p_c) * tf.math.log(1.0 - p_c)
            )

        clip_norm = self.config.gradient_clip_norm

        # ========== Step 0: Default Policy Update ==========
        v_cont_for_default = tf.stop_gradient(
            self.target_continuous_net(inputs, training=False)
        )
        with tf.GradientTape() as default_tape:
            d_prime_train = self.default_policy_net(inputs, training=True)

            default_label_hard = tf.cast(
                v_cont_for_default <= 0.0,
                TENSORFLOW_DTYPE,
            )
            mse_default_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    default_label_hard,
                    d_prime_train,
                    from_logits=False,
                )
            )

            # Bernoulli entropy bonus for default policy
            d_c = tf.clip_by_value(d_prime_train, self.entropy_eps, 1.0 - self.entropy_eps)
            entropy_default = tf.reduce_mean(
                -d_c * tf.math.log(d_c)
                - (1.0 - d_c) * tf.math.log(1.0 - d_c)
            )
            default_loss = mse_default_loss - self.entropy_default_weight * entropy_default

        default_grads = default_tape.gradient(
            default_loss, self._default_variables,
        )
        if clip_norm is not None and clip_norm > 0:
            default_grads, _ = tf.clip_by_global_norm(default_grads, clip_norm)
        self.default_optimizer.apply_gradients(
            zip(default_grads, self._default_variables),
        )

        # ========== Step 1: Unified Policy Update (invest + wait) ==========
        with tf.GradientTape() as policy_tape:
            # --- Shared forward passes (one call each) ---
            k_sigmoid = self.capital_policy_net(inputs, training=True)
            k_prime = self.normalizer_states.denormalize_capital(k_sigmoid)
            b_sigmoid = self.debt_policy_net(inputs, training=True)
            b_prime = self.normalizer_states.denormalize_debt(b_sigmoid)

            # Equity issuance
            equity_issuance_invest, issuance_gate_prob_invest = (
                self._compute_equity_issuance_outputs(
                    self.equity_issuance_net_invest, inputs, training=True
                )
            )
            equity_issuance_noinvest, issuance_gate_prob_noinvest = (
                self._compute_equity_issuance_outputs(
                    self.equity_issuance_net_noinvest, inputs, training=True
                )
            )

            # --- Invest path value objective ---
            q_invest_live = self.estimate_bond_price(
                k_prime, b_prime, z, rho, std, convex, fixed, eta0, eta1,
                eps_bond, use_target=False,
            )
            dividend_invest, payout_invest = self._compute_dividend(
                k, b, z, k_prime, b_prime, q_invest_live,
                equity_issuance_invest, issuance_gate_prob_invest,
                convex, fixed, eta0, eta1,
            )
            inp_inv_1_live = _next_inputs(k_prime, b_prime, z_prime_1)
            inp_inv_2_live = _next_inputs(k_prime, b_prime, z_prime_2)


            ###########################################################

            v_inv_1_raw = self.continuous_net(inp_inv_1_live, training=False)
            v_inv_2_raw = self.continuous_net(inp_inv_2_live, training=False)
            v_inv_1 = tf.where(v_inv_1_raw < 0 , self.continuous_leak_weight * v_inv_1_raw, v_inv_1_raw)
            v_inv_2 = tf.where(v_inv_2_raw < 0 , self.continuous_leak_weight * v_inv_2_raw, v_inv_2_raw)

            ###########################################################

            # v_inv_1 = self.value_net(inp_inv_1_live, training=False)
            # v_inv_2 = self.value_net(inp_inv_2_live, training=False)

            rhs_invest_1 = dividend_invest + self.beta * v_inv_1
            rhs_invest_2 = dividend_invest + self.beta * v_inv_2
            avg_rhs_invest = (rhs_invest_1 + rhs_invest_2) / self.two
            value_obj_invest_loss = -tf.reduce_mean(avg_rhs_invest)

            # --- No-invest path value objective (shared b_prime) ---
            q_no_invest_live = self.estimate_bond_price(
                k_prime_no_invest, b_prime, z, rho, std, convex, fixed, eta0, eta1,
                eps_bond, use_target=False,
            )
            dividend_no_invest, payout_no_invest = self._compute_dividend_no_invest(
                k, b, z, b_prime, q_no_invest_live,
                equity_issuance_noinvest, issuance_gate_prob_noinvest,
                eta0, eta1,
            )
            inp_wait_1 = _next_inputs(k_prime_no_invest, b_prime, z_prime_1)
            inp_wait_2 = _next_inputs(k_prime_no_invest, b_prime, z_prime_2)


            ############################################################

            v_noinv_1_raw = self.continuous_net(inp_wait_1, training=False)
            v_noinv_2_raw = self.continuous_net(inp_wait_2, training=False)
            v_noinv_1 = tf.where(v_noinv_1_raw < 0 , self.continuous_leak_weight * v_noinv_1_raw, v_noinv_1_raw)
            v_noinv_2 = tf.where(v_noinv_2_raw < 0 , self.continuous_leak_weight * v_noinv_2_raw, v_noinv_2_raw)

            ############################################################


            # v_noinv_1 = self.value_net(inp_wait_1, training=False)
            # v_noinv_2 = self.value_net(inp_wait_2, training=False)

            rhs_noinv_1_live = dividend_no_invest + self.beta * v_noinv_1
            rhs_noinv_2_live = dividend_no_invest + self.beta * v_noinv_2
            avg_rhs_noinvest = (rhs_noinv_1_live + rhs_noinv_2_live) / self.two
            value_obj_wait_loss = -tf.reduce_mean(avg_rhs_noinvest)

            # --- Equity FB (both paths) ---
            fb_eq_invest = FischerBurmeisterLoss.fb_function(
                equity_issuance_invest,
                payout_invest + equity_issuance_invest,
            )
            equity_fb_invest_loss = tf.reduce_mean(tf.square(fb_eq_invest))

            fb_eq_wait = FischerBurmeisterLoss.fb_function(
                equity_issuance_noinvest,
                payout_no_invest + equity_issuance_noinvest,
            )
            equity_fb_wait_loss = tf.reduce_mean(tf.square(fb_eq_wait))

            equity_fb_loss = (equity_fb_invest_loss + equity_fb_wait_loss) / self.two

            # --- Entropy: capital + debt (single forward pass) ---
            entropy_capital = _bernoulli_entropy(k_sigmoid)
            entropy_debt = _bernoulli_entropy(b_sigmoid)

            # --- Combined policy loss ---
            unified_policy_loss = (
                value_obj_invest_loss
                + value_obj_wait_loss
                + self.equity_fb_weight * equity_fb_loss
                - self.entropy_capital_weight * entropy_capital
                - self.entropy_debt_weight * entropy_debt
            )

        policy_grads = policy_tape.gradient(
            unified_policy_loss, self._policy_variables,
        )
        if clip_norm is not None and clip_norm > 0:
            policy_grads, _ = tf.clip_by_global_norm(policy_grads, clip_norm)
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self._policy_variables),
        )

        # ========== Step 1b: Investment Decision Update ==========
        with tf.GradientTape() as investment_tape:
            invest_prob = self.investment_decision_net(inputs, training=True)
            investment_bce_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    investment_label, invest_prob, from_logits=False,
                )
            )

        investment_grads = investment_tape.gradient(
            investment_bce_loss, self._investment_variables,
        )
        if clip_norm is not None and clip_norm > 0:
            investment_grads, _ = tf.clip_by_global_norm(investment_grads, clip_norm)
        self.investment_optimizer.apply_gradients(
            zip(investment_grads, self._investment_variables),
        )

        # ========== Compute frozen RHS targets for critic updates ==========
        # Critic target: max across 2 paths
        rhs_target_1 = tf.stop_gradient(
            tf.maximum(rhs_inv_1, rhs_noinv_1)
        )
        rhs_target_2 = tf.stop_gradient(
            tf.maximum(rhs_inv_2, rhs_noinv_2)
        )

        # ========== Step 2: Continuous Critic Update ==========
        with tf.GradientTape() as continuous_tape:
            v_cont = self.continuous_net(inputs, training=True)
            continuous_loss = tf.reduce_mean(
                (v_cont - rhs_target_1) * (v_cont - rhs_target_2)
            )

        continuous_grads = continuous_tape.gradient(
            continuous_loss, self._continuous_variables,
        )
        if clip_norm is not None and clip_norm > 0:
            continuous_grads, _ = tf.clip_by_global_norm(continuous_grads, clip_norm)
        self.continuous_optimizer.apply_gradients(
            zip(continuous_grads, self._continuous_variables),
        )

        # ========== Step 3: Value Critic Update (FB loss) ==========
        v_cont_for_fb = tf.stop_gradient(self.continuous_net(inputs, training=False))
        with tf.GradientTape() as value_tape:
            v_value = self.value_net(inputs, training=True)
            gap = v_value - v_cont_for_fb
            fb_residual = FischerBurmeisterLoss.fb_function(v_value, gap)
            fb_loss = tf.reduce_mean(tf.square(fb_residual))

        value_grads = value_tape.gradient(fb_loss, self._value_variables)
        if clip_norm is not None and clip_norm > 0:
            value_grads, _ = tf.clip_by_global_norm(value_grads, clip_norm)
        self.value_optimizer.apply_gradients(
            zip(value_grads, self._value_variables),
        )

        # ========== Compute logging metrics ==========
        avg_rhs_target = (rhs_target_1 + rhs_target_2) / self.two

        value_obj_loss = (value_obj_invest_loss + value_obj_wait_loss) / self.two

        d_prime_frozen = self.default_policy_net(inputs, training=False)
        optimal_default_for_logging = tf.cast(
            v_cont_for_default <= self.config.epsilon_value_default,
            TENSORFLOW_DTYPE,
        )
        predicted_default = tf.cast(d_prime_frozen > self.half, TENSORFLOW_DTYPE)
        default_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted_default, optimal_default_for_logging), TENSORFLOW_DTYPE)
        )

        total_policy_loss = value_obj_loss + investment_bce_loss + equity_fb_loss

        # ---- Paired statistics: negative payout vs equity issuance ----
        _eps_count = tf.constant(1e-8, dtype=TENSORFLOW_DTYPE)
        neg_mask_inv = tf.cast(payout_invest < 0.0, TENSORFLOW_DTYPE)
        neg_mask_noinv = tf.cast(payout_no_invest < 0.0, TENSORFLOW_DTYPE)
        neg_count_inv = tf.reduce_sum(neg_mask_inv) + _eps_count
        neg_count_noinv = tf.reduce_sum(neg_mask_noinv) + _eps_count

        # Mean of -payout where payout < 0
        mean_neg_payout_inv = tf.reduce_sum(
            neg_mask_inv * (-payout_invest)) / neg_count_inv
        mean_neg_payout_noinv = tf.reduce_sum(
            neg_mask_noinv * (-payout_no_invest)) / neg_count_noinv

        # Mean of equity issuance where payout < 0
        mean_iss_at_neg_inv = tf.reduce_sum(
            neg_mask_inv * equity_issuance_invest) / neg_count_inv
        mean_iss_at_neg_noinv = tf.reduce_sum(
            neg_mask_noinv * equity_issuance_noinvest) / neg_count_noinv

        # Fraction with negative payout
        frac_neg_inv = tf.reduce_mean(neg_mask_inv)
        frac_neg_noinv = tf.reduce_mean(neg_mask_noinv)

        # Invest entropy
        p_clipped = tf.clip_by_value(invest_prob, self.entropy_eps, 1.0 - self.entropy_eps)
        invest_entropy = tf.reduce_mean(
            -p_clipped * tf.math.log(p_clipped)
            - (1.0 - p_clipped) * tf.math.log(1.0 - p_clipped)
        )

        return {
            "policy_loss": total_policy_loss,
            "value_obj_loss": value_obj_loss,
            "value_obj_invest_loss": value_obj_invest_loss,
            "value_obj_wait_loss": value_obj_wait_loss,
            "investment_bce_loss": investment_bce_loss,
            "default_loss": mse_default_loss,
            "default_accuracy": default_accuracy,
            "optimal_default_rate": tf.reduce_mean(optimal_default_for_logging),
            "predicted_default_rate": tf.reduce_mean(d_prime_frozen),
            "avg_rhs": tf.reduce_mean(avg_rhs_target),
            "avg_k_prime": tf.reduce_mean(k_prime),
            "avg_b_prime": tf.reduce_mean(b_prime),
            "avg_d_prime": tf.reduce_mean(d_prime_frozen),
            "avg_invest_prob": tf.reduce_mean(invest_prob),
            "avg_invest_STE": tf.reduce_mean(
                tf.cast(invest_prob > self.half, TENSORFLOW_DTYPE)
            ),
            "avg_bond_price": tf.reduce_mean(q_invest),
            "avg_dividend": tf.reduce_mean(div_inv_tgt),
            "continuous_loss": continuous_loss,
            "avg_v_cont": tf.reduce_mean(v_cont),
            "fb_loss": fb_loss,
            "avg_v_value": tf.reduce_mean(v_value),
            "invest_entropy": invest_entropy,
            "invest_advantage_mean": tf.reduce_mean(
                (investment_advantage_1 + investment_advantage_2) / self.two
            ),
            "equity_fb_loss": equity_fb_loss,
            "avg_equity_issuance_invest": tf.reduce_mean(equity_issuance_invest),
            "avg_issuance_gate_prob_invest": tf.reduce_mean(issuance_gate_prob_invest),
            "avg_equity_issuance_noinvest": tf.reduce_mean(equity_issuance_noinvest),
            "avg_issuance_gate_prob_noinvest": tf.reduce_mean(issuance_gate_prob_noinvest),
            # Paired neg-payout vs issuance diagnostics
            "frac_neg_payout_inv": frac_neg_inv,
            "frac_neg_payout_noinv": frac_neg_noinv,
            "mean_neg_payout_inv": mean_neg_payout_inv,
            "mean_neg_payout_noinv": mean_neg_payout_noinv,
            "mean_iss_at_neg_inv": mean_iss_at_neg_inv,
            "mean_iss_at_neg_noinv": mean_iss_at_neg_noinv,
            # Bernoulli entropy diagnostics
            "entropy_capital": entropy_capital,
            "entropy_debt": entropy_debt,
            "entropy_default": entropy_default,
        }

    def train_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Public interface — delegates to XLA-optimised implementation."""
        return self._optimized_train_step(
            k, b, z, rho, std, convex, fixed, eta0, eta1,
        )

    # =========================================================================
    # Epoch and Training Loop Methods
    # =========================================================================

    def _run_epoch(self, data_iter, current_decay: float) -> Dict[str, float]:
        """Run one training epoch with optimized target updates."""
        epoch_logs = {}
        target_update_freq = self.config.target_update_freq
        decay_tensor = tf.constant(current_decay, dtype=TENSORFLOW_DTYPE)

        for step in range(self.config.steps_per_epoch):
            k, b, z, rho, std, convex, fixed, eta0, eta1 = next(data_iter)
            logs = self.train_step(k, b, z, rho, std, convex, fixed, eta0, eta1)

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            # Reduced frequency target network updates
            if step % target_update_freq == 0:
                self._soft_update_targets(decay_tensor)

        return epoch_logs

    def train(self) -> None:
        """Run the full training loop."""
        total_steps = self.config.epochs * self.config.steps_per_epoch
        warmup_str = (
            f"{self._policy_warmup_epochs} epochs"
            if self._policy_warmup_epochs > 0 else "disabled"
        )
        print(
            f"Risky Dist Model Training: {self.config.epochs} epochs, "
            f"batch={self.config.batch_size}, "
            f"XLA={self.optimization_config.use_xla}, "
            f"MC={self.config.mc_sample_number_bond_priceing}\n"
            f"  LR: CosineDecay alpha={self._lr_decay_rate}, "
            f"steps={total_steps}, warmup={warmup_str}\n"
            f"  Polyak: {self._polyak_start}→{self._polyak_end} "
            f"over {self._polyak_anneal_epochs} epochs, "
            f"update every {self.config.target_update_freq} steps\n"
            f"  Equity FB weight: {self._sched_eq_fb[0]} → {self._sched_eq_fb[1]} "
            f"over epochs [{self._sched_eq_fb[2]}, {self._sched_eq_fb[3]}]\n"
            f"  Continuous leak: {self._sched_cont_leak[0]} → {self._sched_cont_leak[1]} "
            f"over epochs [{self._sched_cont_leak[2]}, {self._sched_cont_leak[3]}]\n"
            f"  Entropy capital: {self._sched_ent_cap[0]} → {self._sched_ent_cap[1]} "
            f"over epochs [{self._sched_ent_cap[2]}, {self._sched_ent_cap[3]}]\n"
            f"  Entropy debt: {self._sched_ent_debt[0]} → {self._sched_ent_debt[1]} "
            f"over epochs [{self._sched_ent_debt[2]}, {self._sched_ent_debt[3]}]\n"
            f"  Entropy default: {self._sched_ent_def[0]} → {self._sched_ent_def[1]} "
            f"over epochs [{self._sched_ent_def[2]}, {self._sched_ent_def[3]}]\n"
            f"  Pretrained: {self.pretrained_checkpoint_dir} "
            f"(epoch {self.pretrained_epoch})\n"
            f"  Input dimension: {INPUT_DIM} (3 state + 6 param)"
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

        # Build dataset pipeline
        dataset = self._build_dataset()
        data_iter = iter(dataset)

        # Warm-up: trace the function once before timing (includes XLA compilation)
        print("Warming up XLA compilation...")
        k_w, b_w, z_w, rho_w, std_w, convex_w, fixed_w, eta0_w, eta1_w = next(data_iter)
        _ = self.train_step(k_w, b_w, z_w, rho_w, std_w, convex_w, fixed_w, eta0_w, eta1_w)
        print("Warm-up complete.\n")

        import time
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.epoch_var.assign(epoch)

            current_decay = self._get_annealed_decay(epoch)

            # Update scheduled weights for this epoch
            scheduled_vars = [
                ("continuous_leak_weight", self.continuous_leak_weight, self._sched_cont_leak),
                ("equity_fb_weight", self.equity_fb_weight, self._sched_eq_fb),
                ("entropy_capital_weight", self.entropy_capital_weight, self._sched_ent_cap),
                ("entropy_debt_weight", self.entropy_debt_weight, self._sched_ent_debt),
                ("entropy_default_weight", self.entropy_default_weight, self._sched_ent_def),
            ]

            current_log_updates = {}
            for name, var, schedule in scheduled_vars:
                val = self._linear_anneal(epoch, schedule)
                var.assign(val)
                current_log_updates[name] = val

            # Pass the current_decay to _run_epoch
            epoch_logs = self._run_epoch(data_iter, current_decay)
            epoch_logs['polyak_decay'] = current_decay
            epoch_logs.update(current_log_updates)
            epoch_time = time.time() - epoch_start_time

            # TensorBoard logging (every epoch)
            self._log_tensorboard(epoch, epoch_logs)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs, epoch_time)

            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                save_checkpoint_risky_final_dist(
                    investment_decision_net=self.investment_decision_net,
                    value_net=self.value_net,
                    capital_policy_net=self.capital_policy_net,
                    default_policy_net=self.default_policy_net,
                    continuous_net=self.continuous_net,
                    debt_policy_net=self.debt_policy_net,
                    equity_issuance_net=self.equity_issuance_net_invest,
                    equity_issuance_net_noinvest=self.equity_issuance_net_noinvest,
                    epoch=epoch,
                )

        self.summary_writer.flush()

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def _get_current_lr(self) -> float:
        """Return the current learning rate as a Python float."""
        if isinstance(
            self.value_optimizer.learning_rate,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            return float(
                self.value_optimizer.learning_rate(self.value_optimizer.iterations)
            )
        return float(self.value_optimizer.learning_rate)

    def _get_current_policy_lr(self) -> float:
        """Return the current policy learning rate as a Python float."""
        if isinstance(
            self.policy_optimizer.learning_rate,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            return float(
                self.policy_optimizer.learning_rate(
                    self.policy_optimizer.iterations
                )
            )
        return float(self.policy_optimizer.learning_rate)

    def _log_tensorboard(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
    ) -> None:
        """Write essential training-health metrics to TensorBoard.

        Metrics are grouped into panels for easy navigation:

        - **loss/** — policy, FB, continuous, default, investment BCE
        - **policy/** — mean K', B', bond price, dividend
        - **value/** — mean V, V_cont, RHS target
        - **decision/** — invest prob/STE, entropy, advantage, default accuracy
        - **optimizer/** — learning rate, Polyak decay
        - **weights/** — L2 norms of all networks
        """
        steps = self.config.steps_per_epoch
        step = epoch

        with self.summary_writer.as_default(step=step):
            # ---- Losses (per-step averages) ----
            tf.summary.scalar("loss/policy", epoch_logs.get("policy_loss", 0) / steps)
            tf.summary.scalar("loss/value_objective", epoch_logs.get("value_obj_loss", 0) / steps)
            tf.summary.scalar("loss/value_obj_invest", epoch_logs.get("value_obj_invest_loss", 0) / steps)
            tf.summary.scalar("loss/value_obj_wait", epoch_logs.get("value_obj_wait_loss", 0) / steps)
            tf.summary.scalar("loss/investment_bce", epoch_logs.get("investment_bce_loss", 0) / steps)
            tf.summary.scalar("loss/fischer_burmeister", epoch_logs.get("fb_loss", 0) / steps)
            tf.summary.scalar("loss/continuous", epoch_logs.get("continuous_loss", 0) / steps)
            tf.summary.scalar("loss/default", epoch_logs.get("default_loss", 0) / steps)
            tf.summary.scalar("loss/equity_fb", epoch_logs.get("equity_fb_loss", 0) / steps)

            # ---- Policy diagnostics ----
            tf.summary.scalar("policy/avg_k_prime", epoch_logs.get("avg_k_prime", 0) / steps)
            tf.summary.scalar("policy/avg_b_prime", epoch_logs.get("avg_b_prime", 0) / steps)
            tf.summary.scalar("policy/avg_bond_price", epoch_logs.get("avg_bond_price", 0) / steps)
            tf.summary.scalar("policy/avg_dividend", epoch_logs.get("avg_dividend", 0) / steps)
            tf.summary.scalar("policy/avg_equity_issuance_invest", epoch_logs.get("avg_equity_issuance_invest", 0) / steps)
            tf.summary.scalar("policy/avg_issuance_gate_inv", epoch_logs.get("avg_issuance_gate_prob_invest", 0) / steps)
            tf.summary.scalar("policy/avg_equity_issuance_noinvest", epoch_logs.get("avg_equity_issuance_noinvest", 0) / steps)
            tf.summary.scalar("policy/avg_issuance_gate_noinv", epoch_logs.get("avg_issuance_gate_prob_noinvest", 0) / steps)

            # ---- Decision diagnostics ----
            tf.summary.scalar("decision/avg_invest_prob", epoch_logs.get("avg_invest_prob", 0) / steps)
            tf.summary.scalar("decision/avg_invest_STE", epoch_logs.get("avg_invest_STE", 0) / steps)
            tf.summary.scalar("decision/invest_entropy", epoch_logs.get("invest_entropy", 0) / steps)
            tf.summary.scalar("decision/invest_advantage_mean", epoch_logs.get("invest_advantage_mean", 0) / steps)
            tf.summary.scalar("decision/predicted_default_rate", epoch_logs.get("predicted_default_rate", 0) / steps)
            tf.summary.scalar("decision/optimal_default_rate", epoch_logs.get("optimal_default_rate", 0) / steps)
            tf.summary.scalar("decision/default_accuracy", epoch_logs.get("default_accuracy", 0) / steps)

            # ---- Value function diagnostics ----
            tf.summary.scalar("value/avg_v_value", epoch_logs.get("avg_v_value", 0) / steps)
            tf.summary.scalar("value/avg_v_cont", epoch_logs.get("avg_v_cont", 0) / steps)
            tf.summary.scalar("value/avg_rhs", epoch_logs.get("avg_rhs", 0) / steps)

            # ---- Neg-payout vs issuance paired diagnostics ----
            tf.summary.scalar("issuance_pair/frac_neg_payout_inv", epoch_logs.get("frac_neg_payout_inv", 0) / steps)
            tf.summary.scalar("issuance_pair/frac_neg_payout_noinv", epoch_logs.get("frac_neg_payout_noinv", 0) / steps)
            tf.summary.scalar("issuance_pair/mean_neg_payout_inv", epoch_logs.get("mean_neg_payout_inv", 0) / steps)
            tf.summary.scalar("issuance_pair/mean_neg_payout_noinv", epoch_logs.get("mean_neg_payout_noinv", 0) / steps)
            tf.summary.scalar("issuance_pair/mean_iss_at_neg_inv", epoch_logs.get("mean_iss_at_neg_inv", 0) / steps)
            tf.summary.scalar("issuance_pair/mean_iss_at_neg_noinv", epoch_logs.get("mean_iss_at_neg_noinv", 0) / steps)

            # ---- Optimizer ----
            tf.summary.scalar("optimizer/learning_rate", self._get_current_lr())
            tf.summary.scalar("optimizer/policy_learning_rate", self._get_current_policy_lr())
            tf.summary.scalar("optimizer/polyak_decay", epoch_logs.get("polyak_decay", 0))
            tf.summary.scalar("optimizer/continuous_leak_weight", epoch_logs.get("continuous_leak_weight", 0))
            tf.summary.scalar("optimizer/equity_fb_weight", epoch_logs.get("equity_fb_weight", 0))
            tf.summary.scalar("optimizer/entropy_capital_weight", epoch_logs.get("entropy_capital_weight", 0))
            tf.summary.scalar("optimizer/entropy_debt_weight", epoch_logs.get("entropy_debt_weight", 0))
            tf.summary.scalar("optimizer/entropy_default_weight", epoch_logs.get("entropy_default_weight", 0))

            # ---- Entropy diagnostics ----
            tf.summary.scalar("entropy/capital", epoch_logs.get("entropy_capital", 0) / steps)
            tf.summary.scalar("entropy/debt", epoch_logs.get("entropy_debt", 0) / steps)
            tf.summary.scalar("entropy/default", epoch_logs.get("entropy_default", 0) / steps)

            # ---- Weight norms (detect explosion / collapse) ----
            tf.summary.scalar("weights/value_net_norm",
                tf.linalg.global_norm(self.value_net.trainable_variables))
            tf.summary.scalar("weights/continuous_net_norm",
                tf.linalg.global_norm(self.continuous_net.trainable_variables))
            tf.summary.scalar("weights/capital_policy_norm",
                tf.linalg.global_norm(self.capital_policy_net.trainable_variables))
            tf.summary.scalar("weights/debt_policy_norm",
                tf.linalg.global_norm(self.debt_policy_net.trainable_variables))
            tf.summary.scalar("weights/investment_net_norm",
                tf.linalg.global_norm(self.investment_decision_net.trainable_variables))
            tf.summary.scalar("weights/default_net_norm",
                tf.linalg.global_norm(self.default_policy_net.trainable_variables))
            tf.summary.scalar("weights/target_default_net_norm",
                tf.linalg.global_norm(self.target_default_policy_net.trainable_variables))
            tf.summary.scalar("weights/equity_issuance_invest_norm",
                tf.linalg.global_norm(self.equity_issuance_net_invest.trainable_variables))
            tf.summary.scalar("weights/equity_issuance_noinvest_norm",
                tf.linalg.global_norm(self.equity_issuance_net_noinvest.trainable_variables))

    def _log_epoch(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
        epoch_time: float = None,
    ) -> None:
        """Log epoch training metrics to stdout."""
        steps = self.config.steps_per_epoch
        progress_val = float(self.training_progress.numpy())

        fb_loss = epoch_logs.get('fb_loss', 0) / steps
        policy_loss = epoch_logs.get('policy_loss', 0) / steps
        inv_bce = epoch_logs.get('investment_bce_loss', 0) / steps
        continues_loss = epoch_logs.get('continuous_loss', 0) / steps
        opt_def_rate = epoch_logs.get('optimal_default_rate', 0) / steps
        pred_def_rate = epoch_logs.get('predicted_default_rate', 0) / steps
        avg_v = epoch_logs.get('avg_v_value', 0) / steps
        avg_q = epoch_logs.get('avg_bond_price', 0) / steps
        avg_k_prime = epoch_logs.get('avg_k_prime', 0) / steps
        avg_b_prime = epoch_logs.get('avg_b_prime', 0) / steps
        avg_inv_prob = epoch_logs.get('avg_invest_prob', 0) / steps
        avg_inv_STE = epoch_logs.get('avg_invest_STE', 0) / steps
        invest_adv = epoch_logs.get('invest_advantage_mean', 0) / steps
        eq_fb = epoch_logs.get('equity_fb_loss', 0) / steps
        avg_eq_inv = epoch_logs.get('avg_equity_issuance_invest', 0) / steps
        avg_gate_inv = epoch_logs.get('avg_issuance_gate_prob_invest', 0) / steps
        avg_eq_noinv = epoch_logs.get('avg_equity_issuance_noinvest', 0) / steps
        avg_gate_noinv = epoch_logs.get('avg_issuance_gate_prob_noinvest', 0) / steps
        frac_neg_inv = epoch_logs.get('frac_neg_payout_inv', 0) / steps
        frac_neg_noinv = epoch_logs.get('frac_neg_payout_noinv', 0) / steps
        neg_pay_inv = epoch_logs.get('mean_neg_payout_inv', 0) / steps
        neg_pay_noinv = epoch_logs.get('mean_neg_payout_noinv', 0) / steps
        iss_neg_inv = epoch_logs.get('mean_iss_at_neg_inv', 0) / steps
        iss_neg_noinv = epoch_logs.get('mean_iss_at_neg_noinv', 0) / steps

        time_str = f" | Time: {epoch_time:.2f}s" if epoch_time else ""
        current_lr = self._get_current_lr()
        current_policy_lr = self._get_current_policy_lr()
        polyak_decay = epoch_logs.get('polyak_decay', None)
        polyak_str = f" | Polyak: {polyak_decay:.6f}" if polyak_decay is not None else ""
        cont_leak = epoch_logs.get('continuous_leak_weight', None)
        leak_str = f" | Leak: {cont_leak:.4f}" if cont_leak is not None else ""
        eq_fb_w = epoch_logs.get('equity_fb_weight', None)
        eq_fb_w_str = f" | EqFBw: {eq_fb_w:.2f}" if eq_fb_w is not None else ""

        print(
            f"Epoch {epoch:4d} | "
            f"Prog: {progress_val:.3f} | "
            f"FB: {fb_loss:.4e} | "
            f"Pol: {policy_loss:.4e} | "
            f"InvBCE: {inv_bce:.4e} | "
            f"EqFB: {eq_fb:.4e} | "
            f"ContL: {continues_loss:.4e} | "
            f"OptD: {opt_def_rate:.3f} | "
            f"PredD: {pred_def_rate:.3f} | "
            f"InvP: {avg_inv_prob:.2%} | "
            f"InvSTE: {avg_inv_STE:.2%} | "
            f"V: {avg_v:.2f} | "
            f"Q: {avg_q:.3f} | "
            f"K': {avg_k_prime:.3f} | "
            f"B': {avg_b_prime:.3f} | "
            f"EqInv: {avg_eq_inv:.4f} | "
            f"GateInv: {avg_gate_inv:.3f} | "
            f"EqNoinv: {avg_eq_noinv:.4f} | "
            f"GateNoinv: {avg_gate_noinv:.3f} | "
            f"InvAdv: {invest_adv:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"pLR: {current_policy_lr:.2e}"
            f"{polyak_str}"
            f"{leak_str}"
            f"{eq_fb_w_str}"
            f"{time_str}"
        )
        print(
            f"         └─ NegPay vs Iss │ "
            f"Inv: frac={frac_neg_inv:.2%} -pay={neg_pay_inv:.4f} iss={iss_neg_inv:.4f} │ "
            f"NoInv: frac={frac_neg_noinv:.2%} -pay={neg_pay_noinv:.4f} iss={iss_neg_noinv:.4f}"
        )

    def _get_annealed_decay(self, epoch: int) -> float:
        """Linearly anneal the Polyak averaging decay over training.

        Starts at ``_polyak_start`` (lower value → faster target updates)
        and linearly ramps to ``_polyak_end`` (higher value → slower,
        more stable updates) over ``_polyak_anneal_epochs`` epochs.
        After the annealing window the decay stays at ``_polyak_end``.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Polyak averaging coefficient for this epoch.
        """
        t = min(epoch / max(self._polyak_anneal_epochs, 1), 1.0)
        return self._polyak_start + t * (self._polyak_end - self._polyak_start)

    @staticmethod
    def _linear_anneal(
        epoch: int, schedule: Tuple[float, float, int, int],
    ) -> float:
        """Linearly anneal a weight from ``w_start`` to ``w_end`` over
        ``[epoch_start, epoch_end]``, clamped at both ends.

        Args:
            epoch: Current training epoch (0-indexed).
            schedule: ``(w_start, w_end, epoch_start, epoch_end)``.
        """
        w_start, w_end, ep_start, ep_end = schedule
        if epoch <= ep_start:
            return w_start
        if epoch >= ep_end:
            return w_end
        t = (epoch - ep_start) / max(ep_end - ep_start, 1)
        return w_start + t * (w_end - w_start)
