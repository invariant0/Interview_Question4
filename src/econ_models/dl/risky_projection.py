"""Actor-critic deep learning solver for the risky debt model (simultaneous updates).

Replicate the risky_final.py architecture with key modifications:

- **Two-branch selection network**:
    A single branch_selection_net outputs probabilities for two branches:
    invest and wait.
    Invest branch outputs (k', b', equity_raw) — 3 outputs.
    Wait branch outputs (b', equity_raw) — 2 outputs.

- **Three-branch selection with learned default**:
    V(s) = V_cont(s) * (1 - default_prob). Default probability
    comes from the branch selection network's third output.

- **Sample-based bond pricing**:
  Bond price is estimated via Monte Carlo sampling: the continuous
  value network is evaluated per MC sample to compute expected payoff.

- **Simultaneous gradient updates**:
    All gradient tapes read from the same weight snapshot; gradients are
    computed first, then all optimiser updates are applied together.

Key design choices:

- Sigmoid-denormalised capital / debt policies bounded to the state space.
- Three-branch architecture separating invest/wait/default decisions.
- Equity issuance merged into each policy net: softplus(raw) = issuance amount,
  sigmoid(raw) = issuance probability (gate).
- Target network for continuous critic with Polyak averaging.
- XLA compilation for accelerated training steps.
"""

from typing import Tuple, Dict, NamedTuple

import datetime
import os

import tensorflow as tf
import tensorflow_probability as tfp
from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.io.checkpoints import save_checkpoint_risky_projection
from econ_models.core.math import FischerBurmeisterLoss
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    BondPricingCalculator,
)
from econ_models.econ.debt_flow import DebtFlowCalculator
tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class PolicyOutputs(NamedTuple):
    """Container for all policy-network outputs at a given state.

    Capital, debt, and equity outputs come from merged policy nets.
    Branch probabilities come from the three-branch selection net
    (invest, wait, default).
    """

    k_prime_invest: tf.Tensor
    b_prime_invest: tf.Tensor
    b_prime_wait: tf.Tensor
    branch_probs: tf.Tensor
    k_prime_no_invest: tf.Tensor
    equity_issuance_invest: tf.Tensor
    equity_issuance_wait: tf.Tensor
    equity_prob_invest: tf.Tensor
    equity_prob_wait: tf.Tensor
    default_prob: tf.Tensor


class OptimizationConfig(NamedTuple):
    """XLA and prefetch settings for the training loop."""

    use_xla: bool = True
    prefetch_buffer: int = tf.data.AUTOTUNE


class _AnnealSchedule(NamedTuple):
    """Linear weight-annealing schedule between two epoch boundaries.

    The weight holds at *w_start* until *epoch_start*, linearly ramps
    to *w_end* by *epoch_end*, then holds at *w_end*.
    """

    w_start: float
    w_end: float
    epoch_start: int
    epoch_end: int


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RiskyModelDL_PROJECTION:
    """Deep learning solver for the Risky Debt model (simultaneous updates).

    Replicates the risky_final architecture but applies all gradient
    updates simultaneously rather than sequentially.

    Neural networks are jointly optimised:

    * Continuous critic is trained via double-sampling Bellman residuals.
    * Capital, debt, investment, and equity policies are updated via
      a unified gradient tape with a single policy optimiser.
    * Two-branch selection (invest/wait) is trained via BCE.
    * Default is determined by V_cont(s) <= 0 (no default network).

    All three gradient computations read the same weight snapshot; updates
    are applied together after all gradients are collected.

    Args:
        params: Economic parameters (discount factor, depreciation, etc.).
        config: Deep learning configuration (layers, learning rate, etc.).
        bounds: State-space bounds with keys ``k_min``, ``k_max``,
            ``b_min``, ``b_max``, ``z_min``, ``z_max``.
        optimization_config: Optional training-optimisation settings.
        pretrained_checkpoint_dir: Directory containing risk-free checkpoint
            weight files used for warm-start initialisation.
        pretrained_epoch: Epoch number of the risk-free checkpoint to load.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bounds: Dict[str, Tuple[float, float]],
        optimization_config: OptimizationConfig | None = None,
        pretrained_checkpoint_dir: str = "checkpoints_pretrain/risk_free",
        pretrained_epoch: int = 400,
        log_dir_prefix: str = "logs/risky_projection",
    ) -> None:
        self.params = params
        self.config = config
        self.bounds = bounds
        self.pretrained_checkpoint_dir = pretrained_checkpoint_dir
        self.pretrained_epoch = pretrained_epoch
        self.log_dir_prefix = log_dir_prefix
        self.optimization_config = optimization_config or OptimizationConfig()

        self.config.update_value_scale(self.params)
        self.normalizer = StateSpaceNormalizer(self.config)
        self.epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.training_progress = tf.Variable(
            1.0, dtype=TENSORFLOW_DTYPE, trainable=False
        )

        self._cache_constants()
        self._build_networks()
        self._build_optimizers()
        self._build_shock_distribution()
        self._compile_train_functions()
        self._build_summary_writer()
        self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights from risk-free checkpoints.

        Validate that every required weight file exists before loading
        any, so the model is never left in a partially-initialised state.

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
        }

        # Validate all files exist before loading any.
        missing = [
            name for name, path in weight_paths.items() if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing pretrained weight files for: {missing}. "
                f"Searched in {ckpt_dir}"
            )
        ## continuous net initialised from pretrained value weights
        self.continuous_net.load_weights(str(weight_paths["value"]))
        self.target_continuous_net.set_weights(self.continuous_net.get_weights())

        # Load pretrained debt weights into both invest / no-invest nets
        # NOTE: value net weights not loaded — V is a hard projection of V_cont.
        # self.capital_policy_net.load_weights(str(weight_paths["capital"]))
        # self.debt_policy_net_invest.load_weights(str(weight_paths["debt"]))
        # self.debt_policy_net_noinvest.load_weights(str(weight_paths["debt"]))

        print(f"[pretrained] Loaded weights from {ckpt_dir} (epoch {epoch})")
        print("[pretrained] Branch selection net left randomly initialised.")
        print("[pretrained] Target networks synchronised.")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_anneal(
        config_start,
        config_end,
        config_epoch_start,
        config_epoch_end,
        default_w: float,
        default_epoch_start: int = 0,
        default_epoch_end: int = 1,
    ) -> Tuple["_AnnealSchedule", tf.Variable]:
        """Create an annealing schedule paired with a non-trainable variable.

        The ``tf.Variable`` is initialised to *w_start* so that
        XLA-compiled training steps pick up per-epoch updates via
        ``Variable.assign``.
        """
        w_start = config_start if config_start is not None else default_w
        w_end = config_end if config_end is not None else default_w
        ep_start = (
            config_epoch_start if config_epoch_start is not None
            else default_epoch_start
        )
        ep_end = (
            config_epoch_end if config_epoch_end is not None
            else default_epoch_end
        )
        schedule = _AnnealSchedule(w_start, w_end, ep_start, ep_end)
        var = tf.Variable(w_start, dtype=TENSORFLOW_DTYPE, trainable=False)
        return schedule, var

    @staticmethod
    def _linear_anneal(schedule: "_AnnealSchedule", epoch: int) -> float:
        """Return the linearly annealed weight for *epoch*.

        Holds at *w_start* before *epoch_start*, ramps linearly to
        *w_end* by *epoch_end*, and holds at *w_end* afterwards.
        """
        if epoch <= schedule.epoch_start:
            return schedule.w_start
        if epoch >= schedule.epoch_end:
            return schedule.w_end
        span = max(schedule.epoch_end - schedule.epoch_start, 1)
        t = (epoch - schedule.epoch_start) / span
        return schedule.w_start + t * (schedule.w_end - schedule.w_start)

    def _build_net(
        self,
        activation: str,
        name: str,
        output_dim: int = 1,
        hidden_layers: tuple[int, ...] | None = None,
    ) -> tf.keras.Model:
        """Build a 3-input MLP with the shared config."""
        return NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=output_dim,
            config=self.config,
            output_activation=activation,
            hidden_layers=hidden_layers,
            scale_factor=1.0,
            name=name,
        )

    def _build_net_with_target(
        self, activation: str, name: str,
    ) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Build an online network and its Polyak-averaged target copy."""
        net = self._build_net(activation, name)
        target = self._build_net(activation, f"Target{name}")
        target.set_weights(net.get_weights())
        return net, target

    @staticmethod
    def _ste_default(d_soft: tf.Tensor) -> tf.Tensor:
        """Straight-through estimator for default probability.

        Forward pass returns hard 0/1 (threshold at 0.5).
        Backward pass uses the sigmoid gradient.
        """
        d_hard = tf.cast(d_soft > 0.5, TENSORFLOW_DTYPE)
        return d_soft + tf.stop_gradient(d_hard - d_soft)

    @staticmethod
    def _clip_gradients(gradients, clip_norm: float | None):
        """Clip gradients by global norm while preserving ``None`` entries."""
        if clip_norm is None or clip_norm <= 0:
            return gradients

        non_none_gradients = [gradient for gradient in gradients if gradient is not None]
        if not non_none_gradients:
            return gradients

        clipped_gradients, _ = tf.clip_by_global_norm(non_none_gradients, clip_norm)
        clipped_iterator = iter(clipped_gradients)
        return [
            next(clipped_iterator) if gradient is not None else None
            for gradient in gradients
        ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _cache_constants(self) -> None:
        """Pre-cast economic parameters and build annealing schedules."""
        self.beta = tf.constant(self.params.discount_factor, dtype=TENSORFLOW_DTYPE)
        self.one_minus_delta = tf.constant(
            1.0 - self.params.depreciation_rate, dtype=TENSORFLOW_DTYPE
        )
        self.half = tf.constant(0.5, dtype=TENSORFLOW_DTYPE)
        self.two = tf.constant(2.0, dtype=TENSORFLOW_DTYPE)
        self.four = tf.constant(4.0, dtype=TENSORFLOW_DTYPE)

        # Validate required scheduling fields
        self.config.validate_scheduling_fields(
            ['lr_policy_scale',
             'polyak_averaging_decay', 'polyak_decay_end',
             'polyak_decay_epochs', 'target_update_freq'],
            'RiskyModelDL_PROJECTION',
        )
        self._lr_policy_scale = self.config.lr_policy_scale

        # Bond pricing and equity-issuance cost constants
        self.one_minus_tax = tf.constant(
            1.0 - self.params.corporate_tax_rate, dtype=TENSORFLOW_DTYPE
        )
        self.eta_0 = tf.constant(
            self.params.equity_issuance_cost_fixed, dtype=TENSORFLOW_DTYPE
        )
        self.eta_1 = tf.constant(
            self.params.equity_issuance_cost_linear, dtype=TENSORFLOW_DTYPE
        )

        # Equity issuance softplus transform: softplus(multiplier * raw) / divisor
        self._eq_sp_mul = tf.constant(
            self.config.equity_softplus_multiplier, dtype=TENSORFLOW_DTYPE
        )
        self._eq_sp_div = tf.constant(
            self.config.equity_softplus_divisor, dtype=TENSORFLOW_DTYPE
        )
        self._eq_gate_mul = tf.constant(
            self.config.equity_gate_prob_multiplier or 1.0, dtype=TENSORFLOW_DTYPE
        )

        # --- Annealing schedules (Polyak, loss weights) ---
        self._polyak_schedule = _AnnealSchedule(
            w_start=self.config.polyak_averaging_decay,
            w_end=self.config.polyak_decay_end,
            epoch_start=0,
            epoch_end=self.config.polyak_decay_epochs,
        )

    def _build_networks(self) -> None:
        """Construct all neural networks and cache trainable-variable lists.

        Three-branch architecture:
          - invest_policy_net: [k', b', equity_raw] (3 outputs, linear)
          - wait_policy_net: [b', equity_raw] (2 outputs, linear)
          - branch_selection_net: 3-way softmax (invest, wait, default)
          - continuous_net / target_continuous_net: V_cont(s)
        """
        # Critics (with Polyak-averaged targets)
        self.continuous_net, self.target_continuous_net = self._build_net_with_target(
            "linear", "ContinuousNet",
        )

        # Policy networks (sigmoid outputs denormalised to state bounds)
        # Three-branch selection net: [invest, wait, default]
        policy_layers = self.config.policy_hidden_layers
        self.branch_selection_net = self._build_net(
            "softmax", "BranchSelectionNet", output_dim=3, hidden_layers=policy_layers
        )

        # Invest policy: [k', b', equity_raw] (3 outputs, linear)
        # sigmoid applied manually for k'/b', softplus for equity
        self.invest_policy_net = self._build_net(
            "linear",
            "InvestPolicyNet",
            output_dim=3,
            hidden_layers=policy_layers,
        )

        # Wait policy: [b', equity_raw] (2 outputs, linear)
        # sigmoid applied manually for b', softplus for equity
        self.wait_policy_net = self._build_net(
            "linear",
            "WaitPolicyNet",
            output_dim=2,
            hidden_layers=policy_layers,
        )

        # Cached trainable-variable lists.
        self._invest_policy_variables = (
            self.invest_policy_net.trainable_variables
        )
        self._wait_policy_variables = (
            self.wait_policy_net.trainable_variables
        )
        self._policy_variables = (
            self._invest_policy_variables
            + self._wait_policy_variables
        )
        self._branch_selection_variables = self.branch_selection_net.trainable_variables
        self._continuous_variables = self.continuous_net.trainable_variables

    def _build_optimizers(self) -> None:
        """Build Adam optimisers with constant learning rates."""
        base_lr = self.config.learning_rate
        policy_lr = base_lr * self._lr_policy_scale

        self.continuous_optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_lr)
        self.branch_selection_optimizer = tf.keras.optimizers.Adam(
            learning_rate=policy_lr
        )

        print(
            f"[LR] Constant: critic={base_lr:.2e}, "
            f"policy={policy_lr:.2e}"
        )

    def _compile_train_functions(self) -> None:
        """Wrap core training functions with ``tf.function`` / XLA."""
        use_xla = self.optimization_config.use_xla
        self._optimized_train_step = tf.function(
            self._train_step_impl, jit_compile=use_xla,
        )
        self._soft_update_targets = tf.function(
            self._soft_update_targets_impl, jit_compile=use_xla,
        )

    def _soft_update_targets_impl(self, decay: tf.Tensor) -> None:
        """Polyak-average all target networks toward their online sources."""
        target_source_pairs = [
            (self.target_continuous_net, self.continuous_net),
        ]
        for target_net, source_net in target_source_pairs:
            for t_var, s_var in zip(
                target_net.trainable_variables,
                source_net.trainable_variables,
            ):
                t_var.assign(decay * t_var + (1.0 - decay) * s_var)

    def _build_summary_writer(self) -> None:
        """Create a TensorBoard summary writer under *log_dir_prefix*."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.log_dir_prefix, timestamp)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoard logs -> {log_dir}")

    def _build_shock_distribution(self) -> None:
        """Build the normal distribution for productivity-shock sampling."""
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE)
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Build the training dataset with prefetching."""
        dataset = DatasetBuilder.build_dataset(
            self.config,
            self.bounds,
            include_debt=True,
        )
        dataset = dataset.prefetch(self.optimization_config.prefetch_buffer)
        return dataset

    def _prepare_inputs(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
    ) -> tf.Tensor:
        """Normalise and concatenate state variables for network input."""
        return tf.concat([
            self.normalizer.normalize_capital(k),
            self.normalizer.normalize_debt(b),
            self.normalizer.normalize_productivity(z)
        ], axis=1)

    def estimate_bond_price(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_curr: tf.Tensor,
        eps: tf.Tensor = None,
        use_target: bool = True,
    ) -> tf.Tensor:
        """Estimate bond price via sample-based Monte Carlo.

        When ``use_target=True`` (label computation, outside gradient
        tapes) the *target* continuous net is used with a hard
        ``V_cont <= 0`` threshold for default.
        When ``use_target=False`` (inside policy tapes, gradient flow
        required) the *branch selection net's* default probability
        (3rd output) is used with a straight-through estimator so that
        gradients flow through the policy.

        Args:
            k_prime: Next-period capital, shape ``(batch, 1)``.
            b_prime: Next-period debt, shape ``(batch, 1)``.
            z_curr: Current productivity, shape ``(batch, 1)``.
            eps: Pre-sampled shocks ``(batch, n_samples)``. Sampled
                fresh when *None*.
            use_target: If *True*, use the target continuous net with a
                hard default threshold.  If *False*, use the branch
                selection net's default prob with STE.

        Returns:
            Bond price *q* of shape ``(batch, 1)``.
        """
        batch_size = tf.shape(z_curr)[0]
        n_samples = self.config.mc_sample_number_bond_priceing

        # 1. Sample future shocks and transition Z (vectorized)
        if eps is None:
            eps = self.shock_dist.sample(sample_shape=(batch_size, n_samples))
        z_curr_bc = tf.broadcast_to(z_curr, (batch_size, n_samples))
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )

        # 2. Prepare broadcasted inputs
        k_prime_bc = tf.broadcast_to(k_prime, (batch_size, n_samples))
        b_prime_bc = tf.broadcast_to(b_prime, (batch_size, n_samples))
        flat_shape = (batch_size * n_samples, 1)

        k_flat = tf.reshape(k_prime_bc, flat_shape)
        b_flat = tf.reshape(b_prime_bc, flat_shape)
        z_flat = tf.reshape(z_prime, flat_shape)

        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(k_flat),
            self.normalizer.normalize_debt(b_flat),
            self.normalizer.normalize_productivity(z_flat)
        ], axis=1)

        # 3. Default indicator
        if use_target:
            # Outside gradient tapes: hard threshold on target net
            v_cont_samples = self.target_continuous_net(
                inputs_eval, training=False
            )
            is_default = tf.cast(v_cont_samples <= 0.0, TENSORFLOW_DTYPE)
        else:
            # Inside policy tapes: branch selection net default prob + STE
            branch_probs_mc = self.branch_selection_net(
                inputs_eval, training=False
            )
            d_soft = branch_probs_mc[:, 2:3]  # default probability
            d_hard = tf.cast(d_soft > 0.5, TENSORFLOW_DTYPE)
            is_default = d_soft + tf.stop_gradient(d_hard - d_soft)

        # 4. Recovery and payoff via centralized econ modules
        profit_next = self.one_minus_tax * ProductionFunctions.cobb_douglas(
            k_flat, z_flat, self.params
        )
        recovery = BondPricingCalculator.recovery_value(
            profit_next, k_flat, self.params
        )
        payoff = BondPricingCalculator.bond_payoff(
            recovery, b_flat, is_default
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
            risk_free_price_val=1.0 / (1.0 + self.params.risk_free_rate)
        )

        return bond_price

    def _estimate_bond_price_from_grid(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_flat: tf.Tensor,
        profit_next: tf.Tensor,
        batch_size: tf.Tensor,
        n_samples: int,
        use_target: bool = True,
    ) -> tf.Tensor:
        """Bond price from a pre-computed MC grid (avoids redundant z-transition).

        This method reuses *z_flat* and *profit_next* that were already
        computed once for the shared MC grid, eliminating redundant
        ``log_ar1_transition`` and ``cobb_douglas`` calls.

        Args:
            k_prime: Next-period capital ``(batch, 1)``.
            b_prime: Next-period debt ``(batch, 1)``.
            z_flat: Flattened MC z-primes ``(batch * n_samples, 1)``.
            profit_next: Pre-computed ``(1-tax) * f(k, z)`` on the MC
                grid with shape ``(batch * n_samples, 1)``.  Note: this
                uses ``k_prime`` (broadcast-flattened) and ``z_flat``.
                Callers must pass the correct profit tensor that matches
                *k_prime*.
            batch_size: Scalar batch dimension.
            n_samples: Number of MC samples.
            use_target: Use target vs online continuous net.

        Returns:
            Bond price *q* ``(batch, 1)``.
        """
        flat_shape = (batch_size * n_samples, 1)
        k_prime_bc = tf.broadcast_to(k_prime, (batch_size, n_samples))
        b_prime_bc = tf.broadcast_to(b_prime, (batch_size, n_samples))
        k_flat = tf.reshape(k_prime_bc, flat_shape)
        b_flat = tf.reshape(b_prime_bc, flat_shape)

        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(k_flat),
            self.normalizer.normalize_debt(b_flat),
            self.normalizer.normalize_productivity(z_flat)
        ], axis=1)

        if use_target:
            v_cont_samples = self.target_continuous_net(
                inputs_eval, training=False
            )
            is_default = tf.cast(v_cont_samples <= 0.0, TENSORFLOW_DTYPE)
        else:
            # Inside policy tapes: branch selection net default prob + STE
            branch_probs_mc = self.branch_selection_net(
                inputs_eval, training=False
            )
            d_soft = branch_probs_mc[:, 2:3]  # default probability
            d_hard = tf.cast(d_soft > 0.5, TENSORFLOW_DTYPE)
            is_default = d_soft + tf.stop_gradient(d_hard - d_soft)

        # Recompute profit for this k_prime (different per call)
        profit_for_k = self.one_minus_tax * ProductionFunctions.cobb_douglas(
            k_flat, z_flat, self.params
        )
        recovery = BondPricingCalculator.recovery_value(
            profit_for_k, k_flat, self.params
        )
        payoff = BondPricingCalculator.bond_payoff(
            recovery, b_flat, is_default
        )

        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, n_samples)),
            axis=1,
            keepdims=True,
        )

        bond_price = BondPricingCalculator.risk_neutral_price(
            expected_payoff,
            b_prime,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price,
            risk_free_price_val=1.0 / (1.0 + self.params.risk_free_rate)
        )

        return bond_price

    def _compute_dividend(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        q: tf.Tensor,
        equity_issuance: tf.Tensor,
        equity_prob: tf.Tensor,
        revenue: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute dividend for the invest branch (with equity issuance gate).

        equity_issuance = softplus(raw), equity_prob = sigmoid(raw).
        issuance_cost = equity_prob * (eta_0 + eta_1 * equity_issuance).
        dividend = payout - issuance_cost.

        Returns:
            ``(dividend, payout)`` where *payout* is the raw operating
            payout before equity issuance and its cost.
        """
        if revenue is None:
            revenue = self.one_minus_tax * ProductionFunctions.cobb_douglas(k, z, self.params)
        investment = ProductionFunctions.calculate_investment(k, k_prime, self.params)
        adj_cost, _ = AdjustmentCostCalculator.calculate(investment, k, self.params)
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(b_prime, q, self.params)

        payout = revenue + debt_inflow + tax_shield - adj_cost - investment - b
        issuance_cost = equity_prob*(self.eta_0 + self.eta_1 * equity_issuance)
        dividend = payout - issuance_cost
        components = {
            "profit": revenue,
            "investment": investment,
            "adj_cost": adj_cost,
            "debt_inflow": debt_inflow,
            "tax_shield": tax_shield,
            "current_debt": b,
            "payout": payout,
            "equity_issuance": equity_issuance,
            "issuance_cost": issuance_cost,
        }
        return dividend, payout, components

    def _compute_dividend_no_invest(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        b_prime: tf.Tensor,
        q: tf.Tensor,
        equity_issuance: tf.Tensor,
        equity_prob: tf.Tensor,
        revenue: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute dividend for the wait branch (with equity issuance gate).

        No investment or adjustment cost; ``k' = (1-delta) k``.
        equity_issuance = softplus(raw), equity_prob = sigmoid(raw).
        When equity_prob > 0.5 (STE), issuance is active:
          dividend = payout - issuance_cost.

        Returns:
            ``(dividend, payout)`` where *payout* is the raw operating
            payout before equity issuance and its cost.
        """
        if revenue is None:
            revenue = self.one_minus_tax * ProductionFunctions.cobb_douglas(k, z, self.params)
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(b_prime, q, self.params)

        payout = revenue + debt_inflow + tax_shield - b
        issuance_cost = equity_prob*(self.eta_0 + self.eta_1 * equity_issuance)
        dividend = payout - issuance_cost
        components = {
            "profit": revenue,
            "investment": tf.zeros_like(revenue),
            "adj_cost": tf.zeros_like(revenue),
            "debt_inflow": debt_inflow,
            "tax_shield": tax_shield,
            "current_debt": b,
            "payout": payout,
            "equity_issuance": equity_issuance,
            "issuance_cost": issuance_cost,
        }
        return dividend, payout, components

    def _train_step_impl(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Execute one simultaneous training step (XLA-compatible).

        All gradient tapes read the same weight snapshot.
        Gradients are computed first, then all optimizer updates
        are applied together at the end.

        Default probability comes from the branch selection net (3rd output).

        Optimisations vs. naive implementation:
        - Policy nets evaluated once (inside tape), with ``tf.stop_gradient``
          for label/bond-price computation — eliminates redundant pre-compute
          forward passes.
        - MC z-transition computed once; ``_estimate_bond_price_from_grid``
          reuses the flattened z grid.
        - 4 online ``continuous_net`` calls inside tapes batched into 1.
        - Logging metrics trimmed; component decomposition removed from XLA.
        - Dead ``frac_v_lt_vcont`` metric removed (always 0 by construction).
        - ``tf.reduce_mean(is_default)`` computed once and reused.

        Loss groups (computed in parallel on the same weights):

        0. Invest policy branch (k', b'_invest).
        1. Wait policy branch (b'_noinvest).
        2. Branch selection (invest / wait / default categorical CE supervision).
        3. Continuous critic (double-sampling loss).
        """
        batch_size = tf.shape(k)[0]
        clip_norm = self.config.gradient_clip_norm
        n_mc = self.config.mc_sample_number_bond_priceing

        # Sample all shocks once (shared across all updates within this step)
        eps1 = self.shock_dist.sample(sample_shape=(batch_size, 1))
        eps2 = self.shock_dist.sample(sample_shape=(batch_size, 1))
        eps_bond = self.shock_dist.sample(
            sample_shape=(batch_size, n_mc)
        )

        inputs = self._prepare_inputs(k, b, z)
        k_prime_no_invest = self.one_minus_delta * k

        # z_prime for Bellman shocks (shared across all updates)
        z_prime_1 = TransitionFunctions.log_ar1_transition(
            z, self.params.productivity_persistence, eps1
        )
        z_prime_2 = TransitionFunctions.log_ar1_transition(
            z, self.params.productivity_persistence, eps2
        )

        # Pre-compute MC z-grid once (reused by all bond-pricing calls)
        z_curr_bc = tf.broadcast_to(z, (batch_size, n_mc))
        z_prime_bond = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps_bond
        )
        z_flat_bond = tf.reshape(z_prime_bond, (batch_size * n_mc, 1))

        # Compute revenue once (shared across all dividend computations)
        revenue = self.one_minus_tax * ProductionFunctions.cobb_douglas(k, z, self.params)

        # Pre-normalize shared z components
        z1_norm = self.normalizer.normalize_productivity(z_prime_1)
        z2_norm = self.normalizer.normalize_productivity(z_prime_2)
        k_noinv_norm = self.normalizer.normalize_capital(k_prime_no_invest)

        # ==================================================================
        # SINGLE FORWARD PASS: policy nets inside tapes, stop_gradient for labels
        # ==================================================================

        # ========== Tape 1: Invest Policy Branch ==========
        with tf.GradientTape() as invest_policy_tape:
            invest_policy_raw = self.invest_policy_net(inputs, training=True)
            k_prime = self.normalizer.denormalize_capital(
                tf.math.sigmoid(invest_policy_raw[:, 0:1])
            )
            b_prime_invest = self.normalizer.denormalize_debt(
                tf.math.sigmoid(invest_policy_raw[:, 1:2])
            )

            # Equity issuance from 3rd output of invest policy net
            equity_raw_invest = invest_policy_raw[:, 2:3]
            _gate_inv = tf.math.sigmoid(self._eq_gate_mul * equity_raw_invest)
            equity_prob_invest = _gate_inv + tf.stop_gradient(
                tf.cast(_gate_inv > 0.5, TENSORFLOW_DTYPE) - _gate_inv
            )
            equity_issuance_invest = tf.math.softplus(self._eq_sp_mul * equity_raw_invest) / self._eq_sp_div

            # Bond price: sample-based with online continuous net + STE
            q_invest_live = self._estimate_bond_price_from_grid(
                k_prime, b_prime_invest, z_flat_bond, None,
                batch_size, n_mc, use_target=False,
            )
            dividend_invest, payout_invest, _ = self._compute_dividend(
                k, b, z, k_prime, b_prime_invest, q_invest_live,
                equity_issuance_invest, equity_prob_invest,
                revenue=revenue,
            )

            # Batch 2 continuous_net calls → 1 inside this tape
            k_prime_live_norm = self.normalizer.normalize_capital(k_prime)
            b_inv_live_norm = self.normalizer.normalize_debt(b_prime_invest)
            inp_inv_12 = tf.concat([
                tf.concat([k_prime_live_norm, b_inv_live_norm, z1_norm], axis=1),
                tf.concat([k_prime_live_norm, b_inv_live_norm, z2_norm], axis=1),
            ], axis=0)
            v_cont_inv_12 = self.continuous_net(inp_inv_12, training=False)
            v_cont_inv_1, v_cont_inv_2 = tf.split(v_cont_inv_12, 2, axis=0)
            # V = V_cont * (1 - default_prob) from branch selection net
            branch_inv_12 = self.branch_selection_net(inp_inv_12, training=False)
            branch_inv_1, branch_inv_2 = tf.split(branch_inv_12, 2, axis=0)
            d_soft_inv_1 = branch_inv_1[:, 2:3]
            d_hard_inv_1 = tf.cast(d_soft_inv_1 > 0.5, TENSORFLOW_DTYPE)
            default_prob_inv_1 = d_soft_inv_1 + tf.stop_gradient(d_hard_inv_1 - d_soft_inv_1)
            d_soft_inv_2 = branch_inv_2[:, 2:3]
            d_hard_inv_2 = tf.cast(d_soft_inv_2 > 0.5, TENSORFLOW_DTYPE)
            default_prob_inv_2 = d_soft_inv_2 + tf.stop_gradient(d_hard_inv_2 - d_soft_inv_2)
            v_inv_1 = v_cont_inv_1 * (1.0 - default_prob_inv_1)
            v_inv_2 = v_cont_inv_2 * (1.0 - default_prob_inv_2)

            rhs_invest_1 = dividend_invest + self.beta * v_inv_1
            rhs_invest_2 = dividend_invest + self.beta * v_inv_2
            avg_rhs_invest = (rhs_invest_1 + rhs_invest_2) / self.two
            value_obj_invest_loss = -tf.reduce_mean(avg_rhs_invest)

            fb_invest = FischerBurmeisterLoss.fb_function(
                equity_issuance_invest,
                payout_invest + equity_issuance_invest,
            )
            fb_invest_loss = tf.reduce_mean(fb_invest ** 2)
            invest_policy_loss = value_obj_invest_loss + fb_invest_loss * 50

        invest_policy_grads = invest_policy_tape.gradient(
            invest_policy_loss,
            self._invest_policy_variables,
        )

        # ========== Tape 2: Wait Policy Branch ==========
        with tf.GradientTape() as wait_policy_tape:
            wait_policy_raw = self.wait_policy_net(inputs, training=True)
            b_prime_noinvest = self.normalizer.denormalize_debt(
                tf.math.sigmoid(wait_policy_raw[:, 0:1])
            )

            # Equity issuance from 2nd output of wait policy net
            equity_raw_noinvest = wait_policy_raw[:, 1:2]
            _gate_noinv = tf.math.sigmoid(self._eq_gate_mul * equity_raw_noinvest)
            equity_prob_noinvest = _gate_noinv + tf.stop_gradient(
                tf.cast(_gate_noinv > 0.5, TENSORFLOW_DTYPE) - _gate_noinv
            )
            equity_issuance_noinvest = tf.math.softplus(self._eq_sp_mul * equity_raw_noinvest) / self._eq_sp_div

            # Bond price: sample-based with online continuous net + STE
            q_no_invest_live = self._estimate_bond_price_from_grid(
                k_prime_no_invest, b_prime_noinvest, z_flat_bond, None,
                batch_size, n_mc, use_target=False,
            )
            dividend_no_invest, payout_no_invest, _ = self._compute_dividend_no_invest(
                k, b, z, b_prime_noinvest, q_no_invest_live,
                equity_issuance_noinvest, equity_prob_noinvest,
                revenue=revenue,
            )

            # Batch 2 continuous_net calls → 1 inside this tape
            b_noinv_live_norm = self.normalizer.normalize_debt(b_prime_noinvest)
            inp_wait_12 = tf.concat([
                tf.concat([k_noinv_norm, b_noinv_live_norm, z1_norm], axis=1),
                tf.concat([k_noinv_norm, b_noinv_live_norm, z2_norm], axis=1),
            ], axis=0)
            v_cont_noinv_12 = self.continuous_net(inp_wait_12, training=False)
            v_cont_noinv_1, v_cont_noinv_2 = tf.split(v_cont_noinv_12, 2, axis=0)
            # V = V_cont * (1 - default_prob) from branch selection net
            branch_noinv_12 = self.branch_selection_net(inp_wait_12, training=False)
            branch_noinv_1, branch_noinv_2 = tf.split(branch_noinv_12, 2, axis=0)
            d_soft_noinv_1 = branch_noinv_1[:, 2:3]
            d_hard_noinv_1 = tf.cast(d_soft_noinv_1 > 0.5, TENSORFLOW_DTYPE)
            default_prob_noinv_1 = d_soft_noinv_1 + tf.stop_gradient(d_hard_noinv_1 - d_soft_noinv_1)
            d_soft_noinv_2 = branch_noinv_2[:, 2:3]
            d_hard_noinv_2 = tf.cast(d_soft_noinv_2 > 0.5, TENSORFLOW_DTYPE)
            default_prob_noinv_2 = d_soft_noinv_2 + tf.stop_gradient(d_hard_noinv_2 - d_soft_noinv_2)
            v_noinv_1 = v_cont_noinv_1 * (1.0 - default_prob_noinv_1)
            v_noinv_2 = v_cont_noinv_2 * (1.0 - default_prob_noinv_2)

            rhs_noinv_1_live = dividend_no_invest + self.beta * v_noinv_1
            rhs_noinv_2_live = dividend_no_invest + self.beta * v_noinv_2
            avg_rhs_noinvest = (rhs_noinv_1_live + rhs_noinv_2_live) / self.two
            value_obj_wait_loss = -tf.reduce_mean(avg_rhs_noinvest)

            fb_wait = FischerBurmeisterLoss.fb_function(
                equity_issuance_noinvest,
                payout_no_invest + equity_issuance_noinvest,
            )
            fb_wait_loss = tf.reduce_mean(fb_wait ** 2)
            wait_policy_loss = value_obj_wait_loss + fb_wait_loss * 50

        wait_policy_grads = wait_policy_tape.gradient(
            wait_policy_loss,
            self._wait_policy_variables,
        )

        # Shared clip and optimizer step for all policy variables
        policy_grads = invest_policy_grads + wait_policy_grads
        policy_grads = self._clip_gradients(policy_grads, clip_norm)

        # ========== Label computation (stop_gradient from tape outputs) ==========
        # Use tf.stop_gradient on policy outputs to build target labels
        # without needing a separate pre-compute forward pass.
        k_prime_sg = tf.stop_gradient(k_prime)
        b_prime_inv_sg = tf.stop_gradient(b_prime_invest)
        b_prime_noinv_sg = tf.stop_gradient(b_prime_noinvest)
        equity_issuance_inv_sg = tf.stop_gradient(equity_issuance_invest)
        equity_prob_inv_sg = tf.stop_gradient(equity_prob_invest)
        equity_issuance_noinv_sg = tf.stop_gradient(equity_issuance_noinvest)
        equity_prob_noinv_sg = tf.stop_gradient(equity_prob_noinvest)

        # Bond prices for labels (target net, reuse MC z-grid)
        q_invest_tgt = self._estimate_bond_price_from_grid(
            k_prime_sg, b_prime_inv_sg, z_flat_bond, None,
            batch_size, n_mc, use_target=True,
        )
        q_no_invest_tgt = self._estimate_bond_price_from_grid(
            k_prime_no_invest, b_prime_noinv_sg, z_flat_bond, None,
            batch_size, n_mc, use_target=True,
        )

        # Target V: batch 5 evals into 1 (4 next-state + 1 current for default)
        k_inv_norm_sg = self.normalizer.normalize_capital(k_prime_sg)
        b_inv_norm_sg = self.normalizer.normalize_debt(b_prime_inv_sg)
        b_noinv_norm_sg = self.normalizer.normalize_debt(b_prime_noinv_sg)

        inp_inv_1_tgt = tf.concat([k_inv_norm_sg, b_inv_norm_sg, z1_norm], axis=1)
        inp_inv_2_tgt = tf.concat([k_inv_norm_sg, b_inv_norm_sg, z2_norm], axis=1)
        inp_noinv_1_tgt = tf.concat([k_noinv_norm, b_noinv_norm_sg, z1_norm], axis=1)
        inp_noinv_2_tgt = tf.concat([k_noinv_norm, b_noinv_norm_sg, z2_norm], axis=1)

        inp_all_4_tgt = tf.concat([
            inp_inv_1_tgt, inp_inv_2_tgt, inp_noinv_1_tgt, inp_noinv_2_tgt,
        ], axis=0)
        v_cont_all_4_tgt = self.target_continuous_net(inp_all_4_tgt, training=False)
        (v_cont_inv_1_tgt, v_cont_inv_2_tgt,
         v_cont_noinv_1_tgt, v_cont_noinv_2_tgt,
         ) = tf.split(v_cont_all_4_tgt, 4, axis=0)

        # # Target V: hard default from V_cont <= 0 (same as max(0, V_cont))
        # v_inv_1_tgt = tf.maximum(v_cont_inv_1_tgt, 0.0)
        # v_inv_2_tgt = tf.maximum(v_cont_inv_2_tgt, 0.0)
        # v_noinv_1_tgt = tf.maximum(v_cont_noinv_1_tgt, 0.0)
        # v_noinv_2_tgt = tf.maximum(v_cont_noinv_2_tgt, 0.0)

        # Target V: V = V_cont * (1 - default_prob) from branch selection net
        branch_all_4_tgt = self.branch_selection_net(inp_all_4_tgt, training=False)
        (br_inv_1_tgt, br_inv_2_tgt,
         br_noinv_1_tgt, br_noinv_2_tgt,
         ) = tf.split(branch_all_4_tgt, 4, axis=0)
        # v_inv_1_tgt = v_cont_inv_1_tgt * (1.0 - br_inv_1_tgt[:, 2:3])
        # v_inv_2_tgt = v_cont_inv_2_tgt * (1.0 - br_inv_2_tgt[:, 2:3])
        # v_noinv_1_tgt = v_cont_noinv_1_tgt * (1.0 - br_noinv_1_tgt[:, 2:3])
        # v_noinv_2_tgt = v_cont_noinv_2_tgt * (1.0 - br_noinv_2_tgt[:, 2:3])
        v_inv_1_tgt = tf.maximum(v_cont_inv_1_tgt, 0.0)
        v_inv_2_tgt = tf.maximum(v_cont_inv_2_tgt, 0.0)
        v_noinv_1_tgt = tf.maximum(v_cont_noinv_1_tgt, 0.0)
        v_noinv_2_tgt = tf.maximum(v_cont_noinv_2_tgt, 0.0)

        # Dividends for label computation
        div_inv_tgt, _, _ = self._compute_dividend(
            k, b, z, k_prime_sg, b_prime_inv_sg, q_invest_tgt,
            equity_issuance_inv_sg, equity_prob_inv_sg,
            revenue=revenue,
        )
        div_noinv_tgt, _, _ = self._compute_dividend_no_invest(
            k, b, z, b_prime_noinv_sg, q_no_invest_tgt,
            equity_issuance_noinv_sg, equity_prob_noinv_sg,
            revenue=revenue,
        )

        # RHS targets for 2 operating paths
        rhs_inv_1 = div_inv_tgt + self.beta * v_inv_1_tgt
        rhs_inv_2 = div_inv_tgt + self.beta * v_inv_2_tgt
        rhs_noinv_1 = div_noinv_tgt + self.beta * v_noinv_1_tgt
        rhs_noinv_2 = div_noinv_tgt + self.beta * v_noinv_2_tgt

        # Branch labels: 3-way (invest, wait, default)
        # Default derived from target continuous net: V_cont(s) <= 0 means default.
        investment_advantage_1 = rhs_inv_1 - rhs_noinv_1
        investment_advantage_2 = rhs_inv_2 - rhs_noinv_2
        avg_rhs_inv = (rhs_inv_1 + rhs_inv_2) / self.two
        avg_rhs_noinv = (rhs_noinv_1 + rhs_noinv_2) / self.two

        # Default label from target continuous net at current state
        v_cont_current_tgt = self.target_continuous_net(inputs, training=False)
        is_default = tf.cast(v_cont_current_tgt <= 0.0, TENSORFLOW_DTYPE)
        avg_default_rate = tf.reduce_mean(is_default)

        # For non-default states, pick best operating branch; default states get label=2
        operating_rhs = tf.concat([avg_rhs_inv, avg_rhs_noinv], axis=1)
        best_operating_idx = tf.argmax(operating_rhs, axis=1, output_type=tf.int32)
        # Default overrides: if V_cont(s) <= 0, label = 2 (default)
        is_default_int = tf.cast(tf.squeeze(is_default, axis=1), tf.int32)
        branch_label_idx = tf.stop_gradient(
            tf.where(tf.equal(is_default_int, 1), 2, best_operating_idx)
        )
        branch_label = tf.stop_gradient(
            tf.one_hot(branch_label_idx, depth=3, dtype=TENSORFLOW_DTYPE)
        )
        invest_selected = tf.equal(branch_label_idx, 0)

        # Continuous-net RHS targets (0 for default states)
        invest_selected_expanded = tf.expand_dims(invest_selected, axis=1)
        default_selected = tf.equal(branch_label_idx, 2)
        default_selected_expanded = tf.expand_dims(default_selected, axis=1)
        operating_rhs_1 = tf.where(invest_selected_expanded, rhs_inv_1, rhs_noinv_1)
        operating_rhs_2 = tf.where(invest_selected_expanded, rhs_inv_2, rhs_noinv_2)
        # rhs_cont_1 = tf.stop_gradient(
        #     tf.where(default_selected_expanded, tf.zeros_like(operating_rhs_1), operating_rhs_1)
        # )
        # rhs_cont_2 = tf.stop_gradient(
        #     tf.where(default_selected_expanded, tf.zeros_like(operating_rhs_2), operating_rhs_2)
        # )
        rhs_cont_1 = tf.stop_gradient(operating_rhs_1)
        rhs_cont_2 = tf.stop_gradient(operating_rhs_2)

        # ========== Tape 3: Branch Selection ==========
        with tf.GradientTape() as branch_selection_tape:
            branch_probs = self.branch_selection_net(inputs, training=True)
            branch_bce_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    branch_label, branch_probs, from_logits=False
                )
            )

        branch_selection_grads = branch_selection_tape.gradient(
            branch_bce_loss, self._branch_selection_variables
        )
        branch_selection_grads = self._clip_gradients(
            branch_selection_grads, clip_norm
        )

        # ========== Tape 4: Continuous Critic ==========
        with tf.GradientTape() as continuous_tape:
            v_cont = self.continuous_net(inputs, training=True)
            per_sample_loss = (v_cont - rhs_cont_1) * (v_cont - rhs_cont_2)
            continuous_loss = tf.reduce_mean(per_sample_loss)

        continuous_grads = continuous_tape.gradient(
            continuous_loss, self._continuous_variables
        )
        continuous_grads = self._clip_gradients(continuous_grads, clip_norm)

        # V = V_cont * (1 - default_prob) from branch selection net
        branch_probs_for_v = tf.stop_gradient(branch_probs)
        default_prob_for_v = branch_probs_for_v[:, 2:3]
        v_value = v_cont * (1.0 - default_prob_for_v)

        # ==================================================================
        # SIMULTANEOUS GRADIENT APPLICATION
        # ==================================================================
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self._policy_variables)
        )
        self.branch_selection_optimizer.apply_gradients(
            zip(branch_selection_grads, self._branch_selection_variables)
        )
        self.continuous_optimizer.apply_gradients(
            zip(continuous_grads, self._continuous_variables)
        )

        # ========== Logging metrics (slim) ==========
        avg_rhs_target = (rhs_cont_1 + rhs_cont_2) / self.two
        value_obj_loss = (value_obj_invest_loss + value_obj_wait_loss) / self.two

        predicted_branch_idx = tf.argmax(
            tf.stop_gradient(branch_probs), axis=1, output_type=tf.int32
        )
        branch_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted_branch_idx, branch_label_idx), TENSORFLOW_DTYPE)
        )
        total_policy_loss = value_obj_loss + branch_bce_loss + (fb_invest_loss + fb_wait_loss) / self.two

        # Negative payout diagnostics
        frac_neg_inv = tf.reduce_mean(tf.cast(payout_invest < 0.0, TENSORFLOW_DTYPE))
        frac_neg_noinv = tf.reduce_mean(tf.cast(payout_no_invest < 0.0, TENSORFLOW_DTYPE))

        # V<0 fraction (frac_v_lt_vcont removed — always 0 by construction)
        frac_v_negative = tf.reduce_mean(
            tf.cast(v_value < 0.0, TENSORFLOW_DTYPE)
        )

        invest_prob_val = tf.stop_gradient(branch_probs[:, 0:1])
        wait_prob_val = tf.stop_gradient(branch_probs[:, 1:2])
        default_prob_val = tf.stop_gradient(branch_probs[:, 2:3])

        return {
            "policy_loss": total_policy_loss,
            "value_obj_loss": value_obj_loss,
            "value_obj_invest_loss": value_obj_invest_loss,
            "value_obj_wait_loss": value_obj_wait_loss,
            "branch_bce_loss": branch_bce_loss,
            "branch_accuracy": branch_accuracy,
            "optimal_default_rate": avg_default_rate,
            "avg_wait_prob": tf.reduce_mean(wait_prob_val),
            "avg_default_prob": tf.reduce_mean(default_prob_val),
            "avg_rhs": tf.reduce_mean(avg_rhs_target),
            "avg_k_prime": tf.reduce_mean(k_prime),
            "avg_b_prime_invest": tf.reduce_mean(b_prime_invest),
            "avg_b_prime_noinvest": tf.reduce_mean(b_prime_noinvest),
            "avg_invest_prob": tf.reduce_mean(invest_prob_val),
            "avg_invest_STE": tf.reduce_mean(
                tf.cast(tf.equal(predicted_branch_idx, 0), TENSORFLOW_DTYPE)
            ),
            "avg_bond_price": tf.reduce_mean(q_invest_tgt),
            "avg_dividend": tf.reduce_mean(div_inv_tgt),
            "continuous_loss": continuous_loss,
            "avg_v_cont": tf.reduce_mean(v_cont),
            "avg_v_value": tf.reduce_mean(v_value),
            "invest_advantage_mean": tf.reduce_mean(
                (investment_advantage_1 + investment_advantage_2) / self.two
            ),
            "frac_neg_payout_inv": frac_neg_inv,
            "frac_neg_payout_noinv": frac_neg_noinv,
            "frac_v_negative": frac_v_negative,
            "fb_invest_loss": fb_invest_loss,
            "fb_wait_loss": fb_wait_loss,
            "avg_equity_issuance_invest": tf.reduce_mean(equity_issuance_invest),
            "avg_equity_issuance_noinvest": tf.reduce_mean(equity_issuance_noinvest),
            "avg_equity_prob_invest": tf.reduce_mean(equity_prob_invest),
            "avg_equity_prob_noinvest": tf.reduce_mean(equity_prob_noinvest),
            "avg_payout_invest": tf.reduce_mean(payout_invest),
            "avg_payout_noinvest": tf.reduce_mean(payout_no_invest),
        }

    def train_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Public training-step interface; delegates to XLA-compiled impl."""
        return self._optimized_train_step(k, b, z)

    def _run_epoch(self, data_iter, current_decay: float) -> Dict[str, float]:
        """Run one training epoch with periodic target-network updates."""
        epoch_logs = {}
        target_update_freq = self.config.target_update_freq
        decay_tensor = tf.constant(current_decay, dtype=TENSORFLOW_DTYPE)

        for step in range(self.config.steps_per_epoch):
            k, b, z = next(data_iter)
            logs = self.train_step(k, b, z)

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            # Reduced frequency target network updates
            if step % target_update_freq == 0:
                self._soft_update_targets(decay_tensor)

        return epoch_logs

    def train(self) -> None:
        """Run the full training loop with logging and checkpointing."""
        ps = self._polyak_schedule
        print(f"Starting Risky Projection Model Training for {self.config.epochs} epochs...")
        print(f"  |-- Pretrained from: {self.pretrained_checkpoint_dir} "
              f"(epoch {self.pretrained_epoch})")
        print(f"\n=== Optimization Settings (SIMULTANEOUS updates) ===")
        print(f"Batch size: {self.config.batch_size}")
        print(f"LR: critic={self.config.learning_rate:.2e}, "
              f"policy={self.config.learning_rate * self._lr_policy_scale:.2e}")
        print(f"Gradient clip norm: {self.config.gradient_clip_norm}")
        print(f"Polyak decay: {ps.w_start} -> {ps.w_end} over {ps.epoch_end} epochs")
        print(f"MC bond pricing samples: {self.config.mc_sample_number_bond_priceing}")
        for label, sched in self._anneal_schedules().items():
            s = sched[0]
            print(f"{label}: {s.w_start} -> {s.w_end} over epochs [{s.epoch_start}, {s.epoch_end}]")
        print(f"XLA={self.optimization_config.use_xla} | "
              f"Target update freq: every {self.config.target_update_freq} steps")
        print("=" * 40 + "\n")

        dataset = self._build_dataset()
        data_iter = iter(dataset)

        # XLA warm-up
        print("Warming up XLA compilation (this may take a minute)...")
        k_warmup, b_warmup, z_warmup = next(data_iter)
        _ = self.train_step(k_warmup, b_warmup, z_warmup)
        print("Warm-up complete.\n")

        import time
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.epoch_var.assign(epoch)

            # Anneal Polyak decay
            current_decay = self._linear_anneal(self._polyak_schedule, epoch)

            # Anneal all scheduled weights and collect for epoch_logs
            anneal_logs: Dict[str, float] = {"polyak_decay": current_decay}
            for key, (sched, var) in self._anneal_schedules().items():
                val = self._linear_anneal(sched, epoch)
                var.assign(val)
                anneal_logs[key] = val

            epoch_logs = self._run_epoch(data_iter, current_decay)
            epoch_logs.update(anneal_logs)
            epoch_time = time.time() - epoch_start_time

            self._log_tensorboard(epoch, epoch_logs)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs, epoch_time)

            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                save_checkpoint_risky_projection(
                    invest_policy_net=self.invest_policy_net,
                    debt_net_noinvest=self.wait_policy_net,
                    investment_decision_net=self.branch_selection_net,
                    continuous_net=self.continuous_net,
                    epoch=epoch,
                )

        self.summary_writer.flush()

    def _anneal_schedules(self) -> Dict[str, Tuple["_AnnealSchedule", tf.Variable]]:
        """Return all epoch-annealed (schedule, variable) pairs."""
        return {}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _get_current_lr(self) -> float:
        """Return the current critic learning rate."""
        lr = self.continuous_optimizer.learning_rate
        if callable(lr):
            return float(lr(self.continuous_optimizer.iterations))
        return float(lr)

    def _get_current_policy_lr(self) -> float:
        """Return the current policy learning rate."""
        lr = self.policy_optimizer.learning_rate
        if callable(lr):
            return float(lr(self.policy_optimizer.iterations))
        return float(lr)

    def _log_tensorboard(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
    ) -> None:
        """Write per-epoch training metrics to TensorBoard."""
        steps = self.config.steps_per_epoch
        step = epoch

        with self.summary_writer.as_default(step=step):
            # ---- Losses (per-step averages) ----
            tf.summary.scalar("loss/policy", epoch_logs.get("policy_loss", 0) / steps)
            tf.summary.scalar("loss/value_objective", epoch_logs.get("value_obj_loss", 0) / steps)
            tf.summary.scalar("loss/value_obj_invest", epoch_logs.get("value_obj_invest_loss", 0) / steps)
            tf.summary.scalar("loss/value_obj_wait", epoch_logs.get("value_obj_wait_loss", 0) / steps)
            tf.summary.scalar("loss/branch_bce", epoch_logs.get("branch_bce_loss", 0) / steps)
            tf.summary.scalar("loss/continuous", epoch_logs.get("continuous_loss", 0) / steps)
            tf.summary.scalar("loss/fb_invest", epoch_logs.get("fb_invest_loss", 0) / steps)
            tf.summary.scalar("loss/fb_wait", epoch_logs.get("fb_wait_loss", 0) / steps)

            # ---- Policy diagnostics ----
            tf.summary.scalar("policy/avg_k_prime", epoch_logs.get("avg_k_prime", 0) / steps)
            tf.summary.scalar("policy/avg_b_prime_invest", epoch_logs.get("avg_b_prime_invest", 0) / steps)
            tf.summary.scalar("policy/avg_b_prime_noinvest", epoch_logs.get("avg_b_prime_noinvest", 0) / steps)
            tf.summary.scalar("policy/avg_bond_price", epoch_logs.get("avg_bond_price", 0) / steps)
            tf.summary.scalar("policy/avg_dividend", epoch_logs.get("avg_dividend", 0) / steps)

            # ---- Decision diagnostics ----
            tf.summary.scalar("decision/avg_invest_prob", epoch_logs.get("avg_invest_prob", 0) / steps)
            tf.summary.scalar("decision/avg_wait_prob", epoch_logs.get("avg_wait_prob", 0) / steps)
            tf.summary.scalar("decision/avg_default_prob", epoch_logs.get("avg_default_prob", 0) / steps)
            tf.summary.scalar("decision/avg_invest_STE", epoch_logs.get("avg_invest_STE", 0) / steps)

            tf.summary.scalar("decision/invest_advantage_mean", epoch_logs.get("invest_advantage_mean", 0) / steps)
            tf.summary.scalar("decision/vcont_default_rate", epoch_logs.get("optimal_default_rate", 0) / steps)
            tf.summary.scalar("decision/branch_accuracy", epoch_logs.get("branch_accuracy", 0) / steps)

            # ---- Value function diagnostics ----
            tf.summary.scalar("value/avg_v_value", epoch_logs.get("avg_v_value", 0) / steps)
            tf.summary.scalar("value/avg_v_cont", epoch_logs.get("avg_v_cont", 0) / steps)
            tf.summary.scalar("value/avg_rhs", epoch_logs.get("avg_rhs", 0) / steps)
            tf.summary.scalar("value/frac_v_negative", epoch_logs.get("frac_v_negative", 0) / steps)

            # ---- Negative payout diagnostics ----
            tf.summary.scalar("payout/frac_neg_payout_inv", epoch_logs.get("frac_neg_payout_inv", 0) / steps)
            tf.summary.scalar("payout/frac_neg_payout_noinv", epoch_logs.get("frac_neg_payout_noinv", 0) / steps)

            # ---- Equity issuance diagnostics ----
            tf.summary.scalar("equity/avg_issuance_invest", epoch_logs.get("avg_equity_issuance_invest", 0) / steps)
            tf.summary.scalar("equity/avg_issuance_noinvest", epoch_logs.get("avg_equity_issuance_noinvest", 0) / steps)
            tf.summary.scalar("equity/avg_prob_invest", epoch_logs.get("avg_equity_prob_invest", 0) / steps)
            tf.summary.scalar("equity/avg_prob_noinvest", epoch_logs.get("avg_equity_prob_noinvest", 0) / steps)
            tf.summary.scalar("equity/avg_payout_invest", epoch_logs.get("avg_payout_invest", 0) / steps)
            tf.summary.scalar("equity/avg_payout_noinvest", epoch_logs.get("avg_payout_noinvest", 0) / steps)

            # ---- Optimizer ----
            tf.summary.scalar("optimizer/learning_rate", self._get_current_lr())
            tf.summary.scalar("optimizer/policy_learning_rate", self._get_current_policy_lr())
            tf.summary.scalar("optimizer/polyak_decay", epoch_logs.get("polyak_decay", 0))

            # ---- Weight norms (detect explosion / collapse) ----
            tf.summary.scalar("weights/continuous_net_norm",
                tf.linalg.global_norm(self.continuous_net.trainable_variables))
            tf.summary.scalar("weights/invest_policy_norm",
                tf.linalg.global_norm(self.invest_policy_net.trainable_variables))
            tf.summary.scalar("weights/wait_policy_norm",
                tf.linalg.global_norm(self.wait_policy_net.trainable_variables))
            tf.summary.scalar("weights/branch_selection_norm",
                tf.linalg.global_norm(self.branch_selection_net.trainable_variables))

    def _log_epoch(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
        epoch_time: float = None,
    ) -> None:
        """Print a formatted summary line for the given epoch."""
        steps = self.config.steps_per_epoch
        progress_val = float(self.training_progress.numpy())

        policy_loss = epoch_logs.get('policy_loss', 0) / steps
        branch_bce = epoch_logs.get('branch_bce_loss', 0) / steps
        continuous_loss = epoch_logs.get('continuous_loss', 0) / steps
        vcont_def_rate = epoch_logs.get('optimal_default_rate', 0) / steps
        branch_accuracy = epoch_logs.get('branch_accuracy', 0) / steps
        avg_v = epoch_logs.get('avg_v_value', 0) / steps
        avg_q = epoch_logs.get('avg_bond_price', 0) / steps
        avg_k_prime = epoch_logs.get('avg_k_prime', 0) / steps
        avg_b_prime_invest = epoch_logs.get('avg_b_prime_invest', 0) / steps
        avg_b_prime_noinvest = epoch_logs.get('avg_b_prime_noinvest', 0) / steps
        avg_inv_prob = epoch_logs.get('avg_invest_prob', 0) / steps
        avg_inv_STE = epoch_logs.get('avg_invest_STE', 0) / steps
        invest_adv = epoch_logs.get('invest_advantage_mean', 0) / steps
        frac_neg_inv = epoch_logs.get('frac_neg_payout_inv', 0) / steps
        frac_neg_noinv = epoch_logs.get('frac_neg_payout_noinv', 0) / steps
        fb_inv = epoch_logs.get('fb_invest_loss', 0) / steps
        fb_wait = epoch_logs.get('fb_wait_loss', 0) / steps
        avg_eq_inv = epoch_logs.get('avg_equity_issuance_invest', 0) / steps
        avg_eq_noinv = epoch_logs.get('avg_equity_issuance_noinvest', 0) / steps
        avg_eq_prob_inv = epoch_logs.get('avg_equity_prob_invest', 0) / steps
        avg_eq_prob_noinv = epoch_logs.get('avg_equity_prob_noinvest', 0) / steps
        avg_default_prob = epoch_logs.get('avg_default_prob', 0) / steps

        time_str = f" | Time: {epoch_time:.2f}s" if epoch_time else ""
        current_lr = self._get_current_lr()
        current_policy_lr = self._get_current_policy_lr()
        polyak_decay = epoch_logs.get('polyak_decay', None)
        polyak_str = f" | Polyak: {polyak_decay:.6f}" if polyak_decay is not None else ""

        print(
            f"Epoch {epoch:4d} | "
            f"Prog: {progress_val:.3f} | "
            f"Pol: {policy_loss:.4e} | "
            f"BranchBCE: {branch_bce:.4e} | "
            f"ContL: {continuous_loss:.4e} | "
            f"BranchAcc: {branch_accuracy:.3f} | "
            f"DefRate: {vcont_def_rate:.3f} | "
            f"DefP: {avg_default_prob:.3f} | "
            f"InvP: {avg_inv_prob:.2%} | "
            f"InvSTE: {avg_inv_STE:.2%} | "
            f"V: {avg_v:.2f} | "
            f"Q: {avg_q:.3f} | "
            f"K': {avg_k_prime:.3f} | "
            f"Bi': {avg_b_prime_invest:.3f} | "
            f"Bn': {avg_b_prime_noinvest:.3f} | "
            f"InvAdv: {invest_adv:.4f} | "
            f"NegInv: {frac_neg_inv:.2%} | "
            f"NegNoinv: {frac_neg_noinv:.2%} | "
            f"LR: {current_lr:.2e} | "
            f"pLR: {current_policy_lr:.2e}"
            f"{polyak_str}"
            f"{time_str}"
        )
        frac_v_neg = epoch_logs.get('frac_v_negative', 0) / steps
        print(
            f"         |-- Value Violations | "
            f"V<0: {frac_v_neg:.2%}"
        )
        print(
            f"         |-- Equity Issuance  | "
            f"FB_inv: {fb_inv:.4e} | "
            f"FB_wait: {fb_wait:.4e} | "
            f"EqP_inv: {avg_eq_prob_inv:.3f} | "
            f"EqP_noinv: {avg_eq_prob_noinv:.3f}"
        )
