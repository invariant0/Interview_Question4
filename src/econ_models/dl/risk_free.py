"""
Deep Learning solver for the Risk-Free Debt Model using All-in-One (AiO) approach.

Faithful implementation of:
  'All-in-One (AiO) Loss Formulation for Corporate Model with Debt'

Minimizes the product of residuals from two independent samples for
unbiased gradient estimation. Implements FOC-based losses:
  A. Bellman           (Eq. 2) — additive, target-net continuation
  B. Capital Euler     (Eq. 4) — additive, normalized
  C. Multiplier        (Eq. 8) — additive, normalized
  D. Fischer-Burmeister complementarity (Eq. 9)

Policy networks are parameterized as:
  - Capital policy    →  sigmoid output  ∈ (0,1)  → K' = denormalize_capital(σ(x))
  - Debt policy       →  sigmoid output  ∈ (0,1)  → B' = denormalize_debt(σ(x))
  - Multiplier network →  μ̂(K,B,Z) = softplus(x) ≥ 0  (Lagrange multiplier)

Loss residuals use the *additive* form (unbiased under expectation):
  - Bellman (Eq. 2):
      D^Bell = [V(K,B,Z) − D − β·V(K',B',Z')] / max(|V|, 1)
  - Capital Euler (Eq. 4):
      D^Cap = [λ(e)·(1 + ψ₀·I/K) − β·∂V/∂K' − μ̂·dB̄/dK']
              / max(|λ(e)·(1 + ψ₀·I/K)|, 1)
  - Debt Multiplier (Eq. 8):
      D^Mult = [λ(e)·α_B + β·∂V/∂B' − μ̂] / max(|λ(e)·α_B|, 1)
  - Fischer-Burmeister (Eq. 9):
      [Ψ^FB((B̄−B')/B̄, μ̂/B̄)]²

where:
  λ(e) = 1 + η₁·𝟙{e<0}  is the marginal cost of external funds,
  α_B = q + τrq²          is the marginal cash inflow per unit face-value debt,
  dB̄/dK'                  is the collateral constraint derivative w.r.t. K'.

Additive residuals eliminate Jensen's inequality bias that arises in
ratio-form residuals, ensuring E[f(ε)] = 0 at the true solution and
valid double-sample unbiased gradient estimation.

Semi-gradient Bellman: a target value network (Polyak-averaged)
provides stable continuation values V'(K',B',Z') in the Bellman
residual, with stop_gradient on V'.  FOC losses (Capital Euler,
Debt) still flow full gradients through ∂V/∂K' and ∂V/∂B' via
the online value net for joint AiO learning.

UPDATED: XLA compilation for acceleration. Economic computations
delegate to centralized econ modules (all use standard TF ops that
are fully XLA-compatible). GPU-side sampling eliminates CPU→GPU
data transfer bottleneck.
"""

import datetime
import os
from typing import Tuple, Dict, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.math import FischerBurmeisterLoss
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.io.checkpoints import save_checkpoint_risk_free
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CollateralCalculator,
    IssuanceCostCalculator,
)

tfd = tfp.distributions


class OptimizationConfig(NamedTuple):
    """Configuration for training optimizations."""
    use_xla: bool = False
    use_mixed_precision: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE
    cache_dataset: bool = True


class RiskFreeModelDL:
    """
    Solves the corporate model with risk-free debt using the AiO approach.

    Minimizes the product of residuals from two independent samples
    (Section 2.1) across four loss components:

      Ξ(Θ) = E[ D₁^Bell × D₂^Bell
               + ν_K · D₁^Cap × D₂^Cap
               + ν_μ · D₁^Mult × D₂^Mult
               + ν_FB · [Ψ^FB((B̄−B')/B̄, μ̂/B̄)]² ]

    All FOC residuals use *additive* form (f = LHS − RHS), normalized
    by shock-independent state-dependent scales wrapped in stop_gradient
    to avoid Jensen's inequality bias:

      Bellman  (Eq. 2):   D^Bell = [V − D − β·Vtarg'] / max(|V|, 1)  (target net)
      Capital  (Eq. 4):   D^Cap  = [λ(e)(1+ψ₀I/K) − β∂V'/∂K' − μ̂·dB̄/dK']
                                    / max(|λ(e)(1+ψ₀I/K)|, 1)
      Debt     (Eq. 8):   D^Mult = [λ(e)·α_B + β·∂V'/∂B' − μ̂]
                                    / max(|λ(e)·α_B|, 1)

    where:
      λ(e) = 1 + η₁·𝟙{e<0}  — marginal cost of external funds
      α_B = q + τrq²          — marginal cash per unit face-value debt

    Policy parameterisation:
      - Capital policy    →  sigmoid ∈ (0,1)  → K' = denormalize_capital(σ(x))
      - Debt policy       →  sigmoid ∈ (0,1)  → B' = denormalize_debt(σ(x))
      - Multiplier network →  μ̂(K,B,Z) = softplus(x) ≥ 0

    KKT conditions (via Fischer-Burmeister):
      μ̂ ≥ 0,  B̄ − B' ≥ 0,  μ̂·(B̄ − B') = 0

    Reference: 'All-in-One (AiO) Loss Formulation for Corporate Model with Debt'
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bounds: Dict[str, Tuple[float, float]],
        optimization_config: OptimizationConfig = None,
        log_dir_prefix: str = "logs/risk_free",
    ) -> None:
        self.params = params
        self.config = config
        self.bounds = bounds
        self.log_dir_prefix = log_dir_prefix
        self.optimization_config = optimization_config or OptimizationConfig()

        self.config.update_value_scale(self.params)
        self.normalizer = StateSpaceNormalizer(self.config)
        self.epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

        self._setup_cached_constants()
        self._build_networks()
        self._build_optimizers()
        self._build_shock_distribution()
        self._build_optimized_train_step()
        self._build_summary_writer()

        # Polyak averaging schedule for target network
        self._polyak_start = self.config.polyak_averaging_decay
        self._polyak_end = self.config.polyak_decay_end
        self._polyak_anneal_epochs = self.config.polyak_decay_epochs
        self._target_update_freq = self.config.target_update_freq 

        self.training_progress = tf.Variable(
            1.0, dtype=TENSORFLOW_DTYPE, trainable=False
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _setup_cached_constants(self) -> None:
        """Pre-cast and cache constants that appear in FOC / loss expressions.

        Only constants appearing directly in the Bellman equation (Eq. 1),
        Capital FOC (Eq. 4), Debt FOC (Eq. 8), and AiO loss (Eq. 9)
        are cached here.

        Key derivation for α_B:
          The firm issues face-value B' debt at price q = 1/(1+r),
          receiving cash qB' plus a tax shield τrq²B'.
          The total marginal cash per unit face-value is:
            α_B = q + τrq² = q(1 + τrq)
          This is ∂e/∂B' — the marginal increase in payout from
          issuing one additional unit of face-value debt.
        """
        r = self.params.risk_free_rate
        tau = self.params.corporate_tax_rate

        # Discount factor β
        self.beta = tf.constant(self.params.discount_factor, dtype=TENSORFLOW_DTYPE)

        # Risk-free rate r
        self.risk_free_rate_t = tf.constant(r, dtype=TENSORFLOW_DTYPE)

        # Bond price q = 1/(1+r)
        self.q_rf = tf.constant(1.0 / (1.0 + r), dtype=TENSORFLOW_DTYPE)

        # 1 − δ  (used for capital law of motion)
        self.one_minus_delta = tf.constant(
            1.0 - self.params.depreciation_rate, dtype=TENSORFLOW_DTYPE
        )

        # Capital bounds (for clipping K')
        self.k_min = tf.constant(self.config.capital_min, dtype=TENSORFLOW_DTYPE)
        self.k_max = tf.constant(self.config.capital_max, dtype=TENSORFLOW_DTYPE)

        # Tax-related constants for the dividend (Eq. 1)
        self.one_minus_tau = tf.constant(1.0 - tau, dtype=TENSORFLOW_DTYPE)
        self.tau_t = tf.constant(tau, dtype=TENSORFLOW_DTYPE)

        # -------------------------------------------------------
        # Marginal cash inflow per unit of face-value debt issued:
        #   α_B = q + τrq² = q(1 + τrq)
        #
        # This is ∂e/∂B' — the marginal increase in payout from
        # issuing one additional unit of face-value debt.
        # Appears in the Debt FOC (Eq. 8):
        #   μ̂ = λ(e)·α_B + β·E[∂V/∂B']
        # -------------------------------------------------------
        q = 1.0 / (1.0 + r)
        self.alpha_B = tf.constant(
            q + tau * r * q * q, dtype=TENSORFLOW_DTYPE
        )

        # ----------------------------------------------------------
        # AiO loss weights: ν_K, ν_μ, ν_FB
        #
        # All residuals are additive and normalized to ~O(1).
        # Weights balance gradient signal across the four loss
        # components.
        # ----------------------------------------------------------
        self.nu_K = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
        self.nu_mu = tf.constant(10.0, dtype=TENSORFLOW_DTYPE)
        self.nu_FB = tf.constant(10.0, dtype=TENSORFLOW_DTYPE)

        # Cached economic parameter for collateral derivative
        self.capital_share_t = tf.constant(
            self.params.capital_share, dtype=TENSORFLOW_DTYPE
        )

        # ----------------------------------------------------------
        # Equity issuance cost parameters
        #
        # When payout e < 0 the firm must raise external equity at
        # cost η(e) = η₀ + η₁|e|.  The marginal cost of funds is:
        #   λ(e) = 1 + η₁·𝟙{e<0}
        # This multiplies the marginal benefit/cost terms in both
        # the Capital Euler and Debt FOC.
        # ----------------------------------------------------------
        self.eta_0_t = tf.constant(
            self.params.equity_issuance_cost_fixed, dtype=TENSORFLOW_DTYPE
        )
        self.eta_1_t = tf.constant(
            self.params.equity_issuance_cost_linear, dtype=TENSORFLOW_DTYPE
        )

        # ----------------------------------------------------------
        # Collateral constraint derivative  dB̄/dK'
        #
        # B̄(K') = (1−τ)·z_min·(K')^θ + τδK' + sK'
        # dB̄/dK' = (1−τ)·z_min·θ·(K')^(θ−1) + τδ + s
        #
        # The constant part τδ + s is cached here; the
        # power-law term is computed per-batch in the train step.
        # ----------------------------------------------------------
        self.z_min_t = tf.constant(
            self.config.productivity_min, dtype=TENSORFLOW_DTYPE
        )
        self.tau_delta_t = tf.constant(
            tau * self.params.depreciation_rate, dtype=TENSORFLOW_DTYPE
        )
        self.recovery_fraction_t = tf.constant(
            self.params.collateral_recovery_fraction, dtype=TENSORFLOW_DTYPE
        )

    def _build_networks(self) -> None:
        # 1. Value Network V(K, B, Z)  — Eq. (2)
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueNet",
        )

        # 2. Capital Policy: sigmoid output ∈ (0,1)
        #    K' = denormalize_capital(σ(x)) maps to [k_min, k_max]
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            name="CapitalPolicyNet",
        )

        # 3. Debt Policy: sigmoid output ∈ (0,1)
        #    B' = denormalize_debt(σ(x)) maps to [b_min, b_max]
        self.debt_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            name="DebtPolicyNet",
        )

        # 4. Multiplier Network  μ̂(K,B,Z) ≥ 0
        #    Softplus activation guarantees non-negativity: softplus(x) ≥ 0.
        #    This is the direct Lagrange multiplier on the collateral
        #    constraint B' ≤ B̄(K').  At the optimum:
        #      μ̂ = 0    when constraint is slack  (B' < B̄)
        #      μ̂ > 0    when constraint binds     (B' = B̄)
        #    Trained jointly via AiO loss to satisfy the debt FOC (Eq. 8)
        #    and Fischer-Burmeister complementarity (Eq. 9).
        self.multiplier_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="softplus",
            name="MultiplierNet",
        )

        # 5. Target Value Network Vtarg(K,B,Z; θ⁻)
        #    A slow-moving copy of the value net used to compute
        #    continuation values V'(K',B',Z') in the Bellman
        #    residual.  Weights are updated via Polyak averaging:
        #      θ⁻ ← τ·θ⁻ + (1−τ)·θ
        #    This stabilises learning by decoupling the Bellman
        #    target from the rapidly changing online value net.
        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="TargetValueNet",
        )
        self.target_value_net.set_weights(self.value_net.get_weights())

        # 6. Default Policy (auxiliary — used for downstream risky model)
        self.default_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            name="DefaultPolicyNet",
        )

        # Cache trainable variables for the joint AiO gradient.
        # Includes multiplier_net whose output μ̂ appears in the
        # debt FOC and Fischer-Burmeister losses.
        self._all_foc_variables = (
            self.value_net.trainable_variables
            + self.capital_policy_net.trainable_variables
            + self.debt_policy_net.trainable_variables
            + self.multiplier_net.trainable_variables
        )

    def _build_optimizers(self) -> None:
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.lr_decay_steps,
            decay_rate=self.config.lr_decay_rate,
            staircase=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.default_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule
        )

    def _build_shock_distribution(self) -> None:
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE),
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

    def _build_optimized_train_step(self) -> None:
        """Build the optimized training step function.

        Both training step and target updates use XLA compilation when
        enabled.  All operations inside the training step are standard
        TF ops (dense layers, sampling, reshaping, reductions) which
        are fully XLA-compatible.
        """
        if self.optimization_config.use_xla:
            self._optimized_train_step = tf.function(
                self._train_step_impl, jit_compile=True
            )
            self._soft_update_targets = tf.function(
                self._soft_update_targets_impl, jit_compile=True
            )
        else:
            self._optimized_train_step = tf.function(self._train_step_impl)
            self._soft_update_targets = tf.function(
                self._soft_update_targets_impl
            )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _build_dataset(self) -> tf.data.Dataset:
        dataset = DatasetBuilder.build_dataset(
            self.config,
            self.bounds,
            include_debt=True,
        )
        dataset = dataset.prefetch(self.optimization_config.prefetch_buffer)
        return dataset

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor
    ) -> tf.Tensor:
        return tf.concat(
            [
                self.normalizer.normalize_capital(k),
                self.normalizer.normalize_debt(b),
                self.normalizer.normalize_productivity(z),
            ],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Core training step  (implements Eq. 9 of the blueprint)
    # ------------------------------------------------------------------

    def _train_step_impl(
        self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """XLA-compatible training step — corrected AiO formulation.

        Corrections vs. original implementation:

        1. **Multiplier baseline removed**:
           μ̂ = softplus(x) ≥ 0  (was: 1 + softplus(x) ≥ 1).
           The +1 offset made the debt FOC unsatisfiable when the
           constraint was slack (KKT requires μ = 0 ⟹ ĥ = 1, but
           the ratio form lhs/1 − 1 ≠ 0).  It also injected a
           spurious collateral-relaxation benefit into the capital
           FOC for unconstrained firms.

        2. **Additive residuals** (not ratio form):
           Ratio residuals f(ε) = LHS/RHS(ε) − 1 have E[f] ≠ 0
           at the true solution by Jensen's inequality (1/x is convex),
           biasing the double-sample product estimator.
           Additive residuals f(ε) = LHS − RHS(ε) satisfy E[f] = 0
           exactly, giving unbiased gradient estimates.

        3. **Semi-gradient Bellman (target net)**:
           A Polyak-averaged target value network computes V'
           for the Bellman residual.  stop_gradient on V'
           prevents destabilising feedback loops.  The online
           value net V(K,B,Z) still receives full gradient as
           the LHS of the Bellman equation.  FOC losses provide
           additional gradient flow through ∂V/∂K' and ∂V/∂B'
           via the online net.

        4. **Direct multiplier in capital FOC**:
           μ̂·dB̄/dK' (was: ĥ·q·dB̄/dK').  The q factor was an
           artifact of the rescaled multiplier convention ĥ = (1+r)μ.
           With the direct multiplier, no q is needed.

        5. **STE steepness reduced**: k=10 (was: 25) for broader
           gradient support during early training.

        Structure:
        1. Draw two independent shocks  ε₁, ε₂  (Section 2.1)
        2. Evaluate policies and compute the dividend  D  (Eq. 1)
        3. Form additive residual pairs for each loss component
        4. Normalize by shock-independent scales (stop_gradient)
        5. Combine via the AiO objective  Ξ(Θ)
        """
        batch_size = tf.shape(k)[0]

        # ============================================================
        # Step 1: Double Sampling  (Section 2.1)
        # Two independent shock draws for unbiased gradient estimation
        # ============================================================
        eps_1 = self.shock_dist.sample(sample_shape=(batch_size, 1))
        eps_2 = self.shock_dist.sample(sample_shape=(batch_size, 1))

        with tf.GradientTape() as tape:
            # ========================================================
            # Step 2: Current-Period Policies and Dividend
            # ========================================================

            inputs = self._prepare_inputs(k, b, z)
            v_current = self.value_net(inputs, training=True)

            # --- Policy outputs ---
            # Capital policy: sigmoid ∈ (0,1), soft-squeezed to
            # [ε, 1−ε] via affine rescaling.  Unlike clip_by_value
            # (which has zero gradient at boundaries), this
            # preserves gradient everywhere:
            #   k_norm = ε + (1 − 2ε) · σ(x)
            # Gradient = (1 − 2ε) · σ'(x) > 0  ∀ x.
            k_prime_norm = self.capital_policy_net(inputs, training=True)
            k_prime = self.normalizer.denormalize_capital(k_prime_norm)

            # --- Derived quantities (capital) ---
            # Investment:  I = K' − (1−δ)K
            investment = k_prime - self.one_minus_delta * k
            # Investment rate:  I/K
            i_rate = investment / k

            # --- Collateral limit B̄(K') ---
            # B̄ = collateral fraction × liquidation value of K'
            # This is the upper bound on debt issuance (Eq. 3).
            collateral_limit = CollateralCalculator.calculate_limit(
                k_prime,
                self.config.productivity_min,
                self.params,
                self.params.collateral_recovery_fraction,
            )

            # --- Debt policy ---
            # Debt policy: sigmoid ∈ (0,1), soft-squeezed to [ε, 1−ε]
            b_prime_norm = self.debt_policy_net(inputs, training=True)
            b_prime = self.normalizer.denormalize_debt(b_prime_norm)

            # Leverage ratio for diagnostics: B'/B̄
            leverage_ratio = b_prime / (collateral_limit + 1e-8)

            # --- Multiplier estimate ---
            # μ̂(K,B,Z) = softplus(x) ≥ 0
            #
            # Direct Lagrange multiplier on the collateral constraint
            # B' ≤ B̄(K').  No baseline offset.
            #
            # At the optimum:
            #   Slack (B' < B̄):  μ̂ → 0  (no shadow price)
            #   Binding (B' = B̄): μ̂ > 0  (positive shadow price)
            #
            # The softplus activation guarantees μ̂ ≥ 0 and provides
            # smooth gradients (unlike ReLU which has dead neurons).
            mu_hat = self.multiplier_net(inputs, training=True)

            # --- Dividend  D(K, B, Z, K', B')   (Eq. 1) ---
            #
            #   D = (1−τ)ZK^θ − I − Φ(I,K)
            #       + B'/(1+r)           [bond proceeds]
            #       + τrB'/(1+r)²        [tax shield on new debt]
            #       − B                  [repay maturing debt]
            #       − η(e)·𝟙{e<0}       [issuance cost]

            # After-tax revenue: (1−τ)ZK^θ
            revenue = self.one_minus_tau * ProductionFunctions.cobb_douglas(
                k, z, self.params
            )

            # Adjustment cost  Φ(I,K) = (ψ₀/2)(I²/K)  and
            # marginal cost   Φ_I = ψ₀(I/K)
            adj_cost, phi_I = AdjustmentCostCalculator.calculate(
                investment, k, self.params
            )

            # Bond proceeds:  B'/(1+r) = B' · q
            bond_proceeds = b_prime * self.q_rf

            # Tax shield:  τrB'/(1+r)² = τ · r · B' · q²
            tax_shield = (
                self.tau_t
                * self.risk_free_rate_t
                * b_prime
                * self.q_rf
                * self.q_rf
            )

            # Raw payout  e  (Eq. 1, before issuance costs)
            payout = (
                revenue - investment - adj_cost + bond_proceeds + tax_shield - b
            )

            # --- Equity issuance cost  η(e) ---
            # When payout e < 0, the firm raises external equity at cost
            #   η(e) = (η₀ + η₁|e|) · 𝟙{e < 0}
            # Uses STE (Straight-Through Estimator) for differentiability:
            # forward pass uses exact hard threshold, backward pass uses
            # smooth sigmoid surrogate for gradient flow.
            issuance_cost = IssuanceCostCalculator.calculate_with_grad(
                payout, self.params
            )

            # Dividend  D = e − η(e)   (Eq. 1, after issuance costs)
            dividend = payout - issuance_cost

            # --- Marginal cost of funds  λ(e) ---
            # λ(e) = ∂D/∂e = 1 + η₁·𝟙{e < 0}
            # STE: exact forward indicator, smooth backward gradient.
            # Steepness k=10 (reduced from 25 for broader gradient
            # support during early training when policies are imprecise).
            is_issuing_hard = tf.cast(payout < 0.0, TENSORFLOW_DTYPE)
            k_ste = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
            is_issuing_soft = tf.math.sigmoid(-k_ste * payout)
            is_issuing = is_issuing_soft + tf.stop_gradient(
                is_issuing_hard - is_issuing_soft
            )
            lambda_e = 1.0 + self.eta_1_t * is_issuing

            # --- Collateral constraint derivative  dB̄/dK' ---
            # B̄(K') = (1−τ)·z_min·(K')^θ + τδK' + sK'
            # dB̄/dK' = (1−τ)·z_min·θ·(K')^(θ−1) + τδ + s
            dBbar_dKp = (
                self.one_minus_tau * self.z_min_t * self.capital_share_t
                * tf.pow(k_prime, self.capital_share_t - 1.0)
                + self.tau_delta_t
                + self.recovery_fraction_t
            )

            # ========================================================
            # Step 3: Future States  (Section 2.1)
            # ========================================================
            z_prime_1 = TransitionFunctions.log_ar1_transition(
                z, self.params.productivity_persistence, eps_1
            )
            z_prime_2 = TransitionFunctions.log_ar1_transition(
                z, self.params.productivity_persistence, eps_2
            )

            # ========================================================
            # Step 4: Future Value and Partial Derivatives
            # ========================================================
            inputs_next_1 = self._prepare_inputs(k_prime, b_prime, z_prime_1)
            inputs_next_2 = self._prepare_inputs(k_prime, b_prime, z_prime_2)

            # --- Semi-gradient Bellman via TARGET value net ---
            # The target value net (Polyak-averaged) provides stable
            # continuation values for the Bellman residual.  Using
            # stop_gradient prevents the deadly feedback loop where
            # V and V' co-inflate to reduce the Bellman loss.
            #
            # --- FOC partials via ONLINE value net ---
            # Full gradients flow through ∂V/∂K' and ∂V/∂B' for
            # joint AiO learning of policies and value.
            #
            # _prepare_inputs must be called INSIDE the v_tape
            # context so the tape records the path from
            # (k_prime, b_prime) → normalize → concat → value_net.
            with tf.GradientTape(persistent=True) as v_tape:
                v_tape.watch([k_prime, b_prime])
                inputs_next_1 = self._prepare_inputs(k_prime, b_prime, z_prime_1)
                inputs_next_2 = self._prepare_inputs(k_prime, b_prime, z_prime_2)
                v_next_1 = self.value_net(inputs_next_1, training=True)
                v_next_2 = self.value_net(inputs_next_2, training=True)

            # ∂V/∂K'  and  ∂V/∂B'  for each shock draw (online net)
            dV_dKp_1 = v_tape.gradient(v_next_1, k_prime)
            dV_dBp_1 = v_tape.gradient(v_next_1, b_prime)
            dV_dKp_2 = v_tape.gradient(v_next_2, k_prime)
            dV_dBp_2 = v_tape.gradient(v_next_2, b_prime)
            del v_tape

            # --- Target value net for Bellman continuation ---
            # V_target'(K',B',Z') with stop_gradient: only the
            # current V(K,B,Z) receives Bellman gradient, while
            # the slowly-updated target provides a stable anchor.
            v_next_target_1 = tf.stop_gradient(
                self.target_value_net(inputs_next_1, training=False)
            )
            v_next_target_2 = tf.stop_gradient(
                self.target_value_net(inputs_next_2, training=False)
            )

            # ========================================================
            # Step 5: Loss Components — Additive Residuals
            #
            # All residuals use the ADDITIVE form f = LHS − RHS.
            # This ensures E[f(ε)] = 0 at the true solution,
            # which is required for the double-sample product
            # E[f(ε₁)·f(ε₂)] = E[f]² = 0 to be unbiased.
            #
            # Ratio-form residuals f = LHS/RHS(ε) − 1 violate
            # this: E[1/RHS(ε)] ≠ 1/E[RHS(ε)] by Jensen's
            # inequality, biasing the loss minimum away from the
            # true solution.
            #
            # Each residual is normalized by a shock-INDEPENDENT,
            # state-dependent scale factor (wrapped in stop_gradient)
            # to keep magnitudes ~O(1) without introducing bias.
            # ========================================================

            # --------------------------------------------------
            # A.  Bellman Residuals — Semi-gradient, Additive
            #
            #   f(ε) = V(K,B,Z) − D − β·V_target(K',B',Z'(ε))
            #
            # Semi-gradient: the TARGET value net (Polyak-averaged)
            # provides V' with stop_gradient.  Only the online
            # V(K,B,Z) receives gradient from the Bellman loss.
            # This prevents the deadly V-inflation feedback loop
            # where V and V' co-inflate to trivially reduce the
            # Bellman residual.
            #
            # stop_gradient on DIVIDEND (actor-critic separation):
            # The Bellman loss trains the value net V(K,B,Z)
            # to satisfy the Bellman equation but does NOT push
            # policy gradients through ∂D/∂K' or ∂D/∂B'.
            # All policy gradients (K', B', μ̂) come exclusively
            # from the FOC and FB losses, which encode the
            # economic optimality conditions.
            # --------------------------------------------------
            dividend_sg = tf.stop_gradient(dividend)
            res_bell_1 = v_current - dividend_sg - self.beta * v_next_target_1
            res_bell_2 = v_current - dividend_sg - self.beta * v_next_target_2

            # Normalize by the DIVIDEND scale rather than V.
            #
            # Using max(|V|, 1) is problematic because V grows
            # unboundedly over training (V: 0.1 → 38+ by epoch 90).
            # This progressively suppresses the residual's
            # sensitivity to the B dimension, causing ∂V/∂B to
            # deteriorate from -0.83 (epoch 10) to -0.62 (epoch 90)
            # instead of converging to the theoretical -λ(e) ≈ -1.07.
            #
            # The dividend D is O(1) and shock-independent (it only
            # depends on current-period states and policies), making
            # it a stable normaliser that doesn't grow over training.
            bell_scale = tf.stop_gradient(
                tf.maximum(tf.abs(dividend), 1.0)
            )
            res_bell_1 = res_bell_1 / bell_scale
            res_bell_2 = res_bell_2 / bell_scale

            loss_bellman = tf.reduce_mean(res_bell_1 * res_bell_2)

            # --------------------------------------------------
            # B.  Capital Euler Residuals — Additive  (Eq. 4)
            #
            # From the Lagrangian:
            #   L = D + β·E[V'] + μ·(B̄(K') − B')
            #
            # FOC w.r.t. K':
            #   ∂D/∂K' + β·E[∂V'/∂K'] + μ·dB̄/dK' = 0
            #
            # Since ∂D/∂K' = −λ(e)·(1 + ψ₀I/K):
            #   λ(e)·(1 + ψ₀I/K) = β·E[∂V'/∂K'] + μ·dB̄/dK'
            #
            # Additive residual:
            #   f(ε) = λ(e)·(1+ψ₀I/K) − β·∂V'/∂K'|_ε − μ̂·dB̄/dK'
            #
            # No q factor: μ̂ is the direct Lagrange multiplier
            # on B' ≤ B̄(K'), so the collateral relaxation
            # benefit is μ̂·dB̄/dK' directly.
            # --------------------------------------------------
            lhs_cap = (1.0 + phi_I) * lambda_e  # λ(e)·(1 + ψ₀I/K)
            collateral_benefit = mu_hat * dBbar_dKp  # μ̂·dB̄/dK'

            res_cap_1 = lhs_cap - self.beta * dV_dKp_1 - collateral_benefit
            res_cap_2 = lhs_cap - self.beta * dV_dKp_2 - collateral_benefit

            # Normalize by the deterministic LHS (shock-independent).
            # This makes the residual dimensionless and ~O(1).
            cap_scale = tf.stop_gradient(
                tf.maximum(tf.abs(lhs_cap), 1.0)
            )
            res_cap_1 = res_cap_1 / cap_scale
            res_cap_2 = res_cap_2 / cap_scale

            loss_foc_k = tf.reduce_mean(res_cap_1 * res_cap_2)

            # --------------------------------------------------
            # C.  Debt FOC Residuals — Additive  (Eq. 8)
            #
            # From the Lagrangian:
            #   FOC w.r.t. B':
            #     ∂D/∂B' + β·E[∂V'/∂B'] − μ = 0
            #
            # Since ∂D/∂B' = λ(e)·α_B where α_B = q + τrq²:
            #   μ = λ(e)·α_B + β·E[∂V'/∂B']
            #
            # Additive residual:
            #   f(ε) = λ(e)·α_B + β·∂V'/∂B'|_ε − μ̂
            #
            # At the optimum:
            #   Slack (μ̂=0): λ(e)·α_B + β·E[∂V'/∂B'] = 0
            #     (marginal benefit of debt = marginal cost,
            #      i.e. E[∂V'/∂B'] < 0)
            #   Binding (μ̂>0): μ̂ = λ(e)·α_B + β·E[∂V'/∂B']
            #     (multiplier equals net marginal benefit)
            #
            # This replaces the old ratio form lhs/ĥ − 1 which
            # was unsatisfiable when the constraint was slack
            # (ĥ ≥ 1 forced a nonzero denominator, giving
            # residual = lhs/1 − 1 = −1 ≠ 0 when lhs = 0).
            # --------------------------------------------------
            debt_benefit = lambda_e * self.alpha_B  # λ(e)·α_B

            res_mult_1 = debt_benefit + self.beta * dV_dBp_1 - mu_hat
            res_mult_2 = debt_benefit + self.beta * dV_dBp_2 - mu_hat

            # Normalize by the deterministic benefit term.
            debt_scale = tf.stop_gradient(
                tf.maximum(tf.abs(debt_benefit), 1.0)
            )
            res_mult_1 = res_mult_1 / debt_scale
            res_mult_2 = res_mult_2 / debt_scale

            loss_multiplier = tf.reduce_mean(res_mult_1 * res_mult_2)

            # --------------------------------------------------
            # D.  Fischer-Burmeister Complementarity  (Eq. 9)
            #
            #   Ψ^FB(a, b) = a + b − √(a² + b²) = 0
            #   ⟺  a ≥ 0, b ≥ 0, a·b = 0
            #
            # Enforces the KKT complementary slackness condition:
            #   μ̂ ≥ 0,  (B̄ − B') ≥ 0,  μ̂·(B̄ − B') = 0
            #
            # RAW (unnormalized) FB arguments:
            #   a = B̄ − B'    (constraint slack)
            #   b = μ̂          (multiplier)
            #
            # Normalization by B̄ is REMOVED because when the
            # constraint is very slack (B' << B̄), dividing by
            # B̄ makes the μ term negligible: μ/B̄ ≈ 0.01,
            # so FB² ≈ 1e-4 even with μ = 0.07.  This allows
            # the multiplier to persist at positive values
            # without penalty, violating complementary slackness.
            #
            # With raw arguments, FB(12.5, 0.074) ≈ 0.074,
            # FB² ≈ 0.005 — a 36× stronger signal pushing μ→0.
            # --------------------------------------------------
            fb_val = FischerBurmeisterLoss.fb_function(
                collateral_limit - b_prime, mu_hat
            )
            loss_fb = tf.reduce_mean(tf.square(fb_val))

            # ========================================================
            # Total AiO Objective  Ξ(Θ)
            #
            # Ξ = E[ f₁^Bell · f₂^Bell
            #      + ν_K  · f₁^Cap  · f₂^Cap
            #      + ν_μ  · f₁^Mult · f₂^Mult
            #      + ν_FB · (Ψ^FB)² ]
            #
            # The double-sample products give unbiased gradient
            # estimates (Section 2.1) because E[f(ε)] = 0 at
            # the true solution for each additive residual.
            # ========================================================
            total_loss = (
                loss_bellman
                + self.nu_K * loss_foc_k
                + self.nu_mu * loss_multiplier
                + self.nu_FB * loss_fb
            )

        # ============================================================
        # Gradient update
        # ============================================================
        grads = tape.gradient(total_loss, self._all_foc_variables)
        # Replace NaN/Inf gradients with zeros to prevent corruption
        grads = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            for g in grads
        ]
        # Track raw gradient norm before clipping (training health)
        grad_norm_raw = tf.linalg.global_norm(grads)
        if self.config.gradient_clip_norm:
            grads, _ = tf.clip_by_global_norm(
                grads, self.config.gradient_clip_norm
            )
        grad_norm_clipped = tf.linalg.global_norm(grads)
        self.optimizer.apply_gradients(zip(grads, self._all_foc_variables))

        # ============================================================
        # Diagnostics
        # ============================================================
        violation_rate = tf.reduce_mean(
            tf.cast(b_prime > collateral_limit, TENSORFLOW_DTYPE)
        )

        # ============================================================
        # Auxiliary: train default policy on frozen value net
        # ============================================================
        with tf.GradientTape() as default_tape:
            default_inputs = self._prepare_inputs(k, b, z)
            default_prob = self.default_policy_net(
                default_inputs, training=True
            )
            v_frozen = tf.stop_gradient(
                self.value_net(default_inputs, training=False)
            )
            is_default_soft = tf.sigmoid(-v_frozen)
            default_loss = tf.reduce_mean(
                tf.square(default_prob - is_default_soft)
            )

        default_grads = default_tape.gradient(
            default_loss, self.default_policy_net.trainable_variables
        )
        default_grads = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            for g in default_grads
        ]
        self.default_optimizer.apply_gradients(
            zip(default_grads, self.default_policy_net.trainable_variables)
        )

        return {
            "total_loss": total_loss,
            "loss_bellman": loss_bellman,
            "loss_foc_k": loss_foc_k,
            "loss_multiplier": loss_multiplier,
            "loss_fb": loss_fb,
            "default_loss": default_loss,
            "violation_rate": violation_rate,
            "avg_mu_hat": tf.reduce_mean(mu_hat),
            "avg_b_prime": tf.reduce_mean(b_prime),
            "avg_k_prime": tf.reduce_mean(k_prime),
            "avg_i_rate": tf.reduce_mean(i_rate),
            "avg_leverage_ratio": tf.reduce_mean(leverage_ratio),
            "avg_dividend": tf.reduce_mean(dividend),
            "avg_collateral_limit": tf.reduce_mean(collateral_limit),
            "avg_V": tf.reduce_mean(v_current),
            "avg_dV_dKp": tf.reduce_mean(0.5 * (dV_dKp_1 + dV_dKp_2)),
            "avg_dV_dBp": tf.reduce_mean(0.5 * (dV_dBp_1 + dV_dBp_2)),
            "avg_lambda_e": tf.reduce_mean(lambda_e),
            "avg_dBbar_dKp": tf.reduce_mean(dBbar_dKp),
            "avg_issuance_cost": tf.reduce_mean(issuance_cost),
            "issuing_fraction": tf.reduce_mean(is_issuing),
            "grad_norm_raw": grad_norm_raw,
            "grad_norm_clipped": grad_norm_clipped,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_step(
        self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Public interface — delegates to XLA-optimized implementation."""
        return self._optimized_train_step(k, b, z)

    def train(self) -> None:
        print(
            f"Starting AiO Training (Double Sample, Additive Residuals) "
            f"for {self.config.epochs} epochs..."
        )
        print(f"\n=== Optimization Settings ===")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Training step: XLA={self.optimization_config.use_xla}")
        print(f"Target value net: Polyak decay {self._polyak_start} → {self._polyak_end} over {self._polyak_anneal_epochs} epochs")
        print(f"Target update freq: every {self._target_update_freq} steps")
        print(f"Additive residuals: unbiased under expectation")
        print(f"Multiplier: μ̂ = softplus(x) ≥ 0 (direct Lagrange multiplier)")
        print(f"α_B (debt marginal cash): {float(self.alpha_B):.6f}")
        print(
            f"Loss weights: ν_K={float(self.nu_K):.1f}  "
            f"ν_μ={float(self.nu_mu):.1f}  ν_FB={float(self.nu_FB):.1f}"
        )
        print(
            f"Issuance cost params: η₀={float(self.eta_0_t):.4f}  "
            f"η₁={float(self.eta_1_t):.4f}"
        )
        print(f"STE steepness: k=10 (reduced for broader gradient support)")
        print("=" * 40 + "\n")

        # Build dataset pipeline
        dataset = self._build_dataset()
        data_iter = iter(dataset)

        # Warm-up: trace the function once before timing
        print("Warming up XLA compilation (this may take a minute)...")
        k_warmup, b_warmup, z_warmup = next(data_iter)
        _ = self.train_step(k_warmup, b_warmup, z_warmup)
        print("Warm-up complete.\n")

        import time

        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.epoch_var.assign(epoch)

            current_decay = self._get_annealed_decay(epoch)
            epoch_logs = self._run_epoch(data_iter, current_decay)
            epoch_logs['polyak_decay'] = current_decay
            epoch_time = time.time() - epoch_start_time

            self._log_tensorboard(epoch, epoch_logs)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs, epoch_time)

            if epoch % 20 == 0:
                self._save_checkpoint(epoch)

        self.summary_writer.flush()

    def _soft_update_targets_impl(self, decay: tf.Tensor) -> None:
        """Polyak-average update of target value network weights.

        θ⁻ ← decay · θ⁻ + (1 − decay) · θ

        Higher decay retains more of the existing target network,
        giving smoother / more stable Bellman targets.

        Args:
            decay: Averaging coefficient in [0, 1].
        """
        for target_var, source_var in zip(
            self.target_value_net.trainable_variables,
            self.value_net.trainable_variables,
        ):
            target_var.assign(
                decay * target_var + (1.0 - decay) * source_var
            )

    def _get_annealed_decay(self, epoch: int) -> float:
        """Linearly anneal the Polyak averaging decay over training.

        Starts at ``_polyak_start`` and linearly ramps to
        ``_polyak_end`` over ``_polyak_anneal_epochs`` epochs.
        After the annealing window the decay stays at ``_polyak_end``.
        """
        t = min(epoch / max(self._polyak_anneal_epochs, 1), 1.0)
        return self._polyak_start + t * (self._polyak_end - self._polyak_start)

    def _run_epoch(
        self, data_iter, current_decay: float
    ) -> Dict[str, float]:
        """Run one training epoch with Polyak target updates."""
        epoch_logs: Dict[str, float] = {}
        target_update_freq = self._target_update_freq
        decay_tensor = tf.constant(current_decay, dtype=TENSORFLOW_DTYPE)

        for step in range(self.config.steps_per_epoch):
            k, b, z = next(data_iter)
            logs = self.train_step(k, b, z)

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            # Soft-update target value net
            if step % target_update_freq == 0:
                self._soft_update_targets(decay_tensor)

        return epoch_logs

    def _get_current_lr(self) -> float:
        """Return the current learning rate from the optimizer."""
        if hasattr(self.optimizer, 'learning_rate'):
            lr = self.optimizer.learning_rate
            if callable(lr):
                return float(lr(self.optimizer.iterations))
            return float(lr)
        return 0.0

    def _log_tensorboard(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
    ) -> None:
        """Write essential training-health metrics to TensorBoard.

        Metrics are grouped into panels for easy navigation:

        - **loss/** — total, bellman, capital FOC, multiplier, FB,
                      default (per-step averages)
        - **policy/** — mean K', B', I/K, leverage, collateral limit,
                        dividend
        - **value/** — mean V, dV/dK', dV/dB', multiplier μ̂
        - **constraint/** — violation rate, FB loss
        - **gradients/** — raw and clipped gradient norms
        - **optimizer/** — learning rate
        - **weights/** — L2 norms of all networks
        """
        steps = self.config.steps_per_epoch
        step = epoch

        with self.summary_writer.as_default(step=step):
            # ---- Losses (per-step averages) ----
            tf.summary.scalar(
                "loss/total", epoch_logs.get("total_loss", 0) / steps)
            tf.summary.scalar(
                "loss/bellman", epoch_logs.get("loss_bellman", 0) / steps)
            tf.summary.scalar(
                "loss/capital_foc", epoch_logs.get("loss_foc_k", 0) / steps)
            tf.summary.scalar(
                "loss/multiplier", epoch_logs.get("loss_multiplier", 0) / steps)
            tf.summary.scalar(
                "loss/fischer_burmeister", epoch_logs.get("loss_fb", 0) / steps)
            tf.summary.scalar(
                "loss/default", epoch_logs.get("default_loss", 0) / steps)

            # ---- Policy diagnostics ----
            tf.summary.scalar(
                "policy/avg_k_prime", epoch_logs.get("avg_k_prime", 0) / steps)
            tf.summary.scalar(
                "policy/avg_b_prime", epoch_logs.get("avg_b_prime", 0) / steps)
            tf.summary.scalar(
                "policy/avg_i_rate", epoch_logs.get("avg_i_rate", 0) / steps)
            tf.summary.scalar(
                "policy/avg_leverage_ratio",
                epoch_logs.get("avg_leverage_ratio", 0) / steps)
            tf.summary.scalar(
                "policy/avg_dividend", epoch_logs.get("avg_dividend", 0) / steps)
            tf.summary.scalar(
                "policy/avg_collateral_limit",
                epoch_logs.get("avg_collateral_limit", 0) / steps)

            # ---- Value function diagnostics ----
            tf.summary.scalar(
                "value/avg_V", epoch_logs.get("avg_V", 0) / steps)
            tf.summary.scalar(
                "value/avg_dV_dKp", epoch_logs.get("avg_dV_dKp", 0) / steps)
            tf.summary.scalar(
                "value/avg_dV_dBp", epoch_logs.get("avg_dV_dBp", 0) / steps)
            tf.summary.scalar(
                "value/avg_mu_hat", epoch_logs.get("avg_mu_hat", 0) / steps)

            # ---- Constraint diagnostics ----
            tf.summary.scalar(
                "constraint/violation_rate",
                epoch_logs.get("violation_rate", 0) / steps)

            # ---- Issuance cost diagnostics ----
            tf.summary.scalar(
                "issuance/avg_lambda_e",
                epoch_logs.get("avg_lambda_e", 0) / steps)
            tf.summary.scalar(
                "issuance/avg_issuance_cost",
                epoch_logs.get("avg_issuance_cost", 0) / steps)
            tf.summary.scalar(
                "issuance/issuing_fraction",
                epoch_logs.get("issuing_fraction", 0) / steps)
            tf.summary.scalar(
                "issuance/avg_dBbar_dKp",
                epoch_logs.get("avg_dBbar_dKp", 0) / steps)

            # ---- Gradient health ----
            tf.summary.scalar(
                "gradients/norm_raw",
                epoch_logs.get("grad_norm_raw", 0) / steps)
            tf.summary.scalar(
                "gradients/norm_clipped",
                epoch_logs.get("grad_norm_clipped", 0) / steps)
            tf.summary.scalar(
                "gradients/clip_ratio",
                (epoch_logs.get("grad_norm_raw", 1e-8)
                 / max(epoch_logs.get("grad_norm_clipped", 1e-8), 1e-8)))

            # ---- Optimizer ----
            tf.summary.scalar("optimizer/learning_rate", self._get_current_lr())
            tf.summary.scalar(
                "optimizer/polyak_decay",
                epoch_logs.get("polyak_decay", 0.0))

            # ---- Weight norms (detect explosion / collapse) ----
            tf.summary.scalar(
                "weights/value_net_norm",
                tf.linalg.global_norm(self.value_net.trainable_variables))
            tf.summary.scalar(
                "weights/capital_policy_norm",
                tf.linalg.global_norm(
                    self.capital_policy_net.trainable_variables))
            tf.summary.scalar(
                "weights/debt_policy_norm",
                tf.linalg.global_norm(
                    self.debt_policy_net.trainable_variables))
            tf.summary.scalar(
                "weights/multiplier_net_norm",
                tf.linalg.global_norm(
                    self.multiplier_net.trainable_variables))
            tf.summary.scalar(
                "weights/default_net_norm",
                tf.linalg.global_norm(
                    self.default_policy_net.trainable_variables))

    def _log_epoch(
        self,
        epoch: int,
        epoch_logs: Dict[str, float],
        epoch_time: float = None,
    ) -> None:
        steps = self.config.steps_per_epoch
        time_str = f" | Time: {epoch_time:.2f}s" if epoch_time else ""
        print(
            f"Ep {epoch:<4} | "
            f"Tot: {epoch_logs['total_loss']/steps:.4f} | "
            f"Bell: {epoch_logs['loss_bellman']/steps:.2e} | "
            f"K_FOC: {epoch_logs['loss_foc_k']/steps:.2e} | "
            f"Mult: {epoch_logs['loss_multiplier']/steps:.2e} | "
            f"FB: {epoch_logs['loss_fb']/steps:.2e} | "
            f"Def: {epoch_logs['default_loss']/steps:.2e} | "
            f"K': {epoch_logs['avg_k_prime']/steps:.3f} | "
            f"B': {epoch_logs['avg_b_prime']/steps:.3f} | "
            f"μ: {epoch_logs['avg_mu_hat']/steps:.3f} | "
            f"I/K: {epoch_logs['avg_i_rate']/steps:.3f} | "
            f"lev: {epoch_logs['avg_leverage_ratio']/steps:.3f} | "
            f"viol: {epoch_logs['violation_rate']/steps:.3f} | "
            f"V: {epoch_logs['avg_V']/steps:.3f} | "
            f"dV/dK': {epoch_logs['avg_dV_dKp']/steps:.4f} | "
            f"dV/dB': {epoch_logs['avg_dV_dBp']/steps:.4f} | "
            f"λ(e): {epoch_logs['avg_lambda_e']/steps:.3f} | "
            f"iss%: {epoch_logs['issuing_fraction']/steps:.3f} | "
            f"τ: {epoch_logs.get('polyak_decay', 0.0):.4f} | "
            f"lr: {self._get_current_lr():.2e}"
            f"{time_str}"
        )

    def _save_checkpoint(self, epoch: int):
        try:
            save_checkpoint_risk_free(
                value_net=self.value_net,
                capital_policy_net=self.capital_policy_net,
                debt_policy_net=self.debt_policy_net,
                epoch=epoch,
                default_policy_net=self.default_policy_net,
            )
        except Exception:
            pass