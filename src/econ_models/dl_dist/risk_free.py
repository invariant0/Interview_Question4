"""Deep Learning solver for the Risk-Free Debt Model — Distributional AiO approach.

Faithful distributional extension of ``dl.risk_free.RiskFreeModelDL``.
Minimises the product of residuals from two independent samples for
unbiased gradient estimation (Maliar, Maliar & Winant 2021).

Unlike the single-parameter ``dl.risk_free`` module, this variant accepts
economic parameters ``(rho, std, convex, fixed, eta0, eta1)`` as additional
network inputs so that a *single* trained model generalises over the
parameter distribution.  All networks have ``input_dim = 9``:
``(k, b, z, rho, std, convex, fixed, eta0, eta1)``.

Equation System (from the reference PDF)
-----------------------------------------

Eq. 1 — Dividend (cash flow identity):

  D = (1−τ)ZKᶿ − I − Φ(I,K) + B'/(1+r) + τrB'/(1+r)² − B − η(e)·𝟙{e<0}

  where  Φ(I,K) = (ψ₀/2)(I²/K) + ψ₁·K·𝟙{I≠0}   (adjustment cost)
         I = K' − (1−δ)K                           (investment)
         η(e) = (η₀ + η₁|e|)·𝟙{e<0}               (issuance cost)

Eq. 2 — Bellman equation:

  V(K,B,Z,p) = D + β·𝔼[V(K',B',Z',p) | Z]

Eq. 3 — Collateral (borrowing) constraint:

  B' ≤ B̄(K') = (1−τ)z_min·(K')ᶿ + τδK' + s·K'

Eq. 4 — Capital Euler (FOC w.r.t. K'):

  λ(e)·(1 + ψ₀I/K) = β·E[∂V'/∂K'] + μ̂·dB̄/dK'

  where λ(e) = 1 + η₁·𝟙{e<0} is the marginal cost of external funds

Eq. 8 — Debt Multiplier FOC (FOC w.r.t. B'):

  μ̂ = λ(e)·α_B + β·E[∂V'/∂B']

  where α_B = q + τrq² is the marginal cash per unit face-value debt

Eq. 9 — AiO loss with Fischer-Burmeister complementarity:

  Ξ(Θ) = 𝔼[ D₁^Bell·D₂^Bell
           + ν_K · D₁^Cap · D₂^Cap
           + ν_μ · D₁^Mult · D₂^Mult
           + ν_FB · [Ψ^FB(B̄−B', μ̂)]² ]

  Ψ^FB(a,b) = a + b − √(a²+b²) enforces:
    (i)   B̄ − B' ≥ 0      (primal feasibility)
    (ii)  μ̂ ≥ 0             (dual feasibility)
    (iii) μ̂·(B̄ − B') = 0  (complementary slackness)

Architecture
------------
- **Value Net** ``V(s, p)`` with Polyak-averaged target network.
- **Capital Policy** — sigmoid ∈ (0,1) → K' via ``denormalize_capital``.
- **Debt Policy** — sigmoid ∈ (0,1) → B' via ``denormalize_debt``.
- **Multiplier Net** — softplus output ``μ̂ ≥ 0`` (direct Lagrange multiplier).
- **Default Policy** — auxiliary ``P(default)`` for downstream risky model.

Optimisations
-------------
- Fused training step with persistent ``GradientTape``.
- XLA compilation for graph optimisation.
- Polyak-averaged target-network updates.
- ``ParamSpaceNormalizer`` normalises parameters to ``[0, 1]``.
- TensorBoard logging for all training metrics.

Example
-------
>>> from econ_models.dl_dist.risk_free import RiskFreeModelDL
>>> model = RiskFreeModelDL(params, config, bonds_config)
>>> model.train()
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, NamedTuple, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.math import FischerBurmeisterLoss
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import ParamSpaceNormalizer, StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl_dist.training.dataset_builder import DatasetBuilder
from econ_models.econ import (
    AdjustmentCostCalculator,
    CollateralCalculator,
    IssuanceCostCalculator,
    ProductionFunctions,
)
from econ_models.io.checkpoints import save_checkpoint_risk_free_dist

tfd = tfp.distributions

# State dimensions: k, b, z = 3
# Parameter dimensions: rho, std, convex, fixed, eta0, eta1 = 6
# Total input_dim = 9
INPUT_DIM = 9


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class OptimizationConfig(NamedTuple):
    """Configuration for training optimisations.

    Attributes:
        use_xla: Enable XLA compilation for faster execution.
        use_mixed_precision: Enable mixed-precision (float16) training.
        target_update_freq: Training steps between target-network updates.
        prefetch_buffer: ``tf.data`` prefetch buffer size.
        cache_dataset: Whether to cache the training dataset in memory.
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

    use_xla: bool = False
    use_mixed_precision: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE
    cache_dataset: bool = True
    cross_product_sampling: bool = False
    batch_size_states: int | None = None
    batch_size_params: int | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RiskFreeModelDL:
    """Distributional AiO solver for the risk-free debt model.

    Minimises the product of residuals from two independent samples
    (Section 2.1) across four loss components:

      Ξ(Θ) = E[ D₁^Bell × D₂^Bell
               + ν_K · D₁^Cap × D₂^Cap
               + ν_μ · D₁^Mult × D₂^Mult
               + ν_FB · [Ψ^FB((B̄−B'), μ̂)]² ]

    All FOC residuals use *additive* form (f = LHS − RHS), normalized
    by shock-independent state-dependent scales wrapped in stop_gradient
    to avoid Jensen's inequality bias:

      Bellman  (Eq. 2):   D^Bell = [V − D − β·Vtarg'] / max(|D|, 1)  (target net)
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
      - Multiplier network →  μ̂(K,B,Z,p) = softplus(x) ≥ 0

    KKT conditions (via Fischer-Burmeister):
      μ̂ ≥ 0,  B̄ − B' ≥ 0,  μ̂·(B̄ − B') = 0

    Networks take ``(k, b, z, rho, std, convex, fixed, eta0, eta1)`` as
    normalised inputs (``input_dim = 9``).

    Args:
        params: Economic parameters (discount factor, depreciation, etc.).
        config: Deep learning configuration (layers, learning rate, etc.).
        bonds_config: State-space and parameter bounds dictionary.
        optimization_config: Optional training-optimisation settings.
        log_dir_prefix: Directory prefix for TensorBoard logs.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        bonds_config: dict,
        optimization_config: OptimizationConfig | None = None,
        log_dir_prefix: str = "logs/risk_free_dist",
    ) -> None:
        self.params = params
        self.config = config
        self.bonds_config = bonds_config
        self.log_dir_prefix = log_dir_prefix
        self.optimization_config = optimization_config or OptimizationConfig()

        self.config.update_value_scale(self.params)
        self.normalizer_states = StateSpaceNormalizer(self.config)
        self.normalizer_params = ParamSpaceNormalizer(self.bonds_config)
        self.epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.training_progress = tf.Variable(
            1.0, dtype=TENSORFLOW_DTYPE, trainable=False,
        )

        self._cache_constants()
        self._build_networks()
        self._build_optimizers()
        self._compile_train_functions()
        self._build_summary_writer()

        # Polyak averaging schedule for target network
        self._polyak_start = self.config.polyak_averaging_decay
        self._polyak_end = self.config.polyak_decay_end
        self._polyak_anneal_epochs = self.config.polyak_decay_epochs
        self._target_update_freq = self.config.target_update_freq

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _cache_constants(self) -> None:
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
        self.beta = tf.constant(
            self.params.discount_factor, dtype=TENSORFLOW_DTYPE,
        )

        # Risk-free rate r
        self.risk_free_rate_t = tf.constant(r, dtype=TENSORFLOW_DTYPE)

        # Bond price q = 1/(1+r)
        self.q_rf = tf.constant(1.0 / (1.0 + r), dtype=TENSORFLOW_DTYPE)

        # 1 − δ  (used for capital law of motion)
        self.one_minus_delta = tf.constant(
            1.0 - self.params.depreciation_rate, dtype=TENSORFLOW_DTYPE,
        )

        # Capital bounds (for clipping K')
        self.k_min = tf.constant(
            self.bonds_config["k_min"], dtype=TENSORFLOW_DTYPE,
        )
        self.k_max = tf.constant(
            self.bonds_config["k_max"], dtype=TENSORFLOW_DTYPE,
        )

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
            q + tau * r * q * q, dtype=TENSORFLOW_DTYPE,
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
            self.params.capital_share, dtype=TENSORFLOW_DTYPE,
        )

        # ----------------------------------------------------------
        # Tauchen width for per-sample z_min computation
        #
        # In the distributional variant, z follows an AR(1) process
        # with per-sample (rho, std).  The minimum productivity is
        # derived from Tauchen's discretisation:
        #   z_min(rho, std) = exp(-m · std / sqrt(1 − rho²))
        # where m = tauchen_width (default 3.0).
        # ----------------------------------------------------------
        self.tauchen_width = tf.constant(3.0, dtype=TENSORFLOW_DTYPE)

        # ----------------------------------------------------------
        # Collateral constraint derivative  dB̄/dK'
        #
        # B̄(K') = (1−τ)·z_min·(K')^θ + τδK' + sK'
        # dB̄/dK' = (1−τ)·z_min·θ·(K')^(θ−1) + τδ + s
        #
        # The constant part τδ + s is cached here; the
        # power-law term and per-sample z_min are computed
        # per-batch in the train step.
        # ----------------------------------------------------------
        self.tau_delta_t = tf.constant(
            tau * self.params.depreciation_rate, dtype=TENSORFLOW_DTYPE,
        )
        self.recovery_fraction_t = tf.constant(
            self.params.collateral_recovery_fraction, dtype=TENSORFLOW_DTYPE,
        )

    def _build_networks(self) -> None:
        """Construct neural networks with parameter-aware inputs (dim=9)."""
        # 1. Value Network V(K, B, Z, p)
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueNet",
        )

        # Target Value Network V_targ
        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="TargetValueNet",
        )
        self.target_value_net.set_weights(self.value_net.get_weights())

        # 2. Capital Policy: sigmoid ∈ (0,1) → K' = denormalize_capital(σ(x))
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="sigmoid",
            name="CapitalPolicyNet",
        )

        # 3. Debt Policy: sigmoid ∈ (0,1) → B' = denormalize_debt(σ(x))
        self.debt_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="sigmoid",
            name="DebtPolicyNet",
        )

        # 4. Multiplier Network ĥ(s, p) ≥ 0 (softplus guarantee)
        self.multiplier_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="softplus",
            name="MultiplierNet",
        )

        # 5. Default Policy (auxiliary — used for downstream risky model)
        self.default_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="sigmoid",
            name="DefaultPolicyNet",
        )

        # Cache trainable variables for the joint AiO gradient.
        self._all_foc_variables = (
            self.value_net.trainable_variables
            + self.capital_policy_net.trainable_variables
            + self.debt_policy_net.trainable_variables
            + self.multiplier_net.trainable_variables
        )

    def _build_optimizers(self) -> None:
        """Build Adam optimizers with exponential-decay learning-rate schedules.

        All scheduling parameters are read exclusively from
        ``self.config`` (populated from the JSON config file).
        Raises ``ValueError`` if any required field is missing.
        """
        self.config.validate_scheduling_fields(
            ['lr_decay_rate', 'lr_decay_steps',
             'polyak_averaging_decay', 'polyak_decay_end',
             'polyak_decay_epochs', 'target_update_freq'],
            'RiskFreeModelDL_Dist'
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
        self.default_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr_schedule,
        )
        if decay_rate < 1.0:
            print(
                f"[LR] ExponentialDecay: {self.config.learning_rate:.2e} "
                f"× {decay_rate} every {decay_steps} steps"
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
        return eps[:, 0:1], eps[:, 1:1 + 1]

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
    # Normalised-input helpers
    # ------------------------------------------------------------------

    def _build_inputs(
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
        """Normalise and concatenate state + parameter inputs.

        Returns:
            Tensor of shape ``(batch, 9)``.
        """
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
                decay * target_var + (1.0 - decay) * source_var,
            )

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
        eps1_normal: tf.Tensor,
        eps2_normal: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """XLA-compatible AiO training step — distributional variant.

        Structure follows Section 2 of the reference document:

        1. Draw two independent shocks ε₁, ε₂ (Section 2.1)
        2. Evaluate policies and compute the dividend D (Eq. 1)
        3. Form residual pairs for each loss component (Section 2.2 A–C)
        4. Combine via the AiO objective Ξ(Θ) (Eq. 9)

        Args:
            k, b, z: State variables, shape ``(batch, 1)``.
            rho, std, convex, fixed, eta0, eta1: Parameters, ``(batch, 1)``.
            eps1_normal, eps2_normal: Unit-normal shocks, ``(batch, 1)``.

        Returns:
            Dictionary of scalar training metrics.
        """
        # --- Scale shocks by per-sample std and compute future productivity ---
        eps1 = std * eps1_normal
        eps2 = std * eps2_normal

        with tf.GradientTape() as tape:
            # ========================================================
            # Step 2: Current-Period Policies and Dividend
            # ========================================================
            inputs = self._build_inputs(
                k, b, z, rho, std, convex, fixed, eta0, eta1,
            )

            v_current = self.value_net(inputs, training=True)

            # --- Policy outputs ---
            # Capital policy: sigmoid ∈ (0,1), soft-squeezed to
            # [ε, 1−ε] via affine rescaling.  Unlike clip_by_value
            # (which has zero gradient at boundaries), this
            # preserves gradient everywhere:
            #   k_norm = ε + (1 − 2ε) · σ(x)
            # Gradient = (1 − 2ε) · σ'(x) > 0  ∀ x.
            k_prime_norm = self.capital_policy_net(inputs, training=True)
            k_prime = self.normalizer_states.denormalize_capital(k_prime_norm)

            # --- Derived quantities (capital) ---
            # Investment:  I = K' − (1−δ)K
            investment = k_prime - self.one_minus_delta * k
            # Investment rate:  I/K
            i_rate = investment / k

            # --- Per-sample z_min from AR(1) parameters ---
            # z follows log-AR(1):  ln z' = rho·ln z + eps,  eps ~ N(0, std²)
            # Tauchen grid minimum:  z_min = exp(-m · std / sqrt(1 − rho²))
            z_min_sample = tf.exp(
                -self.tauchen_width * std
                / tf.sqrt(1.0 - rho * rho)
            )

            # --- Collateral limit B̄(K') ---
            # B̄ = collateral fraction × liquidation value of K'
            # This is the upper bound on debt issuance (Eq. 3).
            # Uses per-sample z_min derived from (rho, std).
            collateral_limit = CollateralCalculator.calculate_limit(
                k_prime,
                z_min_sample,
                self.params,
                self.params.collateral_recovery_fraction,
            )

            # --- Debt policy ---
            # Debt policy: sigmoid ∈ (0,1), soft-squeezed to [ε, 1−ε]
            b_prime_norm = self.debt_policy_net(inputs, training=True)
            b_prime = self.normalizer_states.denormalize_debt(b_prime_norm)

            # Leverage ratio for diagnostics: B'/B̄
            leverage_ratio = b_prime / (collateral_limit + 1e-8)

            # --- Multiplier estimate ---
            # μ̂(K,B,Z,p) = softplus(x) ≥ 0
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
                k, z, self.params,
            )

            # Adjustment cost  Φ(I,K) = (ψ₀/2)(I²/K) + ψ₁·K·𝟙{I≠0}
            # marginal cost   Φ_I = ∂Φ/∂I = ψ₀·(I/K)
            adj_cost, phi_I = AdjustmentCostCalculator.calculate_dist(
                investment, k, convex, fixed,
            )

            # Bond proceeds:  B'/(1+r) = B'·q
            bond_proceeds = b_prime * self.q_rf

            # Tax shield:  τrB'/(1+r)² = τ·r·B'·q²
            tax_shield = (
                self.tau_t * self.risk_free_rate_t
                * b_prime * self.q_rf * self.q_rf
            )

            # Raw payout  e  (Eq. 1, before issuance costs)
            payout = (
                revenue - investment - adj_cost
                + bond_proceeds + tax_shield - b
            )

            # --- Equity issuance cost  η(e) ---
            # When payout e < 0, the firm raises external equity at cost
            #   η(e) = (η₀ + η₁|e|) · 𝟙{e < 0}
            # Uses STE (Straight-Through Estimator) for differentiability:
            # forward pass uses exact hard threshold, backward pass uses
            # smooth sigmoid surrogate for gradient flow.
            # Uses per-sample eta0, eta1 for distributional training.
            issuance_cost = IssuanceCostCalculator.calculate_with_grad_dist(
                payout, eta0, eta1,
            )

            # Dividend  D = e − η(e)   (Eq. 1, after issuance costs)
            dividend = payout - issuance_cost

            # --- Marginal cost of funds  λ(e) ---
            # λ(e) = ∂D/∂e = 1 + η₁·𝟙{e < 0}
            # STE: exact forward indicator, smooth backward gradient.
            # Steepness k=1.0 for broad gradient support during early
            # training when policies are imprecise.
            is_issuing_hard = tf.cast(payout < 0.0, TENSORFLOW_DTYPE)
            k_ste = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
            is_issuing_soft = tf.math.sigmoid(-k_ste * payout)
            is_issuing = is_issuing_soft + tf.stop_gradient(
                is_issuing_hard - is_issuing_soft
            )
            # lambda_e = 1.0 + eta1 * is_issuing
            lambda_e = 1.0 + eta1 * is_issuing

            # --- Collateral constraint derivative  dB̄/dK' ---
            # B̄(K') = (1−τ)·z_min·(K')^θ + τδK' + sK'
            # dB̄/dK' = (1−τ)·z_min·θ·(K')^(θ−1) + τδ + s
            # Uses per-sample z_min derived from (rho, std).
            dBbar_dKp = (
                self.one_minus_tau * z_min_sample * self.capital_share_t
                * tf.pow(k_prime, self.capital_share_t - 1.0)
                + self.tau_delta_t
                + self.recovery_fraction_t
            )

            # ========================================================
            # Step 3: Future States  (Section 2.1)
            # ========================================================
            z_prime_1 = TransitionFunctions.log_ar1_transition(
                z, rho, eps1,
            )
            z_prime_2 = TransitionFunctions.log_ar1_transition(
                z, rho, eps2,
            )

            # ========================================================
            # Step 4: Future Value and Partial Derivatives
            # ========================================================

            # --- FOC partials via ONLINE value net ---
            # Full gradients flow through ∂V/∂K' and ∂V/∂B' for
            # joint AiO learning of policies and value.
            #
            # _build_inputs must be called INSIDE the v_tape
            # context so the tape records the path from
            # (k_prime, b_prime) → normalize → concat → value_net.
            with tf.GradientTape(persistent=True) as v_tape:
                v_tape.watch([k_prime, b_prime])
                inputs_next_1 = self._build_inputs(
                    k_prime, b_prime, z_prime_1,
                    rho, std, convex, fixed, eta0, eta1,
                )
                inputs_next_2 = self._build_inputs(
                    k_prime, b_prime, z_prime_2,
                    rho, std, convex, fixed, eta0, eta1,
                )
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
            inputs_next_1_targ = self._build_inputs(
                tf.stop_gradient(k_prime),
                tf.stop_gradient(b_prime),
                tf.stop_gradient(z_prime_1),
                rho, std, convex, fixed, eta0, eta1,
            )
            inputs_next_2_targ = self._build_inputs(
                tf.stop_gradient(k_prime),
                tf.stop_gradient(b_prime),
                tf.stop_gradient(z_prime_2),
                rho, std, convex, fixed, eta0, eta1,
            )
            v_next_target_1 = tf.stop_gradient(
                self.target_value_net(inputs_next_1_targ, training=False),
            )
            v_next_target_2 = tf.stop_gradient(
                self.target_value_net(inputs_next_2_targ, training=False),
            )

            # ========================================================
            # Step 5: Loss Components — Additive Residuals
            #
            # All residuals use the ADDITIVE form f = LHS − RHS.
            # This ensures E[f(ε)] = 0 at the true solution,
            # which is required for the double-sample product
            # E[f(ε₁)·f(ε₂)] = E[f]² = 0 to be unbiased.
            #
            # Each residual is normalized by a shock-INDEPENDENT,
            # state-dependent scale factor (wrapped in stop_gradient)
            # to keep magnitudes ~O(1) without introducing bias.
            # ========================================================

            # --------------------------------------------------
            # A.  Bellman Residuals — Semi-gradient, Additive
            #
            #   f(ε) = V(K,B,Z,p) − D − β·V_target(K',B',Z'(ε),p)
            #
            # Semi-gradient: the TARGET value net (Polyak-averaged)
            # provides V' with stop_gradient.  Only the online
            # V(K,B,Z,p) receives gradient from the Bellman loss.
            #
            # stop_gradient on DIVIDEND (actor-critic separation):
            # The Bellman loss trains the value net V(K,B,Z,p)
            # to satisfy the Bellman equation but does NOT push
            # policy gradients through ∂D/∂K' or ∂D/∂B'.
            # --------------------------------------------------
            dividend_sg = tf.stop_gradient(dividend)
            res_bell_1 = v_current - dividend_sg - self.beta * v_next_target_1
            res_bell_2 = v_current - dividend_sg - self.beta * v_next_target_2

            # Normalize by the DIVIDEND scale rather than V.
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
            # --------------------------------------------------
            lhs_cap = (1.0 + phi_I) * lambda_e  # λ(e)·(1 + ψ₀I/K)
            collateral_benefit = mu_hat * dBbar_dKp  # μ̂·dB̄/dK'

            res_cap_1 = lhs_cap - self.beta * dV_dKp_1 - collateral_benefit
            res_cap_2 = lhs_cap - self.beta * dV_dKp_2 - collateral_benefit

            # Normalize by the deterministic LHS (shock-independent).
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
            # --------------------------------------------------
            fb_val = FischerBurmeisterLoss.fb_function(
                collateral_limit - b_prime, mu_hat,
            )
            loss_fb = tf.reduce_mean(tf.square(fb_val))

            # ========================================================
            # Total AiO Objective  Ξ(Θ)
            #
            # Ξ = E[ f₁^Bell · f₂^Bell
            #      + ν_K  · f₁^Cap  · f₂^Cap
            #      + ν_μ  · f₁^Mult · f₂^Mult
            #      + ν_FB · (Ψ^FB)² ]
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
                grads, self.config.gradient_clip_norm,
            )
        grad_norm_clipped = tf.linalg.global_norm(grads)
        self.optimizer.apply_gradients(
            zip(grads, self._all_foc_variables),
        )

        # ============================================================
        # Diagnostics
        # ============================================================
        violation_rate = tf.reduce_mean(
            tf.cast(b_prime > collateral_limit, TENSORFLOW_DTYPE),
        )

        # ============================================================
        # Auxiliary: train default policy on frozen value net
        # ============================================================
        with tf.GradientTape() as default_tape:
            default_inputs = self._build_inputs(
                k, b, z, rho, std, convex, fixed, eta0, eta1,
            )
            default_prob = self.default_policy_net(
                default_inputs, training=True,
            )
            v_frozen = tf.stop_gradient(
                self.value_net(default_inputs, training=False),
            )
            is_default_soft = tf.sigmoid(-v_frozen)
            default_loss = tf.reduce_mean(
                tf.square(default_prob - is_default_soft),
            )

        default_grads = default_tape.gradient(
            default_loss, self.default_policy_net.trainable_variables,
        )
        default_grads = [
            tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            for g in default_grads
        ]
        self.default_optimizer.apply_gradients(
            zip(
                default_grads,
                self.default_policy_net.trainable_variables,
            ),
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
        eps1: tf.Tensor,
        eps2: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Public interface — delegates to XLA-optimised implementation.

        Args:
            k, b, z: State variables, shape ``(batch, 1)``.
            rho, std, convex, fixed, eta0, eta1: Parameters, ``(batch, 1)``.
            eps1, eps2: Unit-normal shock draws, shape ``(batch, 1)``.

        Returns:
            Dictionary of scalar training metrics.
        """
        return self._compiled_train_step(
            k, b, z, rho, std, convex, fixed, eta0, eta1, eps1, eps2,
        )

    def train(self) -> None:
        """Run the full training loop."""
        import time

        print(
            f"Starting AiO Training (Double Sample, Additive Residuals) "
            f"for {self.config.epochs} epochs..."
        )
        print(f"\n=== Optimization Settings ===")
        print(f"Batch size: {self.config.batch_size}")
        print(
            "Architecture: ValueNet + CapitalPolicyNet + DebtPolicyNet "
            "+ MultiplierNet + DefaultPolicyNet (9-dim input)"
        )
        print(f"Training step: XLA={self.optimization_config.use_xla}")
        print(
            f"Target value net: Polyak decay {self._polyak_start} → "
            f"{self._polyak_end} over {self._polyak_anneal_epochs} epochs"
        )
        print(f"Target update freq: every {self._target_update_freq} steps")
        print(f"Additive residuals: unbiased under expectation")
        print(f"Multiplier: μ̂ = softplus(x) ≥ 0 (direct Lagrange multiplier)")
        print(f"α_B (debt marginal cash): {float(self.alpha_B):.6f}")
        print(
            f"Loss weights: ν_K={float(self.nu_K):.1f}  "
            f"ν_μ={float(self.nu_mu):.1f}  ν_FB={float(self.nu_FB):.1f}"
        )
        print(f"STE steepness: k=1.0 (broad gradient support)")
        print("=" * 60 + "\n")

        # Build dataset pipeline
        dataset = self._build_dataset()
        data_iter = iter(dataset)

        # Warm up XLA compilation with one step.
        print("Warming up XLA compilation (this may take a minute)...")
        k_w, b_w, z_w, rho_w, std_w, conv_w, fix_w, eta0_w, eta1_w = (
            next(data_iter)
        )
        eps1_w, eps2_w = self._sample_shocks(tf.shape(k_w)[0])
        _ = self.train_step(
            k_w, b_w, z_w, rho_w, std_w, conv_w, fix_w, eta0_w, eta1_w,
            eps1_w, eps2_w,
        )
        print("Warm-up complete.\n")

        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.epoch_var.assign(epoch)

            current_decay = self._get_annealed_decay(epoch)
            epoch_logs = self._run_epoch(data_iter, current_decay)
            epoch_logs["polyak_decay"] = current_decay
            epoch_time = time.time() - epoch_start_time

            self._log_tensorboard(epoch, epoch_logs)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs, epoch_time)

            if epoch % 20 == 0:
                self._save_checkpoint(epoch)

        self.summary_writer.flush()

    def _run_epoch(
        self, data_iter, current_decay: float,
    ) -> Dict[str, float]:
        """Run one training epoch with Polyak target updates.

        Args:
            data_iter: Iterator over
                ``(k, b, z, rho, std, convex, fixed, eta0, eta1)`` batches.
            current_decay: Polyak-averaging decay for target-network updates.

        Returns:
            Accumulated metrics over all steps in the epoch.
        """
        epoch_logs: Dict[str, float] = {}
        target_update_freq = self._target_update_freq
        decay_tensor = tf.constant(current_decay, dtype=TENSORFLOW_DTYPE)

        for step in range(self.config.steps_per_epoch):
            k, b, z, rho, std, convex, fixed, eta0, eta1 = next(data_iter)
            eps1, eps2 = self._sample_shocks(tf.shape(k)[0])
            logs = self.train_step(
                k, b, z, rho, std, convex, fixed, eta0, eta1, eps1, eps2,
            )

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            # Soft-update target value net
            if step % target_update_freq == 0:
                self._compiled_soft_update(decay_tensor)

        return epoch_logs

    def _get_annealed_decay(self, epoch: int) -> float:
        """Linearly anneal the Polyak averaging decay over training.

        Starts at ``_polyak_start`` and linearly ramps to
        ``_polyak_end`` over ``_polyak_anneal_epochs`` epochs.
        After the annealing window the decay stays at ``_polyak_end``.
        """
        t = min(epoch / max(self._polyak_anneal_epochs, 1), 1.0)
        return self._polyak_start + t * (self._polyak_end - self._polyak_start)

    def _get_current_lr(self) -> float:
        """Return the current learning rate from the optimizer."""
        if hasattr(self.optimizer, 'learning_rate'):
            lr = self.optimizer.learning_rate
            if callable(lr):
                return float(lr(self.optimizer.iterations))
            return float(lr)
        return 0.0

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
        epoch_time: float | None = None,
    ) -> None:
        """Log epoch training metrics to stdout."""
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

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        try:
            save_checkpoint_risk_free_dist(
                value_net=self.value_net,
                capital_policy_net=self.capital_policy_net,
                debt_policy_net=self.debt_policy_net,
                epoch=epoch,
                default_policy_net=self.default_policy_net,
            )
        except Exception:
            pass
