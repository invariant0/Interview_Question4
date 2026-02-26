# src/econ_models/simulator/dl/risky_final_dist.py
"""Deep Learning Simulator for the distributional Risky Debt Model.

Simulates the economy using trained neural network policy functions
that take 9-dimensional inputs ``(k, b, z, rho, std, convex, fixed, eta0, eta1)``
spanning the distribution of economic parameters.

Architecture (matches ``dl_dist.risky_final.RiskyModelDL_FINAL``)
-----------------------------------------------------------------
- **Capital Policy** (sigmoid) — outputs normalised ``k'`` in [0, 1];
  ``denormalize_capital`` maps to physical units.
- **Debt Policy** (sigmoid) — outputs normalised ``b'``, shared across paths.
- **Investment Decision** (sigmoid) — binary invest/wait threshold.
- **Default Policy** (sigmoid) — binary default decision.
- **Equity Issuance Invest** (linear) — relu(x) gated by sigmoid(x) > 0.5.
- **Equity Issuance Noinvest** (linear) — same architecture, no-invest path.
- **Value Network** (linear) — V(s, p) for diagnostics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import ParamSpaceNormalizer, StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.econ import (
    BondPricingCalculator,
    ProductionFunctions,
)

tfd = tfp.distributions

# 3 state dims (k, b, z) + 6 param dims (rho, std, convex, fixed, eta0, eta1)
INPUT_DIM = 9


class DLSimulatorRiskyFinal_dist:
    """Simulator for the distributional risky debt model.

    Uses trained neural networks with 9-dim inputs matching the
    ``dl_dist.risky_final.RiskyModelDL_FINAL`` training code.

    Args:
        config: Deep learning configuration (network architecture, bounds).
        bonds: Parameter-space bounds dictionary for normalisation.
        hard_threshold: Threshold for binary decisions.
    """

    def __init__(
        self,
        config: DeepLearningConfig,
        bonds: dict,
        hard_threshold: float = 0.5,
    ) -> None:
        self.config = config
        self.normalizer_states = StateSpaceNormalizer(config)
        self.normalizer_params = ParamSpaceNormalizer(bonds)
        self.hard_threshold = hard_threshold
        self._build_networks()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        """Build policy network architectures (all input_dim=9).

        Activations match ``RiskyModelDL_FINAL`` (dist version).
        """
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="sigmoid", scale_factor=1.0,
            name="CapitalPolicyNet",
        )
        self.debt_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="sigmoid", scale_factor=1.0,
            name="DebtPolicyNet",
        )
        self.investment_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="sigmoid", scale_factor=1.0,
            name="InvestmentNet",
        )
        self.default_policy_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="sigmoid", scale_factor=1.0,
            name="DefaultPolicyNet",
        )
        self.equity_issuance_invest_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="linear", scale_factor=1.0,
            name="EquityIssuanceNetInvest",
        )
        self.equity_issuance_noinvest_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="linear", scale_factor=1.0,
            name="EquityIssuanceNetNoinvest",
        )
        self.value_function_net = NeuralNetFactory.build_mlp(
            input_dim=INPUT_DIM, output_dim=1, config=self.config,
            output_activation="linear", scale_factor=1.0,
            name="ValueNet",
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_solved_dl_solution(
        self,
        capital_policy_filepath: str,
        debt_filepath: str,
        investment_policy_filepath: str,
        default_policy_filepath: str,
        value_function_filepath: str,
        equity_issuance_invest_filepath: str = "",
        equity_issuance_noinvest_filepath: str = "",
        **kwargs,
    ) -> None:
        """Load trained network weights from ``.weights.h5`` files.

        Args:
            capital_policy_filepath: Path to capital policy weights.
            debt_filepath: Path to shared debt policy weights.
            investment_policy_filepath: Path to investment policy weights.
            default_policy_filepath: Path to default policy weights.
            value_function_filepath: Path to value function weights.
            equity_issuance_invest_filepath: Path to invest equity weights.
            equity_issuance_noinvest_filepath: Path to noinvest equity weights.
            **kwargs: Ignored (absorbs legacy params).
        """
        dummy_input = tf.zeros((1, INPUT_DIM), dtype=TENSORFLOW_DTYPE)

        networks = [
            (self.capital_policy_net, capital_policy_filepath, "capital policy"),
            (self.debt_policy_net, debt_filepath, "debt policy"),
            (self.investment_policy_net, investment_policy_filepath, "investment policy"),
            (self.default_policy_net, default_policy_filepath, "default policy"),
            (self.equity_issuance_invest_net, equity_issuance_invest_filepath, "equity issuance invest"),
            (self.equity_issuance_noinvest_net, equity_issuance_noinvest_filepath, "equity issuance noinvest"),
            (self.value_function_net, value_function_filepath, "value function"),
        ]

        for net, filepath, name in networks:
            _ = net(dummy_input)
            try:
                net.load_weights(filepath)
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load {name} weights from {filepath}: {e}"
                )

    # ------------------------------------------------------------------
    # Normalised-input helpers
    # ------------------------------------------------------------------

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
        """Normalise and concatenate the 9-dim input vector.

        Args:
            k: Capital, shape ``(batch, 1)``.
            b: Debt, shape ``(batch, 1)``.
            z: Productivity, shape ``(batch, 1)``.
            rho: AR(1) persistence, shape ``(batch, 1)``.
            std: Productivity std dev, shape ``(batch, 1)``.
            convex: Convex adjustment cost, shape ``(batch, 1)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)``.
            eta0: Equity issuance fixed cost, shape ``(batch, 1)``.
            eta1: Equity issuance linear cost, shape ``(batch, 1)``.

        Returns:
            Concatenated normalised inputs, shape ``(batch, 9)``.
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
    # Policy inference
    # ------------------------------------------------------------------

    def _get_policy_action(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        eta0: tf.Tensor,
        eta1: tf.Tensor,
        depreciation_rate: float,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get next-period capital, debt, and decision outputs.

        Returns:
            Tuple of (k_prime, b_prime, invest_prob, default_prob,
            equity_issuance, issuance_gate_prob), each shape ``(batch,)``.
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        b_2d = tf.reshape(b_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))

        tensors = [rho, std, convex, fixed, eta0, eta1]
        for i, t in enumerate(tensors):
            if len(t.shape) == 1:
                tensors[i] = tf.reshape(t, (-1, 1))
        rho, std, convex, fixed, eta0, eta1 = tensors

        inputs = self._prepare_inputs(k_2d, b_2d, z_2d, rho, std, convex, fixed, eta0, eta1)

        # Capital policy: sigmoid -> denormalise
        k_target = self.normalizer_states.denormalize_capital(
            self.capital_policy_net(inputs, training=False)
        )

        # Shared debt policy
        b_prime = self.normalizer_states.denormalize_debt(
            self.debt_policy_net(inputs, training=False)
        )

        # Decision nets
        invest_prob = self.investment_policy_net(inputs, training=False)
        default_prob = self.default_policy_net(inputs, training=False)

        # Separate equity issuance per branch (STE hard gate)
        issuance_logit_inv = self.equity_issuance_invest_net(inputs, training=False)
        equity_issuance_inv = tf.nn.relu(issuance_logit_inv)
        issuance_gate_prob_inv = tf.cast(
            tf.math.sigmoid(issuance_logit_inv) > 0.5, TENSORFLOW_DTYPE
        )

        issuance_logit_noinv = self.equity_issuance_noinvest_net(inputs, training=False)
        equity_issuance_noinv = tf.nn.relu(issuance_logit_noinv)
        issuance_gate_prob_noinv = tf.cast(
            tf.math.sigmoid(issuance_logit_noinv) > 0.5, TENSORFLOW_DTYPE
        )

        # Investment decision -> select capital and equity path
        k_prime_no_invest = k_2d * (1.0 - depreciation_rate)
        hard_invest = tf.cast(
            invest_prob > self.hard_threshold, TENSORFLOW_DTYPE,
        )
        k_prime = hard_invest * k_target + (1.0 - hard_invest) * k_prime_no_invest
        equity_issuance = (
            hard_invest * equity_issuance_inv
            + (1.0 - hard_invest) * equity_issuance_noinv
        )
        issuance_gate_prob = (
            hard_invest * issuance_gate_prob_inv
            + (1.0 - hard_invest) * issuance_gate_prob_noinv
        )

        # Clip to valid ranges
        k_prime = tf.clip_by_value(
            k_prime,
            clip_value_min=self.config.capital_min,
            clip_value_max=self.config.capital_max,
        )
        b_prime = tf.clip_by_value(
            b_prime,
            clip_value_min=self.config.debt_min,
            clip_value_max=self.config.debt_max,
        )

        return (
            tf.squeeze(k_prime, axis=-1),
            tf.squeeze(b_prime, axis=-1),
            tf.squeeze(invest_prob, axis=-1),
            tf.squeeze(default_prob, axis=-1),
            tf.squeeze(equity_issuance, axis=-1),
            tf.squeeze(issuance_gate_prob, axis=-1),
        )

    # ------------------------------------------------------------------
    # Bond price estimation via MC sampling
    # ------------------------------------------------------------------

    def _estimate_bond_price(
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
        econ_params: EconomicParams,
        mc_samples: Optional[int] = None,
    ) -> tf.Tensor:
        """Estimate bond price via MC sampling of default probability.

        Args:
            k_prime: Next-period capital, shape ``(batch,)``.
            b_prime: Next-period debt, shape ``(batch,)``.
            z_curr: Current productivity, shape ``(batch,)``.
            rho-eta1: Parameter tensors, shape ``(batch,)`` or ``(batch,1)``.
            econ_params: Economic parameters for the run.
            mc_samples: Number of Monte Carlo samples.

        Returns:
            Bond price q, shape ``(batch,)``.
        """
        if mc_samples is None:
            mc_samples = self.config.mc_sample_number_bond_priceing

        batch_size = tf.shape(k_prime)[0]
        one_minus_tax = tf.cast(1.0 - econ_params.corporate_tax_rate, TENSORFLOW_DTYPE)

        # Sample future shocks and transition Z
        eps = tfd.Normal(0.0, econ_params.productivity_std_dev).sample(
            [batch_size, mc_samples]
        )
        z_curr_bc = tf.broadcast_to(
            tf.reshape(z_curr, (-1, 1)), (batch_size, mc_samples)
        )
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, econ_params.productivity_persistence, eps
        )

        # Broadcast inputs
        k_prime_bc = tf.broadcast_to(tf.reshape(k_prime, (-1, 1)), (batch_size, mc_samples))
        b_prime_bc = tf.broadcast_to(tf.reshape(b_prime, (-1, 1)), (batch_size, mc_samples))
        flat_shape = (batch_size * mc_samples, 1)

        k_flat = tf.reshape(k_prime_bc, flat_shape)
        b_flat = tf.reshape(b_prime_bc, flat_shape)
        z_flat = tf.reshape(z_prime, flat_shape)

        # Broadcast param tensors
        def _bc_param(p):
            p = tf.reshape(p, (-1, 1)) if len(p.shape) == 1 else p
            return tf.reshape(
                tf.broadcast_to(p, (batch_size, mc_samples)), flat_shape
            )

        rho_flat = _bc_param(rho)
        std_flat = _bc_param(std)
        convex_flat = _bc_param(convex)
        fixed_flat = _bc_param(fixed)
        eta0_flat = _bc_param(eta0)
        eta1_flat = _bc_param(eta1)

        inputs_eval = self._prepare_inputs(
            k_flat, b_flat, z_flat,
            rho_flat, std_flat, convex_flat, fixed_flat, eta0_flat, eta1_flat,
        )

        # Get default probability
        is_default = self.default_policy_net(inputs_eval, training=False)

        # Recovery and payoff
        profit_next = one_minus_tax * ProductionFunctions.cobb_douglas(
            k_flat, z_flat, econ_params
        )
        recovery = BondPricingCalculator.recovery_value(
            profit_next, k_flat, econ_params
        )
        payoff = BondPricingCalculator.bond_payoff(recovery, b_flat, is_default)

        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, mc_samples)), axis=1,
        )

        b_prime_2d = tf.reshape(b_prime, (-1, 1))
        expected_payoff_2d = tf.reshape(expected_payoff, (-1, 1))
        bond_price_2d = BondPricingCalculator.risk_neutral_price(
            expected_payoff_2d, b_prime_2d,
            econ_params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price,
            risk_free_price_val=1.0 / (1.0 + econ_params.risk_free_rate),
        )

        return tf.squeeze(bond_price_2d, axis=-1)

    # ------------------------------------------------------------------
    # Compiled simulation helpers (pure TF ops, @tf.function-safe)
    # ------------------------------------------------------------------

    def _estimate_bond_price_tensor(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_curr: tf.Tensor,
        rho_t: tf.Tensor,
        std_t: tf.Tensor,
        convex_t: tf.Tensor,
        fixed_t: tf.Tensor,
        eta0_t: tf.Tensor,
        eta1_t: tf.Tensor,
        prod_persist: tf.Tensor,
        prod_std: tf.Tensor,
        corp_tax: tf.Tensor,
        cap_share: tf.Tensor,
        default_cost: tf.Tensor,
        dep_rate: tf.Tensor,
        risk_free_rate: tf.Tensor,
    ) -> tf.Tensor:
        """Bond pricing via MC — pure TF ops, no Python/EconomicParams.

        All economic parameters are pre-converted to ``tf.Tensor``
        scalars by the caller.  Uses ``self.config`` for fixed
        hyper-parameters (``epsilon_debt``, ``min_q_price``,
        ``mc_sample_number_bond_priceing``) which never change.

        Args:
            k_prime: Next-period capital, shape ``(batch,)``.
            b_prime: Next-period debt, shape ``(batch,)``.
            z_curr: Current productivity, shape ``(batch,)``.
            rho_t … eta1_t: Per-firm distributional-param tensors,
                shape ``(batch,)``.
            prod_persist … risk_free_rate: Scalar ``tf.Tensor`` econ
                parameters.

        Returns:
            Bond price ``q``, shape ``(batch,)``.
        """
        mc_samples = self.config.mc_sample_number_bond_priceing
        epsilon_debt = tf.constant(
            self.config.epsilon_debt, dtype=TENSORFLOW_DTYPE,
        )
        min_q_price = tf.constant(
            self.config.min_q_price, dtype=TENSORFLOW_DTYPE,
        )
        one = tf.cast(1.0, TENSORFLOW_DTYPE)
        one_minus_tax = one - corp_tax

        batch_size = tf.shape(k_prime)[0]

        # --- Sample future shocks ---
        eps = tf.random.normal(
            shape=(batch_size, mc_samples), dtype=TENSORFLOW_DTYPE,
        ) * prod_std

        # --- Z transition: ln(z') = ρ·ln(z) + ε ---
        z_curr_bc = tf.broadcast_to(
            tf.reshape(z_curr, (-1, 1)), (batch_size, mc_samples),
        )
        z_safe = tf.maximum(z_curr_bc, tf.cast(1e-12, TENSORFLOW_DTYPE))
        z_prime = tf.exp(prod_persist * tf.math.log(z_safe) + eps)

        # --- Broadcast & flatten to (batch*mc, 1) ---
        flat_n = batch_size * mc_samples
        k_flat = tf.reshape(
            tf.broadcast_to(
                tf.reshape(k_prime, (-1, 1)), (batch_size, mc_samples),
            ),
            (flat_n, 1),
        )
        b_flat = tf.reshape(
            tf.broadcast_to(
                tf.reshape(b_prime, (-1, 1)), (batch_size, mc_samples),
            ),
            (flat_n, 1),
        )
        z_flat = tf.reshape(z_prime, (flat_n, 1))

        def _bc_flat(p: tf.Tensor) -> tf.Tensor:
            return tf.reshape(
                tf.broadcast_to(
                    tf.reshape(p, (-1, 1)), (batch_size, mc_samples),
                ),
                (flat_n, 1),
            )

        inputs_eval = self._prepare_inputs(
            k_flat, b_flat, z_flat,
            _bc_flat(rho_t), _bc_flat(std_t), _bc_flat(convex_t),
            _bc_flat(fixed_t), _bc_flat(eta0_t), _bc_flat(eta1_t),
        )

        # --- Default probability ---
        is_default = self.default_policy_net(inputs_eval, training=False)

        # --- Recovery: (1-ξ) · ((1-τ)·z·k^α + (1-δ)·k) ---
        k_pos = tf.maximum(k_flat, tf.cast(1e-8, TENSORFLOW_DTYPE))
        profit_next = one_minus_tax * z_flat * tf.pow(k_pos, cap_share)
        firm_value_gross = profit_next + (one - dep_rate) * k_flat
        recovery = (one - default_cost) * firm_value_gross

        # --- Bond payoff ---
        b_face = tf.maximum(b_flat, tf.cast(0.0, TENSORFLOW_DTYPE))
        payoff_default = tf.minimum(recovery, b_face)
        payoff = is_default * payoff_default + (one - is_default) * b_face

        # --- Average over MC samples ---
        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, mc_samples)), axis=1,
        )

        # --- Risk-neutral pricing ---
        q_rf = one / (one + risk_free_rate)
        b_1d = tf.reshape(b_prime, (-1,))
        has_debt = b_1d > epsilon_debt
        denom = (one + risk_free_rate) * tf.maximum(b_1d, epsilon_debt)
        q_risky = expected_payoff / denom
        q_final = tf.where(has_debt, q_risky, q_rf * tf.ones_like(q_risky))
        return tf.clip_by_value(q_final, min_q_price, q_rf)

    def _simulate_step_tensor(
        self,
        k_t: tf.Tensor,
        b_t: tf.Tensor,
        z_t: tf.Tensor,
        alive_mask: tf.Tensor,
        eps_t: tf.Tensor,
        rho_t: tf.Tensor,
        std_t: tf.Tensor,
        convex_t: tf.Tensor,
        fixed_t: tf.Tensor,
        eta0_t: tf.Tensor,
        eta1_t: tf.Tensor,
        dep_rate: tf.Tensor,
        default_threshold: tf.Tensor,
        prod_persist: tf.Tensor,
        prod_std: tf.Tensor,
        corp_tax: tf.Tensor,
        cap_share: tf.Tensor,
        default_cost: tf.Tensor,
        risk_free_rate: tf.Tensor,
    ) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
        tf.Tensor, tf.Tensor,
    ]:
        """Single compiled simulation step — all TF, no Python branching.

        Dead firms (``alive_mask == 0``) keep their last-alive state so
        that neural-network inputs remain numerically valid.  Outputs
        for dead firms are zeroed; the caller applies NaN masking after
        exiting the compiled loop.

        Returns:
            ``(k_next, b_next, z_next, new_alive, q_t, equity_t)``
        """
        # --- Policy action (all firms; dead ones have safe frozen states) ---
        k_next, b_next, inv_prob, default_prob, equity_iss, gate_prob = (
            self._get_policy_action(
                k_t, b_t, z_t,
                rho_t, std_t, convex_t, fixed_t, eta0_t, eta1_t,
                dep_rate,
            )
        )

        # --- Default decision (only alive firms can newly default) ---
        newly_default = (
            tf.cast(default_prob > default_threshold, TENSORFLOW_DTYPE)
            * alive_mask
        )
        new_alive = alive_mask * (tf.cast(1.0, TENSORFLOW_DTYPE) - newly_default)

        # Freeze dead firms' states at last-alive values (avoids NaN in NNs)
        k_next = tf.where(new_alive > 0.5, k_next, k_t)
        b_next = tf.where(new_alive > 0.5, b_next, b_t)

        # --- Bond pricing (full batch; dead firms get valid but unused q) ---
        q_t = self._estimate_bond_price_tensor(
            k_next, b_next, z_t,
            rho_t, std_t, convex_t, fixed_t, eta0_t, eta1_t,
            prod_persist, prod_std,
            corp_tax, cap_share, default_cost, dep_rate, risk_free_rate,
        )

        # --- Equity issuance (VFI-style cash-flow calculation) ---
        # Reference: simulator/vfi/risky.py — payout-based equity logic.
        _one = tf.cast(1.0, TENSORFLOW_DTYPE)
        _one_minus_tax = _one - corp_tax
        _k_safe = tf.maximum(k_t, tf.cast(1e-8, TENSORFLOW_DTYPE))
        _revenue = _one_minus_tax * z_t * tf.pow(_k_safe, cap_share)

        _debt_inflow = q_t * b_next
        _interest = (_one - q_t) * tf.maximum(b_next, tf.cast(0.0, TENSORFLOW_DTYPE))
        _tax_shield = corp_tax * _interest / (_one + risk_free_rate)

        _investment = k_next - (_one - dep_rate) * k_t
        _hard_invest = tf.cast(inv_prob > self.hard_threshold, TENSORFLOW_DTYPE)
        _convex_cost = convex_t * (_investment ** 2) / (tf.cast(2.0, TENSORFLOW_DTYPE) * _k_safe)
        _fixed_cost = fixed_t * k_t * _hard_invest
        _adj_cost = _convex_cost + _fixed_cost

        _payout = _revenue + _debt_inflow + _tax_shield - b_t
        _payout = _payout - _hard_invest * (_investment + _adj_cost)

        equity_t = tf.maximum(tf.cast(0.0, TENSORFLOW_DTYPE), -_payout)

        # # --- OLD: NN-based equity issuance (uncomment to restore) ---
        # equity_t = equity_iss * gate_prob

        # --- Productivity transition ---
        z_safe = tf.maximum(z_t, tf.cast(1e-12, TENSORFLOW_DTYPE))
        z_next = tf.exp(prod_persist * tf.math.log(z_safe) + eps_t)
        z_next = tf.where(new_alive > 0.5, z_next, z_t)

        return k_next, b_next, z_next, new_alive, q_t, equity_t

    def _simulate_loop(
        self,
        k_init: tf.Tensor,
        b_init: tf.Tensor,
        z_init: tf.Tensor,
        eps_path: tf.Tensor,
        rho_t: tf.Tensor,
        std_t: tf.Tensor,
        convex_t: tf.Tensor,
        fixed_t: tf.Tensor,
        eta0_t: tf.Tensor,
        eta1_t: tf.Tensor,
        dep_rate: tf.Tensor,
        default_threshold: tf.Tensor,
        prod_persist: tf.Tensor,
        prod_std: tf.Tensor,
        corp_tax: tf.Tensor,
        cap_share: tf.Tensor,
        default_cost: tf.Tensor,
        risk_free_rate: tf.Tensor,
    ) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor,
        tf.Tensor, tf.Tensor, tf.Tensor,
    ]:
        """Full T-step simulation in pure TF ops.

        Used as the body of ``@tf.function`` (and optionally XLA).
        Returns stacked tensors of shape ``(T+1, batch)`` for states
        and ``(T, batch)`` for per-step outputs.

        Returns:
            ``(k_stack, b_stack, z_stack, alive_stack, eq_stack, q_stack)``
        """
        T = tf.shape(eps_path)[1] - 1

        k_ta = tf.TensorArray(
            TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        b_ta = tf.TensorArray(
            TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        z_ta = tf.TensorArray(
            TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        alive_ta = tf.TensorArray(
            TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        eq_ta = tf.TensorArray(
            TENSORFLOW_DTYPE, size=T,
            dynamic_size=False, clear_after_read=False,
        )
        q_ta = tf.TensorArray(
            TENSORFLOW_DTYPE, size=T,
            dynamic_size=False, clear_after_read=False,
        )

        # Initial state
        k_ta = k_ta.write(0, k_init)
        b_ta = b_ta.write(0, b_init)
        z_ta = z_ta.write(0, z_init)
        alive_ta = alive_ta.write(0, tf.ones_like(k_init))

        k_t = k_init
        b_t = b_init
        z_t = z_init
        alive = tf.ones_like(k_init)

        for t in tf.range(T):
            k_t, b_t, z_t, alive, q_step, eq_step = (
                self._simulate_step_tensor(
                    k_t, b_t, z_t, alive, eps_path[:, t + 1],
                    rho_t, std_t, convex_t, fixed_t, eta0_t, eta1_t,
                    dep_rate, default_threshold,
                    prod_persist, prod_std,
                    corp_tax, cap_share, default_cost, risk_free_rate,
                )
            )
            k_ta = k_ta.write(t + 1, k_t)
            b_ta = b_ta.write(t + 1, b_t)
            z_ta = z_ta.write(t + 1, z_t)
            alive_ta = alive_ta.write(t + 1, alive)
            eq_ta = eq_ta.write(t, eq_step)
            q_ta = q_ta.write(t, q_step)

        return (
            k_ta.stack(),       # (T+1, batch)
            b_ta.stack(),       # (T+1, batch)
            z_ta.stack(),       # (T+1, batch)
            alive_ta.stack(),   # (T+1, batch)
            eq_ta.stack(),      # (T, batch)
            q_ta.stack(),       # (T, batch)
        )

    # ------------------------------------------------------------------
    # Public simulation API
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_states: Tuple[tf.Tensor, ...],
        innovation_sequence: tf.Tensor,
        econ_params: EconomicParams,
        default_threshold: float = 0.5,
        use_compiled: bool = True,
        jit_compile: bool = False,
    ) -> dict:
        """Simulate the risky debt economy using distributional neural networks.

        The distributional model is trained once and can simulate under
        *different* economic parameters without re-training.

        When ``use_compiled=True`` (default) the entire T-step loop
        runs inside a single ``@tf.function`` graph — the same pattern
        used by the basic-model simulator.  This eliminates per-step
        Python ↔ TF dispatch overhead and gives ~10-50× speedup.

        Setting ``jit_compile=True`` additionally enables XLA fusion,
        which can provide a further 2-5× gain when batch and time
        dimensions are fixed across calls (the normal SMM case).

        Args:
            initial_states: Tuple ``(K_0, B_0, Z_0)`` or ``(K_0, Z_0)``.
            innovation_sequence: Shape ``(batch, T+1)`` standard normal shocks.
            econ_params: Economic parameters for this simulation run.
            default_threshold: Probability threshold for default decision.
            use_compiled: If ``True`` run the compiled TensorArray loop;
                ``False`` falls back to the original eager Python loop.
            jit_compile: If ``True`` (and ``use_compiled``), enable XLA
                compilation for additional kernel fusion.

        Returns:
            Dictionary with simulation results:
                ``K_curr``, ``K_next``, ``B_curr``, ``B_next``,
                ``Z_curr``, ``Z_next``, ``equity_issuance``, ``bond_price``.
        """
        # Parse initial states
        if len(initial_states) == 2:
            k_curr, z_curr = initial_states
            b_curr = tf.zeros_like(k_curr)
        elif len(initial_states) == 3:
            k_curr, b_curr, z_curr = initial_states
        else:
            raise ValueError(
                f"Expected 2 or 3 initial states, got {len(initial_states)}"
            )

        # Ensure 1-D float tensors
        if len(k_curr.shape) > 1:
            k_curr = tf.squeeze(k_curr, axis=-1)
        if len(z_curr.shape) > 1:
            z_curr = tf.squeeze(z_curr, axis=-1)
        if len(b_curr.shape) > 1:
            b_curr = tf.squeeze(b_curr, axis=-1)
        k_curr = tf.cast(k_curr, TENSORFLOW_DTYPE)
        z_curr = tf.cast(z_curr, TENSORFLOW_DTYPE)
        b_curr = tf.cast(b_curr, TENSORFLOW_DTYPE)

        eps_path = tf.cast(
            innovation_sequence * econ_params.productivity_std_dev,
            TENSORFLOW_DTYPE,
        )

        if use_compiled:
            return self._simulate_compiled(
                k_curr, b_curr, z_curr, eps_path,
                econ_params, default_threshold, jit_compile,
            )

        return self._simulate_eager(
            k_curr, b_curr, z_curr, eps_path,
            econ_params, default_threshold,
        )

    # ------------------------------------------------------------------
    # Compiled path
    # ------------------------------------------------------------------

    def _simulate_compiled(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
        eps_path: tf.Tensor,
        econ_params: EconomicParams,
        default_threshold: float,
        jit_compile: bool,
    ) -> dict:
        """Run the simulation via ``@tf.function``-compiled loop.

        All varying economic parameters are broadcast to per-firm
        tensors; fixed parameters are converted to scalar
        ``tf.Tensor`` so that the traced graph is reusable across
        different ``econ_params`` values without retracing.
        """
        batch_size = tf.shape(k_curr)[0]

        # --- Per-firm parameter tensors (change each SMM evaluation) ---
        ones = tf.ones((batch_size,), dtype=TENSORFLOW_DTYPE)
        rho_t = ones * tf.constant(
            econ_params.productivity_persistence, dtype=TENSORFLOW_DTYPE,
        )
        std_t = ones * tf.constant(
            econ_params.productivity_std_dev, dtype=TENSORFLOW_DTYPE,
        )
        convex_t = ones * tf.constant(
            econ_params.adjustment_cost_convex, dtype=TENSORFLOW_DTYPE,
        )
        fixed_t = ones * tf.constant(
            econ_params.adjustment_cost_fixed, dtype=TENSORFLOW_DTYPE,
        )
        eta0_t = ones * tf.constant(
            econ_params.equity_issuance_cost_fixed, dtype=TENSORFLOW_DTYPE,
        )
        eta1_t = ones * tf.constant(
            econ_params.equity_issuance_cost_linear, dtype=TENSORFLOW_DTYPE,
        )

        # --- Scalar tensors for transition / bond-pricing math ---
        dep_rate = tf.constant(
            econ_params.depreciation_rate, dtype=TENSORFLOW_DTYPE,
        )
        default_thresh = tf.constant(
            default_threshold, dtype=TENSORFLOW_DTYPE,
        )
        prod_persist = tf.constant(
            econ_params.productivity_persistence, dtype=TENSORFLOW_DTYPE,
        )
        prod_std = tf.constant(
            econ_params.productivity_std_dev, dtype=TENSORFLOW_DTYPE,
        )
        corp_tax = tf.constant(
            econ_params.corporate_tax_rate, dtype=TENSORFLOW_DTYPE,
        )
        cap_share = tf.constant(
            econ_params.capital_share, dtype=TENSORFLOW_DTYPE,
        )
        default_cost = tf.constant(
            econ_params.default_cost_proportional, dtype=TENSORFLOW_DTYPE,
        )
        risk_free_rate = tf.constant(
            econ_params.risk_free_rate, dtype=TENSORFLOW_DTYPE,
        )

        # --- Select / create the compiled function ---
        attr = "_compiled_fn_xla" if jit_compile else "_compiled_fn"
        if not hasattr(self, attr):
            fn = tf.function(
                self._simulate_loop,
                reduce_retracing=True,
                jit_compile=jit_compile,
            )
            setattr(self, attr, fn)
        compiled_fn = getattr(self, attr)

        # --- Execute compiled loop ---
        k_stack, b_stack, z_stack, alive_stack, eq_stack, q_stack = (
            compiled_fn(
                k_curr, b_curr, z_curr, eps_path,
                rho_t, std_t, convex_t, fixed_t, eta0_t, eta1_t,
                dep_rate, default_thresh,
                prod_persist, prod_std,
                corp_tax, cap_share, default_cost, risk_free_rate,
            )
        )

        # --- Convert to NumPy & apply NaN mask for dead firms ---
        # Shapes: stacks are (T+1, batch) or (T, batch); transpose
        # to (batch, T+1) / (batch, T) to match the output convention.
        k_path = tf.transpose(k_stack).numpy()      # (batch, T+1)
        b_path = tf.transpose(b_stack).numpy()
        z_path = tf.transpose(z_stack).numpy()
        alive_path = tf.transpose(alive_stack).numpy()  # (batch, T+1)
        eq_vals = tf.transpose(eq_stack).numpy()     # (batch, T)
        q_vals = tf.transpose(q_stack).numpy()

        # alive_path[:, t] == 1 means firm was alive entering step t.
        # K_curr[:, t] should be NaN if alive_path[:, t] == 0.
        # K_next[:, t] should be NaN if alive_path[:, t+1] == 0.
        alive_curr = alive_path[:, :-1]   # (batch, T)
        alive_next = alive_path[:, 1:]    # (batch, T)
        nan_curr = alive_curr < 0.5
        nan_next = alive_next < 0.5

        K_curr_out = k_path[:, :-1].copy()
        K_next_out = k_path[:, 1:].copy()
        B_curr_out = b_path[:, :-1].copy()
        B_next_out = b_path[:, 1:].copy()
        Z_curr_out = z_path[:, :-1].copy()
        Z_next_out = z_path[:, 1:].copy()

        K_curr_out[nan_curr] = np.nan
        K_next_out[nan_next] = np.nan
        B_curr_out[nan_curr] = np.nan
        B_next_out[nan_next] = np.nan
        Z_curr_out[nan_curr] = np.nan
        Z_next_out[nan_next] = np.nan
        eq_vals[nan_next] = np.nan
        q_vals[nan_next] = np.nan

        return {
            "K_curr": K_curr_out,
            "K_next": K_next_out,
            "B_curr": B_curr_out,
            "B_next": B_next_out,
            "Z_curr": Z_curr_out,
            "Z_next": Z_next_out,
            "equity_issuance": eq_vals,
            "bond_price": q_vals,
        }

    # ------------------------------------------------------------------
    # Eager fallback (original implementation)
    # ------------------------------------------------------------------

    def _simulate_eager(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
        eps_path: tf.Tensor,
        econ_params: EconomicParams,
        default_threshold: float,
    ) -> dict:
        """Original eager Python-loop simulation (kept as fallback)."""
        batch_size = eps_path.shape[0]
        T = eps_path.shape[1] - 1

        # Build per-sample parameter tensors
        ones = tf.ones((batch_size,), dtype=TENSORFLOW_DTYPE)
        rho_t = ones * econ_params.productivity_persistence
        std_t = ones * econ_params.productivity_std_dev
        convex_t = ones * econ_params.adjustment_cost_convex
        fixed_t = ones * econ_params.adjustment_cost_fixed
        eta0_t = ones * econ_params.equity_issuance_cost_fixed
        eta1_t = ones * econ_params.equity_issuance_cost_linear
        dep_rate = econ_params.depreciation_rate

        # Storage arrays
        K_curr_out = np.full((batch_size, T), np.nan)
        K_next_out = np.full((batch_size, T), np.nan)
        B_curr_out = np.full((batch_size, T), np.nan)
        B_next_out = np.full((batch_size, T), np.nan)
        Z_curr_out = np.full((batch_size, T), np.nan)
        Z_next_out = np.full((batch_size, T), np.nan)
        equity_issuance_out = np.full((batch_size, T), np.nan)
        bond_price_out = np.full((batch_size, T), np.nan)

        has_defaulted = np.zeros(batch_size, dtype=bool)

        k_sim = k_curr.numpy().copy()
        b_sim = b_curr.numpy().copy()
        z_sim = z_curr.numpy().copy()

        for t in range(T):
            K_curr_out[:, t] = k_sim
            B_curr_out[:, t] = b_sim
            Z_curr_out[:, t] = z_sim

            alive = ~has_defaulted
            if not np.any(alive):
                break

            k_t = tf.constant(k_sim, dtype=TENSORFLOW_DTYPE)
            b_t = tf.constant(b_sim, dtype=TENSORFLOW_DTYPE)
            z_t = tf.constant(z_sim, dtype=TENSORFLOW_DTYPE)

            (k_next_tf, b_next_tf, invest_prob, default_prob,
             equity_iss_tf, gate_prob_tf) = self._get_policy_action(
                k_t, b_t, z_t,
                rho_t, std_t, convex_t, fixed_t, eta0_t, eta1_t,
                dep_rate,
            )

            # Default decision
            default_prob_np = default_prob.numpy()
            newly_default = np.zeros(batch_size, dtype=bool)
            newly_default[alive] = default_prob_np[alive] > default_threshold
            has_defaulted |= newly_default
            alive = ~has_defaulted

            k_next_np = np.full(batch_size, np.nan)
            b_next_np = np.full(batch_size, np.nan)

            if np.any(alive):
                k_next_np[alive] = k_next_tf.numpy()[alive]
                b_next_np[alive] = b_next_tf.numpy()[alive]

            # Bond price via MC
            q_t = np.full(batch_size, np.nan)
            if np.any(alive):
                q_val = self._estimate_bond_price(
                    tf.constant(k_next_np[alive], dtype=TENSORFLOW_DTYPE),
                    tf.constant(b_next_np[alive], dtype=TENSORFLOW_DTYPE),
                    tf.constant(z_sim[alive], dtype=TENSORFLOW_DTYPE),
                    rho_t[:tf.reduce_sum(tf.cast(alive, tf.int32))],
                    std_t[:tf.reduce_sum(tf.cast(alive, tf.int32))],
                    convex_t[:tf.reduce_sum(tf.cast(alive, tf.int32))],
                    fixed_t[:tf.reduce_sum(tf.cast(alive, tf.int32))],
                    eta0_t[:tf.reduce_sum(tf.cast(alive, tf.int32))],
                    eta1_t[:tf.reduce_sum(tf.cast(alive, tf.int32))],
                    econ_params,
                )
                q_t[alive] = q_val.numpy()
            bond_price_out[:, t] = q_t

            # Equity issuance (VFI-style cash-flow calculation)
            # Reference: simulator/vfi/risky.py — payout-based equity logic.
            eq_iss_t = np.full(batch_size, np.nan)
            if np.any(alive):
                kc = k_sim[alive]
                bc = b_sim[alive]
                zc = z_sim[alive]
                kn = k_next_np[alive]
                bn = b_next_np[alive]
                qa = q_t[alive]

                revenue = (
                    (1.0 - econ_params.corporate_tax_rate)
                    * zc
                    * np.power(np.maximum(kc, 1e-8), econ_params.capital_share)
                )
                debt_inflow = qa * bn
                interest = (1.0 - qa) * np.maximum(bn, 0.0)
                tax_shield = (
                    econ_params.corporate_tax_rate * interest
                    / (1.0 + econ_params.risk_free_rate)
                )

                investment = kn - (1.0 - dep_rate) * kc
                safe_k = np.maximum(kc, 1e-8)
                inv_prob_np = invest_prob.numpy()[alive]
                hard_invest_np = (inv_prob_np > self.hard_threshold).astype(np.float64)
                convex_cost = (
                    econ_params.adjustment_cost_convex
                    * (investment ** 2) / (2.0 * safe_k)
                )
                fixed_cost = econ_params.adjustment_cost_fixed * kc * hard_invest_np
                adj_cost = convex_cost + fixed_cost

                payout = revenue + debt_inflow + tax_shield - bc
                payout -= hard_invest_np * (investment + adj_cost)

                eq_iss_t[alive] = np.maximum(0.0, -payout)

            # # --- OLD: NN-based equity issuance (uncomment to restore) ---
            # eq_iss_t = np.full(batch_size, np.nan)
            # if np.any(alive):
            #     eq_iss_t[alive] = (
            #         equity_iss_tf.numpy() * gate_prob_tf.numpy()
            #     )[alive]

            equity_issuance_out[:, t] = eq_iss_t

            # Productivity transition
            z_next_tf = TransitionFunctions.log_ar1_transition(
                z_curr=tf.constant(z_sim, dtype=TENSORFLOW_DTYPE),
                rho=econ_params.productivity_persistence,
                epsilon=eps_path[:, t + 1],
            )
            z_next_np = z_next_tf.numpy().copy()
            z_next_np[~alive] = np.nan

            K_next_out[:, t] = k_next_np
            B_next_out[:, t] = b_next_np
            Z_next_out[:, t] = z_next_np

            k_sim = k_next_np.copy()
            b_sim = b_next_np.copy()
            z_sim = z_next_np.copy()

        return {
            "K_curr": K_curr_out,
            "K_next": K_next_out,
            "B_curr": B_curr_out,
            "B_next": B_next_out,
            "Z_curr": Z_curr_out,
            "Z_next": Z_next_out,
            "equity_issuance": equity_issuance_out,
            "bond_price": bond_price_out,
        }
