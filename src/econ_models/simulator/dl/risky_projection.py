# src/econ_models/simulator/dl/risky_projection.py
"""Deep Learning Simulator for the Risky Debt Projection Model.

Simulates the economy with endogenous default using trained neural
network policy functions for capital, debt, investment, and default
decisions.

Architecture (matches ``dl.risky_projection.RiskyModelDL_PROJECTION``)
----------------------------------------------------------------------
Three-branch architecture with branch-selection network:

    invest:   [k', b', equity_raw]   (3 outputs)
    wait:     [b', equity_raw]       (2 outputs)

Branch selection is performed by a softmax network with 3 outputs.
At simulation time the branch is selected by argmax (0=invest, 1=wait, 2=default).

- **Default Decision** — determined by the branch selection network:
  argmax == 2 triggers default.
- **Continuous Network** V_cont(s) (linear activation) — continuation value.
- **Value Network** V(s) = V_cont(s) * (1 - default_prob) where default_prob
  is the 3rd output of the branch selection network.
- **Bond Pricing** — MC sample-based using the branch selection network
  to determine default: default_prob(s') from 3rd softmax output.
- **Equity issuance** — learned from policy network outputs:
  equity_issuance = softplus(M * equity_raw) / D  (M, D from config),
  equity_prob = sigmoid(equity_raw) with straight-through estimator.
  issuance_cost = equity_prob * (eta_0 + eta_1 * equity_issuance).
  dividend = payout - issuance_cost.

Decision hierarchy:
1. Default: argmax of branch selection == 2 (default branch).
2. Branch decision: argmax of softmax [invest, wait, default] (0=invest, 1=wait, 2=default).
3. Equity issuance: from policy net outputs; softplus(M*raw)/D (M, D from config) = amount,
   sigmoid(raw) with STE = probability gate.

Bond price estimated via MC sampling of default_prob(s') from branch selection net.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.econ import (
    AdjustmentCostCalculator,
    BondPricingCalculator,
    DebtFlowCalculator,
    ProductionFunctions,
)

tfd = tfp.distributions


class DLSimulatorRiskyProjection:
    """Simulator for the risky debt projection model (3-branch architecture).

    Uses trained neural networks matching the ``RiskyModelDL_PROJECTION``
    training architecture.

    Decision networks:
    - Branch Selection: softmax (3-output) [invest, wait, default]
    - Default Decision: argmax == 2 from branch selection network

    Equity issuance from policy network outputs (matching training):
    equity_issuance = softplus(M * equity_raw) / D  (M, D from config),
    equity_prob = sigmoid(equity_raw) with STE.
    issuance_cost = equity_prob * (eta_0 + eta_1 * equity_issuance).
    dividend = payout - issuance_cost.

    Value networks:
    - Continuous Network: V_cont(k, b, z) for continuation value
    - Value Network: V(k, b, z) = V_cont(k, b, z) * (1 - default_prob)

    Bond pricing:
    - MC sample-based using branch_selection_net to determine default at s':
      is_default = cast(default_prob(s') > 0.5)

    Attributes:
        config: Deep learning configuration.
        params: Economic parameters.
        normalizer: State space normalizer for network inputs.
        hard_threshold: Threshold for binary decisions.
    """

    def __init__(
        self,
        config: DeepLearningConfig,
        params: EconomicParams,
        hard_threshold: float = 0.5,
    ) -> None:
        self.config = config
        self.params = params
        self.normalizer = StateSpaceNormalizer(config)
        self.hard_threshold = hard_threshold
        self.config.update_value_scale(self.params)

        # Equity issuance cost constants (analytical, matching training)
        self.eta_0 = self.params.equity_issuance_cost_fixed
        self.eta_1 = self.params.equity_issuance_cost_linear

        # Equity issuance softplus transform: softplus(multiplier * raw) / divisor
        self._eq_sp_mul = self.config.equity_softplus_multiplier
        self._eq_sp_div = self.config.equity_softplus_divisor

        # Equity gate probability multiplier: sigmoid(gate_mul * raw)
        self._eq_gate_mul = self.config.equity_gate_prob_multiplier or 1.0

        self._build_networks()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        """Build all network architectures matching ``RiskyModelDL_PROJECTION``.

        Three-branch architecture:
          - invest_policy_net: [k', b', equity_raw] (3 outputs, linear)
          - wait_policy_net: [b', equity_raw] (2 outputs, linear)
          - branch_selection_net: softmax (3 outputs: [invest, wait, default])
          - continuous_net: V_cont(s)
        """
        policy_layers = self.config.policy_hidden_layers

        # Branch selection: softmax (3 outputs: index 0=invest, 1=wait, 2=default)
        self.branch_selection_net = NeuralNetFactory.build_mlp(
            input_dim=3, output_dim=3, config=self.config,
            output_activation="softmax",
            hidden_layers=policy_layers,
            scale_factor=1.0,
            name="BranchSelectionNet",
        )

        # invest_policy: [k', b', equity_raw] (3 linear outputs)
        self.invest_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3, output_dim=3, config=self.config,
            output_activation="linear", hidden_layers=policy_layers,
            scale_factor=1.0,
            name="InvestPolicyNet",
        )

        # wait_policy: [b', equity_raw] (2 linear outputs)
        self.wait_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3, output_dim=2, config=self.config,
            output_activation="linear", hidden_layers=policy_layers,
            scale_factor=1.0,
            name="WaitPolicyNet",
        )

        # --- Continuous value network V_cont(s) ---
        self.continuous_net = NeuralNetFactory.build_mlp(
            input_dim=3, output_dim=1, config=self.config,
            output_activation="linear", scale_factor=1.0,
            name="ContinuousNet",
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_solved_dl_solution(
        self,
        invest_policy_filepath: str,
        wait_policy_filepath: str,
        branch_selection_filepath: str,
        continuous_net_filepath: str,
        **kwargs,
    ) -> None:
        """Load trained network weights from ``.weights.h5`` files.

        Matches checkpoint naming from
        ``save_checkpoint_risky_projection`` (3-branch architecture).

        Args:
            invest_policy_filepath: Path to invest policy weights [k', b'].
            wait_policy_filepath: Path to wait policy weights [b'].
            branch_selection_filepath: Path to 3-way branch-selection weights.
            continuous_net_filepath: Path to continuation value weights.
            **kwargs: Ignored (absorbs legacy params).
        """
        dummy_input = tf.zeros((1, 3), dtype=TENSORFLOW_DTYPE)

        networks = [
            (self.invest_policy_net, invest_policy_filepath, "invest policy"),
            (self.wait_policy_net, wait_policy_filepath, "wait policy"),
            (self.branch_selection_net, branch_selection_filepath, "branch selection"),
            (self.continuous_net, continuous_net_filepath, "continuous net"),
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
    ) -> tf.Tensor:
        """Normalise and concatenate state variables for network input."""
        if len(k.shape) == 1:
            k = tf.reshape(k, (-1, 1))
        if len(b.shape) == 1:
            b = tf.reshape(b, (-1, 1))
        if len(z.shape) == 1:
            z = tf.reshape(z, (-1, 1))

        return tf.concat([
            self.normalizer.normalize_capital(k),
            self.normalizer.normalize_debt(b),
            self.normalizer.normalize_productivity(z),
        ], axis=1)

    # ------------------------------------------------------------------
    # Policy / value inference
    # ------------------------------------------------------------------

    def estimate_bond_price(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_curr: tf.Tensor,
        eps: tf.Tensor = None,
    ) -> tf.Tensor:
        """Estimate bond price via sample-based Monte Carlo using branch selection net.

        Default probability at next-period states is determined by the
        branch selection network: ``default_prob(s') > 0.5`` indicates default.

        Args:
            k_prime: Next-period capital, shape ``(batch, 1)``.
            b_prime: Next-period debt, shape ``(batch, 1)``.
            z_curr: Current productivity, shape ``(batch, 1)``.
            eps: Pre-sampled shocks ``(batch, n_samples)``. Sampled
                fresh when *None*.

        Returns:
            Bond price *q* of shape ``(batch, 1)``.
        """
        batch_size = tf.shape(z_curr)[0]
        n_samples = self.config.mc_sample_number_bond_priceing
        one_minus_tax = tf.constant(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )

        if eps is None:
            eps = tfd.Normal(
                tf.cast(0.0, TENSORFLOW_DTYPE),
                tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE),
            ).sample(sample_shape=(batch_size, n_samples))
        z_curr_bc = tf.broadcast_to(z_curr, (batch_size, n_samples))
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )

        k_prime_bc = tf.broadcast_to(k_prime, (batch_size, n_samples))
        b_prime_bc = tf.broadcast_to(b_prime, (batch_size, n_samples))
        flat_shape = (batch_size * n_samples, 1)

        k_flat = tf.reshape(k_prime_bc, flat_shape)
        b_flat = tf.reshape(b_prime_bc, flat_shape)
        z_flat = tf.reshape(z_prime, flat_shape)

        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(k_flat),
            self.normalizer.normalize_debt(b_flat),
            self.normalizer.normalize_productivity(z_flat),
        ], axis=1)

        # Default from branch selection net (3rd output)
        branch_probs_mc = self.branch_selection_net(inputs_eval, training=False)
        is_default = tf.cast(branch_probs_mc[:, 2:3] > 0.5, TENSORFLOW_DTYPE)

        profit_next = one_minus_tax * ProductionFunctions.cobb_douglas(
            k_flat, z_flat, self.params
        )
        recovery = BondPricingCalculator.recovery_value(
            profit_next, k_flat, self.params
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
            risk_free_price_val=1.0 / (1.0 + self.params.risk_free_rate),
        )
        return bond_price

    @tf.function(reduce_retracing=True)
    def _get_policy_action(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
               tf.Tensor, tf.Tensor]:
        """Get next-period capital, debt, equity, and branch decision (3-branch).

        Branch indices: 0=invest, 1=wait, 2=default.

        Returns:
            Tuple of (k_prime, b_prime, branch_idx, default_decision,
                      equity_issuance, equity_prob).
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        b_2d = tf.reshape(b_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))

        inputs = self._prepare_inputs(k_2d, b_2d, z_2d)

        one_minus_delta = tf.constant(
            1.0 - self.params.depreciation_rate, TENSORFLOW_DTYPE
        )
        k_prime_no_invest = k_2d * one_minus_delta

        # --- Invest branch: [k', b', equity_raw] ---
        invest_raw = self.invest_policy_net(inputs, training=False)
        k_prime_invest = self.normalizer.denormalize_capital(
            tf.math.sigmoid(invest_raw[:, 0:1])
        )
        b_prime_invest = self.normalizer.denormalize_debt(
            tf.math.sigmoid(invest_raw[:, 1:2])
        )
        equity_raw_invest = invest_raw[:, 2:3]
        _gate_inv = tf.math.sigmoid(equity_raw_invest)
        equity_prob_invest = _gate_inv + tf.stop_gradient(
            tf.cast(_gate_inv > 0.5, TENSORFLOW_DTYPE) - _gate_inv
        )
        equity_issuance_invest = tf.math.softplus(
            self._eq_sp_mul * equity_raw_invest
        ) / self._eq_sp_div

        # --- Wait branch: [b', equity_raw] ---
        wait_raw = self.wait_policy_net(inputs, training=False)
        b_prime_wait = self.normalizer.denormalize_debt(
            tf.math.sigmoid(wait_raw[:, 0:1])
        )
        equity_raw_wait = wait_raw[:, 1:2]
        _gate_wait = tf.math.sigmoid(equity_raw_wait)
        equity_prob_wait = _gate_wait + tf.stop_gradient(
            tf.cast(_gate_wait > 0.5, TENSORFLOW_DTYPE) - _gate_wait
        )
        equity_issuance_wait = tf.math.softplus(
            self._eq_sp_mul * equity_raw_wait
        ) / self._eq_sp_div

        # Branch selection: argmax of softmax (0=invest, 1=wait, 2=default)
        branch_probs = self.branch_selection_net(inputs, training=False)
        branch_idx_3way = tf.argmax(branch_probs, axis=1, output_type=tf.int32)

        # Default decision: branch_idx == 2
        default_selected = tf.cast(
            tf.equal(branch_idx_3way, 2), TENSORFLOW_DTYPE
        )

        # For policy selection, map default (2) to wait (1) for k'/b' gathering
        operating_branch_idx = tf.minimum(branch_idx_3way, 1)

        # Stack and gather selected outputs
        k_prime_all = tf.concat([k_prime_invest, k_prime_no_invest], axis=1)
        b_prime_all = tf.concat([b_prime_invest, b_prime_wait], axis=1)
        equity_issuance_all = tf.concat([
            equity_issuance_invest, equity_issuance_wait,
        ], axis=1)
        equity_prob_all = tf.concat([
            equity_prob_invest, equity_prob_wait,
        ], axis=1)

        batch_indices = tf.range(tf.shape(k_2d)[0])
        k_prime_selected = tf.gather_nd(
            k_prime_all,
            tf.stack([batch_indices, operating_branch_idx], axis=1),
        )
        b_prime_selected = tf.gather_nd(
            b_prime_all,
            tf.stack([batch_indices, operating_branch_idx], axis=1),
        )
        equity_issuance_selected = tf.gather_nd(
            equity_issuance_all,
            tf.stack([batch_indices, operating_branch_idx], axis=1),
        )
        equity_prob_selected = tf.gather_nd(
            equity_prob_all,
            tf.stack([batch_indices, operating_branch_idx], axis=1),
        )

        k_prime_selected = tf.clip_by_value(
            k_prime_selected,
            clip_value_min=self.config.capital_min,
            clip_value_max=self.config.capital_max,
        )
        b_prime_selected = tf.clip_by_value(
            b_prime_selected,
            clip_value_min=self.config.debt_min,
            clip_value_max=self.config.debt_max,
        )

        return (
            k_prime_selected,
            b_prime_selected,
            tf.cast(branch_idx_3way, TENSORFLOW_DTYPE),
            default_selected,
            equity_issuance_selected,
            equity_prob_selected,
        )

    @tf.function(reduce_retracing=True)
    def _get_continuous_value(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
    ) -> tf.Tensor:
        """Get continuation value estimate V_cont(s)."""
        k_2d = tf.reshape(k_curr, (-1, 1))
        b_2d = tf.reshape(b_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))

        inputs = self._prepare_inputs(k_2d, b_2d, z_2d)
        v_cont = self.continuous_net(inputs, training=False)
        return tf.squeeze(v_cont, axis=-1)

    @tf.function(reduce_retracing=True)
    def _get_value(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
    ) -> tf.Tensor:
        """Get value estimate V(s) = V_cont(s) * (1 - default_prob(s))."""
        k_2d = tf.reshape(k_curr, (-1, 1))
        b_2d = tf.reshape(b_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))

        inputs = self._prepare_inputs(k_2d, b_2d, z_2d)
        v_cont = self.continuous_net(inputs, training=False)
        branch_probs = self.branch_selection_net(inputs, training=False)
        default_prob = branch_probs[:, 2:3]
        v_value = v_cont * (1.0 - default_prob)
        return tf.squeeze(v_value, axis=-1)

    @tf.function(reduce_retracing=True)
    def _estimate_bond_price(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_curr: tf.Tensor,
        mc_samples: Optional[int] = None,
    ) -> tf.Tensor:
        """Estimate bond price using branch selection net for default determination."""
        if mc_samples is None:
            mc_samples = self.config.mc_sample_number_bond_priceing

        batch_size = tf.shape(k_prime)[0]
        one_minus_tax = tf.cast(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )

        eps = tfd.Normal(0.0, self.params.productivity_std_dev).sample(
            [batch_size, mc_samples]
        )
        z_curr_bc = tf.broadcast_to(
            tf.reshape(z_curr, (-1, 1)), (batch_size, mc_samples)
        )
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )

        k_prime_bc = tf.broadcast_to(
            tf.reshape(k_prime, (-1, 1)), (batch_size, mc_samples)
        )
        b_prime_bc = tf.broadcast_to(
            tf.reshape(b_prime, (-1, 1)), (batch_size, mc_samples)
        )
        flat_shape = (batch_size * mc_samples, 1)

        k_flat = tf.reshape(k_prime_bc, flat_shape)
        b_flat = tf.reshape(b_prime_bc, flat_shape)
        z_flat = tf.reshape(z_prime, flat_shape)

        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(k_flat),
            self.normalizer.normalize_debt(b_flat),
            self.normalizer.normalize_productivity(z_flat),
        ], axis=1)

        # Default from branch selection net (3rd output)
        branch_probs_mc = self.branch_selection_net(inputs_eval, training=False)
        is_default = tf.cast(branch_probs_mc[:, 2:3] > 0.5, TENSORFLOW_DTYPE)

        profit_next = one_minus_tax * ProductionFunctions.cobb_douglas(
            k_flat, z_flat, self.params
        )
        recovery = BondPricingCalculator.recovery_value(
            profit_next, k_flat, self.params
        )
        payoff = BondPricingCalculator.bond_payoff(
            recovery, b_flat, is_default
        )

        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, mc_samples)),
            axis=1,
            keepdims=True,
        )

        b_prime_2d = tf.reshape(b_prime, (-1, 1))
        bond_price_2d = BondPricingCalculator.risk_neutral_price(
            expected_payoff,
            b_prime_2d,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price,
            risk_free_price_val=1.0 / (1.0 + self.params.risk_free_rate),
        )

        return tf.squeeze(bond_price_2d, axis=-1)

    # ------------------------------------------------------------------
    # Simulation (fused single-step for tf.function)
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _simulate_step(
        self,
        k_sim: tf.Tensor,
        b_sim: tf.Tensor,
        z_sim: tf.Tensor,
        alive_mask: tf.Tensor,
        default_threshold: tf.Tensor,
    ) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
        tf.Tensor, tf.Tensor, tf.Tensor,
    ]:
        """Execute one simulation step entirely in TensorFlow (3-branch).

        Returns:
            Tuple of (k_next, b_next, branch_idx, newly_alive_mask,
                      bond_price, dividend, equity_issuance).
        """
        (
            k_next, b_next, branch_idx, default_decision,
            equity_issuance_sel, equity_prob_sel,
        ) = self._get_policy_action(k_sim, b_sim, z_sim)

        newly_defaulted = default_decision * alive_mask
        newly_alive_mask = alive_mask * (
            tf.ones_like(newly_defaulted) - newly_defaulted
        )

        one_minus_tax = tf.cast(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )
        eta_0 = tf.cast(self.eta_0, TENSORFLOW_DTYPE)
        eta_1 = tf.cast(self.eta_1, TENSORFLOW_DTYPE)

        q_selected = self._estimate_bond_price(k_next, b_next, z_sim)

        k_2d = tf.reshape(k_sim, (-1, 1))
        b_2d = tf.reshape(b_sim, (-1, 1))
        z_2d = tf.reshape(z_sim, (-1, 1))
        k_next_2d = tf.reshape(k_next, (-1, 1))
        b_next_2d = tf.reshape(b_next, (-1, 1))
        q_2d = tf.reshape(q_selected, (-1, 1))
        equity_issuance_2d = tf.reshape(equity_issuance_sel, (-1, 1))
        equity_prob_2d = tf.reshape(equity_prob_sel, (-1, 1))

        revenue = one_minus_tax * ProductionFunctions.cobb_douglas(
            k_2d, z_2d, self.params
        )

        debt_inflow, tax_shield = DebtFlowCalculator.calculate(
            b_next_2d, q_2d, self.params
        )

        investment = ProductionFunctions.calculate_investment(
            k_2d, k_next_2d, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_2d, self.params
        )

        payout_invest = revenue + debt_inflow + tax_shield - adj_cost - investment - b_2d
        payout_wait = revenue + debt_inflow + tax_shield - b_2d

        issuance_cost = equity_prob_2d * (eta_0 + eta_1 * equity_issuance_2d)

        dividend_invest = payout_invest - issuance_cost
        dividend_wait = payout_wait - issuance_cost

        dividends_all = tf.concat([dividend_invest, dividend_wait], axis=1)

        batch_indices = tf.range(tf.shape(k_2d)[0])
        # Clamp branch_idx to [0, 1] for dividend gathering
        # (default branch=2 mapped to wait=1; defaulted firms are masked out)
        operating_idx = tf.minimum(tf.cast(branch_idx, tf.int32), 1)
        dividend_selected = tf.gather_nd(
            dividends_all,
            tf.stack([batch_indices, operating_idx], axis=1),
        )

        gated_equity_issuance = equity_issuance_sel * equity_prob_sel

        return (
            k_next, b_next, branch_idx, newly_alive_mask,
            q_selected, dividend_selected, gated_equity_issuance,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_states: Tuple[tf.Tensor, ...],
        innovation_sequence: tf.Tensor,
        default_threshold: float = 0.0,
    ) -> dict:
        """Simulate the risky debt economy using neural network policies.

        Args:
            initial_states: Tuple (K_0, B_0, Z_0) or (K_0, Z_0).
            innovation_sequence: Tensor of shape (batch_size, T+1) with
                standard normal innovations.
            default_threshold: Unused, retained for API compatibility.

        Returns:
            Dictionary with simulation results (aligned with VFI simulator).
        """
        if len(initial_states) == 2:
            k_curr, z_curr = initial_states
            b_curr = tf.zeros_like(k_curr)
        elif len(initial_states) == 3:
            k_curr, b_curr, z_curr = initial_states
        else:
            raise ValueError(
                f"Expected 2 or 3 initial states, got {len(initial_states)}"
            )

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
            innovation_sequence * self.params.productivity_std_dev,
            TENSORFLOW_DTYPE,
        )

        batch_size = eps_path.shape[0]
        T = eps_path.shape[1] - 1
        default_threshold_tf = tf.constant(default_threshold, TENSORFLOW_DTYPE)

        def _make_ta(name: str) -> tf.TensorArray:
            return tf.TensorArray(
                dtype=TENSORFLOW_DTYPE, size=T,
                dynamic_size=False, clear_after_read=True,
                element_shape=(batch_size,),
            )

        ta_K_curr = _make_ta("K_curr")
        ta_K_next = _make_ta("K_next")
        ta_B_curr = _make_ta("B_curr")
        ta_B_next = _make_ta("B_next")
        ta_Z_curr = _make_ta("Z_curr")
        ta_Z_next = _make_ta("Z_next")
        ta_dividend = _make_ta("dividend")
        ta_equity = _make_ta("equity")
        ta_bond_price = _make_ta("bond_price")
        ta_branch_idx = _make_ta("branch_idx")
        ta_alive = _make_ta("alive")

        k_sim = k_curr
        b_sim = b_curr
        z_sim = z_curr
        alive_mask = tf.ones((batch_size,), dtype=TENSORFLOW_DTYPE)

        for t in range(T):
            ta_K_curr = ta_K_curr.write(t, k_sim)
            ta_B_curr = ta_B_curr.write(t, b_sim)
            ta_Z_curr = ta_Z_curr.write(t, z_sim)
            ta_alive = ta_alive.write(t, alive_mask)

            (
                k_next, b_next, branch_idx, alive_mask,
                bond_price, dividend, equity_issuance,
            ) = self._simulate_step(
                k_sim, b_sim, z_sim, alive_mask, default_threshold_tf,
            )

            ta_K_next = ta_K_next.write(t, k_next)
            ta_B_next = ta_B_next.write(t, b_next)
            ta_branch_idx = ta_branch_idx.write(t, branch_idx)
            ta_bond_price = ta_bond_price.write(t, bond_price)
            ta_dividend = ta_dividend.write(t, dividend)
            ta_equity = ta_equity.write(t, equity_issuance)

            z_next = TransitionFunctions.log_ar1_transition(
                z_curr=z_sim,
                rho=self.params.productivity_persistence,
                epsilon=eps_path[:, t + 1],
            )
            ta_Z_next = ta_Z_next.write(t, z_next)

            k_sim = k_next
            b_sim = b_next
            z_sim = z_next

        def _stack_transpose(ta: tf.TensorArray) -> np.ndarray:
            return tf.transpose(ta.stack(), [1, 0]).numpy()

        K_curr_out = _stack_transpose(ta_K_curr)
        K_next_out = _stack_transpose(ta_K_next)
        B_curr_out = _stack_transpose(ta_B_curr)
        B_next_out = _stack_transpose(ta_B_next)
        Z_curr_out = _stack_transpose(ta_Z_curr)
        Z_next_out = _stack_transpose(ta_Z_next)
        dividend_out = _stack_transpose(ta_dividend)
        equity_issuance_out = _stack_transpose(ta_equity)
        bond_price_out = _stack_transpose(ta_bond_price)
        branch_idx_out = _stack_transpose(ta_branch_idx)
        alive_out = _stack_transpose(ta_alive)

        dead_mask = alive_out == 0.0
        for arr in (
            K_curr_out, K_next_out, B_curr_out, B_next_out,
            Z_curr_out, Z_next_out, dividend_out, equity_issuance_out,
            bond_price_out, branch_idx_out,
        ):
            arr[dead_mask] = np.nan

        return {
            "K_curr": K_curr_out,
            "K_next": K_next_out,
            "B_curr": B_curr_out,
            "B_next": B_next_out,
            "Z_curr": Z_curr_out,
            "Z_next": Z_next_out,
            "dividend": dividend_out,
            "equity_issuance": equity_issuance_out,
            "bond_price": bond_price_out,
            "branch_idx": branch_idx_out,
        }

    # ------------------------------------------------------------------
    # Bellman residual diagnostic
    # ------------------------------------------------------------------

    def simulate_bellman_residual(
        self,
        initial_states: Tuple[tf.Tensor, ...],
    ) -> dict:
        """Compute Bellman equation residuals on a set of states.

        Returns:
            Dictionary with relative_error, absolute_error, mean_value.
        """
        if len(initial_states) == 2:
            k_curr, z_curr = initial_states
            b_curr = tf.zeros_like(k_curr)
        elif len(initial_states) == 3:
            k_curr, b_curr, z_curr = initial_states
        else:
            raise ValueError(
                f"Expected 2 or 3 initial states, got {len(initial_states)}"
            )

        if len(k_curr.shape) > 1:
            k_curr = tf.squeeze(k_curr, axis=-1)
        if len(z_curr.shape) > 1:
            z_curr = tf.squeeze(z_curr, axis=-1)
        if len(b_curr.shape) > 1:
            b_curr = tf.squeeze(b_curr, axis=-1)
        k_curr = tf.cast(k_curr, TENSORFLOW_DTYPE)
        z_curr = tf.cast(z_curr, TENSORFLOW_DTYPE)
        b_curr = tf.cast(b_curr, TENSORFLOW_DTYPE)

        (
            k_next, b_next, branch_idx, default_decision,
            equity_issuance_sel, equity_prob_sel,
        ) = self._get_policy_action(k_curr, b_curr, z_curr)

        one_minus_tax = tf.cast(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )
        eta_0 = tf.cast(self.eta_0, TENSORFLOW_DTYPE)
        eta_1 = tf.cast(self.eta_1, TENSORFLOW_DTYPE)

        q_selected = self._estimate_bond_price(k_next, b_next, z_curr)

        revenue = one_minus_tax * ProductionFunctions.cobb_douglas(
            k_curr, z_curr, self.params
        )
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(
            b_next, q_selected, self.params
        )
        debt_inflow = tf.squeeze(debt_inflow, axis=-1) if len(debt_inflow.shape) > 1 else debt_inflow
        tax_shield = tf.squeeze(tax_shield, axis=-1) if len(tax_shield.shape) > 1 else tax_shield

        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr, self.params
        )

        payout_invest = revenue + debt_inflow + tax_shield - adj_cost - investment - b_curr
        payout_wait = revenue + debt_inflow + tax_shield - b_curr

        issuance_cost = equity_prob_sel * (eta_0 + eta_1 * equity_issuance_sel)
        dividend_invest = payout_invest - issuance_cost
        dividend_wait = payout_wait - issuance_cost

        dividends_all = tf.stack([dividend_invest, dividend_wait], axis=1)
        batch_indices = tf.range(tf.shape(k_curr)[0])
        operating_idx = tf.minimum(tf.cast(branch_idx, tf.int32), 1)
        cash_flow = tf.gather_nd(
            dividends_all,
            tf.stack([batch_indices, operating_idx], axis=1),
        )

        mc_num = self.config.mc_sample_number_bond_priceing
        eps = tfd.Normal(0.0, self.params.productivity_std_dev).sample(
            [tf.shape(k_curr)[0], mc_num]
        )

        z_curr_expanded = tf.repeat(
            tf.expand_dims(z_curr, axis=1), mc_num, axis=1
        )
        z_next = TransitionFunctions.log_ar1_transition(
            z_curr_expanded, self.params.productivity_persistence, eps
        )

        k_next_expanded = tf.repeat(
            tf.expand_dims(k_next, axis=1), mc_num, axis=1
        )
        b_next_expanded = tf.repeat(
            tf.expand_dims(b_next, axis=1), mc_num, axis=1
        )

        k_next_flat = tf.reshape(k_next_expanded, [-1])
        b_next_flat = tf.reshape(b_next_expanded, [-1])
        z_next_flat = tf.reshape(z_next, [-1])

        v_cont_next_flat = self._get_continuous_value(
            k_next_flat, b_next_flat, z_next_flat
        )
        # V = V_cont * (1 - default_prob) from branch selection net
        v_next_flat = self._get_value(
            k_next_flat, b_next_flat, z_next_flat
        )
        expected_value = tf.reshape(
            v_next_flat, [tf.shape(k_curr)[0], mc_num]
        )
        expected_value = tf.reduce_mean(expected_value, axis=1)

        rhs = cash_flow + self.params.discount_factor * expected_value
        v_curr = self._get_value(k_curr, b_curr, z_curr)
        residuals = rhs - v_curr
        rel_error = tf.abs(residuals) / (tf.abs(rhs) + 1e-8)
        abs_residual = tf.reduce_mean(tf.abs(residuals))

        return {
            "relative_error": float(tf.reduce_mean(rel_error)),
            "absolute_error": float(abs_residual),
            "mean_value": float(tf.reduce_mean(v_curr)),
        }

    # ------------------------------------------------------------------
    # Lifetime reward
    # ------------------------------------------------------------------

    def simulate_life_time_reward(
        self,
        initial_states: Tuple[tf.Tensor, ...],
        innovation_sequence: tf.Tensor,
        default_threshold: float = 0.0,
    ) -> tf.Tensor:
        """Compute discounted lifetime reward from simulation.

        Returns:
            Mean discounted lifetime reward across batch.
        """
        if len(initial_states) == 2:
            k_curr, z_curr = initial_states
            b_curr = tf.zeros_like(k_curr)
        elif len(initial_states) == 3:
            k_curr, b_curr, z_curr = initial_states
        else:
            raise ValueError(
                f"Expected 2 or 3 initial states, got {len(initial_states)}"
            )

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
            innovation_sequence * self.params.productivity_std_dev,
            TENSORFLOW_DTYPE,
        )

        batch_size = eps_path.shape[0]
        T = eps_path.shape[1] - 1

        one_minus_tax = tf.cast(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )
        beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)

        accumulated_reward = tf.zeros(
            shape=(batch_size,), dtype=TENSORFLOW_DTYPE
        )
        has_defaulted = tf.zeros(
            shape=(batch_size,), dtype=TENSORFLOW_DTYPE
        )

        k_t = k_curr
        z_t = z_curr
        b_t = b_curr

        for t in range(T):
            (
                k_next, b_next, branch_idx, default_decision,
                equity_issuance_sel, equity_prob_sel,
            ) = self._get_policy_action(k_t, b_t, z_t)

            has_defaulted = tf.maximum(has_defaulted, default_decision)
            alive_mask = 1.0 - has_defaulted

            q_selected = self._estimate_bond_price(k_next, b_next, z_t)

            revenue = one_minus_tax * ProductionFunctions.cobb_douglas(
                k_t, z_t, self.params
            )

            debt_inflow, tax_shield = DebtFlowCalculator.calculate(
                b_next, q_selected, self.params
            )

            investment = ProductionFunctions.calculate_investment(
                k_t, k_next, self.params
            )
            adj_cost, _ = AdjustmentCostCalculator.calculate(
                investment, k_t, self.params
            )

            payout_invest = revenue + debt_inflow + tax_shield - adj_cost - investment - b_t
            payout_wait = revenue + debt_inflow + tax_shield - b_t

            eta_0_t = tf.cast(self.eta_0, TENSORFLOW_DTYPE)
            eta_1_t = tf.cast(self.eta_1, TENSORFLOW_DTYPE)
            issuance_cost = equity_prob_sel * (eta_0_t + eta_1_t * equity_issuance_sel)

            dividend_invest = payout_invest - issuance_cost
            dividend_wait = payout_wait - issuance_cost

            dividends_all = tf.stack([dividend_invest, dividend_wait], axis=1)
            batch_indices = tf.range(tf.shape(k_t)[0])
            operating_idx = tf.minimum(tf.cast(branch_idx, tf.int32), 1)
            cash_flow = tf.gather_nd(
                dividends_all,
                tf.stack([batch_indices, operating_idx], axis=1),
            )
            cash_flow = alive_mask * cash_flow

            accumulated_reward = accumulated_reward + cash_flow * (beta ** t)

            z_next = TransitionFunctions.log_ar1_transition(
                z_t,
                self.params.productivity_persistence,
                eps_path[:, t + 1],
            )
            k_t = k_next
            z_t = z_next
            b_t = b_next

        return tf.reduce_mean(accumulated_reward)

    # ------------------------------------------------------------------
    # Value function gap
    # ------------------------------------------------------------------

    def compute_value_function_gap(
        self,
        grid_points: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        value_labels: tf.Tensor,
    ) -> dict[str, float]:
        """Compute the gap between predicted and true value function on a grid.

        Returns:
            Dictionary containing 'mae' and 'mape'.
        """
        k_grid, b_grid, z_grid = grid_points

        k_curr = tf.reshape(tf.cast(k_grid, TENSORFLOW_DTYPE), [-1])
        b_curr = tf.reshape(tf.cast(b_grid, TENSORFLOW_DTYPE), [-1])
        z_curr = tf.reshape(tf.cast(z_grid, TENSORFLOW_DTYPE), [-1])
        value_labels_flat = tf.reshape(
            tf.cast(value_labels, TENSORFLOW_DTYPE), [-1]
        )

        v_cont = self._get_continuous_value(k_curr, b_curr, z_curr)
        # V = V_cont * (1 - default_prob) from branch selection net
        value_pred = self._get_value(k_curr, b_curr, z_curr)
        value_pred = tf.reshape(value_pred, [-1])

        residual = value_labels_flat - value_pred

        mae = tf.reduce_mean(tf.abs(residual))
        safe_denominator = tf.abs(value_labels_flat) + 1e-4
        mape = tf.reduce_mean(tf.abs(residual) / safe_denominator)

        return {
            "mae": float(mae.numpy()),
            "mape": float(mape.numpy()),
        }
