# src/econ_models/simulator/dl/risky_final.py
"""
Deep Learning Simulator for the Risky Debt Model with Investment Decision.

Simulates the economy with endogenous default using trained neural
network policy functions for capital, debt, investment, and default
decisions.

Shared debt policy / separate issuance architecture matching RiskyModelDL_FINAL:
- Capital Policy: sigmoid -> denormalize to K' in [k_min, k_max]
- Debt Policy: sigmoid -> denormalize to B' in [b_min, b_max] (shared across paths)
- Investment Decision: prob(invest) in [0, 1]
- Default Policy: prob(default) in [0, 1]
- Equity Issuance (invest): linear net -> inline relu (value) & sigmoid (gate)
- Equity Issuance (noinvest): linear net -> inline relu (value) & sigmoid (gate)
- Value Network V(s) (linear activation)

Decision hierarchy:
1. Investment: invest_prob > threshold -> k_target, else (1-delta)k
2. Debt: shared network for both paths
3. Equity: separate networks for invest and noinvest paths
4. Default: if d_prob > threshold, firm defaults -> NaN-masked

Bond price estimated via MC sampling of default probability.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    DebtFlowCalculator,
    BondPricingCalculator,
)
import tensorflow_probability as tfp
tfd = tfp.distributions


class DLSimulatorRiskyFinal:
    """Simulator for the risky debt model with shared debt / separate issuance.
    
    Uses trained neural networks matching the RiskyModelDL_FINAL training
    architecture:

    Policy networks:
    - Capital Policy: sigmoid -> denormalize to K' in [k_min, k_max]
    - Debt Policy: sigmoid -> denormalize to B' in [b_min, b_max] (shared)

    Decision networks:
    - Investment Decision: prob(invest) for binary invest/wait decision
    - Default Policy: prob(default) for binary default decision
    - Equity Issuance (invest): linear -> inline relu (value) & sigmoid (gate)
    - Equity Issuance (noinvest): linear -> inline relu (value) & sigmoid (gate)
    - Value Network: V(k, b, z) for diagnostics and Bellman residuals
    
    Attributes:
        config: Deep learning configuration.
        params: Economic parameters.
        normalizer: State space normalizer for network inputs.
        capital_policy_net: Capital choice k' (sigmoid -> denorm).
        debt_policy_net: Debt choice b' shared across paths (sigmoid -> denorm).
        investment_policy_net: Investment probability network.
        default_policy_net: Default probability network.
        equity_issuance_invest_net: Equity issuance for invest path (linear output).
        equity_issuance_noinvest_net: Equity issuance for noinvest path (linear output).
        value_function_net: Value network V(s) for diagnostics.
        hard_threshold: Threshold for binary decisions.
    """
    
    def __init__(
        self,
        config: DeepLearningConfig,
        params: EconomicParams,
        hard_threshold: float = 0.5,
    ):
        """Initialize the DL simulator.
        
        Args:
            config: Deep learning configuration.
            params: Economic parameters.
            hard_threshold: Threshold for binary investment/default decisions.
        """
        self.config = config
        self.params = params
        self.normalizer = StateSpaceNormalizer(config)
        self.hard_threshold = hard_threshold
        self.config.update_value_scale(self.params)
        
        self.capital_policy_net: Optional[tf.keras.Model] = None
        self.debt_policy_net: Optional[tf.keras.Model] = None
        self.investment_policy_net: Optional[tf.keras.Model] = None
        self.default_policy_net: Optional[tf.keras.Model] = None
        self.equity_issuance_invest_net: Optional[tf.keras.Model] = None
        self.equity_issuance_noinvest_net: Optional[tf.keras.Model] = None
        self.value_function_net: Optional[tf.keras.Model] = None
        
        # Equity issuance cost constants (inline, matching training)
        self.eta_0 = self.params.equity_issuance_cost_fixed
        self.eta_1 = self.params.equity_issuance_cost_linear
        
        self._build_networks()
    
    def _build_networks(self) -> None:
        """Build all network architectures matching the training code.
        
        All policy networks use input_dim=3 for (K, B, Z) state.
        Activations match RiskyModelDL_FINAL:
        - Capital: sigmoid (denormalized to physical units)
        - Debt: sigmoid (denormalized to physical units, shared across paths)
        - Investment/Default: sigmoid
        - Equity Issuance invest: linear (relu & sigmoid computed inline)
        - Equity Issuance noinvest: linear (relu & sigmoid computed inline)
        - Value: linear
        """
        # Capital Policy Network -> sigmoid in (0,1) -> denorm to [k_min, k_max]
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="CapitalPolicyNet",
        )
        
        # Debt Policy Network (shared) -> sigmoid in (0,1) -> denorm to [b_min, b_max]
        self.debt_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="DebtPolicyNet",
        )
        
        # Investment Decision Network -> prob(invest)
        self.investment_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="InvestmentNet",
        )
        
        # Default Policy Network -> prob(default)
        self.default_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="DefaultPolicyNet",
        )
        
        # Equity Issuance Network (invest) -> linear output x
        #   issuance_value = relu(x), gate = sigmoid(x), computed inline
        self.equity_issuance_invest_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="EquityIssuanceNetInvest",
        )
        
        # Equity Issuance Network (noinvest) -> linear output x
        #   issuance_value = relu(x), gate = sigmoid(x), computed inline
        self.equity_issuance_noinvest_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="EquityIssuanceNetNoinvest",
        )
        
        # Value Function Network V(s)
        self.value_function_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueNet",
        )
    
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
        """Load all trained policy networks from weights files.
        
        Matches the shared debt checkpoint naming from
        save_checkpoint_risky_final in the training code.
        
        Args:
            capital_policy_filepath: Path to capital .weights.h5.
            debt_filepath: Path to shared debt policy .weights.h5.
            investment_policy_filepath: Path to investment policy .weights.h5.
            default_policy_filepath: Path to default policy .weights.h5.
            value_function_filepath: Path to value function .weights.h5.
            equity_issuance_invest_filepath: Path to invest equity .weights.h5.
            equity_issuance_noinvest_filepath: Path to noinvest equity .weights.h5.
            **kwargs: Ignored (absorbs legacy params).
        """
        dummy_input = tf.zeros((1, 3), dtype=TENSORFLOW_DTYPE)
        
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
    
    def _prepare_inputs(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
    ) -> tf.Tensor:
        """Prepare normalized inputs for neural networks.
        
        Matches training code's _prepare_inputs method.
        
        Args:
            k: Capital tensor, shape (batch_size, 1) or (batch_size,).
            b: Debt tensor, shape (batch_size, 1) or (batch_size,).
            z: Productivity tensor, shape (batch_size, 1) or (batch_size,).
            
        Returns:
            Concatenated normalized inputs, shape (batch_size, 3).
        """
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
    
    @tf.function
    def _get_policy_action(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get next-period capital, debt, and decision outputs.
        
        Shared debt architecture: single debt network for both paths.
        Separate equity issuance networks for invest and noinvest paths.
        The investment decision selects which equity branch to use.
        
        1. Capital: sigmoid -> denormalize to k_target
        2. Investment: invest_prob > threshold -> k_target, else (1-delta)k
        3. Debt: shared network (same B' for both paths)
        4. Equity: invest branch or noinvest branch based on decision
        
        Args:
            k_curr: Current capital, shape (batch_size,).
            b_curr: Current debt, shape (batch_size,).
            z_curr: Current productivity, shape (batch_size,).
            
        Returns:
            Tuple of:
                - k_prime: Next-period capital, shape (batch_size,)
                - b_prime: Next-period debt, shape (batch_size,)
                - invest_prob: Investment probability, shape (batch_size,)
                - default_prob: Default probability, shape (batch_size,)
                - equity_issuance: Effective equity issuance, shape (batch_size,)
                - issuance_gate_prob: Issuance gate probability, shape (batch_size,)
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        b_2d = tf.reshape(b_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))
        
        inputs = self._prepare_inputs(k_2d, b_2d, z_2d)
        
        # --- Capital policy: sigmoid -> denormalize ---
        k_target = self.normalizer.denormalize_capital(
            self.capital_policy_net(inputs, training=False)
        )
        
        # --- Shared debt policy ---
        b_prime = self.normalizer.denormalize_debt(
            self.debt_policy_net(inputs, training=False)
        )
        
        # --- Decision nets ---
        invest_prob = self.investment_policy_net(inputs, training=False)
        default_prob = self.default_policy_net(inputs, training=False)
        
        # --- Separate equity issuance per branch (hard gate matching training STE) ---
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

        # Investment decision: hard threshold -> select capital and equity path
        # Debt is shared (same B' regardless of invest decision)
        k_prime_no_invest = k_2d * (1.0 - self.params.depreciation_rate)
        hard_invest = tf.cast(
            invest_prob > self.hard_threshold, TENSORFLOW_DTYPE,
        )
        k_prime = hard_invest * k_target + (1.0 - hard_invest) * k_prime_no_invest
        equity_issuance = hard_invest * equity_issuance_inv + (1.0 - hard_invest) * equity_issuance_noinv
        issuance_gate_prob = hard_invest * issuance_gate_prob_inv + (1.0 - hard_invest) * issuance_gate_prob_noinv
        
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
    
    @tf.function
    def _get_value_function(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
    ) -> tf.Tensor:
        """Get value function estimate for given states.
        
        Args:
            k_curr: Current capital, shape (batch_size,).
            b_curr: Current debt, shape (batch_size,).
            z_curr: Current productivity, shape (batch_size,).
            
        Returns:
            Value function estimates, shape (batch_size,).
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        b_2d = tf.reshape(b_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))
        
        inputs = self._prepare_inputs(k_2d, b_2d, z_2d)
        v_values = self.value_function_net(inputs, training=False)
        return tf.squeeze(v_values, axis=-1)
    
    @tf.function
    def _estimate_bond_price(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_curr: tf.Tensor,
        mc_samples: Optional[int] = None,
    ) -> tf.Tensor:
        """Estimate bond price via MC sampling of default probability.
        
        Matches the training code's estimate_bond_price method using
        centralized econ modules:
        - Sample future shocks -> z'
        - Evaluate default policy at (k', b', z')
        - Compute recovery value via BondPricingCalculator
        - Risk-neutral pricing via BondPricingCalculator
        
        Args:
            k_prime: Next-period capital, shape (batch_size,).
            b_prime: Next-period debt, shape (batch_size,).
            z_curr: Current productivity, shape (batch_size,).
            mc_samples: Number of Monte Carlo samples. Defaults to
                config.mc_sample_number_bond_priceing.
            
        Returns:
            Bond price q, shape (batch_size,).
        """
        if mc_samples is None:
            mc_samples = self.config.mc_sample_number_bond_priceing

        batch_size = tf.shape(k_prime)[0]
        one_minus_tax = tf.cast(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )
        
        # 1. Sample future shocks and transition Z
        eps = tfd.Normal(0.0, self.params.productivity_std_dev).sample(
            [batch_size, mc_samples]
        )
        z_curr_bc = tf.broadcast_to(
            tf.reshape(z_curr, (-1, 1)), (batch_size, mc_samples)
        )
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )
        
        # 2. Prepare broadcasted inputs
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
        
        # 3. Get default probability (single batched forward pass)
        is_default = self.default_policy_net(inputs_eval, training=False)
        
        # 4. Recovery and payoff via centralized econ modules
        profit_next = one_minus_tax * ProductionFunctions.cobb_douglas(
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
            tf.reshape(payoff, (batch_size, mc_samples)),
            axis=1,
        )
        
        # 6. Risk-neutral pricing via centralized module
        b_prime_2d = tf.reshape(b_prime, (-1, 1))
        expected_payoff_2d = tf.reshape(expected_payoff, (-1, 1))
        bond_price_2d = BondPricingCalculator.risk_neutral_price(
            expected_payoff_2d,
            b_prime_2d,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price,
            risk_free_price_val=1.0 / (1.0 + self.params.risk_free_rate),
        )
        
        return tf.squeeze(bond_price_2d, axis=-1)
    
    def simulate(
        self,
        initial_states: Tuple[tf.Tensor, ...],
        innovation_sequence: tf.Tensor,
        default_threshold: float = 0.5,
    ) -> dict:
        """Simulate the risky debt economy using neural network policies.
        
        Uses TensorArray-based loop matching the basic model simulator pattern.
        Handles default events by NaN-masking dead firms (matching VFI convention).
        Computes equity issuance via MC bond pricing.
        
        Args:
            initial_states: Tuple (K_0, B_0, Z_0) or (K_0, Z_0) of initial states.
                Convention: (K, B, Z) for 3-element, (K, Z) for 2-element.
            innovation_sequence: Tensor of shape (batch_size, T+1) with
                standard normal innovations (will be scaled by sigma).
            default_threshold: Probability threshold for default decision.
            
        Returns:
            Dictionary with simulation results (aligned with VFI simulator):
                K_curr, K_next, B_curr, B_next, Z_curr, Z_next,
                equity_issuance.
        """
        # Parse initial states: convention is (K, B, Z)
        if len(initial_states) == 2:
            k_curr, z_curr = initial_states
            b_curr = tf.zeros_like(k_curr)
        elif len(initial_states) == 3:
            k_curr, b_curr, z_curr = initial_states
        else:
            raise ValueError(
                f"Expected 2 or 3 initial states, got {len(initial_states)}"
            )
        
        # Ensure 1D shape
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
        
        # Storage arrays (aligned with VFI simulator output)
        K_curr_out = np.full((batch_size, T), np.nan)
        K_next_out = np.full((batch_size, T), np.nan)
        B_curr_out = np.full((batch_size, T), np.nan)
        B_next_out = np.full((batch_size, T), np.nan)
        Z_curr_out = np.full((batch_size, T), np.nan)
        Z_next_out = np.full((batch_size, T), np.nan)
        equity_issuance_out = np.full((batch_size, T), np.nan)
        bond_price_out = np.full((batch_size, T), np.nan)
        
        # Cumulative default mask: once defaulted, always dead (matches VFI)
        has_defaulted = np.zeros(batch_size, dtype=bool)
        
        # Current state as numpy for masking, as tensors for policy calls
        k_sim = k_curr.numpy().copy()
        b_sim = b_curr.numpy().copy()
        z_sim = z_curr.numpy().copy()
        
        # Simulation loop
        for t in range(T):
            # Record current state
            K_curr_out[:, t] = k_sim
            B_curr_out[:, t] = b_sim
            Z_curr_out[:, t] = z_sim
            
            alive = ~has_defaulted
            
            if not np.any(alive):
                # All firms dead — remaining periods are NaN
                break
            
            # Get policy actions for alive firms
            k_t = tf.constant(k_sim, dtype=TENSORFLOW_DTYPE)
            b_t = tf.constant(b_sim, dtype=TENSORFLOW_DTYPE)
            z_t = tf.constant(z_sim, dtype=TENSORFLOW_DTYPE)
            
            k_next_tf, b_next_tf, invest_prob, default_prob, equity_iss_tf, gate_prob_tf = (
                self._get_policy_action(k_t, b_t, z_t)
            )
            
            # Default decision
            default_prob_np = default_prob.numpy()
            newly_default = np.zeros(batch_size, dtype=bool)
            newly_default[alive] = default_prob_np[alive] > default_threshold
            has_defaulted |= newly_default
            alive = ~has_defaulted
            
            # Next-period state (NaN for dead firms, matching VFI)
            k_next_np = np.full(batch_size, np.nan)
            b_next_np = np.full(batch_size, np.nan)
            
            if np.any(alive):
                k_next_np[alive] = k_next_tf.numpy()[alive]
                b_next_np[alive] = b_next_tf.numpy()[alive]
            
            # Bond price via MC sampling of default policy (compute BEFORE payout)
            q_t = np.full(batch_size, np.nan)
            if np.any(alive):
                q_val = self._estimate_bond_price(
                    tf.constant(k_next_np[alive], dtype=TENSORFLOW_DTYPE),
                    tf.constant(b_next_np[alive], dtype=TENSORFLOW_DTYPE),
                    tf.constant(z_sim[alive], dtype=TENSORFLOW_DTYPE),
                )
                q_t[alive] = q_val.numpy()
            bond_price_out[:, t] = q_t

            # Use neural network equity issuance output directly (gated)
            eq_iss_t = np.full(batch_size, np.nan)
            if np.any(alive):
                eq_iss_t[alive] = (equity_iss_tf.numpy() * gate_prob_tf.numpy())[alive]
            equity_issuance_out[:, t] = eq_iss_t

            # # --- Commented out: payout-based equity issuance (VFI convention) ---
            # eq_iss = max(0, -payout) where payout is the operating cash flow
            # eq_iss_t = np.full(batch_size, np.nan)
            # if np.any(alive):
            #     kc = k_sim[alive]
            #     zc = z_sim[alive]
            #     bc = b_sim[alive]
            #     kn = k_next_np[alive]
            #     bn = b_next_np[alive]
            #     qa = q_t[alive]
            
            #     one_minus_tax = 1.0 - self.params.corporate_tax_rate
            #     one_minus_delta = 1.0 - self.params.depreciation_rate
            #     r_f = self.params.risk_free_rate
            
            #     # Revenue: (1-tau) * Z * K^alpha
            #     revenue = one_minus_tax * zc * np.power(kc, self.params.capital_share)
            
            #     # Investment: K' - (1-delta)*K
            #     inv = kn - one_minus_delta * kc
            
            #     # Adjustment cost: psi0/2 * I^2/K + psi1 * K * 1{I!=0}
            #     safe_k = np.maximum(kc, 1e-8)
            #     psi0 = self.params.adjustment_cost_convex
            #     psi1 = self.params.adjustment_cost_fixed
            #     adj_cost = (
            #         psi0 * (inv ** 2) / (2.0 * safe_k)
            #         + psi1 * kc * (np.abs(inv) > 1e-8).astype(np.float32)
            #     )
            
            #     # Debt flows: q*b' and tax shield
            #     debt_inflow = qa * bn
            #     interest = (1.0 - qa) * np.maximum(bn, 0.0)
            #     tax_shield = self.params.corporate_tax_rate * interest / (1.0 + r_f)
            
            #     # Payout (before equity issuance cost)
            #     payout = revenue + debt_inflow + tax_shield - bc - inv - adj_cost
            
            #     eq_iss_t[alive] = np.maximum(0.0, -payout)
            
            # equity_issuance_out[:, t] = eq_iss_t
            
            # Productivity transition
            z_next_tf = TransitionFunctions.log_ar1_transition(
                z_curr=tf.constant(z_sim, dtype=TENSORFLOW_DTYPE),
                rho=self.params.productivity_persistence,
                epsilon=eps_path[:, t + 1],
            )
            z_next_np = z_next_tf.numpy().copy()
            z_next_np[~alive] = np.nan
            
            # Record next state
            K_next_out[:, t] = k_next_np
            B_next_out[:, t] = b_next_np
            Z_next_out[:, t] = z_next_np
            
            # Advance state
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
    
    def simulate_bellman_residual(
        self,
        initial_states: Tuple[tf.Tensor, ...],
    ) -> dict:
        """Compute Bellman equation residuals on a set of states.
        
        For the risky model:
            RHS = dividend + β * E[V(k', b', z')]  (if continuing)
            V(s) should equal RHS at optimum → residual = RHS - V(s)
        
        Uses Monte Carlo integration for the expectation over z'.
        
        Args:
            initial_states: Tuple (K, B, Z) or (K, Z) of state tensors
                to evaluate residuals on.
                Convention: (K, B, Z) for 3-element, (K, Z) for 2-element.
            
        Returns:
            Dictionary with relative_error, absolute_error, mean_value.
        """
        # Parse states: convention is (K, B, Z)
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
        
        # Get policy actions
        k_next, b_next, invest_prob, default_prob, equity_issuance, issuance_gate_prob = (
            self._get_policy_action(k_curr, b_curr, z_curr)
        )
        
        # --- Compute cash flow for continuers ---
        one_minus_tax = tf.cast(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )
        
        # Production via centralized module: (1-tau) * Z * K^theta
        profit = one_minus_tax * ProductionFunctions.cobb_douglas(
            k_curr, z_curr, self.params
        )
        
        # Investment via centralized module: K' - (1-delta)K
        investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, self.params
        )
        
        # Adjustment cost: invest_decision already encoded in k' from
        # _get_policy_action — non-investors have k'=(1-delta)k so investment=0,
        # giving adj_cost=0 automatically
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr, self.params
        )
        
        # Bond pricing
        bond_price = self._estimate_bond_price(k_next, b_next, z_curr)
        
        # Debt flow
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(
            b_next, bond_price, self.params
        )
        debt_inflow = tf.squeeze(debt_inflow, axis=-1) if len(debt_inflow.shape) > 1 else debt_inflow
        tax_shield = tf.squeeze(tax_shield, axis=-1) if len(tax_shield.shape) > 1 else tax_shield
        
        # Payout (before issuance cost)
        payout = profit + debt_inflow + tax_shield - adj_cost - investment - b_curr
        
        # Equity issuance from budget constraint (matching VFI):
        # When payout < 0, the firm must issue equity to cover the shortfall.
        # equity_issuance = max(0, -payout)
        # issuance_cost = (eta_0 + eta_1 * equity_iss) * 1{payout < 0}
        equity_iss = tf.nn.relu(-payout)
        needs_issuance = tf.cast(payout < 0.0, TENSORFLOW_DTYPE)
        issuance_cost = needs_issuance * (self.eta_0 + self.eta_1 * equity_iss)
        cash_flow = payout - issuance_cost
        
        # --- Monte Carlo expectation of V(k', b', z') ---
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
        
        expected_value_flat = self._get_value_function(
            k_next_flat, b_next_flat, z_next_flat
        )
        expected_value = tf.reshape(
            expected_value_flat, [tf.shape(k_curr)[0], mc_num]
        )
        expected_value = tf.reduce_mean(expected_value, axis=1)
        
        rhs = cash_flow + self.params.discount_factor * expected_value
        v_curr = self._get_value_function(k_curr, b_curr, z_curr)
        residuals = rhs - v_curr
        rel_error = tf.abs(residuals) / (tf.abs(rhs) + 1e-8)
        abs_residual = tf.reduce_mean(tf.abs(residuals))
        
        return {
            "relative_error": float(tf.reduce_mean(rel_error)),
            "absolute_error": float(abs_residual),
            "mean_value": float(tf.reduce_mean(v_curr)),
        }
    
    def simulate_life_time_reward(
        self,
        initial_states: Tuple[tf.Tensor, ...],
        innovation_sequence: tf.Tensor,
        default_threshold: float = 0.5,
    ) -> tf.Tensor:
        """Compute discounted lifetime reward from simulation.
        
        Simulates forward and accumulates discounted cash flows,
        properly handling investment decisions, default events,
        bond pricing, and issuance costs.
        
        Args:
            initial_states: Tuple (K_0, B_0, Z_0) or (K_0, Z_0).
                Convention: (K, B, Z) for 3-element, (K, Z) for 2-element.
            innovation_sequence: Shape (batch_size, T+1) standard normals.
            default_threshold: Threshold for default decision.
            
        Returns:
            Mean discounted lifetime reward across batch.
        """
        # Parse states: convention is (K, B, Z)
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
        
        # Pre-cast constants (module internals handle their own casting)
        one_minus_tax = tf.cast(
            1.0 - self.params.corporate_tax_rate, TENSORFLOW_DTYPE
        )
        beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
        
        accumulated_reward = tf.zeros(
            shape=(batch_size,), dtype=TENSORFLOW_DTYPE
        )
        
        # Track cumulative defaults: once dead, always dead
        # (matches VFI convention — no re-entry after default)
        has_defaulted = tf.zeros(
            shape=(batch_size,), dtype=TENSORFLOW_DTYPE
        )
        
        k_t = k_curr
        z_t = z_curr
        b_t = b_curr
        
        for t in range(T):
            # Get policy outputs matching training architecture
            k_next, b_next, invest_prob, default_prob, equity_issuance, issuance_gate_prob = (
                self._get_policy_action(k_t, b_t, z_t)
            )

            default_decision = tf.cast(
                default_prob > default_threshold, TENSORFLOW_DTYPE
            )

            # Update cumulative default: once dead, always dead
            has_defaulted = tf.maximum(has_defaulted, default_decision)
            alive_mask = 1.0 - has_defaulted

            # Production via centralized module: (1-tau) * Z * K^theta
            profit = one_minus_tax * ProductionFunctions.cobb_douglas(
                k_t, z_t, self.params
            )
            
            # Investment via centralized module: K' - (1-delta)K
            investment = ProductionFunctions.calculate_investment(
                k_t, k_next, self.params
            )
            
            # Adjustment cost: invest_decision already encoded in k_next --
            # non-investors have k'=(1-delta)k so investment=0, adj_cost=0 automatically
            adj_cost, _ = AdjustmentCostCalculator.calculate(
                investment, k_t, self.params
            )
            
            # Bond pricing
            bond_price = self._estimate_bond_price(k_next, b_next, z_t)
            
            # Debt flow via centralized module
            debt_inflow, tax_shield = DebtFlowCalculator.calculate(
                b_next, bond_price, self.params
            )
            
            # Payout (before issuance cost)
            payout = profit + debt_inflow + tax_shield - adj_cost - investment - b_t
            
            # Equity issuance from budget constraint (matching VFI):
            # When payout < 0, the firm must issue equity to cover the shortfall.
            # equity_issuance = max(0, -payout)
            # issuance_cost = (eta_0 + eta_1 * equity_iss) * 1{payout < 0}
            equity_iss = tf.nn.relu(-payout)
            needs_issuance = tf.cast(payout < 1e-4, TENSORFLOW_DTYPE)
            issuance_cost = needs_issuance * (self.eta_0 + self.eta_1 * equity_iss)
            
            # Cash flow: zero for ALL permanently dead firms
            cash_flow = payout - issuance_cost
            cash_flow = alive_mask * cash_flow
            
            accumulated_reward = accumulated_reward + cash_flow * (beta ** t)
            
            # Productivity transition (evolves for everyone)
            z_next = TransitionFunctions.log_ar1_transition(
                z_t,
                self.params.productivity_persistence,
                eps_path[:, t + 1],
            )
            # Dead firms: no restart, state is irrelevant since
            # alive_mask zeros out their cash flow permanently
            k_t = k_next
            z_t = z_next
            b_t = b_next
        
        return tf.reduce_mean(accumulated_reward)
    
    def compute_value_function_gap(
        self,
        grid_points: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        value_labels: tf.Tensor,
    ) -> dict[str, float]:
        """Compute the gap between predicted and true value function on a grid.
        
        Args:
            grid_points: Tuple (K_grid, B_grid, Z_grid) of grid point tensors.
                Expected shape (n_k, n_b, n_z) or flattened.
            value_labels: True value function values from VFI.
                Expected shape matching grid_points or flattened.
            
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
        
        value_pred = self._get_value_function(k_curr, b_curr, z_curr)
        value_pred = tf.reshape(value_pred, [-1])
        
        residual = value_labels_flat - value_pred
        
        mae = tf.reduce_mean(tf.abs(residual))
        safe_denominator = tf.abs(value_labels_flat) + 1e-4
        mape = tf.reduce_mean(tf.abs(residual) / safe_denominator)
        
        return {
            "mae": float(mae.numpy()),
            "mape": float(mape.numpy()),
        }
