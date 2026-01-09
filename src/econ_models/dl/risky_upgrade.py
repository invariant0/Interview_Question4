# econ_models/dl/risky_upgrade.py
"""
Deep Learning solver for the Risky Debt Model using Actor-Critic with Policy Networks.

This module implements a neural network approach to solving the
risky debt model with endogenous default and bond pricing.
Uses policy networks for capital and debt decisions with
Fischer-Burmeister complementarity conditions.

Architecture follows the FB function formulation:
- Value Net V(s): Estimates V(s) with target network for stable bootstrapping
- Continuous Net Vcont(s): Estimates value assuming operation (no default)
- Capital Policy πk(s): Deterministic policy for k'
- Debt Policy πb(s): Deterministic policy for b'

Training order within each step:
1. Compute RHS targets (d + β*Vtarg(s')) for two shocks
2. Policy update: Maximize RHS values
3. Continuous critic update: Fit Vcont to RHS targets using AiO loss
4. Value critic update: Apply FB(V, Vcont) loss

Example:
    >>> from econ_models.dl.risky import RiskyModelDL
    >>> model = RiskyModelDL(params, config)
    >>> model.train()
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CashFlowCalculator,
    BondPricingCalculator
)
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.math import FischerBurmeisterLoss
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.io.checkpoints import save_checkpoint_risky_upgrade
from econ_models.dl.simulation_history import RiskyDLSimulationHistory
tfd = tfp.distributions


class RiskyModelDL_UPGRADE:
    """
    Deep learning solver for the Risky Debt model using Actor-Critic.

    This class implements the FB function formulation with:
    - Value Net V(s; θV): Estimates V(s), has target network Vtarg
    - Continuous Net Vcont(s; θC): Estimates value assuming operation
    - Capital Policy πk(s; θπ): Deterministic policy for k'
    - Debt Policy πb(s; θπ): Deterministic policy for b'
    
    Training order within each step (shared shocks eps1, eps2):
    1. Compute RHS targets: y_i = d(s, π(s)) + β*Vtarg(s'_i) for i=1,2
    2. Policy update: Maximize E[y_1 + y_2]/2 w.r.t. policy parameters
    3. Continuous update: Minimize E[(Vcont - y_1)(Vcont - y_2)] (AiO loss)
    4. Value update: Minimize FB(V, Vcont) loss

    Attributes:
        params: Economic parameters.
        config: Deep learning configuration.
        normalizer: State space normalizer.
        value_net: Value function neural network V(s).
        target_value_net: Target value network Vtarg(s).
        continuous_net: Continuous value network Vcont(s).
        capital_policy_net: Policy network for next period capital.
        debt_policy_net: Policy network for next period debt.
        value_optimizer: Optimizer for value network.
        continuous_optimizer: Optimizer for continuous network.
        policy_optimizer: Optimizer for policy networks.
        shock_dist: Normal distribution for productivity shocks.
        training_progress: Curriculum learning progress variable.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig
    ) -> None:
        """
        Initialize the risky debt deep learning model.

        Args:
            params: Economic parameters.
            config: Deep learning configuration.
        """
        self.params = params
        self.config = config

        self.config.update_value_scale(self.params)
        self.normalizer = StateSpaceNormalizer(self.config)

        self._build_networks()
        self._build_optimizers()
        self._build_shock_distribution()
        self._initialize_curriculum()

    def _build_networks(self) -> None:
        """
        Construct neural networks following the blueprint architecture.
        
        Networks:
        - Value Net V(s): Has target network, linear output activation
        - Continuous Net Vcont(s): No target network, linear output activation
        - Capital Policy πk(s): No target network, sigmoid output
        - Debt Policy πb(s): No target network, sigmoid output
        """
        # Value Network V(s; θV) - estimates V(s)
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=self.config.value_scale_factor,
            name="ValueNet"
        )

        # Target Value Network Vtarg(s; θV-)
        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=self.config.value_scale_factor,
            name="TargetValueNet"
        )
        self.target_value_net.set_weights(self.value_net.get_weights())

        # Continuous Network Vcont(s; θC) - estimates value assuming operation
        self.continuous_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="linear",
            scale_factor=self.config.value_scale_factor,
            name="ContinuousNet"
        )

        # Capital Policy Network πk(s; θπ) -> k'
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="CapitalPolicyNet"
        )

        # Debt Policy Network πb(s; θπ) -> b'
        self.debt_policy_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="sigmoid",
            scale_factor=1.0,
            name="DebtPolicyNet"
        )

    def _build_optimizers(self) -> None:
        """Construct separate optimizers for each network type."""
        self.value_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.steps_per_epoch * 20,
            decay_rate=0.9,
            staircase=False
        )

        self.continuous_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.steps_per_epoch * 20,
            decay_rate=0.9,
            staircase=False
        )

        self.policy_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate * 0.1,
            decay_steps=self.config.steps_per_epoch * 20,
            decay_rate=0.9,
            staircase=False
        )

        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.value_lr_schedule
        )
        self.continuous_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.continuous_lr_schedule
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.policy_lr_schedule
        )

    def _build_shock_distribution(self) -> None:
        """Construct productivity shock distribution."""
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE)
        )

    def _initialize_curriculum(self) -> None:
        """Initialize curriculum learning state."""
        self.training_progress = tf.Variable(
            1.0, dtype=TENSORFLOW_DTYPE, trainable=False
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Create training data pipeline with curriculum support."""
        return DatasetBuilder.build_dataset(
            self.config,
            include_debt=True,
            progress_variable=self.training_progress
        )

    @tf.function
    def _prepare_inputs(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor
    ) -> tf.Tensor:
        """
        Prepare normalized inputs for network forward passes.
        
        Args:
            k: Capital values.
            b: Debt values.
            z: Productivity values.
            
        Returns:
            Concatenated normalized input tensor.
        """
        return tf.concat([
            self.normalizer.normalize_capital(k),
            self.normalizer.normalize_debt(b),
            self.normalizer.normalize_productivity(z)
        ], axis=1)

    @tf.function
    def _get_policy_actions(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get policy network outputs (k', b').
        
        Args:
            inputs: Normalized input tensor [k_norm, b_norm, z_norm].
            training: Whether in training mode.
            
        Returns:
            Tuple of (k_prime, b_prime) in original scale.
        """
        k_prime_norm = self.capital_policy_net(inputs, training=training)
        b_prime_norm = self.debt_policy_net(inputs, training=training)

        k_prime = self.normalizer.denormalize_capital(k_prime_norm)
        b_prime = self.normalizer.denormalize_debt(b_prime_norm)
        
        return k_prime, b_prime
    @tf.function
    def _compute_rhs_targets_online(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        eps1: tf.Tensor,
        eps2: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute RHS using ONLINE value network for policy gradients.
        
        This provides immediate feedback on policy choices.
        """
        beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
        
        q = self.estimate_bond_price(k_prime, b_prime, z)
        dividend = self._compute_dividend(k, b, z, k_prime, b_prime, q)
        
        z_prime_1 = TransitionFunctions.log_ar1_transition(
            z, self.params.productivity_persistence, eps1
        )
        z_prime_2 = TransitionFunctions.log_ar1_transition(
            z, self.params.productivity_persistence, eps2
        )
        
        inputs_next_1 = tf.concat([
            self.normalizer.normalize_capital(k_prime),
            self.normalizer.normalize_debt(b_prime),
            self.normalizer.normalize_productivity(z_prime_1)
        ], axis=1)
        
        inputs_next_2 = tf.concat([
            self.normalizer.normalize_capital(k_prime),
            self.normalizer.normalize_debt(b_prime),
            self.normalizer.normalize_productivity(z_prime_2)
        ], axis=1)
        
        v_prime_1 = self.value_net(inputs_next_1, training=False)
        v_prime_2 = self.value_net(inputs_next_2, training=False)
        
        rhs_target_1 = dividend + beta * v_prime_1
        rhs_target_2 = dividend + beta * v_prime_2
        
        return rhs_target_1, rhs_target_2
    @tf.function
    def estimate_bond_price(
        self,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        z_curr: tf.Tensor
    ) -> tf.Tensor:
        """
        Estimate bond price with proper gradient flow.
        
        Gradients flow through:
        1. k_prime, b_prime → affect next period state → affect default probability
        2. k_prime → affects recovery value in default
        3. b_prime → affects face value and price calculation

        """
        batch_size = tf.shape(z_curr)[0]
        n_samples = self.config.mc_sample_number_bond_priceing
        
        # Sample future shocks
        eps = self.shock_dist.sample(sample_shape=(batch_size, n_samples))
        z_curr_bc = tf.broadcast_to(z_curr, (batch_size, n_samples))
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )
        
        k_prime_bc = tf.broadcast_to(k_prime, (batch_size, n_samples))
        b_prime_bc = tf.broadcast_to(b_prime, (batch_size, n_samples))
        
        flat_shape = (batch_size * n_samples, 1)
        
        # === Compute value at next period state ===
        # Gradients flow through k_prime and b_prime to the inputs
        # but NOT through the target network weights
        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(tf.reshape(k_prime_bc, flat_shape)),  # NO stop_gradient!
            self.normalizer.normalize_debt(tf.reshape(b_prime_bc, flat_shape)),      # NO stop_gradient!
            self.normalizer.normalize_productivity(tf.reshape(z_prime, flat_shape))
        ], axis=1)
        
        # Target network: gradients flow through inputs but not weights
        v_prime_eval = self.target_value_net(inputs_eval, training=False)
        
        # === Default indicator ===
        # We want gradients to flow: if policy chooses k', b' that lead to
        # lower V(k', b', z'), this should increase default probability,
        # which should lower bond price, which should affect the dividend
        # 
        # However, the step function (casting to 0/1) has zero gradient.
        # We need a soft approximation for gradient flow.

        # Use soft sigmoid:
        temperature = 0.1  # Tune this
        is_default = tf.sigmoid((1 - v_prime_eval) / temperature)
        
        # Note: is_default itself has no gradient (step function), but
        # that's okay - the main gradient channel is through recovery value
        
        # === Recovery value: HAS gradient through k_prime ===
        profit = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(
                tf.reshape(k_prime_bc, flat_shape),
                tf.reshape(z_prime, flat_shape),
                self.params
            )
        )
        recovery = BondPricingCalculator.recovery_value(
            profit, 
            tf.reshape(k_prime_bc, flat_shape),
            self.params
        )
        
        # === Payoff calculation ===
        recovery_capped = tf.minimum(recovery, tf.reshape(b_prime_bc, flat_shape))
        
        # Gradients flow through both branches weighted by is_default
        payoff = (
            is_default * recovery_capped + 
            (1.0 - is_default) * tf.reshape(b_prime_bc, flat_shape)
        )
        
        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, n_samples)), axis=1, keepdims=True
        )
        
        bond_price = BondPricingCalculator.risk_neutral_price(
            expected_payoff,
            b_prime,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price
        )
        
        return bond_price

    @tf.function
    def _compute_dividend(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        q: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute dividend d(s, k', b').
        
        Args:
            k: Current capital.
            b: Current debt.
            z: Current productivity.
            k_prime: Next period capital.
            b_prime: Next period debt.
            q: Bond price.
            
        Returns:
            Dividend tensor.
        """
        revenue = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(k, z, self.params)
        )
        
        investment = ProductionFunctions.calculate_investment(k, k_prime, self.params)
        adj_cost, _ = AdjustmentCostCalculator.calculate(investment, k, self.params)
        
        dividend = CashFlowCalculator.risky_cash_flow(
            revenue, investment, adj_cost, b, b_prime, q, self.params
        )
        
        return dividend

    @tf.function
    def _compute_rhs_targets(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        k_prime: tf.Tensor,
        b_prime: tf.Tensor,
        eps1: tf.Tensor,
        eps2: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute RHS Bellman targets for two shock samples.
        
        RHS_i = d(s, k', b') + β * Vtarg(k', b', z'_i)
        
        where z'_i = transition(z, eps_i)
        
        Args:
            k: Current capital.
            b: Current debt.
            z: Current productivity.
            k_prime: Next period capital (from policy).
            b_prime: Next period debt (from policy).
            eps1: First shock sample, shape (batch_size, 1).
            eps2: Second shock sample, shape (batch_size, 1).
            
        Returns:
            Tuple of (rhs_target_1, rhs_target_2).
        """
        beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
        
        # Compute bond price (pure statistical, no gradients)
        q = self.estimate_bond_price(k_prime, b_prime, z)
        
        # Compute dividend
        dividend = self._compute_dividend(k, b, z, k_prime, b_prime, q)
        
        # Compute next period productivity for both shocks
        z_prime_1 = TransitionFunctions.log_ar1_transition(
            z, self.params.productivity_persistence, eps1
        )
        z_prime_2 = TransitionFunctions.log_ar1_transition(
            z, self.params.productivity_persistence, eps2
        )
        
        # Prepare inputs for target network
        inputs_next_1 = tf.concat([
            self.normalizer.normalize_capital(k_prime),
            self.normalizer.normalize_debt(b_prime),
            self.normalizer.normalize_productivity(z_prime_1)
        ], axis=1)
        
        inputs_next_2 = tf.concat([
            self.normalizer.normalize_capital(k_prime),
            self.normalizer.normalize_debt(b_prime),
            self.normalizer.normalize_productivity(z_prime_2)
        ], axis=1)
        
        # Get target values (no gradient flow through target network)
        v_prime_1 = self.target_value_net(inputs_next_1, training=False)
        v_prime_2 = self.target_value_net(inputs_next_2, training=False)
        
        # Compute RHS targets
        rhs_target_1 = dividend + beta * v_prime_1
        rhs_target_2 = dividend + beta * v_prime_2
        
        return rhs_target_1, rhs_target_2

    @tf.function
    def train_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one complete training step with shared shocks.
        
        Training order:
        1. Sample two shocks eps1, eps2 (shared across all updates)
        2. Policy update: Maximize E[(RHS_1 + RHS_2)/2] w.r.t. policy params
        3. Continuous update: Minimize E[(Vcont - RHS_1)(Vcont - RHS_2)] 
        4. Value update: Minimize FB(V, Vcont) loss
        5. Soft update target network

        Args:
            k: Capital values.
            b: Debt values.
            z: Productivity values.

        Returns:
            Dictionary of training metrics.
        """
        batch_size = tf.shape(k)[0]
        
        # ========== Sample two shocks once (shared across all updates) ==========
        eps1 = self.shock_dist.sample(sample_shape=(batch_size, 1))
        eps2 = self.shock_dist.sample(sample_shape=(batch_size, 1))
        
        # ========== Prepare normalized inputs ==========
        inputs = self._prepare_inputs(k, b, z)
        
        # ========== Step 1: Policy Update ==========
        # Maximize RHS = d + β*Vtarg(s') w.r.t. policy parameters
        policy_variables = (
            self.capital_policy_net.trainable_variables +
            self.debt_policy_net.trainable_variables
        )
        
        with tf.GradientTape() as policy_tape:
            # Get policy actions with gradients
            k_prime, b_prime = self._get_policy_actions(inputs, training=True)
            
            # Compute RHS targets
            rhs_1, rhs_2 = self._compute_rhs_targets_online(
                k, b, z, k_prime, b_prime, eps1, eps2
            )
            
            # Policy objective: maximize average of two RHS values
            avg_rhs = (rhs_1 + rhs_2) / 2.0
            policy_loss = -tf.reduce_mean(avg_rhs)
        
        policy_grads = policy_tape.gradient(policy_loss, policy_variables)
        if self.config.gradient_clip_norm:
            policy_grads, _ = tf.clip_by_global_norm(
                policy_grads, self.config.gradient_clip_norm
            )
        self.policy_optimizer.apply_gradients(zip(policy_grads, policy_variables))
        
        # ========== Compute frozen RHS targets for critic updates ==========
        # Use updated policy but stop gradients for critic training
        k_prime_frozen, b_prime_frozen = self._get_policy_actions(inputs, training=False)
        k_prime_frozen = tf.stop_gradient(k_prime_frozen)
        b_prime_frozen = tf.stop_gradient(b_prime_frozen)
        
        rhs_target_1, rhs_target_2 = self._compute_rhs_targets(
            k, b, z, k_prime_frozen, b_prime_frozen, eps1, eps2
        )
        rhs_target_1 = tf.stop_gradient(rhs_target_1)
        rhs_target_2 = tf.stop_gradient(rhs_target_2)
        
        # ========== Step 2: Continuous Critic Update ==========
        # Fit Vcont to RHS targets using AiO loss
        with tf.GradientTape() as continuous_tape:
            v_cont = self.continuous_net(inputs, training=True)
            
            # AiO loss: E[(Vcont - RHS_1) * (Vcont - RHS_2)]
            continuous_loss = tf.reduce_mean(
                (v_cont - rhs_target_1) * (v_cont - rhs_target_2)
            )
        
        continuous_grads = continuous_tape.gradient(
            continuous_loss, self.continuous_net.trainable_variables
        )
        if self.config.gradient_clip_norm:
            continuous_grads, _ = tf.clip_by_global_norm(
                continuous_grads, self.config.gradient_clip_norm
            )
        self.continuous_optimizer.apply_gradients(
            zip(continuous_grads, self.continuous_net.trainable_variables)
        )
        
        # ========== Step 3: Value Critic Update ==========
        # Get Vcont outputs for both shocks (with stopped gradient)
        # Recompute Vcont-style targets from the frozen RHS values
        v_cont_for_fb = tf.stop_gradient(self.continuous_net(inputs, training=False))
        with tf.GradientTape() as value_tape:
            v_value = self.value_net(inputs, training=True)
            
            # Two-shock FB loss for unbiased gradients
            fb_loss = FischerBurmeisterLoss.compute_loss(
                v_value, v_cont_for_fb
            )
        
        value_grads = value_tape.gradient(fb_loss, self.value_net.trainable_variables)
        if self.config.gradient_clip_norm:
            value_grads, _ = tf.clip_by_global_norm(
                value_grads, self.config.gradient_clip_norm
            )
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value_net.trainable_variables)
        )
        
        # ========== Compute logging metrics ==========
        avg_rhs_target = (rhs_target_1 + rhs_target_2) / 2.0
        q_frozen = self.estimate_bond_price(k_prime_frozen, b_prime_frozen, z)
        dividend_frozen = self._compute_dividend(
            k, b, z, k_prime_frozen, b_prime_frozen, q_frozen
        )
        
        return {
            "policy_loss": policy_loss,
            "avg_rhs": tf.reduce_mean(avg_rhs),
            "avg_k_prime": tf.reduce_mean(k_prime_frozen),
            "avg_b_prime": tf.reduce_mean(b_prime_frozen),
            "avg_bond_price": tf.reduce_mean(q_frozen),
            "avg_dividend": tf.reduce_mean(dividend_frozen),
            "continuous_loss": continuous_loss,
            "avg_v_cont": tf.reduce_mean(v_cont),
            "avg_rhs_target": tf.reduce_mean(avg_rhs_target),
            "fb_loss": fb_loss,
            "avg_v_value": tf.reduce_mean(v_value),
            "avg_v_cont_for_fb": tf.reduce_mean(v_cont_for_fb),
        }

    @tf.function
    def _get_optimal_actions_for_simulation(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Get optimal (k', b', q, V) for simulation.

        Uses the same denormalization as training for consistency.

        Args:
            k: Current capital values.
            b: Current debt values.
            z: Current productivity values.

        Returns:
            Tuple of (optimal k', optimal b', bond price, value).
        """
        inputs = self._prepare_inputs(k, b, z)
        
        # Use inference mode
        k_opt, b_opt = self._get_policy_actions(inputs, training=False)
        
        q_opt = self.estimate_bond_price(k_opt, b_opt, z)
        v_opt = self.value_net(inputs, training=False)
        
        return k_opt, b_opt, q_opt, v_opt

    def train(self) -> None:
        """Execute the main training loop."""
        print(f"Starting Risky Model Training for {self.config.epochs} epochs...")
        print("Architecture: Value Net + Continuous Net + Policy Nets")
        print("Training order: Policy -> Continuous (AiO) -> Value (FB)")
        print("Bond pricing is pure statistical with no gradient flow")

        dataset = self._build_dataset()
        data_iter = iter(dataset)

        for epoch in range(self.config.epochs):
            epoch_logs = self._run_epoch(data_iter)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs)

            if epoch % 10 == 0:
                save_checkpoint_risky_upgrade(
                    value_net=self.value_net,
                    capital_policy_net=self.capital_policy_net,
                    debt_policy_net=self.debt_policy_net,
                    epoch=epoch,
                )

    def _run_epoch(self, data_iter) -> Dict[str, float]:
        """Run one training epoch."""
        epoch_logs = {}

        for step in range(self.config.steps_per_epoch):
            k, b, z = next(data_iter)
            logs = self.train_step(k, b, z)

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            # Soft update target network after each step
            NeuralNetFactory.soft_update(
                self.value_net,
                self.target_value_net,
                self.config.polyak_averaging_decay
            )

        return epoch_logs

    def _log_epoch(self, epoch: int, epoch_logs: Dict[str, float]) -> None:
        """Log training progress for an epoch."""
        steps = self.config.steps_per_epoch
        progress_val = float(self.training_progress.numpy())

        fb_loss = epoch_logs.get('fb_loss', 0) / steps
        cont_loss = epoch_logs.get('continuous_loss', 0) / steps
        policy_loss = epoch_logs.get('policy_loss', 0) / steps
        avg_v = epoch_logs.get('avg_v_value', 0) / steps
        avg_v_cont = epoch_logs.get('avg_v_cont', 0) / steps
        avg_q = epoch_logs.get('avg_bond_price', 0) / steps
        avg_k_prime = epoch_logs.get('avg_k_prime', 0) / steps
        avg_b_prime = epoch_logs.get('avg_b_prime', 0) / steps

        print(
            f"Epoch {epoch:4d} | "
            f"Prg: {progress_val:.2f} | "
            f"FB: {fb_loss:.4e} | "
            f"Cont: {cont_loss:.4e} | "
            f"Pol: {policy_loss:.4e} | "
            f"V: {avg_v:.2f} | "
            f"Vc: {avg_v_cont:.2f} | "
            f"Q: {avg_q:.3f} | "
            f"K': {avg_k_prime:.3f} | "
            f"B': {avg_b_prime:.3f}"
        )

    def simulate(
        self,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: Optional[int] = None,
        sim_batch_size: int = 200
    ) -> Tuple[RiskyDLSimulationHistory, Dict[str, float]]:
        """
        Simulate economy using the learned policy networks.
        
        Includes default logic: if Value(s) <= epsilon, the firm defaults at state s.
        Defaulted firms are marked in the 'd' trajectory (d=1 means defaulted).
        Once a firm defaults, it remains in default state for all future periods.
        
        Args:
            n_steps: Number of time steps per simulation.
            n_batches: Number of parallel simulations.
            seed: Random seed for reproducibility.
            sim_batch_size: Batch size for chunked processing to manage memory.
            
        Returns:
            Tuple of (simulation history, statistics dictionary).
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Extract bounds
        k_min = tf.constant(self.config.capital_min, dtype=TENSORFLOW_DTYPE)
        k_max = tf.constant(self.config.capital_max, dtype=TENSORFLOW_DTYPE)
        b_min = tf.constant(self.config.debt_min, dtype=TENSORFLOW_DTYPE)
        b_max = tf.constant(self.config.debt_max, dtype=TENSORFLOW_DTYPE)
        z_min = tf.constant(self.config.productivity_min, dtype=TENSORFLOW_DTYPE)
        z_max = tf.constant(self.config.productivity_max, dtype=TENSORFLOW_DTYPE)

        # Define Distributions using TFD
        dist_k_init = tfd.Uniform(low=k_min, high=k_max)
        dist_b_init = tfd.Uniform(low=b_min, high=b_max)
        dist_z_init = tfd.Uniform(low=z_min, high=z_max)

        # Pre-allocate CPU memory for results
        k_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        b_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        z_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        q_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        d_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)

        # Process simulation in chunks
        for start_idx in range(0, n_batches, sim_batch_size):
            end_idx = min(start_idx + sim_batch_size, n_batches)
            current_batch_size = end_idx - start_idx
            
            print(f"Simulating batch {start_idx} to {end_idx}...")

            # Sample Initial States on GPU using TFD
            k_curr = dist_k_init.sample(sample_shape=(current_batch_size, 1))
            b_curr = dist_b_init.sample(sample_shape=(current_batch_size, 1))
            z_curr = dist_z_init.sample(sample_shape=(current_batch_size, 1))
            
            # Track cumulative default status for this batch (0=active, 1=defaulted)
            cumulative_defaults = tf.zeros((current_batch_size, 1), dtype=TENSORFLOW_DTYPE)

            for t in range(n_steps):
                # --- A. Compute optimal actions and value at current state ---
                k_opt, b_opt, q_opt, v_opt = self._get_optimal_actions_for_simulation(
                    k_curr, b_curr, z_curr
                )

                # --- B. Check Default Condition at current state ---
                # Firm defaults if value of operating <= epsilon
                is_default_this_step = tf.cast(
                    v_opt <= self.config.epsilon_value_default, TENSORFLOW_DTYPE
                )
                
                # Update cumulative defaults (once defaulted, always defaulted)
                cumulative_defaults = tf.maximum(cumulative_defaults, is_default_this_step)

                # --- C. Store current state and default status ---
                k_hist_np[t, start_idx:end_idx] = k_curr.numpy().flatten()
                b_hist_np[t, start_idx:end_idx] = b_curr.numpy().flatten()
                z_hist_np[t, start_idx:end_idx] = z_curr.numpy().flatten()
                d_hist_np[t, start_idx:end_idx] = cumulative_defaults.numpy().flatten()
                
                # Store bond price (set to 0 for defaulted firms)
                q_masked = tf.where(
                    tf.cast(cumulative_defaults, bool), 
                    tf.zeros_like(q_opt), 
                    q_opt
                )
                q_hist_np[t, start_idx:end_idx] = q_masked.numpy().flatten()

                # --- D. Transition to next period ---
                # Sample shock for productivity transition
                eps = self.shock_dist.sample(sample_shape=(current_batch_size, 1))
                
                # Transition productivity (always evolves, even for defaulted firms)
                z_next = TransitionFunctions.log_ar1_transition(
                    z_curr, self.params.productivity_persistence, eps
                )

                # Update state for next period
                # Active firms: use optimal policy
                # Defaulted firms: remain at minimum values (they've exited)
                active_mask = tf.cast(1.0 - cumulative_defaults, bool)
                
                k_curr = tf.where(active_mask, k_opt, k_min * tf.ones_like(k_opt))
                b_curr = tf.where(active_mask, b_opt, b_min * tf.ones_like(b_opt))
                z_curr = z_next
                
                # Clean up intermediate tensors
                del k_opt, b_opt, q_opt, q_masked, eps, v_opt
                del is_default_this_step, active_mask, z_next

        # Build history object
        history = RiskyDLSimulationHistory(
            trajectories={
                "k": k_hist_np,
                "b": b_hist_np,
                "z": z_hist_np,
                "q": q_hist_np,
                "d": d_hist_np,
                "steady_state_capital": self.config.capital_steady_state,
            },
            n_batches=n_batches,
            n_steps=n_steps
        )

        # Calculate statistics using NumPy (CPU)
        stats = self._compute_simulation_stats(
            k_hist_np, b_hist_np, z_hist_np, d_hist_np,
            float(k_min), float(k_max),
            float(b_min), float(b_max),
            float(z_min), float(z_max)
        )

        return history, stats

    def _compute_simulation_stats(
        self,
        k_history: np.ndarray,
        b_history: np.ndarray,
        z_history: np.ndarray,
        d_history: np.ndarray,
        k_min: float, k_max: float,
        b_min: float, b_max: float,
        z_min: float, z_max: float,
        boundary_tolerance: float = 0.01
    ) -> Dict[str, float]:
        """
        Compute boundary hit statistics from simulation history using NumPy.
        Excludes defaulted observations from boundary stats.
        
        Args:
            k_history: Capital history array of shape (n_steps, n_batches).
            b_history: Debt history array of shape (n_steps, n_batches).
            z_history: Productivity history array of shape (n_steps, n_batches).
            d_history: Default indicator array of shape (n_steps, n_batches).
            k_min, k_max: Capital bounds.
            b_min, b_max: Debt bounds.
            z_min, z_max: Productivity bounds.
            boundary_tolerance: Fraction of range to consider "near" boundary.
            
        Returns:
            Dictionary of simulation statistics.
        """
        total_obs = k_history.size
        
        # Calculate Default Rate (fraction of observations that are in default)
        total_defaults = np.sum(d_history == 1.0)
        default_rate = total_defaults / total_obs
        
        # Calculate unique default events (first time each batch defaults)
        n_steps, n_batches = d_history.shape
        n_firms_that_defaulted = 0
        for batch_idx in range(n_batches):
            if np.any(d_history[:, batch_idx] == 1.0):
                n_firms_that_defaulted += 1
        firm_default_rate = n_firms_that_defaulted / n_batches

        # Create mask for valid (active) observations
        valid_mask = d_history == 0.0
        valid_k = k_history[valid_mask]
        valid_b = b_history[valid_mask]
        valid_z = z_history[valid_mask]
        
        valid_obs_count = valid_k.size

        # Calculate boundary thresholds
        k_tol = boundary_tolerance * (k_max - k_min)
        b_tol = boundary_tolerance * (b_max - b_min)
        z_tol = boundary_tolerance * (z_max - z_min)

        def calc_var_stats(history_subset, v_min, v_max, v_tol, prefix):
            if history_subset.size == 0:
                return {
                    f"{prefix}_below_min_pct": 0.0,
                    f"{prefix}_above_max_pct": 0.0,
                    f"{prefix}_near_min_pct": 0.0,
                    f"{prefix}_near_max_pct": 0.0,
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_std": 0.0,
                    f"{prefix}_min_observed": 0.0,
                    f"{prefix}_max_observed": 0.0,
                }

            below_min = np.sum(history_subset < v_min)
            above_max = np.sum(history_subset > v_max)
            near_min = np.sum(history_subset < v_min + v_tol)
            near_max = np.sum(history_subset > v_max - v_tol)
            
            return {
                f"{prefix}_below_min_pct": float(below_min / valid_obs_count),
                f"{prefix}_above_max_pct": float(above_max / valid_obs_count),
                f"{prefix}_near_min_pct": float(near_min / valid_obs_count),
                f"{prefix}_near_max_pct": float(near_max / valid_obs_count),
                f"{prefix}_mean": float(np.mean(history_subset)),
                f"{prefix}_std": float(np.std(history_subset)),
                f"{prefix}_min_observed": float(np.min(history_subset)),
                f"{prefix}_max_observed": float(np.max(history_subset)),
            }

        stats = {}
        stats.update(calc_var_stats(valid_k, k_min, k_max, k_tol, "k"))
        stats.update(calc_var_stats(valid_b, b_min, b_max, b_tol, "b"))
        stats.update(calc_var_stats(valid_z, z_min, z_max, z_tol, "z"))

        stats.update({
            "k_min_bound": k_min, "k_max_bound": k_max,
            "b_min_bound": b_min, "b_max_bound": b_max,
            "z_min_bound": z_min, "z_max_bound": z_max,
            "total_observations": int(total_obs),
            "valid_observations": int(valid_obs_count),
            "default_rate": float(default_rate),
            "firm_default_rate": float(firm_default_rate),
        })

        return stats