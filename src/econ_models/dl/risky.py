# econ_models/dl/risky.py
"""
Deep Learning solver for the Risky Debt Model.

This module implements a neural network approach to solving the
risky debt model with endogenous default and bond pricing.

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
from econ_models.core.sampling.candidate_sampler import CandidateSampler
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CashFlowCalculator,
    BondPricingCalculator
)
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.io.checkpoints import save_checkpoint_risky
from econ_models.dl.simulation_history import RiskyDLSimulationHistory
tfd = tfp.distributions

class RiskyModelDL:
    """
    Deep learning solver for the Risky Debt model.

    This class uses a value function neural network with a target network
    for stability, similar to Deep Q-Learning approaches. Bond prices are
    computed via Monte Carlo integration over default probabilities.

    Attributes:
        params: Economic parameters.
        config: Deep learning configuration.
        normalizer: State space normalizer.
        value_net: Value function neural network.
        target_value_net: Target network for stable bootstrapping.
        optimizer: Adam optimizer with learning rate schedule.
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
        self._build_optimizer()
        self._build_shock_distribution()
        self._initialize_curriculum()

    def _build_networks(self) -> None:
        """Construct value and target neural networks."""
        self.value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="relu",
            scale_factor=self.config.value_scale_factor,
            name="RiskyValueNet"
        )

        self.target_value_net = NeuralNetFactory.build_mlp(
            input_dim=3,
            output_dim=1,
            config=self.config,
            output_activation="relu",
            scale_factor=self.config.value_scale_factor,
            name="RiskyTargetValueNet"
        )

        # Initialize target weights to match source
        self.target_value_net.set_weights(self.value_net.get_weights())

    def _build_optimizer(self) -> None:
        """Construct optimizer with learning rate schedule."""
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.steps_per_epoch * 20,
            decay_rate=0.9,
            staircase=False
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

    def _build_shock_distribution(self) -> None:
        """Construct productivity shock distribution."""
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(self.params.productivity_std_dev, TENSORFLOW_DTYPE)
        )

    def _initialize_curriculum(self) -> None:
        """Initialize curriculum learning state."""
        self.training_progress = tf.Variable(
            0.0, dtype=TENSORFLOW_DTYPE, trainable=False
        )

    def _build_dataset(self) -> tf.data.Dataset:
        """Create training data pipeline with curriculum support."""
        return DatasetBuilder.build_dataset(
            self.config,
            include_debt=True,
            progress_variable=self.training_progress
        )

    @tf.function
    def estimate_bond_price(
        self,
        k_prime_cand: tf.Tensor,
        b_prime_cand: tf.Tensor,
        z_curr: tf.Tensor
    ) -> tf.Tensor:
        """
        Estimate bond price q(k', b', z) via Monte Carlo.

        Args:
            k_prime_cand: Candidate next capital values.
            b_prime_cand: Candidate next debt values.
            z_curr: Current productivity values.

        Returns:
            Estimated bond prices for each candidate.
        """
        batch_size = tf.shape(z_curr)[0]
        n_cand = tf.shape(k_prime_cand)[1]
        n_samples = self.config.mc_sample_number_bond_priceing

        # Sample future shocks
        eps = self.shock_dist.sample(sample_shape=(batch_size, n_cand, n_samples))
        z_curr_bc = tf.expand_dims(
            tf.broadcast_to(z_curr, (batch_size, n_cand)), -1
        )
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )

        # Broadcast candidates
        k_prime_bc = tf.broadcast_to(
            tf.expand_dims(k_prime_cand, -1), (batch_size, n_cand, n_samples)
        )
        b_prime_bc = tf.broadcast_to(
            tf.expand_dims(b_prime_cand, -1), (batch_size, n_cand, n_samples)
        )

        # Evaluate value at (k', b', z') for default check
        flat_shape = (batch_size * n_cand * n_samples, 1)
        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(tf.reshape(k_prime_bc, flat_shape)),
            self.normalizer.normalize_debt(tf.reshape(b_prime_bc, flat_shape)),
            self.normalizer.normalize_productivity(tf.reshape(z_prime, flat_shape))
        ], axis=1)

        v_prime_eval = self.target_value_net(inputs_eval, training=False)
        is_default = tf.cast(
            v_prime_eval <= self.config.epsilon_value_default,
            TENSORFLOW_DTYPE
        )

        # Calculate payoff
        profit = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(
                tf.reshape(k_prime_bc, flat_shape),
                tf.reshape(z_prime, flat_shape),
                self.params
            )
        )
        recovery = BondPricingCalculator.recovery_value(
            profit, tf.reshape(k_prime_bc, flat_shape), self.params
        )
        payoff = BondPricingCalculator.bond_payoff(
            recovery, tf.reshape(b_prime_bc, flat_shape), is_default
        )

        # Average over samples
        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, n_cand, n_samples)), axis=2
        )

        return BondPricingCalculator.risk_neutral_price(
            expected_payoff,
            b_prime_cand,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price
        )

    @tf.function
    def _optimize_next_state(
        self,
        k_curr: tf.Tensor,
        b_curr: tf.Tensor,
        z_curr: tf.Tensor,
        test_mode: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Find optimal (k', b') by evaluating candidate grid.

        Args:
            k_curr: Current capital values.
            b_curr: Current debt values.
            z_curr: Current productivity values.

        Returns:
            Tuple of (optimal k', optimal b', optimal bond price).
        """
        batch_size = tf.shape(k_curr)[0]
        n_cand = self.config.mc_next_candidate_sample
        n_mc_samples = self.config.mc_sample_number_continuation_value
        # Generate candidates
        if test_mode:
            n_cand = 2500
            k_cand, b_cand = CandidateSampler.sample_candidate_grid(
                batch_size, n_cand, k_curr, b_curr,
                self.config, progress=1.0
            )
        else:
            k_cand, b_cand = CandidateSampler.sample_candidate(
                batch_size, n_cand, k_curr, b_curr,
                self.config, progress=self.training_progress
            )
        n_actual = tf.shape(k_cand)[1]

        # Estimate bond prices
        q_cand = self.estimate_bond_price(k_cand, b_cand, z_curr)

        # Calculate current cash flows
        revenue = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)
        )

        k_curr_bc = tf.broadcast_to(k_curr, (batch_size, n_actual))
        b_curr_bc = tf.broadcast_to(b_curr, (batch_size, n_actual))

        investment = ProductionFunctions.calculate_investment(
            k_curr_bc, k_cand, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr_bc, self.params
        )

        dividend = CashFlowCalculator.risky_cash_flow(
            tf.broadcast_to(revenue, (batch_size, n_actual)),
            investment, adj_cost,
            b_curr_bc, b_cand, q_cand, self.params
        )

        # Estimate expected continuation value
        eps_samples = self.shock_dist.sample(
            sample_shape=(batch_size, n_actual, n_mc_samples)
        )

        z_curr_expanded = tf.expand_dims(
            tf.broadcast_to(z_curr, (batch_size, n_actual)), -1
        )
        z_prime_samples = TransitionFunctions.log_ar1_transition(
            z_curr_expanded, self.params.productivity_persistence, eps_samples
        )

        k_cand_expanded = tf.broadcast_to(
            tf.expand_dims(k_cand, -1), (batch_size, n_actual, n_mc_samples)
        )
        b_cand_expanded = tf.broadcast_to(
            tf.expand_dims(b_cand, -1), (batch_size, n_actual, n_mc_samples)
        )

        flat_dim = batch_size * n_actual * n_mc_samples
        inputs_next = tf.concat([
            self.normalizer.normalize_capital(
                tf.reshape(k_cand_expanded, (flat_dim, 1))
            ),
            self.normalizer.normalize_debt(
                tf.reshape(b_cand_expanded, (flat_dim, 1))
            ),
            self.normalizer.normalize_productivity(
                tf.reshape(z_prime_samples, (flat_dim, 1))
            )
        ], axis=1)

        v_prime_flat = self.target_value_net(inputs_next, training=False)
        v_prime_samples = tf.reshape(
            v_prime_flat, (batch_size, n_actual, n_mc_samples)
        )
        expected_v_prime = tf.reduce_mean(v_prime_samples, axis=2)

        # Maximize total value
        beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
        rhs_cand = dividend + beta * expected_v_prime
        best_indices = tf.argmax(rhs_cand, axis=1)
        # Get best value (for default checking)
        best_val = tf.reduce_max(rhs_cand, axis=1)
        gather_indices = tf.stack(
            [tf.range(batch_size, dtype=tf.int64), best_indices], axis=1
        )

        k_opt = tf.expand_dims(tf.gather_nd(k_cand, gather_indices), 1)
        b_opt = tf.expand_dims(tf.gather_nd(b_cand, gather_indices), 1)
        q_opt = tf.expand_dims(tf.gather_nd(q_cand, gather_indices), 1)
        v_opt = tf.expand_dims(best_val, 1)
        return k_opt, b_opt, q_opt, v_opt

    @tf.function
    def train_step(
        self,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one gradient descent step on the value network.

        Args:
            k: Capital values.
            b: Debt values.
            z: Productivity values.

        Returns:
            Dictionary of training metrics.
        """
        with tf.GradientTape() as tape:
            # Find optimal policy
            k_opt, b_opt, q_opt, v_opt = self._optimize_next_state(k, b, z)

            # Calculate implied value target
            revenue = (
                (1.0 - self.params.corporate_tax_rate)
                * ProductionFunctions.cobb_douglas(k, z, self.params)
            )
            investment = ProductionFunctions.calculate_investment(k, k_opt, self.params)
            adj_cost, _ = AdjustmentCostCalculator.calculate(investment, k, self.params)
            dividend = CashFlowCalculator.risky_cash_flow(
                revenue, investment, adj_cost, b, b_opt, q_opt, self.params
            )

            # Calculate continuation values with two independent samples
            eps1 = self.shock_dist.sample(sample_shape=tf.shape(z))
            eps2 = self.shock_dist.sample(sample_shape=tf.shape(z))
            z_prime_1 = TransitionFunctions.log_ar1_transition(
                z, self.params.productivity_persistence, eps1
            )
            z_prime_2 = TransitionFunctions.log_ar1_transition(
                z, self.params.productivity_persistence, eps2
            )

            k_opt_norm = self.normalizer.normalize_capital(k_opt)
            b_opt_norm = self.normalizer.normalize_debt(b_opt)

            v_prime_1 = self.target_value_net(
                tf.concat([
                    k_opt_norm, b_opt_norm,
                    self.normalizer.normalize_productivity(z_prime_1)
                ], 1),
                training=False
            )
            v_prime_2 = self.target_value_net(
                tf.concat([
                    k_opt_norm, b_opt_norm,
                    self.normalizer.normalize_productivity(z_prime_2)
                ], 1),
                training=False
            )

            # Current value
            v_curr = self.value_net(
                tf.concat([
                    self.normalizer.normalize_capital(k),
                    self.normalizer.normalize_debt(b),
                    self.normalizer.normalize_productivity(z)
                ], 1),
                training=True
            )

            # Calculate AiO loss
            beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
            target_1 = tf.keras.activations.relu(dividend + beta * v_prime_1)
            target_2 = tf.keras.activations.relu(dividend + beta * v_prime_2)

            bellman_res1 = v_curr - target_1
            bellman_res2 = v_curr - target_2
            loss = tf.reduce_mean(bellman_res1 * bellman_res2)

        # Apply gradients
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        if self.config.gradient_clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip_norm)
        self.optimizer.apply_gradients(
            zip(grads, self.value_net.trainable_variables)
        )

        return {
            "loss": loss,
            "avg_val": tf.reduce_mean(v_curr),
            "avg_bond_price": tf.reduce_mean(q_opt),
            "avg_k_prime": tf.reduce_mean(k_opt),
            "avg_b_prime": tf.reduce_mean(b_opt),
        }

    def train(self) -> None:
        """Execute the main training loop."""
        print(f"Starting Risky Model Training for {self.config.epochs} epochs...")

        dataset = self._build_dataset()
        data_iter = iter(dataset)

        for epoch in range(self.config.epochs):
            # Update curriculum progress
            self._update_curriculum(epoch)

            epoch_logs = self._run_epoch(data_iter)

            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch(epoch, epoch_logs)

            if epoch % 10 == 0:
                save_checkpoint_risky(self.value_net, epoch)

    def _update_curriculum(self, epoch: int) -> None:
        """Update curriculum learning progress."""
        if self.config.curriculum_epochs > 0:
            progress_val = min(1.0, epoch / self.config.curriculum_epochs)
        else:
            progress_val = 1.0
        self.training_progress.assign(tf.cast(progress_val, TENSORFLOW_DTYPE))

    def _run_epoch(self, data_iter) -> Dict[str, float]:
        """Run one training epoch."""
        epoch_logs = {}

        for step in range(self.config.steps_per_epoch):
            k, b, z = next(data_iter)
            logs = self.train_step(k, b, z)

            for key, value in logs.items():
                epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)

            # Soft update target network every step
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

        avg_loss = epoch_logs['loss'] / steps
        avg_val = epoch_logs['avg_val'] / steps
        avg_q = epoch_logs['avg_bond_price'] / steps
        avg_k_prime = epoch_logs['avg_k_prime'] / steps
        avg_b_prime = epoch_logs['avg_b_prime'] / steps

        print(
            f"Epoch {epoch:4d} | "
            f"Prg: {progress_val:.2f} | "
            f"Loss: {avg_loss:.4e} | "
            f"Val: {avg_val:.2f} | "
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
        Simulate economy using the learned value function and candidate search.
        
        Includes default logic: if Value(Repay) <= epsilon, the firm defaults.
        Defaulted firms are marked in the 'd' trajectory and stop operating.
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Extract bounds (Keep these on GPU)
        k_min = tf.constant(self.config.capital_min, dtype=TENSORFLOW_DTYPE)
        k_max = tf.constant(self.config.capital_max, dtype=TENSORFLOW_DTYPE)
        b_min = tf.constant(self.config.debt_min, dtype=TENSORFLOW_DTYPE)
        b_max = tf.constant(self.config.debt_max, dtype=TENSORFLOW_DTYPE)
        z_min = tf.constant(self.config.productivity_min, dtype=TENSORFLOW_DTYPE)
        z_max = tf.constant(self.config.productivity_max, dtype=TENSORFLOW_DTYPE)

        # 1. Define Distributions using TFD
        dist_k_init = tfd.Uniform(low=k_min, high=k_max)
        dist_b_init = tfd.Uniform(low=b_min, high=b_max)
        dist_z_init = tfd.Uniform(low=z_min, high=z_max)

        # Pre-allocate CPU memory (RAM) for results
        k_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        b_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        z_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        q_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32)
        d_hist_np = np.zeros((n_steps, n_batches), dtype=np.float32) # Default history

        # Process simulation in chunks
        for start_idx in range(0, n_batches, sim_batch_size):
            end_idx = min(start_idx + sim_batch_size, n_batches)
            current_batch_size = end_idx - start_idx
            
            print(f"Simulating batch {start_idx} to {end_idx}...")

            # 2. Sample Initial States on GPU using TFD
            k_curr = dist_k_init.sample(sample_shape=(current_batch_size, 1))
            b_curr = dist_b_init.sample(sample_shape=(current_batch_size, 1))
            z_curr = dist_z_init.sample(sample_shape=(current_batch_size, 1))
            
            # Track current default status for this batch (0=active, 1=defaulted)
            # Once 1, stays 1.
            current_defaults = tf.zeros((current_batch_size, 1), dtype=TENSORFLOW_DTYPE)

            for t in range(n_steps):
                # --- A. Offload to CPU ---
                k_hist_np[t, start_idx:end_idx] = k_curr.numpy().flatten()
                b_hist_np[t, start_idx:end_idx] = b_curr.numpy().flatten()
                z_hist_np[t, start_idx:end_idx] = z_curr.numpy().flatten()
                d_hist_np[t, start_idx:end_idx] = current_defaults.numpy().flatten()

                # --- B. Compute Next Step (GPU) ---
                # Get optimal policy AND the value of that policy
                k_opt, b_opt, q_opt, v_opt = self._optimize_next_state(
                    k_curr, b_curr, z_curr, test_mode=True
                )

                # --- C. Check Default Condition ---
                # Logic referenced from VFI: if Value <= Epsilon, default.
                # Note: We check if the *continuation* value is too low.
                is_default_step = tf.cast(
                    v_opt <= self.config.epsilon_value_default, TENSORFLOW_DTYPE
                )
                
                # Update cumulative defaults: (already_defaulted OR newly_defaulted)
                new_defaults = tf.maximum(current_defaults, is_default_step)
                
                # Store bond price (mask defaulted ones to 0 for clarity)
                q_masked = tf.where(
                    tf.cast(new_defaults, bool), 
                    tf.zeros_like(q_opt), 
                    q_opt
                )
                q_hist_np[t, start_idx:end_idx] = q_masked.numpy().flatten()

                # Sample shock (GPU)
                eps = self.shock_dist.sample(sample_shape=(current_batch_size, 1))
                
                # Transition productivity (GPU)
                z_next = TransitionFunctions.log_ar1_transition(
                    z_curr, self.params.productivity_persistence, eps
                )

                # --- D. Update State Pointers ---
                # If defaulted, we effectively stop updating k and b (set to 0 or hold const)
                # Here we set to 0.0 to indicate "exit" clearly in the data, similar to -1 in VFI.
                active_mask = tf.cast(1.0 - new_defaults, bool)
                
                k_curr = tf.where(active_mask, k_opt, tf.zeros_like(k_opt))
                b_curr = tf.where(active_mask, b_opt, tf.zeros_like(b_opt))
                
                # Z continues to evolve exogenously, or we can freeze it. 
                # Evolving it is harmless and keeps array shapes consistent.
                z_curr = z_next
                
                # Update default tracker
                current_defaults = new_defaults
                
                # Cleanup
                del q_opt, q_masked, eps, v_opt

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
        """
        total_obs = k_history.size
        
        # Calculate Default Rate
        # d_history contains 1.0 for defaults.
        total_defaults = np.sum(d_history == 1.0)
        default_rate = total_defaults / total_obs

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

        # Helper to calculate stats for a single variable
        def calc_var_stats(history_subset, v_min, v_max, v_tol, prefix):
            if history_subset.size == 0:
                return {
                    f"{prefix}_below_min_pct": 0.0,
                    f"{prefix}_above_max_pct": 0.0,
                    f"{prefix}_near_min_pct": 0.0,
                    f"{prefix}_near_max_pct": 0.0,
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_std": 0.0,
                }

            below_min = np.sum(history_subset < v_min)
            above_max = np.sum(history_subset > v_max)
            near_min = np.sum(history_subset < v_min + v_tol)
            near_max = np.sum(history_subset > v_max - v_tol)
            
            # Denominator is valid_obs_count, not total_obs
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

        # Add bounds used for reference
        stats.update({
            "k_min_bound": k_min, "k_max_bound": k_max,
            "b_min_bound": b_min, "b_max_bound": b_max,
            "z_min_bound": z_min, "z_max_bound": z_max,
            "total_observations": int(total_obs),
            "valid_observations": int(valid_obs_count),
            "default_rate": float(default_rate)
        })

        return stats