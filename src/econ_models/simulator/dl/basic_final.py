# src/econ_models/simulator/dl/basic_final.py
"""Deep Learning Simulator for the Basic RBC Model with investment decision.

Simulates the economy using trained neural network policy functions that
include both a capital policy network and an investment decision network.

Architecture (matches ``dl.basic_final`` training code)
-------------------------------------------------------
- **Capital Policy** (sigmoid) -- outputs normalised ``k'`` in [0, 1];
  ``denormalize_capital`` maps to physical units.
- **Investment Policy** (sigmoid) -- outputs invest probability,
  hard-thresholded for binary decision.

Decision logic per time step:
    If ``invest_prob > threshold``:
        ``k' = denormalize(capital_net(inputs))``
    Else:
        ``k' = (1 - delta) * k``
"""

from __future__ import annotations

from typing import Tuple

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
    ProductionFunctions,
)

tfd = tfp.distributions


class DLSimulatorBasicFinal:
    """Simulator for the basic RBC model with investment decision network.

    Uses two trained neural networks:

    - ``capital_policy_net`` (sigmoid) -- outputs normalised ``k'``,
      denormalised to get next-period capital when investing.
    - ``investment_policy_net`` (sigmoid) -- outputs investment probability,
      thresholded for binary go/no-go decision.

    Args:
        config: Deep learning configuration (contains network architecture
            and state-space bounds).
        params: Economic parameters.
        hard_threshold: Threshold for the binary investment decision.
    """

    def __init__(
        self,
        config: DeepLearningConfig,
        params: EconomicParams,
        hard_threshold: float = 0.5,
    ) -> None:
        self.config = config
        self.params = params
        self.hard_threshold = hard_threshold

        self.config.update_value_scale(self.params)
        self.normalizer = StateSpaceNormalizer(config)

        self._build_networks()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        """Build policy and value network architectures.

        Activations match ``BasicModelDL_FINAL``:
        - Capital policy: sigmoid (outputs normalised k').
        - Investment policy: sigmoid (outputs invest probability).
        - Value net: linear with scale factor.
        """
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
        self.value_function_net = NeuralNetFactory.build_mlp(
            input_dim=2, output_dim=1, config=self.config,
            output_activation="linear",
            scale_factor=1.0,
            name="ValueFunctionNet",
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_solved_dl_solution(
        self,
        capital_policy_filepath: str,
        investment_policy_filepath: str,
        value_function_filepath: str,
    ) -> None:
        """Load trained network weights from ``.weights.h5`` files.

        Args:
            capital_policy_filepath: Path to capital policy weights.
            investment_policy_filepath: Path to investment policy weights.
            value_function_filepath: Path to value function weights.
        """
        dummy = tf.zeros((1, 2), dtype=TENSORFLOW_DTYPE)

        # Capital policy
        _ = self.capital_policy_net(dummy)
        try:
            self.capital_policy_net.load_weights(capital_policy_filepath)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load capital policy weights from "
                f"{capital_policy_filepath}: {e}"
            ) from e

        # Investment policy
        _ = self.investment_policy_net(dummy)
        try:
            self.investment_policy_net.load_weights(
                investment_policy_filepath,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load investment policy weights from "
                f"{investment_policy_filepath}: {e}"
            ) from e

        # Value function
        _ = self.value_function_net(dummy)
        try:
            self.value_function_net.load_weights(value_function_filepath)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load value function weights from "
                f"{value_function_filepath}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Normalised-input helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, k: tf.Tensor, z: tf.Tensor,
    ) -> tf.Tensor:
        """Normalise and concatenate ``(k, z)`` for network input.

        Args:
            k: Capital, shape ``(batch, 1)`` or ``(batch,)``.
            z: Productivity, shape ``(batch, 1)`` or ``(batch,)``.

        Returns:
            Concatenated normalised inputs, shape ``(batch, 2)``.
        """
        if len(k.shape) == 1:
            k = tf.reshape(k, (-1, 1))
        if len(z.shape) == 1:
            z = tf.reshape(z, (-1, 1))
        return tf.concat([
            self.normalizer.normalize_capital(k),
            self.normalizer.normalize_productivity(z),
        ], axis=1)

    # ------------------------------------------------------------------
    # Policy / value inference
    # ------------------------------------------------------------------

    @tf.function
    def _get_policy_action(
        self, k_curr: tf.Tensor, z_curr: tf.Tensor,
    ) -> tf.Tensor:
        """Get next-period capital from policy networks.

        Implements the training-code logic:
            k_target = denormalize(capital_policy_net(inputs))
            invest_prob = investment_policy_net(inputs)
            invest = 1{invest_prob > threshold}
            k' = invest * k_target + (1 - invest) * (1 - delta) * k

        Args:
            k_curr: Current capital, shape ``(batch,)``.
            z_curr: Current productivity, shape ``(batch,)``.

        Returns:
            ``k_prime``, shape ``(batch,)``.
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))
        inputs = self._prepare_inputs(k_2d, z_2d)

        # Capital policy: sigmoid → denormalise to physical k'
        k_prime_norm = self.capital_policy_net(inputs, training=False)
        k_prime_invest = self.normalizer.denormalize_capital(k_prime_norm)

        # No-invest branch: depreciation only
        k_prime_no_invest = (
            (1.0 - self.params.depreciation_rate) * k_2d
        )

        # Binary investment decision
        invest_prob = self.investment_policy_net(inputs, training=False)
        invest_decision = tf.cast(
            invest_prob > self.hard_threshold, TENSORFLOW_DTYPE,
        )
        k_prime = (
            invest_decision * k_prime_invest
            + (1.0 - invest_decision) * k_prime_no_invest
        )

        k_prime = tf.clip_by_value(
            k_prime,
            clip_value_min=self.config.capital_min,
            clip_value_max=self.config.capital_max,
        )
        return tf.squeeze(k_prime, axis=-1)

    @tf.function
    def _get_value_function(
        self, k_curr: tf.Tensor, z_curr: tf.Tensor,
    ) -> tf.Tensor:
        """Get value-function estimate.

        Args:
            k_curr: Capital, shape ``(batch,)``.
            z_curr: Productivity, shape ``(batch,)``.

        Returns:
            Value estimates, shape ``(batch,)``.
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))
        inputs = self._prepare_inputs(k_2d, z_2d)
        return tf.squeeze(
            self.value_function_net(inputs, training=False), axis=-1,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_states: Tuple[tf.Tensor, tf.Tensor],
        innovation_sequence: tf.Tensor,
    ) -> dict:
        """Simulate the economy using neural network policies.

        Args:
            initial_states: ``(K_0, Z_0)`` initial-state tensors.
            innovation_sequence: Shock tensor of shape ``(batch, T+1)``.

        Returns:
            Dictionary with keys ``K_curr``, ``K_next``, ``Z_curr``,
            ``Z_next`` as NumPy arrays.
        """
        k_curr = tf.cast(tf.squeeze(initial_states[0]), TENSORFLOW_DTYPE)
        z_curr = tf.cast(tf.squeeze(initial_states[1]), TENSORFLOW_DTYPE)

        eps_path = tf.cast(
            innovation_sequence * self.params.productivity_std_dev,
            TENSORFLOW_DTYPE,
        )
        T = eps_path.shape[1] - 1

        k_history = tf.TensorArray(
            dtype=TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        z_history = tf.TensorArray(
            dtype=TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        k_history = k_history.write(0, k_curr)
        z_history = z_history.write(0, z_curr)

        for t in range(T):
            k_t = k_history.read(t)
            z_t = z_history.read(t)

            k_next = self._get_policy_action(k_t, z_t)
            z_next = TransitionFunctions.log_ar1_transition(
                z_curr=z_t,
                rho=self.params.productivity_persistence,
                epsilon=eps_path[:, t + 1],
            )
            k_history = k_history.write(t + 1, k_next)
            z_history = z_history.write(t + 1, z_next)

        k_path = tf.transpose(k_history.stack(), perm=[1, 0]).numpy()
        z_path = tf.transpose(z_history.stack(), perm=[1, 0]).numpy()

        return {
            "K_curr": k_path[:, :-1],
            "K_next": k_path[:, 1:],
            "Z_curr": z_path[:, :-1],
            "Z_next": z_path[:, 1:],
        }

    # ------------------------------------------------------------------
    # Bellman residual diagnostic
    # ------------------------------------------------------------------

    def simulate_bellman_residual(
        self,
        initial_states: Tuple[tf.Tensor, tf.Tensor],
    ) -> dict:
        """Compute Bellman residuals on a batch of states.

        Uses Monte Carlo integration (100 draws) for the conditional
        expectation of continuation value.

        Args:
            initial_states: ``(K, Z)`` state tensors.

        Returns:
            Dictionary with ``relative_error``, ``absolute_error``,
            ``mean_value``.
        """
        k_curr = tf.cast(tf.squeeze(initial_states[0]), TENSORFLOW_DTYPE)
        z_curr = tf.cast(tf.squeeze(initial_states[1]), TENSORFLOW_DTYPE)

        k_2d = tf.reshape(k_curr, (-1, 1))
        z_2d = tf.reshape(z_curr, (-1, 1))
        inputs = self._prepare_inputs(k_2d, z_2d)

        # --- Invest branch: k' from capital policy (sigmoid → denormalise) ---
        k_prime_norm = self.capital_policy_net(inputs, training=False)
        k_prime_invest = tf.squeeze(
            self.normalizer.denormalize_capital(k_prime_norm), axis=-1,
        )
        k_prime_invest = tf.clip_by_value(
            k_prime_invest, self.config.capital_min, self.config.capital_max,
        )

        # --- No-invest branch ---
        k_prime_no_invest = k_curr * (1.0 - self.params.depreciation_rate)

        # --- Cash flow ---
        profit = ProductionFunctions.cobb_douglas(k_curr, z_curr, self.params)

        investment = k_prime_invest - (1.0 - self.params.depreciation_rate) * k_curr
        adj_cost, _ = AdjustmentCostCalculator.calculate_with_grad(
            investment, k_curr, 1.0, self.params,
        )
        cash_flow_invest = profit - investment - adj_cost
        cash_flow_no_invest = profit

        # --- Monte Carlo continuation values ---
        mc_num = 100
        eps = tfd.Normal(0.0, self.params.productivity_std_dev).sample(
            [tf.shape(k_curr)[0], mc_num],
        )
        z_curr_exp = tf.repeat(
            tf.expand_dims(z_curr, axis=1), mc_num, axis=1,
        )
        z_next = TransitionFunctions.log_ar1_transition(
            z_curr_exp, self.params.productivity_persistence, eps,
        )

        # Invest branch continuation
        k_inv_exp = tf.repeat(
            tf.expand_dims(k_prime_invest, axis=1), mc_num, axis=1,
        )
        ev_invest = tf.reduce_mean(
            tf.reshape(
                self._get_value_function(
                    tf.reshape(k_inv_exp, [-1]),
                    tf.reshape(z_next, [-1]),
                ),
                [tf.shape(k_curr)[0], mc_num],
            ),
            axis=1,
        )

        # No-invest branch continuation
        k_noinv_exp = tf.repeat(
            tf.expand_dims(k_prime_no_invest, axis=1), mc_num, axis=1,
        )
        ev_no_invest = tf.reduce_mean(
            tf.reshape(
                self._get_value_function(
                    tf.reshape(k_noinv_exp, [-1]),
                    tf.reshape(z_next, [-1]),
                ),
                [tf.shape(k_curr)[0], mc_num],
            ),
            axis=1,
        )

        # --- Bellman RHS = max(invest, no-invest) ---
        beta = self.params.discount_factor
        rhs_invest = cash_flow_invest + beta * ev_invest
        rhs_no_invest = cash_flow_no_invest + beta * ev_no_invest
        rhs = tf.maximum(rhs_invest, rhs_no_invest)

        v_curr = self._get_value_function(k_curr, z_curr)
        residuals = rhs - v_curr
        rel_error = tf.abs(residuals) / (tf.abs(rhs) + 1e-8)

        return {
            "relative_error": float(tf.reduce_mean(rel_error)),
            "absolute_error": float(tf.reduce_mean(tf.abs(residuals))),
            "mean_value": float(tf.reduce_mean(v_curr)),
        }

    # ------------------------------------------------------------------
    # Lifetime reward diagnostic
    # ------------------------------------------------------------------

    def simulate_life_time_reward(
        self,
        initial_states: Tuple[tf.Tensor, tf.Tensor],
        innovation_sequence: tf.Tensor,
    ) -> tf.Tensor:
        """Compute mean discounted lifetime reward.

        Args:
            initial_states: ``(K_0, Z_0)`` initial-state tensors.
            innovation_sequence: Shock tensor of shape ``(batch, T+1)``.

        Returns:
            Scalar tensor — mean accumulated discounted reward.
        """
        k_t = tf.cast(tf.squeeze(initial_states[0]), TENSORFLOW_DTYPE)
        z_t = tf.cast(tf.squeeze(initial_states[1]), TENSORFLOW_DTYPE)
        eps_path = tf.cast(
            innovation_sequence * self.params.productivity_std_dev,
            TENSORFLOW_DTYPE,
        )
        batch_size = eps_path.shape[0]
        T = eps_path.shape[1] - 1

        accumulated_reward = tf.zeros(
            shape=(batch_size,), dtype=TENSORFLOW_DTYPE,
        )
        delta = self.params.depreciation_rate
        beta = self.params.discount_factor

        for t in range(T):
            k_2d = tf.reshape(k_t, (-1, 1))
            z_2d = tf.reshape(z_t, (-1, 1))
            inputs = self._prepare_inputs(k_2d, z_2d)

            # Investment decision
            invest_prob = self.investment_policy_net(inputs, training=False)
            invest_decision = tf.cast(
                invest_prob > self.hard_threshold, TENSORFLOW_DTYPE,
            )
            invest_decision = tf.squeeze(invest_decision, axis=-1)

            # Capital target from sigmoid policy
            k_prime_norm = self.capital_policy_net(inputs, training=False)
            k_target = tf.squeeze(
                self.normalizer.denormalize_capital(k_prime_norm), axis=-1,
            )

            k_depreciated = k_t * (1.0 - delta)
            k_next = (
                invest_decision * k_target
                + (1.0 - invest_decision) * k_depreciated
            )
            k_next = tf.clip_by_value(
                k_next, self.config.capital_min, self.config.capital_max,
            )

            # Cash flow
            profit = ProductionFunctions.cobb_douglas(k_t, z_t, self.params)
            investment = k_next - k_depreciated

            adj_cost_if_adjust, _ = AdjustmentCostCalculator.calculate(
                investment, k_t, self.params,
            )
            adjustment_cost = invest_decision * adj_cost_if_adjust

            cash_flow = profit - investment - adjustment_cost
            accumulated_reward = accumulated_reward + cash_flow * (beta ** t)

            # Transition
            z_t = TransitionFunctions.log_ar1_transition(
                z_t, self.params.productivity_persistence,
                eps_path[:, t + 1],
            )
            k_t = k_next

        return tf.reduce_mean(accumulated_reward)

    # ------------------------------------------------------------------
    # Value-function gap diagnostic
    # ------------------------------------------------------------------

    def compute_value_function_gap(
        self,
        grid_points: Tuple[tf.Tensor, tf.Tensor],
        value_labels: tf.Tensor,
    ) -> dict[str, float]:
        """Compute gap between predicted and true value function on a grid.

        Args:
            grid_points: ``(K_grid, Z_grid)`` from ``meshgrid``.
            value_labels: True VFI value function values.

        Returns:
            Dictionary with ``mae`` and ``mape``.
        """
        k_flat = tf.reshape(tf.cast(grid_points[0], TENSORFLOW_DTYPE), [-1])
        z_flat = tf.reshape(tf.cast(grid_points[1], TENSORFLOW_DTYPE), [-1])
        v_true = tf.reshape(tf.cast(value_labels, TENSORFLOW_DTYPE), [-1])

        v_pred = tf.reshape(self._get_value_function(k_flat, z_flat), [-1])
        residual = v_true - v_pred

        mae = tf.reduce_mean(tf.abs(residual))
        mape = tf.reduce_mean(
            tf.abs(residual) / (tf.abs(v_true) + 1e-4)
        )
        return {
            "mae": float(mae.numpy()),
            "mape": float(mape.numpy()),
        }
