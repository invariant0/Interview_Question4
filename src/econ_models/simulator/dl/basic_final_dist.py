# src/econ_models/simulator/dl/basic_final_dist.py
"""Deep Learning Simulator for the distributional Basic RBC Model.

Simulates the economy using trained neural network policy functions
that take 6-dimensional inputs ``(k, z, rho, std, convex, fixed)``
spanning the distribution of economic parameters.

Architecture (matches ``dl_dist.basic_final`` training code)
-------------------------------------------------------------
- **Capital Policy** (sigmoid) -- outputs normalised ``k'`` in [0, 1];
  ``denormalize_capital`` maps to physical units.
- **Investment Policy** (sigmoid) -- outputs invest probability,
  hard-thresholded for binary decision.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import ParamSpaceNormalizer, StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE


class DLSimulatorBasicFinal_dist:
    """Simulator for the distributional basic RBC model.

    Uses two trained neural networks with 6-dim inputs:

    - ``capital_policy_net`` (sigmoid) -- outputs normalised ``k'``,
      denormalised to get next-period capital when investing.
    - ``investment_policy_net`` (sigmoid) -- outputs investment probability,
      thresholded for a binary go/no-go decision.

    Args:
        config: Deep learning configuration (network architecture, bounds).
        bonds: Parameter-space bounds for normalisation.
        hard_threshold: Threshold for the binary investment decision.
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
        """Build policy network architectures.

        Activations match ``BasicModelDL_dist_FINAL``:
        - Capital policy: sigmoid (outputs normalised k').
        - Investment policy: sigmoid (outputs invest probability).
        """
        self.capital_policy_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="sigmoid",
            name="CapitalPolicyNet",
        )
        self.investment_policy_net = NeuralNetFactory.build_mlp(
            input_dim=6, output_dim=1, config=self.config,
            output_activation="sigmoid", scale_factor=1.0,
            name="InvestmentPolicyNet",
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_solved_dl_solution(
        self,
        capital_policy_filepath: str,
        investment_policy_filepath: str,
    ) -> None:
        """Load trained network weights from ``.weights.h5`` files.

        Args:
            capital_policy_filepath: Path to capital policy weights.
            investment_policy_filepath: Path to investment policy weights.
        """
        dummy = tf.zeros((1, 6), dtype=TENSORFLOW_DTYPE)

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

    # ------------------------------------------------------------------
    # Normalised-input helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        k: tf.Tensor,
        z: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
    ) -> tf.Tensor:
        """Normalise and concatenate the 6-dim input vector.

        Args:
            k: Capital, shape ``(batch, 1)`` or ``(batch,)``.
            z: Productivity, shape ``(batch, 1)`` or ``(batch,)``.
            rho: AR(1) persistence, shape ``(batch, 1)`` or ``(batch,)``.
            std: Productivity std dev, shape ``(batch, 1)`` or ``(batch,)``.
            convex: Convex adjustment cost, shape ``(batch, 1)`` or ``(batch,)``.
            fixed: Fixed adjustment cost, shape ``(batch, 1)`` or ``(batch,)``.

        Returns:
            Concatenated normalised inputs, shape ``(batch, 6)``.
        """
        tensors = [k, z, rho, std, convex, fixed]
        for i, t in enumerate(tensors):
            if len(t.shape) == 1:
                tensors[i] = tf.reshape(t, (-1, 1))
        k, z, rho, std, convex, fixed = tensors

        return tf.concat([
            self.normalizer_states.normalize_capital(k),
            self.normalizer_states.normalize_productivity(z),
            self.normalizer_params.normalize_rho(rho),
            self.normalizer_params.normalize_std(std),
            self.normalizer_params.normalize_convex(convex),
            self.normalizer_params.normalize_fixed(fixed),
        ], axis=1)

    # ------------------------------------------------------------------
    # Policy inference
    # ------------------------------------------------------------------

    @tf.function
    def _get_policy_action(
        self,
        k_curr: tf.Tensor,
        z_curr: tf.Tensor,
        rho: tf.Tensor,
        std: tf.Tensor,
        convex: tf.Tensor,
        fixed: tf.Tensor,
        depreciation_rate: float,
    ) -> tf.Tensor:
        """Get next-period capital from policy networks.

        Args:
            k_curr: Current capital, shape ``(batch,)``.
            z_curr: Current productivity, shape ``(batch,)``.
            rho: AR(1) persistence tensor.
            std: Productivity std dev tensor.
            convex: Convex adjustment cost tensor.
            fixed: Fixed adjustment cost tensor.
            depreciation_rate: Rate of depreciation.

        Returns:
            ``k_prime``, shape ``(batch,)``.
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        inputs = self._prepare_inputs(k_2d, z_curr, rho, std, convex, fixed)

        # Capital policy: sigmoid → denormalise to physical k'
        k_prime_norm = self.capital_policy_net(inputs, training=False)
        k_prime_invest = self.normalizer_states.denormalize_capital(
            k_prime_norm,
        )

        # No-invest branch: depreciation only
        k_prime_no_invest = (1.0 - depreciation_rate) * k_2d

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

    # ------------------------------------------------------------------
    # Compiled simulation loop
    # ------------------------------------------------------------------

    def _simulate_step(
        self,
        k_t: tf.Tensor,
        z_t: tf.Tensor,
        eps_t: tf.Tensor,
        rho_tensor: tf.Tensor,
        std_tensor: tf.Tensor,
        convex_tensor: tf.Tensor,
        fixed_tensor: tf.Tensor,
        rho_scalar: tf.Tensor,
        depreciation_rate: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Single time-step: policy + AR(1) transition (all TF ops)."""
        k_2d = tf.reshape(k_t, (-1, 1))
        inputs = self._prepare_inputs(
            k_2d, z_t, rho_tensor, std_tensor, convex_tensor, fixed_tensor,
        )

        # Capital policy
        k_prime_norm = self.capital_policy_net(inputs, training=False)
        k_prime_invest = self.normalizer_states.denormalize_capital(k_prime_norm)

        # No-invest branch
        k_prime_no_invest = (1.0 - depreciation_rate) * k_2d

        # Binary investment decision
        invest_prob = self.investment_policy_net(inputs, training=False)
        invest_decision = tf.cast(
            invest_prob > self.hard_threshold, TENSORFLOW_DTYPE,
        )
        k_next = (
            invest_decision * k_prime_invest
            + (1.0 - invest_decision) * k_prime_no_invest
        )
        k_next = tf.clip_by_value(
            k_next,
            clip_value_min=self.config.capital_min,
            clip_value_max=self.config.capital_max,
        )
        k_next = tf.squeeze(k_next, axis=-1)

        # AR(1) productivity transition
        z_safe = tf.maximum(z_t, tf.cast(1e-12, TENSORFLOW_DTYPE))
        ln_z_prime = rho_scalar * tf.math.log(z_safe) + eps_t
        z_next = tf.exp(ln_z_prime)

        return k_next, z_next

    @tf.function(reduce_retracing=True)
    def _simulate_compiled(
        self,
        k_init: tf.Tensor,
        z_init: tf.Tensor,
        eps_path: tf.Tensor,
        rho_tensor: tf.Tensor,
        std_tensor: tf.Tensor,
        convex_tensor: tf.Tensor,
        fixed_tensor: tf.Tensor,
        rho_scalar: tf.Tensor,
        depreciation_rate: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Run the full T-step simulation inside a compiled TF graph.

        Returns stacked ``(k_path, z_path)``, each of shape ``(T+1, batch)``.
        """
        T = tf.shape(eps_path)[1] - 1

        k_ta = tf.TensorArray(
            dtype=TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        z_ta = tf.TensorArray(
            dtype=TENSORFLOW_DTYPE, size=T + 1,
            dynamic_size=False, clear_after_read=False,
        )
        k_ta = k_ta.write(0, k_init)
        z_ta = z_ta.write(0, z_init)

        k_t = k_init
        z_t = z_init
        for t in tf.range(T):
            eps_t = eps_path[:, t + 1]
            k_t, z_t = self._simulate_step(
                k_t, z_t, eps_t,
                rho_tensor, std_tensor, convex_tensor, fixed_tensor,
                rho_scalar, depreciation_rate,
            )
            k_ta = k_ta.write(t + 1, k_t)
            z_ta = z_ta.write(t + 1, z_t)

        return k_ta.stack(), z_ta.stack()

    # ------------------------------------------------------------------
    # Public simulation API
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_states: Tuple[tf.Tensor, tf.Tensor],
        innovation_sequence: tf.Tensor,
        econ_params: EconomicParams,
    ) -> dict:
        """Simulate the economy using neural network policies.

        The heavy time-step loop runs inside a ``@tf.function`` compiled
        graph, eliminating Python↔TF dispatch overhead.

        Args:
            initial_states: ``(K_0, Z_0)`` initial-state tensors.
            innovation_sequence: Shock tensor of shape ``(batch, T+1)``.
            econ_params: Economic parameters for the simulation run.

        Returns:
            Dictionary with keys ``K_curr``, ``K_next``, ``Z_curr``,
            ``Z_next`` as NumPy arrays.
        """
        ones_like_k = tf.ones_like(
            initial_states[0], dtype=TENSORFLOW_DTYPE,
        )
        rho_tensor = (
            tf.constant(econ_params.productivity_persistence, dtype=TENSORFLOW_DTYPE)
            * ones_like_k
        )
        std_tensor = (
            tf.constant(econ_params.productivity_std_dev, dtype=TENSORFLOW_DTYPE)
            * ones_like_k
        )
        convex_tensor = (
            tf.constant(econ_params.adjustment_cost_convex, dtype=TENSORFLOW_DTYPE)
            * ones_like_k
        )
        fixed_tensor = (
            tf.constant(econ_params.adjustment_cost_fixed, dtype=TENSORFLOW_DTYPE)
            * ones_like_k
        )

        k_init = tf.cast(tf.squeeze(initial_states[0]), TENSORFLOW_DTYPE)
        z_init = tf.cast(tf.squeeze(initial_states[1]), TENSORFLOW_DTYPE)

        eps_path = tf.cast(
            innovation_sequence * econ_params.productivity_std_dev,
            TENSORFLOW_DTYPE,
        )

        rho_scalar = tf.constant(
            econ_params.productivity_persistence, dtype=TENSORFLOW_DTYPE,
        )
        dep_rate = tf.constant(
            econ_params.depreciation_rate, dtype=TENSORFLOW_DTYPE,
        )

        # ----- compiled loop (traced once, reused) -----
        k_stack, z_stack = self._simulate_compiled(
            k_init, z_init, eps_path,
            rho_tensor, std_tensor, convex_tensor, fixed_tensor,
            rho_scalar, dep_rate,
        )

        k_path = tf.transpose(k_stack, perm=[1, 0]).numpy()
        z_path = tf.transpose(z_stack, perm=[1, 0]).numpy()

        return {
            "K_curr": k_path[:, :-1],
            "K_next": k_path[:, 1:],
            "Z_curr": z_path[:, :-1],
            "Z_next": z_path[:, 1:],
        }
 