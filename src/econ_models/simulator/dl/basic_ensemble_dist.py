# src/econ_models/simulator/dl/basic_ensemble_dist.py
"""Ensemble Deep Learning Simulator for the distributional Basic RBC Model.

Loads *N* independently trained copies of the capital and investment policy
networks (e.g. from different random-seed runs) and averages their outputs
at every simulation step.  This reduces variance from seed-dependent local
optima while preserving the distributional conditioning on (ρ, σ, γ, F).

Ensemble strategy
-----------------
- **Capital policy**: Each member produces a normalised k' ∈ [0, 1].
  The ensemble **averages the normalised outputs** before denormalising
  to physical units.  This is equivalent to averaging in the sigmoid
  output space, which keeps the average inside the valid range.
- **Investment policy**: Each member produces an invest probability.
  The ensemble **averages the probabilities** and applies the hard
  threshold to the averaged value.  This gives a "majority vote" style
  decision with soft weighting.

Usage::

    sim = DLSimulatorBasicEnsemble_dist(config, bonds, n_members=4)
    sim.load_ensemble_weights(
        capital_paths=["run_0/cap.h5", "run_1/cap.h5", ...],
        investment_paths=["run_0/inv.h5", "run_1/inv.h5", ...],
    )
    results = sim.simulate(initial_states, shocks, econ_params)
"""

from __future__ import annotations

from typing import List, Tuple

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.config.economic_params import EconomicParams
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.standardize import ParamSpaceNormalizer, StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE


class DLSimulatorBasicEnsemble_dist:
    """Ensemble simulator for the distributional basic RBC model.

    Holds *n_members* copies of the capital + investment policy networks,
    averages their outputs at inference time, and simulates the economy
    using the averaged policy.

    Args:
        config: Deep learning configuration (network architecture, bounds).
        bonds: Parameter-space bounds for normalisation.
        n_members: Number of ensemble members.
        hard_threshold: Threshold for the binary investment decision
            applied to the **averaged** investment probability.
    """

    def __init__(
        self,
        config: DeepLearningConfig,
        bonds: dict,
        n_members: int = 4,
        hard_threshold: float = 0.5,
    ) -> None:
        self.config = config
        self.n_members = n_members
        self.normalizer_states = StateSpaceNormalizer(config)
        self.normalizer_params = ParamSpaceNormalizer(bonds)
        self.hard_threshold = hard_threshold
        self._build_networks()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_networks(self) -> None:
        """Build *n_members* copies of both policy networks."""
        self.capital_policy_nets: List[tf.keras.Model] = []
        self.investment_policy_nets: List[tf.keras.Model] = []

        for i in range(self.n_members):
            cap_net = NeuralNetFactory.build_mlp(
                input_dim=6, output_dim=1, config=self.config,
                output_activation="sigmoid",
                name=f"CapitalPolicyNet_{i}",
            )
            inv_net = NeuralNetFactory.build_mlp(
                input_dim=6, output_dim=1, config=self.config,
                output_activation="sigmoid", scale_factor=1.0,
                name=f"InvestmentPolicyNet_{i}",
            )
            self.capital_policy_nets.append(cap_net)
            self.investment_policy_nets.append(inv_net)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_ensemble_weights(
        self,
        capital_paths: List[str],
        investment_paths: List[str],
    ) -> None:
        """Load trained weights for every ensemble member.

        Args:
            capital_paths: One ``.weights.h5`` path per member (capital policy).
            investment_paths: One ``.weights.h5`` path per member (investment policy).

        Raises:
            ValueError: If the number of paths doesn't match *n_members*.
            FileNotFoundError: If a weight file cannot be loaded.
        """
        if len(capital_paths) != self.n_members:
            raise ValueError(
                f"Expected {self.n_members} capital weight paths, "
                f"got {len(capital_paths)}."
            )
        if len(investment_paths) != self.n_members:
            raise ValueError(
                f"Expected {self.n_members} investment weight paths, "
                f"got {len(investment_paths)}."
            )

        dummy = tf.zeros((1, 6), dtype=TENSORFLOW_DTYPE)

        for i in range(self.n_members):
            # Capital policy
            _ = self.capital_policy_nets[i](dummy)
            try:
                self.capital_policy_nets[i].load_weights(capital_paths[i])
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load capital policy weights for member {i} "
                    f"from {capital_paths[i]}: {e}"
                ) from e

            # Investment policy
            _ = self.investment_policy_nets[i](dummy)
            try:
                self.investment_policy_nets[i].load_weights(investment_paths[i])
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load investment policy weights for member {i} "
                    f"from {investment_paths[i]}: {e}"
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
    # Ensemble policy inference
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
        """Get next-period capital from the **ensemble-averaged** policies.

        1. Each member produces normalised k' and invest probability.
        2. Normalised k' values are averaged → single denormalised k'_invest.
        3. Invest probabilities are averaged → single hard-threshold decision.

        Returns:
            ``k_prime``, shape ``(batch,)``.
        """
        k_2d = tf.reshape(k_curr, (-1, 1))
        inputs = self._prepare_inputs(k_2d, z_curr, rho, std, convex, fixed)

        # --- Accumulate ensemble outputs ---
        cap_sum = tf.zeros_like(k_2d)
        inv_sum = tf.zeros_like(k_2d)

        for i in range(self.n_members):
            cap_sum = cap_sum + self.capital_policy_nets[i](inputs, training=False)
            inv_sum = inv_sum + self.investment_policy_nets[i](inputs, training=False)

        n = tf.constant(self.n_members, dtype=TENSORFLOW_DTYPE)
        k_prime_norm_avg = cap_sum / n      # averaged normalised k'
        invest_prob_avg = inv_sum / n       # averaged invest probability

        # Denormalise the averaged normalised capital
        k_prime_invest = self.normalizer_states.denormalize_capital(
            k_prime_norm_avg,
        )

        # No-invest branch: depreciation only
        k_prime_no_invest = (1.0 - depreciation_rate) * k_2d

        # Binary investment decision on the averaged probability
        invest_decision = tf.cast(
            invest_prob_avg > self.hard_threshold, TENSORFLOW_DTYPE,
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
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_states: Tuple[tf.Tensor, tf.Tensor],
        innovation_sequence: tf.Tensor,
        econ_params: EconomicParams,
    ) -> dict:
        """Simulate the economy using ensemble-averaged neural network policies.

        Args:
            initial_states: ``(K_0, Z_0)`` initial-state tensors.
            innovation_sequence: Shock tensor of shape ``(batch, T+1)``.
            econ_params: Economic parameters for the simulation run.

        Returns:
            Dictionary with keys ``K_curr``, ``K_next``, ``Z_curr``,
            ``Z_next`` as NumPy arrays.
        """
        # Build parameter tensors matching batch dimension
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

        k_curr = tf.cast(tf.squeeze(initial_states[0]), TENSORFLOW_DTYPE)
        z_curr = tf.cast(tf.squeeze(initial_states[1]), TENSORFLOW_DTYPE)

        eps_path = tf.cast(
            innovation_sequence * econ_params.productivity_std_dev,
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

            k_next = self._get_policy_action(
                k_t, z_t,
                rho_tensor, std_tensor, convex_tensor, fixed_tensor,
                econ_params.depreciation_rate,
            )
            z_next = TransitionFunctions.log_ar1_transition(
                z_curr=z_t,
                rho=econ_params.productivity_persistence,
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
