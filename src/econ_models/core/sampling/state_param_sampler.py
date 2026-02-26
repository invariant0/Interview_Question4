# econ_models/core/sampling/state_param_sampler.py
"""
State variable sampling for deep learning training.

This module provides functions for generating random samples of
state variables with optional curriculum learning support.

Sampling strategies
-------------------
- **Independent** (``sample_states_params_gpu``): Each sample draws states
  and economic parameters independently.  Fast but high-variance because
  each parameter combo sees only a random slice of state space.
- **Cross-product** (``sample_states_params_cross_product_gpu``): Draws
  *N_s* states and *N_p* parameter combos, then tiles them into an
  *N_s × N_p* batch so that **every parameter sees the full state sample**.
  Lower variance at the same total batch size, which stabilises training.
"""

from typing import Tuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.core.types import TENSORFLOW_DTYPE

tfd = tfp.distributions

class StateParamSampler:
    """Sample economic state variables and model parameters jointly.

    Provide two sampling strategies for generating training batches
    that pair state-space points (k, z, [b]) with economic-parameter
    combos (rho, std, convex, fixed, [eta0, eta1]):

    * **Independent** (``sample_states_params_gpu``) -- each sample
      draws states and parameters independently.  Fast but
      high-variance because each parameter combo sees only a random
      slice of state space.
    * **Cross-product** (``sample_states_params_cross_product_gpu``)
      -- draws *N_s* states and *N_p* parameter combos, then tiles
      them into an *N_s x N_p* batch so that every parameter sees
      the full state sample.
    """

    # ------------------------------------------------------------------
    # Cross-product sampling (new)
    # ------------------------------------------------------------------

    @staticmethod
    def sample_states_params_cross_product_gpu(
        n_states: int,
        n_params: int,
        bonds_config: dict,
        include_debt: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        """Sample states and params independently then form their cross product.

        Draws ``n_states`` state points (k, z) and ``n_params`` economic-
        parameter combos (rho, std, convex, fixed), then tiles them so that
        every parameter combo is paired with every state point.

        The total output batch size is ``n_states * n_params``.

        This ensures that each economic-parameter region receives gradient
        signal from the **full** sampled state space, eliminating the
        sampling-noise imbalance that causes seed-dependent instability.

        Args:
            n_states:  Number of distinct (k, z) points.
            n_params:  Number of distinct (rho, std, convex, fixed) combos.
            bonds_config: Sampling bounds dictionary.
            include_debt: Whether to include debt + equity issuance params.

        Returns:
            Tuple of tensors each with shape ``(n_states * n_params, 1)``.
        """
        # --- Sample states: (n_states, 1) ---
        s_shape = (n_states, 1)
        k_s = tf.random.uniform(s_shape, minval=bonds_config['k_min'], maxval=bonds_config['k_max'], dtype=TENSORFLOW_DTYPE)
        z_s = tf.random.uniform(s_shape, minval=bonds_config['z_min'], maxval=bonds_config['z_max'], dtype=TENSORFLOW_DTYPE)

        # --- Sample params: (n_params, 1) ---
        p_shape = (n_params, 1)
        rho_p = tf.random.uniform(p_shape, minval=bonds_config['rho_min'], maxval=bonds_config['rho_max'], dtype=TENSORFLOW_DTYPE)
        std_p = tf.random.uniform(p_shape, minval=bonds_config['std_min'], maxval=bonds_config['std_max'], dtype=TENSORFLOW_DTYPE)
        convex_p = tf.random.uniform(p_shape, minval=bonds_config['convex_min'], maxval=bonds_config['convex_max'], dtype=TENSORFLOW_DTYPE)
        fixed_p = tf.random.uniform(p_shape, minval=bonds_config['fixed_min'], maxval=bonds_config['fixed_max'], dtype=TENSORFLOW_DTYPE)

        # --- Cross-product via tile + repeat ---
        # States: repeat each state n_params times  → (n_states * n_params, 1)
        # Params: tile the whole param block n_states times → (n_states * n_params, 1)
        k = tf.repeat(k_s, repeats=n_params, axis=0)       # [s0,s0,..,s1,s1,..]
        z = tf.repeat(z_s, repeats=n_params, axis=0)

        rho = tf.tile(rho_p, [n_states, 1])                # [p0,p1,..,p0,p1,..]
        std = tf.tile(std_p, [n_states, 1])
        convex = tf.tile(convex_p, [n_states, 1])
        fixed = tf.tile(fixed_p, [n_states, 1])

        if not include_debt:
            return k, z, rho, std, convex, fixed

        # Debt variables are state-like — sample n_states and repeat
        b_s = tf.random.uniform(s_shape, minval=bonds_config['b_min'], maxval=bonds_config['b_max'], dtype=TENSORFLOW_DTYPE)
        b = tf.repeat(b_s, repeats=n_params, axis=0)

        # Equity issuance costs are param-like — sample n_params and tile
        eta0_p = tf.random.uniform(p_shape, minval=bonds_config['eta0_min'], maxval=bonds_config['eta0_max'], dtype=TENSORFLOW_DTYPE)
        eta1_p = tf.random.uniform(p_shape, minval=bonds_config['eta1_min'], maxval=bonds_config['eta1_max'], dtype=TENSORFLOW_DTYPE)
        eta0 = tf.tile(eta0_p, [n_states, 1])
        eta1 = tf.tile(eta1_p, [n_states, 1])

        return k, b, z, rho, std, convex, fixed, eta0, eta1

    # ------------------------------------------------------------------
    # Original independent sampling
    # ------------------------------------------------------------------

    @staticmethod
    def sample_states_params_gpu(
        batch_size: int,
        bonds_config: dict,
        include_debt: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        """
        Generate random state+param samples using tf.random.uniform (XLA-compatible).

        GPU-friendly counterpart of sample_states_params(). Uses tf.random.uniform
        instead of tfp.distributions.Beta for full XLA compatibility and direct GPU
        execution without CPU→GPU transfer. With Beta(1,1), the distributions are
        identical (uniform).

        Args:
            batch_size: Number of samples to generate.
            bonds_config: Dict with sampling bounds for all state/param variables.
            include_debt: Whether to include debt + equity issuance params.

        Returns:
            Tuple of tensors with shape (batch_size, 1).
        """
        shape = (batch_size, 1)

        k = tf.random.uniform(shape, minval=bonds_config['k_min'], maxval=bonds_config['k_max'], dtype=TENSORFLOW_DTYPE)
        z = tf.random.uniform(shape, minval=bonds_config['z_min'], maxval=bonds_config['z_max'], dtype=TENSORFLOW_DTYPE)
        rho = tf.random.uniform(shape, minval=bonds_config['rho_min'], maxval=bonds_config['rho_max'], dtype=TENSORFLOW_DTYPE)
        std = tf.random.uniform(shape, minval=bonds_config['std_min'], maxval=bonds_config['std_max'], dtype=TENSORFLOW_DTYPE)
        convex = tf.random.uniform(shape, minval=bonds_config['convex_min'], maxval=bonds_config['convex_max'], dtype=TENSORFLOW_DTYPE)
        fixed = tf.random.uniform(shape, minval=bonds_config['fixed_min'], maxval=bonds_config['fixed_max'], dtype=TENSORFLOW_DTYPE)

        if not include_debt:
            return k, z, rho, std, convex, fixed

        b = tf.random.uniform(shape, minval=bonds_config['b_min'], maxval=bonds_config['b_max'], dtype=TENSORFLOW_DTYPE)
        eta0 = tf.random.uniform(shape, minval=bonds_config['eta0_min'], maxval=bonds_config['eta0_max'], dtype=TENSORFLOW_DTYPE)
        eta1 = tf.random.uniform(shape, minval=bonds_config['eta1_min'], maxval=bonds_config['eta1_max'], dtype=TENSORFLOW_DTYPE)
        return k, b, z, rho, std, convex, fixed, eta0, eta1

    @staticmethod
    def sample_states_params(
        batch_size: int,
        bonds_config: dict,
        include_debt: bool = False,
    ) -> Tuple[tf.Tensor, ...]:
        """Generate random samples of state variables and model parameters.

        Use scaled Beta(1, 1) distributions (equivalent to uniform) for
        each variable, scaled to the domain specified in *bonds_config*.

        Args:
            batch_size: Number of samples to generate.
            bonds_config: Dictionary with sampling bounds for all
                state and parameter variables.
            include_debt: Whether to include debt and equity-issuance
                cost parameters.

        Returns:
            Tuple of tensors each with shape ``(batch_size, 1)``.
        """

        k_samples = StateParamSampler._sample_capital(
            batch_size, bonds_config
        )
        z_samples = StateParamSampler._sample_productivity(
            batch_size, bonds_config
        )
        rho_samples = StateParamSampler._sample_rho(
            batch_size, bonds_config
        )
        std_samples = StateParamSampler._sample_std(
            batch_size, bonds_config
        )
        convex_samples = StateParamSampler._sample_convex(
            batch_size, bonds_config
        )
        fix_samples = StateParamSampler._sample_fix(
            batch_size, bonds_config
        )

        if not include_debt:
            return k_samples, z_samples, rho_samples, std_samples, convex_samples, fix_samples

        eta_fixed_samples = StateParamSampler._sample_equity_issuance_fixed(
            batch_size, bonds_config
        )
        eta_linear_samples = StateParamSampler._sample_equity_issuance_linear(
            batch_size, bonds_config
        )

        b_samples = StateParamSampler._sample_debt(
            batch_size, bonds_config
        )
        return k_samples, b_samples, z_samples, rho_samples, std_samples, convex_samples, fix_samples, eta_fixed_samples, eta_linear_samples

    @staticmethod
    def _sample_capital(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample capital values with curriculum bounds."""
        k_min, k_max = bonds_config['k_min'], bonds_config['k_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=k_min, maxval=k_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_productivity(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample productivity values with curriculum bounds."""
        z_min, z_max = bonds_config['z_min'], bonds_config['z_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=z_min, maxval=z_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_debt(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample debt values with curriculum bounds."""
        b_min, b_max = bonds_config['b_min'], bonds_config['b_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=b_min, maxval=b_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_rho(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample rho values with curriculum bounds."""
        rho_min, rho_max = bonds_config['rho_min'], bonds_config['rho_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=rho_min, maxval=rho_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_std(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample std values with curriculum bounds."""
        std_min, std_max = bonds_config['std_min'], bonds_config['std_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=std_min, maxval=std_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_convex(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample convex adjustment cost values with curriculum bounds."""
        convex_min, convex_max = bonds_config['convex_min'], bonds_config['convex_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=convex_min, maxval=convex_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_fix(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample fixed adjustment cost values with curriculum bounds."""
        fixed_min, fixed_max = bonds_config['fixed_min'], bonds_config['fixed_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=fixed_min, maxval=fixed_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_equity_issuance_fixed(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample equity issuance fixed cost values with curriculum bounds."""
        equity_issuance_fixed_min = bonds_config['eta0_min']
        equity_issuance_fixed_max = bonds_config['eta0_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=equity_issuance_fixed_min, maxval=equity_issuance_fixed_max
        )
        return tf.reshape(samples, (-1, 1))
    
    @staticmethod
    def _sample_equity_issuance_linear(
        batch_size: tf.Tensor,
        bonds_config: dict,
    ) -> tf.Tensor:
        """Sample equity issuance linear cost values with curriculum bounds."""
        equity_issuance_linear_min = bonds_config['eta1_min']
        equity_issuance_linear_max = bonds_config['eta1_max']
        samples = StateParamSampler._sample_beta_scaled(
            shape=(batch_size,), minval=equity_issuance_linear_min, maxval=equity_issuance_linear_max
        )
        return tf.reshape(samples, (-1, 1))

    @staticmethod
    def _sample_beta_scaled(
        shape: Tuple[int, ...],
        minval: tf.Tensor,
        maxval: tf.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> tf.Tensor:
        """
        Sample from a scaled Beta distribution.

        With alpha=beta=1.0, this is equivalent to uniform sampling.

        Args:
            shape: Output shape.
            minval: Minimum value.
            maxval: Maximum value.
            alpha: Beta distribution alpha parameter.
            beta: Beta distribution beta parameter.

        Returns:
            Samples scaled to [minval, maxval].
        """
        alpha_t = tf.cast(alpha, TENSORFLOW_DTYPE)
        beta_t = tf.cast(beta, TENSORFLOW_DTYPE)
        dist = tfd.Beta(concentration1=alpha_t, concentration0=beta_t)
        beta_samples = dist.sample(shape)
        return minval + (maxval - minval) * beta_samples