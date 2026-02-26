# econ_models/dl_dist/training/dataset_builder.py
"""
Dataset construction for deep learning training.

This module provides utilities for building TensorFlow data pipelines
that sample state variables for training.

Two sampling strategies are available:

- **Independent** (default, ``cross_product=False``): Each element in the
  batch has independently drawn states and parameters.  Total batch size
  equals ``dl_config.batch_size``.

- **Cross-product** (``cross_product=True``): Draws ``n_states`` state
  points and ``n_params`` parameter combos, then tiles them into
  ``n_states × n_params`` total samples so every parameter combo sees the
  same full set of states.  This greatly reduces gradient variance across
  the parameter space and improves training stability.  The effective
  batch size is ``n_states × n_params`` (which should equal
  ``dl_config.batch_size`` for a fair comparison).
"""

import math

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.sampling.state_param_sampler import StateParamSampler
from econ_models.core.types import TENSORFLOW_DTYPE


class DatasetBuilder:
    """
    Factory for creating training data pipelines.

    This class constructs efficient TensorFlow Dataset objects that
    sample economic state variables with curriculum learning support.
    """

    @staticmethod
    def build_dataset(
        dl_config: DeepLearningConfig,
        bonds_config: dict,
        include_debt: bool = False,
        cross_product: bool = False,
        n_states: int | None = None,
        n_params: int | None = None,
    ) -> tf.data.Dataset:
        """
        Create an infinite dataset pipeline for training.

        Args:
            dl_config: Deep learning configuration.
            bonds_config: Sampling bounds dictionary.
            include_debt: Whether to include debt as a state variable.
            cross_product: If True, use cross-product sampling so every
                parameter combo sees the full sampled state space.
            n_states: Number of distinct state points when
                ``cross_product=True``.  Defaults to
                ``int(sqrt(batch_size))``.
            n_params: Number of distinct parameter combos when
                ``cross_product=True``.  Defaults to
                ``batch_size // n_states``.

        Returns:
            Prefetched TensorFlow Dataset.
        """
        total_steps = (dl_config.epochs+1) * dl_config.steps_per_epoch
        dataset = tf.data.Dataset.range(total_steps)

        if cross_product:
            # Resolve grid dimensions
            if n_states is None:
                n_states = int(math.isqrt(dl_config.batch_size))
            if n_params is None:
                n_params = dl_config.batch_size // n_states

            print(f"[DatasetBuilder] CROSS-PRODUCT sampling enabled")
            print(f"[DatasetBuilder]   n_states={n_states}  n_params={n_params}  "
                  f"effective_batch={n_states * n_params}")

            def sample_fn(_):
                return StateParamSampler.sample_states_params_cross_product_gpu(
                    n_states,
                    n_params,
                    bonds_config,
                    include_debt=include_debt,
                )
        else:
            print(f"[DatasetBuilder] INDEPENDENT sampling  batch_size={dl_config.batch_size}")

            def sample_fn(_):
                return StateParamSampler.sample_states_params_gpu(
                    dl_config.batch_size,
                    bonds_config,
                    include_debt=include_debt,
                )

        dataset = dataset.map(
            sample_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)