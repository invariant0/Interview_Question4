# econ_models/dl/training/dataset_builder.py
"""
Dataset construction for deep learning training.

This module provides utilities for building TensorFlow data pipelines
that sample state variables for training.
"""

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.sampling.state_sampler import StateSampler
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
    ) -> tf.data.Dataset:
        """
        Create an infinite dataset pipeline for training.

        Args:
            config: Deep learning configuration.
            include_debt: Whether to include debt as a state variable.
            progress_variable: Curriculum progress variable.

        Returns:
            Prefetched TensorFlow Dataset.
        """
        total_steps = (dl_config.epochs + 1) * dl_config.steps_per_epoch
        dataset = tf.data.Dataset.range(total_steps)

        def sample_fn(_):
            return StateSampler.sample_states_gpu(
                dl_config.batch_size,
                bonds_config,
                include_debt=include_debt,
            )

        dataset = dataset.map(
            sample_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)