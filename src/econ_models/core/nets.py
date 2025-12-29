# econ_models/core/nets.py
"""
Neural network factory for building standardized MLP architectures.

This module provides utilities for constructing multi-layer perceptrons
with consistent architecture and for managing target network updates.

Example:
    >>> from econ_models.core.nets import NeuralNetFactory
    >>> value_net = NeuralNetFactory.build_mlp(
    ...     input_dim=2, output_dim=1, config=config, name="ValueNet"
    ... )
"""

from typing import Optional, List

import tensorflow as tf
import numpy as np

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.types import TENSORFLOW_DTYPE


class NeuralNetFactory:
    """
    Factory for building MLPs and managing network weights.

    This class provides static methods for constructing neural networks
    with consistent architecture and for performing soft updates between
    source and target networks.
    """

    @staticmethod
    def build_mlp(
        input_dim: int,
        output_dim: int,
        config: DeepLearningConfig,
        output_activation: str = "linear",
        scale_factor: Optional[float] = None,
        name: str = "MLP"
    ) -> tf.keras.Model:
        """
        Build a Multi-Layer Perceptron using the Keras Functional API.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output neurons.
            config: Configuration with layer sizes and activation.
            output_activation: Activation function for the final layer.
            scale_factor: If provided, multiplies output by this constant.
            name: Name of the Keras model.

        Returns:
            Compiled (but untrained) Keras model.
        """
        inputs = tf.keras.layers.Input(
            shape=(input_dim,),
            dtype=TENSORFLOW_DTYPE,
            name=f"{name}_Input"
        )

        x = inputs
        for i, units in enumerate(config.hidden_layers):
            x = tf.keras.layers.Dense(
                units,
                activation=config.activation_function,
                dtype=TENSORFLOW_DTYPE,
                name=f"{name}_Dense_{i}"
            )(x)

        outputs = tf.keras.layers.Dense(
            output_dim,
            activation=output_activation,
            dtype=TENSORFLOW_DTYPE,
            name=f"{name}_Output_Raw"
        )(x)

        # Apply optional scaling for economic value normalization
        if scale_factor is not None and scale_factor != 1.0:
            outputs = NeuralNetFactory._apply_scale_layer(
                outputs, scale_factor, name
            )

        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    @staticmethod
    def _apply_scale_layer(
        outputs: tf.Tensor,
        scale_factor: float,
        name: str
    ) -> tf.Tensor:
        """
        Apply a scaling layer to network outputs.

        Args:
            outputs: Raw network outputs.
            scale_factor: Multiplicative scale factor.
            name: Base name for the layer.

        Returns:
            Scaled outputs.
        """
        scale_tensor = tf.constant(scale_factor, dtype=TENSORFLOW_DTYPE)
        return tf.keras.layers.Lambda(
            lambda t: t * scale_tensor,
            name=f"{name}_Output_Scaled"
        )(outputs)

    @staticmethod
    def soft_update(
        source_model: tf.keras.Model,
        target_model: tf.keras.Model,
        polyak: float
    ) -> None:
        """
        Perform soft update of target network using Polyak averaging.

        Formula:
            theta_target = polyak * theta_source + (1 - polyak) * theta_target

        This technique stabilizes training by slowly updating the target
        network, similar to techniques used in Deep Q-Networks.

        Args:
            source_model: Model with latest gradients/weights.
            target_model: Stable target model to update.
            polyak: Interpolation factor (typically small, e.g., 0.005).
        """
        source_weights = source_model.get_weights()
        target_weights = target_model.get_weights()

        new_weights: List[np.ndarray] = []
        for s_w, t_w in zip(source_weights, target_weights):
            updated_w = polyak * s_w + (1.0 - polyak) * t_w
            new_weights.append(updated_w)

        target_model.set_weights(new_weights)