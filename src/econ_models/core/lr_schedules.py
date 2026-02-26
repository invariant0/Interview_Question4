# econ_models/core/lr_schedules.py
"""
Custom learning-rate schedules for deep-learning training loops.

Provides schedules that compose a linear warm-up phase with standard
decay schedules such as cosine annealing.
"""

import tensorflow as tf


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with a linear warm-up prefix.

    During the first ``warmup_steps`` optimiser steps the learning rate
    increases linearly from ``initial_learning_rate * warmup_start_fraction``
    to ``initial_learning_rate``.  After the warm-up window the rate
    follows a standard half-cosine decay to
    ``initial_learning_rate * alpha``.

    This schedule is XLA-compatible and can be used inside
    ``tf.function(jit_compile=True)`` training steps.

    Args:
        initial_learning_rate: Peak learning rate reached at the end of
            the warm-up window.
        warmup_steps: Number of steps for the linear ramp-up.  Set to 0
            to disable warm-up (equivalent to plain ``CosineDecay``).
        decay_steps: Total number of *post-warm-up* steps over which the
            cosine decay is applied.
        alpha: Minimum LR fraction at the end of training
            (``final_lr = initial_learning_rate * alpha``).
        warmup_start_fraction: LR fraction at step 0
            (``start_lr = initial_learning_rate * warmup_start_fraction``).
            Defaults to 0.0 (start from zero).

    Example::

        schedule = WarmupCosineDecay(
            initial_learning_rate=5e-4,
            warmup_steps=5000,
            decay_steps=95000,
            alpha=0.01,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    """

    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        alpha: float = 0.0,
        warmup_start_fraction: float = 0.0,
    ) -> None:
        """Initialize the warmup cosine decay schedule.

        Store schedule hyper-parameters and pre-build the internal
        ``CosineDecay`` schedule that takes over after the warm-up
        window.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_start_fraction = warmup_start_fraction

        # Pre-build the cosine decay that takes over after warm-up
        self._cosine = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha,
        )

    def __call__(self, step):
        """Return the learning rate for the given optimizer step.

        Args:
            step: Current optimizer step (scalar or 0-d tensor).

        Returns:
            Scalar learning rate for *step*.
        """
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        # Linear ramp: start_fraction → 1.0 over warmup_steps
        warmup_lr = self.initial_learning_rate * (
            self.warmup_start_fraction
            + (1.0 - self.warmup_start_fraction) * step / tf.maximum(warmup_steps, 1.0)
        )

        # Cosine decay phase (step count restarts from 0 after warm-up)
        cosine_lr = self._cosine(step - warmup_steps)

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        """Return the schedule configuration as a serializable dict."""
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_start_fraction": self.warmup_start_fraction,
        }
