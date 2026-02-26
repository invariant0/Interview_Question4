"""Protocol definitions for VFI solver components.

Defines ``typing.Protocol`` classes that formalise the interfaces
between the orchestrator and its components.  Contains no
implementation — only type signatures.

Using these protocols, the orchestrator (``RiskyDebtModelVFI.solve``)
depends only on abstract interfaces.  In tests, any protocol can be
satisfied by a lightweight stub that returns pre-canned tensors.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import tensorflow as tf


@runtime_checkable
class FlowComputer(Protocol):
    """Interface for flow-tensor construction."""

    def build_adjust_flow(self, **kwargs: Any) -> tf.Tensor:
        """Build the ADJUST-branch flow tensor."""
        ...

    def build_wait_flow(self, **kwargs: Any) -> tf.Tensor:
        """Build the WAIT-branch flow tensor."""
        ...

    def build_debt_components(self, **kwargs: Any) -> tf.Tensor:
        """Build the debt-component tensor."""
        ...


@runtime_checkable
class BellmanUpdater(Protocol):
    """Interface for Bellman-iteration primitives."""

    def compute_ev(
        self, v_curr: tf.Tensor, P: tf.Tensor, beta: tf.Tensor
    ) -> tf.Tensor:
        """Compute discounted expected continuation value."""
        ...

    def bellman_update(
        self,
        v_adjust: tf.Tensor,
        v_wait: tf.Tensor,
        v_curr: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute V_new and sup-norm diff."""
        ...

    def sup_norm_diff(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Compute ‖a − b‖∞."""
        ...


@runtime_checkable
class BondPriceUpdater(Protocol):
    """Interface for bond-price updates."""

    def update(
        self,
        value_function: tf.Tensor,
        bond_prices_old: tf.Tensor,
        **kwargs: Any,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Update bond prices and return (q_updated, diff)."""
        ...


@runtime_checkable
class TileExecutor(Protocol):
    """Interface for chunked ADJUST-branch computation."""

    def execute(self, **kwargs: Any) -> Tuple[tf.Tensor, tf.Tensor]:
        """Run the tiled ADJUST computation."""
        ...


@runtime_checkable
class PolicyExtractor(Protocol):
    """Interface for mapping indices to continuous policy values."""

    def extract(self, **kwargs: Any) -> Dict[str, tf.Tensor]:
        """Extract policies from raw value-function and index tensors."""
        ...


@runtime_checkable
class Simulator(Protocol):
    """Interface for post-solve simulation."""

    def run(
        self, solution: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Run simulation and return (history, stats)."""
        ...
