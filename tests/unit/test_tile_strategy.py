"""Unit tests for tile_strategy: compute_optimal_chunks."""

from __future__ import annotations

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import pytest

from econ_models.vfi.chunking.tile_strategy import compute_optimal_chunks


class TestComputeOptimalChunks:
    """Tests for compute_optimal_chunks."""

    def test_small_grid_single_tile(self):
        """Small grid fits in VRAM → returns full-grid chunks."""
        nk, nb = 10, 10
        k_cs, b_cs, kp_cs, bp_cs = compute_optimal_chunks(nk, nb, n_z=5, vram_limit_gb=28.0)
        assert k_cs == nk
        assert b_cs == nb
        assert kp_cs == nk
        assert bp_cs == nb

    def test_medium_grid(self):
        """Medium grid should still return valid chunks."""
        nk, nb = 100, 100
        k_cs, b_cs, kp_cs, bp_cs = compute_optimal_chunks(nk, nb, n_z=15, vram_limit_gb=28.0)
        assert 1 <= k_cs <= nk
        assert 1 <= b_cs <= nb
        assert 1 <= kp_cs <= nk
        assert 1 <= bp_cs <= nb

    def test_large_grid_chunks_smaller(self):
        """Large grid must chunk — choice dims should be < full."""
        nk, nb = 1000, 1000
        k_cs, b_cs, kp_cs, bp_cs = compute_optimal_chunks(nk, nb, n_z=15, vram_limit_gb=28.0)
        # With 1000^4 * 15 = way over budget, so must chunk
        assert kp_cs < nk or bp_cs < nb

    def test_vram_respects_budget(self):
        """Tile size does not exceed VRAM budget."""
        nk, nb, nz = 200, 200, 15
        k_cs, b_cs, kp_cs, bp_cs = compute_optimal_chunks(nk, nb, n_z=nz, vram_limit_gb=2.0)

        tile_bytes = k_cs * b_cs * kp_cs * bp_cs * nz * 4 * 3
        assert tile_bytes <= 2.0 * (1024 ** 3) * 1.5  # Allow 50% margin for rounding

    def test_symmetric_choice_dims(self):
        """kp_chunk and bp_chunk should be equal (strategy uses kp=bp)."""
        nk, nb = 500, 500
        k_cs, b_cs, kp_cs, bp_cs = compute_optimal_chunks(nk, nb, n_z=15, vram_limit_gb=10.0)
        assert kp_cs == bp_cs

    def test_tiny_vram_still_works(self):
        """Even with very small VRAM, returns valid (≥10) chunks."""
        nk, nb = 100, 100
        k_cs, b_cs, kp_cs, bp_cs = compute_optimal_chunks(nk, nb, n_z=15, vram_limit_gb=0.001)
        assert kp_cs >= 10
        assert bp_cs >= 10
