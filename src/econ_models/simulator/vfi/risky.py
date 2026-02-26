# src/econ_models/simulator/vfi/risky.py
"""
Continuous-Interpolation VFI Simulator for the Risky Debt Model.

Uses trilinear interpolation of value functions and policies to
ensure smooth, grid-resolution-independent dynamics — following
the same design as the basic-model VFI simulator.

Design Principles
-----------------
1. **Continuous state space** — capital, debt, and productivity
   remain continuous throughout simulation (no grid-snapping).
2. **Interpolated decisions** — default and adjust/wait decisions
   use trilinear interpolation of V_adjust and V_wait, producing
   smooth decision boundaries that converge as grid resolution
   increases.
3. **Interpolated policies** — next-period (k', b') are read from
   continuously interpolated policy surfaces, not index arrays.
4. **Continuous Z transition** — productivity follows the AR(1)
   process without snapping to the nearest grid point (matching
   the basic-model approach).
"""

from typing import Optional, Dict, Tuple, Any

import numpy as np
import scipy.ndimage as ndimage

from econ_models.config.economic_params import EconomicParams


# ═══════════════════════════════════════════════════════════════════════════
#  Pure-NumPy Trilinear Interpolation (avoids TF dispatch overhead)
# ═══════════════════════════════════════════════════════════════════════════

def _interp_3d_np(
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    values: np.ndarray,
    kq: np.ndarray,
    bq: np.ndarray,
    zq: np.ndarray,
) -> np.ndarray:
    """
    Vectorised trilinear interpolation — pure NumPy, no TF overhead.

    Args:
        k_grid: (Nk,) sorted knots.
        b_grid: (Nb,) sorted knots.
        z_grid: (Nz,) sorted knots.
        values: (Nk, Nb, Nz) function values on the regular grid.
        kq, bq, zq: (B,) query coordinates (clamped to grid bounds).

    Returns:
        (B,) interpolated values.
    """
    Nk = len(k_grid)
    Nb = len(b_grid)
    Nz = len(z_grid)

    # Clamp
    kq = np.clip(kq, k_grid[0], k_grid[-1])
    bq = np.clip(bq, b_grid[0], b_grid[-1])
    zq = np.clip(zq, z_grid[0], z_grid[-1])

    # Bracket indices
    ki = np.searchsorted(k_grid, kq, side='right') - 1
    bi = np.searchsorted(b_grid, bq, side='right') - 1
    zi = np.searchsorted(z_grid, zq, side='right') - 1
    np.clip(ki, 0, Nk - 2, out=ki)
    np.clip(bi, 0, Nb - 2, out=bi)
    np.clip(zi, 0, Nz - 2, out=zi)

    # Grid values at bracket boundaries
    k0 = k_grid[ki];  k1 = k_grid[ki + 1]
    b0 = b_grid[bi];  b1 = b_grid[bi + 1]
    z0 = z_grid[zi];  z1 = z_grid[zi + 1]

    # Weights
    EPS = 1e-10
    wk = (kq - k0) / np.maximum(k1 - k0, EPS)
    wb = (bq - b0) / np.maximum(b1 - b0, EPS)
    wz = (zq - z0) / np.maximum(z1 - z0, EPS)

    # Eight corner lookups via flat indexing
    flat = values.ravel()
    stride_k = Nb * Nz
    stride_b = Nz
    base = ki * stride_k + bi * stride_b + zi

    v000 = flat[base]
    v001 = flat[base + 1]
    v010 = flat[base + stride_b]
    v011 = flat[base + stride_b + 1]
    v100 = flat[base + stride_k]
    v101 = flat[base + stride_k + 1]
    v110 = flat[base + stride_k + stride_b]
    v111 = flat[base + stride_k + stride_b + 1]

    # Trilinear combination
    wk1 = 1.0 - wk
    wb1 = 1.0 - wb
    wz1 = 1.0 - wz

    return (
        v000 * (wk1 * wb1 * wz1)
        + v001 * (wk1 * wb1 * wz)
        + v010 * (wk1 * wb  * wz1)
        + v011 * (wk1 * wb  * wz)
        + v100 * (wk  * wb1 * wz1)
        + v101 * (wk  * wb1 * wz)
        + v110 * (wk  * wb  * wz1)
        + v111 * (wk  * wb  * wz)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a numpy array with the nearest non-NaN value.

    Essential for preventing interpolation artefacts at default-state
    boundaries where the VFI solver stores NaN for policy values.
    """
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    indices = ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True,
    )
    return arr[tuple(indices)]


class VFISimulator_risky:
    """
    Simulator using trilinear interpolation of VFI solutions.

    Mirrors the design of ``VFISimulator_basic`` but extends to the
    three-dimensional ``(k, b, z)`` state space of the risky debt model.

    Design principles (matching basic model):

    1. Capital, debt, and productivity remain **continuous** throughout
       the simulation — no grid-snapping.
    2. **Adjust / wait / default** decisions use interpolated value
       functions at the exact continuous state.
    3. Target ``(k', b')`` when adjusting (and ``b'`` when waiting)
       are interpolated from continuous policy surfaces.
    4. Productivity follows the continuous AR(1) process with no
       post-transition grid-snapping.
    """

    def __init__(self, econ_params: EconomicParams):
        self.econ_params = econ_params
        self.solution_loaded = False

        # Grids
        self.k_grid: Optional[np.ndarray] = None
        self.b_grid: Optional[np.ndarray] = None
        self.z_grid: Optional[np.ndarray] = None

        # Value functions (float64, shape: nk × nb × nz)
        self.V: Optional[np.ndarray] = None
        self.V_adjust: Optional[np.ndarray] = None
        self.V_wait: Optional[np.ndarray] = None

        # Continuous policy arrays (float64, shape: nk × nb × nz)
        # NaN-filled at default states; _fill_nan_nearest applied on load.
        self.policy_k_values: Optional[np.ndarray] = None
        self.policy_b_values: Optional[np.ndarray] = None
        self.policy_b_wait_values: Optional[np.ndarray] = None

        # Bond-price schedule (float64, shape: nk × nb × nz)
        self.Q: Optional[np.ndarray] = None

        # Grid bounds (set on load)
        self.k_min: float = 0.0
        self.k_max: float = 1.0
        self.b_min: float = 0.0
        self.b_max: float = 1.0
        self.z_min: float = 0.0
        self.z_max: float = 1.0

        # Metadata
        self.delta: float = 0.0
        self._v_default_eps: float = 1e-10

    # ── Load VFI Solution ──────────────────────────────────────────────

    def load_solved_vfi_solution(self, res: Dict[str, Any]) -> None:
        """
        Load converged VFI arrays and prepare for continuous interpolation.

        Stores grids, value functions, continuous policy arrays (with
        NaN-filling for default states), and the bond-price schedule.
        Mirrors the basic model's ``load_solved_vfi_solution`` but
        extended to the 3-D ``(k, b, z)`` state space.

        Args:
            res: Dictionary (or NpzFile) from the VFI solver's ``solve()``.
        """
        # 1. Grids
        self.k_grid = np.array(res["K"], dtype=np.float64)
        self.b_grid = np.array(res["B"], dtype=np.float64)
        self.z_grid = np.array(res["Z"], dtype=np.float64)

        # 2. Value functions
        self.V = np.array(res["V"], dtype=np.float64)
        self.V_adjust = np.array(res["V_adjust"], dtype=np.float64)
        self.V_wait = np.array(res["V_wait"], dtype=np.float64)

        # 3. Bond-price schedule
        self.Q = np.array(res["Q"], dtype=np.float64)

        # 4. Continuous policy values
        #    The VFI solver stores NaN at default states; fill with
        #    nearest-neighbor so the interpolator always returns finite
        #    values (default states are filtered by value-function check
        #    before policies are used).
        if "policy_k_values" in res:
            raw_k = np.array(res["policy_k_values"], dtype=np.float64)
            raw_b = np.array(res["policy_b_values"], dtype=np.float64)
            raw_bw = np.array(res["policy_b_wait_values"], dtype=np.float64)
        else:
            # Backward compatibility: convert indices to values
            raw_k_idx = np.clip(
                np.array(res["policy_k_idx"], dtype=np.int32),
                0, len(self.k_grid) - 1,
            )
            raw_b_idx = np.clip(
                np.array(res["policy_b_idx"], dtype=np.int32),
                0, len(self.b_grid) - 1,
            )
            raw_bw_idx = np.clip(
                np.array(res["policy_b_wait_idx"], dtype=np.int32),
                0, len(self.b_grid) - 1,
            )
            raw_k = self.k_grid[raw_k_idx].astype(np.float64)
            raw_b = self.b_grid[raw_b_idx].astype(np.float64)
            raw_bw = self.b_grid[raw_bw_idx].astype(np.float64)

        self.policy_k_values = _fill_nan_nearest(raw_k)
        self.policy_b_values = _fill_nan_nearest(raw_b)
        self.policy_b_wait_values = _fill_nan_nearest(raw_bw)

        # 5. Grid bounds
        self.k_min = float(self.k_grid[0])
        self.k_max = float(self.k_grid[-1])
        self.b_min = float(self.b_grid[0])
        self.b_max = float(self.b_grid[-1])
        self.z_min = float(self.z_grid[0])
        self.z_max = float(self.z_grid[-1])

        # 6. Metadata
        self.delta = float(
            res.get("depreciation_rate", self.econ_params.depreciation_rate)
        )
        self._v_default_eps = float(res.get("v_default_eps", 1e-10))

        self.solution_loaded = True

        nk, nb, nz = len(self.k_grid), len(self.b_grid), len(self.z_grid)
        print(
            f"Continuous VFI solution loaded (trilinear interpolation).\n"
            f"  Grid : nk={nk}, nb={nb}, nz={nz} "
            f"({nk * nb * nz:,} total states)\n"
            f"  Decisions via interpolated V_adjust / V_wait.\n"
            f"  Policies via interpolated continuous k', b', b_wait'.\n"
            f"  Firms evolve in continuous state space (no grid-snapping)."
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Pure-NumPy Financial Helpers (float64)
    # ══════════════════════════════════════════════════════════════════════

    def _revenue(
        self, k: np.ndarray, z: np.ndarray,
    ) -> np.ndarray:
        """After-tax Cobb-Douglas revenue: ``(1−τ) · z · k^θ``."""
        return (
            (1.0 - self.econ_params.corporate_tax_rate)
            * z
            * np.power(k, self.econ_params.capital_share)
        )

    def _debt_flows(
        self, q: np.ndarray, b_next: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Debt inflow and tax shield from new bond issuance.

        Returns:
            (debt_inflow, tax_shield) — same shapes as inputs.
        """
        debt_inflow = q * b_next
        interest = (1.0 - q) * np.maximum(b_next, 0.0)
        tax_shield = (
            self.econ_params.corporate_tax_rate
            * interest
            / (1.0 + self.econ_params.risk_free_rate)
        )
        return debt_inflow, tax_shield

    def _adj_cost(
        self, investment: np.ndarray, k_curr: np.ndarray,
    ) -> np.ndarray:
        """Convex + fixed adjustment cost: ``ψ₀/2 · I²/K + ψ₁ · K · 𝟙{I≠0}``."""
        psi0 = self.econ_params.adjustment_cost_convex
        psi1 = self.econ_params.adjustment_cost_fixed
        safe_k = np.maximum(k_curr, 1e-8)
        convex = psi0 * (investment ** 2) / (2.0 * safe_k)
        fixed = psi1 * k_curr * (investment != 0.0).astype(np.float64)
        return convex + fixed

    def _issuance_cost(self, payout: np.ndarray) -> np.ndarray:
        """Equity issuance cost: ``(η₀ + η₁·|e|) · 𝟙{e<0}``."""
        return np.where(
            payout < 0.0,
            self.econ_params.equity_issuance_cost_fixed
            + self.econ_params.equity_issuance_cost_linear * np.abs(payout),
            0.0,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Main Simulation
    # ══════════════════════════════════════════════════════════════════════

    def simulate(
        self,
        initial_states: Tuple[np.ndarray, np.ndarray, np.ndarray],
        innovation_sequence: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate the risky debt model using continuous interpolation.

        Performance optimisations vs. the original implementation:

        * Pure-NumPy trilinear interpolation (no TF dispatch overhead).
        * Full Z sequence pre-computed outside the loop.
        * V_adjust / V_wait cached from default check and reused for
          adjust-vs-wait decision (saves 2 interp calls per step).
        * Branchless ``np.where`` replaces conditional fancy-indexing.
        * Policy interpolation done for ALL alive firms at once; results
          combined via ``np.where`` (no variable-length sub-arrays).

        Args:
            initial_states: ``(K_0, B_0, Z_0)`` arrays, shape ``(batch,)``.
            innovation_sequence: ``(batch, T+1)`` standard-normal shocks.

        Returns:
            Dictionary of ``(batch, T)`` arrays:
            ``K_curr``, ``K_next``, ``B_curr``, ``B_next``, ``Z_curr``, ``Z_next``.
        """
        if not self.solution_loaded:
            raise ValueError("VFI solution not loaded.")

        # ── Setup ──────────────────────────────────────────────────
        k_init = np.asarray(initial_states[0], dtype=np.float64).ravel()
        b_init = np.asarray(initial_states[1], dtype=np.float64).ravel()
        z_init = np.asarray(initial_states[2], dtype=np.float64).ravel()
        eps_seq = (
            np.asarray(innovation_sequence, dtype=np.float64)
            * self.econ_params.productivity_std_dev
        )

        batch = len(k_init)
        T = eps_seq.shape[1] - 1

        q_rf = 1.0 / (1.0 + self.econ_params.risk_free_rate)

        # ── Pre-compute the entire Z sequence (independent of K/B) ──
        rho = self.econ_params.productivity_persistence
        Z_all = np.empty((batch, T + 1), dtype=np.float64)
        Z_all[:, 0] = np.clip(z_init, self.z_min, self.z_max)
        for t in range(T):
            ln_z = np.log(np.maximum(Z_all[:, t], 1e-12))
            Z_all[:, t + 1] = np.exp(rho * ln_z + eps_seq[:, t + 1])

        # ── Local refs for hot-loop ────────────────────────────────
        k_grid = np.ascontiguousarray(self.k_grid, dtype=np.float64)
        b_grid = np.ascontiguousarray(self.b_grid, dtype=np.float64)
        z_grid = np.ascontiguousarray(self.z_grid, dtype=np.float64)
        V_adj  = np.ascontiguousarray(self.V_adjust, dtype=np.float64)
        V_wt   = np.ascontiguousarray(self.V_wait, dtype=np.float64)
        pol_k  = np.ascontiguousarray(self.policy_k_values, dtype=np.float64)
        pol_b  = np.ascontiguousarray(self.policy_b_values, dtype=np.float64)
        pol_bw = np.ascontiguousarray(self.policy_b_wait_values, dtype=np.float64)
        Q_arr  = np.ascontiguousarray(self.Q, dtype=np.float64)
        k_lo, k_hi = self.k_min, self.k_max
        b_lo, b_hi = self.b_min, self.b_max
        one_minus_delta = 1.0 - self.delta
        v_def_eps = self._v_default_eps
        interp = _interp_3d_np  # local function ref

        # Clip initial states to grid bounds (keep continuous)
        k_sim = np.clip(k_init, k_lo, k_hi)
        b_sim = np.clip(b_init, b_lo, b_hi)

        # Storage
        K_curr = np.full((batch, T), np.nan, dtype=np.float64)
        K_next = np.full((batch, T), np.nan, dtype=np.float64)
        B_curr = np.full((batch, T), np.nan, dtype=np.float64)
        B_next = np.full((batch, T), np.nan, dtype=np.float64)
        Z_curr = Z_all[:, :T].copy()
        Z_next = Z_all[:, 1:T + 1].copy()
        equity_issuance = np.full((batch, T), np.nan, dtype=np.float64)
        bond_price = np.full((batch, T), np.nan, dtype=np.float64)

        has_defaulted = np.zeros(batch, dtype=bool)

        # ── Simulation loop ────────────────────────────────────────
        for t in range(T):
            z_sim = Z_all[:, t]
            K_curr[:, t] = k_sim
            B_curr[:, t] = b_sim

            active = ~has_defaulted
            if not np.any(active):
                # All firms have defaulted — remaining columns stay NaN
                break

            # ── 1. Interpolate V_adjust and V_wait ONCE for active ──
            ka, ba, za = k_sim[active], b_sim[active], z_sim[active]
            v_adj_vals = interp(k_grid, b_grid, z_grid, V_adj, ka, ba, za)
            v_wt_vals  = interp(k_grid, b_grid, z_grid, V_wt,  ka, ba, za)

            # ── 2. Default decision (reuse cached V values) ────────
            v_continue = np.maximum(v_adj_vals, v_wt_vals)
            newly_default_active = v_continue <= v_def_eps

            newly_default = np.zeros(batch, dtype=bool)
            newly_default[active] = newly_default_active
            has_defaulted |= newly_default
            alive = ~has_defaulted

            if not np.any(alive):
                break

            # ── 3. Adjust vs Wait (reuse cached values) ───────────
            survived_in_active = alive[active]
            should_adjust = np.zeros(batch, dtype=bool)
            active_idx = np.where(active)[0]
            should_adjust_in_active = (v_adj_vals > v_wt_vals) & survived_in_active
            should_adjust[active_idx[should_adjust_in_active]] = True

            # ── 4. Next-period state (branchless policies) ────────
            k_dep = k_sim * one_minus_delta
            k_next = np.full(batch, np.nan, dtype=np.float64)
            b_next = np.full(batch, np.nan, dtype=np.float64)

            # Default for alive: wait (depreciated k, same b)
            k_next[alive] = k_dep[alive]
            b_next[alive] = b_sim[alive]

            # Interpolate ALL policy surfaces for alive firms at once
            al_k = k_sim[alive]
            al_b = b_sim[alive]
            al_z = z_sim[alive]

            k_targ = interp(k_grid, b_grid, z_grid, pol_k, al_k, al_b, al_z)
            b_targ = interp(k_grid, b_grid, z_grid, pol_b, al_k, al_b, al_z)
            b_wait_targ = interp(k_grid, b_grid, z_grid, pol_bw, al_k, al_b, al_z)

            np.clip(k_targ, k_lo, k_hi, out=k_targ)
            np.clip(b_targ, b_lo, b_hi, out=b_targ)
            np.clip(b_wait_targ, b_lo, b_hi, out=b_wait_targ)

            is_adj_alive = should_adjust[alive]
            k_next[alive] = np.where(is_adj_alive, k_targ, k_dep[alive])
            b_next[alive] = np.where(is_adj_alive, b_targ, b_wait_targ)

            # ── 5. Equity issuance (branchless payout) ────────────
            eq_iss_t = np.full(batch, np.nan, dtype=np.float64)
            q_t = np.full(batch, np.nan, dtype=np.float64)
            kc = al_k
            zc = al_z
            bc = al_b
            kn = k_next[alive]
            bn = b_next[alive]

            revenue = self._revenue(kc, zc)

            q = interp(k_grid, b_grid, z_grid, Q_arr, kn, bn, zc)
            np.clip(q, 0.0, q_rf, out=q)
            q_t[alive] = q

            d_inflow, t_shield = self._debt_flows(q, bn)

            inv_alive = kn - (one_minus_delta * kc)
            ac = self._adj_cost(inv_alive, kc)

            # Branchless payout
            payout = revenue + d_inflow + t_shield - bc
            payout -= np.where(is_adj_alive, inv_alive + ac, 0.0)

            eq_iss_t[alive] = np.maximum(0.0, -payout)
            equity_issuance[:, t] = eq_iss_t
            bond_price[:, t] = q_t

            # ── 6. Record next state & advance ─────────────────────
            K_next[:, t] = k_next
            B_next[:, t] = b_next
            # Z_next already filled from precomputed array;
            # mark dead firms as NaN (both Z_curr and Z_next for consistency
            # with K_curr / B_curr NaN-masking convention)
            if np.any(has_defaulted):
                Z_curr[has_defaulted, t] = np.nan
                Z_next[has_defaulted, t] = np.nan

            k_sim = k_next
            b_sim = b_next

        return {
            "K_curr": K_curr,
            "K_next": K_next,
            "B_curr": B_curr,
            "B_next": B_next,
            "Z_curr": Z_curr,
            "Z_next": Z_next,
            "equity_issuance": equity_issuance,
            "bond_price": bond_price,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  Discounted Lifetime Reward
    # ══════════════════════════════════════════════════════════════════════

    def simulate_life_time_reward(
        self,
        initial_states: Tuple[np.ndarray, np.ndarray, np.ndarray],
        innovation_sequence: np.ndarray,
    ) -> float:
        """
        Discounted lifetime reward using continuous interpolation.

        Same logic as ``simulate`` but accumulates
        ``β^t · (payout − issuance_cost)`` for survivors and zero for
        defaulted firms.

        Optimised: pure-NumPy interpolation, precomputed Z, cached
        V_adjust/V_wait, branchless np.where, pre-computed discount
        powers.

        Args:
            initial_states: ``(K_0, B_0, Z_0)`` arrays.
            innovation_sequence: ``(batch, T+1)`` standard-normal shocks.

        Returns:
            Mean discounted lifetime reward across the batch (scalar).
        """
        if not self.solution_loaded:
            raise ValueError("VFI solution not loaded.")

        # ── Setup ──────────────────────────────────────────────────
        k_init = np.asarray(initial_states[0], dtype=np.float64).ravel()
        b_init = np.asarray(initial_states[1], dtype=np.float64).ravel()
        z_init = np.asarray(initial_states[2], dtype=np.float64).ravel()
        eps_seq = (
            np.asarray(innovation_sequence, dtype=np.float64)
            * self.econ_params.productivity_std_dev
        )

        batch = len(k_init)
        T = eps_seq.shape[1] - 1

        q_rf = 1.0 / (1.0 + self.econ_params.risk_free_rate)
        beta = float(self.econ_params.discount_factor)

        # ── Pre-compute Z sequence ─────────────────────────────────
        rho = self.econ_params.productivity_persistence
        Z_all = np.empty((batch, T + 1), dtype=np.float64)
        Z_all[:, 0] = np.clip(z_init, self.z_min, self.z_max)
        for t in range(T):
            ln_z = np.log(np.maximum(Z_all[:, t], 1e-12))
            Z_all[:, t + 1] = np.clip(
                np.exp(rho * ln_z + eps_seq[:, t + 1]),
                self.z_min, self.z_max,
            )

        # ── Local refs ─────────────────────────────────────────────
        k_grid = np.ascontiguousarray(self.k_grid, dtype=np.float64)
        b_grid = np.ascontiguousarray(self.b_grid, dtype=np.float64)
        z_grid = np.ascontiguousarray(self.z_grid, dtype=np.float64)
        V_adj  = np.ascontiguousarray(self.V_adjust, dtype=np.float64)
        V_wt   = np.ascontiguousarray(self.V_wait, dtype=np.float64)
        pol_k  = np.ascontiguousarray(self.policy_k_values, dtype=np.float64)
        pol_b  = np.ascontiguousarray(self.policy_b_values, dtype=np.float64)
        pol_bw = np.ascontiguousarray(self.policy_b_wait_values, dtype=np.float64)
        Q_arr  = np.ascontiguousarray(self.Q, dtype=np.float64)
        k_lo, k_hi = self.k_min, self.k_max
        b_lo, b_hi = self.b_min, self.b_max
        one_minus_delta = 1.0 - self.delta
        v_def_eps = self._v_default_eps
        interp = _interp_3d_np

        # Pre-compute discount powers
        disc_powers = np.power(beta, np.arange(T, dtype=np.float64))

        k_sim = np.clip(k_init, k_lo, k_hi)
        b_sim = np.clip(b_init, b_lo, b_hi)

        reward = np.zeros(batch, dtype=np.float64)
        has_defaulted = np.zeros(batch, dtype=bool)

        # ── Simulation loop ────────────────────────────────────────
        for t in range(T):
            z_sim = Z_all[:, t]

            active = ~has_defaulted
            if not np.any(active):
                break

            # ── 1. Interpolate V once; cache for default + adjust ──
            ka, ba, za = k_sim[active], b_sim[active], z_sim[active]
            v_adj_vals = interp(k_grid, b_grid, z_grid, V_adj, ka, ba, za)
            v_wt_vals  = interp(k_grid, b_grid, z_grid, V_wt,  ka, ba, za)

            # Default
            v_continue = np.maximum(v_adj_vals, v_wt_vals)
            newly_default = np.zeros(batch, dtype=bool)
            newly_default[active] = v_continue <= v_def_eps
            has_defaulted |= newly_default
            alive = ~has_defaulted

            if not np.any(alive):
                break

            # Adjust / Wait (reuse cached V values)
            survived_in_active = alive[active]
            should_adjust = np.zeros(batch, dtype=bool)
            active_idx = np.where(active)[0]
            should_adjust_in_active = (v_adj_vals > v_wt_vals) & survived_in_active
            should_adjust[active_idx[should_adjust_in_active]] = True

            # ── 2. Next state (branchless) ─────────────────────────
            k_dep = k_sim * one_minus_delta
            k_next = np.zeros(batch, dtype=np.float64)
            b_next = np.zeros(batch, dtype=np.float64)

            al_k = k_sim[alive]
            al_b = b_sim[alive]
            al_z = z_sim[alive]

            k_targ = interp(k_grid, b_grid, z_grid, pol_k, al_k, al_b, al_z)
            b_targ = interp(k_grid, b_grid, z_grid, pol_b, al_k, al_b, al_z)
            b_wait_targ = interp(k_grid, b_grid, z_grid, pol_bw, al_k, al_b, al_z)
            np.clip(k_targ, k_lo, k_hi, out=k_targ)
            np.clip(b_targ, b_lo, b_hi, out=b_targ)
            np.clip(b_wait_targ, b_lo, b_hi, out=b_wait_targ)

            is_adj_alive = should_adjust[alive]
            k_next[alive] = np.where(is_adj_alive, k_targ, k_dep[alive])
            b_next[alive] = np.where(is_adj_alive, b_targ, b_wait_targ)

            # ── 3. Cash flow (branchless) ──────────────────────────
            cf = np.zeros(batch, dtype=np.float64)

            kc = al_k
            zc = al_z
            bc = al_b
            kn = k_next[alive]
            bn = b_next[alive]

            revenue = self._revenue(kc, zc)

            q = interp(k_grid, b_grid, z_grid, Q_arr, kn, bn, zc)
            np.clip(q, 0.0, q_rf, out=q)

            d_inflow, t_shield = self._debt_flows(q, bn)

            inv_alive = kn - (one_minus_delta * kc)
            ac = self._adj_cost(inv_alive, kc)

            payout = revenue + d_inflow + t_shield - bc
            payout -= np.where(is_adj_alive, inv_alive + ac, 0.0)

            icost = self._issuance_cost(payout)
            cf[alive] = payout - icost

            reward += disc_powers[t] * cf

            # ── 4. Advance state ───────────────────────────────────
            k_sim = k_next
            b_sim = b_next

        return float(np.mean(reward))
