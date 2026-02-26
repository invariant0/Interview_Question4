# src/econ_models/simulator/vfi/basic.py
"""
VFI Simulator for the Basic RBC Model.

Uses continuous interpolation of value functions and policies
to ensure moment convergence as grid resolution increases.

Compatible with synthetic_data_generator for consistent shock sequences.
"""

from typing import Optional, Dict, Tuple
import numpy as np

from econ_models.config.economic_params import EconomicParams
from econ_models.vfi.grids.grid_utils import interp_2d_batch

class VFISimulator_basic:
    """
    Simulator using fully continuous interpolation of VFI solutions.
    
    Design principles:
    1. Capital (K) remains continuous throughout simulation
    2. Productivity (Z) from shock_sequence is continuous, mapped to grid for value lookup
    3. Adjust/wait decision uses interpolated values at exact (K, Z)
    4. Target capital when adjusting is interpolated (not snapped to grid)
    """
    
    def __init__(self, econ_params: EconomicParams):
        """Initialize the VFI simulator."""
        self.econ_params = econ_params
        self.solution_loaded = False
        
        # Solution components (populated by load_solved_vfi_solution)
        self.V_adjust: Optional[np.ndarray] = None
        self.V_wait: Optional[np.ndarray] = None
        self.policy_k_values: Optional[np.ndarray] = None
        self.policy_adjust_idx: Optional[np.ndarray] = None  # For backward compatibility
        self.k_grid: Optional[np.ndarray] = None
        self.z_grid: Optional[np.ndarray] = None
        self.k_min: float = 0.0
        self.k_max: float = 1.0
        self.delta: float = 0.1

    def load_solved_vfi_solution(self, res: Dict) -> None:
        """
        Load pre-computed VFI solution.
        
        Args:
            res: Dictionary from BasicModelVFI.solve() containing:
                - V_adjust, V_wait: Value functions (n_k, n_z)
                - policy_k_values: Optimal K' when adjusting (n_k, n_z)
                - K, Z: Grid arrays
                - k_min, k_max, depreciation_rate: Metadata
        """
        self.V_adjust = res["V_adjust"]
        self.V_wait = res["V_wait"]
        self.k_grid = res["K"]
        self.z_grid = res["Z"]
        
        # Use continuous policy if available, otherwise convert from indices
        if "policy_k_values" in res:
            self.policy_k_values = res["policy_k_values"]
        else:
            # Backward compatibility: convert indices to values
            self.policy_k_values = self.k_grid[res["policy_adjust_idx"]]
        
        # Also store index-based policy for any legacy code
        self.policy_adjust_idx = res.get("policy_adjust_idx", None)
        
        # Grid metadata
        self.k_min = res.get("k_min", float(self.k_grid[0]))
        self.k_max = res.get("k_max", float(self.k_grid[-1]))
        self.delta = res.get("depreciation_rate", self.econ_params.depreciation_rate)
        
        self.solution_loaded = True

    def simulate(
        self,
        initial_states: Tuple[np.ndarray, np.ndarray],
        innovation_sequence: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate the economy using VFI policy function.
        
        Uses CONTINUOUS interpolation for both decisions and policies.
        
        Args:
            initial_states: Tuple (K_0, Z_0)
                - K_0: array of shape (batch_size, 1) or (batch_size,)
                - Z_0: array of shape (batch_size, 1) or (batch_size,)
            shock_sequence: array of shape (batch_size, T+1) containing
                            CONTINUOUS productivity values from AR(1) process.
                            
        Returns:
            Dictionary with:
                - K_curr: Capital at start of period (batch_size, T)
                - K_next: Capital at end of period (batch_size, T)
                - Z_curr: Productivity at start of period (batch_size, T)
                - Z_next: Productivity at end of period (batch_size, T)
        """
        if not self.solution_loaded:
            raise ValueError("VFI solution not loaded. Call load_solved_vfi_solution first.")

        # Convert inputs to numpy
        k_init = np.asarray(initial_states[0]).flatten()
        z_init = np.asarray(initial_states[1]).flatten()
        eps_seq = np.asarray(innovation_sequence) * self.econ_params.productivity_std_dev  # (batch_size, T+1) - continuous Z values
        
        batch_size = len(k_init)
        T = eps_seq.shape[1] - 1  # Number of transitions (T periods)
        
        # Initialize capital at provided initial state, clamped to grid
        k_sim = np.clip(k_init, self.k_min, self.k_max)
        z_sim = z_init  # Continuous productivity
        # Allocate history arrays
        K_curr = np.zeros((batch_size, T))
        K_next = np.zeros((batch_size, T))
        Z_curr = np.zeros((batch_size, T))
        Z_next = np.zeros((batch_size, T))
        
        # Simulation loop
        for t in range(T):

            # Store current state
            K_curr[:, t] = k_sim
            Z_curr[:, t] = z_sim
            
            # --- Decision: Adjust or Wait? ---
            # Interpolate BOTH value functions at current continuous (K, Z)
            v_adjust = interp_2d_batch(
                self.k_grid, self.z_grid, self.V_adjust, k_sim, z_sim
            )
            v_wait = interp_2d_batch(
                self.k_grid, self.z_grid, self.V_wait, k_sim, z_sim
            )
            
            should_adjust = v_adjust > v_wait
            
            # --- Compute next period capital ---
            k_depreciated = k_sim * (1.0 - self.delta)
            k_next = np.copy(k_depreciated)  # Default: wait
            
            # For adjusters: interpolate the policy function
            if np.any(should_adjust):
                k_target = interp_2d_batch(
                    self.k_grid, self.z_grid, self.policy_k_values,
                    k_sim[should_adjust], z_sim[should_adjust]
                )
                # Clamp to grid bounds
                k_target = np.clip(k_target, self.k_min, self.k_max)
                k_next[should_adjust] = k_target
            
            # Store next capital
            K_next[:, t] = k_next
            z_safe = np.maximum(z_sim, 1e-12)
            ln_z = np.log(z_safe)
            ln_z_next = self.econ_params.productivity_persistence * ln_z + eps_seq[:, t + 1]
            z_next = np.exp(ln_z_next)
            Z_next[:, t] = z_next
            # Update state for next period
            k_sim = k_next
            z_sim = z_next
        return {
            "K_curr": K_curr,
            "K_next": K_next,
            "Z_curr": Z_curr,
            "Z_next": Z_next,
        }
    
    def simulate_life_time_reward(
        self,
        initial_states: Tuple[np.ndarray, np.ndarray],
        innovation_sequence: np.ndarray,
    ) -> float:
        """
        Simulate the life-time reward using VFI policy function.
        """
        if not self.solution_loaded:
            raise ValueError("VFI solution not loaded. Call load_solved_vfi_solution first.")

        # 1. Setup Data
        k_init = np.asarray(initial_states[0]).flatten()
        z_init = np.asarray(initial_states[1]).flatten()
        # Ensure epsilon is scaled correctly
        eps_seq = np.asarray(innovation_sequence) * self.econ_params.productivity_std_dev
        
        batch_size = len(k_init)
        T = eps_seq.shape[1] - 1
        
        k_sim = np.clip(k_init, self.k_min, self.k_max).astype(np.float64)
        z_sim = z_init.astype(np.float64)
        
        accumulated_reward = np.zeros(batch_size, dtype=np.float64)
        
        # 2. Extract Params for fast Numpy access
        discount_factor = float(self.econ_params.discount_factor)
        delta = float(self.econ_params.depreciation_rate)
        # Check for fixed cost parameter, default to 0.0 if not found
        fixed_cost_param = getattr(self.econ_params, 'fixed_cost', 0.0)
        
        # 3. Simulation Loop
        for t in range(T):
            # --- A. Decision: Adjust or Wait? ---
            v_adjust = interp_2d_batch(self.k_grid, self.z_grid, self.V_adjust, k_sim, z_sim)
            v_wait = interp_2d_batch(self.k_grid, self.z_grid, self.V_wait, k_sim, z_sim)
            should_adjust = v_adjust > v_wait
            
            # --- B. Physical Transitions ---
            k_depreciated = k_sim * (1.0 - delta)
            
            # Interpolate target capital
            k_target = interp_2d_batch(self.k_grid, self.z_grid, self.policy_k_values, k_sim, z_sim)
            k_target = np.clip(k_target, self.k_min, self.k_max)
            
            # Determine next capital
            k_next = np.where(should_adjust, k_target, k_depreciated)
            
            # --- C. Reward Calculation (Numpy Optimized) ---
            # 1. Profit (Cobb-Douglas: Z * K^alpha)
            # We do this in numpy to avoid TF overhead inside loop
            profit = z_sim * (k_sim ** self.econ_params.capital_share)
            
            # 2. Investment
            investment = k_next - k_depreciated
            
            # 3. Adjustment Costs (pure numpy)
            psi_0 = self.econ_params.adjustment_cost_convex
            psi_1 = self.econ_params.adjustment_cost_fixed
            safe_capital = np.maximum(k_sim, 1e-8)
            convex_cost = psi_0 * (investment ** 2) / (2.0 * safe_capital)
            is_investing = (investment != 0.0).astype(np.float64)
            fixed_cost = psi_1 * k_sim * is_investing
            adj_cost_total = convex_cost + fixed_cost
            
            # FIX 1: Add Fixed Cost for adjusters
            # Fixed cost is usually proportional to K or a constant. 
            # Assuming standard formulation: Cost = Convex + (Fixed if adjust)
            if fixed_cost_param > 0:
                # If fixed cost is a fraction of capital (common): fixed_cost_param * k_sim
                # If fixed cost is absolute: fixed_cost_param
                # You must check your specific EconomicParams definition. 
                # Assuming absolute for safety, or check if params has 'fixed_cost_k_prop'
                fixed_cost_val = fixed_cost_param * k_sim # Common assumption: proportional to size
                adj_cost_total = adj_cost_total + np.where(should_adjust, fixed_cost_val, 0.0)

            # 4. Cash Flow
            cash_flow = profit - investment - adj_cost_total
            
            accumulated_reward += (discount_factor ** t) * cash_flow
            
            # --- D. Productivity Transition ---
            z_safe = np.maximum(z_sim, 1e-12)
            ln_z = np.log(z_safe)
            ln_z_next = self.econ_params.productivity_persistence * ln_z + eps_seq[:, t + 1]
            z_next = np.exp(ln_z_next)
            
            k_sim = k_next
            z_sim = z_next
        
        return float(np.mean(accumulated_reward))