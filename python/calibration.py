"""
VIX model calibration using Particle EM with regime-switching CIR processes.
"""

import numpy as np
from typing import Dict, Tuple, List
import logging
from particle_filter import ParticleEM

logger = logging.getLogger(__name__)


class VIXParticleEM(ParticleEM):
    """Particle EM for regime-switching VIX model with CIR processes."""
    
    def __init__(
        self,
        num_particles: int = 1000,
        num_regimes: int = 2,
        seed: int = 42
    ):
        super().__init__(num_particles, num_regimes, state_dim=1, seed=seed)
        self.dt = 1.0 / 252.0  # Daily time step
        self.rng = np.random.default_rng(seed)
        
    def _cir_transition_vectorized(
        self,
        x: np.ndarray,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float
    ) -> np.ndarray:
        """
        Vectorized CIR transition for multiple particles.
        
        Args:
            x: (N,) array of current states
            
        Returns:
            (N,) array of next states
        """
        c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
        d = 4 * kappa * theta / sigma**2
        nc_param = 4 * kappa * np.exp(-kappa * dt) / (
            sigma**2 * (1 - np.exp(-kappa * dt))
        ) * x
        
        result = np.zeros_like(x)
        
        if d > 1:
            # Vectorized non-central chi-squared
            chi_sq = self.rng.noncentral_chisquare(d, nc_param)
            result = c * chi_sq
        else:
            # Vectorized Euler-Maruyama
            drift = kappa * (theta - x) * dt
            diffusion = sigma * np.sqrt(np.maximum(x, 0)) * np.sqrt(dt) * self.rng.standard_normal(len(x))
            result = np.maximum(x + drift + diffusion, 0)
        
        return result
    
    def _ou_transition_vectorized(
        self,
        x: np.ndarray,
        kappa: float,
        theta: float,
        sigma: float,
        dt: float
    ) -> np.ndarray:
        """
        Vectorized OU transition for multiple particles.
        
        Args:
            x: (N,) array of current states
            
        Returns:
            (N,) array of next states
        """
        drift = kappa * (theta - x) * dt
        diffusion = sigma * np.sqrt(dt) * self.rng.standard_normal(len(x))
        return x + drift + diffusion
    
    def _sample_regimes_vectorized(
        self,
        current_regimes: np.ndarray,
        P: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized regime sampling.
        
        Args:
            current_regimes: (N,) array of current regimes
            P: (K, K) transition matrix
            
        Returns:
            (N,) array of next regimes
        """
        N = len(current_regimes)
        new_regimes = np.zeros(N, dtype=int)
        
        # Sample all at once using transition probabilities
        uniform_draws = self.rng.uniform(size=N)
        
        for k in range(self.K):
            mask = (current_regimes == k)
            if mask.any():
                cumsum = np.cumsum(P[k])
                new_regimes[mask] = np.searchsorted(cumsum, uniform_draws[mask])
        
        return new_regimes
    
    def _e_step(
        self,
        observations: np.ndarray,
        params: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run particle filter for VIX model with vectorized operations."""
        
        P = params['transition_matrix']
        process_params = params['process_params']
        obs_noise_std = params['observation_noise_std']
        
        def initial_state_sampler():
            """Vectorized initial state sampling."""
            regimes = self.rng.choice(self.K, size=self.pf.N, p=params['initial_regime_probs'])
            states = np.full((self.pf.N, 1), observations[0] / 100.0)
            return states, regimes
        
        def transition_fn(states, regimes, t):
            """Vectorized transition function."""
            N = len(states)
            new_states = np.zeros((N, 1))
            
            # Process regime 0 (CIR) particles
            mask_0 = (regimes == 0)
            if mask_0.any():
                p0 = process_params[0]
                new_states[mask_0, 0] = self._cir_transition_vectorized(
                    states[mask_0, 0], p0['kappa'], p0['theta'], p0['sigma'], self.dt
                )
            
            # Process regime 1 (OU) particles
            mask_1 = (regimes == 1)
            if mask_1.any():
                p1 = process_params[1]
                new_states[mask_1, 0] = self._ou_transition_vectorized(
                    states[mask_1, 0], p1['kappa'], p1['theta'], p1['sigma'], self.dt
                )
            
            # Transition regimes (vectorized)
            new_regimes = self._sample_regimes_vectorized(regimes, P)
            
            return new_states, new_regimes
        
        def observation_fn(states, regimes):
            """Vectorized observation function."""
            return states[:, 0] * 100.0
        
        return self.pf.filter(
            observations,
            initial_state_sampler,
            transition_fn,
            observation_fn,
            obs_noise_std
        )
    
    def _m_step(
        self,
        observations: np.ndarray,
        trajectories_states: List[np.ndarray],
        trajectories_regimes: List[np.ndarray],
        old_params: Dict
    ) -> Dict:
        """Update parameters based on sampled trajectories (vectorized)."""
        
        new_params = old_params.copy()
        
        # Update transition matrix
        new_params['transition_matrix'] = self.estimate_transition_matrix(
            trajectories_regimes
        )
        
        # Update process parameters for each regime (vectorized)
        new_process_params = []
        
        for k in range(self.K):
            # Collect data for this regime using vectorized operations
            regime_states = []
            regime_next_states = []
            
            for traj_states, traj_regimes in zip(trajectories_states, trajectories_regimes):
                mask = (traj_regimes == k)
                if mask.any():
                    indices = np.where(mask)[0]
                    # Only include if we have next state
                    valid_indices = indices[indices < len(traj_regimes) - 1]
                    if len(valid_indices) > 0:
                        regime_states.append(traj_states[valid_indices, 0])
                        regime_next_states.append(traj_states[valid_indices + 1, 0])
            
            if len(regime_states) == 0:
                logger.warning(f"No data for regime {k}, keeping old parameters")
                new_process_params.append(old_params['process_params'][k])
                continue
            
            # Stack all data
            x_vals = np.concatenate(regime_states)
            next_x_vals = np.concatenate(regime_next_states)
            
            if len(x_vals) < 10:
                logger.warning(f"Insufficient data for regime {k}, keeping old parameters")
                new_process_params.append(old_params['process_params'][k])
                continue
            
            # Vectorized moment matching
            mean_x = np.mean(x_vals)
            var_x = np.var(x_vals)
            
            # Estimate theta (long-term mean)
            theta = mean_x
            
            # Vectorized increment calculations
            dx = next_x_vals - x_vals
            mean_dx = np.mean(dx)
            
            # Estimate kappa (mean reversion speed)
            # E[dX] ≈ kappa * (theta - X) * dt
            kappa = max(-mean_dx / (mean_x - theta + 1e-10) / self.dt, 0.01)
            
            # Estimate sigma (volatility)
            var_dx = np.var(dx)
            if k == 0:  # CIR: Var[dX] ≈ sigma^2 * X * dt
                sigma = np.sqrt(max(var_dx / (mean_x * self.dt + 1e-10), 0.001))
            else:  # OU: Var[dX] ≈ sigma^2 * dt
                sigma = np.sqrt(max(var_dx / (self.dt + 1e-10), 0.001))
            
            # Regularization to prevent extreme values
            kappa = np.clip(kappa, 0.1, 20.0)
            theta = np.clip(theta, 0.05, 1.0)  # VIX in decimal
            sigma = np.clip(sigma, 0.1, 5.0)
            
            new_process_params.append({
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma
            })
            
            logger.info(f"  Regime {k}: kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}")
        
        new_params['process_params'] = new_process_params
        
        # Update observation noise (vectorized)
        # Calculate all residuals at once
        all_residuals = []
        for traj_states, traj_regimes in zip(trajectories_states, trajectories_regimes):
            pred_obs = traj_states[:, 0] * 100.0
            residuals = observations[:len(traj_regimes)] - pred_obs
            all_residuals.append(residuals)
        
        all_residuals = np.concatenate(all_residuals)
        new_params['observation_noise_std'] = np.std(all_residuals)
        
        return new_params


def initialize_vix_params(
    observations: np.ndarray,
    num_regimes: int = 2
) -> Dict:
    """
    Initialize parameters for VIX regime-switching model.
    
    Args:
        observations: (T,) array of VIX values in percentage points
        num_regimes: Number of regimes
        
    Returns:
        Dictionary with initial parameters
    """
    
    # Convert to decimal
    vix_decimal = observations / 100.0
    
    mean_vix = np.mean(vix_decimal)
    std_vix = np.std(vix_decimal)
    
    # Initialize transition matrix (persistent regimes)
    P = np.zeros((num_regimes, num_regimes))
    for i in range(num_regimes):
        for j in range(num_regimes):
            if i == j:
                P[i, j] = 0.95
            else:
                P[i, j] = (1 - 0.95) / (num_regimes - 1)
    # Normalize each row
    P = P / P.sum(axis=1, keepdims=True)
    
    # Initialize process parameters
    # Regime 0: Low volatility
    # Regime 1: High volatility
    process_params = []
    
    for k in range(num_regimes):
        # Stagger the long-term means
        theta = mean_vix * (0.7 + 0.6 * k / max(num_regimes - 1, 1))
        
        # Higher volatility in higher regimes
        sigma = std_vix * (1.0 + k * 0.5)
        
        # Moderate mean reversion
        kappa = 3.0 / (k + 1)  # Slower mean reversion in higher regimes
        
        process_params.append({
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma
        })
    
    return {
        'transition_matrix': P,
        'process_params': process_params,
        'observation_noise_std': std_vix * 0.1,  # Small observation noise
        'initial_regime_probs': np.ones(num_regimes) / num_regimes
    }


def calibrate_vix_model(
    vix_data: np.ndarray,
    num_regimes: int = 2,
    num_particles: int = 1000,
    max_iterations: int = 30,
    num_trajectories: int = 10,
    seed: int = 42
) -> Tuple[Dict, List[float]]:
    """
    Calibrate regime-switching VIX model to historical data.
    
    Args:
        vix_data: (T,) array of VIX values in percentage points
        num_regimes: Number of regimes
        num_particles: Number of particles for filter
        max_iterations: Maximum EM iterations
        num_trajectories: Number of trajectories for M-step
        seed: Random seed
        
    Returns:
        params: Calibrated parameters
        log_likelihoods: Log likelihood history
    """
    
    logger.info(f"Calibrating {num_regimes}-regime VIX model")
    logger.info(f"Data: {len(vix_data)} observations")
    logger.info(f"VIX range: [{vix_data.min():.2f}, {vix_data.max():.2f}]")
    
    # Initialize parameters
    initial_params = initialize_vix_params(vix_data, num_regimes)
    
    # Run Particle EM
    pem = VIXParticleEM(num_particles, num_regimes, seed)
    
    params, log_likelihoods = pem.fit(
        vix_data,
        initial_params,
        max_iterations=max_iterations,
        tolerance=1e-4,
        num_trajectories=num_trajectories
    )
    
    return params, log_likelihoods
