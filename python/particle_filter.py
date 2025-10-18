"""
Particle filter and Particle EM for regime-switching VIX models.
Implements Bootstrap Particle Filter with systematic resampling.
Optimized with vectorized NumPy operations.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, List, Dict, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParticleFilter:
    """Bootstrap particle filter for regime-switching models."""
    
    def __init__(
        self,
        num_particles: int,
        num_regimes: int,
        state_dim: int,
        seed: int = 42
    ):
        """
        Initialize particle filter.
        
        Args:
            num_particles: Number of particles
            num_regimes: Number of regimes (K)
            state_dim: Dimension of state space
            seed: Random seed for reproducibility
        """
        self.N = num_particles
        self.K = num_regimes
        self.d = state_dim
        self.rng = np.random.default_rng(seed)
        
    def systematic_resampling(self, weights: np.ndarray) -> np.ndarray:
        """
        Systematic resampling of particles.
        Low-variance resampling method.
        
        Args:
            weights: (N,) array of normalized particle weights
            
        Returns:
            indices: (N,) array of resampled particle indices
        """
        N = len(weights)
        positions = (np.arange(N) + self.rng.uniform()) / N
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        return indices
    
    def filter(
        self,
        observations: np.ndarray,
        initial_state_sampler: Callable,
        transition_fn: Callable,
        observation_fn: Callable,
        observation_noise_std: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Run particle filter with vectorized operations.
        
        Args:
            observations: (T,) array of observations
            initial_state_sampler: Function that returns (states, regimes) arrays
            transition_fn: Vectorized function (states, regimes, t) -> (new_states, new_regimes)
            observation_fn: Vectorized function (states, regimes) -> observations
            observation_noise_std: Standard deviation of observation noise
            
        Returns:
            states: (T, N, d) filtered states
            regimes: (T, N) filtered regimes
            weights: (T, N) particle weights
            log_likelihood: Log likelihood of observations
        """
        T = len(observations)
        
        # Initialize storage
        states = np.zeros((T, self.N, self.d))
        regimes = np.zeros((T, self.N), dtype=int)
        weights = np.zeros((T, self.N))
        
        # Initialize particles (vectorized)
        states[0], regimes[0] = initial_state_sampler()
        
        # Compute initial weights (vectorized)
        pred_obs = observation_fn(states[0], regimes[0])
        weights[0] = norm.pdf(observations[0], loc=pred_obs, scale=observation_noise_std)
        
        # Normalize weights
        weight_sum = weights[0].sum()
        if weight_sum > 0:
            weights[0] /= weight_sum
            log_likelihood = np.log(weight_sum / self.N)
        else:
            weights[0] = np.ones(self.N) / self.N
            log_likelihood = -np.inf
            logger.warning("All weights zero at time 0")
        
        # Filter loop
        for t in range(1, T):
            # Resample
            indices = self.systematic_resampling(weights[t-1])
            states[t-1] = states[t-1, indices]
            regimes[t-1] = regimes[t-1, indices]
            
            # Propagate (vectorized)
            states[t], regimes[t] = transition_fn(states[t-1], regimes[t-1], t)
            
            # Weight (vectorized)
            pred_obs = observation_fn(states[t], regimes[t])
            weights[t] = norm.pdf(observations[t], loc=pred_obs, scale=observation_noise_std)
            
            # Normalize and update log likelihood
            weight_sum = weights[t].sum()
            if weight_sum > 0:
                weights[t] /= weight_sum
                log_likelihood += np.log(weight_sum / self.N)
            else:
                weights[t] = np.ones(self.N) / self.N
                logger.warning(f"All weights zero at time {t}")
        
        return states, regimes, weights, log_likelihood


class ParticleEM:
    """Particle EM algorithm for regime-switching models."""
    
    def __init__(
        self,
        num_particles: int,
        num_regimes: int,
        state_dim: int,
        seed: int = 42
    ):
        """
        Initialize Particle EM algorithm.
        
        Args:
            num_particles: Number of particles for filtering
            num_regimes: Number of regimes (K)
            state_dim: Dimension of state space
            seed: Random seed for reproducibility
        """
        self.pf = ParticleFilter(num_particles, num_regimes, state_dim, seed)
        self.K = num_regimes
        self.d = state_dim
        self.rng = np.random.default_rng(seed)
        
    def backward_sampling(
        self,
        states: np.ndarray,
        regimes: np.ndarray,
        weights: np.ndarray,
        transition_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward sampling to get a single trajectory.
        Uses Forward Filtering Backward Sampling (FFBS).
        
        Args:
            states: (T, N, d) filtered states
            regimes: (T, N) filtered regimes  
            weights: (T, N) particle weights
            transition_matrix: (K, K) transition matrix
            
        Returns:
            sampled_states: (T, d) single trajectory
            sampled_regimes: (T,) regime sequence
        """
        T, N, d = states.shape
        
        sampled_states = np.zeros((T, d))
        sampled_regimes = np.zeros(T, dtype=int)
        
        # Sample final time
        idx = self.rng.choice(N, p=weights[-1])
        sampled_states[-1] = states[-1, idx]
        sampled_regimes[-1] = regimes[-1, idx]
        
        # Backward pass
        for t in range(T-2, -1, -1):
            # Compute backward weights (vectorized)
            next_regime = sampled_regimes[t+1]
            bwd_weights = weights[t] * transition_matrix[regimes[t], next_regime]
            
            if bwd_weights.sum() > 0:
                bwd_weights /= bwd_weights.sum()
                idx = self.rng.choice(N, p=bwd_weights)
            else:
                idx = self.rng.choice(N, p=weights[t])
            
            sampled_states[t] = states[t, idx]
            sampled_regimes[t] = regimes[t, idx]
        
        return sampled_states, sampled_regimes
    
    def estimate_transition_matrix(
        self,
        regime_trajectories: List[np.ndarray]
    ) -> np.ndarray:
        """
        Estimate transition matrix from regime trajectories (vectorized).
        Uses maximum likelihood estimation (count-based).
        
        Args:
            regime_trajectories: List of (T,) regime sequences
            
        Returns:
            P: (K, K) transition matrix
        """
        # Stack all trajectories
        all_regimes = np.concatenate(regime_trajectories)
        T_total = len(all_regimes)
        
        # Count transitions (vectorized)
        curr_regimes = []
        next_regimes = []
        
        for trajectory in regime_trajectories:
            if len(trajectory) > 1:
                curr_regimes.append(trajectory[:-1])
                next_regimes.append(trajectory[1:])
        
        if len(curr_regimes) == 0:
            P = np.ones((self.K, self.K)) / self.K
            return P
        
        curr_regimes = np.concatenate(curr_regimes)
        next_regimes = np.concatenate(next_regimes)
        
        # Count transitions using bincount
        counts = np.zeros((self.K, self.K))
        for i in range(self.K):
            mask = (curr_regimes == i)
            if mask.any():
                counts[i] = np.bincount(next_regimes[mask], minlength=self.K)
        
        # Normalize rows
        P = np.zeros((self.K, self.K))
        for i in range(self.K):
            row_sum = counts[i].sum()
            if row_sum > 0:
                P[i] = counts[i] / row_sum
            else:
                P[i] = 1.0 / self.K  # Uniform if no observations
        
        return P
    
    def fit(
        self,
        observations: np.ndarray,
        initial_params: Dict,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        num_trajectories: int = 10
    ) -> Tuple[Dict, List[float]]:
        """
        Fit model using Particle EM.
        
        Args:
            observations: (T,) array of VIX observations
            initial_params: Dictionary with initial parameters
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance
            num_trajectories: Number of trajectories for M-step
            
        Returns:
            params: Dictionary with estimated parameters
            log_likelihoods: List of log likelihoods per iteration
        """
        params = initial_params.copy()
        log_likelihoods = []
        
        for iteration in range(max_iterations):
            logger.info(f"EM Iteration {iteration + 1}/{max_iterations}")
            
            # E-step: Run particle filter
            states, regimes, weights, log_lik = self._e_step(
                observations, params
            )
            log_likelihoods.append(log_lik)
            
            logger.info(f"  Log-likelihood: {log_lik:.4f}")
            
            # Check convergence
            if iteration > 0:
                improvement = log_lik - log_likelihoods[-2]
                logger.info(f"  Improvement: {improvement:.6f}")
                if abs(improvement) < tolerance:
                    logger.info("Converged!")
                    break
            
            # Sample trajectories for M-step
            trajectories_states = []
            trajectories_regimes = []
            
            for _ in range(num_trajectories):
                traj_states, traj_regimes = self.backward_sampling(
                    states, regimes, weights, params['transition_matrix']
                )
                trajectories_states.append(traj_states)
                trajectories_regimes.append(traj_regimes)
            
            # M-step: Update parameters
            params = self._m_step(
                observations, trajectories_states, trajectories_regimes, params
            )
        
        return params, log_likelihoods
    
    def _e_step(
        self,
        observations: np.ndarray,
        params: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        E-step: run particle filter.
        Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")
    
    def _m_step(
        self,
        observations: np.ndarray,
        trajectories_states: List[np.ndarray],
        trajectories_regimes: List[np.ndarray],
        old_params: Dict
    ) -> Dict:
        """
        M-step: update parameters.
        Must be implemented by subclass.
        """
        raise NotImplementedError("Must be implemented by subclass")
    