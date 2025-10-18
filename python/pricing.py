"""
VIX option pricing using calibrated regime-switching model.
Complete pipeline: load data -> calibrate -> price options.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import logging
import sys
import os

# Import our modules
from calibration import calibrate_vix_model
from particle_filter import ParticleFilter

# Import C++ extension
try:
    import vixmodels
except ImportError:
    print("Error: vixmodels module not found. Did you build the C++ extension?")
    print("Run: mkdir build && cd build && cmake .. && make && cd ..")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_vix_data(filepath: str, date_col: str = 'Date', vix_col: str = 'VIX') -> pd.DataFrame:
    """
    Load VIX data from CSV file.
    
    Args:
        filepath: Path to CSV file
        date_col: Name of date column
        vix_col: Name of VIX value column
        
    Returns:
        DataFrame with Date and VIX columns
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    # df = df[[date_col, vix_col]].rename(columns={date_col: 'Date', vix_col: 'VIX'})
    return df


def create_regime_process(params: Dict, dt: float = 1.0/252.0):
    """
    Create RegimeProcess object from calibrated parameters.
    Regime 0: CIR (calm), Regime 1: OU (volatile)
    
    Args:
        params: Dictionary with calibrated parameters
        dt: Time step size
        
    Returns:
        RegimeProcess object
    """
    K = len(params['process_params'])
    
    # Create individual processes for each regime
    processes = []
    for k in range(K):
        p = params['process_params'][k]
        x0 = p['theta']  # Start at long-term mean
        
        if k == 0:
            # Regime 0: CIR process (calm)
            # Note: QuantLib CIR process constructor is (kappa, theta, sigma, x0)
            process = vixmodels.create_cir_process(
                x0=x0,
                kappa=p['kappa'],
                theta=p['theta'],
                sigma=p['sigma']
            )
        else:
            # Regime 1: OU process (volatile)
            # Note: QuantLib OU process constructor is (kappa, sigma, x0, theta)
            process = vixmodels.create_ou_process(
                kappa=p['kappa'],
                sigma=p['sigma'],
                x0=x0,
                theta=p['theta']
            )
        
        processes.append(process)
    
    # Create transition matrix
    P = vixmodels.Matrix(K, K)
    for i in range(K):
        for j in range(K):
            P[i, j] = params['transition_matrix'][i, j]
    
    # Create regime-switching process
    regime_process = vixmodels.RegimeProcess(processes, P, dt)
    
    return regime_process


def infer_current_regime(
    vix_data: np.ndarray,
    params: Dict,
    num_particles: int = 1000,
    window_size: int = 20
) -> Tuple[int, np.ndarray]:
    """
    Infer the most likely current regime using particle filter.
    
    Args:
        vix_data: Historical VIX data
        params: Calibrated parameters
        num_particles: Number of particles
        window_size: Number of recent observations to use
        
    Returns:
        most_likely_regime: Index of most likely regime
        regime_probs: Probability distribution over regimes
    """
    # Use recent data
    recent_data = vix_data[-window_size:]
    
    pf = ParticleFilter(
        num_particles=num_particles,
        num_regimes=len(params['process_params']),
        state_dim=1
    )
    
    dt = 1.0 / 252.0
    rng = np.random.default_rng(42)
    
    def initial_state_sampler():
        """Vectorized initial state sampling."""
        K = len(params['process_params'])
        regimes = rng.choice(K, size=num_particles, p=params['initial_regime_probs'])
        states = np.full((num_particles, 1), recent_data[0] / 100.0)
        return states, regimes
    
    def transition_fn(states, regimes, t):
        """Vectorized transition function."""
        N = len(states)
        new_states = np.zeros((N, 1))
        
        # Process each regime's particles
        for k in range(len(params['process_params'])):
            mask = (regimes == k)
            if not mask.any():
                continue
                
            p = params['process_params'][k]
            kappa, theta, sigma = p['kappa'], p['theta'], p['sigma']
            x = states[mask, 0]
            
            if k == 0:
                # CIR transition (vectorized)
                c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
                d = 4 * kappa * theta / sigma**2
                nc_param = 4 * kappa * np.exp(-kappa * dt) / (
                    sigma**2 * (1 - np.exp(-kappa * dt))
                ) * x
                
                if d > 1:
                    chi_sq = rng.noncentral_chisquare(d, nc_param)
                    new_state_val = c * chi_sq
                else:
                    drift = kappa * (theta - x) * dt
                    diffusion = sigma * np.sqrt(np.maximum(x, 0)) * np.sqrt(dt) * rng.standard_normal(len(x))
                    new_state_val = np.maximum(x + drift + diffusion, 0)
            else:
                # OU transition (vectorized)
                drift = kappa * (theta - x) * dt
                diffusion = sigma * np.sqrt(dt) * rng.standard_normal(len(x))
                new_state_val = x + drift + diffusion
            
            new_states[mask, 0] = new_state_val
        
        # Transition regimes (vectorized)
        uniform_draws = rng.uniform(size=N)
        new_regimes = np.zeros(N, dtype=int)
        
        for k in range(len(params['process_params'])):
            mask = (regimes == k)
            if mask.any():
                cumsum = np.cumsum(params['transition_matrix'][k])
                new_regimes[mask] = np.searchsorted(cumsum, uniform_draws[mask])
        
        return new_states, new_regimes
    
    def observation_fn(states, regimes):
        """Vectorized observation function."""
        return states[:, 0] * 100.0
    
    states, regimes, weights, _ = pf.filter(
        recent_data,
        initial_state_sampler,
        transition_fn,
        observation_fn,
        params['observation_noise_std']
    )
    
    # Calculate regime probabilities from final particles
    final_regimes = regimes[-1]
    final_weights = weights[-1]
    
    regime_probs = np.zeros(len(params['process_params']))
    for k in range(len(params['process_params'])):
        regime_probs[k] = final_weights[final_regimes == k].sum()
    
    most_likely_regime = np.argmax(regime_probs)
    
    return most_likely_regime, regime_probs


def price_vix_option(
    regime_process,
    initial_vix: float,
    initial_regime: int,
    strike: float,
    expiry_days: int,
    risk_free_rate: float = 0.05,
    option_type: str = 'call',
    num_paths: int = 50000,
    seed: int = 42
) -> float:
    """
    Price a VIX option using Monte Carlo simulation.
    
    Args:
        regime_process: RegimeProcess object
        initial_vix: Current VIX level (percentage points)
        initial_regime: Current regime index
        strike: Strike price
        expiry_days: Days to expiration
        risk_free_rate: Risk-free rate (annual)
        option_type: 'call' or 'put'
        num_paths: Number of Monte Carlo paths
        seed: Random seed
        
    Returns:
        option_price: Estimated option price
    """
    pricer = vixmodels.VIXOptionPricer(regime_process, num_paths, seed)
    
    if option_type.lower() == 'call':
        price = pricer.price_call(
            initial_vix,
            initial_regime,
            strike,
            expiry_days,
            risk_free_rate
        )
    elif option_type.lower() == 'put':
        price = pricer.price_put(
            initial_vix,
            initial_regime,
            strike,
            expiry_days,
            risk_free_rate
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price


if __name__ == "__main__":
    # Example usage
    data_path = os.path.join('data', 'vix_data.csv')
    vix_df = load_vix_data(data_path)
    vix_values = vix_df['VIX'].values
    
    logger.info("Calibrating VIX model...")
    calibrated_params, log_likelihoods = calibrate_vix_model(vix_values)
    # calibrated_params = initialize_vix_params(vix_values, num_regimes=2)
    
    logger.info("Inferring current regime...")
    current_regime, regime_probs = infer_current_regime(vix_values, calibrated_params)
    logger.info(f"Most likely current regime: {current_regime}")
    logger.info(f"Regime probabilities: {regime_probs}")
    
    logger.info("Creating regime process...")
    regime_process = create_regime_process(calibrated_params)
    
    current_vix = vix_values[-1]
    strike_price = 13.0  # ATM option
    expiry = 30  # 30 days to expiration
    
    logger.info("Pricing VIX call option...")
    option_price = price_vix_option(
        regime_process,
        initial_vix=current_vix,
        initial_regime=current_regime,
        strike=strike_price,
        expiry_days=expiry,
        option_type='call'
    )
    
    logger.info(f"Estimated VIX call option price (strike={strike_price}, expiry={expiry} days): {option_price:.2f}")
