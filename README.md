# VIX Regime-Switching Model with Particle EM Calibration

A complete system for calibrating regime-switching stochastic volatility models to VIX data and pricing VIX options using Monte Carlo simulation. Combines C++ (QuantLib) for performance-critical Monte Carlo pricing with Python for statistical estimation via Particle Filter and Particle EM algorithms.

## Features

- **Regime-Switching CIR Processes**: Multiple volatility regimes with Markov switching
- **Particle Filter**: Bootstrap particle filter with systematic resampling
- **Particle EM Algorithm**: Maximum likelihood estimation of regime-switching parameters
- **Monte Carlo Pricing**: High-performance option pricing using QuantLib
- **Python-C++ Integration**: Seamless interface via pybind11

## Project Structure

```
project/
├── src/                          # C++ source files
│   ├── RegimeProcess.hpp         # Regime-switching process wrapper
│   ├── RegimeProcess.cpp
│   ├── MonteCarloEngine.hpp      # Monte Carlo pricing engine
│   ├── MonteCarloEngine.cpp
│   └── bindings.cpp              # Python bindings
├── python/                       # Python modules
│   ├── particle_filter.py        # Particle filter & Particle EM
│   ├── calibration.py            # VIX model calibration
│   ├── pricing.py                # Pricing infrastructure
|   ├── pricing_test.ipynb        # Test of pricing engines
│   └── regime_viz.ipynb          # Visual of regime probabilities
├── data/                         # Data directory
│   ├── vix_data.csv              # Short-window VIX data (user-provided)
│   ├── long_vix_data.csv         # Long-window VIX data (user-provided)
│   ├── short_active.png          # Short-window VIX chart
│   └── long_active.png           # Long-window VIX chart
├── download_vix_data.py          # yfinance download script
├── requirements.txt              # Python requirements
├── CMakeLists.txt                # Build configuration
├── build.sh                      # Build script
└── README.md                     # This file
```

## Dependencies

### C++ Libraries
- **QuantLib** (>= 1.30): Financial mathematics library
- **Boost**: Required by QuantLib
- **pybind11**: Python bindings

### Python Packages
- numpy >= 1.20
- scipy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4
- yfinance >= 0.2.66

## Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libquantlib0-dev \
    python3-dev \
    pkg-config
```

**macOS:**
```bash
brew install cmake boost quantlib python3
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the C++ Extension

```bash
chmod +x build.sh
./build.sh
```

This will:
- Check dependencies
- Build the C++ extension module
- Copy the module to the project root

## Usage

### 1. Prepare VIX Data

Run the download script `download_vix_data.py` to place your historical VIX data in `data/vix_data.csv`. The CSV should have at least two columns:
- `Date`: Date in YYYY-MM-DD format
- `VIX`: VIX closing value

Example:
```csv
Date,VIX
2020-01-02,13.78
2020-01-03,12.47
2020-01-06,13.12
...
```

You can download VIX data from Yahoo Finance (^VIX).

### 2. Run the Notebook

This will:
1. Load VIX historical data
2. Calibrate regime-switching model using Particle EM
3. Infer the current regime
4. Simulate future VIX paths
5. Price VIX call and put options at various strikes and expiries

## Model Description

### Regime-Switching CIR Process

The VIX follows a regime-switching pair of CIR and OU processes:

```
dV_t = κ_s(θ_s - V_t)dt + σ_sg_{S_t}(V_t) dW_t
```

where `s ∈ {0, 1, ..., K-1}` is the regime at time t, following a discrete Markov chain with transition matrix P.

### Parameters

For each regime k:
- **κ_k**: Mean reversion speed
- **θ_k**: Long-term mean level
- **σ_k**: Volatility of volatility

Plus:
- **P**: K×K transition probability matrix
- **σ_obs**: Observation noise standard deviation

### Calibration Algorithm

1. **Particle Filter (E-step)**:
   - Approximate posterior distribution of latent states and regimes
   - Use bootstrap particle filter with systematic resampling
   - Compute likelihood of observations

2. **Parameter Update (M-step)**:
   - Sample trajectories via backward sampling
   - Estimate transition matrix from regime sequences
   - Estimate CIR parameters using method of moments
   - Update observation noise

3. **Iterate** until convergence

### Pricing

Monte Carlo simulation:
1. Initialize VIX at current level in inferred regime
2. Simulate N paths using regime-switching CIR dynamics
3. Calculate option payoffs at maturity
4. Discount and average

## Troubleshooting

### Build Errors

**"QuantLib not found"**
```bash
# Check installation
pkg-config --modversion QuantLib

# Ubuntu/Debian
sudo apt-get install libquantlib0-dev

# macOS
brew install quantlib
```

**"pybind11 not found"**
```bash
pip install pybind11
```

### Runtime Errors

**"vixmodels module not found"**
- Ensure build was successful: `ls vixmodels*.so`
- Run from project root directory
- Check Python can import: `python3 -c "import vixmodels"`

**"All weights zero at time t"**
- Model parameters may be extreme
- Try adjusting initial parameters in `calibration.py`
- Reduce number of regimes
- Check data quality

## References

1. Doucet, A., & Johansen, A. M. (2009). A tutorial on particle filtering and smoothing.
2. Cappé, O., Moulines, E., & Rydén, T. (2005). Inference in Hidden Markov Models.
3. Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A theory of the term structure of interest rates.
4. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series.

## License

This project uses QuantLib which is licensed under a BSD-style license.

## Authors

Developed for regime-switching VIX option pricing with particle-based calibration.
