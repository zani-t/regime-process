#include "MonteCarloEngine.hpp"
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/math/randomnumbers/inversecumulativersg.hpp>
#include <ql/math/randomnumbers/randomsequencegenerator.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <cmath>

RegimeSwitchingMCEngine::RegimeSwitchingMCEngine(
    const boost::shared_ptr<RegimeProcess>& process,
    Size numPaths,
    Size timeSteps,
    BigNatural seed)
: process_(process), numPaths_(numPaths), timeSteps_(timeSteps), seed_(seed) {
    QL_REQUIRE(process_, "process cannot be null");
    QL_REQUIRE(numPaths_ > 0, "number of paths must be positive");
    QL_REQUIRE(timeSteps_ > 0, "number of time steps must be positive");
}

Real RegimeSwitchingMCEngine::priceEuropean(
    Time maturity,
    const Array& initialState,
    Size initialRegime,
    Rate riskFreeRate,
    const boost::function<Real(const Array&)>& payoff) {
    
    Time dt = maturity / timeSteps_;
    Size dimensions = process_->size();
    
    Real sumPayoffs = 0.0;

    // Create uniform random sequence generator
    MersenneTwisterUniformRng uniformGen(seed_);
    RandomSequenceGenerator unifSeqGen(dimensions, uniformGen);
    
    // Create inverse cumulative RSG for normal variates
    InverseCumulativeRsg<RandomSequenceGenerator<MersenneTwisterUniformRng>, InverseCumulativeNormal> rng(unifSeqGen);
    
    for (Size path = 0; path < numPaths_; ++path) {
        process_->setRegime(initialRegime);
        Array state = initialState;
        Time t = 0.0;
        
        for (Size step = 0; step < timeSteps_; ++step) {
            // Get random normal vector
            const std::vector<Real>& gaussian = rng.nextSequence().value;
            Array dw(dimensions);
            for (Size d = 0; d < dimensions; ++d) {
                dw[d] = gaussian[d] * std::sqrt(dt);
            }
            
            // Evolve the process
            state = process_->evolve(t, state, dt, dw);
            t += dt;
        }
        
        // Calculate payoff at maturity
        Real thisPayoff = payoff(state);
        sumPayoffs += thisPayoff;
    }
    
    // Discount and average
    Real discount = std::exp(-riskFreeRate * maturity);
    return (sumPayoffs / numPaths_) * discount;
}

std::vector<std::vector<Array>> RegimeSwitchingMCEngine::simulatePaths(
    Time maturity,
    const Array& initialState,
    Size initialRegime,
    Size numPathsToReturn) {
    
    if (numPathsToReturn == 0 || numPathsToReturn > numPaths_) {
        numPathsToReturn = numPaths_;
    }
    
    Time dt = maturity / timeSteps_;
    Size dimensions = process_->size();
    
    std::vector<std::vector<Array>> paths(numPathsToReturn);
    
    // Create uniform random sequence generator
    MersenneTwisterUniformRng uniformGen(seed_);
    RandomSequenceGenerator unifSeqGen(dimensions, uniformGen);
    
    // Create inverse cumulative RSG for normal variates
    InverseCumulativeRsg<RandomSequenceGenerator<MersenneTwisterUniformRng>, InverseCumulativeNormal> rng(unifSeqGen);
    
    for (Size path = 0; path < numPathsToReturn; ++path) {
        process_->setRegime(initialRegime);
        Array state = initialState;
        Time t = 0.0;
        
        paths[path].resize(timeSteps_ + 1);
        paths[path][0] = state;
        
        for (Size step = 0; step < timeSteps_; ++step) {
            const std::vector<Real>& gaussian = rng.nextSequence().value;
            Array dw(dimensions);
            for (Size d = 0; d < dimensions; ++d) {
                dw[d] = gaussian[d] * std::sqrt(dt);
            }
            
            state = process_->evolve(t, state, dt, dw);
            t += dt;
            paths[path][step + 1] = state;
        }
    }
    
    return paths;
}

std::vector<Size> RegimeSwitchingMCEngine::getRegimeHistory(
    Time maturity,
    const Array& initialState,
    Size initialRegime) {
    
    Time dt = maturity / timeSteps_;
    Size dimensions = process_->size();
    
    std::vector<Size> regimes(timeSteps_ + 1);
    
    // Create uniform random sequence generator
    MersenneTwisterUniformRng uniformGen(seed_);
    RandomSequenceGenerator unifSeqGen(dimensions, uniformGen);
    
    // Create inverse cumulative RSG for normal variates
    InverseCumulativeRsg<RandomSequenceGenerator<MersenneTwisterUniformRng>, InverseCumulativeNormal> rng(unifSeqGen);
    
    process_->setRegime(initialRegime);
    Array state = initialState;
    Time t = 0.0;
    
    regimes[0] = initialRegime;
    
    for (Size step = 0; step < timeSteps_; ++step) {
        const std::vector<Real>& gaussian = rng.nextSequence().value;
        Array dw(dimensions);
        for (Size d = 0; d < dimensions; ++d) {
            dw[d] = gaussian[d] * std::sqrt(dt);
        }
        
        state = process_->evolve(t, state, dt, dw);
        t += dt;
        regimes[step + 1] = process_->regime();
    }
    
    return regimes;
}

// VIXOptionPricer implementation

VIXOptionPricer::VIXOptionPricer(
    const boost::shared_ptr<RegimeProcess>& process,
    Size numPaths,
    BigNatural seed)
: process_(process), numPaths_(numPaths), seed_(seed) {
    QL_REQUIRE(process_, "process cannot be null");
    QL_REQUIRE(numPaths_ > 0, "number of paths must be positive");
}

Real VIXOptionPricer::priceCall(
    Real initialVIX,
    Size initialRegime,
    Real strike,
    Size expiryDays,
    Rate riskFreeRate) {
    
    Time dt = 1.0 / 252.0;  // Daily steps (business days convention)
    Time maturity = expiryDays * dt;
    Size dimensions = process_->size();
    
    Real sumPayoffs = 0.0;
    
    // Create uniform random sequence generator
    MersenneTwisterUniformRng uniformGen(seed_);
    RandomSequenceGenerator unifSeqGen(dimensions, uniformGen);
    
    // Create inverse cumulative RSG for normal variates
    InverseCumulativeRsg<RandomSequenceGenerator<MersenneTwisterUniformRng>, InverseCumulativeNormal> rng(unifSeqGen);
    
    for (Size path = 0; path < numPaths_; ++path) {
        process_->setRegime(initialRegime);
        Array state = process_->initialValues();
        state[0] = initialVIX / 100.0;  // Convert to decimal
        
        Time t = 0.0;
        
        for (Size step = 0; step < expiryDays; ++step) {
            const std::vector<Real>& gaussian = rng.nextSequence().value;
            Array dw(dimensions);
            for (Size d = 0; d < dimensions; ++d) {
                dw[d] = gaussian[d] * std::sqrt(dt);
            }
            
            state = process_->evolve(t, state, dt, dw);
            t += dt;
            
            // Ensure non-negative VIX
            if (state[0] < 0.0) state[0] = 0.0;
        }
        
        // Convert back to VIX percentage terms
        Real finalVIX = state[0] * 100.0;
        
        // Call payoff
        Real payoff = std::max(finalVIX - strike, 0.0);
        sumPayoffs += payoff;
    }
    
    // Discount and average
    Real discount = std::exp(-riskFreeRate * maturity);
    return (sumPayoffs / numPaths_) * discount;
}

Real VIXOptionPricer::pricePut(
    Real initialVIX,
    Size initialRegime,
    Real strike,
    Size expiryDays,
    Rate riskFreeRate) {
    
    Time dt = 1.0 / 252.0;
    Time maturity = expiryDays * dt;
    Size dimensions = process_->size();
    
    Real sumPayoffs = 0.0;
    
    // Create uniform random sequence generator
    MersenneTwisterUniformRng uniformGen(seed_);
    RandomSequenceGenerator unifSeqGen(dimensions, uniformGen);
    
    // Create inverse cumulative RSG for normal variates
    InverseCumulativeRsg<RandomSequenceGenerator<MersenneTwisterUniformRng>, InverseCumulativeNormal> rng(unifSeqGen);
    
    for (Size path = 0; path < numPaths_; ++path) {
        process_->setRegime(initialRegime);
        Array state = process_->initialValues();
        state[0] = initialVIX / 100.0;
        
        Time t = 0.0;
        
        for (Size step = 0; step < expiryDays; ++step) {
            const std::vector<Real>& gaussian = rng.nextSequence().value;
            Array dw(dimensions);
            for (Size d = 0; d < dimensions; ++d) {
                dw[d] = gaussian[d] * std::sqrt(dt);
            }
            
            state = process_->evolve(t, state, dt, dw);
            t += dt;
            
            if (state[0] < 0.0) state[0] = 0.0;
        }
        
        Real finalVIX = state[0] * 100.0;
        
        // Put payoff
        Real payoff = std::max(strike - finalVIX, 0.0);
        sumPayoffs += payoff;
    }
    
    Real discount = std::exp(-riskFreeRate * maturity);
    return (sumPayoffs / numPaths_) * discount;
}

Matrix VIXOptionPricer::simulateVIXPaths(
    Real initialVIX,
    Size initialRegime,
    Size numDays,
    Size numPathsToReturn) {
    
    if (numPathsToReturn == 0 || numPathsToReturn > numPaths_) {
        numPathsToReturn = numPaths_;
    }
    
    Time dt = 1.0 / 252.0;
    Size dimensions = process_->size();
    
    Matrix paths(numPathsToReturn, numDays + 1);
    
    // Create uniform random sequence generator
    MersenneTwisterUniformRng uniformGen(seed_);
    RandomSequenceGenerator unifSeqGen(dimensions, uniformGen);
    
    // Create inverse cumulative RSG for normal variates
    InverseCumulativeRsg<RandomSequenceGenerator<MersenneTwisterUniformRng>, InverseCumulativeNormal> rng(unifSeqGen);
    
    for (Size path = 0; path < numPathsToReturn; ++path) {
        process_->setRegime(initialRegime);
        Array state = process_->initialValues();
        state[0] = initialVIX / 100.0;
        
        paths[path][0] = initialVIX;
        
        Time t = 0.0;
        
        for (Size step = 0; step < numDays; ++step) {
            const std::vector<Real>& gaussian = rng.nextSequence().value;
            Array dw(dimensions);
            for (Size d = 0; d < dimensions; ++d) {
                dw[d] = gaussian[d] * std::sqrt(dt);
            }
            
            state = process_->evolve(t, state, dt, dw);
            t += dt;
            
            if (state[0] < 0.0) state[0] = 0.0;
            
            paths[path][step + 1] = state[0] * 100.0;
        }
    }
    
    return paths;
}
