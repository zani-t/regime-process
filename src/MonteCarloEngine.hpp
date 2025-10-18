#ifndef MONTE_CARLO_ENGINE_HPP
#define MONTE_CARLO_ENGINE_HPP

#include "RegimeProcess.hpp"
#include <ql/methods/montecarlo/path.hpp>
#include <ql/methods/montecarlo/pathgenerator.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/math/randomnumbers/inversecumulativerng.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <boost/function.hpp>
#include <vector>

using namespace QuantLib;

// Generic Monte Carlo engine for regime-switching processes
class RegimeSwitchingMCEngine {
public:
    // Constructor
    RegimeSwitchingMCEngine(
        const boost::shared_ptr<RegimeProcess>& process,
        Size numPaths,
        Size timeSteps,
        BigNatural seed = 42
    );

    // Price a European option with custom payoff
    Real priceEuropean(
        Time maturity,
        const Array& initialState,
        Size initialRegime,
        Rate riskFreeRate,
        const boost::function<Real(const Array&)>& payoff
    );

    // Simulate paths and return them
    std::vector<std::vector<Array>> simulatePaths(
        Time maturity,
        const Array& initialState,
        Size initialRegime,
        Size numPathsToReturn = 0
    );

    // Get regime history for a single path
    std::vector<Size> getRegimeHistory(
        Time maturity,
        const Array& initialState,
        Size initialRegime
    );

private:
    boost::shared_ptr<RegimeProcess> process_;
    Size numPaths_;
    Size timeSteps_;
    BigNatural seed_;
};

// Specific VIX option pricer
class VIXOptionPricer {
public:
    VIXOptionPricer(
        const boost::shared_ptr<RegimeProcess>& process,
        Size numPaths,
        BigNatural seed = 42
    );

    // Price VIX call option
    Real priceCall(
        Real initialVIX,
        Size initialRegime,
        Real strike,
        Size expiryDays,
        Rate riskFreeRate
    );

    // Price VIX put option
    Real pricePut(
        Real initialVIX,
        Size initialRegime,
        Real strike,
        Size expiryDays,
        Rate riskFreeRate
    );

    // Simulate VIX paths (returns matrix: paths × time)
    Matrix simulateVIXPaths(
        Real initialVIX,
        Size initialRegime,
        Size numDays,
        Size numPathsToReturn = 0
    );

private:
    boost::shared_ptr<RegimeProcess> process_;
    Size numPaths_;
    BigNatural seed_;
};

#endif // MONTE_CARLO_ENGINE_HPP
