#ifndef REGIME_PROCESS_HPP
#define REGIME_PROCESS_HPP

#include <ql/stochasticprocess.hpp>
#include <ql/math/matrix.hpp>

#include <random>
#include <vector>

using namespace QuantLib;

// A minimal regime-switching wrapper over StochasticProcess.
class RegimeProcess : public StochasticProcess {
public:
    RegimeProcess(const std::vector<boost::shared_ptr<StochasticProcess>>& processes,
                  const Matrix& transitionMatrix,
                  Time dt);

    Size size() const override;

    virtual Array initialValues() const override;

    virtual Array drift(Time t, const Array& x) const override;

    virtual Matrix diffusion(Time t, const Array& x) const override;

    virtual Array expectation(Time t0, const Array& x0, Time dt) const override;

    virtual Matrix stdDeviation(Time t0, const Array& x0, Time dt) const override;

    virtual Matrix covariance(Time t0, const Array& x0, Time dt) const override;

    virtual Array evolve(Time t0, const Array& x0, Time dt, const Array& dw) const override;

    void setRegime(Size r);
    Size regime() const;

private:
    Size sampleNextRegime(Size current) const;

    std::vector<boost::shared_ptr<StochasticProcess>> processes_;
    Matrix P_;
    Time dt_;
    Size K_;
    mutable Size currentRegime_;
    mutable std::mt19937 rng_;
};

#endif // REGIME_PROCESS_HPP
