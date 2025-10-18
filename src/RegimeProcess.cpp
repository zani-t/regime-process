#include "RegimeProcess.hpp"

RegimeProcess::RegimeProcess(
    const std::vector<boost::shared_ptr<StochasticProcess>>& processes,
    const Matrix& transitionMatrix,
    Time dt)
: processes_(processes), P_(transitionMatrix), dt_(dt), K_(processes.size()) {
    QL_REQUIRE(K_ > 0, "must provide at least one regime process");
    QL_REQUIRE(P_.rows() == static_cast<Size>(K_) && P_.columns() == static_cast<Size>(K_),
               "transition matrix must be KxK");
    
    // Check that all processes have the same dimensions
    Size d = processes[0]->size();
    for (const auto& p : processes) {
        QL_REQUIRE(p->size() == d, "all processes must have the same dimension");
    }
    
    // Initialize RNG
    rng_.seed(123456);
    currentRegime_ = 0;
}

Size RegimeProcess::size() const {
    return processes_[0]->size();
}

Array RegimeProcess::initialValues() const {
    return processes_[0]->initialValues();
}

Array RegimeProcess::drift(Time t, const Array& x) const {
    return processes_[currentRegime_]->drift(t, x);
}

Matrix RegimeProcess::diffusion(Time t, const Array& x) const {
    return processes_[currentRegime_]->diffusion(t, x);
}

Array RegimeProcess::expectation(Time t0, const Array& x0, Time dt) const {
    return processes_[currentRegime_]->expectation(t0, x0, dt);
}

Matrix RegimeProcess::stdDeviation(Time t0, const Array& x0, Time dt) const {
    return processes_[currentRegime_]->stdDeviation(t0, x0, dt);
}

Matrix RegimeProcess::covariance(Time t0, const Array& x0, Time dt) const {
    return processes_[currentRegime_]->covariance(t0, x0, dt);
}

Array RegimeProcess::evolve(Time t0, const Array& x0, Time dt, const Array& dw) const {
    // Sample next regime index
    Size next = sampleNextRegime(currentRegime_);
    
    // Delegate evolution to the selected regime's process
    Array x1 = processes_[next]->evolve(t0, x0, dt, dw);
    
    // Update current regime
    currentRegime_ = next;
    return x1;
}

void RegimeProcess::setRegime(Size r) {
    QL_REQUIRE(r < K_, "invalid regime");
    currentRegime_ = r;
}

Size RegimeProcess::regime() const {
    return currentRegime_;
}

Size RegimeProcess::sampleNextRegime(Size current) const {
    std::vector<double> probs(K_);
    for (Size j = 0; j < K_; ++j)
        probs[j] = P_[current][j];
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return static_cast<Size>(dist(rng_));
}
