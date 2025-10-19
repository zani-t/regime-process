#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <ql/option.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/processes/coxingersollrossprocess.hpp>
#include <ql/processes/ornsteinuhlenbeckprocess.hpp>
#include "RegimeProcess.hpp"
#include "MonteCarloEngine.hpp"

namespace py = pybind11;
using namespace QuantLib;

// Tell pybind11 that boost::shared_ptr is a holder type
PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

// Helper to create CIR process
boost::shared_ptr<StochasticProcess> create_cir_process(
    Real x0, Real kappa, Real theta, Real sigma) {
    return boost::shared_ptr<StochasticProcess>(
        new CoxIngersollRossProcess(kappa, theta, sigma, x0)
    );
}

// Helper to create OU process
boost::shared_ptr<StochasticProcess> create_ou_process(
    Real kappa, Real sigma, Real x0, Real theta) {
    return boost::shared_ptr<StochasticProcess>(
        new OrnsteinUhlenbeckProcess(kappa, sigma, x0, theta)
    );
}

// Convert numpy array to QuantLib Array
Array numpy_to_array(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Array must be 1-dimensional");
    }
    double* ptr = static_cast<double*>(buf.ptr);
    Array result(buf.shape[0]);
    for (size_t i = 0; i < buf.shape[0]; ++i) {
        result[i] = ptr[i];
    }
    return result;
}

// Convert QuantLib Array to numpy array
py::array_t<double> array_to_numpy(const Array& arr) {
    py::array_t<double> result(arr.size());
    auto r = result.mutable_unchecked<1>();
    for (Size i = 0; i < arr.size(); ++i) {
        r(i) = arr[i];
    }
    return result;
}

// Convert QuantLib Matrix to numpy array
py::array_t<double> matrix_to_numpy(const Matrix& mat) {
    py::array_t<double> result({mat.rows(), mat.columns()});
    auto r = result.mutable_unchecked<2>();
    for (Size i = 0; i < mat.rows(); ++i) {
        for (Size j = 0; j < mat.columns(); ++j) {
            r(i, j) = mat[i][j];
        }
    }
    return result;
}

PYBIND11_MODULE(vixmodels, m) {
    m.doc() = "VIX regime-switching models with QuantLib backend";
    
    // Expose QuantLib Array
    py::class_<Array>(m, "Array")
        .def(py::init<Size>())
        .def("__len__", &Array::size)
        .def("__getitem__", [](const Array& a, Size i) {
            if (i >= a.size()) throw py::index_error();
            return a[i];
        })
        .def("__setitem__", [](Array& a, Size i, Real val) {
            if (i >= a.size()) throw py::index_error();
            a[i] = val;
        })
        .def("size", &Array::size)
        .def("to_numpy", &array_to_numpy);
    
    // Expose QuantLib Matrix
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<Size, Size>())
        .def("__getitem__", [](const Matrix& m, py::tuple idx) {
            if (idx.size() != 2) throw std::runtime_error("Matrix index must be 2D");
            Size i = idx[0].cast<Size>();
            Size j = idx[1].cast<Size>();
            if (i >= m.rows() || j >= m.columns()) throw py::index_error();
            return m[i][j];
        })
        .def("__setitem__", [](Matrix& m, py::tuple idx, Real val) {
            if (idx.size() != 2) throw std::runtime_error("Matrix index must be 2D");
            Size i = idx[0].cast<Size>();
            Size j = idx[1].cast<Size>();
            if (i >= m.rows() || j >= m.columns()) throw py::index_error();
            m[i][j] = val;
        })
        .def("rows", &Matrix::rows)
        .def("columns", &Matrix::columns)
        .def("to_numpy", &matrix_to_numpy);
    
    // Expose Option::Type enum
    py::enum_<Option::Type>(m, "Option")
        .value("Call", Option::Call)
        .value("Put", Option::Put);
    
    // Expose Payoff base class
    py::class_<Payoff, boost::shared_ptr<Payoff>>(m, "Payoff")
        .def("__call__", &Payoff::operator());
    
    // Expose PlainVanillaPayoff
    py::class_<PlainVanillaPayoff, Payoff, boost::shared_ptr<PlainVanillaPayoff>>(
        m, "PlainVanillaPayoff")
        .def(py::init<Option::Type, Real>(),
             py::arg("type"), py::arg("strike"));
    
    // Expose StochasticProcess base class (must come before derived classes)
    py::class_<StochasticProcess, boost::shared_ptr<StochasticProcess>>(m, "StochasticProcess")
        .def("size", &StochasticProcess::size);
    
    // Helper functions to create processes (return boost::shared_ptr)
    m.def("create_cir_process", &create_cir_process,
          py::return_value_policy::take_ownership,
          py::arg("x0"), py::arg("kappa"), py::arg("theta"), py::arg("sigma"),
          "Create a Cox-Ingersoll-Ross process");
    
    m.def("create_ou_process", &create_ou_process,
          py::return_value_policy::take_ownership,
          py::arg("kappa"), py::arg("sigma"), py::arg("x0"), py::arg("theta"),
          "Create an Ornstein-Uhlenbeck process");
    
    m.def("numpy_to_array", &numpy_to_array,
          py::arg("arr"), "Convert numpy array to QuantLib Array");
    
    // RegimeProcess wrapper
    py::class_<RegimeProcess, boost::shared_ptr<RegimeProcess>>(m, "RegimeProcess")
        .def(py::init<const std::vector<boost::shared_ptr<StochasticProcess>>&,
                      const Matrix&, Time>(),
             py::arg("processes"), py::arg("transition_matrix"), py::arg("dt"))
        .def("set_regime", &RegimeProcess::setRegime, py::arg("regime"))
        .def("regime", &RegimeProcess::regime)
        .def("size", &RegimeProcess::size)
        .def("initial_values", &RegimeProcess::initialValues)
        .def("drift", &RegimeProcess::drift, py::arg("t"), py::arg("x"))
        .def("diffusion", &RegimeProcess::diffusion, py::arg("t"), py::arg("x"));
    
    // RegimeSwitchingMCEngine
    py::class_<RegimeSwitchingMCEngine>(m, "RegimeSwitchingMCEngine")
        .def(py::init<const boost::shared_ptr<RegimeProcess>&, Size, Size, BigNatural>(),
             py::arg("process"), py::arg("num_paths"), py::arg("time_steps"),
             py::arg("seed") = 42)
        .def("price_european", [](RegimeSwitchingMCEngine& self,
                                   Time maturity,
                                   py::array_t<double> initial_state,
                                   Size initial_regime,
                                   Rate risk_free_rate,
                                   py::function payoff_func) {
            Array state = numpy_to_array(initial_state);
            
            // Wrap Python payoff function
            auto payoff = [payoff_func](const Array& x) -> Real {
                py::array_t<double> np_x = array_to_numpy(x);
                return payoff_func(np_x).cast<Real>();
            };
            
            return self.priceEuropean(maturity, state, initial_regime,
                                     risk_free_rate, payoff);
        }, py::arg("maturity"), py::arg("initial_state"), py::arg("initial_regime"),
           py::arg("risk_free_rate"), py::arg("payoff"),
           "Price a European option with custom payoff")
        .def("simulate_paths", [](RegimeSwitchingMCEngine& self,
                                   Time maturity,
                                   py::array_t<double> initial_state,
                                   Size initial_regime,
                                   Size num_paths_to_return) {
            Array state = numpy_to_array(initial_state);
            auto paths = self.simulatePaths(maturity, state, initial_regime,
                                           num_paths_to_return);
            
            // Convert to numpy array
            Size nPaths = paths.size();
            Size nSteps = paths[0].size();
            Size nDims = paths[0][0].size();
            
            py::array_t<double> result({nPaths, nSteps, nDims});
            auto r = result.mutable_unchecked<3>();
            
            for (Size i = 0; i < nPaths; ++i) {
                for (Size j = 0; j < nSteps; ++j) {
                    for (Size k = 0; k < nDims; ++k) {
                        r(i, j, k) = paths[i][j][k];
                    }
                }
            }
            
            return result;
        }, py::arg("maturity"), py::arg("initial_state"), py::arg("initial_regime"),
           py::arg("num_paths_to_return") = 0,
           "Simulate paths and return as numpy array")
        .def("get_regime_history", [](RegimeSwitchingMCEngine& self,
                                       Time maturity,
                                       py::array_t<double> initial_state,
                                       Size initial_regime) {
            Array state = numpy_to_array(initial_state);
            auto regimes = self.getRegimeHistory(maturity, state, initial_regime);
            
            py::array_t<int> result(regimes.size());
            auto r = result.mutable_unchecked<1>();
            for (Size i = 0; i < regimes.size(); ++i) {
                r(i) = static_cast<int>(regimes[i]);
            }
            return result;
        }, py::arg("maturity"), py::arg("initial_state"), py::arg("initial_regime"),
           "Get regime history for a single path");
    
    // VIXRSMCEngine
    py::class_<VIXRSMCEngine>(m, "VIXRSMCEngine")
        .def(py::init<const boost::shared_ptr<RegimeProcess>&, Size, BigNatural>(),
             py::arg("process"), py::arg("num_paths"), py::arg("seed") = 42)
        .def("price_call", &VIXRSMCEngine::priceCall,
             py::arg("initial_vix"), py::arg("initial_regime"),
             py::arg("strike"), py::arg("expiry_days"),
             py::arg("risk_free_rate"),
             "Price a VIX call option using Monte Carlo")
        .def("price_put", &VIXRSMCEngine::pricePut,
             py::arg("initial_vix"), py::arg("initial_regime"),
             py::arg("strike"), py::arg("expiry_days"),
             py::arg("risk_free_rate"),
             "Price a VIX put option using Monte Carlo")
        .def("simulate_vix_paths", [](VIXRSMCEngine& self,
                                       Real initial_vix,
                                       Size initial_regime,
                                       Size num_days,
                                       Size num_paths_to_return) {
            Matrix paths = self.simulateVIXPaths(initial_vix, initial_regime,
                                                 num_days, num_paths_to_return);
            return matrix_to_numpy(paths);
        }, py::arg("initial_vix"), py::arg("initial_regime"),
           py::arg("num_days"), py::arg("num_paths_to_return") = 0,
           "Simulate VIX paths and return as numpy array");
}
