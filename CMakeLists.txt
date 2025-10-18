cmake_minimum_required(VERSION 3.15)
project(RegimeVIX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Boost REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
pkg_check_modules(QUANTLIB REQUIRED QuantLib)

# Add pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(pybind11_DIR)
    message(STATUS "Found pybind11 at: ${pybind11_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
else()
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()

# Link directories
link_directories(/usr/local/lib)

# Python module
pybind11_add_module(vixmodels 
    src/bindings.cpp
    src/RegimeProcess.cpp
    src/MonteCarloEngine.cpp
)

target_include_directories(vixmodels PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${QUANTLIB_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(vixmodels PRIVATE 
    ${QUANTLIB_LIBRARIES} 
    ${Boost_LIBRARIES}
)

target_compile_options(vixmodels PRIVATE 
    ${QUANTLIB_CFLAGS_OTHER}
)

# Install the Python module
install(TARGETS vixmodels LIBRARY DESTINATION .)
