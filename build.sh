#!/bin/bash

# Build script for VIX regime-switching model

set -e  # Exit on error

echo "======================================"
echo "Building VIX Regime-Switching Model"
echo "======================================"

# Check for required packages
echo ""
echo "Checking dependencies..."

# Check for QuantLib
if ! pkg-config --exists QuantLib; then
    echo "Error: QuantLib not found!"
    echo "Install with: sudo apt-get install libquantlib0-dev (Ubuntu/Debian)"
    echo "           or brew install quantlib (macOS)"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found!"
    exit 1
fi

# Check for pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "pybind11 not found. Installing..."
    pip install pybind11
fi

# Check for other Python dependencies
echo "Checking Python dependencies..."
pip install numpy scipy pandas matplotlib 2>/dev/null || true

echo ""
echo "Building C++ extension..."

# Create build directory
mkdir -p build
cd build

# Run CMake
cmake ..

# Build
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Copy the built module to project root
if [ -f vixmodels*.so ]; then
    cp vixmodels*.so ../python/
    echo ""
    echo "✓ Build successful!"
    echo "  Module: $(ls ../vixmodels*.so)"
elif [ -f vixmodels*.dylib ]; then
    cp vixmodels*.dylib ../python/
    echo ""
    echo "✓ Build successful!"
    echo "  Module: $(ls ../vixmodels*.dylib)"
else
    echo ""
    echo "✗ Build failed: Module not found"
    exit 1
fi

cd ..

echo ""
echo "======================================"
echo "Build Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Place your VIX data in data/vix_data.csv"
echo "2. Run: python3 python/pricing.py"
echo ""
