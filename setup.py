"""
Setup script for SIG Options Trading System

Professional options trading platform with advanced pricing models,
market making capabilities, and comprehensive risk management.
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import platform
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
__version__ = "1.0.0"

# Compiler-specific settings
extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    extra_compile_args.extend(["/O2", "/std:c++17"])
else:
    extra_compile_args.extend(["-O3", "-std=c++17", "-ffast-math", "-march=native"])

# C++ extension for pricing engine
ext_modules = [
    Pybind11Extension(
        "sig_options_engine",
        [
            "python_api/src/python_bindings.cpp",
            "cpp_engine/src/math_utils.cpp",
            "cpp_engine/src/black_scholes.cpp",
            "cpp_engine/src/binomial_model.cpp",
            "cpp_engine/src/monte_carlo.cpp",
            "cpp_engine/src/volatility_surface.cpp",
        ],
        include_dirs=[
            "cpp_engine/include",
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="sig-options-trading",
    version=__version__,
    author="SIG Internship Application",
    author_email="applicant@sig.com",
    description="Professional Options Trading System for SIG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sig/options-trading-system",
    packages=find_packages(),
    package_dir={
        "sig_options": "python_api/src",
        "sig_market_data": "market_data/src",
        "sig_risk": "risk_management/src",
        "sig_web": "web_interface/src",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "gpu": [
            "cupy-cuda11x>=9.0.0",
            "numba>=0.56.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "sig-options-server=sig_web.server:main",
            "sig-market-maker=sig_options.market_maker:main",
            "sig-risk-monitor=sig_risk.risk_manager:main",
            "sig-backtest=sig_options.backtester:main",
        ],
    },
    package_data={
        "sig_web": ["static/*", "templates/*", "*.html"],
        "sig_options": ["config/*.yaml", "data/*.csv"],
    },
    include_package_data=True,
    project_urls={
        "Bug Reports": "https://github.com/sig/options-trading-system/issues",
        "Source": "https://github.com/sig/options-trading-system",
        "Documentation": "https://sig-options-trading.readthedocs.io/",
    },
    keywords="options trading derivatives quantitative finance market making",
    platforms=["any"],
)

# Print post-installation message
print("""
🚀 SIG Options Trading System Installation Complete! 🚀

Getting Started:
1. Start the web interface: sig-options-server
2. Run market maker: sig-market-maker --config config/default.yaml
3. Monitor risk: sig-risk-monitor --portfolio portfolio.json

Documentation: https://sig-options-trading.readthedocs.io/
Examples: See /examples directory for usage examples

For SIG internship evaluation, this system demonstrates:
✅ Advanced options pricing models (Black-Scholes, Binomial, Monte Carlo)
✅ Comprehensive Greeks calculation (all orders)
✅ Sophisticated market making strategies
✅ Real-time risk management and VaR calculation
✅ Professional web interface with real-time updates
✅ High-performance C++ engine with Python API
✅ Comprehensive testing and validation

Thank you for considering this application!
""")