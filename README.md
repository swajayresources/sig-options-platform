# SIG Options Market Making Platform

Sophisticated options trading system built for the SIG internship application. Implements a complete options market making platform with real-time Greeks, implied volatility surfaces, and automated trading strategies.

## What It Does

- Black-Scholes, Binomial, Monte Carlo options pricing
- Full Greeks: Delta, Gamma, Vega, Theta, Rho + higher-order (Vomma, Vanna, Charm)
- Implied volatility surface construction
- Automated market making with bid-ask spread optimization
- Real-time options chain visualization
- Portfolio-level risk aggregation and hedging

## Quick Start

### Prerequisites
- Python 3.9+
- No external services required

### Run the Streamlit Platform

```bash
pip install streamlit plotly pandas numpy scipy
cd streamlit_options_platform
streamlit run main.py
```

Open http://localhost:8501

### Run the Demo

```bash
pip install streamlit plotly pandas numpy scipy
cd streamlit_options_platform
python demo.py
```

### Full install

```bash
pip install -r requirements.txt
cd streamlit_options_platform
streamlit run main.py
```

## Dependency Notes

- **C++ engine**: Optional. The Python implementation covers all functionality. C++ source is in `cpp_engine/` for performance-critical paths.
- **No external APIs required**: Demo data is included in `streamlit_options_platform/demo_data/`.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit, Plotly |
| Pricing engine | Python (NumPy, SciPy) + optional C++ |
| Vol surface | Custom SABR/Heston implementation |
| Market data | Simulated (demo) / yfinance (live) |

## Project Structure

```
SIG3/
├── streamlit_options_platform/
│   ├── main.py                # Entry point — run this
│   ├── demo.py                # Standalone demo
│   ├── utils/                 # Pricing, vol surface, data utils
│   ├── strategies/            # Market making strategies
│   ├── backtesting/           # Strategy backtesting
│   └── demo_data/             # Sample data for demo mode
├── cpp_engine/                # High-performance C++ pricing (optional)
├── python_api/                # Python wrappers
├── market_data/               # Data integration
├── risk_management/           # Hedging tools
└── web_interface/             # Alternative web UI
```
