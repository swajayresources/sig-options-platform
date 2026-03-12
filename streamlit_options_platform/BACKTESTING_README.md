# Professional Options Backtesting Framework

## 🎯 Overview

This comprehensive backtesting framework provides institutional-grade testing and validation capabilities for options trading strategies. Built with rigorous statistical methods and realistic market simulation, it enables robust strategy development and risk assessment.

## 🚀 Key Features

### 📊 **Backtesting Engine**
- **Historical Data Replay**: Realistic market data simulation with bid-ask spreads
- **Transaction Cost Modeling**: Commission, slippage, and market impact simulation
- **Greeks-based Portfolio Simulation**: Real-time delta, gamma, theta, vega tracking
- **Exercise and Assignment**: Automatic handling of option expiry and early exercise
- **Risk Management**: Real-time risk limit monitoring and violation alerts

### ✅ **Strategy Validation Framework**
- **Out-of-sample Testing**: Robust train/test split validation
- **Walk-forward Analysis**: Rolling window performance validation
- **Statistical Significance Testing**: T-tests and confidence intervals
- **Overfitting Detection**: Automated detection of curve-fitting
- **Model Validation**: Comprehensive scoring system (0-100)

### 🎲 **Monte Carlo Simulation**
- **Multi-scenario Analysis**: Bull, bear, high volatility, crash scenarios
- **Correlation Modeling**: Realistic cross-asset correlation simulation
- **Volatility Surface Evolution**: Dynamic volatility smile modeling
- **Jump Risk**: Rare event simulation and tail risk analysis
- **Confidence Intervals**: 90%, 95%, 99% confidence bounds

### 📈 **Performance Analytics**
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar ratios
- **Greeks Attribution**: Delta, gamma, theta, vega P&L breakdown
- **Risk Analysis**: VaR, Expected Shortfall, maximum drawdown
- **Benchmark Comparison**: Alpha generation and tracking error
- **Trade Analysis**: Win rate, profit factor, holding periods

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install streamlit plotly pandas numpy scipy scikit-learn psutil
```

### Directory Structure
```
backtesting/
├── __init__.py                    # Framework initialization
├── backtesting_engine.py          # Core backtesting engine
├── strategy_validator.py          # Strategy validation framework
├── performance_analytics.py       # Performance analysis tools
├── monte_carlo_engine.py          # Monte Carlo simulation
└── backtesting_dashboard.py       # Streamlit dashboard

tests/
├── __init__.py
├── test_backtesting.py            # Comprehensive test suite
└── run_tests.py                   # Automated test runner

strategies/
├── __init__.py
└── example_strategies.py          # Example strategy implementations
```

## 📋 Quick Start

### 1. Basic Backtesting

```python
from backtesting import BacktestingEngine, Order, OptionContract, OptionType, OrderType
from datetime import datetime, timedelta

# Initialize engine
engine = BacktestingEngine(initial_capital=1000000)

# Define strategy
def simple_strategy(market_data, positions, capital):
    orders = []

    if len(positions) == 0:
        # Buy ATM straddle
        strike = market_data.underlying_price
        expiry = market_data.timestamp + timedelta(days=30)

        call_order = Order(
            OptionContract("SPY", strike, expiry, OptionType.CALL),
            quantity=10,
            order_type=OrderType.BUY
        )

        put_order = Order(
            OptionContract("SPY", strike, expiry, OptionType.PUT),
            quantity=10,
            order_type=OrderType.BUY
        )

        orders.extend([call_order, put_order])

    return orders

# Run backtest
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

results = engine.run_backtest(simple_strategy, start_date, end_date)

# Access results
print(f"Total Return: {results['performance_metrics']['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']:.2f}%")
```

### 2. Strategy Validation

```python
from backtesting import StrategyValidator

# Initialize validator
validator = StrategyValidator()

# Run comprehensive validation
validation_result = validator.validate_strategy(
    strategy_function=simple_strategy,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    strategy_name="Simple Straddle"
)

# Generate validation report
report = validator.generate_validation_report(validation_result)
print(report)

# Access validation metrics
print(f"Validation Score: {validation_result.validation_score:.1f}/100")
print(f"Overfitting Score: {validation_result.overfitting_score:.3f}")
```

### 3. Monte Carlo Analysis

```python
from backtesting import MonteCarloEngine

# Initialize Monte Carlo engine
mc_engine = MonteCarloEngine(num_simulations=1000)

# Run Monte Carlo analysis
mc_summary = mc_engine.run_monte_carlo_backtest(
    strategy_function=simple_strategy,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Access results
print(f"Mean Return: {mc_summary.mean_return:.2f}%")
print(f"Success Probability: {mc_summary.success_probability:.1f}%")
print(f"95th Percentile: {mc_summary.percentiles['95th']:.2f}%")
print(f"5th Percentile: {mc_summary.percentiles['5th']:.2f}%")
```

### 4. Performance Analytics

```python
from backtesting import PerformanceAnalyzer

# Initialize analyzer
analyzer = PerformanceAnalyzer()

# Calculate comprehensive metrics
metrics = analyzer.calculate_comprehensive_metrics(
    equity_curve=results['equity_curve'],
    trades=results['trades']
)

# Greeks attribution analysis
attribution = analyzer.analyze_greeks_attribution(
    greeks_history=results['greeks_history'],
    underlying_prices=underlying_price_series
)

print(f"Delta P&L: ${attribution.delta_pnl:.2f}")
print(f"Gamma P&L: ${attribution.gamma_pnl:.2f}")
print(f"Theta P&L: ${attribution.theta_pnl:.2f}")
print(f"Vega P&L: ${attribution.vega_pnl:.2f}")
```

## 🎮 Interactive Dashboard

### Launch Backtesting Dashboard

```python
from backtesting import run_backtesting_dashboard

# Launch Streamlit dashboard
run_backtesting_dashboard()
```

The dashboard provides:
- **Strategy Backtesting Interface**: Interactive parameter configuration
- **Validation Framework**: Comprehensive strategy validation
- **Monte Carlo Analysis**: Multi-scenario simulation
- **Performance Analytics**: Advanced performance metrics
- **Model Validation**: Pricing model accuracy testing
- **Test Reports**: Automated quality assurance reports

## 🧪 Testing Framework

### Run Comprehensive Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python -m unittest tests.test_backtesting.TestBlackScholesCalculations
python -m unittest tests.test_backtesting.TestBacktestingEngine
python -m unittest tests.test_backtesting.TestStrategyValidator
```

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end workflow testing
3. **Performance Tests**: Speed and memory benchmarks
4. **Validation Tests**: Model accuracy verification
5. **Stress Tests**: Large-scale and edge case testing

## 📊 Example Strategies

### Delta-Neutral Strategy
```python
from strategies import DeltaNeutralStrategy

strategy = DeltaNeutralStrategy(
    target_delta=0.0,
    rebalance_threshold=10.0,
    max_position_size=0.1
)

def delta_neutral_orders(market_data, positions, capital):
    return strategy.generate_orders(market_data, positions, capital)
```

### Iron Condor Strategy
```python
from strategies import IronCondorStrategy

strategy = IronCondorStrategy(
    wing_width=10.0,
    target_dte=45,
    profit_target=0.5,
    stop_loss=2.0
)

def iron_condor_orders(market_data, positions, capital):
    return strategy.generate_orders(market_data, positions, capital)
```

### Volatility Trading Strategy
```python
from strategies import VolatilityTradingStrategy

strategy = VolatilityTradingStrategy(
    vol_threshold_low=0.15,
    vol_threshold_high=0.35,
    target_dte=30
)

def volatility_orders(market_data, positions, capital):
    return strategy.generate_orders(market_data, positions, capital)
```

## 📈 Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: Excess return per unit of risk
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return-to-maximum drawdown ratio
- **Information Ratio**: Excess return vs tracking error

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at confidence levels
- **Expected Shortfall**: Average loss beyond VaR
- **Maximum Drawdown**: Peak-to-trough decline
- **Greeks Risk**: Delta, gamma, vega exposure limits

### Trade Analytics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss ratio
- **Average Win/Loss**: Mean profit and loss per trade
- **Holding Period**: Average time in trades

## ⚙️ Configuration Options

### Backtesting Parameters
```python
engine = BacktestingEngine(
    initial_capital=1000000,           # Starting capital
    commission_per_contract=1.0,       # Commission cost
    slippage_pct=0.005,               # Slippage percentage
    bid_ask_spread_pct=0.02,          # Bid-ask spread
    market_impact_factor=0.001,        # Market impact cost
    max_position_size=0.1,            # Maximum position size
    max_portfolio_delta=1000,         # Delta limit
    max_portfolio_gamma=500,          # Gamma limit
    max_portfolio_vega=10000          # Vega limit
)
```

### Validation Parameters
```python
validator = StrategyValidator(
    min_sample_size=252               # Minimum data points
)
```

### Monte Carlo Parameters
```python
mc_engine = MonteCarloEngine(
    num_simulations=1000              # Number of simulations
)
```

## 🔬 Advanced Features

### Custom Market Scenarios
```python
custom_scenarios = {
    'volatility_regime': {
        'low_vol': {'prob': 0.3, 'vol_multiplier': 0.6},
        'high_vol': {'prob': 0.2, 'vol_multiplier': 2.0}
    },
    'jump_risk': {
        'prob_jump': 0.02,
        'jump_mean': -0.02,
        'jump_std': 0.05
    }
}

mc_results = mc_engine.run_monte_carlo_backtest(
    strategy_function=your_strategy,
    start_date=start_date,
    end_date=end_date,
    market_scenarios=custom_scenarios
)
```

### Performance Benchmarking
```python
benchmark_data = [...]  # Benchmark price series

comparison = analyzer.benchmark_comparison(
    strategy_metrics=performance_metrics,
    benchmark_data=benchmark_data
)

print(f"Excess Return: {comparison['excess_return']:.2f}%")
print(f"Information Ratio: {comparison['information_ratio']:.3f}")
```

### Risk Report Generation
```python
risk_report = analyzer.create_risk_report(
    equity_curve=results['equity_curve'],
    greeks_history=results['greeks_history']
)

print(f"95% VaR: ${risk_report['var_95']:.2f}")
print(f"Expected Shortfall: ${risk_report['expected_shortfall_95']:.2f}")
```

## 🚨 Best Practices

### Strategy Development
1. **Start Simple**: Begin with basic strategies before adding complexity
2. **Use Realistic Assumptions**: Include transaction costs and market impact
3. **Validate Thoroughly**: Use multiple validation methods
4. **Check for Overfitting**: Monitor in-sample vs out-of-sample performance
5. **Test Edge Cases**: Stress test under extreme market conditions

### Risk Management
1. **Set Position Limits**: Limit exposure per trade and strategy
2. **Monitor Greeks**: Track delta, gamma, vega exposure
3. **Implement Stop Losses**: Define maximum acceptable losses
4. **Diversify Strategies**: Use multiple uncorrelated approaches
5. **Regular Rebalancing**: Maintain target risk profile

### Validation Process
1. **Reserve Out-of-Sample Data**: Never use for strategy development
2. **Use Walk-Forward Analysis**: Test robustness over time
3. **Run Monte Carlo**: Understand range of potential outcomes
4. **Check Statistical Significance**: Ensure results aren't due to chance
5. **Document Everything**: Maintain detailed records of tests and results

## 📚 API Reference

### Core Classes
- `BacktestingEngine`: Main backtesting functionality
- `StrategyValidator`: Strategy validation and testing
- `PerformanceAnalyzer`: Performance metrics calculation
- `MonteCarloEngine`: Monte Carlo simulation

### Data Classes
- `Order`: Trade order representation
- `Trade`: Executed trade information
- `Position`: Current position tracking
- `OptionContract`: Option contract specification
- `MarketData`: Market data snapshot

### Strategy Classes
- `DeltaNeutralStrategy`: Delta-neutral implementation
- `IronCondorStrategy`: Iron condor implementation
- `VolatilityTradingStrategy`: Volatility-based trading
- `MomentumStrategy`: Momentum-based directional trading

## 🤝 Contributing

This framework is designed for professional options trading research and development. Contributions should focus on:

1. **Enhanced Pricing Models**: Additional pricing methodologies
2. **Advanced Strategies**: New strategy implementations
3. **Risk Metrics**: Additional risk measurement tools
4. **Performance Analytics**: Enhanced attribution analysis
5. **Testing Coverage**: Expanded test scenarios

## ⚠️ Disclaimer

This backtesting framework is for research and educational purposes. Past performance does not guarantee future results. Always validate strategies thoroughly before live trading and consider:

- Model limitations and assumptions
- Market regime changes
- Transaction costs and liquidity
- Regulatory requirements
- Risk management protocols

## 📞 Support

For technical support or questions about the backtesting framework:
- Review the comprehensive test suite
- Check the example strategies
- Use the interactive dashboard for analysis
- Validate models using the built-in testing tools

The framework provides institutional-grade backtesting capabilities with rigorous validation to support professional options trading strategy development.