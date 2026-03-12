# Options Trading Analytics Platform

A comprehensive, professional-grade options trading analytics platform with real-time monitoring, advanced signal generation, and sophisticated risk management capabilities.

## 🚀 Features

### Portfolio Analytics
- **Real-time Portfolio Greeks Aggregation**: Live tracking of Delta, Gamma, Theta, Vega, and Rho
- **P&L Attribution Analysis**: Detailed breakdown by strategy, symbol, and risk factors
- **Time Decay Optimization**: Advanced theta tracking and decay analysis
- **Scenario Analysis & Stress Testing**: Monte Carlo simulations and stress scenarios
- **Cross-asset Correlation Impact**: Portfolio correlation analysis and risk assessment

### Market Analysis Tools
- **Options Flow Sentiment Analysis**: Real-time sentiment from options order flow
- **Put/Call Ratio Monitoring**: Advanced P/C ratio analysis with historical context
- **Volatility Surface Analysis**: 3D volatility surface visualization and analysis
- **Implied Volatility Percentile Ranking**: IV rank and percentile calculations
- **Unusual Options Activity Detection**: Real-time detection of large and unusual trades

### Trading Signals
- **Volatility Mean Reversion Signals**: Statistical arbitrage based on IV mean reversion
- **Options Mispricing Detection**: Black-Scholes theoretical vs market price analysis
- **Cross-asset Volatility Arbitrage**: Inter-market volatility spread opportunities
- **Earnings Volatility Prediction**: Pre-earnings volatility crush/expansion signals
- **Statistical Arbitrage Identification**: Pair trading and spread opportunities

### Professional Features
- **Real-time Options Chain Visualization**: Interactive options chain with Greeks
- **Greeks Ladder with Risk Levels**: Color-coded risk visualization
- **Interactive Strategy P&L Diagrams**: Dynamic profit/loss visualization
- **Monte Carlo Scenario Analysis**: Advanced risk simulation
- **Real-time Alert Management**: Sophisticated alerting with multiple notification channels

## 🏗️ Architecture

### Core Components

1. **Analytics Framework** (`analytics_framework.py`)
   - Central analytics engine
   - Market data processing
   - Greeks calculations
   - Risk metrics computation

2. **Portfolio Monitor** (`portfolio_monitor.py`)
   - Real-time portfolio tracking
   - Performance monitoring
   - Risk limit monitoring
   - Position analytics

3. **Market Analysis** (`market_analysis.py`)
   - Signal generation engine
   - Volatility analysis
   - Arbitrage detection
   - Market regime detection

4. **Flow Analysis** (`flow_analysis.py`)
   - Options flow processing
   - Sentiment calculation
   - Unusual activity detection
   - Flow pattern recognition

5. **Visualization Components** (`visualization_components.py`)
   - Interactive charting
   - 3D visualizations
   - Dashboard components
   - Real-time updates

6. **Alert System** (`alert_system.py`)
   - Rule-based alerting
   - Multi-channel notifications
   - Alert aggregation
   - Performance tracking

7. **Performance Analytics** (`performance_analytics.py`)
   - Comprehensive performance metrics
   - Attribution analysis
   - Benchmark comparison
   - Risk-adjusted returns

8. **Trading Interface** (`trading_interface.py`)
   - Web-based professional interface
   - Real-time WebSocket updates
   - Interactive dashboards
   - REST API endpoints

## 🚦 Getting Started

### Prerequisites

```bash
pip install numpy pandas plotly scipy aiohttp aiohttp-cors python-socketio asyncio jinja2
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd options_analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the trading interface:
```bash
python src/trading_interface.py
```

4. Access the platform at `http://localhost:8080`

### Configuration

Create a configuration file `config.json`:

```json
{
  "host": "0.0.0.0",
  "port": 8080,
  "risk_limits": {
    "max_delta": 1000,
    "max_gamma": 500,
    "max_theta": -200,
    "max_vega": 2000,
    "max_var": 100000
  },
  "notifications": {
    "email": {
      "enabled": true,
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "username": "your_email@gmail.com",
      "password": "your_password",
      "from_email": "alerts@yourcompany.com",
      "to_email": "trader@yourcompany.com"
    },
    "slack": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/services/..."
    }
  },
  "chart_theme": "plotly_dark",
  "update_frequency_ms": 1000
}
```

## 📊 Usage Examples

### Portfolio Monitoring

```python
from src.portfolio_monitor import PortfolioMonitor
from src.analytics_framework import Position

# Initialize portfolio monitor
config = {"risk_limits": {"max_delta": 1000}}
monitor = PortfolioMonitor(config)
await monitor.initialize()

# Update positions
positions = {
    "AAPL_240115C150": Position(
        symbol="AAPL_240115C150",
        quantity=10,
        average_price=5.50,
        current_price=6.20,
        market_value=6200,
        unrealized_pnl=700,
        realized_pnl=0,
        delta=0.65,
        gamma=0.02,
        theta=-0.05,
        vega=0.15,
        rho=0.03
    )
}

await monitor.update_positions(positions)

# Get portfolio summary
summary = await monitor.get_portfolio_summary()
print(f"Total P&L: ${summary['total_pnl']:.2f}")
print(f"Portfolio Delta: {summary['greeks']['delta']:.2f}")
```

### Signal Generation

```python
from src.market_analysis import MarketAnalysisEngine

# Initialize market analysis
analysis = MarketAnalysisEngine(config)
await analysis.initialize()

# Get active signals
signals = await analysis.get_active_signals(min_strength=0.7)

for signal in signals:
    print(f"Signal: {signal.symbol} - {signal.signal_type}")
    print(f"Strength: {signal.strength:.2f}, Confidence: {signal.confidence:.2f}")
    print(f"Direction: {signal.direction}")
```

### Flow Analysis

```python
from src.flow_analysis import FlowAnalysisEngine

# Initialize flow analysis
flow_engine = FlowAnalysisEngine(config)
await flow_engine.initialize()

# Get flow analytics
analytics = await flow_engine.get_flow_analytics(timeframe='1h')

for symbol, data in analytics.items():
    print(f"{symbol}: P/C Ratio = {data.put_call_ratio:.2f}")
    print(f"Sentiment Score: {data.sentiment_score:.2f}")
    print(f"Flow Direction: {data.flow_direction}")
```

### Visualization

```python
from src.visualization_components import VisualizationEngine

# Initialize visualization
viz = VisualizationEngine(config)
await viz.initialize()

# Generate portfolio dashboard
portfolio_data = await monitor.get_portfolio_summary()
dashboard = await viz.generate_portfolio_dashboard(portfolio_data)

# Convert to HTML
html = viz.get_chart_html(dashboard['portfolio_overview'])
```

## 🎯 Key Analytics

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Calmar Ratio**: Return vs maximum drawdown
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss ratio

### Risk Metrics
- **Portfolio Greeks**: Aggregated sensitivity measures
- **Concentration Risk**: Position size distribution
- **Correlation Risk**: Inter-asset correlation exposure
- **Leverage Ratio**: Total exposure vs capital
- **Stress Test Results**: Performance under adverse scenarios

### Options Flow Metrics
- **Put/Call Ratio**: Market sentiment indicator
- **Volume Trends**: Unusual volume detection
- **Large Block Activity**: Institutional flow tracking
- **Sentiment Scores**: Composite sentiment indicators
- **Flow Patterns**: Sweep, accumulation, hedge detection

## 🔧 Advanced Features

### Custom Signal Development

```python
from src.market_analysis import SignalGenerator

class CustomVolatilitySignal(SignalGenerator):
    async def generate_signals(self, market_data, historical_data):
        signals = []

        for symbol, data in market_data.items():
            # Custom signal logic
            iv_zscore = self.calculate_iv_zscore(data, historical_data[symbol])

            if abs(iv_zscore) > 2.0:
                signal = TradingSignal(
                    signal_id=f"custom_{symbol}",
                    symbol=symbol,
                    signal_type="custom_volatility",
                    direction="sell" if iv_zscore > 0 else "buy",
                    strength=min(abs(iv_zscore) / 3.0, 1.0),
                    confidence=0.8,
                    metadata={"iv_zscore": iv_zscore}
                )
                signals.append(signal)

        return signals
```

### Custom Alert Rules

```python
from src.alert_system import AlertRule, AlertType, AlertPriority

# Create custom alert rule
custom_rule = AlertRule(
    rule_id="portfolio_concentration",
    name="Portfolio Concentration Alert",
    description="Alert when single position exceeds 20% of portfolio",
    alert_type=AlertType.RISK_LIMIT,
    priority=AlertPriority.HIGH,
    condition="max(position.market_value for position in positions.values()) / sum(position.market_value for position in positions.values()) > 0.2",
    parameters={"max_concentration": 0.2},
    notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
)

await alert_manager.add_alert_rule(custom_rule)
```

## 📈 Performance Optimization

- **Async/Await**: Full asynchronous operation for real-time performance
- **Data Caching**: Intelligent caching of frequently accessed calculations
- **WebSocket Updates**: Efficient real-time data transmission
- **Vectorized Calculations**: NumPy-based high-performance computations
- **Connection Pooling**: Optimized database and API connections

## 🔒 Risk Management

### Built-in Risk Controls
- **Position Limits**: Configurable position size limits
- **Greeks Limits**: Delta, Gamma, Theta, Vega exposure limits
- **VaR Limits**: Value at Risk thresholds
- **Concentration Limits**: Maximum single position exposure
- **Leverage Limits**: Total leverage constraints

### Real-time Monitoring
- **Continuous Risk Assessment**: Real-time risk metric updates
- **Automatic Alerts**: Immediate notification of limit breaches
- **Stress Testing**: Regular portfolio stress test execution
- **Scenario Analysis**: What-if analysis capabilities

## 🌐 API Reference

### REST Endpoints

- `GET /api/portfolio/summary` - Portfolio overview
- `GET /api/portfolio/positions` - Detailed position data
- `GET /api/market/overview` - Market analysis summary
- `GET /api/market/signals` - Active trading signals
- `GET /api/flow/analytics` - Options flow analysis
- `GET /api/alerts/active` - Current active alerts
- `GET /api/performance/metrics` - Performance analytics

### WebSocket Events

- `portfolio_update` - Real-time portfolio changes
- `market_update` - Market data updates
- `new_signal` - New trading signal generated
- `alert_triggered` - New alert created
- `flow_update` - Options flow data update

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Black-Scholes Model**: Options pricing and Greeks calculations
- **Plotly**: Interactive visualization library
- **NumPy/SciPy**: High-performance numerical computing
- **AsyncIO**: Asynchronous programming framework
- **SocketIO**: Real-time bidirectional communication

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Email: support@optionsanalytics.com
- Documentation: [docs.optionsanalytics.com](https://docs.optionsanalytics.com)

---

**Disclaimer**: This software is for educational and research purposes. Options trading involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results.