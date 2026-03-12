"""
Options Backtesting Framework
Professional-grade backtesting and validation system for options trading strategies
"""

from .backtesting_engine import (
    BacktestingEngine,
    Order,
    Trade,
    Position,
    OptionContract,
    MarketData,
    OptionType,
    OrderType
)

from .strategy_validator import (
    StrategyValidator,
    ValidationResult,
    PerformanceAttributor
)

from .performance_analytics import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    GreeksAttribution
)

from .monte_carlo_engine import (
    MonteCarloEngine,
    MonteCarloResult,
    MonteCarloSummary
)

from .backtesting_dashboard import (
    BacktestingDashboard,
    run_backtesting_dashboard
)

__version__ = "1.0.0"
__author__ = "Professional Trading Systems"

__all__ = [
    # Core backtesting
    "BacktestingEngine",
    "Order",
    "Trade",
    "Position",
    "OptionContract",
    "MarketData",
    "OptionType",
    "OrderType",

    # Strategy validation
    "StrategyValidator",
    "ValidationResult",
    "PerformanceAttributor",

    # Performance analytics
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "GreeksAttribution",

    # Monte Carlo simulation
    "MonteCarloEngine",
    "MonteCarloResult",
    "MonteCarloSummary",

    # Dashboard
    "BacktestingDashboard",
    "run_backtesting_dashboard"
]