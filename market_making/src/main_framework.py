"""
Options Market Making Framework - Main Integration Module

This module provides a unified interface to the complete options market making system,
integrating all components including strategies, quote management, hedging, portfolio
optimization, execution, performance monitoring, and the real-time dashboard.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import threading
import time
import warnings

# Import all framework components
from market_making_strategies import (
 StrategyManager, DeltaNeutralMarketMaker, VolatilityArbitrageStrategy,
 StatisticalArbitrageStrategy, PinRiskManagementStrategy, StrategyConfig,
 MarketData, Position, Trade, Greeks
)
from quote_management import QuoteEngine, QuoteRequest, AdverseSelectionProtector
from hedging_risk_management import HedgingRiskManager, RiskLimits, HedgingConfig
from portfolio_optimization import PortfolioOptimizer, OptimizationObjective, PortfolioConstraints
from execution_algorithms import ExecutionEngine, ExecutionOrder, ExecutionAlgorithm
from performance_monitoring import StrategyPerformanceMonitor, RiskMonitor
from backtesting_optimization import BacktestEngine, ParameterOptimizer, BacktestConfig
from monitoring_dashboard import MonitoringDashboard, DashboardConfig


@dataclass
class MarketMakingConfig:
 """Master configuration for the entire market making system"""
 # Strategy settings
 strategies_to_run: List[str] = None
 strategy_weights: Dict[str, float] = None

 # Risk management
 max_portfolio_value: float = 10000000.0 # $10M max portfolio
 max_daily_loss: float = 100000.0 # $100K max daily loss
 max_position_concentration: float = 0.2 # 20% max in single position

 # Quote management
 quote_refresh_frequency_ms: int = 100 # 100ms quote refresh
 max_spread_width: float = 0.25 # 25% max spread
 min_spread_width: float = 0.01 # 1% min spread

 # Execution settings
 default_execution_algorithm: str = "smart_order_routing"
 max_execution_time_seconds: int = 30

 # Performance monitoring
 enable_real_time_monitoring: bool = True
 dashboard_port: int = 5000

 # Data retention
 max_history_days: int = 30
 backup_frequency_hours: int = 6

 def __post_init__(self):
 if self.strategies_to_run is None:
 self.strategies_to_run = ["delta_neutral", "volatility_arbitrage"]

 if self.strategy_weights is None:
 self.strategy_weights = {
 "delta_neutral": 0.6,
 "volatility_arbitrage": 0.3,
 "statistical_arbitrage": 0.1
 }


class OptionsMarketMakingFramework:
 """
 Main framework class that orchestrates the entire options market making system

 This class integrates:
 - Multiple market making strategies
 - Real-time quote generation and management
 - Automated hedging and risk management
 - Portfolio optimization
 - Smart execution algorithms
 - Performance monitoring and attribution
 - Real-time dashboard
 """

 def __init__(self, config: MarketMakingConfig = None):
 self.config = config or MarketMakingConfig()

 # Initialize core components
 self._initialize_strategies()
 self._initialize_quote_engine()
 self._initialize_hedging_system()
 self._initialize_portfolio_optimizer()
 self._initialize_execution_engine()
 self._initialize_monitoring()
 self._initialize_dashboard()

 # State management
 self.is_running = False
 self.current_market_data: Dict[str, MarketData] = {}
 self.active_positions: Dict[str, Position] = {}
 self.trade_history: List[Trade] = []

 # Threading
 self.main_loop_thread: Optional[threading.Thread] = None
 self.monitoring_thread: Optional[threading.Thread] = None

 print("Options Market Making Framework initialized successfully!")

 def _initialize_strategies(self):
 """Initialize strategy manager and individual strategies"""
 self.strategy_manager = StrategyManager()

 # Create strategy configurations
 base_config = StrategyConfig(
 max_position_size=100,
 max_portfolio_delta=1000.0,
 min_spread_width=self.config.min_spread_width,
 max_spread_width=self.config.max_spread_width
 )

 # Initialize requested strategies
 for strategy_name in self.config.strategies_to_run:
 weight = self.config.strategy_weights.get(strategy_name, 1.0)

 if strategy_name == "delta_neutral":
 strategy = DeltaNeutralMarketMaker(f"delta_neutral_{int(time.time())}", base_config)
 self.strategy_manager.add_strategy(strategy, weight)

 elif strategy_name == "volatility_arbitrage":
 strategy = VolatilityArbitrageStrategy(f"vol_arb_{int(time.time())}", base_config)
 self.strategy_manager.add_strategy(strategy, weight)

 elif strategy_name == "statistical_arbitrage":
 strategy = StatisticalArbitrageStrategy(f"stat_arb_{int(time.time())}", base_config)
 self.strategy_manager.add_strategy(strategy, weight)

 elif strategy_name == "pin_risk":
 strategy = PinRiskManagementStrategy(f"pin_risk_{int(time.time())}", base_config)
 self.strategy_manager.add_strategy(strategy, weight)

 print(f"Initialized {len(self.strategy_manager.strategies)} strategies")

 def _initialize_quote_engine(self):
 """Initialize quote generation and management engine"""
 quote_config = {
 'quote_refresh_interval_ms': self.config.quote_refresh_frequency_ms,
 'max_processing_time_ms': 10,
 'worker_threads': 8,
 'queue_size_limit': 2000
 }

 self.quote_engine = QuoteEngine(quote_config)
 print("Quote engine initialized")

 def _initialize_hedging_system(self):
 """Initialize hedging and risk management system"""
 risk_limits = RiskLimits(
 max_portfolio_delta=1000.0,
 max_portfolio_gamma=500.0,
 max_portfolio_vega=2000.0,
 max_daily_loss=self.config.max_daily_loss,
 max_portfolio_value=self.config.max_portfolio_value
 )

 hedging_config = HedgingConfig(
 delta_hedge_threshold=50.0,
 gamma_scalp_threshold=10.0,
 continuous_hedge_interval_seconds=30
 )

 self.hedging_manager = HedgingRiskManager(risk_limits, hedging_config)
 print("Hedging and risk management system initialized")

 def _initialize_portfolio_optimizer(self):
 """Initialize portfolio optimization system"""
 optimization_objective = OptimizationObjective(
 maximize_sharpe=True,
 minimize_greeks_variance=True,
 risk_aversion=3.0
 )

 constraints = PortfolioConstraints(
 max_weight_per_position=self.config.max_position_concentration,
 delta_neutral_tolerance=100.0,
 max_portfolio_var=0.02
 )

 self.portfolio_optimizer = PortfolioOptimizer(optimization_objective, constraints)
 print("Portfolio optimizer initialized")

 def _initialize_execution_engine(self):
 """Initialize execution algorithm engine"""
 self.execution_engine = ExecutionEngine()
 print("Execution engine initialized")

 def _initialize_monitoring(self):
 """Initialize performance monitoring"""
 self.performance_monitors = {}
 self.risk_monitor = RiskMonitor()

 # Create performance monitors for each strategy
 for strategy_id in self.strategy_manager.strategies:
 self.performance_monitors[strategy_id] = StrategyPerformanceMonitor(strategy_id)

 print("Performance monitoring initialized")

 def _initialize_dashboard(self):
 """Initialize real-time monitoring dashboard"""
 if self.config.enable_real_time_monitoring:
 dashboard_config = DashboardConfig(
 port=self.config.dashboard_port,
 update_frequency_ms=1000
 )

 self.dashboard = MonitoringDashboard(dashboard_config)

 # Register strategy managers with dashboard
 for strategy_id, strategy in self.strategy_manager.strategies.items():
 self.dashboard.add_strategy_manager(strategy_id, self.strategy_manager)

 print(f"Dashboard initialized on port {self.config.dashboard_port}")
 else:
 self.dashboard = None

 def start_trading(self, market_data_feed: Optional[Callable] = None):
 """Start the market making system"""
 if self.is_running:
 print("System is already running!")
 return

 self.is_running = True

 # Start quote engine
 self.quote_engine.start()

 # Start main trading loop
 self.main_loop_thread = threading.Thread(target=self._main_trading_loop)
 self.main_loop_thread.daemon = True
 self.main_loop_thread.start()

 # Start dashboard if enabled
 if self.dashboard:
 self.monitoring_thread = threading.Thread(target=self._start_dashboard)
 self.monitoring_thread.daemon = True
 self.monitoring_thread.start()

 print("🚀 Options Market Making System Started!")
 print(f" - Strategies running: {len(self.strategy_manager.strategies)}")
 print(f" - Dashboard: {'Enabled' if self.dashboard else 'Disabled'}")
 print(f" - Risk limits: ${self.config.max_daily_loss:,.0f} daily loss limit")

 if self.dashboard:
 print(f" - Dashboard URL: http://localhost:{self.config.dashboard_port}")

 def stop_trading(self):
 """Stop the market making system"""
 if not self.is_running:
 print("System is not running!")
 return

 print("Stopping market making system...")

 self.is_running = False

 # Stop quote engine
 self.quote_engine.stop()

 # Stop dashboard
 if self.dashboard:
 self.dashboard.stop_monitoring()

 # Wait for threads to finish
 if self.main_loop_thread:
 self.main_loop_thread.join(timeout=5)

 if self.monitoring_thread:
 self.monitoring_thread.join(timeout=5)

 print("✅ Market making system stopped")

 def _main_trading_loop(self):
 """Main trading loop that orchestrates all components"""
 print("Main trading loop started")

 while self.is_running:
 try:
 # Update market data (would come from real feed)
 self._update_market_data()

 # Generate quotes from all strategies
 self._generate_and_submit_quotes()

 # Check for hedge requirements
 self._check_and_execute_hedges()

 # Update portfolio optimization
 self._optimize_portfolio()

 # Update performance monitoring
 self._update_performance_monitoring()

 # Sleep before next iteration
 time.sleep(0.1) # 100ms cycle

 except Exception as e:
 warnings.warn(f"Error in main trading loop: {e}")
 time.sleep(1) # Longer sleep on error

 def _update_market_data(self):
 """Update market data (mock implementation)"""
 # In practice, this would receive real market data
 # For demo, we'll generate some mock data

 current_time = datetime.now()

 # Mock option symbols
 symbols = [
 "AAPL_231215C150", "AAPL_231215P150",
 "AAPL_231215C155", "AAPL_231215P145",
 "SPY_231215C450", "SPY_231215P450"
 ]

 for symbol in symbols:
 # Generate realistic market data
 bid = 5.0 + np.random.normal(0, 0.1)
 ask = bid + 0.05 + np.random.exponential(0.05)
 last = (bid + ask) / 2 + np.random.normal(0, 0.02)

 market_data = MarketData(
 symbol=symbol,
 timestamp=current_time,
 bid=max(0.01, bid),
 ask=max(bid + 0.01, ask),
 last=max(0.01, last),
 bid_size=np.random.randint(10, 100),
 ask_size=np.random.randint(10, 100),
 volume=np.random.randint(100, 1000),
 open_interest=np.random.randint(1000, 10000),
 implied_vol=0.15 + np.random.normal(0, 0.02),
 underlying_price=150.0 + np.random.normal(0, 1.0)
 )

 self.current_market_data[symbol] = market_data

 def _generate_and_submit_quotes(self):
 """Generate quotes from strategies and submit them"""
 try:
 # Generate consolidated quotes from all strategies
 all_quotes = self.strategy_manager.generate_consolidated_quotes(self.current_market_data)

 # Submit quotes to quote engine
 for quote in all_quotes:
 # Create quote request
 request = QuoteRequest(
 symbol=quote.symbol,
 market_data=self.current_market_data.get(quote.symbol),
 greeks=None, # Would have real Greeks
 position=self.active_positions.get(quote.symbol),
 strategy_id=quote.strategy_id
 )

 # Submit to quote engine
 self.quote_engine.request_quotes(request, self._quote_callback)

 except Exception as e:
 warnings.warn(f"Error generating quotes: {e}")

 def _quote_callback(self, response):
 """Callback for quote responses"""
 if response.quotes:
 # Simulate some fills
 for quote in response.quotes[:1]: # Fill first quote sometimes
 if np.random.random() < 0.1: # 10% fill rate
 self._simulate_fill(quote)

 def _simulate_fill(self, quote):
 """Simulate a quote fill"""
 fill = Trade(
 symbol=quote.symbol,
 side=quote.side,
 quantity=quote.size,
 price=quote.price,
 timestamp=datetime.now(),
 strategy_id=quote.strategy_id,
 commission=0.5,
 slippage=0.01
 )

 self.trade_history.append(fill)

 # Update positions
 if quote.symbol not in self.active_positions:
 self.active_positions[quote.symbol] = Position(
 symbol=quote.symbol,
 quantity=0,
 avg_price=0.0,
 market_value=0.0,
 unrealized_pnl=0.0,
 realized_pnl=0.0
 )

 position = self.active_positions[quote.symbol]

 # Update position with fill
 if fill.side.value == 'bid':
 position.quantity += fill.quantity
 else:
 position.quantity -= fill.quantity

 # Update market value
 if quote.symbol in self.current_market_data:
 market_price = self.current_market_data[quote.symbol].last
 position.market_value = position.quantity * market_price

 def _check_and_execute_hedges(self):
 """Check if hedging is needed and execute hedge orders"""
 try:
 # Calculate current portfolio Greeks
 self.strategy_manager.update_portfolio_greeks(self.current_market_data)
 portfolio_greeks = self.strategy_manager.total_portfolio_greeks

 # Update hedging system
 result = self.hedging_manager.update_portfolio_state(
 portfolio_greeks,
 self.active_positions,
 self.current_market_data,
 self.trade_history[-10:] # Recent trades
 )

 # Execute hedge orders if generated
 for hedge_order in result.get('hedge_orders', []):
 self._execute_hedge_order(hedge_order)

 except Exception as e:
 warnings.warn(f"Error in hedging: {e}")

 def _execute_hedge_order(self, hedge_order):
 """Execute a hedge order"""
 # Create execution order
 execution_order = ExecutionOrder(
 symbol=hedge_order.symbol,
 side=hedge_order.side,
 quantity=hedge_order.quantity,
 order_type=hedge_order.order_type,
 algorithm=ExecutionAlgorithm.SMART_ORDER_ROUTING,
 urgency=hedge_order.urgency
 )

 # Mock execution
 hedge_trade = Trade(
 symbol=hedge_order.symbol,
 side=hedge_order.side,
 quantity=hedge_order.quantity,
 price=self.current_market_data.get(hedge_order.symbol, MarketData(hedge_order.symbol, datetime.now(), 100, 101, 100.5, 50, 50, 1000, 5000)).last,
 timestamp=datetime.now(),
 strategy_id="hedge",
 commission=0.25,
 slippage=0.005
 )

 self.trade_history.append(hedge_trade)

 def _optimize_portfolio(self):
 """Run portfolio optimization"""
 try:
 # Run portfolio optimization every 5 minutes
 if len(self.trade_history) % 3000 == 0: # Approximately every 5 minutes
 # Mock returns history
 returns_history = {symbol: [np.random.normal(0.0001, 0.02) for _ in range(30)]
 for symbol in self.active_positions.keys()}

 # Mock Greeks data
 greeks_data = {symbol: Greeks(np.random.normal(0.5, 0.1), np.random.normal(0.05, 0.01),
 np.random.normal(-0.1, 0.02), np.random.normal(0.3, 0.05),
 np.random.normal(0.01, 0.005), 150.0, datetime.now())
 for symbol in self.active_positions.keys()}

 if self.active_positions and returns_history:
 result = self.portfolio_optimizer.optimize_portfolio(
 self.active_positions,
 self.current_market_data,
 returns_history,
 greeks_data
 )

 if result.constraints_satisfied:
 print(f"Portfolio optimization completed. Expected improvement: {result.expected_improvement}")

 except Exception as e:
 warnings.warn(f"Error in portfolio optimization: {e}")

 def _update_performance_monitoring(self):
 """Update performance monitoring for all strategies"""
 try:
 for strategy_id, monitor in self.performance_monitors.items():
 # Mock Greeks for the strategy
 strategy_greeks = {symbol: Greeks(np.random.normal(0.5, 0.1), np.random.normal(0.05, 0.01),
 np.random.normal(-0.1, 0.02), np.random.normal(0.3, 0.05),
 np.random.normal(0.01, 0.005), 150.0, datetime.now())
 for symbol in self.active_positions.keys()}

 # Update performance
 snapshot = monitor.update_performance(
 self.active_positions,
 strategy_greeks,
 self.current_market_data,
 [t for t in self.trade_history if t.strategy_id == strategy_id]
 )

 except Exception as e:
 warnings.warn(f"Error updating performance monitoring: {e}")

 def _start_dashboard(self):
 """Start the monitoring dashboard"""
 try:
 self.dashboard.start_monitoring()
 except Exception as e:
 warnings.warn(f"Error starting dashboard: {e}")

 def get_system_status(self) -> Dict[str, Any]:
 """Get current system status"""
 total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.active_positions.values())

 portfolio_value = sum(abs(pos.market_value) for pos in self.active_positions.values())

 return {
 'is_running': self.is_running,
 'active_strategies': len(self.strategy_manager.strategies),
 'open_positions': len([p for p in self.active_positions.values() if p.quantity != 0]),
 'total_trades': len(self.trade_history),
 'total_pnl': total_pnl,
 'portfolio_value': portfolio_value,
 'portfolio_greeks': {
 'delta': self.strategy_manager.total_portfolio_greeks.delta,
 'gamma': self.strategy_manager.total_portfolio_greeks.gamma,
 'vega': self.strategy_manager.total_portfolio_greeks.vega,
 'theta': self.strategy_manager.total_portfolio_greeks.theta,
 },
 'last_update': datetime.now().isoformat()
 }

 def get_performance_summary(self) -> Dict[str, Any]:
 """Get performance summary across all strategies"""
 total_return = 0.0
 total_trades = len(self.trade_history)

 strategy_performance = {}
 for strategy_id, monitor in self.performance_monitors.items():
 perf = monitor.get_strategy_performance(7) # 7-day performance
 strategy_performance[strategy_id] = {
 'total_return': perf.total_return,
 'sharpe_ratio': perf.sharpe_ratio,
 'max_drawdown': perf.max_drawdown,
 'win_rate': perf.win_rate
 }
 total_return += perf.total_return

 return {
 'total_return': total_return,
 'total_trades': total_trades,
 'strategy_performance': strategy_performance,
 'system_uptime_hours': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds() / 3600
 }


# Factory functions for easy setup
def create_production_framework() -> OptionsMarketMakingFramework:
 """Create production-ready market making framework"""
 config = MarketMakingConfig(
 strategies_to_run=["delta_neutral", "volatility_arbitrage", "statistical_arbitrage"],
 max_portfolio_value=50000000.0, # $50M
 max_daily_loss=500000.0, # $500K
 quote_refresh_frequency_ms=50, # 50ms for high frequency
 enable_real_time_monitoring=True,
 dashboard_port=5000
 )

 return OptionsMarketMakingFramework(config)


def create_demo_framework() -> OptionsMarketMakingFramework:
 """Create demo framework with reduced limits"""
 config = MarketMakingConfig(
 strategies_to_run=["delta_neutral", "volatility_arbitrage"],
 max_portfolio_value=1000000.0, # $1M
 max_daily_loss=10000.0, # $10K
 quote_refresh_frequency_ms=200, # 200ms for demo
 enable_real_time_monitoring=True,
 dashboard_port=5000
 )

 return OptionsMarketMakingFramework(config)


def create_backtesting_framework() -> OptionsMarketMakingFramework:
 """Create framework optimized for backtesting"""
 config = MarketMakingConfig(
 strategies_to_run=["delta_neutral"],
 enable_real_time_monitoring=False, # Disable dashboard for backtesting
 quote_refresh_frequency_ms=1000 # Slower for backtesting
 )

 return OptionsMarketMakingFramework(config)


# Example usage and main execution
if __name__ == "__main__":
 import signal
 import sys

 def signal_handler(sig, frame):
 print('\nShutdown signal received...')
 if 'framework' in locals():
 framework.stop_trading()
 sys.exit(0)

 signal.signal(signal.SIGINT, signal_handler)

 print("🎯 Options Market Making Framework")
 print("=" * 50)

 # Create framework
 print("Creating market making framework...")
 framework = create_demo_framework()

 # Start trading
 print("Starting trading system...")
 framework.start_trading()

 try:
 # Keep running and display status
 while framework.is_running:
 time.sleep(10) # Status update every 10 seconds

 status = framework.get_system_status()
 print(f"\n📊 System Status:")
 print(f" Total P&L: ${status['total_pnl']:,.2f}")
 print(f" Open Positions: {status['open_positions']}")
 print(f" Total Trades: {status['total_trades']}")
 print(f" Portfolio Delta: {status['portfolio_greeks']['delta']:.2f}")
 print(f" Portfolio Value: ${status['portfolio_value']:,.2f}")

 # Performance summary every minute
 if status['total_trades'] % 6 == 0: # Approximately every minute
 perf = framework.get_performance_summary()
 print(f"\n📈 Performance Summary:")
 print(f" Total Return: {perf['total_return']:.2%}")
 print(f" System Uptime: {perf['system_uptime_hours']:.1f} hours")

 for strategy_id, strategy_perf in perf['strategy_performance'].items():
 print(f" {strategy_id}: {strategy_perf['total_return']:.2%} return, "
 f"{strategy_perf['sharpe_ratio']:.2f} Sharpe")

 except KeyboardInterrupt:
 print("\nShutdown requested by user")
 finally:
 print("Stopping framework...")
 framework.stop_trading()
 print("✅ Framework stopped successfully")