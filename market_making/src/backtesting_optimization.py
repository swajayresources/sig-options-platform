"""
Strategy Backtesting and Optimization Framework

This module implements comprehensive backtesting and optimization capabilities for
options market making strategies, including historical simulation, parameter optimization,
walk-forward analysis, and strategy comparison tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.optimize import minimize, differential_evolution
import json

from market_making_strategies import (
 MarketMakingStrategy, DeltaNeutralMarketMaker, VolatilityArbitrageStrategy,
 StatisticalArbitrageStrategy, StrategyConfig, MarketData, Position, Trade, Greeks
)
from performance_monitoring import StrategyPerformanceMonitor, PerformanceCalculator


@dataclass
class BacktestConfig:
 """Configuration for backtesting"""
 start_date: datetime
 end_date: datetime
 initial_capital: float = 1000000.0
 commission_per_contract: float = 0.50
 slippage_bps: int = 5 # 5 basis points

 # Market data settings
 data_frequency: str = "1min" # 1min, 5min, 1h, 1d
 include_after_hours: bool = False

 # Execution settings
 execution_delay_ms: int = 100 # Execution delay
 market_impact_model: str = "sqrt" # linear, sqrt, power

 # Risk settings
 max_leverage: float = 1.0
 position_limits: Dict[str, int] = field(default_factory=dict)

 # Rebalancing
 rebalance_frequency: str = "intraday" # intraday, daily, weekly


@dataclass
class BacktestResult:
 """Results from strategy backtesting"""
 strategy_id: str
 config: BacktestConfig

 # Performance metrics
 total_return: float
 annualized_return: float
 volatility: float
 sharpe_ratio: float
 sortino_ratio: float
 max_drawdown: float
 calmar_ratio: float

 # Trading metrics
 total_trades: int
 win_rate: float
 profit_factor: float
 avg_trade_pnl: float

 # Risk metrics
 var_95: float
 expected_shortfall: float
 beta: float

 # Time series data
 daily_returns: List[float]
 cumulative_returns: List[float]
 equity_curve: List[Tuple[datetime, float]]
 drawdown_series: List[Tuple[datetime, float]]

 # Trade history
 trades: List[Trade]
 positions_history: List[Dict[str, Position]]

 # Attribution
 pnl_attribution: Dict[str, float]

 # Execution statistics
 avg_slippage: float
 avg_commission: float
 execution_quality: float


@dataclass
class OptimizationResult:
 """Results from parameter optimization"""
 strategy_class: type
 optimal_parameters: Dict[str, Any]
 optimal_score: float
 optimization_metric: str

 # Optimization details
 iterations: int
 function_evaluations: int
 optimization_time: float

 # Parameter sensitivity
 parameter_sensitivity: Dict[str, float]
 parameter_bounds: Dict[str, Tuple[float, float]]

 # Out-of-sample validation
 in_sample_result: BacktestResult
 out_of_sample_result: Optional[BacktestResult]

 # Robustness metrics
 parameter_stability: float
 performance_consistency: float


class HistoricalDataGenerator:
 """Generates realistic historical market data for backtesting"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'base_vol': 0.20,
 'vol_of_vol': 0.1,
 'skew_param': -0.05,
 'term_structure_slope': 0.02,
 'mean_reversion_speed': 0.5,
 'jump_intensity': 0.1,
 'jump_size_mean': -0.05,
 'jump_size_std': 0.15
 }

 def generate_option_data(self, underlying_prices: pd.Series,
 strikes: List[float], expiries: List[float],
 start_date: datetime, end_date: datetime) -> pd.DataFrame:
 """Generate historical option data"""

 data_points = []

 # Create date range
 date_range = pd.date_range(start_date, end_date, freq='1H')

 for timestamp in date_range:
 underlying_price = self._interpolate_underlying_price(underlying_prices, timestamp)

 for expiry in expiries:
 time_to_expiry = (datetime.combine(timestamp.date(), datetime.min.time()) + timedelta(days=expiry*365) - timestamp).total_seconds() / (365.25 * 24 * 3600)

 if time_to_expiry <= 0:
 continue

 for strike in strikes:
 # Generate option data for this strike/expiry
 option_data = self._generate_option_quote(
 underlying_price, strike, time_to_expiry, timestamp
 )

 if option_data:
 data_points.append(option_data)

 return pd.DataFrame(data_points)

 def _interpolate_underlying_price(self, prices: pd.Series, timestamp: pd.Timestamp) -> float:
 """Interpolate underlying price for given timestamp"""
 # Simple interpolation - in practice would use more sophisticated methods
 try:
 if timestamp in prices.index:
 return prices[timestamp]

 # Find nearest prices
 before_prices = prices[prices.index <= timestamp]
 after_prices = prices[prices.index > timestamp]

 if len(before_prices) > 0 and len(after_prices) > 0:
 # Linear interpolation
 t1, p1 = before_prices.index[-1], before_prices.iloc[-1]
 t2, p2 = after_prices.index[0], after_prices.iloc[0]

 weight = (timestamp - t1).total_seconds() / (t2 - t1).total_seconds()
 return p1 + weight * (p2 - p1)
 elif len(before_prices) > 0:
 return before_prices.iloc[-1]
 elif len(after_prices) > 0:
 return after_prices.iloc[0]
 else:
 return 100.0 # Default price

 except Exception:
 return 100.0

 def _generate_option_quote(self, underlying_price: float, strike: float,
 time_to_expiry: float, timestamp: pd.Timestamp) -> Optional[Dict]:
 """Generate option quote for given parameters"""

 try:
 # Calculate moneyness
 moneyness = np.log(underlying_price / strike)

 # Generate implied volatility with smile
 base_vol = self.config['base_vol']
 skew = self.config['skew_param'] * moneyness
 vol_smile = base_vol * (1 + skew + 0.1 * moneyness**2)

 # Add term structure effect
 term_effect = self.config['term_structure_slope'] * np.sqrt(time_to_expiry)
 implied_vol = vol_smile + term_effect

 # Add noise
 vol_noise = np.random.normal(0, 0.01)
 implied_vol = max(0.05, implied_vol + vol_noise)

 # Calculate theoretical price using Black-Scholes
 call_price = self._black_scholes_call(underlying_price, strike, time_to_expiry, 0.02, implied_vol)
 put_price = self._black_scholes_put(underlying_price, strike, time_to_expiry, 0.02, implied_vol)

 # Generate bid-ask spreads
 call_spread = max(0.01, call_price * 0.02 + 0.005)
 put_spread = max(0.01, put_price * 0.02 + 0.005)

 # Create option symbols
 expiry_str = (timestamp + timedelta(days=time_to_expiry*365)).strftime('%y%m%d')
 call_symbol = f"{underlying_price:.0f}_{expiry_str}C{strike:.0f}"
 put_symbol = f"{underlying_price:.0f}_{expiry_str}P{strike:.0f}"

 # Generate volume and open interest
 volume = max(1, int(np.random.exponential(50)))
 open_interest = max(volume, int(np.random.exponential(500)))

 return {
 'timestamp': timestamp,
 'call_symbol': call_symbol,
 'put_symbol': put_symbol,
 'underlying_price': underlying_price,
 'strike': strike,
 'expiry': time_to_expiry,
 'call_bid': call_price - call_spread/2,
 'call_ask': call_price + call_spread/2,
 'call_last': call_price,
 'put_bid': put_price - put_spread/2,
 'put_ask': put_price + put_spread/2,
 'put_last': put_price,
 'implied_vol': implied_vol,
 'volume': volume,
 'open_interest': open_interest
 }

 except Exception as e:
 warnings.warn(f"Failed to generate option quote: {e}")
 return None

 def _black_scholes_call(self, S: float, K: float, T: float, r: float, vol: float) -> float:
 """Black-Scholes call option price"""
 if T <= 0:
 return max(0, S - K)

 d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
 d2 = d1 - vol*np.sqrt(T)

 from scipy.stats import norm
 call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
 return max(0, call_price)

 def _black_scholes_put(self, S: float, K: float, T: float, r: float, vol: float) -> float:
 """Black-Scholes put option price"""
 if T <= 0:
 return max(0, K - S)

 d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
 d2 = d1 - vol*np.sqrt(T)

 from scipy.stats import norm
 put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
 return max(0, put_price)


class BacktestEngine:
 """Core backtesting engine"""

 def __init__(self, config: BacktestConfig):
 self.config = config
 self.data_generator = HistoricalDataGenerator()
 self.performance_calculator = PerformanceCalculator()

 def backtest_strategy(self, strategy: MarketMakingStrategy,
 historical_data: pd.DataFrame) -> BacktestResult:
 """Run backtest for a strategy"""

 # Initialize tracking variables
 portfolio_value = self.config.initial_capital
 positions: Dict[str, Position] = {}
 trades: List[Trade] = []
 equity_curve: List[Tuple[datetime, float]] = []
 positions_history: List[Dict[str, Position]] = []

 # Performance tracking
 daily_returns: List[float] = []
 last_portfolio_value = portfolio_value

 # Process historical data chronologically
 data_by_timestamp = historical_data.groupby('timestamp')

 for timestamp, group_data in data_by_timestamp:
 # Convert group data to market data format
 market_data = self._convert_to_market_data(group_data)

 # Update strategy with current market data
 quotes = strategy.generate_quotes(market_data)

 # Simulate quote fills
 new_trades = self._simulate_quote_fills(quotes, market_data, timestamp)
 trades.extend(new_trades)

 # Update positions
 for trade in new_trades:
 strategy.update_position(trade)
 self._update_position_tracking(positions, trade)

 # Calculate current portfolio value
 current_value = self._calculate_portfolio_value(positions, market_data)
 portfolio_value = current_value

 # Record equity curve
 equity_curve.append((timestamp, portfolio_value))

 # Calculate daily returns
 if timestamp.hour == 16 and timestamp.minute == 0: # Market close
 daily_return = (portfolio_value - last_portfolio_value) / last_portfolio_value
 daily_returns.append(daily_return)
 last_portfolio_value = portfolio_value

 # Store positions snapshot
 positions_history.append(positions.copy())

 # Check for hedging requirements
 portfolio_greeks = strategy.calculate_portfolio_greeks(market_data)
 if strategy.should_hedge(portfolio_greeks):
 hedge_orders = strategy.calculate_hedge_orders(portfolio_greeks)
 hedge_trades = self._execute_hedge_orders(hedge_orders, market_data, timestamp)
 trades.extend(hedge_trades)

 # Calculate performance metrics
 total_return = (portfolio_value - self.config.initial_capital) / self.config.initial_capital
 annualized_return = total_return * (365 / (self.config.end_date - self.config.start_date).days)

 cumulative_returns = np.cumprod(1 + np.array(daily_returns)).tolist()

 # Risk metrics
 volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
 sharpe_ratio = self.performance_calculator.calculate_sharpe_ratio(daily_returns)
 sortino_ratio = self.performance_calculator.calculate_sortino_ratio(daily_returns)
 max_drawdown, _, _ = self.performance_calculator.calculate_max_drawdown(cumulative_returns)
 calmar_ratio = self.performance_calculator.calculate_calmar_ratio(daily_returns)

 var_95 = self.performance_calculator.calculate_var(daily_returns)
 expected_shortfall = self.performance_calculator.calculate_expected_shortfall(daily_returns)

 # Trading metrics
 total_trades = len(trades)
 winning_trades = [t for t in trades if self._calculate_trade_pnl(t, positions) > 0]
 win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

 profitable_pnl = sum(self._calculate_trade_pnl(t, positions) for t in winning_trades)
 losing_pnl = sum(abs(self._calculate_trade_pnl(t, positions)) for t in trades if self._calculate_trade_pnl(t, positions) < 0)
 profit_factor = profitable_pnl / losing_pnl if losing_pnl > 0 else float('inf')

 avg_trade_pnl = sum(self._calculate_trade_pnl(t, positions) for t in trades) / total_trades if total_trades > 0 else 0

 # Calculate drawdown series
 running_max = np.maximum.accumulate([v for _, v in equity_curve])
 drawdown_series = [(ts, (val - max_val) / max_val) for (ts, val), max_val in zip(equity_curve, running_max)]

 # P&L Attribution (simplified)
 pnl_attribution = self._calculate_pnl_attribution(trades, positions)

 # Execution statistics
 avg_slippage = np.mean([t.slippage for t in trades]) if trades else 0
 avg_commission = np.mean([t.commission for t in trades]) if trades else 0
 execution_quality = 0.95 # Simplified metric

 return BacktestResult(
 strategy_id=strategy.strategy_id,
 config=self.config,
 total_return=total_return,
 annualized_return=annualized_return,
 volatility=volatility,
 sharpe_ratio=sharpe_ratio,
 sortino_ratio=sortino_ratio,
 max_drawdown=max_drawdown,
 calmar_ratio=calmar_ratio,
 total_trades=total_trades,
 win_rate=win_rate,
 profit_factor=profit_factor,
 avg_trade_pnl=avg_trade_pnl,
 var_95=var_95,
 expected_shortfall=expected_shortfall,
 beta=0.0, # Would calculate vs market benchmark
 daily_returns=daily_returns,
 cumulative_returns=cumulative_returns,
 equity_curve=equity_curve,
 drawdown_series=drawdown_series,
 trades=trades,
 positions_history=positions_history,
 pnl_attribution=pnl_attribution,
 avg_slippage=avg_slippage,
 avg_commission=avg_commission,
 execution_quality=execution_quality
 )

 def _convert_to_market_data(self, group_data: pd.DataFrame) -> Dict[str, MarketData]:
 """Convert historical data to MarketData format"""
 market_data = {}

 for _, row in group_data.iterrows():
 # Call option
 if 'call_symbol' in row and pd.notna(row['call_symbol']):
 call_data = MarketData(
 symbol=row['call_symbol'],
 timestamp=row['timestamp'],
 bid=row['call_bid'],
 ask=row['call_ask'],
 last=row['call_last'],
 bid_size=50, # Default size
 ask_size=50,
 volume=row['volume'],
 open_interest=row['open_interest'],
 implied_vol=row['implied_vol'],
 underlying_price=row['underlying_price']
 )
 market_data[row['call_symbol']] = call_data

 # Put option
 if 'put_symbol' in row and pd.notna(row['put_symbol']):
 put_data = MarketData(
 symbol=row['put_symbol'],
 timestamp=row['timestamp'],
 bid=row['put_bid'],
 ask=row['put_ask'],
 last=row['put_last'],
 bid_size=50,
 ask_size=50,
 volume=row['volume'],
 open_interest=row['open_interest'],
 implied_vol=row['implied_vol'],
 underlying_price=row['underlying_price']
 )
 market_data[row['put_symbol']] = put_data

 return market_data

 def _simulate_quote_fills(self, quotes: List, market_data: Dict[str, MarketData],
 timestamp: datetime) -> List[Trade]:
 """Simulate quote fills based on market conditions"""
 fills = []

 for quote in quotes:
 if quote.symbol not in market_data:
 continue

 data = market_data[quote.symbol]

 # Simple fill simulation based on price competitiveness
 if quote.side.value == 'bid':
 # We're bidding - check if our bid is competitive
 if quote.price >= data.bid * 0.99: # Within 1% of market bid
 fill_probability = 0.3 + min(0.4, (quote.price - data.bid) / (data.ask - data.bid))
 else:
 fill_probability = 0.1
 else:
 # We're offering - check if our ask is competitive
 if quote.price <= data.ask * 1.01: # Within 1% of market ask
 fill_probability = 0.3 + min(0.4, (data.ask - quote.price) / (data.ask - data.bid))
 else:
 fill_probability = 0.1

 # Simulate fill
 if np.random.random() < fill_probability:
 # Calculate slippage
 if quote.side.value == 'bid':
 market_price = data.ask
 else:
 market_price = data.bid

 execution_price = quote.price
 slippage = abs(execution_price - market_price)

 fill = Trade(
 symbol=quote.symbol,
 side=quote.side,
 quantity=quote.size,
 price=execution_price,
 timestamp=timestamp,
 strategy_id=quote.strategy_id,
 commission=self.config.commission_per_contract,
 slippage=slippage
 )
 fills.append(fill)

 return fills

 def _update_position_tracking(self, positions: Dict[str, Position], trade: Trade):
 """Update position tracking with new trade"""
 symbol = trade.symbol

 if symbol not in positions:
 positions[symbol] = Position(
 symbol=symbol,
 quantity=0,
 avg_price=0.0,
 market_value=0.0,
 unrealized_pnl=0.0,
 realized_pnl=0.0
 )

 position = positions[symbol]

 # Update position
 if trade.side.value == 'bid': # We bought
 new_quantity = position.quantity + trade.quantity
 else: # We sold
 new_quantity = position.quantity - trade.quantity

 # Update average price
 if new_quantity != 0:
 total_cost = position.quantity * position.avg_price
 trade_cost = trade.quantity * trade.price * (1 if trade.side.value == 'bid' else -1)
 new_avg_price = (total_cost + trade_cost) / new_quantity
 position.avg_price = new_avg_price

 position.quantity = new_quantity

 def _calculate_portfolio_value(self, positions: Dict[str, Position],
 market_data: Dict[str, MarketData]) -> float:
 """Calculate current portfolio value"""
 total_value = self.config.initial_capital

 for symbol, position in positions.items():
 if position.quantity == 0:
 continue

 if symbol in market_data:
 data = market_data[symbol]
 market_price = (data.bid + data.ask) / 2 if data.bid > 0 and data.ask > 0 else data.last
 position_value = position.quantity * market_price
 total_value += position_value - position.quantity * position.avg_price

 return total_value

 def _execute_hedge_orders(self, hedge_orders: List, market_data: Dict[str, MarketData],
 timestamp: datetime) -> List[Trade]:
 """Execute hedge orders"""
 hedge_trades = []

 for order in hedge_orders:
 # Simplified hedge execution - in practice would be more sophisticated
 symbol = order[0] # Extract symbol from order tuple
 side = order[1]
 quantity = order[2]

 # Assume hedge executes at market price
 if symbol in market_data:
 data = market_data[symbol]
 if side.value == 'bid':
 price = data.ask
 else:
 price = data.bid

 hedge_trade = Trade(
 symbol=symbol,
 side=side,
 quantity=quantity,
 price=price,
 timestamp=timestamp,
 strategy_id="hedge",
 commission=self.config.commission_per_contract * 0.5, # Lower hedge commission
 slippage=0.01
 )
 hedge_trades.append(hedge_trade)

 return hedge_trades

 def _calculate_trade_pnl(self, trade: Trade, positions: Dict[str, Position]) -> float:
 """Calculate P&L for a trade"""
 # Simplified P&L calculation
 if trade.side.value == 'bid':
 return -trade.quantity * trade.price # Cost of buying
 else:
 return trade.quantity * trade.price # Revenue from selling

 def _calculate_pnl_attribution(self, trades: List[Trade],
 positions: Dict[str, Position]) -> Dict[str, float]:
 """Calculate P&L attribution"""
 attribution = {
 'trading_pnl': sum(self._calculate_trade_pnl(t, positions) for t in trades),
 'commission_cost': sum(t.commission for t in trades),
 'slippage_cost': sum(t.slippage for t in trades),
 }

 return attribution


class ParameterOptimizer:
 """Optimizes strategy parameters using various algorithms"""

 def __init__(self, backtest_engine: BacktestEngine):
 self.backtest_engine = backtest_engine
 self.optimization_history: List[OptimizationResult] = []

 def optimize_parameters(self,
 strategy_class: type,
 parameter_bounds: Dict[str, Tuple[float, float]],
 historical_data: pd.DataFrame,
 optimization_metric: str = 'sharpe_ratio',
 method: str = 'differential_evolution',
 validation_split: float = 0.3) -> OptimizationResult:
 """Optimize strategy parameters"""

 start_time = time.time()

 # Split data for in-sample and out-of-sample testing
 split_point = int(len(historical_data) * (1 - validation_split))
 in_sample_data = historical_data.iloc[:split_point]
 out_of_sample_data = historical_data.iloc[split_point:]

 # Define objective function
 def objective_function(params_array):
 try:
 # Convert parameter array to dictionary
 params_dict = {param_name: params_array[i]
 for i, param_name in enumerate(parameter_bounds.keys())}

 # Create strategy with parameters
 strategy = self._create_strategy_with_params(strategy_class, params_dict)

 # Run backtest
 result = self.backtest_engine.backtest_strategy(strategy, in_sample_data)

 # Return negative value for maximization problems
 score = getattr(result, optimization_metric)
 return -score if optimization_metric in ['sharpe_ratio', 'calmar_ratio', 'total_return'] else score

 except Exception as e:
 warnings.warn(f"Parameter evaluation failed: {e}")
 return 1e6 # Large penalty for failed evaluations

 # Set up optimization bounds
 bounds = list(parameter_bounds.values())

 # Run optimization
 if method == 'differential_evolution':
 opt_result = differential_evolution(
 objective_function,
 bounds,
 maxiter=100,
 popsize=15,
 seed=42
 )
 elif method == 'minimize':
 # Use initial guess as midpoint of bounds
 x0 = [(b[0] + b[1]) / 2 for b in bounds]
 opt_result = minimize(
 objective_function,
 x0,
 bounds=bounds,
 method='L-BFGS-B'
 )
 else:
 raise ValueError(f"Unknown optimization method: {method}")

 # Extract optimal parameters
 optimal_params = {param_name: opt_result.x[i]
 for i, param_name in enumerate(parameter_bounds.keys())}

 # Create optimal strategy and run full backtest
 optimal_strategy = self._create_strategy_with_params(strategy_class, optimal_params)
 in_sample_result = self.backtest_engine.backtest_strategy(optimal_strategy, in_sample_data)

 # Out-of-sample validation
 out_of_sample_result = None
 if len(out_of_sample_data) > 0:
 oos_strategy = self._create_strategy_with_params(strategy_class, optimal_params)
 out_of_sample_result = self.backtest_engine.backtest_strategy(oos_strategy, out_of_sample_data)

 # Calculate parameter sensitivity
 parameter_sensitivity = self._calculate_parameter_sensitivity(
 strategy_class, optimal_params, parameter_bounds, in_sample_data, optimization_metric
 )

 # Performance consistency metrics
 performance_consistency = self._calculate_performance_consistency(in_sample_result, out_of_sample_result)
 parameter_stability = self._calculate_parameter_stability(optimal_params, parameter_bounds)

 optimization_time = time.time() - start_time

 result = OptimizationResult(
 strategy_class=strategy_class,
 optimal_parameters=optimal_params,
 optimal_score=-opt_result.fun if opt_result.fun < 0 else opt_result.fun,
 optimization_metric=optimization_metric,
 iterations=getattr(opt_result, 'nit', 0),
 function_evaluations=getattr(opt_result, 'nfev', 0),
 optimization_time=optimization_time,
 parameter_sensitivity=parameter_sensitivity,
 parameter_bounds=parameter_bounds,
 in_sample_result=in_sample_result,
 out_of_sample_result=out_of_sample_result,
 parameter_stability=parameter_stability,
 performance_consistency=performance_consistency
 )

 self.optimization_history.append(result)
 return result

 def _create_strategy_with_params(self, strategy_class: type, params: Dict[str, Any]) -> MarketMakingStrategy:
 """Create strategy instance with given parameters"""

 # Create base config
 config = StrategyConfig()

 # Update config with optimized parameters
 for param_name, param_value in params.items():
 if hasattr(config, param_name):
 setattr(config, param_name, param_value)

 # Create strategy instance
 strategy_id = f"{strategy_class.__name__}_optimized"

 if strategy_class == DeltaNeutralMarketMaker:
 strategy = DeltaNeutralMarketMaker(strategy_id, config)
 elif strategy_class == VolatilityArbitrageStrategy:
 strategy = VolatilityArbitrageStrategy(strategy_id, config)
 elif strategy_class == StatisticalArbitrageStrategy:
 strategy = StatisticalArbitrageStrategy(strategy_id, config)
 else:
 raise ValueError(f"Unknown strategy class: {strategy_class}")

 # Set additional parameters specific to strategy
 for param_name, param_value in params.items():
 if hasattr(strategy, param_name):
 setattr(strategy, param_name, param_value)

 return strategy

 def _calculate_parameter_sensitivity(self,
 strategy_class: type,
 optimal_params: Dict[str, Any],
 parameter_bounds: Dict[str, Tuple[float, float]],
 data: pd.DataFrame,
 metric: str) -> Dict[str, float]:
 """Calculate parameter sensitivity"""

 sensitivity = {}
 base_strategy = self._create_strategy_with_params(strategy_class, optimal_params)
 base_result = self.backtest_engine.backtest_strategy(base_strategy, data)
 base_score = getattr(base_result, metric)

 for param_name, (min_val, max_val) in parameter_bounds.items():
 # Test parameter perturbation
 perturbation = (max_val - min_val) * 0.1 # 10% perturbation

 # Test positive perturbation
 perturbed_params = optimal_params.copy()
 perturbed_params[param_name] = min(max_val, optimal_params[param_name] + perturbation)

 try:
 perturbed_strategy = self._create_strategy_with_params(strategy_class, perturbed_params)
 perturbed_result = self.backtest_engine.backtest_strategy(perturbed_strategy, data)
 perturbed_score = getattr(perturbed_result, metric)

 sensitivity[param_name] = abs(perturbed_score - base_score) / abs(base_score) if base_score != 0 else 0
 except:
 sensitivity[param_name] = 0.0

 return sensitivity

 def _calculate_performance_consistency(self,
 in_sample: BacktestResult,
 out_of_sample: Optional[BacktestResult]) -> float:
 """Calculate performance consistency between in-sample and out-of-sample"""

 if out_of_sample is None:
 return 0.0

 # Compare key metrics
 metrics_comparison = {
 'sharpe_ratio': (in_sample.sharpe_ratio, out_of_sample.sharpe_ratio),
 'max_drawdown': (in_sample.max_drawdown, out_of_sample.max_drawdown),
 'win_rate': (in_sample.win_rate, out_of_sample.win_rate)
 }

 consistency_scores = []
 for metric, (is_val, oos_val) in metrics_comparison.items():
 if is_val != 0:
 consistency = 1 - abs(oos_val - is_val) / abs(is_val)
 consistency_scores.append(max(0, consistency))

 return np.mean(consistency_scores) if consistency_scores else 0.0

 def _calculate_parameter_stability(self,
 optimal_params: Dict[str, Any],
 parameter_bounds: Dict[str, Tuple[float, float]]) -> float:
 """Calculate parameter stability (how close to bounds)"""

 stability_scores = []

 for param_name, param_value in optimal_params.items():
 min_val, max_val = parameter_bounds[param_name]

 # Calculate distance from bounds (normalized)
 dist_from_min = (param_value - min_val) / (max_val - min_val)
 dist_from_max = (max_val - param_value) / (max_val - min_val)

 # Stability is higher when parameter is not at bounds
 stability = min(dist_from_min, dist_from_max) * 2 # Scale to 0-1
 stability_scores.append(min(1.0, stability))

 return np.mean(stability_scores)


class WalkForwardAnalyzer:
 """Performs walk-forward analysis for strategy validation"""

 def __init__(self, backtest_engine: BacktestEngine, optimizer: ParameterOptimizer):
 self.backtest_engine = backtest_engine
 self.optimizer = optimizer

 def walk_forward_analysis(self,
 strategy_class: type,
 parameter_bounds: Dict[str, Tuple[float, float]],
 historical_data: pd.DataFrame,
 window_months: int = 6,
 step_months: int = 1,
 reoptimize_frequency: int = 3) -> Dict[str, Any]:
 """Perform walk-forward analysis"""

 results = []
 data_length = len(historical_data)

 # Convert months to approximate rows (assuming daily data)
 window_size = window_months * 21 # Approximate trading days per month
 step_size = step_months * 21

 current_params = None
 months_since_optimization = 0

 for start_idx in range(0, data_length - window_size, step_size):
 end_idx = start_idx + window_size

 # Get training and testing windows
 train_data = historical_data.iloc[start_idx:end_idx]
 test_start = end_idx
 test_end = min(test_start + step_size, data_length)
 test_data = historical_data.iloc[test_start:test_end]

 if len(test_data) == 0:
 break

 # Reoptimize parameters if needed
 if current_params is None or months_since_optimization >= reoptimize_frequency:
 print(f"Optimizing parameters for period {start_idx}-{end_idx}")

 # Split training data for optimization
 opt_result = self.optimizer.optimize_parameters(
 strategy_class, parameter_bounds, train_data,
 validation_split=0.2
 )
 current_params = opt_result.optimal_parameters
 months_since_optimization = 0

 # Test with current parameters
 test_strategy = self.optimizer._create_strategy_with_params(strategy_class, current_params)
 test_result = self.backtest_engine.backtest_strategy(test_strategy, test_data)

 period_result = {
 'period_start': test_data.iloc[0]['timestamp'],
 'period_end': test_data.iloc[-1]['timestamp'],
 'parameters': current_params.copy(),
 'result': test_result,
 'months_since_optimization': months_since_optimization
 }

 results.append(period_result)
 months_since_optimization += step_months

 # Aggregate results
 aggregate_metrics = self._aggregate_walkforward_results(results)

 return {
 'period_results': results,
 'aggregate_metrics': aggregate_metrics,
 'parameter_stability': self._analyze_parameter_stability(results),
 'performance_consistency': self._analyze_performance_consistency(results)
 }

 def _aggregate_walkforward_results(self, results: List[Dict]) -> Dict[str, float]:
 """Aggregate walk-forward results"""

 if not results:
 return {}

 all_returns = []
 total_trades = 0
 total_wins = 0

 for period in results:
 result = period['result']
 all_returns.extend(result.daily_returns)
 total_trades += result.total_trades
 total_wins += int(result.total_trades * result.win_rate)

 if not all_returns:
 return {}

 calculator = PerformanceCalculator()

 aggregate_metrics = {
 'total_return': np.prod(1 + np.array(all_returns)) - 1,
 'annualized_return': np.mean(all_returns) * 252,
 'volatility': np.std(all_returns) * np.sqrt(252),
 'sharpe_ratio': calculator.calculate_sharpe_ratio(all_returns),
 'max_drawdown': calculator.calculate_max_drawdown(np.cumprod(1 + np.array(all_returns)).tolist())[0],
 'win_rate': total_wins / total_trades if total_trades > 0 else 0,
 'total_periods': len(results)
 }

 return aggregate_metrics

 def _analyze_parameter_stability(self, results: List[Dict]) -> Dict[str, float]:
 """Analyze parameter stability across periods"""

 if len(results) < 2:
 return {}

 param_names = list(results[0]['parameters'].keys())
 stability_metrics = {}

 for param_name in param_names:
 param_values = [period['parameters'][param_name] for period in results]

 # Calculate coefficient of variation
 mean_val = np.mean(param_values)
 std_val = np.std(param_values)
 cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')

 stability_metrics[param_name] = 1 / (1 + cv) # Higher is more stable

 return stability_metrics

 def _analyze_performance_consistency(self, results: List[Dict]) -> Dict[str, float]:
 """Analyze performance consistency across periods"""

 if not results:
 return {}

 metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate']
 consistency = {}

 for metric in metrics:
 values = [getattr(period['result'], metric) for period in results]

 # Calculate consistency as inverse of coefficient of variation
 mean_val = np.mean(values)
 std_val = np.std(values)
 cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')

 consistency[metric] = 1 / (1 + cv)

 return consistency


# Factory functions and examples
def create_backtest_config(start_date: str, end_date: str, **kwargs) -> BacktestConfig:
 """Create backtest configuration"""
 return BacktestConfig(
 start_date=datetime.strptime(start_date, '%Y-%m-%d'),
 end_date=datetime.strptime(end_date, '%Y-%m-%d'),
 **kwargs
 )


def run_strategy_comparison(strategies: List[type],
 historical_data: pd.DataFrame,
 config: BacktestConfig) -> pd.DataFrame:
 """Compare multiple strategies"""

 engine = BacktestEngine(config)
 results = []

 for strategy_class in strategies:
 # Create default strategy
 strategy = strategy_class(f"{strategy_class.__name__}_default", StrategyConfig())

 # Run backtest
 result = engine.backtest_strategy(strategy, historical_data)

 results.append({
 'Strategy': strategy_class.__name__,
 'Total Return': f"{result.total_return:.2%}",
 'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
 'Max Drawdown': f"{result.max_drawdown:.2%}",
 'Win Rate': f"{result.win_rate:.2%}",
 'Total Trades': result.total_trades
 })

 return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
 # Generate sample historical data
 data_generator = HistoricalDataGenerator()

 # Create sample underlying price series
 dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
 prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
 underlying_prices = pd.Series(prices, index=dates)

 # Generate option data
 strikes = [90, 95, 100, 105, 110]
 expiries = [0.25, 0.5, 1.0] # 3 months, 6 months, 1 year

 historical_data = data_generator.generate_option_data(
 underlying_prices, strikes, expiries,
 datetime(2023, 1, 1), datetime(2023, 6, 30)
 )

 print(f"Generated {len(historical_data)} data points")

 # Create backtest configuration
 config = create_backtest_config('2023-01-01', '2023-06-30', initial_capital=1000000)

 # Run simple backtest
 engine = BacktestEngine(config)
 strategy = DeltaNeutralMarketMaker("test_strategy", StrategyConfig())

 result = engine.backtest_strategy(strategy, historical_data)

 print(f"\nBacktest Results:")
 print(f"Total Return: {result.total_return:.2%}")
 print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
 print(f"Max Drawdown: {result.max_drawdown:.2%}")
 print(f"Total Trades: {result.total_trades}")

 # Parameter optimization example
 optimizer = ParameterOptimizer(engine)

 parameter_bounds = {
 'max_position_size': (50, 200),
 'min_spread_width': (0.005, 0.05),
 'max_spread_width': (0.1, 0.5)
 }

 print(f"\nRunning parameter optimization...")
 opt_result = optimizer.optimize_parameters(
 DeltaNeutralMarketMaker,
 parameter_bounds,
 historical_data,
 optimization_metric='sharpe_ratio'
 )

 print(f"Optimal parameters: {opt_result.optimal_parameters}")
 print(f"Optimal Sharpe ratio: {opt_result.optimal_score:.2f}")
 print(f"Optimization time: {opt_result.optimization_time:.1f}s")