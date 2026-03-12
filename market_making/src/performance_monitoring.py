"""
Performance Monitoring and Attribution System

This module implements comprehensive performance monitoring, attribution analysis,
and real-time strategy performance tracking for options market making operations.
Includes P&L attribution, risk-adjusted returns, strategy comparison, and detailed
performance analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import threading
import warnings
from enum import Enum
from collections import defaultdict, deque
import json
from concurrent.futures import ThreadPoolExecutor

from market_making_strategies import Greeks, Position, Trade, MarketData, StrategyManager
from hedging_risk_management import PnLAttribution


class PerformanceMetric(Enum):
 """Types of performance metrics"""
 TOTAL_PNL = "total_pnl"
 SHARPE_RATIO = "sharpe_ratio"
 SORTINO_RATIO = "sortino_ratio"
 MAX_DRAWDOWN = "max_drawdown"
 CALMAR_RATIO = "calmar_ratio"
 VAR_95 = "var_95"
 EXPECTED_SHORTFALL = "expected_shortfall"
 HIT_RATIO = "hit_ratio"
 PROFIT_FACTOR = "profit_factor"
 AVERAGE_WIN = "average_win"
 AVERAGE_LOSS = "average_loss"


class AttributionCategory(Enum):
 """P&L attribution categories"""
 DELTA_PNL = "delta_pnl"
 GAMMA_PNL = "gamma_pnl"
 THETA_PNL = "theta_pnl"
 VEGA_PNL = "vega_pnl"
 RHO_PNL = "rho_pnl"
 TRADING_PNL = "trading_pnl"
 CARRY_PNL = "carry_pnl"
 HEDGE_PNL = "hedge_pnl"
 COMMISSION_COST = "commission_cost"
 SLIPPAGE_COST = "slippage_cost"


@dataclass
class PerformanceSnapshot:
 """Performance snapshot at a point in time"""
 timestamp: datetime
 strategy_id: str
 total_pnl: float
 daily_pnl: float
 positions_count: int
 total_exposure: float

 # Risk metrics
 portfolio_delta: float
 portfolio_gamma: float
 portfolio_vega: float
 portfolio_theta: float
 var_95: float

 # Performance metrics
 sharpe_ratio: float
 max_drawdown: float
 hit_ratio: float

 # Attribution
 pnl_attribution: Dict[str, float]

 # Trading metrics
 trades_count: int
 avg_trade_pnl: float
 win_rate: float
 avg_win: float
 avg_loss: float


@dataclass
class StrategyPerformance:
 """Comprehensive strategy performance metrics"""
 strategy_id: str
 start_date: datetime
 end_date: datetime

 # Return metrics
 total_return: float
 annualized_return: float
 daily_returns: List[float]

 # Risk metrics
 volatility: float
 max_drawdown: float
 var_95: float
 expected_shortfall: float

 # Risk-adjusted metrics
 sharpe_ratio: float
 sortino_ratio: float
 calmar_ratio: float
 information_ratio: float

 # Trading metrics
 total_trades: int
 win_rate: float
 profit_factor: float
 avg_win: float
 avg_loss: float

 # Greeks performance
 delta_pnl_contribution: float
 gamma_pnl_contribution: float
 theta_pnl_contribution: float
 vega_pnl_contribution: float

 # Costs
 total_commissions: float
 total_slippage: float

 # Additional metrics
 correlation_to_market: float
 beta_to_market: float
 tracking_error: float


@dataclass
class RiskMetrics:
 """Real-time risk metrics"""
 timestamp: datetime
 portfolio_value: float

 # VaR metrics
 var_95_1d: float
 var_99_1d: float
 var_95_10d: float
 expected_shortfall_95: float

 # Greeks risk
 delta_risk: float
 gamma_risk: float
 vega_risk: float
 theta_risk: float

 # Concentration metrics
 concentration_herfindahl: float
 max_position_weight: float
 sector_concentration: Dict[str, float]

 # Liquidity metrics
 liquidity_score: float
 days_to_liquidate: float

 # Stress test results
 stress_test_results: Dict[str, float]


class PerformanceCalculator:
 """Calculates various performance and risk metrics"""

 def __init__(self):
 self.risk_free_rate = 0.02 # 2% annual risk-free rate

 def calculate_sharpe_ratio(self, returns: List[float], annualize: bool = True) -> float:
 """Calculate Sharpe ratio"""
 if not returns or len(returns) < 2:
 return 0.0

 returns_array = np.array(returns)
 excess_returns = returns_array - self.risk_free_rate / 252 # Daily risk-free rate

 mean_excess = np.mean(excess_returns)
 std_excess = np.std(excess_returns, ddof=1)

 if std_excess == 0:
 return 0.0

 sharpe = mean_excess / std_excess

 if annualize:
 sharpe *= np.sqrt(252) # Annualize

 return sharpe

 def calculate_sortino_ratio(self, returns: List[float], annualize: bool = True) -> float:
 """Calculate Sortino ratio (using downside deviation)"""
 if not returns or len(returns) < 2:
 return 0.0

 returns_array = np.array(returns)
 excess_returns = returns_array - self.risk_free_rate / 252

 mean_excess = np.mean(excess_returns)
 downside_returns = excess_returns[excess_returns < 0]

 if len(downside_returns) == 0:
 return float('inf') if mean_excess > 0 else 0.0

 downside_deviation = np.std(downside_returns, ddof=1)

 if downside_deviation == 0:
 return float('inf') if mean_excess > 0 else 0.0

 sortino = mean_excess / downside_deviation

 if annualize:
 sortino *= np.sqrt(252)

 return sortino

 def calculate_max_drawdown(self, cumulative_returns: List[float]) -> Tuple[float, int, int]:
 """Calculate maximum drawdown and its duration"""
 if not cumulative_returns:
 return 0.0, 0, 0

 cumulative = np.array(cumulative_returns)
 running_max = np.maximum.accumulate(cumulative)
 drawdown = (cumulative - running_max) / running_max

 max_dd = np.min(drawdown)
 max_dd_idx = np.argmin(drawdown)

 # Find start of drawdown period
 start_idx = max_dd_idx
 while start_idx > 0 and drawdown[start_idx - 1] == 0:
 start_idx -= 1

 # Find end of drawdown period
 end_idx = max_dd_idx
 while end_idx < len(drawdown) - 1 and drawdown[end_idx] < 0:
 end_idx += 1

 duration = end_idx - start_idx

 return abs(max_dd), start_idx, duration

 def calculate_var(self, returns: List[float], confidence: float = 0.95,
 method: str = 'historical') -> float:
 """Calculate Value at Risk"""
 if not returns:
 return 0.0

 returns_array = np.array(returns)

 if method == 'historical':
 return -np.percentile(returns_array, (1 - confidence) * 100)
 elif method == 'parametric':
 mean_return = np.mean(returns_array)
 std_return = np.std(returns_array, ddof=1)
 z_score = -1.645 if confidence == 0.95 else -2.326 # 95% or 99%
 return -(mean_return + z_score * std_return)
 else:
 raise ValueError(f"Unknown VaR method: {method}")

 def calculate_expected_shortfall(self, returns: List[float], confidence: float = 0.95) -> float:
 """Calculate Expected Shortfall (Conditional VaR)"""
 if not returns:
 return 0.0

 var_threshold = self.calculate_var(returns, confidence)
 returns_array = np.array(returns)
 tail_returns = returns_array[returns_array <= -var_threshold]

 if len(tail_returns) == 0:
 return var_threshold

 return -np.mean(tail_returns)

 def calculate_calmar_ratio(self, returns: List[float]) -> float:
 """Calculate Calmar ratio (annual return / max drawdown)"""
 if not returns:
 return 0.0

 annual_return = np.mean(returns) * 252
 cumulative_returns = np.cumprod(1 + np.array(returns))
 max_dd, _, _ = self.calculate_max_drawdown(cumulative_returns)

 if max_dd == 0:
 return float('inf') if annual_return > 0 else 0.0

 return annual_return / max_dd

 def calculate_information_ratio(self, returns: List[float],
 benchmark_returns: List[float]) -> float:
 """Calculate Information Ratio"""
 if not returns or not benchmark_returns or len(returns) != len(benchmark_returns):
 return 0.0

 excess_returns = np.array(returns) - np.array(benchmark_returns)
 tracking_error = np.std(excess_returns, ddof=1)

 if tracking_error == 0:
 return 0.0

 return np.mean(excess_returns) / tracking_error * np.sqrt(252)

 def calculate_hit_ratio(self, returns: List[float]) -> float:
 """Calculate hit ratio (percentage of positive returns)"""
 if not returns:
 return 0.0

 positive_returns = sum(1 for r in returns if r > 0)
 return positive_returns / len(returns)

 def calculate_profit_factor(self, returns: List[float]) -> float:
 """Calculate profit factor (gross profit / gross loss)"""
 if not returns:
 return 0.0

 returns_array = np.array(returns)
 gross_profit = np.sum(returns_array[returns_array > 0])
 gross_loss = abs(np.sum(returns_array[returns_array < 0]))

 if gross_loss == 0:
 return float('inf') if gross_profit > 0 else 0.0

 return gross_profit / gross_loss


class PnLAttributionEngine:
 """Advanced P&L attribution analysis"""

 def __init__(self):
 self.attribution_history: deque = deque(maxlen=10000)
 self.previous_positions: Dict[str, Position] = {}
 self.previous_greeks: Dict[str, Greeks] = {}
 self.previous_prices: Dict[str, float] = {}

 def calculate_detailed_attribution(self,
 current_positions: Dict[str, Position],
 current_greeks: Dict[str, Greeks],
 current_prices: Dict[str, float],
 trades: List[Trade],
 time_elapsed_hours: float) -> Dict[str, float]:
 """Calculate detailed P&L attribution"""

 attribution = {
 'delta_pnl': 0.0,
 'gamma_pnl': 0.0,
 'theta_pnl': 0.0,
 'vega_pnl': 0.0,
 'rho_pnl': 0.0,
 'trading_pnl': 0.0,
 'carry_pnl': 0.0,
 'hedge_pnl': 0.0,
 'commission_cost': 0.0,
 'slippage_cost': 0.0,
 'unexplained_pnl': 0.0
 }

 # Calculate price changes
 price_changes = {}
 for symbol, current_price in current_prices.items():
 if symbol in self.previous_prices:
 price_changes[symbol] = current_price - self.previous_prices[symbol]
 else:
 price_changes[symbol] = 0.0

 # Calculate Greeks P&L for each position
 for symbol, position in current_positions.items():
 if position.quantity == 0:
 continue

 prev_greeks = self.previous_greeks.get(symbol)
 curr_greeks = current_greeks.get(symbol)
 price_change = price_changes.get(symbol, 0.0)

 if prev_greeks and curr_greeks and price_change != 0:
 # Delta P&L
 delta_pnl = position.quantity * prev_greeks.delta * price_change
 attribution['delta_pnl'] += delta_pnl

 # Gamma P&L
 gamma_pnl = 0.5 * position.quantity * prev_greeks.gamma * (price_change ** 2)
 attribution['gamma_pnl'] += gamma_pnl

 # Theta P&L (time decay)
 theta_pnl = position.quantity * prev_greeks.theta * (time_elapsed_hours / 24)
 attribution['theta_pnl'] += theta_pnl

 # Vega P&L (would need volatility changes - simplified here)
 # vega_pnl = position.quantity * prev_greeks.vega * vol_change
 # attribution['vega_pnl'] += vega_pnl

 # Trading P&L
 for trade in trades:
 # Mark-to-market the trade
 current_price = current_prices.get(trade.symbol, trade.price)

 if trade.side.value == 'bid': # We bought
 trading_pnl = trade.quantity * (current_price - trade.price)
 else: # We sold
 trading_pnl = trade.quantity * (trade.price - current_price)

 attribution['trading_pnl'] += trading_pnl
 attribution['commission_cost'] += trade.commission
 attribution['slippage_cost'] += trade.slippage

 # Carry P&L (funding costs, dividends, etc.)
 total_portfolio_value = sum(abs(pos.market_value) for pos in current_positions.values())
 daily_carry_rate = 0.02 / 365 # 2% annual funding cost
 attribution['carry_pnl'] = -total_portfolio_value * daily_carry_rate * (time_elapsed_hours / 24)

 # Store attribution
 self.attribution_history.append({
 'timestamp': datetime.now(),
 'attribution': attribution.copy(),
 'time_elapsed_hours': time_elapsed_hours
 })

 # Update previous state
 self.previous_positions = current_positions.copy()
 self.previous_greeks = current_greeks.copy()
 self.previous_prices = current_prices.copy()

 return attribution

 def get_attribution_summary(self, period_hours: int = 24) -> Dict[str, float]:
 """Get attribution summary for a period"""
 cutoff_time = datetime.now() - timedelta(hours=period_hours)

 period_attributions = [
 attr['attribution'] for attr in self.attribution_history
 if attr['timestamp'] > cutoff_time
 ]

 if not period_attributions:
 return {}

 summary = {}
 for key in period_attributions[0].keys():
 summary[key] = sum(attr[key] for attr in period_attributions)

 return summary

 def calculate_attribution_statistics(self, period_days: int = 30) -> Dict[str, Dict[str, float]]:
 """Calculate attribution statistics over a period"""
 cutoff_time = datetime.now() - timedelta(days=period_days)

 period_attributions = [
 attr['attribution'] for attr in self.attribution_history
 if attr['timestamp'] > cutoff_time
 ]

 if not period_attributions:
 return {}

 stats = {}
 for key in period_attributions[0].keys():
 values = [attr[key] for attr in period_attributions]
 stats[key] = {
 'mean': np.mean(values),
 'std': np.std(values),
 'min': np.min(values),
 'max': np.max(values),
 'total': np.sum(values),
 'sharpe': np.mean(values) / np.std(values) if np.std(values) > 0 else 0
 }

 return stats


class StrategyPerformanceMonitor:
 """Monitors performance of individual strategies"""

 def __init__(self, strategy_id: str):
 self.strategy_id = strategy_id
 self.performance_history: deque = deque(maxlen=10000)
 self.daily_snapshots: deque = deque(maxlen=365) # 1 year of daily data
 self.calculator = PerformanceCalculator()
 self.attribution_engine = PnLAttributionEngine()

 # State tracking
 self.start_date: Optional[datetime] = None
 self.last_snapshot_date: Optional[datetime] = None
 self.cumulative_pnl: float = 0.0
 self.daily_pnls: List[float] = []

 def update_performance(self,
 positions: Dict[str, Position],
 greeks: Dict[str, Greeks],
 market_data: Dict[str, MarketData],
 trades: List[Trade]) -> PerformanceSnapshot:
 """Update strategy performance with current state"""

 timestamp = datetime.now()

 if self.start_date is None:
 self.start_date = timestamp

 # Calculate current portfolio metrics
 total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in positions.values())
 total_exposure = sum(abs(pos.market_value) for pos in positions.values())
 positions_count = len([pos for pos in positions.values() if pos.quantity != 0])

 # Daily P&L calculation
 if self.last_snapshot_date is None or timestamp.date() != self.last_snapshot_date:
 if self.last_snapshot_date is not None:
 daily_pnl = total_pnl - self.cumulative_pnl
 self.daily_pnls.append(daily_pnl)
 self.last_snapshot_date = timestamp.date()
 self.cumulative_pnl = total_pnl

 daily_pnl = total_pnl - self.cumulative_pnl if self.daily_pnls else 0.0

 # Calculate Greeks
 portfolio_delta = sum(g.delta * positions[symbol].quantity
 for symbol, g in greeks.items()
 if symbol in positions and positions[symbol].quantity != 0)
 portfolio_gamma = sum(g.gamma * positions[symbol].quantity
 for symbol, g in greeks.items()
 if symbol in positions and positions[symbol].quantity != 0)
 portfolio_vega = sum(g.vega * positions[symbol].quantity
 for symbol, g in greeks.items()
 if symbol in positions and positions[symbol].quantity != 0)
 portfolio_theta = sum(g.theta * positions[symbol].quantity
 for symbol, g in greeks.items()
 if symbol in positions and positions[symbol].quantity != 0)

 # Risk metrics
 returns = self.daily_pnls[-30:] if len(self.daily_pnls) >= 30 else self.daily_pnls
 var_95 = self.calculator.calculate_var(returns) if returns else 0.0

 # Performance metrics
 sharpe_ratio = self.calculator.calculate_sharpe_ratio(returns) if len(returns) >= 2 else 0.0

 if len(self.daily_pnls) >= 2:
 cumulative_returns = np.cumprod(1 + np.array(self.daily_pnls))
 max_drawdown, _, _ = self.calculator.calculate_max_drawdown(cumulative_returns.tolist())
 else:
 max_drawdown = 0.0

 hit_ratio = self.calculator.calculate_hit_ratio(returns) if returns else 0.0

 # P&L Attribution
 current_prices = {symbol: data.last for symbol, data in market_data.items()}
 time_elapsed = 1.0 # 1 hour default
 pnl_attribution = self.attribution_engine.calculate_detailed_attribution(
 positions, greeks, current_prices, trades, time_elapsed
 )

 # Trading metrics
 recent_trades = [t for t in trades if t.timestamp > timestamp - timedelta(hours=24)]
 trades_count = len(recent_trades)
 avg_trade_pnl = np.mean([t.quantity * t.price for t in recent_trades]) if recent_trades else 0.0

 win_trades = [t for t in recent_trades if t.quantity * t.price > 0]
 loss_trades = [t for t in recent_trades if t.quantity * t.price < 0]

 win_rate = len(win_trades) / len(recent_trades) if recent_trades else 0.0
 avg_win = np.mean([t.quantity * t.price for t in win_trades]) if win_trades else 0.0
 avg_loss = np.mean([abs(t.quantity * t.price) for t in loss_trades]) if loss_trades else 0.0

 # Create snapshot
 snapshot = PerformanceSnapshot(
 timestamp=timestamp,
 strategy_id=self.strategy_id,
 total_pnl=total_pnl,
 daily_pnl=daily_pnl,
 positions_count=positions_count,
 total_exposure=total_exposure,
 portfolio_delta=portfolio_delta,
 portfolio_gamma=portfolio_gamma,
 portfolio_vega=portfolio_vega,
 portfolio_theta=portfolio_theta,
 var_95=var_95,
 sharpe_ratio=sharpe_ratio,
 max_drawdown=max_drawdown,
 hit_ratio=hit_ratio,
 pnl_attribution=pnl_attribution,
 trades_count=trades_count,
 avg_trade_pnl=avg_trade_pnl,
 win_rate=win_rate,
 avg_win=avg_win,
 avg_loss=avg_loss
 )

 # Store snapshot
 self.performance_history.append(snapshot)

 # Store daily snapshot
 if timestamp.hour == 16 and timestamp.minute == 0: # Market close
 self.daily_snapshots.append(snapshot)

 return snapshot

 def get_strategy_performance(self, period_days: int = 30) -> StrategyPerformance:
 """Get comprehensive strategy performance analysis"""

 cutoff_time = datetime.now() - timedelta(days=period_days)
 period_snapshots = [s for s in self.performance_history if s.timestamp > cutoff_time]

 if not period_snapshots:
 return self._empty_performance()

 # Extract daily returns
 daily_returns = []
 for i in range(1, len(period_snapshots)):
 if period_snapshots[i].total_pnl != 0:
 daily_return = (period_snapshots[i].total_pnl - period_snapshots[i-1].total_pnl) / abs(period_snapshots[i-1].total_pnl)
 daily_returns.append(daily_return)

 if not daily_returns:
 return self._empty_performance()

 # Calculate metrics
 total_return = (period_snapshots[-1].total_pnl - period_snapshots[0].total_pnl) / abs(period_snapshots[0].total_pnl) if period_snapshots[0].total_pnl != 0 else 0
 annualized_return = total_return * (365 / period_days)
 volatility = np.std(daily_returns) * np.sqrt(252)

 sharpe_ratio = self.calculator.calculate_sharpe_ratio(daily_returns)
 sortino_ratio = self.calculator.calculate_sortino_ratio(daily_returns)
 calmar_ratio = self.calculator.calculate_calmar_ratio(daily_returns)

 cumulative_returns = np.cumprod(1 + np.array(daily_returns))
 max_drawdown, _, _ = self.calculator.calculate_max_drawdown(cumulative_returns.tolist())

 var_95 = self.calculator.calculate_var(daily_returns)
 expected_shortfall = self.calculator.calculate_expected_shortfall(daily_returns)

 hit_ratio = self.calculator.calculate_hit_ratio(daily_returns)
 profit_factor = self.calculator.calculate_profit_factor(daily_returns)

 # Trading metrics
 all_trades = []
 for snapshot in period_snapshots:
 all_trades.extend([snapshot.avg_trade_pnl] * snapshot.trades_count)

 total_trades = sum(s.trades_count for s in period_snapshots)
 win_rate = np.mean([s.win_rate for s in period_snapshots if s.win_rate > 0]) if period_snapshots else 0
 avg_win = np.mean([s.avg_win for s in period_snapshots if s.avg_win > 0]) if period_snapshots else 0
 avg_loss = np.mean([s.avg_loss for s in period_snapshots if s.avg_loss > 0]) if period_snapshots else 0

 # Greeks attribution
 delta_pnl = sum(s.pnl_attribution.get('delta_pnl', 0) for s in period_snapshots)
 gamma_pnl = sum(s.pnl_attribution.get('gamma_pnl', 0) for s in period_snapshots)
 theta_pnl = sum(s.pnl_attribution.get('theta_pnl', 0) for s in period_snapshots)
 vega_pnl = sum(s.pnl_attribution.get('vega_pnl', 0) for s in period_snapshots)

 # Costs
 total_commissions = sum(s.pnl_attribution.get('commission_cost', 0) for s in period_snapshots)
 total_slippage = sum(s.pnl_attribution.get('slippage_cost', 0) for s in period_snapshots)

 return StrategyPerformance(
 strategy_id=self.strategy_id,
 start_date=period_snapshots[0].timestamp,
 end_date=period_snapshots[-1].timestamp,
 total_return=total_return,
 annualized_return=annualized_return,
 daily_returns=daily_returns,
 volatility=volatility,
 max_drawdown=max_drawdown,
 var_95=var_95,
 expected_shortfall=expected_shortfall,
 sharpe_ratio=sharpe_ratio,
 sortino_ratio=sortino_ratio,
 calmar_ratio=calmar_ratio,
 information_ratio=0.0, # Would need benchmark
 total_trades=total_trades,
 win_rate=win_rate,
 profit_factor=profit_factor,
 avg_win=avg_win,
 avg_loss=avg_loss,
 delta_pnl_contribution=delta_pnl,
 gamma_pnl_contribution=gamma_pnl,
 theta_pnl_contribution=theta_pnl,
 vega_pnl_contribution=vega_pnl,
 total_commissions=total_commissions,
 total_slippage=total_slippage,
 correlation_to_market=0.0, # Would calculate vs market index
 beta_to_market=0.0, # Would calculate vs market index
 tracking_error=0.0 # Would calculate vs benchmark
 )

 def _empty_performance(self) -> StrategyPerformance:
 """Return empty performance object"""
 return StrategyPerformance(
 strategy_id=self.strategy_id,
 start_date=datetime.now(),
 end_date=datetime.now(),
 total_return=0.0,
 annualized_return=0.0,
 daily_returns=[],
 volatility=0.0,
 max_drawdown=0.0,
 var_95=0.0,
 expected_shortfall=0.0,
 sharpe_ratio=0.0,
 sortino_ratio=0.0,
 calmar_ratio=0.0,
 information_ratio=0.0,
 total_trades=0,
 win_rate=0.0,
 profit_factor=0.0,
 avg_win=0.0,
 avg_loss=0.0,
 delta_pnl_contribution=0.0,
 gamma_pnl_contribution=0.0,
 theta_pnl_contribution=0.0,
 vega_pnl_contribution=0.0,
 total_commissions=0.0,
 total_slippage=0.0,
 correlation_to_market=0.0,
 beta_to_market=0.0,
 tracking_error=0.0
 )


class RiskMonitor:
 """Real-time risk monitoring system"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'var_confidence_levels': [0.95, 0.99],
 'stress_scenarios': ['market_crash', 'vol_spike', 'rate_shock'],
 'concentration_threshold': 0.2,
 'liquidity_threshold': 0.1
 }

 self.risk_history: deque = deque(maxlen=1000)

 def calculate_risk_metrics(self,
 positions: Dict[str, Position],
 market_data: Dict[str, MarketData],
 historical_returns: Dict[str, List[float]]) -> RiskMetrics:
 """Calculate comprehensive risk metrics"""

 timestamp = datetime.now()
 portfolio_value = sum(pos.market_value for pos in positions.values())

 # Portfolio returns for VaR calculation
 portfolio_returns = self._calculate_portfolio_returns(positions, historical_returns)

 # VaR calculations
 var_95_1d = self._calculate_var(portfolio_returns, 0.95) * portfolio_value
 var_99_1d = self._calculate_var(portfolio_returns, 0.99) * portfolio_value
 var_95_10d = var_95_1d * np.sqrt(10) # 10-day VaR
 expected_shortfall_95 = self._calculate_expected_shortfall(portfolio_returns, 0.95) * portfolio_value

 # Greeks risk
 total_delta = sum(pos.greeks.delta * pos.quantity for pos in positions.values() if pos.greeks)
 total_gamma = sum(pos.greeks.gamma * pos.quantity for pos in positions.values() if pos.greeks)
 total_vega = sum(pos.greeks.vega * pos.quantity for pos in positions.values() if pos.greeks)
 total_theta = sum(pos.greeks.theta * pos.quantity for pos in positions.values() if pos.greeks)

 # Estimate 1-day Greeks risk (simplified)
 delta_risk = abs(total_delta) * 0.02 * portfolio_value # 2% price move
 gamma_risk = 0.5 * abs(total_gamma) * (0.02 ** 2) * portfolio_value # Gamma risk
 vega_risk = abs(total_vega) * 0.05 * portfolio_value / 100 # 5% vol move
 theta_risk = abs(total_theta) * portfolio_value # 1-day theta decay

 # Concentration metrics
 position_weights = {symbol: abs(pos.market_value) / portfolio_value
 for symbol, pos in positions.items()
 if portfolio_value > 0}

 concentration_herfindahl = sum(w ** 2 for w in position_weights.values())
 max_position_weight = max(position_weights.values()) if position_weights else 0

 # Sector concentration (simplified)
 sector_concentration = self._calculate_sector_concentration(positions, position_weights)

 # Liquidity metrics
 liquidity_score = self._calculate_liquidity_score(positions, market_data)
 days_to_liquidate = self._estimate_liquidation_time(positions, market_data)

 # Stress test results
 stress_test_results = self._run_stress_tests(positions, market_data)

 risk_metrics = RiskMetrics(
 timestamp=timestamp,
 portfolio_value=portfolio_value,
 var_95_1d=var_95_1d,
 var_99_1d=var_99_1d,
 var_95_10d=var_95_10d,
 expected_shortfall_95=expected_shortfall_95,
 delta_risk=delta_risk,
 gamma_risk=gamma_risk,
 vega_risk=vega_risk,
 theta_risk=theta_risk,
 concentration_herfindahl=concentration_herfindahl,
 max_position_weight=max_position_weight,
 sector_concentration=sector_concentration,
 liquidity_score=liquidity_score,
 days_to_liquidate=days_to_liquidate,
 stress_test_results=stress_test_results
 )

 self.risk_history.append(risk_metrics)
 return risk_metrics

 def _calculate_portfolio_returns(self, positions: Dict[str, Position],
 historical_returns: Dict[str, List[float]]) -> List[float]:
 """Calculate portfolio returns from position weights and asset returns"""

 # Get portfolio weights
 total_value = sum(abs(pos.market_value) for pos in positions.values())
 if total_value == 0:
 return []

 weights = {symbol: pos.market_value / total_value for symbol, pos in positions.items()}

 # Calculate portfolio returns
 min_length = min(len(returns) for returns in historical_returns.values() if returns)
 if min_length == 0:
 return []

 portfolio_returns = []
 for i in range(min_length):
 portfolio_return = sum(weights.get(symbol, 0) * historical_returns[symbol][i]
 for symbol in historical_returns.keys()
 if symbol in weights)
 portfolio_returns.append(portfolio_return)

 return portfolio_returns

 def _calculate_var(self, returns: List[float], confidence: float) -> float:
 """Calculate Value at Risk"""
 if not returns:
 return 0.0
 return -np.percentile(returns, (1 - confidence) * 100)

 def _calculate_expected_shortfall(self, returns: List[float], confidence: float) -> float:
 """Calculate Expected Shortfall"""
 if not returns:
 return 0.0
 var_threshold = self._calculate_var(returns, confidence)
 tail_returns = [r for r in returns if r <= -var_threshold]
 return -np.mean(tail_returns) if tail_returns else var_threshold

 def _calculate_sector_concentration(self, positions: Dict[str, Position],
 position_weights: Dict[str, float]) -> Dict[str, float]:
 """Calculate sector concentration"""
 # Simplified sector mapping based on symbol prefix
 sector_weights = defaultdict(float)

 for symbol, weight in position_weights.items():
 # Extract underlying symbol (before underscore)
 underlying = symbol.split('_')[0] if '_' in symbol else symbol

 # Simple sector mapping (would be more sophisticated in practice)
 if underlying in ['AAPL', 'MSFT', 'GOOGL']:
 sector = 'Technology'
 elif underlying in ['JPM', 'BAC', 'GS']:
 sector = 'Financial'
 elif underlying in ['XOM', 'CVX']:
 sector = 'Energy'
 else:
 sector = 'Other'

 sector_weights[sector] += weight

 return dict(sector_weights)

 def _calculate_liquidity_score(self, positions: Dict[str, Position],
 market_data: Dict[str, MarketData]) -> float:
 """Calculate portfolio liquidity score"""

 total_value = sum(abs(pos.market_value) for pos in positions.values())
 if total_value == 0:
 return 1.0

 weighted_liquidity = 0.0

 for symbol, position in positions.items():
 if symbol in market_data:
 data = market_data[symbol]

 # Liquidity score based on volume and spread
 if data.volume > 0 and data.bid > 0 and data.ask > 0:
 spread_ratio = (data.ask - data.bid) / ((data.ask + data.bid) / 2)
 volume_score = min(1.0, data.volume / 1000) # Normalize by 1000
 spread_score = max(0, 1.0 - spread_ratio * 10) # Penalize wide spreads

 liquidity_score = (volume_score + spread_score) / 2
 else:
 liquidity_score = 0.1 # Low liquidity for missing data

 weight = abs(position.market_value) / total_value
 weighted_liquidity += weight * liquidity_score

 return weighted_liquidity

 def _estimate_liquidation_time(self, positions: Dict[str, Position],
 market_data: Dict[str, MarketData]) -> float:
 """Estimate time to liquidate portfolio in days"""

 max_liquidation_days = 0.0

 for symbol, position in positions.items():
 if symbol in market_data and position.quantity != 0:
 data = market_data[symbol]

 # Estimate liquidation time based on volume
 daily_volume = data.volume * 6.5 # Assume 6.5 trading hours
 position_size = abs(position.quantity)

 # Assume we can trade 10% of daily volume
 participation_rate = 0.1
 liquidation_days = position_size / (daily_volume * participation_rate)

 max_liquidation_days = max(max_liquidation_days, liquidation_days)

 return max_liquidation_days

 def _run_stress_tests(self, positions: Dict[str, Position],
 market_data: Dict[str, MarketData]) -> Dict[str, float]:
 """Run stress tests on portfolio"""

 stress_results = {}
 portfolio_value = sum(pos.market_value for pos in positions.values())

 # Market crash scenario (-20% equity move)
 market_crash_pnl = 0.0
 for symbol, position in positions.items():
 if position.greeks:
 # Simplified: assume 20% underlying move
 price_shock = -0.20 * (market_data.get(symbol, MarketData(symbol, datetime.now(), 0, 0, 100, 0, 0, 0, 0)).last)
 delta_pnl = position.quantity * position.greeks.delta * price_shock
 gamma_pnl = 0.5 * position.quantity * position.greeks.gamma * (price_shock ** 2)
 market_crash_pnl += delta_pnl + gamma_pnl

 stress_results['market_crash'] = market_crash_pnl / portfolio_value if portfolio_value > 0 else 0

 # Volatility spike scenario (+50% vol)
 vol_spike_pnl = 0.0
 for symbol, position in positions.items():
 if position.greeks:
 vol_shock = 0.5 # 50% vol increase
 vega_pnl = position.quantity * position.greeks.vega * vol_shock
 vol_spike_pnl += vega_pnl

 stress_results['vol_spike'] = vol_spike_pnl / portfolio_value if portfolio_value > 0 else 0

 # Interest rate shock (+200bps)
 rate_shock_pnl = 0.0
 for symbol, position in positions.items():
 if position.greeks:
 rate_shock = 0.02 # 200bps increase
 rho_pnl = position.quantity * position.greeks.rho * rate_shock
 rate_shock_pnl += rho_pnl

 stress_results['rate_shock'] = rate_shock_pnl / portfolio_value if portfolio_value > 0 else 0

 return stress_results


class PerformanceReportGenerator:
 """Generates comprehensive performance reports"""

 def __init__(self):
 self.calculator = PerformanceCalculator()

 def generate_strategy_report(self, strategy_performance: StrategyPerformance) -> Dict[str, Any]:
 """Generate comprehensive strategy performance report"""

 report = {
 'strategy_summary': {
 'strategy_id': strategy_performance.strategy_id,
 'period': f"{strategy_performance.start_date.date()} to {strategy_performance.end_date.date()}",
 'total_return': f"{strategy_performance.total_return:.2%}",
 'annualized_return': f"{strategy_performance.annualized_return:.2%}",
 'sharpe_ratio': f"{strategy_performance.sharpe_ratio:.2f}",
 'max_drawdown': f"{strategy_performance.max_drawdown:.2%}",
 },

 'risk_metrics': {
 'volatility': f"{strategy_performance.volatility:.2%}",
 'var_95': f"{strategy_performance.var_95:.2%}",
 'expected_shortfall': f"{strategy_performance.expected_shortfall:.2%}",
 'sortino_ratio': f"{strategy_performance.sortino_ratio:.2f}",
 'calmar_ratio': f"{strategy_performance.calmar_ratio:.2f}",
 },

 'trading_metrics': {
 'total_trades': strategy_performance.total_trades,
 'win_rate': f"{strategy_performance.win_rate:.2%}",
 'profit_factor': f"{strategy_performance.profit_factor:.2f}",
 'avg_win': f"${strategy_performance.avg_win:.2f}",
 'avg_loss': f"${strategy_performance.avg_loss:.2f}",
 },

 'greeks_attribution': {
 'delta_pnl': f"${strategy_performance.delta_pnl_contribution:.2f}",
 'gamma_pnl': f"${strategy_performance.gamma_pnl_contribution:.2f}",
 'theta_pnl': f"${strategy_performance.theta_pnl_contribution:.2f}",
 'vega_pnl': f"${strategy_performance.vega_pnl_contribution:.2f}",
 },

 'costs': {
 'total_commissions': f"${strategy_performance.total_commissions:.2f}",
 'total_slippage': f"${strategy_performance.total_slippage:.2f}",
 'total_costs': f"${strategy_performance.total_commissions + strategy_performance.total_slippage:.2f}",
 }
 }

 return report

 def generate_risk_report(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
 """Generate risk monitoring report"""

 report = {
 'portfolio_overview': {
 'portfolio_value': f"${risk_metrics.portfolio_value:,.2f}",
 'timestamp': risk_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
 },

 'var_metrics': {
 'var_95_1d': f"${risk_metrics.var_95_1d:,.2f}",
 'var_99_1d': f"${risk_metrics.var_99_1d:,.2f}",
 'var_95_10d': f"${risk_metrics.var_95_10d:,.2f}",
 'expected_shortfall_95': f"${risk_metrics.expected_shortfall_95:,.2f}",
 },

 'greeks_risk': {
 'delta_risk': f"${risk_metrics.delta_risk:,.2f}",
 'gamma_risk': f"${risk_metrics.gamma_risk:,.2f}",
 'vega_risk': f"${risk_metrics.vega_risk:,.2f}",
 'theta_risk': f"${risk_metrics.theta_risk:,.2f}",
 },

 'concentration_metrics': {
 'herfindahl_index': f"{risk_metrics.concentration_herfindahl:.3f}",
 'max_position_weight': f"{risk_metrics.max_position_weight:.2%}",
 'sector_concentration': {k: f"{v:.2%}" for k, v in risk_metrics.sector_concentration.items()},
 },

 'liquidity_metrics': {
 'liquidity_score': f"{risk_metrics.liquidity_score:.2f}",
 'days_to_liquidate': f"{risk_metrics.days_to_liquidate:.1f} days",
 },

 'stress_tests': {k: f"{v:.2%}" for k, v in risk_metrics.stress_test_results.items()}
 }

 return report


# Factory functions
def create_performance_monitor(strategy_id: str) -> StrategyPerformanceMonitor:
 """Create a performance monitor for a strategy"""
 return StrategyPerformanceMonitor(strategy_id)


def create_risk_monitor(config: Optional[Dict[str, Any]] = None) -> RiskMonitor:
 """Create a risk monitor with configuration"""
 return RiskMonitor(config)


# Example usage
if __name__ == "__main__":
 # Create performance monitor
 monitor = create_performance_monitor("delta_neutral_strategy")

 # Example positions and data
 positions = {
 "AAPL_231215C150": Position("AAPL_231215C150", 100, 5.0, 50000, 1000, 500),
 "AAPL_231215P140": Position("AAPL_231215P140", -50, 3.0, -15000, -500, 200)
 }

 greeks = {
 "AAPL_231215C150": Greeks(0.6, 0.05, -0.1, 0.3, 0.02, 150.0, datetime.now()),
 "AAPL_231215P140": Greeks(-0.4, 0.05, -0.08, 0.25, -0.01, 150.0, datetime.now())
 }

 market_data = {
 "AAPL_231215C150": MarketData("AAPL_231215C150", datetime.now(), 4.90, 5.10, 5.00, 50, 50, 1000, 5000),
 "AAPL_231215P140": MarketData("AAPL_231215P140", datetime.now(), 2.90, 3.10, 3.00, 30, 30, 800, 3000)
 }

 trades = []

 # Update performance
 snapshot = monitor.update_performance(positions, greeks, market_data, trades)

 print("Performance Snapshot:")
 print(f"Total P&L: ${snapshot.total_pnl:.2f}")
 print(f"Portfolio Delta: {snapshot.portfolio_delta:.2f}")
 print(f"Sharpe Ratio: {snapshot.sharpe_ratio:.2f}")
 print(f"VaR 95%: ${snapshot.var_95:.2f}")

 # Get strategy performance
 performance = monitor.get_strategy_performance(30)

 print(f"\n30-Day Performance:")
 print(f"Total Return: {performance.total_return:.2%}")
 print(f"Annualized Return: {performance.annualized_return:.2%}")
 print(f"Volatility: {performance.volatility:.2%}")
 print(f"Max Drawdown: {performance.max_drawdown:.2%}")

 # Generate report
 report_generator = PerformanceReportGenerator()
 report = report_generator.generate_strategy_report(performance)

 print(f"\nStrategy Report:")
 print(json.dumps(report, indent=2))