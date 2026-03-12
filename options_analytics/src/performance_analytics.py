"""
Performance Measurement and Attribution Tools
Comprehensive performance analytics for options trading strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict, deque
from scipy import stats
import math

from analytics_framework import Position, PortfolioGreeks
from portfolio_monitor import PortfolioSnapshot, PnLAttribution

class PerformanceMetric(Enum):
 TOTAL_RETURN = "total_return"
 ANNUALIZED_RETURN = "annualized_return"
 SHARPE_RATIO = "sharpe_ratio"
 CALMAR_RATIO = "calmar_ratio"
 MAX_DRAWDOWN = "max_drawdown"
 VOLATILITY = "volatility"
 VAR_95 = "var_95"
 WIN_RATE = "win_rate"
 PROFIT_FACTOR = "profit_factor"
 AVERAGE_WIN = "average_win"
 AVERAGE_LOSS = "average_loss"

class AttributionType(Enum):
 BY_STRATEGY = "by_strategy"
 BY_SYMBOL = "by_symbol"
 BY_GREEKS = "by_greeks"
 BY_TIME = "by_time"
 BY_SECTOR = "by_sector"

@dataclass
class PerformanceMetrics:
 period_start: datetime
 period_end: datetime
 total_return: float
 annualized_return: float
 volatility: float
 sharpe_ratio: float
 calmar_ratio: float
 max_drawdown: float
 max_drawdown_duration: int
 var_95: float
 expected_shortfall: float
 win_rate: float
 profit_factor: float
 average_win: float
 average_loss: float
 total_trades: int
 winning_trades: int
 losing_trades: int
 largest_win: float
 largest_loss: float
 consecutive_wins: int
 consecutive_losses: int

@dataclass
class AttributionAnalysis:
 attribution_type: AttributionType
 period_start: datetime
 period_end: datetime
 total_pnl: float
 contributions: Dict[str, float]
 percentage_contributions: Dict[str, float]
 risk_contributions: Dict[str, float]
 return_contributions: Dict[str, float]

@dataclass
class StrategyPerformance:
 strategy_name: str
 period_start: datetime
 period_end: datetime
 positions_count: int
 total_pnl: float
 realized_pnl: float
 unrealized_pnl: float
 total_return: float
 win_rate: float
 sharpe_ratio: float
 max_drawdown: float
 trades: int
 avg_holding_period: float
 risk_adjusted_return: float

@dataclass
class RiskMetrics:
 timestamp: datetime
 portfolio_var: float
 portfolio_cvar: float
 leverage_ratio: float
 concentration_risk: float
 correlation_risk: float
 volatility_risk: float
 greeks_risk: Dict[str, float]
 stress_test_results: Dict[str, float]

class PerformanceAnalyzer:
 """Comprehensive performance analysis engine"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 # Performance data storage
 self.portfolio_history: deque = deque(maxlen=10000)
 self.pnl_history: deque = deque(maxlen=10000)
 self.trade_history: List[Dict[str, Any]] = []
 self.performance_cache: Dict[str, Any] = {}

 # Analysis components
 self.metrics_calculator = MetricsCalculator(config)
 self.attribution_engine = AttributionEngine(config)
 self.risk_analyzer = RiskAnalyzer(config)
 self.benchmark_analyzer = BenchmarkAnalyzer(config)

 # Configuration
 self.benchmark_symbol = config.get('benchmark_symbol', 'SPY')
 self.risk_free_rate = config.get('risk_free_rate', 0.02)
 self.analysis_periods = config.get('analysis_periods', ['1d', '1w', '1m', '3m', '6m', '1y'])

 async def initialize(self):
 """Initialize performance analyzer"""
 self.logger.info("Initializing Performance Analyzer")
 await asyncio.gather(
 self.metrics_calculator.initialize(),
 self.attribution_engine.initialize(),
 self.risk_analyzer.initialize(),
 self.benchmark_analyzer.initialize()
 )

 async def update_portfolio_data(self, snapshot: PortfolioSnapshot):
 """Update portfolio performance data"""
 self.portfolio_history.append(snapshot)
 await self._update_performance_cache()

 async def update_pnl_data(self, pnl_attribution: PnLAttribution):
 """Update P&L attribution data"""
 self.pnl_history.append(pnl_attribution)

 async def add_trade(self, trade_data: Dict[str, Any]):
 """Add trade to history"""
 self.trade_history.append(trade_data)

 # Limit trade history size
 if len(self.trade_history) > 50000:
 self.trade_history = self.trade_history[-50000:]

 async def calculate_performance_metrics(self, period: str = '1m') -> PerformanceMetrics:
 """Calculate comprehensive performance metrics"""
 return await self.metrics_calculator.calculate_metrics(
 list(self.portfolio_history), period, self.risk_free_rate
 )

 async def calculate_attribution_analysis(self, attribution_type: AttributionType,
 period: str = '1m') -> AttributionAnalysis:
 """Calculate performance attribution analysis"""
 return await self.attribution_engine.calculate_attribution(
 list(self.pnl_history), attribution_type, period
 )

 async def analyze_strategy_performance(self, strategy_name: str,
 period: str = '1m') -> StrategyPerformance:
 """Analyze performance of specific strategy"""
 strategy_trades = [t for t in self.trade_history if t.get('strategy') == strategy_name]
 strategy_pnl = [p for p in self.pnl_history if strategy_name in p.strategy_attribution]

 return await self._calculate_strategy_performance(
 strategy_name, strategy_trades, strategy_pnl, period
 )

 async def calculate_risk_metrics(self, positions: Dict[str, Position]) -> RiskMetrics:
 """Calculate comprehensive risk metrics"""
 return await self.risk_analyzer.calculate_risk_metrics(positions, list(self.portfolio_history))

 async def generate_performance_report(self, period: str = '1m') -> Dict[str, Any]:
 """Generate comprehensive performance report"""
 # Calculate all metrics
 performance_metrics = await self.calculate_performance_metrics(period)
 risk_metrics = await self.calculate_risk_metrics({})

 # Attribution analyses
 strategy_attribution = await self.calculate_attribution_analysis(
 AttributionType.BY_STRATEGY, period
 )
 symbol_attribution = await self.calculate_attribution_analysis(
 AttributionType.BY_SYMBOL, period
 )
 greeks_attribution = await self.calculate_attribution_analysis(
 AttributionType.BY_GREEKS, period
 )

 # Benchmark comparison
 benchmark_comparison = await self.benchmark_analyzer.compare_to_benchmark(
 list(self.portfolio_history), period
 )

 # Trading statistics
 trading_stats = await self._calculate_trading_statistics(period)

 # Performance attribution summary
 attribution_summary = {
 'by_strategy': strategy_attribution,
 'by_symbol': symbol_attribution,
 'by_greeks': greeks_attribution
 }

 return {
 'period': period,
 'performance_metrics': performance_metrics,
 'risk_metrics': risk_metrics,
 'attribution_analysis': attribution_summary,
 'benchmark_comparison': benchmark_comparison,
 'trading_statistics': trading_stats,
 'generated_at': datetime.now().isoformat()
 }

 async def get_performance_summary(self) -> Dict[str, Any]:
 """Get quick performance summary"""
 if not self.portfolio_history:
 return {}

 latest_snapshot = self.portfolio_history[-1]
 recent_pnl = list(self.pnl_history)[-10:] if self.pnl_history else []

 # Calculate basic metrics
 total_pnl = sum(pnl.total_pnl for pnl in recent_pnl)
 daily_returns = await self._calculate_daily_returns()

 summary = {
 'current_portfolio_value': latest_snapshot.total_value,
 'total_pnl': latest_snapshot.total_pnl,
 'daily_pnl': total_pnl,
 'positions_count': latest_snapshot.positions_count,
 'var_95': latest_snapshot.var_95,
 'sharpe_ratio': latest_snapshot.sharpe_ratio,
 'max_drawdown': latest_snapshot.max_drawdown,
 'last_updated': latest_snapshot.timestamp.isoformat()
 }

 if daily_returns:
 summary.update({
 'daily_volatility': np.std(daily_returns),
 'best_day': max(daily_returns),
 'worst_day': min(daily_returns),
 'win_rate': len([r for r in daily_returns if r > 0]) / len(daily_returns)
 })

 return summary

 async def get_drawdown_analysis(self, period: str = '6m') -> Dict[str, Any]:
 """Analyze drawdown periods"""
 period_data = await self._get_period_data(period)
 if not period_data:
 return {}

 # Calculate drawdown series
 portfolio_values = [snap.total_value for snap in period_data]
 cumulative_returns = np.array(portfolio_values) / portfolio_values[0] - 1
 running_max = np.maximum.accumulate(cumulative_returns)
 drawdowns = cumulative_returns - running_max

 # Find drawdown periods
 drawdown_periods = []
 in_drawdown = False
 start_idx = 0

 for i, dd in enumerate(drawdowns):
 if dd < -0.01 and not in_drawdown: # Start of drawdown (>1%)
 in_drawdown = True
 start_idx = i
 elif dd >= -0.001 and in_drawdown: # End of drawdown
 in_drawdown = False
 duration = i - start_idx
 max_dd = min(drawdowns[start_idx:i])

 drawdown_periods.append({
 'start_date': period_data[start_idx].timestamp,
 'end_date': period_data[i].timestamp,
 'duration_days': duration,
 'max_drawdown': max_dd,
 'recovery_days': 0 # Would calculate recovery time
 })

 return {
 'current_drawdown': drawdowns[-1] if drawdowns else 0,
 'max_drawdown': min(drawdowns) if drawdowns else 0,
 'avg_drawdown': np.mean([dd for dd in drawdowns if dd < 0]) if any(dd < 0 for dd in drawdowns) else 0,
 'drawdown_periods': drawdown_periods,
 'time_in_drawdown': len([dd for dd in drawdowns if dd < -0.01]) / len(drawdowns) if drawdowns else 0
 }

 async def _update_performance_cache(self):
 """Update performance cache with latest calculations"""
 if len(self.portfolio_history) < 2:
 return

 # Cache frequently used calculations
 daily_returns = await self._calculate_daily_returns()
 if daily_returns:
 self.performance_cache['daily_returns'] = daily_returns
 self.performance_cache['volatility'] = np.std(daily_returns) * np.sqrt(252)
 self.performance_cache['sharpe_ratio'] = (
 np.mean(daily_returns) * 252 - self.risk_free_rate
 ) / max(np.std(daily_returns) * np.sqrt(252), 0.01)

 async def _calculate_daily_returns(self) -> List[float]:
 """Calculate daily returns from portfolio history"""
 if len(self.portfolio_history) < 2:
 return []

 returns = []
 prev_value = None

 for snapshot in self.portfolio_history:
 if prev_value is not None and prev_value != 0:
 daily_return = (snapshot.total_value - prev_value) / prev_value
 returns.append(daily_return)
 prev_value = snapshot.total_value

 return returns

 async def _get_period_data(self, period: str) -> List[PortfolioSnapshot]:
 """Get portfolio data for specified period"""
 if not self.portfolio_history:
 return []

 end_time = datetime.now()

 if period == '1d':
 start_time = end_time - timedelta(days=1)
 elif period == '1w':
 start_time = end_time - timedelta(weeks=1)
 elif period == '1m':
 start_time = end_time - timedelta(days=30)
 elif period == '3m':
 start_time = end_time - timedelta(days=90)
 elif period == '6m':
 start_time = end_time - timedelta(days=180)
 elif period == '1y':
 start_time = end_time - timedelta(days=365)
 else:
 start_time = end_time - timedelta(days=30)

 return [snap for snap in self.portfolio_history if snap.timestamp >= start_time]

 async def _calculate_strategy_performance(self, strategy_name: str,
 strategy_trades: List[Dict[str, Any]],
 strategy_pnl: List[PnLAttribution],
 period: str) -> StrategyPerformance:
 """Calculate performance metrics for specific strategy"""
 if not strategy_trades and not strategy_pnl:
 return StrategyPerformance(
 strategy_name=strategy_name,
 period_start=datetime.now() - timedelta(days=30),
 period_end=datetime.now(),
 positions_count=0,
 total_pnl=0.0,
 realized_pnl=0.0,
 unrealized_pnl=0.0,
 total_return=0.0,
 win_rate=0.0,
 sharpe_ratio=0.0,
 max_drawdown=0.0,
 trades=0,
 avg_holding_period=0.0,
 risk_adjusted_return=0.0
 )

 # Calculate period boundaries
 end_time = datetime.now()
 start_time = end_time - timedelta(days=30) # Default to 1 month

 # Calculate strategy P&L
 total_pnl = sum(pnl.strategy_attribution.get(strategy_name, 0.0) for pnl in strategy_pnl)

 # Calculate trade statistics
 winning_trades = len([t for t in strategy_trades if t.get('pnl', 0) > 0])
 win_rate = winning_trades / max(len(strategy_trades), 1)

 # Calculate holding periods
 holding_periods = []
 for trade in strategy_trades:
 if 'entry_time' in trade and 'exit_time' in trade:
 holding_period = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
 holding_periods.append(holding_period)

 avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0

 # Calculate returns for Sharpe ratio
 trade_returns = [t.get('return', 0.0) for t in strategy_trades if 'return' in t]
 if trade_returns:
 sharpe_ratio = np.mean(trade_returns) / max(np.std(trade_returns), 0.01)
 else:
 sharpe_ratio = 0.0

 return StrategyPerformance(
 strategy_name=strategy_name,
 period_start=start_time,
 period_end=end_time,
 positions_count=len(set(t.get('symbol', '') for t in strategy_trades)),
 total_pnl=total_pnl,
 realized_pnl=sum(t.get('realized_pnl', 0.0) for t in strategy_trades),
 unrealized_pnl=sum(t.get('unrealized_pnl', 0.0) for t in strategy_trades),
 total_return=total_pnl / max(sum(abs(t.get('entry_value', 1.0)) for t in strategy_trades), 1.0),
 win_rate=win_rate,
 sharpe_ratio=sharpe_ratio,
 max_drawdown=0.0, # Would calculate from strategy returns
 trades=len(strategy_trades),
 avg_holding_period=avg_holding_period,
 risk_adjusted_return=total_pnl / max(np.std(trade_returns) if trade_returns else 1.0, 0.01)
 )

 async def _calculate_trading_statistics(self, period: str) -> Dict[str, Any]:
 """Calculate trading statistics for period"""
 period_data = await self._get_period_data(period)

 if not period_data:
 return {}

 period_trades = [
 t for t in self.trade_history
 if 'timestamp' in t and t['timestamp'] >= period_data[0].timestamp
 ]

 if not period_trades:
 return {}

 winning_trades = [t for t in period_trades if t.get('pnl', 0) > 0]
 losing_trades = [t for t in period_trades if t.get('pnl', 0) < 0]

 return {
 'total_trades': len(period_trades),
 'winning_trades': len(winning_trades),
 'losing_trades': len(losing_trades),
 'win_rate': len(winning_trades) / len(period_trades),
 'average_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
 'average_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
 'largest_win': max([t['pnl'] for t in winning_trades]) if winning_trades else 0,
 'largest_loss': min([t['pnl'] for t in losing_trades]) if losing_trades else 0,
 'profit_factor': (
 sum(t['pnl'] for t in winning_trades) / abs(sum(t['pnl'] for t in losing_trades))
 if losing_trades else float('inf')
 ),
 'avg_trade_duration': np.mean([
 (t.get('exit_time', t.get('timestamp', datetime.now())) -
 t.get('entry_time', t.get('timestamp', datetime.now()))).total_seconds() / 3600
 for t in period_trades
 if 'exit_time' in t or 'entry_time' in t
 ]) if period_trades else 0
 }

class MetricsCalculator:
 """Calculate performance metrics"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize metrics calculator"""
 self.logger.info("Initializing Metrics Calculator")

 async def calculate_metrics(self, portfolio_history: List[PortfolioSnapshot],
 period: str, risk_free_rate: float) -> PerformanceMetrics:
 """Calculate comprehensive performance metrics"""
 if len(portfolio_history) < 2:
 return self._empty_metrics()

 # Get period data
 period_data = self._filter_period_data(portfolio_history, period)
 if len(period_data) < 2:
 return self._empty_metrics()

 # Calculate returns
 returns = self._calculate_returns(period_data)
 if not returns:
 return self._empty_metrics()

 # Basic metrics
 total_return = (period_data[-1].total_value / period_data[0].total_value) - 1
 periods_per_year = self._get_periods_per_year(period)
 annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

 # Risk metrics
 volatility = np.std(returns) * np.sqrt(periods_per_year)
 sharpe_ratio = (annualized_return - risk_free_rate) / max(volatility, 0.01)

 # Drawdown metrics
 max_drawdown, max_dd_duration = self._calculate_max_drawdown(period_data)

 # Calculate Calmar ratio
 calmar_ratio = annualized_return / max(abs(max_drawdown), 0.01)

 # VaR and Expected Shortfall
 var_95 = np.percentile(returns, 5)
 expected_shortfall = np.mean([r for r in returns if r <= var_95])

 # Trading metrics
 winning_returns = [r for r in returns if r > 0]
 losing_returns = [r for r in returns if r < 0]

 win_rate = len(winning_returns) / len(returns)

 avg_win = np.mean(winning_returns) if winning_returns else 0
 avg_loss = np.mean(losing_returns) if losing_returns else 0

 profit_factor = (
 sum(winning_returns) / abs(sum(losing_returns))
 if losing_returns else float('inf')
 )

 # Consecutive wins/losses
 consecutive_wins, consecutive_losses = self._calculate_consecutive_periods(returns)

 return PerformanceMetrics(
 period_start=period_data[0].timestamp,
 period_end=period_data[-1].timestamp,
 total_return=total_return,
 annualized_return=annualized_return,
 volatility=volatility,
 sharpe_ratio=sharpe_ratio,
 calmar_ratio=calmar_ratio,
 max_drawdown=max_drawdown,
 max_drawdown_duration=max_dd_duration,
 var_95=var_95,
 expected_shortfall=expected_shortfall,
 win_rate=win_rate,
 profit_factor=profit_factor,
 average_win=avg_win,
 average_loss=avg_loss,
 total_trades=len(returns),
 winning_trades=len(winning_returns),
 losing_trades=len(losing_returns),
 largest_win=max(returns) if returns else 0,
 largest_loss=min(returns) if returns else 0,
 consecutive_wins=consecutive_wins,
 consecutive_losses=consecutive_losses
 )

 def _empty_metrics(self) -> PerformanceMetrics:
 """Return empty metrics object"""
 now = datetime.now()
 return PerformanceMetrics(
 period_start=now,
 period_end=now,
 total_return=0.0,
 annualized_return=0.0,
 volatility=0.0,
 sharpe_ratio=0.0,
 calmar_ratio=0.0,
 max_drawdown=0.0,
 max_drawdown_duration=0,
 var_95=0.0,
 expected_shortfall=0.0,
 win_rate=0.0,
 profit_factor=0.0,
 average_win=0.0,
 average_loss=0.0,
 total_trades=0,
 winning_trades=0,
 losing_trades=0,
 largest_win=0.0,
 largest_loss=0.0,
 consecutive_wins=0,
 consecutive_losses=0
 )

 def _filter_period_data(self, portfolio_history: List[PortfolioSnapshot],
 period: str) -> List[PortfolioSnapshot]:
 """Filter portfolio history for specified period"""
 if not portfolio_history:
 return []

 end_time = datetime.now()

 if period == '1d':
 start_time = end_time - timedelta(days=1)
 elif period == '1w':
 start_time = end_time - timedelta(weeks=1)
 elif period == '1m':
 start_time = end_time - timedelta(days=30)
 elif period == '3m':
 start_time = end_time - timedelta(days=90)
 elif period == '6m':
 start_time = end_time - timedelta(days=180)
 elif period == '1y':
 start_time = end_time - timedelta(days=365)
 else:
 start_time = end_time - timedelta(days=30)

 return [snap for snap in portfolio_history if snap.timestamp >= start_time]

 def _calculate_returns(self, portfolio_data: List[PortfolioSnapshot]) -> List[float]:
 """Calculate returns from portfolio data"""
 if len(portfolio_data) < 2:
 return []

 returns = []
 for i in range(1, len(portfolio_data)):
 prev_value = portfolio_data[i-1].total_value
 curr_value = portfolio_data[i].total_value

 if prev_value != 0:
 ret = (curr_value - prev_value) / prev_value
 returns.append(ret)

 return returns

 def _calculate_max_drawdown(self, portfolio_data: List[PortfolioSnapshot]) -> Tuple[float, int]:
 """Calculate maximum drawdown and its duration"""
 if len(portfolio_data) < 2:
 return 0.0, 0

 values = [snap.total_value for snap in portfolio_data]
 peak = values[0]
 max_drawdown = 0.0
 max_duration = 0
 current_duration = 0

 for value in values:
 if value > peak:
 peak = value
 current_duration = 0
 else:
 drawdown = (peak - value) / peak
 max_drawdown = max(max_drawdown, drawdown)
 current_duration += 1
 max_duration = max(max_duration, current_duration)

 return max_drawdown, max_duration

 def _calculate_consecutive_periods(self, returns: List[float]) -> Tuple[int, int]:
 """Calculate maximum consecutive winning and losing periods"""
 if not returns:
 return 0, 0

 max_consecutive_wins = 0
 max_consecutive_losses = 0
 current_wins = 0
 current_losses = 0

 for ret in returns:
 if ret > 0:
 current_wins += 1
 current_losses = 0
 max_consecutive_wins = max(max_consecutive_wins, current_wins)
 elif ret < 0:
 current_losses += 1
 current_wins = 0
 max_consecutive_losses = max(max_consecutive_losses, current_losses)
 else:
 current_wins = 0
 current_losses = 0

 return max_consecutive_wins, max_consecutive_losses

 def _get_periods_per_year(self, period: str) -> int:
 """Get number of periods per year for annualization"""
 if period == '1d':
 return 252 # Trading days
 elif period == '1w':
 return 52
 elif period == '1m':
 return 12
 else:
 return 252 # Default to daily

class AttributionEngine:
 """Performance attribution analysis"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize attribution engine"""
 self.logger.info("Initializing Attribution Engine")

 async def calculate_attribution(self, pnl_history: List[PnLAttribution],
 attribution_type: AttributionType,
 period: str) -> AttributionAnalysis:
 """Calculate performance attribution analysis"""
 if not pnl_history:
 return self._empty_attribution(attribution_type, period)

 # Filter for period
 period_pnl = self._filter_period_pnl(pnl_history, period)
 if not period_pnl:
 return self._empty_attribution(attribution_type, period)

 # Calculate attribution based on type
 if attribution_type == AttributionType.BY_STRATEGY:
 return await self._calculate_strategy_attribution(period_pnl, period)
 elif attribution_type == AttributionType.BY_SYMBOL:
 return await self._calculate_symbol_attribution(period_pnl, period)
 elif attribution_type == AttributionType.BY_GREEKS:
 return await self._calculate_greeks_attribution(period_pnl, period)
 else:
 return self._empty_attribution(attribution_type, period)

 async def _calculate_strategy_attribution(self, pnl_history: List[PnLAttribution],
 period: str) -> AttributionAnalysis:
 """Calculate strategy attribution"""
 strategy_contributions = defaultdict(float)

 for pnl in pnl_history:
 for strategy, contribution in pnl.strategy_attribution.items():
 strategy_contributions[strategy] += contribution

 total_pnl = sum(strategy_contributions.values())

 # Calculate percentages
 percentage_contributions = {}
 if total_pnl != 0:
 percentage_contributions = {
 strategy: (pnl / total_pnl) * 100
 for strategy, pnl in strategy_contributions.items()
 }

 return AttributionAnalysis(
 attribution_type=AttributionType.BY_STRATEGY,
 period_start=pnl_history[0].timestamp,
 period_end=pnl_history[-1].timestamp,
 total_pnl=total_pnl,
 contributions=dict(strategy_contributions),
 percentage_contributions=percentage_contributions,
 risk_contributions={}, # Would calculate risk contribution
 return_contributions=dict(strategy_contributions)
 )

 async def _calculate_symbol_attribution(self, pnl_history: List[PnLAttribution],
 period: str) -> AttributionAnalysis:
 """Calculate symbol attribution"""
 symbol_contributions = defaultdict(float)

 for pnl in pnl_history:
 for symbol, contribution in pnl.symbol_attribution.items():
 symbol_contributions[symbol] += contribution

 total_pnl = sum(symbol_contributions.values())

 percentage_contributions = {}
 if total_pnl != 0:
 percentage_contributions = {
 symbol: (pnl / total_pnl) * 100
 for symbol, pnl in symbol_contributions.items()
 }

 return AttributionAnalysis(
 attribution_type=AttributionType.BY_SYMBOL,
 period_start=pnl_history[0].timestamp,
 period_end=pnl_history[-1].timestamp,
 total_pnl=total_pnl,
 contributions=dict(symbol_contributions),
 percentage_contributions=percentage_contributions,
 risk_contributions={},
 return_contributions=dict(symbol_contributions)
 )

 async def _calculate_greeks_attribution(self, pnl_history: List[PnLAttribution],
 period: str) -> AttributionAnalysis:
 """Calculate Greeks attribution"""
 greeks_contributions = {
 'delta': sum(pnl.delta_pnl for pnl in pnl_history),
 'gamma': sum(pnl.gamma_pnl for pnl in pnl_history),
 'theta': sum(pnl.theta_pnl for pnl in pnl_history),
 'vega': sum(pnl.vega_pnl for pnl in pnl_history),
 'rho': sum(pnl.rho_pnl for pnl in pnl_history)
 }

 total_pnl = sum(greeks_contributions.values())

 percentage_contributions = {}
 if total_pnl != 0:
 percentage_contributions = {
 greek: (pnl / total_pnl) * 100
 for greek, pnl in greeks_contributions.items()
 }

 return AttributionAnalysis(
 attribution_type=AttributionType.BY_GREEKS,
 period_start=pnl_history[0].timestamp,
 period_end=pnl_history[-1].timestamp,
 total_pnl=total_pnl,
 contributions=greeks_contributions,
 percentage_contributions=percentage_contributions,
 risk_contributions={},
 return_contributions=greeks_contributions
 )

 def _filter_period_pnl(self, pnl_history: List[PnLAttribution],
 period: str) -> List[PnLAttribution]:
 """Filter P&L history for specified period"""
 if not pnl_history:
 return []

 end_time = datetime.now()

 if period == '1d':
 start_time = end_time - timedelta(days=1)
 elif period == '1w':
 start_time = end_time - timedelta(weeks=1)
 elif period == '1m':
 start_time = end_time - timedelta(days=30)
 elif period == '3m':
 start_time = end_time - timedelta(days=90)
 elif period == '6m':
 start_time = end_time - timedelta(days=180)
 elif period == '1y':
 start_time = end_time - timedelta(days=365)
 else:
 start_time = end_time - timedelta(days=30)

 return [pnl for pnl in pnl_history if pnl.timestamp >= start_time]

 def _empty_attribution(self, attribution_type: AttributionType,
 period: str) -> AttributionAnalysis:
 """Return empty attribution analysis"""
 now = datetime.now()
 return AttributionAnalysis(
 attribution_type=attribution_type,
 period_start=now,
 period_end=now,
 total_pnl=0.0,
 contributions={},
 percentage_contributions={},
 risk_contributions={},
 return_contributions={}
 )

class RiskAnalyzer:
 """Risk metrics analysis"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize risk analyzer"""
 self.logger.info("Initializing Risk Analyzer")

 async def calculate_risk_metrics(self, positions: Dict[str, Position],
 portfolio_history: List[PortfolioSnapshot]) -> RiskMetrics:
 """Calculate comprehensive risk metrics"""
 if not portfolio_history:
 return self._empty_risk_metrics()

 latest_snapshot = portfolio_history[-1]

 # Portfolio VaR (from latest snapshot)
 portfolio_var = latest_snapshot.var_95

 # Calculate CVaR (Expected Shortfall)
 returns = self._calculate_returns_from_history(portfolio_history)
 if returns:
 var_threshold = np.percentile(returns, 5)
 portfolio_cvar = np.mean([r for r in returns if r <= var_threshold])
 else:
 portfolio_cvar = 0.0

 # Leverage ratio
 leverage_ratio = latest_snapshot.total_value / max(abs(latest_snapshot.total_value), 1.0)

 # Concentration risk (simplified)
 concentration_risk = 0.3 # Would calculate from position sizes

 # Greeks risk
 greeks_risk = {
 'delta_risk': abs(latest_snapshot.total_delta) / 1000, # Normalized
 'gamma_risk': abs(latest_snapshot.total_gamma) / 500,
 'theta_risk': abs(latest_snapshot.total_theta) / 200,
 'vega_risk': abs(latest_snapshot.total_vega) / 2000
 }

 # Stress test results (simplified)
 stress_test_results = {
 'market_crash': -0.15,
 'vol_spike': -0.08,
 'interest_rate_shock': -0.05
 }

 return RiskMetrics(
 timestamp=datetime.now(),
 portfolio_var=portfolio_var,
 portfolio_cvar=portfolio_cvar,
 leverage_ratio=leverage_ratio,
 concentration_risk=concentration_risk,
 correlation_risk=0.1, # Would calculate from correlations
 volatility_risk=0.2, # Would calculate from vol exposure
 greeks_risk=greeks_risk,
 stress_test_results=stress_test_results
 )

 def _calculate_returns_from_history(self, portfolio_history: List[PortfolioSnapshot]) -> List[float]:
 """Calculate returns from portfolio history"""
 if len(portfolio_history) < 2:
 return []

 returns = []
 for i in range(1, len(portfolio_history)):
 prev_value = portfolio_history[i-1].total_value
 curr_value = portfolio_history[i].total_value

 if prev_value != 0:
 ret = (curr_value - prev_value) / prev_value
 returns.append(ret)

 return returns

 def _empty_risk_metrics(self) -> RiskMetrics:
 """Return empty risk metrics"""
 return RiskMetrics(
 timestamp=datetime.now(),
 portfolio_var=0.0,
 portfolio_cvar=0.0,
 leverage_ratio=1.0,
 concentration_risk=0.0,
 correlation_risk=0.0,
 volatility_risk=0.0,
 greeks_risk={},
 stress_test_results={}
 )

class BenchmarkAnalyzer:
 """Benchmark comparison analysis"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize benchmark analyzer"""
 self.logger.info("Initializing Benchmark Analyzer")

 async def compare_to_benchmark(self, portfolio_history: List[PortfolioSnapshot],
 period: str) -> Dict[str, Any]:
 """Compare portfolio performance to benchmark"""
 if len(portfolio_history) < 2:
 return {}

 # Calculate portfolio returns
 portfolio_returns = self._calculate_portfolio_returns(portfolio_history)

 # Generate benchmark returns (mock data)
 benchmark_returns = self._generate_benchmark_returns(len(portfolio_returns))

 if not portfolio_returns or not benchmark_returns:
 return {}

 # Calculate metrics
 portfolio_total_return = np.prod([1 + r for r in portfolio_returns]) - 1
 benchmark_total_return = np.prod([1 + r for r in benchmark_returns]) - 1

 excess_return = portfolio_total_return - benchmark_total_return

 # Calculate tracking error
 excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
 tracking_error = np.std(excess_returns) * np.sqrt(252)

 # Information ratio
 information_ratio = np.mean(excess_returns) * 252 / max(tracking_error, 0.01)

 # Beta calculation
 if np.var(benchmark_returns) > 0:
 beta = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)
 else:
 beta = 1.0

 # Alpha calculation
 risk_free_rate = 0.02
 alpha = (np.mean(portfolio_returns) * 252 - risk_free_rate) - beta * (np.mean(benchmark_returns) * 252 - risk_free_rate)

 return {
 'benchmark_symbol': self.config.get('benchmark_symbol', 'SPY'),
 'portfolio_return': portfolio_total_return,
 'benchmark_return': benchmark_total_return,
 'excess_return': excess_return,
 'tracking_error': tracking_error,
 'information_ratio': information_ratio,
 'beta': beta,
 'alpha': alpha,
 'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0][1] if len(portfolio_returns) > 1 else 0,
 'outperformance_ratio': len([r for r in excess_returns if r > 0]) / len(excess_returns) if excess_returns else 0
 }

 def _calculate_portfolio_returns(self, portfolio_history: List[PortfolioSnapshot]) -> List[float]:
 """Calculate portfolio returns"""
 returns = []
 for i in range(1, len(portfolio_history)):
 prev_value = portfolio_history[i-1].total_value
 curr_value = portfolio_history[i].total_value

 if prev_value != 0:
 ret = (curr_value - prev_value) / prev_value
 returns.append(ret)

 return returns

 def _generate_benchmark_returns(self, n_periods: int) -> List[float]:
 """Generate mock benchmark returns for comparison"""
 # Mock benchmark returns (normal distribution)
 daily_vol = 0.16 / np.sqrt(252) # 16% annual vol
 daily_return = 0.10 / 252 # 10% annual return

 returns = np.random.normal(daily_return, daily_vol, n_periods)
 return returns.tolist()