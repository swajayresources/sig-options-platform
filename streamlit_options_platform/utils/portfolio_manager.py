"""
Portfolio Manager
Portfolio tracking, performance analysis, and risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass, asdict

@dataclass
class Position:
 symbol: str
 quantity: int
 entry_price: float
 current_price: float
 entry_date: datetime
 option_type: str
 strike: float
 expiration: datetime
 strategy: str

@dataclass
class PortfolioMetrics:
 total_value: float
 total_pnl: float
 unrealized_pnl: float
 realized_pnl: float
 daily_pnl: float
 total_delta: float
 total_gamma: float
 total_theta: float
 total_vega: float
 position_count: int
 largest_position: float
 concentration_risk: float

class PortfolioManager:
 """Manages portfolio positions, performance, and risk metrics"""

 def __init__(self):
 self.positions = self._generate_demo_positions()
 self.historical_data = self._generate_historical_data()
 self.performance_history = self._generate_performance_history()
 self.risk_limits = {
 'max_delta': 1000,
 'max_gamma': 500,
 'max_theta': -200,
 'max_vega': 2000,
 'max_position_size': 100000,
 'max_concentration': 0.25
 }

 def get_portfolio_summary(self) -> Dict[str, Any]:
 """Get high-level portfolio summary"""
 metrics = self._calculate_portfolio_metrics()

 return {
 'total_value': metrics.total_value,
 'total_pnl': metrics.total_pnl,
 'daily_pnl': metrics.daily_pnl,
 'position_count': metrics.position_count,
 'total_delta': metrics.total_delta,
 'total_gamma': metrics.total_gamma,
 'total_theta': metrics.total_theta,
 'total_vega': metrics.total_vega,
 'last_update': datetime.now()
 }

 def get_portfolio_greeks(self) -> Dict[str, float]:
 """Get aggregated portfolio Greeks"""
 total_delta = 0
 total_gamma = 0
 total_theta = 0
 total_vega = 0

 for position in self.positions:
 # Calculate position Greeks (simplified)
 greeks = self._calculate_position_greeks(position)

 total_delta += greeks['delta'] * position.quantity
 total_gamma += greeks['gamma'] * position.quantity
 total_theta += greeks['theta'] * position.quantity
 total_vega += greeks['vega'] * position.quantity

 return {
 'delta': total_delta,
 'gamma': total_gamma,
 'theta': total_theta,
 'vega': total_vega
 }

 def get_portfolio_composition(self) -> Dict[str, float]:
 """Get portfolio composition by symbol"""
 composition = {}
 total_value = 0

 for position in self.positions:
 market_value = abs(position.quantity * position.current_price * 100) # Options are in 100s
 symbol = position.symbol.split('_')[0] # Extract underlying symbol

 if symbol not in composition:
 composition[symbol] = 0

 composition[symbol] += market_value
 total_value += market_value

 # Convert to percentages
 if total_value > 0:
 composition = {k: v / total_value * 100 for k, v in composition.items()}

 return composition

 def get_pnl_attribution(self) -> Dict[str, float]:
 """Get P&L attribution by Greeks"""
 # Simulate daily P&L attribution
 return {
 'Delta P&L': np.random.uniform(-2000, 3000),
 'Gamma P&L': np.random.uniform(-1000, 1500),
 'Theta P&L': np.random.uniform(-800, -200),
 'Vega P&L': np.random.uniform(-1500, 2000),
 'Other': np.random.uniform(-500, 500)
 }

 def get_positions_dataframe(self) -> pd.DataFrame:
 """Get positions as DataFrame for display"""
 positions_data = []

 for position in self.positions:
 market_value = position.quantity * position.current_price * 100
 unrealized_pnl = (position.current_price - position.entry_price) * position.quantity * 100

 positions_data.append({
 'Symbol': position.symbol,
 'Strategy': position.strategy,
 'Type': position.option_type,
 'Strike': position.strike,
 'Expiration': position.expiration.strftime('%Y-%m-%d'),
 'Quantity': position.quantity,
 'Entry_Price': f"${position.entry_price:.2f}",
 'Current_Price': f"${position.current_price:.2f}",
 'Market_Value': f"${market_value:,.0f}",
 'Unrealized_PnL': unrealized_pnl,
 'Days_to_Exp': (position.expiration - datetime.now()).days
 })

 return pd.DataFrame(positions_data)

 def get_greeks_history(self) -> Dict[str, List]:
 """Get historical Greeks data"""
 dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')

 # Generate synthetic Greeks history
 base_delta = 150
 base_gamma = 75
 base_theta = -300
 base_vega = 1200

 history = {
 'timestamp': dates,
 'delta': [base_delta + np.random.normal(0, 50) for _ in dates],
 'gamma': [base_gamma + np.random.normal(0, 25) for _ in dates],
 'theta': [base_theta + np.random.normal(0, 100) for _ in dates],
 'vega': [base_vega + np.random.normal(0, 300) for _ in dates]
 }

 return history

 def get_available_strategies(self) -> List[str]:
 """Get list of available strategies"""
 return ['Delta Neutral', 'Iron Condor', 'Straddle', 'Butterfly', 'Calendar Spread', 'Covered Call']

 def get_strategy_performance(self, strategy: str) -> Dict[str, Any]:
 """Get performance data for specific strategy"""
 # Generate synthetic performance data
 dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
 returns = np.random.normal(0.001, 0.02, len(dates)) # Daily returns
 cumulative_pnl = np.cumsum(returns) * 10000 # Scale to dollar amounts

 # Monthly returns
 monthly_dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
 monthly_returns = np.random.normal(0.02, 0.05, len(monthly_dates))

 performance_data = {
 'total_return': np.random.uniform(0.08, 0.25),
 'period_return': np.random.uniform(-0.02, 0.05),
 'sharpe_ratio': np.random.uniform(0.8, 2.5),
 'sharpe_change': np.random.uniform(-0.2, 0.3),
 'max_drawdown': np.random.uniform(-0.15, -0.05),
 'dd_change': np.random.uniform(-0.02, 0.01),
 'win_rate': np.random.uniform(0.55, 0.75),
 'win_rate_change': np.random.uniform(-0.05, 0.05),
 'pnl_history': {
 'date': dates,
 'cumulative_pnl': cumulative_pnl
 },
 'monthly_returns': {
 'month': [d.strftime('%Y-%m') for d in monthly_dates],
 'return': monthly_returns * 100 # Convert to percentage
 },
 'risk_attribution': {
 'Market Risk': 5000,
 'Volatility Risk': 2000,
 'Time Decay': -1500,
 'Interest Rate': 500,
 'Other': -500,
 'Total': 5500
 }
 }

 return performance_data

 def get_strategy_details(self, strategy: str) -> Dict[str, Any]:
 """Get detailed strategy information"""
 strategy_configs = {
 'Delta Neutral': {
 'parameters': {
 'Target Delta': '±50',
 'Rebalance Threshold': '±100',
 'Max Position Size': '$50,000',
 'Volatility Filter': '15-45%'
 },
 'statistics': {
 'Avg Daily P&L': '$245',
 'Win Rate': '68%',
 'Max Daily Loss': '-$1,250',
 'Sharpe Ratio': '1.85'
 }
 },
 'Iron Condor': {
 'parameters': {
 'Wing Width': '10 points',
 'Target DTE': '30-45 days',
 'Profit Target': '50% max profit',
 'Stop Loss': '200% premium'
 },
 'statistics': {
 'Avg Trade P&L': '$125',
 'Win Rate': '72%',
 'Avg Trade Duration': '21 days',
 'Monthly Return': '3.2%'
 }
 }
 }

 return strategy_configs.get(strategy, {
 'parameters': {'Strategy': strategy},
 'statistics': {'Status': 'Active'}
 })

 def get_risk_metrics(self) -> Dict[str, float]:
 """Get portfolio risk metrics"""
 return {
 'var_95': np.random.uniform(15000, 45000),
 'max_drawdown': np.random.uniform(0.08, 0.18),
 'leverage_ratio': np.random.uniform(1.2, 2.8),
 'concentration_risk': np.random.uniform(0.15, 0.35),
 'correlation_risk': np.random.uniform(0.2, 0.6),
 'volatility_risk': np.random.uniform(0.1, 0.4)
 }

 def get_risk_limits(self) -> Dict[str, float]:
 """Get risk limits configuration"""
 return self.risk_limits.copy()

 def get_current_exposure(self) -> Dict[str, float]:
 """Get current risk exposure"""
 greeks = self.get_portfolio_greeks()
 return {
 'delta': greeks['delta'],
 'gamma': greeks['gamma'],
 'theta': greeks['theta'],
 'vega': greeks['vega']
 }

 def get_stress_test_results(self) -> Dict[str, float]:
 """Get stress test scenario results"""
 return {
 'Market Crash (-20%)': np.random.uniform(-25000, -15000),
 'Vol Spike (+10 pts)': np.random.uniform(-8000, 12000),
 'Rate Shock (+2%)': np.random.uniform(-5000, 3000),
 'Time Decay (7 days)': np.random.uniform(-3000, -1000),
 'Correlation Breakdown': np.random.uniform(-12000, -5000)
 }

 def get_risk_decomposition(self) -> Dict[str, float]:
 """Get risk breakdown by component"""
 return {
 'Directional Risk': 35,
 'Volatility Risk': 25,
 'Time Decay Risk': 15,
 'Interest Rate Risk': 8,
 'Correlation Risk': 12,
 'Other': 5
 }

 def get_risk_history(self) -> Dict[str, List]:
 """Get historical risk metrics"""
 dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')

 base_var = 25000
 base_es = 35000

 history = {
 'date': dates,
 'var_95': [base_var + np.random.normal(0, 5000) for _ in dates],
 'expected_shortfall': [base_es + np.random.normal(0, 7000) for _ in dates]
 }

 return history

 def _generate_demo_positions(self) -> List[Position]:
 """Generate demo portfolio positions"""
 positions = []
 symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
 strategies = ['Delta Neutral', 'Iron Condor', 'Straddle', 'Butterfly', 'Calendar Spread']
 option_types = ['CALL', 'PUT']

 for i in range(15): # Generate 15 positions
 symbol = np.random.choice(symbols)
 strategy = np.random.choice(strategies)
 option_type = np.random.choice(option_types)

 base_price = np.random.uniform(150, 400)
 strike = base_price * np.random.uniform(0.95, 1.05)

 position = Position(
 symbol=f"{symbol}_{(datetime.now() + timedelta(days=np.random.randint(7, 90))).strftime('%y%m%d')}{option_type[0]}{int(strike)}",
 quantity=np.random.randint(-10, 10),
 entry_price=np.random.uniform(2, 15),
 current_price=np.random.uniform(1, 20),
 entry_date=datetime.now() - timedelta(days=np.random.randint(1, 30)),
 option_type=option_type,
 strike=strike,
 expiration=datetime.now() + timedelta(days=np.random.randint(7, 90)),
 strategy=strategy
 )
 positions.append(position)

 return positions

 def _calculate_portfolio_metrics(self) -> PortfolioMetrics:
 """Calculate comprehensive portfolio metrics"""
 total_value = 0
 total_pnl = 0
 unrealized_pnl = 0
 position_values = []

 greeks = self.get_portfolio_greeks()

 for position in self.positions:
 market_value = position.quantity * position.current_price * 100
 position_pnl = (position.current_price - position.entry_price) * position.quantity * 100

 total_value += market_value
 total_pnl += position_pnl
 unrealized_pnl += position_pnl
 position_values.append(abs(market_value))

 largest_position = max(position_values) if position_values else 0
 concentration_risk = largest_position / max(sum(position_values), 1)

 return PortfolioMetrics(
 total_value=total_value,
 total_pnl=total_pnl,
 unrealized_pnl=unrealized_pnl,
 realized_pnl=0, # Would track from closed positions
 daily_pnl=np.random.uniform(-2000, 3000),
 total_delta=greeks['delta'],
 total_gamma=greeks['gamma'],
 total_theta=greeks['theta'],
 total_vega=greeks['vega'],
 position_count=len(self.positions),
 largest_position=largest_position,
 concentration_risk=concentration_risk
 )

 def _calculate_position_greeks(self, position: Position) -> Dict[str, float]:
 """Calculate Greeks for a position (simplified)"""
 # Simplified Greeks calculation based on position characteristics
 time_to_expiry = (position.expiration - datetime.now()).days / 365

 if time_to_expiry <= 0:
 return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

 # Simple delta calculation
 if position.option_type == 'CALL':
 delta = 0.5 if position.strike == position.current_price else (
 0.8 if position.strike < position.current_price else 0.2
 )
 else: # PUT
 delta = -0.5 if position.strike == position.current_price else (
 -0.8 if position.strike > position.current_price else -0.2
 )

 # Simple Greeks approximation
 gamma = 0.05 / max(time_to_expiry, 0.01)
 theta = -position.current_price * 0.01 / max(time_to_expiry, 0.01)
 vega = position.current_price * 0.1 * max(time_to_expiry, 0.01)

 return {
 'delta': delta,
 'gamma': gamma,
 'theta': theta,
 'vega': vega
 }

 def _generate_historical_data(self) -> Dict[str, Any]:
 """Generate historical portfolio data"""
 # This would typically load from database
 return {}

 def _generate_performance_history(self) -> Dict[str, Any]:
 """Generate performance history"""
 # This would typically load from database
 return {}