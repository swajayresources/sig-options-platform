"""
Professional Options Backtesting Engine
Comprehensive backtesting system with realistic market simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from enum import Enum
import math
from scipy.stats import norm
import logging

class OrderType(Enum):
 BUY = "BUY"
 SELL = "SELL"

class OptionType(Enum):
 CALL = "CALL"
 PUT = "PUT"

@dataclass
class OptionContract:
 symbol: str
 strike: float
 expiry: datetime
 option_type: OptionType

@dataclass
class MarketData:
 timestamp: datetime
 underlying_price: float
 option_prices: Dict[str, float] # option_id -> price
 bid_ask_spreads: Dict[str, Tuple[float, float]] # option_id -> (bid, ask)
 volatilities: Dict[str, float] # option_id -> implied vol
 volumes: Dict[str, int] # option_id -> volume
 open_interest: Dict[str, int] # option_id -> open interest

@dataclass
class Order:
 contract: OptionContract
 quantity: int
 order_type: OrderType
 limit_price: Optional[float] = None
 timestamp: Optional[datetime] = None

@dataclass
class Trade:
 contract: OptionContract
 quantity: int
 price: float
 timestamp: datetime
 transaction_cost: float
 slippage: float

@dataclass
class Position:
 contract: OptionContract
 quantity: int
 avg_price: float
 timestamp: datetime

class BacktestingEngine:
 """
 Professional options backtesting engine with realistic market simulation
 """

 def __init__(self, initial_capital: float = 1000000.0):
 self.initial_capital = initial_capital
 self.current_capital = initial_capital
 self.positions: Dict[str, Position] = {}
 self.trades: List[Trade] = []
 self.equity_curve: List[Tuple[datetime, float]] = []
 self.greeks_history: List[Dict[str, float]] = []

 # Market simulation parameters
 self.bid_ask_spread_pct = 0.02 # 2% bid-ask spread
 self.slippage_pct = 0.005 # 0.5% slippage on market orders
 self.commission_per_contract = 1.0 # $1 per contract
 self.market_impact_factor = 0.001 # Additional cost for large orders

 # Risk management
 self.max_position_size = 0.1 # 10% of capital per position
 self.max_portfolio_delta = 1000
 self.max_portfolio_gamma = 500
 self.max_portfolio_vega = 10000

 logging.basicConfig(level=logging.INFO)
 self.logger = logging.getLogger(__name__)

 def load_historical_data(self, start_date: datetime, end_date: datetime) -> List[MarketData]:
 """
 Load historical options data for backtesting
 In production, this would connect to a historical data provider
 """
 market_data = []
 current_date = start_date

 # Generate synthetic historical data for demonstration
 base_price = 100.0
 volatility = 0.25

 while current_date <= end_date:
 # Simulate underlying price movement
 daily_return = np.random.normal(0, volatility / np.sqrt(252))
 base_price *= (1 + daily_return)

 # Generate option contracts
 strikes = [base_price * k for k in [0.9, 0.95, 1.0, 1.05, 1.1]]
 expiries = [current_date + timedelta(days=d) for d in [30, 60, 90]]

 option_prices = {}
 bid_ask_spreads = {}
 volatilities = {}
 volumes = {}
 open_interest = {}

 for strike in strikes:
 for expiry in expiries:
 for opt_type in [OptionType.CALL, OptionType.PUT]:
 option_id = f"SPY_{expiry.strftime('%y%m%d')}{opt_type.value[0]}{strike:.0f}"

 # Calculate theoretical price
 time_to_expiry = (expiry - current_date).days / 365.0
 if time_to_expiry > 0:
 price = self._black_scholes_price(
 base_price, strike, volatility, time_to_expiry, opt_type
 )

 # Add realistic bid-ask spread
 spread = price * self.bid_ask_spread_pct
 bid = price - spread / 2
 ask = price + spread / 2

 option_prices[option_id] = price
 bid_ask_spreads[option_id] = (bid, ask)
 volatilities[option_id] = volatility + np.random.normal(0, 0.05)
 volumes[option_id] = np.random.randint(10, 1000)
 open_interest[option_id] = np.random.randint(100, 10000)

 market_data.append(MarketData(
 timestamp=current_date,
 underlying_price=base_price,
 option_prices=option_prices,
 bid_ask_spreads=bid_ask_spreads,
 volatilities=volatilities,
 volumes=volumes,
 open_interest=open_interest
 ))

 current_date += timedelta(days=1)

 return market_data

 def _black_scholes_price(self, spot: float, strike: float, vol: float,
 time_to_expiry: float, option_type: OptionType,
 risk_free_rate: float = 0.05) -> float:
 """Calculate Black-Scholes option price"""
 if time_to_expiry <= 0:
 if option_type == OptionType.CALL:
 return max(0, spot - strike)
 else:
 return max(0, strike - spot)

 d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * vol ** 2) * time_to_expiry) / (vol * math.sqrt(time_to_expiry))
 d2 = d1 - vol * math.sqrt(time_to_expiry)

 if option_type == OptionType.CALL:
 price = spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
 else:
 price = strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

 return max(0, price)

 def calculate_greeks(self, spot: float, strike: float, vol: float,
 time_to_expiry: float, option_type: OptionType,
 risk_free_rate: float = 0.05) -> Dict[str, float]:
 """Calculate option Greeks"""
 if time_to_expiry <= 0:
 return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

 d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * vol ** 2) * time_to_expiry) / (vol * math.sqrt(time_to_expiry))
 d2 = d1 - vol * math.sqrt(time_to_expiry)

 # Delta
 if option_type == OptionType.CALL:
 delta = norm.cdf(d1)
 else:
 delta = norm.cdf(d1) - 1

 # Gamma
 gamma = norm.pdf(d1) / (spot * vol * math.sqrt(time_to_expiry))

 # Theta
 theta_common = -(spot * norm.pdf(d1) * vol) / (2 * math.sqrt(time_to_expiry))
 if option_type == OptionType.CALL:
 theta = theta_common - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
 else:
 theta = theta_common + risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
 theta /= 365 # Convert to daily

 # Vega
 vega = spot * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100

 # Rho
 if option_type == OptionType.CALL:
 rho = strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
 else:
 rho = -strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100

 return {
 'delta': delta,
 'gamma': gamma,
 'theta': theta,
 'vega': vega,
 'rho': rho
 }

 def execute_order(self, order: Order, market_data: MarketData) -> Optional[Trade]:
 """
 Execute an order with realistic market simulation
 """
 option_id = f"{order.contract.symbol}_{order.contract.expiry.strftime('%y%m%d')}{order.contract.option_type.value[0]}{order.contract.strike:.0f}"

 if option_id not in market_data.option_prices:
 self.logger.warning(f"Option {option_id} not found in market data")
 return None

 # Get market prices
 bid, ask = market_data.bid_ask_spreads.get(option_id, (0, 0))
 mid_price = market_data.option_prices[option_id]

 # Determine execution price
 if order.order_type == OrderType.BUY:
 if order.limit_price is None or order.limit_price >= ask:
 execution_price = ask
 else:
 return None # Order not filled
 else: # SELL
 if order.limit_price is None or order.limit_price <= bid:
 execution_price = bid
 else:
 return None # Order not filled

 # Calculate transaction costs
 commission = abs(order.quantity) * self.commission_per_contract

 # Market impact for large orders
 volume = market_data.volumes.get(option_id, 100)
 market_impact = 0
 if abs(order.quantity) > volume * 0.1: # Large order
 market_impact = execution_price * self.market_impact_factor * (abs(order.quantity) / volume)

 # Slippage for market orders
 slippage = 0
 if order.limit_price is None:
 slippage = execution_price * self.slippage_pct

 total_transaction_cost = commission + market_impact + slippage

 # Check capital requirements
 trade_value = abs(order.quantity) * execution_price * 100 # Options are quoted per share, 100 shares per contract
 if trade_value > self.current_capital * self.max_position_size:
 self.logger.warning(f"Order exceeds position size limit: {trade_value}")
 return None

 # Create trade
 trade = Trade(
 contract=order.contract,
 quantity=order.quantity,
 price=execution_price,
 timestamp=market_data.timestamp,
 transaction_cost=total_transaction_cost,
 slippage=slippage
 )

 # Update positions
 self._update_positions(trade)

 # Update capital
 if order.order_type == OrderType.BUY:
 self.current_capital -= (trade_value + total_transaction_cost)
 else:
 self.current_capital += (trade_value - total_transaction_cost)

 self.trades.append(trade)
 return trade

 def _update_positions(self, trade: Trade):
 """Update position tracking"""
 option_id = f"{trade.contract.symbol}_{trade.contract.expiry.strftime('%y%m%d')}{trade.contract.option_type.value[0]}{trade.contract.strike:.0f}"

 if option_id in self.positions:
 existing_pos = self.positions[option_id]
 total_quantity = existing_pos.quantity + trade.quantity

 if total_quantity == 0:
 del self.positions[option_id]
 else:
 # Update average price
 total_cost = (existing_pos.quantity * existing_pos.avg_price +
 trade.quantity * trade.price)
 avg_price = total_cost / total_quantity

 self.positions[option_id] = Position(
 contract=trade.contract,
 quantity=total_quantity,
 avg_price=avg_price,
 timestamp=trade.timestamp
 )
 else:
 if trade.quantity != 0:
 self.positions[option_id] = Position(
 contract=trade.contract,
 quantity=trade.quantity,
 avg_price=trade.price,
 timestamp=trade.timestamp
 )

 def calculate_portfolio_value(self, market_data: MarketData) -> Dict[str, float]:
 """Calculate current portfolio value and Greeks"""
 total_value = self.current_capital
 total_delta = 0
 total_gamma = 0
 total_theta = 0
 total_vega = 0
 total_rho = 0

 for option_id, position in self.positions.items():
 # Get current market price
 current_price = market_data.option_prices.get(option_id, 0)
 position_value = position.quantity * current_price * 100
 total_value += position_value

 # Calculate Greeks
 time_to_expiry = (position.contract.expiry - market_data.timestamp).days / 365.0
 if time_to_expiry > 0:
 vol = market_data.volatilities.get(option_id, 0.25)
 greeks = self.calculate_greeks(
 market_data.underlying_price,
 position.contract.strike,
 vol,
 time_to_expiry,
 position.contract.option_type
 )

 total_delta += position.quantity * greeks['delta'] * 100
 total_gamma += position.quantity * greeks['gamma'] * 100
 total_theta += position.quantity * greeks['theta'] * 100
 total_vega += position.quantity * greeks['vega'] * 100
 total_rho += position.quantity * greeks['rho'] * 100

 return {
 'total_value': total_value,
 'cash': self.current_capital,
 'delta': total_delta,
 'gamma': total_gamma,
 'theta': total_theta,
 'vega': total_vega,
 'rho': total_rho,
 'pnl': total_value - self.initial_capital,
 'return_pct': (total_value - self.initial_capital) / self.initial_capital * 100
 }

 def check_risk_limits(self, portfolio_metrics: Dict[str, float]) -> List[str]:
 """Check if portfolio exceeds risk limits"""
 violations = []

 if abs(portfolio_metrics['delta']) > self.max_portfolio_delta:
 violations.append(f"Delta limit exceeded: {portfolio_metrics['delta']:.2f}")

 if abs(portfolio_metrics['gamma']) > self.max_portfolio_gamma:
 violations.append(f"Gamma limit exceeded: {portfolio_metrics['gamma']:.2f}")

 if abs(portfolio_metrics['vega']) > self.max_portfolio_vega:
 violations.append(f"Vega limit exceeded: {portfolio_metrics['vega']:.2f}")

 return violations

 def run_backtest(self, strategy_function, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
 """
 Run a complete backtest with a given strategy function
 """
 self.logger.info(f"Starting backtest from {start_date} to {end_date}")

 # Load historical data
 market_data_history = self.load_historical_data(start_date, end_date)

 results = {
 'trades': [],
 'equity_curve': [],
 'greeks_history': [],
 'risk_violations': [],
 'performance_metrics': {}
 }

 for market_data in market_data_history:
 # Run strategy
 orders = strategy_function(market_data, self.positions, self.current_capital)

 # Execute orders
 for order in orders:
 trade = self.execute_order(order, market_data)
 if trade:
 results['trades'].append(trade)

 # Calculate portfolio metrics
 portfolio_metrics = self.calculate_portfolio_value(market_data)
 self.equity_curve.append((market_data.timestamp, portfolio_metrics['total_value']))
 self.greeks_history.append(portfolio_metrics)

 # Check risk limits
 violations = self.check_risk_limits(portfolio_metrics)
 if violations:
 results['risk_violations'].extend(violations)

 results['equity_curve'].append(portfolio_metrics)

 # Calculate final performance metrics
 results['performance_metrics'] = self._calculate_performance_metrics()

 self.logger.info(f"Backtest completed. Final P&L: {results['performance_metrics']['total_return']:.2f}%")

 return results

 def _calculate_performance_metrics(self) -> Dict[str, float]:
 """Calculate comprehensive performance metrics"""
 if len(self.equity_curve) < 2:
 return {}

 # Extract values
 values = [point[1] for point in self.equity_curve]
 returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

 total_return = (values[-1] - values[0]) / values[0] * 100
 volatility = np.std(returns) * np.sqrt(252) * 100 # Annualized

 # Sharpe ratio (assuming 2% risk-free rate)
 risk_free_rate = 0.02
 excess_returns = [r - risk_free_rate/252 for r in returns]
 sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

 # Maximum drawdown
 peak = values[0]
 max_drawdown = 0
 for value in values:
 if value > peak:
 peak = value
 drawdown = (peak - value) / peak * 100
 max_drawdown = max(max_drawdown, drawdown)

 # Win rate
 winning_trades = sum(1 for trade in self.trades if trade.quantity > 0) # Simplified
 win_rate = winning_trades / len(self.trades) * 100 if self.trades else 0

 return {
 'total_return': total_return,
 'volatility': volatility,
 'sharpe_ratio': sharpe_ratio,
 'max_drawdown': max_drawdown,
 'win_rate': win_rate,
 'total_trades': len(self.trades),
 'final_capital': values[-1]
 }