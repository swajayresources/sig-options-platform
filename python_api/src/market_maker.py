"""
Market Making Strategy Framework

Sophisticated market making algorithms for options trading with
inventory management, risk control, and dynamic hedging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import logging

from.pricing_engine import (
 PricingEngine, OptionContract, MarketData, Greeks, PricingResult,
 OptionType, PricingModel
)

logger = logging.getLogger(__name__)

class QuoteType(Enum):
 BID = "BID"
 ASK = "ASK"
 BOTH = "BOTH"

@dataclass
class Quote:
 """Market maker quote"""
 symbol: str
 bid_price: float
 ask_price: float
 bid_size: int
 ask_size: int
 timestamp: float
 theoretical_value: float
 edge: float
 confidence: float = 1.0

 @property
 def mid_price(self) -> float:
 return (self.bid_price + self.ask_price) / 2.0

 @property
 def spread(self) -> float:
 return self.ask_price - self.bid_price

 @property
 def spread_bps(self) -> float:
 """Spread in basis points"""
 return (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0

@dataclass
class Position:
 """Position tracking"""
 symbol: str
 quantity: int
 average_price: float
 unrealized_pnl: float = 0.0
 realized_pnl: float = 0.0
 delta: float = 0.0
 gamma: float = 0.0
 theta: float = 0.0
 vega: float = 0.0

@dataclass
class MarketMakerConfig:
 """Market maker configuration"""
 max_position_size: int = 100
 max_portfolio_delta: float = 1000.0
 max_portfolio_gamma: float = 500.0
 max_portfolio_vega: float = 10000.0

 # Spread parameters
 min_spread_bps: float = 10.0 # Minimum spread in basis points
 max_spread_bps: float = 200.0
 inventory_penalty: float = 0.1 # Inventory skew factor

 # Risk management
 max_drawdown: float = 0.05 # 5% max drawdown
 position_limit_multiplier: float = 0.8
 hedge_threshold_delta: float = 100.0
 hedge_threshold_gamma: float = 50.0

 # Pricing
 vol_adjustment_factor: float = 0.02
 edge_target: float = 0.005 # 50bps target edge
 confidence_threshold: float = 0.7

class InventoryManager:
 """Manages options inventory and risk metrics"""

 def __init__(self, config: MarketMakerConfig):
 self.config = config
 self.positions: Dict[str, Position] = {}
 self.total_pnl = 0.0
 self.max_portfolio_value = 0.0

 def add_position(self, symbol: str, quantity: int, price: float, greeks: Greeks):
 """Add or update position"""
 if symbol in self.positions:
 pos = self.positions[symbol]
 total_quantity = pos.quantity + quantity

 if total_quantity != 0:
 pos.average_price = ((pos.average_price * pos.quantity + price * quantity)
 / total_quantity)
 pos.quantity = total_quantity
 else:
 # Position closed
 pos.realized_pnl += (price - pos.average_price) * quantity
 pos.quantity = 0
 pos.average_price = 0
 else:
 self.positions[symbol] = Position(
 symbol=symbol,
 quantity=quantity,
 average_price=price,
 delta=greeks.delta * quantity,
 gamma=greeks.gamma * quantity,
 theta=greeks.theta * quantity,
 vega=greeks.vega * quantity
 )

 def get_portfolio_greeks(self) -> Greeks:
 """Calculate total portfolio Greeks"""
 total_greeks = Greeks()

 for pos in self.positions.values():
 total_greeks.delta += pos.delta
 total_greeks.gamma += pos.gamma
 total_greeks.theta += pos.theta
 total_greeks.vega += pos.vega

 return total_greeks

 def get_inventory_skew(self, symbol: str) -> float:
 """Calculate inventory skew for position sizing"""
 if symbol not in self.positions:
 return 0.0

 position = self.positions[symbol]
 max_pos = self.config.max_position_size

 return position.quantity / max_pos if max_pos > 0 else 0.0

 def should_hedge(self) -> Tuple[bool, str]:
 """Check if portfolio needs hedging"""
 greeks = self.get_portfolio_greeks()

 if abs(greeks.delta) > self.config.hedge_threshold_delta:
 return True, f"Delta exposure: {greeks.delta:.2f}"

 if abs(greeks.gamma) > self.config.hedge_threshold_gamma:
 return True, f"Gamma exposure: {greeks.gamma:.2f}"

 return False, ""

class VolatilityOracle:
 """Volatility surface management and forecasting"""

 def __init__(self):
 self.historical_vols: Dict[str, deque] = {}
 self.realized_vols: Dict[str, float] = {}
 self.vol_forecast_horizon = 30 # days

 def update_realized_vol(self, symbol: str, vol: float):
 """Update realized volatility"""
 self.realized_vols[symbol] = vol

 if symbol not in self.historical_vols:
 self.historical_vols[symbol] = deque(maxlen=252) # 1 year

 self.historical_vols[symbol].append(vol)

 def get_vol_forecast(self, symbol: str, time_horizon: float) -> float:
 """Get volatility forecast for given time horizon"""
 if symbol not in self.realized_vols:
 return 0.2 # Default 20%

 # Simple EWMA forecast
 historical = list(self.historical_vols.get(symbol, []))
 if len(historical) < 10:
 return self.realized_vols[symbol]

 weights = np.exp(-np.arange(len(historical)) * 0.05)
 weights = weights[::-1] / np.sum(weights)

 forecast = np.sum(np.array(historical) * weights)

 # Adjust for time horizon
 vol_adjustment = np.sqrt(time_horizon / (1/252)) # Annualized

 return forecast * vol_adjustment

 def get_vol_term_structure(self, symbol: str) -> Dict[float, float]:
 """Get volatility term structure"""
 base_vol = self.get_vol_forecast(symbol, 1.0)

 # Simple term structure model
 term_structure = {}
 for maturity in [0.25, 0.5, 1.0, 2.0]: # 3M, 6M, 1Y, 2Y
 # Add term structure skew
 skew = -0.02 * np.log(maturity) if maturity > 0.25 else 0
 term_structure[maturity] = base_vol + skew

 return term_structure

class EdgeCalculator:
 """Calculate theoretical edge and confidence for market making"""

 def __init__(self, vol_oracle: VolatilityOracle):
 self.vol_oracle = vol_oracle

 def calculate_edge(self, option: OptionContract, market_data: MarketData,
 market_bid: float, market_ask: float) -> Tuple[float, float]:
 """Calculate edge and confidence for option quote"""

 # Get volatility forecast
 forecast_vol = self.vol_oracle.get_vol_forecast(
 option.underlying, option.expiry
 )

 # Create market data with forecast volatility
 forecast_market_data = MarketData(
 market_data.spot_price,
 market_data.risk_free_rate,
 market_data.dividend_yield,
 forecast_vol,
 market_data.time_to_expiry
 )

 # Calculate theoretical value
 pricing_engine = PricingEngine()
 result = pricing_engine.price_option(option, forecast_market_data)

 if not result.success:
 return 0.0, 0.0

 theoretical = result.price
 market_mid = (market_bid + market_ask) / 2.0

 # Calculate edge
 edge = (theoretical - market_mid) / market_mid if market_mid > 0 else 0

 # Calculate confidence based on vol stability and liquidity
 vol_stability = self._calculate_vol_stability(option.underlying)
 liquidity_score = self._calculate_liquidity_score(market_bid, market_ask)

 confidence = min(vol_stability * liquidity_score, 1.0)

 return edge, confidence

 def _calculate_vol_stability(self, symbol: str) -> float:
 """Calculate volatility stability score"""
 if symbol not in self.vol_oracle.historical_vols:
 return 0.5

 vols = list(self.vol_oracle.historical_vols[symbol])
 if len(vols) < 10:
 return 0.5

 vol_std = np.std(vols)
 vol_mean = np.mean(vols)

 # Lower relative volatility = higher stability
 vol_of_vol = vol_std / vol_mean if vol_mean > 0 else 1.0
 stability = max(0.1, 1.0 - vol_of_vol)

 return min(stability, 1.0)

 def _calculate_liquidity_score(self, bid: float, ask: float) -> float:
 """Calculate liquidity score based on spread"""
 if bid <= 0 or ask <= 0 or ask <= bid:
 return 0.1

 mid = (bid + ask) / 2.0
 spread_pct = (ask - bid) / mid

 # Tighter spreads = higher liquidity
 return max(0.1, 1.0 - min(spread_pct * 10, 0.9))

class RiskManager:
 """Real-time risk management for market making"""

 def __init__(self, config: MarketMakerConfig, inventory: InventoryManager):
 self.config = config
 self.inventory = inventory
 self.risk_alerts: List[str] = []

 def check_position_limits(self, symbol: str, quantity: int) -> bool:
 """Check if new position would violate limits"""
 current_pos = self.inventory.positions.get(symbol, Position(symbol, 0, 0)).quantity
 new_position = current_pos + quantity

 if abs(new_position) > self.config.max_position_size:
 self.risk_alerts.append(f"Position limit exceeded for {symbol}")
 return False

 return True

 def check_portfolio_limits(self) -> bool:
 """Check portfolio-level risk limits"""
 greeks = self.inventory.get_portfolio_greeks()

 if abs(greeks.delta) > self.config.max_portfolio_delta:
 self.risk_alerts.append(f"Portfolio delta limit exceeded: {greeks.delta}")
 return False

 if abs(greeks.gamma) > self.config.max_portfolio_gamma:
 self.risk_alerts.append(f"Portfolio gamma limit exceeded: {greeks.gamma}")
 return False

 if abs(greeks.vega) > self.config.max_portfolio_vega:
 self.risk_alerts.append(f"Portfolio vega limit exceeded: {greeks.vega}")
 return False

 return True

 def calculate_position_sizing(self, symbol: str, edge: float, confidence: float) -> int:
 """Calculate optimal position size based on edge and risk"""
 if edge <= 0 or confidence < self.config.confidence_threshold:
 return 0

 # Kelly criterion with modifications
 win_prob = 0.5 + (edge * confidence) # Adjust probability by confidence
 loss_prob = 1 - win_prob

 if win_prob <= 0.5:
 return 0

 # Expected value and Kelly fraction
 kelly_fraction = (win_prob * edge - loss_prob) / edge
 kelly_fraction = min(kelly_fraction, 0.25) # Cap at 25%

 # Base position size
 base_size = int(self.config.max_position_size * kelly_fraction)

 # Adjust for current inventory
 inventory_skew = self.inventory.get_inventory_skew(symbol)
 inventory_adjustment = 1.0 - abs(inventory_skew) * self.config.inventory_penalty

 adjusted_size = int(base_size * inventory_adjustment)

 return max(1, min(adjusted_size, self.config.max_position_size // 4))

class MarketMaker:
 """Main market making engine"""

 def __init__(self, config: MarketMakerConfig = None):
 self.config = config or MarketMakerConfig()
 self.inventory = InventoryManager(self.config)
 self.vol_oracle = VolatilityOracle()
 self.edge_calculator = EdgeCalculator(self.vol_oracle)
 self.risk_manager = RiskManager(self.config, self.inventory)
 self.pricing_engine = PricingEngine()

 self.active_quotes: Dict[str, Quote] = {}
 self.quote_history: List[Quote] = []

 self.is_running = False
 self.pnl_history: List[float] = []

 def start(self):
 """Start market making engine"""
 self.is_running = True
 logger.info("Market maker started")

 def stop(self):
 """Stop market making engine"""
 self.is_running = False
 logger.info("Market maker stopped")

 def generate_quote(self, option: OptionContract, market_data: MarketData,
 market_bid: float = None, market_ask: float = None) -> Optional[Quote]:
 """Generate market maker quote for option"""

 if not self.is_running:
 return None

 try:
 # Calculate theoretical value
 result = self.pricing_engine.price_option(option, market_data)
 if not result.success:
 return None

 theoretical = result.price

 # Calculate edge and confidence
 if market_bid and market_ask:
 edge, confidence = self.edge_calculator.calculate_edge(
 option, market_data, market_bid, market_ask
 )
 else:
 edge = self.config.edge_target
 confidence = 0.8

 # Check if we should quote
 if confidence < self.config.confidence_threshold:
 return None

 # Calculate spread
 spread = self._calculate_spread(option, market_data, theoretical, edge, confidence)

 # Adjust for inventory
 inventory_skew = self.inventory.get_inventory_skew(option.symbol)
 inventory_adjustment = inventory_skew * self.config.inventory_penalty

 # Apply inventory skew to bid/ask
 bid_price = theoretical - spread/2 - inventory_adjustment
 ask_price = theoretical + spread/2 - inventory_adjustment

 # Ensure minimum spread
 min_spread = theoretical * self.config.min_spread_bps / 10000
 if ask_price - bid_price < min_spread:
 mid_adj = (bid_price + ask_price) / 2
 bid_price = mid_adj - min_spread/2
 ask_price = mid_adj + min_spread/2

 # Calculate position sizes
 edge_adjusted = edge * confidence
 bid_size = self.risk_manager.calculate_position_sizing(
 option.symbol, edge_adjusted, confidence
 )
 ask_size = bid_size

 # Risk checks
 if not self.risk_manager.check_portfolio_limits():
 return None

 quote = Quote(
 symbol=option.symbol,
 bid_price=max(bid_price, 0.01), # Minimum price
 ask_price=ask_price,
 bid_size=bid_size,
 ask_size=ask_size,
 timestamp=time.time(),
 theoretical_value=theoretical,
 edge=edge,
 confidence=confidence
 )

 self.active_quotes[option.symbol] = quote
 self.quote_history.append(quote)

 return quote

 except Exception as e:
 logger.error(f"Error generating quote for {option.symbol}: {e}")
 return None

 def _calculate_spread(self, option: OptionContract, market_data: MarketData,
 theoretical: float, edge: float, confidence: float) -> float:
 """Calculate bid-ask spread"""

 # Base spread from configuration
 base_spread_bps = self.config.min_spread_bps + (
 (self.config.max_spread_bps - self.config.min_spread_bps) *
 (1.0 - confidence)
 )

 base_spread = theoretical * base_spread_bps / 10000

 # Adjust for Greeks (higher gamma = wider spread)
 result = self.pricing_engine.price_option(option, market_data)
 if result.success:
 gamma_adjustment = min(abs(result.greeks.gamma) * 0.01, 0.5)
 base_spread *= (1.0 + gamma_adjustment)

 # Adjust for time to expiry (shorter time = wider spread)
 time_adjustment = max(0.5, 1.0 - market_data.time_to_expiry)
 base_spread *= time_adjustment

 return base_spread

 def handle_fill(self, symbol: str, quantity: int, price: float,
 option: OptionContract, market_data: MarketData):
 """Handle filled order"""

 try:
 # Calculate Greeks for position
 result = self.pricing_engine.price_option(option, market_data)
 greeks = result.greeks if result.success else Greeks()

 # Update inventory
 self.inventory.add_position(symbol, quantity, price, greeks)

 # Update P&L
 if symbol in self.active_quotes:
 quote = self.active_quotes[symbol]
 pnl = (price - quote.theoretical_value) * quantity
 self.inventory.total_pnl += pnl
 self.pnl_history.append(self.inventory.total_pnl)

 # Check if hedging is needed
 should_hedge, reason = self.inventory.should_hedge()
 if should_hedge:
 logger.warning(f"Hedging required: {reason}")
 self._execute_hedge()

 logger.info(f"Fill: {quantity} {symbol} @ {price}")

 except Exception as e:
 logger.error(f"Error handling fill: {e}")

 def _execute_hedge(self):
 """Execute portfolio hedge"""
 greeks = self.inventory.get_portfolio_greeks()

 # Simple delta hedging (in practice, would use underlying or futures)
 if abs(greeks.delta) > self.config.hedge_threshold_delta:
 hedge_quantity = -int(greeks.delta)
 logger.info(f"Delta hedge: {hedge_quantity} shares")

 # In practice, would execute hedge trade here
 # For simulation, just log the hedge

 def get_portfolio_summary(self) -> Dict:
 """Get portfolio summary"""
 greeks = self.inventory.get_portfolio_greeks()

 total_positions = len([p for p in self.inventory.positions.values() if p.quantity != 0])
 total_pnl = self.inventory.total_pnl

 return {
 'total_positions': total_positions,
 'total_pnl': total_pnl,
 'portfolio_greeks': greeks.to_dict(),
 'active_quotes': len(self.active_quotes),
 'risk_alerts': self.risk_manager.risk_alerts.copy()
 }

 def get_performance_metrics(self) -> Dict:
 """Get performance metrics"""
 if not self.pnl_history:
 return {}

 pnl_array = np.array(self.pnl_history)
 returns = np.diff(pnl_array) / (np.abs(pnl_array[:-1]) + 1e-8)

 return {
 'total_pnl': self.pnl_history[-1],
 'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
 'max_drawdown': self._calculate_max_drawdown(),
 'hit_ratio': self._calculate_hit_ratio(),
 'avg_quote_spread': self._calculate_avg_spread()
 }

 def _calculate_max_drawdown(self) -> float:
 """Calculate maximum drawdown"""
 if len(self.pnl_history) < 2:
 return 0.0

 pnl = np.array(self.pnl_history)
 running_max = np.maximum.accumulate(pnl)
 drawdown = (pnl - running_max) / (running_max + 1e-8)

 return float(np.min(drawdown))

 def _calculate_hit_ratio(self) -> float:
 """Calculate hit ratio (profitable trades / total trades)"""
 if not self.quote_history:
 return 0.0

 profitable_quotes = sum(1 for q in self.quote_history if q.edge > 0)
 return profitable_quotes / len(self.quote_history)

 def _calculate_avg_spread(self) -> float:
 """Calculate average quoted spread"""
 if not self.quote_history:
 return 0.0

 spreads = [q.spread_bps for q in self.quote_history]
 return float(np.mean(spreads))