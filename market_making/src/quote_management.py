"""
Real-time Quote Generation and Management System

This module implements a sophisticated automated quoting system for options market making
with real-time bid/ask generation, dynamic spread optimization, inventory management,
adverse selection protection, and latency-optimized order management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import queue
import warnings
from enum import Enum
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from market_making_strategies import (
 Quote, MarketData, Greeks, Position, OrderSide, OrderType,
 MarketMakingStrategy, OptionContract
)


@dataclass
class QuoteParams:
 """Parameters for quote generation"""
 theoretical_value: float
 bid_edge: float
 ask_edge: float
 bid_size: int
 ask_size: int
 confidence: float
 max_position: int
 inventory_skew: float
 volatility_skew: float
 adverse_selection_adjustment: float


@dataclass
class MarketState:
 """Current market state for quote decisions"""
 symbol: str
 bid: float
 ask: float
 last: float
 volume: int
 bid_size: int
 ask_size: int
 volatility: float
 underlying_price: float
 time_to_expiry: float
 moneyness: float
 liquidity_score: float
 adverse_selection_score: float


@dataclass
class QuoteRequest:
 """Request for quote generation"""
 symbol: str
 market_data: MarketData
 greeks: Optional[Greeks]
 position: Optional[Position]
 strategy_id: str
 priority: int = 1
 timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QuoteResponse:
 """Response from quote generation"""
 quotes: List[Quote]
 rejected_reason: Optional[str]
 processing_time_ms: float
 strategy_id: str
 timestamp: datetime


class QuoteQuality(Enum):
 """Quote quality levels"""
 HIGH = "high"
 MEDIUM = "medium"
 LOW = "low"
 REJECT = "reject"


class AdverseSelectionProtector:
 """Protects against adverse selection in market making"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'trade_velocity_threshold': 10, # Trades per minute
 'price_move_threshold': 0.05, # 5% price move
 'volume_spike_threshold': 3.0, # 3x average volume
 'time_window_minutes': 5,
 'protection_duration_seconds': 30
 }

 self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
 self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
 self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
 self.protected_symbols: Dict[str, datetime] = {}

 def assess_adverse_selection_risk(self, symbol: str, market_data: MarketData) -> float:
 """Assess adverse selection risk for a symbol (0-1 scale)"""

 # Update histories
 self.price_history[symbol].append((market_data.last, market_data.timestamp))
 self.volume_history[symbol].append((market_data.volume, market_data.timestamp))

 risk_score = 0.0

 # Check trade velocity
 recent_trades = self._get_recent_activity(symbol, self.trade_history, minutes=self.config['time_window_minutes'])
 if len(recent_trades) > self.config['trade_velocity_threshold']:
 risk_score += 0.3

 # Check price volatility
 price_volatility = self._calculate_recent_volatility(symbol)
 if price_volatility > self.config['price_move_threshold']:
 risk_score += 0.4

 # Check volume spikes
 volume_ratio = self._calculate_volume_ratio(symbol, market_data)
 if volume_ratio > self.config['volume_spike_threshold']:
 risk_score += 0.3

 # Check if symbol is currently protected
 if self.is_symbol_protected(symbol):
 risk_score = max(risk_score, 0.8)

 return min(1.0, risk_score)

 def is_symbol_protected(self, symbol: str) -> bool:
 """Check if symbol is currently under adverse selection protection"""
 if symbol not in self.protected_symbols:
 return False

 protection_expiry = self.protected_symbols[symbol] + timedelta(
 seconds=self.config['protection_duration_seconds']
 )

 if datetime.now() > protection_expiry:
 del self.protected_symbols[symbol]
 return False

 return True

 def trigger_protection(self, symbol: str, reason: str = ""):
 """Trigger adverse selection protection for a symbol"""
 self.protected_symbols[symbol] = datetime.now()
 print(f"Adverse selection protection triggered for {symbol}: {reason}")

 def record_trade(self, symbol: str, side: OrderSide, price: float, size: int):
 """Record a trade for adverse selection analysis"""
 self.trade_history[symbol].append({
 'side': side,
 'price': price,
 'size': size,
 'timestamp': datetime.now()
 })

 def _get_recent_activity(self, symbol: str, history: Dict[str, deque], minutes: int) -> List:
 """Get recent activity within time window"""
 cutoff_time = datetime.now() - timedelta(minutes=minutes)
 recent = []

 for item in history[symbol]:
 if isinstance(item, dict) and 'timestamp' in item:
 if item['timestamp'] > cutoff_time:
 recent.append(item)
 elif isinstance(item, tuple) and len(item) == 2:
 if item[1] > cutoff_time:
 recent.append(item)

 return recent

 def _calculate_recent_volatility(self, symbol: str) -> float:
 """Calculate recent price volatility"""
 recent_prices = self._get_recent_activity(symbol, self.price_history, minutes=self.config['time_window_minutes'])

 if len(recent_prices) < 2:
 return 0.0

 prices = [p[0] for p in recent_prices]
 if not prices or prices[0] == 0:
 return 0.0

 price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
 return max(price_changes) if price_changes else 0.0

 def _calculate_volume_ratio(self, symbol: str, market_data: MarketData) -> float:
 """Calculate current volume vs average volume ratio"""
 recent_volumes = self._get_recent_activity(symbol, self.volume_history, minutes=60) # 1 hour window

 if len(recent_volumes) < 10: # Need history for comparison
 return 1.0

 volumes = [v[0] for v in recent_volumes[:-1]] # Exclude current
 avg_volume = np.mean(volumes) if volumes else 1

 return market_data.volume / max(avg_volume, 1)


class DynamicSpreadOptimizer:
 """Optimizes spreads dynamically based on market conditions and Greeks"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'base_spread_bps': 10, # Base spread in basis points
 'volatility_multiplier': 2.0, # Spread multiplier for volatility
 'liquidity_discount': 0.5, # Discount for high liquidity
 'inventory_penalty': 1.5, # Penalty for inventory imbalance
 'gamma_adjustment': 3.0, # Adjustment for gamma exposure
 'vega_adjustment': 2.0, # Adjustment for vega exposure
 'time_decay_adjustment': 1.2, # Adjustment for time decay
 'min_spread_bps': 5, # Minimum spread
 'max_spread_bps': 100 # Maximum spread
 }

 self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
 self.profitability_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

 def calculate_optimal_spread(self, symbol: str, market_state: MarketState,
 greeks: Optional[Greeks], position: Optional[Position],
 adverse_selection_score: float) -> Tuple[float, float]:
 """Calculate optimal bid and ask spreads"""

 base_spread = self.config['base_spread_bps'] / 10000.0 # Convert to decimal

 # Volatility adjustment
 vol_adjustment = 1.0 + (market_state.volatility * self.config['volatility_multiplier'])

 # Liquidity adjustment
 liquidity_adjustment = max(0.5, 1.0 - (market_state.liquidity_score * self.config['liquidity_discount']))

 # Inventory adjustment
 inventory_adjustment = 1.0
 if position and position.quantity != 0:
 inventory_ratio = abs(position.quantity) / 100 # Normalize by typical position size
 inventory_adjustment = 1.0 + (inventory_ratio * self.config['inventory_penalty'])

 # Greeks adjustments
 greeks_adjustment = 1.0
 if greeks:
 # Gamma adjustment (higher gamma = wider spreads)
 gamma_adj = 1.0 + (abs(greeks.gamma) * self.config['gamma_adjustment'])

 # Vega adjustment (higher vega = wider spreads in volatile markets)
 vega_adj = 1.0 + (abs(greeks.vega) * market_state.volatility * self.config['vega_adjustment'] / 100)

 # Time decay adjustment (higher theta = tighter spreads to encourage flow)
 theta_adj = max(0.5, 1.0 - (abs(greeks.theta) * self.config['time_decay_adjustment'] / 100))

 greeks_adjustment = gamma_adj * vega_adj * theta_adj

 # Adverse selection adjustment
 adverse_adj = 1.0 + (adverse_selection_score * 2.0) # Double spread for high adverse selection

 # Time to expiry adjustment
 tte_adjustment = 1.0
 if market_state.time_to_expiry < 0.1: # Less than 36 days
 tte_adjustment = 1.0 + (0.1 - market_state.time_to_expiry) * 5.0 # Widen spreads near expiry

 # Moneyness adjustment
 moneyness_adjustment = 1.0 + abs(market_state.moneyness) * 0.5 # Widen for OTM options

 # Combine all adjustments
 total_adjustment = (vol_adjustment * liquidity_adjustment * inventory_adjustment *
 greeks_adjustment * adverse_adj * tte_adjustment * moneyness_adjustment)

 adjusted_spread = base_spread * total_adjustment

 # Apply limits
 min_spread = self.config['min_spread_bps'] / 10000.0
 max_spread = self.config['max_spread_bps'] / 10000.0
 adjusted_spread = np.clip(adjusted_spread, min_spread, max_spread)

 # Calculate asymmetric spreads based on inventory
 bid_spread = adjusted_spread / 2
 ask_spread = adjusted_spread / 2

 if position and position.quantity != 0:
 # Skew spreads based on inventory
 skew = min(0.3, abs(position.quantity) / 100) * np.sign(position.quantity)
 bid_spread *= (1 - skew) # Tighter bid if long, wider if short
 ask_spread *= (1 + skew) # Wider ask if long, tighter if short

 # Record spread for analysis
 self.spread_history[symbol].append({
 'spread': adjusted_spread,
 'bid_spread': bid_spread,
 'ask_spread': ask_spread,
 'timestamp': datetime.now(),
 'adjustments': {
 'volatility': vol_adjustment,
 'liquidity': liquidity_adjustment,
 'inventory': inventory_adjustment,
 'greeks': greeks_adjustment,
 'adverse_selection': adverse_adj
 }
 })

 return bid_spread, ask_spread

 def analyze_spread_performance(self, symbol: str) -> Dict[str, float]:
 """Analyze spread performance for optimization"""
 if symbol not in self.spread_history or len(self.spread_history[symbol]) < 10:
 return {}

 spreads = list(self.spread_history[symbol])
 spread_values = [s['spread'] for s in spreads]

 return {
 'avg_spread': np.mean(spread_values),
 'spread_volatility': np.std(spread_values),
 'min_spread': np.min(spread_values),
 'max_spread': np.max(spread_values),
 'recent_trend': (spread_values[-5:] if len(spread_values) >= 5 else spread_values),
 }

 def record_fill(self, symbol: str, side: OrderSide, spread_used: float, pnl: float):
 """Record a fill for spread performance analysis"""
 self.profitability_tracker[symbol].append({
 'side': side,
 'spread': spread_used,
 'pnl': pnl,
 'timestamp': datetime.now()
 })


class InventoryManager:
 """Manages inventory levels and position limits"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'max_gross_position': 1000, # Maximum gross position
 'max_net_position': 500, # Maximum net position per symbol
 'max_portfolio_delta': 2000, # Maximum portfolio delta
 'max_portfolio_gamma': 1000, # Maximum portfolio gamma
 'max_portfolio_vega': 5000, # Maximum portfolio vega
 'inventory_decay_factor': 0.95, # Daily inventory decay target
 'concentration_limit': 0.2, # Max 20% of portfolio in one symbol
 'rebalance_threshold': 0.8 # Rebalance when 80% of limit reached
 }

 self.position_limits: Dict[str, Dict[str, float]] = {}
 self.inventory_targets: Dict[str, float] = defaultdict(float)
 self.risk_utilization: Dict[str, float] = {}

 def check_position_limits(self, symbol: str, side: OrderSide, quantity: int,
 current_position: Optional[Position],
 portfolio_greeks: Greeks) -> Tuple[bool, str, int]:
 """Check if position is within limits and return allowed quantity"""

 current_qty = current_position.quantity if current_position else 0

 # Calculate new position after trade
 new_qty = current_qty + (quantity if side == OrderSide.BID else -quantity)

 # Check individual position limits
 if abs(new_qty) > self.config['max_net_position']:
 allowed_qty = max(0, self.config['max_net_position'] - abs(current_qty))
 if allowed_qty == 0:
 return False, f"Position limit exceeded for {symbol}", 0
 return True, "Reduced quantity due to position limit", allowed_qty

 # Check gross position limits
 total_gross = sum(abs(pos.quantity) for pos in self._get_all_positions())
 new_gross = total_gross + quantity - abs(current_qty) + abs(new_qty)

 if new_gross > self.config['max_gross_position']:
 return False, "Gross position limit exceeded", 0

 # Check portfolio Greeks limits
 # This would require calculating new Greeks after trade
 if abs(portfolio_greeks.delta) > self.config['max_portfolio_delta']:
 return False, "Portfolio delta limit exceeded", 0

 if abs(portfolio_greeks.gamma) > self.config['max_portfolio_gamma']:
 return False, "Portfolio gamma limit exceeded", 0

 if abs(portfolio_greeks.vega) > self.config['max_portfolio_vega']:
 return False, "Portfolio vega limit exceeded", 0

 return True, "Position approved", quantity

 def calculate_inventory_skew(self, symbol: str, position: Optional[Position]) -> float:
 """Calculate inventory skew for quote adjustment"""
 if not position or position.quantity == 0:
 return 0.0

 # Calculate skew based on position size relative to limits
 max_pos = self.config['max_net_position']
 position_ratio = position.quantity / max_pos

 # Calculate target inventory (usually 0 for market making)
 target = self.inventory_targets[symbol]
 target_ratio = target / max_pos

 skew = position_ratio - target_ratio

 # Amplify skew as position gets larger
 skew_factor = 1.0 + abs(position_ratio) ** 2

 return skew * skew_factor

 def update_inventory_targets(self, symbol: str, target_quantity: float):
 """Update inventory target for a symbol"""
 self.inventory_targets[symbol] = target_quantity

 def get_rebalancing_recommendations(self) -> List[Dict[str, Any]]:
 """Get recommendations for rebalancing inventory"""
 recommendations = []

 positions = self._get_all_positions()

 for position in positions:
 symbol = position.symbol
 current_qty = position.quantity
 target_qty = self.inventory_targets.get(symbol, 0)

 # Check if rebalancing is needed
 max_pos = self.config['max_net_position']
 utilization = abs(current_qty) / max_pos

 if utilization > self.config['rebalance_threshold']:
 recommended_trade = target_qty - current_qty

 if abs(recommended_trade) > 1: # Only recommend if significant
 recommendations.append({
 'symbol': symbol,
 'current_position': current_qty,
 'target_position': target_qty,
 'recommended_trade': recommended_trade,
 'urgency': utilization,
 'reason': 'Position limit utilization high'
 })

 return recommendations

 def _get_all_positions(self) -> List[Position]:
 """Get all current positions (placeholder - would integrate with position manager)"""
 # This would integrate with the actual position management system
 return []


class QuoteEngine:
 """Main quote generation and management engine"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'quote_refresh_interval_ms': 100, # 100ms refresh rate
 'max_quotes_per_symbol': 2, # Bid and ask
 'quote_timeout_seconds': 5, # Quote validity period
 'max_processing_time_ms': 10, # Maximum processing time
 'queue_size_limit': 1000, # Maximum queue size
 'worker_threads': 4, # Number of worker threads
 'priority_levels': 3 # Number of priority levels
 }

 # Initialize components
 self.adverse_selection_protector = AdverseSelectionProtector()
 self.spread_optimizer = DynamicSpreadOptimizer()
 self.inventory_manager = InventoryManager()

 # Threading and queues
 self.quote_queue = queue.PriorityQueue(maxsize=self.config['queue_size_limit'])
 self.response_callbacks: Dict[str, Callable] = {}
 self.worker_pool = ThreadPoolExecutor(max_workers=self.config['worker_threads'])

 # State management
 self.active_quotes: Dict[str, List[Quote]] = defaultdict(list)
 self.quote_performance: Dict[str, Dict] = defaultdict(dict)
 self.market_states: Dict[str, MarketState] = {}

 # Control flags
 self.is_running = False
 self.processing_stats = {
 'quotes_processed': 0,
 'quotes_rejected': 0,
 'avg_processing_time': 0.0,
 'queue_high_water': 0
 }

 def start(self):
 """Start the quote engine"""
 self.is_running = True
 self.worker_pool.submit(self._process_quote_queue)
 print("Quote engine started")

 def stop(self):
 """Stop the quote engine"""
 self.is_running = False
 self.worker_pool.shutdown(wait=True)
 print("Quote engine stopped")

 def request_quotes(self, request: QuoteRequest, callback: Optional[Callable] = None) -> str:
 """Request quotes for a symbol"""
 if not self.is_running:
 raise RuntimeError("Quote engine not running")

 # Generate request ID
 request_id = f"{request.symbol}_{int(time.time() * 1000)}"

 # Store callback if provided
 if callback:
 self.response_callbacks[request_id] = callback

 try:
 # Add to priority queue (lower number = higher priority)
 priority_score = self._calculate_request_priority(request)
 self.quote_queue.put((priority_score, request_id, request), timeout=0.1)

 # Update stats
 current_queue_size = self.quote_queue.qsize()
 self.processing_stats['queue_high_water'] = max(
 self.processing_stats['queue_high_water'], current_queue_size
 )

 return request_id

 except queue.Full:
 raise RuntimeError("Quote queue is full")

 def cancel_quotes(self, symbol: str):
 """Cancel all active quotes for a symbol"""
 if symbol in self.active_quotes:
 del self.active_quotes[symbol]

 def get_active_quotes(self, symbol: Optional[str] = None) -> Dict[str, List[Quote]]:
 """Get active quotes"""
 if symbol:
 return {symbol: self.active_quotes.get(symbol, [])}
 return dict(self.active_quotes)

 def update_market_state(self, symbol: str, market_data: MarketData, greeks: Optional[Greeks] = None):
 """Update market state for a symbol"""
 # Calculate market state metrics
 contract = self._parse_option_symbol(symbol)
 if not contract:
 return

 time_to_expiry = (contract.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
 moneyness = np.log(market_data.underlying_price / contract.strike) if market_data.underlying_price and contract.strike else 0

 self.market_states[symbol] = MarketState(
 symbol=symbol,
 bid=market_data.bid,
 ask=market_data.ask,
 last=market_data.last,
 volume=market_data.volume,
 bid_size=market_data.bid_size,
 ask_size=market_data.ask_size,
 volatility=market_data.implied_vol or 0.2,
 underlying_price=market_data.underlying_price or market_data.last,
 time_to_expiry=max(0.001, time_to_expiry),
 moneyness=moneyness,
 liquidity_score=self._calculate_liquidity_score(market_data),
 adverse_selection_score=self.adverse_selection_protector.assess_adverse_selection_risk(symbol, market_data)
 )

 def _process_quote_queue(self):
 """Process quote requests from queue"""
 while self.is_running:
 try:
 # Get request from queue with timeout
 priority, request_id, request = self.quote_queue.get(timeout=0.1)
 start_time = time.time()

 # Process the request
 response = self._generate_quote_response(request)

 # Calculate processing time
 processing_time = (time.time() - start_time) * 1000 # Convert to ms
 response.processing_time_ms = processing_time

 # Update statistics
 self._update_processing_stats(processing_time, len(response.quotes) > 0)

 # Call callback if registered
 if request_id in self.response_callbacks:
 callback = self.response_callbacks.pop(request_id)
 self.worker_pool.submit(callback, response)

 # Update active quotes
 if response.quotes:
 self.active_quotes[request.symbol] = response.quotes

 self.quote_queue.task_done()

 except queue.Empty:
 continue
 except Exception as e:
 warnings.warn(f"Error processing quote request: {e}")
 self.processing_stats['quotes_rejected'] += 1

 def _generate_quote_response(self, request: QuoteRequest) -> QuoteResponse:
 """Generate quote response for a request"""
 start_time = time.time()

 try:
 # Get market state
 market_state = self.market_states.get(request.symbol)
 if not market_state:
 return QuoteResponse(
 quotes=[],
 rejected_reason="No market state available",
 processing_time_ms=0,
 strategy_id=request.strategy_id,
 timestamp=datetime.now()
 )

 # Check adverse selection protection
 if self.adverse_selection_protector.is_symbol_protected(request.symbol):
 return QuoteResponse(
 quotes=[],
 rejected_reason="Symbol under adverse selection protection",
 processing_time_ms=0,
 strategy_id=request.strategy_id,
 timestamp=datetime.now()
 )

 # Calculate optimal spreads
 bid_spread, ask_spread = self.spread_optimizer.calculate_optimal_spread(
 request.symbol, market_state, request.greeks, request.position,
 market_state.adverse_selection_score
 )

 # Calculate theoretical value
 theo_value = self._calculate_theoretical_value(request.symbol, request.market_data, request.greeks)
 if theo_value is None:
 return QuoteResponse(
 quotes=[],
 rejected_reason="Cannot calculate theoretical value",
 processing_time_ms=0,
 strategy_id=request.strategy_id,
 timestamp=datetime.now()
 )

 # Calculate inventory skew
 inventory_skew = self.inventory_manager.calculate_inventory_skew(request.symbol, request.position)

 # Generate quote prices
 bid_price = theo_value - bid_spread + inventory_skew
 ask_price = theo_value + ask_spread + inventory_skew

 # Ensure minimum spread and tick size
 bid_price = self._round_to_tick(max(0.01, bid_price))
 ask_price = self._round_to_tick(ask_price)

 if ask_price - bid_price < 0.01: # Minimum $0.01 spread
 mid = (bid_price + ask_price) / 2
 bid_price = self._round_to_tick(mid - 0.005)
 ask_price = self._round_to_tick(mid + 0.005)

 # Calculate quote sizes
 bid_size, ask_size = self._calculate_quote_sizes(request.symbol, market_state, request.greeks, request.position)

 # Check position limits
 quotes = []
 if bid_size > 0:
 allowed, reason, allowed_qty = self.inventory_manager.check_position_limits(
 request.symbol, OrderSide.BID, bid_size, request.position, Greeks(0, 0, 0, 0, 0, 0, datetime.now())
 )
 if allowed and allowed_qty > 0:
 quotes.append(Quote(
 symbol=request.symbol,
 side=OrderSide.BID,
 price=bid_price,
 size=allowed_qty,
 timestamp=datetime.now(),
 strategy_id=request.strategy_id,
 confidence=self._calculate_quote_confidence(market_state, request.greeks)
 ))

 if ask_size > 0:
 allowed, reason, allowed_qty = self.inventory_manager.check_position_limits(
 request.symbol, OrderSide.ASK, ask_size, request.position, Greeks(0, 0, 0, 0, 0, 0, datetime.now())
 )
 if allowed and allowed_qty > 0:
 quotes.append(Quote(
 symbol=request.symbol,
 side=OrderSide.ASK,
 price=ask_price,
 size=allowed_qty,
 timestamp=datetime.now(),
 strategy_id=request.strategy_id,
 confidence=self._calculate_quote_confidence(market_state, request.greeks)
 ))

 return QuoteResponse(
 quotes=quotes,
 rejected_reason=None,
 processing_time_ms=(time.time() - start_time) * 1000,
 strategy_id=request.strategy_id,
 timestamp=datetime.now()
 )

 except Exception as e:
 return QuoteResponse(
 quotes=[],
 rejected_reason=f"Error generating quotes: {str(e)}",
 processing_time_ms=(time.time() - start_time) * 1000,
 strategy_id=request.strategy_id,
 timestamp=datetime.now()
 )

 def _calculate_request_priority(self, request: QuoteRequest) -> int:
 """Calculate request priority (lower = higher priority)"""
 base_priority = request.priority

 # Higher priority for:
 # - High volume symbols
 # - Near expiry options
 # - High adverse selection risk (need faster updates)

 market_state = self.market_states.get(request.symbol)
 if market_state:
 # Near expiry gets higher priority
 if market_state.time_to_expiry < 0.1: # Less than 36 days
 base_priority -= 1

 # High volume gets higher priority
 if market_state.volume > 1000:
 base_priority -= 1

 # High adverse selection gets higher priority
 if market_state.adverse_selection_score > 0.7:
 base_priority -= 1

 return max(1, base_priority)

 def _calculate_theoretical_value(self, symbol: str, market_data: MarketData, greeks: Optional[Greeks]) -> Optional[float]:
 """Calculate theoretical option value"""
 # Simplified Black-Scholes - in practice would use more sophisticated models
 if not market_data.implied_vol or not market_data.underlying_price:
 return (market_data.bid + market_data.ask) / 2 if market_data.bid > 0 and market_data.ask > 0 else market_data.last

 contract = self._parse_option_symbol(symbol)
 if not contract:
 return None

 # Use mid-market as theoretical for simplicity
 # In practice, this would be a sophisticated pricing model
 if market_data.bid > 0 and market_data.ask > 0:
 return (market_data.bid + market_data.ask) / 2
 else:
 return market_data.last

 def _calculate_liquidity_score(self, market_data: MarketData) -> float:
 """Calculate liquidity score (0-1, higher = more liquid)"""
 # Simple liquidity score based on spreads and size
 if market_data.bid <= 0 or market_data.ask <= 0:
 return 0.0

 spread_ratio = (market_data.ask - market_data.bid) / ((market_data.ask + market_data.bid) / 2)
 spread_score = max(0, 1.0 - spread_ratio * 10) # Penalize wide spreads

 size_score = min(1.0, (market_data.bid_size + market_data.ask_size) / 100) # Normalize by typical size

 volume_score = min(1.0, market_data.volume / 1000) # Normalize by typical volume

 return (spread_score + size_score + volume_score) / 3

 def _calculate_quote_sizes(self, symbol: str, market_state: MarketState,
 greeks: Optional[Greeks], position: Optional[Position]) -> Tuple[int, int]:
 """Calculate optimal quote sizes"""
 base_size = 10

 # Adjust for liquidity
 liquidity_factor = 0.5 + market_state.liquidity_score
 size = int(base_size * liquidity_factor)

 # Adjust for Greeks risk
 if greeks:
 # Reduce size for high gamma
 gamma_factor = max(0.2, 1.0 - abs(greeks.gamma) * 20)
 size = int(size * gamma_factor)

 # Reduce size for high vega in volatile markets
 vega_factor = max(0.3, 1.0 - abs(greeks.vega) * market_state.volatility / 100)
 size = int(size * vega_factor)

 # Adjust for time to expiry
 if market_state.time_to_expiry < 0.1: # Less than 36 days
 size = max(1, size // 2) # Reduce size near expiry

 # Adjust for adverse selection
 if market_state.adverse_selection_score > 0.5:
 size = max(1, int(size * (1 - market_state.adverse_selection_score)))

 # Adjust for inventory
 inventory_factor = 1.0
 if position and position.quantity != 0:
 position_ratio = abs(position.quantity) / 100
 inventory_factor = max(0.2, 1.0 - position_ratio)

 size = max(1, int(size * inventory_factor))

 # Calculate asymmetric sizes based on position
 bid_size = size
 ask_size = size

 if position and position.quantity != 0:
 if position.quantity > 0: # Long position, prefer to sell
 ask_size = int(size * 1.2)
 bid_size = max(1, int(size * 0.8))
 else: # Short position, prefer to buy
 bid_size = int(size * 1.2)
 ask_size = max(1, int(size * 0.8))

 return bid_size, ask_size

 def _calculate_quote_confidence(self, market_state: MarketState, greeks: Optional[Greeks]) -> float:
 """Calculate quote confidence score"""
 confidence = 1.0

 # Reduce confidence for wide spreads
 if market_state.bid > 0 and market_state.ask > 0:
 spread_ratio = (market_state.ask - market_state.bid) / ((market_state.ask + market_state.bid) / 2)
 if spread_ratio > 0.1: # 10% spread
 confidence *= 0.6

 # Reduce confidence for high adverse selection
 confidence *= (1 - market_state.adverse_selection_score * 0.5)

 # Reduce confidence for illiquid markets
 confidence *= market_state.liquidity_score

 # Reduce confidence near expiry
 if market_state.time_to_expiry < 0.05: # Less than 18 days
 confidence *= 0.7

 # Reduce confidence for high Greeks
 if greeks:
 if abs(greeks.gamma) > 0.1:
 confidence *= 0.8
 if abs(greeks.vega) > 50:
 confidence *= 0.8

 return max(0.1, confidence)

 def _round_to_tick(self, price: float) -> float:
 """Round price to minimum tick size"""
 if price < 3:
 tick_size = 0.05 # $0.05 for options under $3
 else:
 tick_size = 0.10 # $0.10 for options $3 and above

 return round(price / tick_size) * tick_size

 def _parse_option_symbol(self, symbol: str) -> Optional[OptionContract]:
 """Parse option symbol - simplified implementation"""
 try:
 parts = symbol.split('_')
 if len(parts) != 2:
 return None

 underlying = parts[0]
 option_part = parts[1]

 if 'C' in option_part:
 expiry_str, strike_str = option_part.split('C')
 option_type = 'call'
 elif 'P' in option_part:
 expiry_str, strike_str = option_part.split('P')
 option_type = 'put'
 else:
 return None

 expiry = datetime.strptime(expiry_str, '%y%m%d')
 strike = float(strike_str)

 return OptionContract(
 symbol=symbol,
 underlying=underlying,
 strike=strike,
 expiry=expiry,
 option_type=option_type
 )
 except:
 return None

 def _update_processing_stats(self, processing_time_ms: float, success: bool):
 """Update processing statistics"""
 if success:
 self.processing_stats['quotes_processed'] += 1
 else:
 self.processing_stats['quotes_rejected'] += 1

 # Update average processing time
 total_processed = self.processing_stats['quotes_processed'] + self.processing_stats['quotes_rejected']
 if total_processed > 0:
 current_avg = self.processing_stats['avg_processing_time']
 self.processing_stats['avg_processing_time'] = (current_avg * (total_processed - 1) + processing_time_ms) / total_processed

 def get_performance_stats(self) -> Dict[str, Any]:
 """Get quote engine performance statistics"""
 return {
 'processing_stats': self.processing_stats.copy(),
 'queue_size': self.quote_queue.qsize(),
 'active_quotes_count': sum(len(quotes) for quotes in self.active_quotes.values()),
 'market_states_count': len(self.market_states),
 'adverse_selection_protected': len(self.adverse_selection_protector.protected_symbols)
 }


# Example usage and factory functions
def create_quote_engine(config: Optional[Dict[str, Any]] = None) -> QuoteEngine:
 """Create a configured quote engine"""
 default_config = {
 'quote_refresh_interval_ms': 50, # High frequency
 'max_processing_time_ms': 5, # Low latency
 'worker_threads': 8, # Multiple workers
 'queue_size_limit': 2000 # Large queue
 }

 if config:
 default_config.update(config)

 return QuoteEngine(default_config)


async def run_quote_engine_example():
 """Example of running the quote engine"""
 engine = create_quote_engine()
 engine.start()

 try:
 # Simulate quote requests
 for i in range(10):
 market_data = MarketData(
 symbol=f"AAPL_231215C150",
 timestamp=datetime.now(),
 bid=1.50 + i * 0.01,
 ask=1.60 + i * 0.01,
 last=1.55 + i * 0.01,
 bid_size=50,
 ask_size=50,
 volume=1000,
 open_interest=5000,
 implied_vol=0.25,
 underlying_price=150.0
 )

 engine.update_market_state(market_data.symbol, market_data)

 request = QuoteRequest(
 symbol=market_data.symbol,
 market_data=market_data,
 greeks=None,
 position=None,
 strategy_id="test_strategy"
 )

 request_id = engine.request_quotes(request)
 await asyncio.sleep(0.01) # 10ms between requests

 # Wait for processing
 await asyncio.sleep(1)

 # Get stats
 stats = engine.get_performance_stats()
 print(f"Processed {stats['processing_stats']['quotes_processed']} quotes")
 print(f"Average processing time: {stats['processing_stats']['avg_processing_time']:.2f}ms")

 finally:
 engine.stop()


if __name__ == "__main__":
 asyncio.run(run_quote_engine_example())