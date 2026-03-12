"""
Automated Hedging and Risk Management System

This module implements sophisticated hedging algorithms and risk management for options
market making, including continuous delta hedging, gamma scalping, vega hedging,
portfolio-level Greeks management, and real-time P&L monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import warnings
from enum import Enum
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from market_making_strategies import (
    Greeks, Position, Trade, OrderSide, OrderType, MarketData, OptionContract
)


@dataclass
class HedgeOrder:
    """Hedge order specification"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    urgency: float = 1.0  # 0-1 scale
    hedge_type: str = "delta"  # delta, gamma, vega, theta
    target_greek: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_portfolio_delta: float = 1000.0
    max_portfolio_gamma: float = 500.0
    max_portfolio_vega: float = 2000.0
    max_portfolio_theta: float = -100.0
    max_position_size: int = 100
    max_portfolio_value: float = 1000000.0
    max_daily_loss: float = 10000.0
    max_var_95: float = 50000.0
    concentration_limit: float = 0.2  # 20% max in single name

    # Dynamic limits based on market conditions
    high_vol_delta_multiplier: float = 0.5  # Reduce delta limit in high vol
    low_liquidity_position_multiplier: float = 0.7  # Reduce position size in illiquid markets
    near_expiry_gamma_multiplier: float = 0.3  # Reduce gamma near expiry


@dataclass
class HedgingConfig:
    """Hedging algorithm configuration"""
    delta_hedge_threshold: float = 50.0  # Delta threshold for hedging
    gamma_scalp_threshold: float = 10.0  # Gamma threshold for scalping
    vega_hedge_threshold: float = 200.0  # Vega threshold for hedging

    # Hedging frequencies
    continuous_hedge_interval_seconds: int = 30
    gamma_scalp_interval_seconds: int = 10
    vega_hedge_interval_seconds: int = 300  # 5 minutes

    # Transaction cost parameters
    underlying_commission: float = 0.005  # $0.005 per share
    option_commission: float = 0.50  # $0.50 per contract
    bid_ask_cost_factor: float = 0.5  # Use 50% of bid-ask spread as cost

    # Hedging ratios
    delta_hedge_ratio: float = 0.8  # Hedge 80% of delta
    gamma_scalp_ratio: float = 1.0  # Full gamma scalp
    vega_hedge_ratio: float = 0.6  # Hedge 60% of vega

    # Market impact parameters
    max_market_impact_bps: int = 10  # Maximum 10bps market impact
    volume_participation_limit: float = 0.1  # Max 10% of average volume


@dataclass
class PnLAttribution:
    """P&L attribution breakdown"""
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    theta_pnl: float = 0.0
    vega_pnl: float = 0.0
    rho_pnl: float = 0.0
    carry_pnl: float = 0.0
    trading_pnl: float = 0.0
    commission_costs: float = 0.0
    slippage_costs: float = 0.0
    total_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class HedgingAlgorithm(Enum):
    """Types of hedging algorithms"""
    CONTINUOUS_DELTA = "continuous_delta"
    GAMMA_SCALPING = "gamma_scalping"
    VEGA_HEDGING = "vega_hedging"
    THETA_DECAY = "theta_decay"
    CROSS_GAMMA = "cross_gamma"
    DYNAMIC_DELTA = "dynamic_delta"


class RiskMonitor:
    """Real-time risk monitoring and alerting system"""

    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.risk_violations: List[Dict] = []
        self.portfolio_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.alert_callbacks: List[Callable] = []
        self.daily_pnl_start: Optional[float] = None

    def check_risk_limits(self, portfolio_greeks: Greeks, positions: Dict[str, Position],
                         current_pnl: float, market_data: Dict[str, MarketData]) -> List[str]:
        """Check all risk limits and return list of violations"""
        violations = []

        # Greeks limits
        if abs(portfolio_greeks.delta) > self.risk_limits.max_portfolio_delta:
            violations.append(f"Portfolio delta {portfolio_greeks.delta:.2f} exceeds limit {self.risk_limits.max_portfolio_delta}")

        if abs(portfolio_greeks.gamma) > self.risk_limits.max_portfolio_gamma:
            violations.append(f"Portfolio gamma {portfolio_greeks.gamma:.2f} exceeds limit {self.risk_limits.max_portfolio_gamma}")

        if abs(portfolio_greeks.vega) > self.risk_limits.max_portfolio_vega:
            violations.append(f"Portfolio vega {portfolio_greeks.vega:.2f} exceeds limit {self.risk_limits.max_portfolio_vega}")

        if portfolio_greeks.theta < self.risk_limits.max_portfolio_theta:
            violations.append(f"Portfolio theta {portfolio_greeks.theta:.2f} exceeds limit {self.risk_limits.max_portfolio_theta}")

        # Position size limits
        for symbol, position in positions.items():
            if abs(position.quantity) > self.risk_limits.max_position_size:
                violations.append(f"Position size {position.quantity} in {symbol} exceeds limit {self.risk_limits.max_position_size}")

        # Portfolio value limit
        total_value = sum(abs(pos.market_value) for pos in positions.values())
        if total_value > self.risk_limits.max_portfolio_value:
            violations.append(f"Portfolio value {total_value:.2f} exceeds limit {self.risk_limits.max_portfolio_value}")

        # Daily loss limit
        if self.daily_pnl_start is not None:
            daily_pnl = current_pnl - self.daily_pnl_start
            if daily_pnl < -self.risk_limits.max_daily_loss:
                violations.append(f"Daily loss {daily_pnl:.2f} exceeds limit {self.risk_limits.max_daily_loss}")

        # Concentration limit
        concentration_violations = self._check_concentration_limits(positions)
        violations.extend(concentration_violations)

        # Market condition adjustments
        market_violations = self._check_market_condition_limits(portfolio_greeks, market_data)
        violations.extend(market_violations)

        # Record violations
        if violations:
            for violation in violations:
                self.risk_violations.append({
                    'timestamp': datetime.now(),
                    'violation': violation,
                    'portfolio_greeks': portfolio_greeks,
                    'current_pnl': current_pnl
                })

            # Trigger alerts
            self._trigger_alerts(violations)

        return violations

    def calculate_var(self, returns_history: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns_history) < 30:  # Need sufficient history
            return 0.0

        returns = np.array(returns_history)
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)

        return abs(var)

    def update_portfolio_history(self, portfolio_greeks: Greeks, total_pnl: float, timestamp: datetime):
        """Update portfolio history for risk calculations"""
        self.portfolio_history.append({
            'timestamp': timestamp,
            'delta': portfolio_greeks.delta,
            'gamma': portfolio_greeks.gamma,
            'vega': portfolio_greeks.vega,
            'theta': portfolio_greeks.theta,
            'total_pnl': total_pnl
        })

    def set_daily_pnl_start(self, start_pnl: float):
        """Set starting P&L for daily tracking"""
        self.daily_pnl_start = start_pnl

    def add_alert_callback(self, callback: Callable):
        """Add callback for risk alerts"""
        self.alert_callbacks.append(callback)

    def _check_concentration_limits(self, positions: Dict[str, Position]) -> List[str]:
        """Check position concentration limits"""
        violations = []

        total_value = sum(abs(pos.market_value) for pos in positions.values())
        if total_value == 0:
            return violations

        # Group by underlying
        underlying_exposure = defaultdict(float)
        for position in positions.values():
            # Extract underlying from option symbol (simplified)
            underlying = position.symbol.split('_')[0] if '_' in position.symbol else position.symbol
            underlying_exposure[underlying] += abs(position.market_value)

        # Check concentration per underlying
        for underlying, exposure in underlying_exposure.items():
            concentration = exposure / total_value
            if concentration > self.risk_limits.concentration_limit:
                violations.append(f"Concentration in {underlying} ({concentration:.1%}) exceeds limit ({self.risk_limits.concentration_limit:.1%})")

        return violations

    def _check_market_condition_limits(self, portfolio_greeks: Greeks, market_data: Dict[str, MarketData]) -> List[str]:
        """Check limits adjusted for market conditions"""
        violations = []

        # Calculate average implied volatility
        avg_vol = 0.0
        vol_count = 0
        for data in market_data.values():
            if data.implied_vol and data.implied_vol > 0:
                avg_vol += data.implied_vol
                vol_count += 1

        if vol_count > 0:
            avg_vol /= vol_count

            # High volatility adjustments
            if avg_vol > 0.4:  # 40% average vol considered high
                adjusted_delta_limit = self.risk_limits.max_portfolio_delta * self.risk_limits.high_vol_delta_multiplier
                if abs(portfolio_greeks.delta) > adjusted_delta_limit:
                    violations.append(f"High volatility delta limit {portfolio_greeks.delta:.2f} exceeds adjusted limit {adjusted_delta_limit}")

        return violations

    def _trigger_alerts(self, violations: List[str]):
        """Trigger alerts for risk violations"""
        for callback in self.alert_callbacks:
            try:
                callback(violations)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}")


class DeltaHedger:
    """Continuous delta hedging algorithm"""

    def __init__(self, config: HedgingConfig):
        self.config = config
        self.last_hedge_time: Optional[datetime] = None
        self.hedge_history: deque = deque(maxlen=1000)
        self.underlying_positions: Dict[str, float] = defaultdict(float)

    def calculate_hedge_requirement(self, portfolio_greeks: Greeks,
                                  positions: Dict[str, Position]) -> Optional[HedgeOrder]:
        """Calculate required delta hedge"""

        if abs(portfolio_greeks.delta) < self.config.delta_hedge_threshold:
            return None

        # Calculate hedge quantity
        hedge_delta = portfolio_greeks.delta * self.config.delta_hedge_ratio
        hedge_quantity = int(abs(hedge_delta))

        if hedge_quantity == 0:
            return None

        # Determine underlying symbol (simplified - would need mapping)
        underlying_symbol = self._get_primary_underlying(positions)

        # Determine side
        side = OrderSide.ASK if hedge_delta > 0 else OrderSide.BID

        # Calculate urgency based on delta size and time since last hedge
        urgency = self._calculate_hedge_urgency(portfolio_greeks.delta)

        return HedgeOrder(
            symbol=underlying_symbol,
            side=side,
            quantity=hedge_quantity,
            order_type=OrderType.MARKET if urgency > 0.8 else OrderType.LIMIT,
            hedge_type="delta",
            target_greek=0.0,
            urgency=urgency
        )

    def should_hedge_now(self, portfolio_greeks: Greeks, market_conditions: Dict[str, Any]) -> bool:
        """Determine if hedge should be executed now"""

        # Always hedge if above threshold
        if abs(portfolio_greeks.delta) > self.config.delta_hedge_threshold:
            return True

        # Time-based hedging
        if self.last_hedge_time:
            time_since_hedge = (datetime.now() - self.last_hedge_time).total_seconds()
            if time_since_hedge > self.config.continuous_hedge_interval_seconds:
                return abs(portfolio_greeks.delta) > self.config.delta_hedge_threshold * 0.5

        return False

    def calculate_transaction_cost(self, hedge_order: HedgeOrder, market_data: MarketData) -> float:
        """Calculate expected transaction cost for hedge"""

        # Commission cost
        commission = hedge_order.quantity * self.config.underlying_commission

        # Bid-ask spread cost
        if market_data.bid > 0 and market_data.ask > 0:
            spread_cost = (market_data.ask - market_data.bid) * hedge_order.quantity * self.config.bid_ask_cost_factor
        else:
            spread_cost = 0.0

        # Market impact cost (simplified model)
        # In practice, would use more sophisticated market impact models
        impact_cost = hedge_order.quantity * 0.001  # $0.001 per share impact

        return commission + spread_cost + impact_cost

    def record_hedge_execution(self, hedge_order: HedgeOrder, execution_price: float):
        """Record hedge execution for analysis"""
        self.last_hedge_time = datetime.now()

        self.hedge_history.append({
            'timestamp': datetime.now(),
            'symbol': hedge_order.symbol,
            'side': hedge_order.side,
            'quantity': hedge_order.quantity,
            'execution_price': execution_price,
            'hedge_type': hedge_order.hedge_type,
            'urgency': hedge_order.urgency
        })

        # Update underlying positions
        position_change = hedge_order.quantity if hedge_order.side == OrderSide.BID else -hedge_order.quantity
        self.underlying_positions[hedge_order.symbol] += position_change

    def _get_primary_underlying(self, positions: Dict[str, Position]) -> str:
        """Get primary underlying for hedging"""
        # Simplified - would need proper underlying mapping
        underlying_exposure = defaultdict(float)

        for position in positions.values():
            if '_' in position.symbol:  # Option symbol
                underlying = position.symbol.split('_')[0]
            else:
                underlying = position.symbol

            underlying_exposure[underlying] += abs(position.market_value)

        if underlying_exposure:
            return max(underlying_exposure.items(), key=lambda x: x[1])[0]

        return "SPY"  # Default

    def _calculate_hedge_urgency(self, current_delta: float) -> float:
        """Calculate hedge urgency based on delta magnitude"""
        delta_ratio = abs(current_delta) / self.config.delta_hedge_threshold

        # Exponential urgency scaling
        urgency = min(1.0, delta_ratio ** 2)

        return urgency


class GammaScalper:
    """Gamma scalping algorithm"""

    def __init__(self, config: HedgingConfig):
        self.config = config
        self.last_scalp_time: Optional[datetime] = None
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.scalp_history: deque = deque(maxlen=1000)

    def calculate_gamma_scalp(self, portfolio_greeks: Greeks, underlying_prices: Dict[str, float],
                            previous_prices: Dict[str, float]) -> List[HedgeOrder]:
        """Calculate gamma scalping opportunities"""

        scalp_orders = []

        if abs(portfolio_greeks.gamma) < self.config.gamma_scalp_threshold:
            return scalp_orders

        # Calculate price moves for each underlying
        for underlying, current_price in underlying_prices.items():
            if underlying not in previous_prices:
                continue

            previous_price = previous_prices[underlying]
            price_move = current_price - previous_price

            if abs(price_move) < 0.01:  # Minimum move threshold
                continue

            # Update price history
            self.price_history[underlying].append(current_price)

            # Calculate gamma scalp quantity
            # Gamma scalping: buy when price falls, sell when price rises
            scalp_quantity = self._calculate_scalp_quantity(portfolio_greeks.gamma, price_move, current_price)

            if scalp_quantity == 0:
                continue

            # Determine side (opposite to price move for gamma scalping)
            side = OrderSide.BID if price_move > 0 else OrderSide.ASK

            scalp_orders.append(HedgeOrder(
                symbol=underlying,
                side=side,
                quantity=abs(scalp_quantity),
                order_type=OrderType.LIMIT,
                hedge_type="gamma",
                target_greek=0.0,
                urgency=0.3  # Lower urgency for gamma scalping
            ))

        return scalp_orders

    def should_scalp_now(self, portfolio_greeks: Greeks) -> bool:
        """Determine if gamma scalping should be performed"""

        # Check gamma threshold
        if abs(portfolio_greeks.gamma) < self.config.gamma_scalp_threshold:
            return False

        # Time-based scalping
        if self.last_scalp_time:
            time_since_scalp = (datetime.now() - self.last_scalp_time).total_seconds()
            return time_since_scalp > self.config.gamma_scalp_interval_seconds

        return True

    def calculate_gamma_pnl(self, price_moves: Dict[str, float], gamma_exposure: float) -> float:
        """Calculate P&L from gamma exposure"""

        total_gamma_pnl = 0.0

        for underlying, price_move in price_moves.items():
            # Gamma P&L = 0.5 * Gamma * (Price Move)^2
            gamma_pnl = 0.5 * gamma_exposure * (price_move ** 2)
            total_gamma_pnl += gamma_pnl

        return total_gamma_pnl

    def _calculate_scalp_quantity(self, portfolio_gamma: float, price_move: float, current_price: float) -> int:
        """Calculate optimal gamma scalp quantity"""

        # Basic gamma scalping formula
        # Quantity proportional to gamma and price move
        base_quantity = abs(portfolio_gamma) * abs(price_move) * self.config.gamma_scalp_ratio

        # Scale by price level (avoid huge quantities for low-priced stocks)
        price_scale = min(1.0, 100.0 / current_price)
        scaled_quantity = base_quantity * price_scale

        # Apply limits
        max_scalp_quantity = 1000  # Maximum scalp size
        quantity = min(max_scalp_quantity, abs(int(scaled_quantity)))

        return quantity

    def record_scalp_execution(self, scalp_order: HedgeOrder, execution_price: float):
        """Record gamma scalp execution"""
        self.last_scalp_time = datetime.now()

        self.scalp_history.append({
            'timestamp': datetime.now(),
            'symbol': scalp_order.symbol,
            'side': scalp_order.side,
            'quantity': scalp_order.quantity,
            'execution_price': execution_price,
            'hedge_type': scalp_order.hedge_type
        })


class VegaHedger:
    """Vega hedging using other options"""

    def __init__(self, config: HedgingConfig):
        self.config = config
        self.last_hedge_time: Optional[datetime] = None
        self.hedge_instruments: Dict[str, List[str]] = {}  # Mapping of underlying to hedge instruments

    def calculate_vega_hedge(self, portfolio_greeks: Greeks, available_options: Dict[str, Dict]) -> List[HedgeOrder]:
        """Calculate vega hedge using available options"""

        hedge_orders = []

        if abs(portfolio_greeks.vega) < self.config.vega_hedge_threshold:
            return hedge_orders

        # Find best hedge instruments
        hedge_instruments = self._find_vega_hedge_instruments(portfolio_greeks.vega, available_options)

        for instrument in hedge_instruments:
            hedge_order = self._create_vega_hedge_order(instrument, portfolio_greeks.vega)
            if hedge_order:
                hedge_orders.append(hedge_order)

        return hedge_orders

    def should_hedge_vega_now(self, portfolio_greeks: Greeks) -> bool:
        """Determine if vega hedging should be performed"""

        if abs(portfolio_greeks.vega) < self.config.vega_hedge_threshold:
            return False

        # Time-based vega hedging (less frequent than delta)
        if self.last_hedge_time:
            time_since_hedge = (datetime.now() - self.last_hedge_time).total_seconds()
            return time_since_hedge > self.config.vega_hedge_interval_seconds

        return True

    def _find_vega_hedge_instruments(self, target_vega: float, available_options: Dict[str, Dict]) -> List[Dict]:
        """Find best options for vega hedging"""

        hedge_candidates = []

        for symbol, option_data in available_options.items():
            if 'greeks' not in option_data or 'market_data' not in option_data:
                continue

            greeks = option_data['greeks']
            market_data = option_data['market_data']

            # Check if this option has significant vega
            if abs(greeks.vega) < 5.0:  # Minimum vega threshold
                continue

            # Calculate hedge efficiency (vega per dollar)
            mid_price = (market_data.bid + market_data.ask) / 2 if market_data.bid > 0 and market_data.ask > 0 else market_data.last
            if mid_price <= 0:
                continue

            vega_efficiency = abs(greeks.vega) / mid_price

            # Calculate liquidity score
            liquidity_score = self._calculate_option_liquidity(market_data)

            hedge_candidates.append({
                'symbol': symbol,
                'vega': greeks.vega,
                'price': mid_price,
                'efficiency': vega_efficiency,
                'liquidity': liquidity_score,
                'greeks': greeks,
                'market_data': market_data
            })

        # Sort by efficiency and liquidity
        hedge_candidates.sort(key=lambda x: (x['efficiency'] * x['liquidity']), reverse=True)

        # Return top candidates
        return hedge_candidates[:3]

    def _create_vega_hedge_order(self, instrument: Dict, target_vega: float) -> Optional[HedgeOrder]:
        """Create vega hedge order for an instrument"""

        instrument_vega = instrument['vega']

        if abs(instrument_vega) < 1.0:
            return None

        # Calculate quantity needed
        hedge_vega = target_vega * self.config.vega_hedge_ratio
        quantity = abs(int(hedge_vega / instrument_vega))

        if quantity == 0:
            return None

        # Determine side (opposite to target vega)
        side = OrderSide.ASK if hedge_vega > 0 else OrderSide.BID

        return HedgeOrder(
            symbol=instrument['symbol'],
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=instrument['price'],
            hedge_type="vega",
            target_greek=0.0,
            urgency=0.5
        )

    def _calculate_option_liquidity(self, market_data: MarketData) -> float:
        """Calculate option liquidity score"""

        # Based on bid-ask spread and size
        if market_data.bid <= 0 or market_data.ask <= 0:
            return 0.0

        spread_ratio = (market_data.ask - market_data.bid) / ((market_data.ask + market_data.bid) / 2)
        spread_score = max(0, 1.0 - spread_ratio * 5)  # Penalize wide spreads

        size_score = min(1.0, (market_data.bid_size + market_data.ask_size) / 50)

        return (spread_score + size_score) / 2


class PnLAttributionEngine:
    """Real-time P&L attribution and monitoring"""

    def __init__(self):
        self.attribution_history: deque = deque(maxlen=10000)
        self.previous_greeks: Optional[Greeks] = None
        self.previous_positions: Dict[str, Position] = {}
        self.previous_prices: Dict[str, float] = {}

    def calculate_attribution(self, current_greeks: Greeks, current_positions: Dict[str, Position],
                            current_prices: Dict[str, float], trades: List[Trade],
                            time_elapsed: float) -> PnLAttribution:
        """Calculate detailed P&L attribution"""

        attribution = PnLAttribution()

        if self.previous_greeks is None:
            self.previous_greeks = current_greeks
            self.previous_positions = current_positions.copy()
            self.previous_prices = current_prices.copy()
            return attribution

        # Calculate price changes
        price_changes = {}
        for symbol, current_price in current_prices.items():
            if symbol in self.previous_prices:
                price_changes[symbol] = current_price - self.previous_prices[symbol]

        # Delta P&L
        attribution.delta_pnl = self._calculate_delta_pnl(price_changes, self.previous_greeks)

        # Gamma P&L
        attribution.gamma_pnl = self._calculate_gamma_pnl(price_changes, self.previous_greeks)

        # Theta P&L
        attribution.theta_pnl = self._calculate_theta_pnl(self.previous_greeks, time_elapsed)

        # Vega P&L (would need volatility changes)
        attribution.vega_pnl = 0.0  # Simplified

        # Trading P&L from new trades
        attribution.trading_pnl = self._calculate_trading_pnl(trades)

        # Commission and slippage costs
        attribution.commission_costs = self._calculate_commission_costs(trades)
        attribution.slippage_costs = self._calculate_slippage_costs(trades)

        # Total P&L
        attribution.total_pnl = (attribution.delta_pnl + attribution.gamma_pnl +
                               attribution.theta_pnl + attribution.vega_pnl +
                               attribution.trading_pnl - attribution.commission_costs -
                               attribution.slippage_costs)

        # Update history
        self.attribution_history.append(attribution)

        # Update previous values
        self.previous_greeks = current_greeks
        self.previous_positions = current_positions.copy()
        self.previous_prices = current_prices.copy()

        return attribution

    def get_attribution_summary(self, period_hours: int = 24) -> Dict[str, float]:
        """Get P&L attribution summary for a period"""

        cutoff_time = datetime.now() - timedelta(hours=period_hours)

        period_attributions = [attr for attr in self.attribution_history
                             if attr.timestamp > cutoff_time]

        if not period_attributions:
            return {}

        summary = {
            'total_pnl': sum(attr.total_pnl for attr in period_attributions),
            'delta_pnl': sum(attr.delta_pnl for attr in period_attributions),
            'gamma_pnl': sum(attr.gamma_pnl for attr in period_attributions),
            'theta_pnl': sum(attr.theta_pnl for attr in period_attributions),
            'vega_pnl': sum(attr.vega_pnl for attr in period_attributions),
            'trading_pnl': sum(attr.trading_pnl for attr in period_attributions),
            'commission_costs': sum(attr.commission_costs for attr in period_attributions),
            'slippage_costs': sum(attr.slippage_costs for attr in period_attributions),
            'trade_count': len(period_attributions)
        }

        return summary

    def _calculate_delta_pnl(self, price_changes: Dict[str, float], greeks: Greeks) -> float:
        """Calculate P&L from delta exposure"""

        # Simplified - assumes single underlying
        # In practice, would need underlying mapping for each position
        total_price_move = sum(price_changes.values())

        if total_price_move == 0:
            return 0.0

        return greeks.delta * total_price_move

    def _calculate_gamma_pnl(self, price_changes: Dict[str, float], greeks: Greeks) -> float:
        """Calculate P&L from gamma exposure"""

        total_gamma_pnl = 0.0

        for price_move in price_changes.values():
            # Gamma P&L = 0.5 * Gamma * (Price Move)^2
            gamma_pnl = 0.5 * greeks.gamma * (price_move ** 2)
            total_gamma_pnl += gamma_pnl

        return total_gamma_pnl

    def _calculate_theta_pnl(self, greeks: Greeks, time_elapsed: float) -> float:
        """Calculate P&L from time decay"""

        # Theta is typically quoted per day, so scale by time elapsed
        days_elapsed = time_elapsed / (24 * 3600)

        return greeks.theta * days_elapsed

    def _calculate_trading_pnl(self, trades: List[Trade]) -> float:
        """Calculate P&L from trading activity"""

        # Simplified calculation - would need more sophisticated attribution
        trading_pnl = 0.0

        for trade in trades:
            # This would involve marking trades to market
            # For now, assume trades are marked at mid-market
            pass

        return trading_pnl

    def _calculate_commission_costs(self, trades: List[Trade]) -> float:
        """Calculate commission costs from trades"""

        total_commission = 0.0

        for trade in trades:
            total_commission += trade.commission

        return total_commission

    def _calculate_slippage_costs(self, trades: List[Trade]) -> float:
        """Calculate slippage costs from trades"""

        total_slippage = 0.0

        for trade in trades:
            total_slippage += trade.slippage

        return total_slippage


class HedgingRiskManager:
    """Main hedging and risk management coordinator"""

    def __init__(self, risk_limits: RiskLimits, hedging_config: HedgingConfig):
        self.risk_limits = risk_limits
        self.hedging_config = hedging_config

        # Initialize components
        self.risk_monitor = RiskMonitor(risk_limits)
        self.delta_hedger = DeltaHedger(hedging_config)
        self.gamma_scalper = GammaScalper(hedging_config)
        self.vega_hedger = VegaHedger(hedging_config)
        self.pnl_engine = PnLAttributionEngine()

        # State management
        self.is_active = True
        self.hedge_queue: deque = deque()
        self.execution_callback: Optional[Callable] = None

    def update_portfolio_state(self, portfolio_greeks: Greeks, positions: Dict[str, Position],
                             market_data: Dict[str, MarketData], trades: List[Trade]):
        """Update portfolio state and generate hedge recommendations"""

        current_prices = {symbol: data.underlying_price or data.last for symbol, data in market_data.items()}
        current_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in positions.values())

        # Check risk limits
        violations = self.risk_monitor.check_risk_limits(portfolio_greeks, positions, current_pnl, market_data)

        if violations:
            self._handle_risk_violations(violations, portfolio_greeks, positions)

        # Calculate P&L attribution
        time_elapsed = 60.0  # Assume 1 minute updates
        attribution = self.pnl_engine.calculate_attribution(
            portfolio_greeks, positions, current_prices, trades, time_elapsed
        )

        # Generate hedge recommendations
        hedge_orders = self._generate_hedge_recommendations(portfolio_greeks, positions, market_data)

        # Execute hedges if conditions are met
        if self.execution_callback and hedge_orders:
            for hedge_order in hedge_orders:
                self.execution_callback(hedge_order)

        return {
            'risk_violations': violations,
            'pnl_attribution': attribution,
            'hedge_orders': hedge_orders,
            'portfolio_greeks': portfolio_greeks
        }

    def set_execution_callback(self, callback: Callable):
        """Set callback for hedge execution"""
        self.execution_callback = callback

    def _generate_hedge_recommendations(self, portfolio_greeks: Greeks, positions: Dict[str, Position],
                                      market_data: Dict[str, MarketData]) -> List[HedgeOrder]:
        """Generate all hedge recommendations"""

        hedge_orders = []

        # Delta hedging
        if self.delta_hedger.should_hedge_now(portfolio_greeks, {}):
            delta_hedge = self.delta_hedger.calculate_hedge_requirement(portfolio_greeks, positions)
            if delta_hedge:
                hedge_orders.append(delta_hedge)

        # Gamma scalping
        if self.gamma_scalper.should_scalp_now(portfolio_greeks):
            current_prices = {symbol: data.underlying_price or data.last for symbol, data in market_data.items()}
            previous_prices = {}  # Would need to track previous prices

            gamma_scalps = self.gamma_scalper.calculate_gamma_scalp(portfolio_greeks, current_prices, previous_prices)
            hedge_orders.extend(gamma_scalps)

        # Vega hedging
        if self.vega_hedger.should_hedge_vega_now(portfolio_greeks):
            available_options = {}  # Would need available options data
            vega_hedges = self.vega_hedger.calculate_vega_hedge(portfolio_greeks, available_options)
            hedge_orders.extend(vega_hedges)

        return hedge_orders

    def _handle_risk_violations(self, violations: List[str], portfolio_greeks: Greeks, positions: Dict[str, Position]):
        """Handle risk limit violations"""

        print(f"RISK VIOLATION: {violations}")

        # Generate emergency hedge orders
        emergency_hedges = []

        # Emergency delta hedge
        if abs(portfolio_greeks.delta) > self.risk_limits.max_portfolio_delta:
            emergency_hedge = HedgeOrder(
                symbol=self.delta_hedger._get_primary_underlying(positions),
                side=OrderSide.ASK if portfolio_greeks.delta > 0 else OrderSide.BID,
                quantity=int(abs(portfolio_greeks.delta * 0.9)),  # Aggressive hedge
                order_type=OrderType.MARKET,
                hedge_type="emergency_delta",
                urgency=1.0
            )
            emergency_hedges.append(emergency_hedge)

        # Execute emergency hedges immediately
        if self.execution_callback:
            for hedge in emergency_hedges:
                self.execution_callback(hedge)

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""

        recent_violations = [v for v in self.risk_monitor.risk_violations
                           if v['timestamp'] > datetime.now() - timedelta(hours=1)]

        pnl_summary = self.pnl_engine.get_attribution_summary(24)

        return {
            'recent_violations': recent_violations,
            'pnl_attribution': pnl_summary,
            'hedge_history_delta': list(self.delta_hedger.hedge_history)[-10:] if self.delta_hedger.hedge_history else [],
            'hedge_history_gamma': list(self.gamma_scalper.scalp_history)[-10:] if self.gamma_scalper.scalp_history else [],
            'risk_limits': {
                'max_portfolio_delta': self.risk_limits.max_portfolio_delta,
                'max_portfolio_gamma': self.risk_limits.max_portfolio_gamma,
                'max_portfolio_vega': self.risk_limits.max_portfolio_vega,
                'max_daily_loss': self.risk_limits.max_daily_loss
            }
        }


# Factory functions
def create_hedging_risk_manager(config: Optional[Dict[str, Any]] = None) -> HedgingRiskManager:
    """Create a configured hedging and risk manager"""

    # Default risk limits
    risk_limits = RiskLimits(
        max_portfolio_delta=1000.0,
        max_portfolio_gamma=500.0,
        max_portfolio_vega=2000.0,
        max_daily_loss=10000.0
    )

    # Default hedging config
    hedging_config = HedgingConfig(
        delta_hedge_threshold=50.0,
        gamma_scalp_threshold=10.0,
        continuous_hedge_interval_seconds=30
    )

    if config:
        # Update configs from provided parameters
        for key, value in config.items():
            if hasattr(risk_limits, key):
                setattr(risk_limits, key, value)
            elif hasattr(hedging_config, key):
                setattr(hedging_config, key, value)

    return HedgingRiskManager(risk_limits, hedging_config)


# Example usage
if __name__ == "__main__":
    # Create hedging manager
    manager = create_hedging_risk_manager()

    # Example portfolio state
    portfolio_greeks = Greeks(
        delta=150.0,  # Above threshold
        gamma=25.0,
        theta=-50.0,
        vega=300.0,
        rho=10.0,
        underlying_price=100.0,
        timestamp=datetime.now()
    )

    positions = {
        "AAPL_231215C150": Position("AAPL_231215C150", 100, 5.0, 50000, 1000, 500),
        "AAPL_231215P140": Position("AAPL_231215P140", -50, 3.0, -15000, -500, 200)
    }

    market_data = {
        "AAPL": MarketData("AAPL", datetime.now(), 149.50, 150.50, 150.0, 100, 100, 1000000, 0, underlying_price=150.0)
    }

    # Update portfolio and get recommendations
    result = manager.update_portfolio_state(portfolio_greeks, positions, market_data, [])

    print("Risk violations:", result['risk_violations'])
    print("Hedge orders:", len(result['hedge_orders']))
    print("P&L attribution:", result['pnl_attribution'].total_pnl)