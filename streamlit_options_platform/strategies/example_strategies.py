"""
Example Options Trading Strategies
Professional strategy implementations for backtesting and validation
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from backtesting.backtesting_engine import (
    Order, OptionContract, OptionType, OrderType, MarketData
)

class DeltaNeutralStrategy:
    """
    Delta-neutral options strategy with dynamic hedging
    """

    def __init__(self, target_delta: float = 0.0, rebalance_threshold: float = 10.0,
                 max_position_size: float = 0.1):
        self.target_delta = target_delta
        self.rebalance_threshold = rebalance_threshold
        self.max_position_size = max_position_size
        self.logger = logging.getLogger(__name__)

    def generate_orders(self, market_data: MarketData, positions: Dict, capital: float) -> List[Order]:
        """Generate orders for delta-neutral strategy"""
        orders = []

        # Calculate current portfolio delta
        current_delta = self._calculate_portfolio_delta(market_data, positions)

        # Check if rebalancing is needed
        delta_deviation = abs(current_delta - self.target_delta)

        if delta_deviation > self.rebalance_threshold:
            # Determine rebalancing orders
            orders.extend(self._generate_rebalancing_orders(
                market_data, positions, current_delta, capital
            ))

        # Check for new position opportunities
        if len(positions) < 5:  # Limit number of active positions
            new_orders = self._generate_new_position_orders(market_data, capital)
            orders.extend(new_orders)

        return orders

    def _calculate_portfolio_delta(self, market_data: MarketData, positions: Dict) -> float:
        """Calculate current portfolio delta"""
        total_delta = 0

        for position_id, position in positions.items():
            # Get option greeks
            time_to_expiry = (position.contract.expiry - market_data.timestamp).days / 365.0

            if time_to_expiry > 0:
                # Simplified delta calculation
                if position.contract.option_type == OptionType.CALL:
                    delta = 0.5  # Simplified ATM delta
                else:
                    delta = -0.5

                total_delta += position.quantity * delta * 100

        return total_delta

    def _generate_rebalancing_orders(self, market_data: MarketData, positions: Dict,
                                   current_delta: float, capital: float) -> List[Order]:
        """Generate orders to rebalance delta"""
        orders = []

        delta_to_hedge = current_delta - self.target_delta

        # Create hedging order
        if abs(delta_to_hedge) > self.rebalance_threshold:
            # Use closest-to-money options for hedging
            strike = market_data.underlying_price
            expiry = market_data.timestamp + timedelta(days=30)

            if delta_to_hedge > 0:  # Need to sell delta
                contract = OptionContract("SPY", strike, expiry, OptionType.PUT)
                quantity = int(abs(delta_to_hedge) / 50)  # Simplified sizing
                order_type = OrderType.BUY
            else:  # Need to buy delta
                contract = OptionContract("SPY", strike, expiry, OptionType.CALL)
                quantity = int(abs(delta_to_hedge) / 50)
                order_type = OrderType.BUY

            if quantity > 0:
                orders.append(Order(contract, quantity, order_type))

        return orders

    def _generate_new_position_orders(self, market_data: MarketData, capital: float) -> List[Order]:
        """Generate orders for new positions"""
        orders = []

        # Only enter new positions if conditions are favorable
        if np.random.random() < 0.1:  # 10% chance to enter new position
            # Create straddle position
            strike = market_data.underlying_price
            expiry = market_data.timestamp + timedelta(days=45)

            call_contract = OptionContract("SPY", strike, expiry, OptionType.CALL)
            put_contract = OptionContract("SPY", strike, expiry, OptionType.PUT)

            quantity = max(1, int(capital * self.max_position_size / 100000))

            orders.append(Order(call_contract, quantity, OrderType.BUY))
            orders.append(Order(put_contract, quantity, OrderType.BUY))

        return orders

class IronCondorStrategy:
    """
    Iron Condor options strategy for range-bound markets
    """

    def __init__(self, wing_width: float = 10.0, target_dte: int = 45,
                 profit_target: float = 0.5, stop_loss: float = 2.0):
        self.wing_width = wing_width
        self.target_dte = target_dte
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.logger = logging.getLogger(__name__)

    def generate_orders(self, market_data: MarketData, positions: Dict, capital: float) -> List[Order]:
        """Generate orders for iron condor strategy"""
        orders = []

        # Check existing positions for profit/loss management
        orders.extend(self._manage_existing_positions(market_data, positions))

        # Look for new entry opportunities
        if len(positions) < 3:  # Limit concurrent positions
            new_orders = self._generate_entry_orders(market_data, capital)
            orders.extend(new_orders)

        return orders

    def _manage_existing_positions(self, market_data: MarketData, positions: Dict) -> List[Order]:
        """Manage existing iron condor positions"""
        orders = []

        # Group positions by expiry
        position_groups = self._group_positions_by_expiry(positions)

        for expiry, group_positions in position_groups.items():
            if self._is_iron_condor_group(group_positions):
                # Check if we should close the position
                pnl_pct = self._calculate_position_pnl(market_data, group_positions)

                if pnl_pct >= self.profit_target or pnl_pct <= -self.stop_loss:
                    # Close the position
                    orders.extend(self._close_iron_condor(group_positions))

        return orders

    def _generate_entry_orders(self, market_data: MarketData, capital: float) -> List[Order]:
        """Generate entry orders for new iron condor"""
        orders = []

        # Check market conditions (simplified)
        current_price = market_data.underlying_price
        expiry = market_data.timestamp + timedelta(days=self.target_dte)

        # Define strikes
        call_short_strike = current_price + 5
        call_long_strike = call_short_strike + self.wing_width
        put_short_strike = current_price - 5
        put_long_strike = put_short_strike - self.wing_width

        # Create iron condor
        quantity = max(1, int(capital * 0.05 / 10000))  # Risk 5% of capital

        orders.extend([
            # Short call spread
            Order(OptionContract("SPY", call_short_strike, expiry, OptionType.CALL),
                  -quantity, OrderType.SELL),
            Order(OptionContract("SPY", call_long_strike, expiry, OptionType.CALL),
                  quantity, OrderType.BUY),

            # Short put spread
            Order(OptionContract("SPY", put_short_strike, expiry, OptionType.PUT),
                  -quantity, OrderType.SELL),
            Order(OptionContract("SPY", put_long_strike, expiry, OptionType.PUT),
                  quantity, OrderType.BUY)
        ])

        return orders

    def _group_positions_by_expiry(self, positions: Dict) -> Dict[datetime, List]:
        """Group positions by expiration date"""
        groups = {}
        for position in positions.values():
            expiry = position.contract.expiry
            if expiry not in groups:
                groups[expiry] = []
            groups[expiry].append(position)
        return groups

    def _is_iron_condor_group(self, positions: List) -> bool:
        """Check if positions form an iron condor"""
        return len(positions) == 4  # Simplified check

    def _calculate_position_pnl(self, market_data: MarketData, positions: List) -> float:
        """Calculate position P&L percentage"""
        # Simplified P&L calculation
        return np.random.uniform(-1.0, 1.0)  # Random P&L for demo

    def _close_iron_condor(self, positions: List) -> List[Order]:
        """Generate orders to close iron condor position"""
        orders = []
        for position in positions:
            # Reverse the position
            close_order = Order(
                position.contract,
                -position.quantity,
                OrderType.SELL if position.quantity > 0 else OrderType.BUY
            )
            orders.append(close_order)
        return orders

class VolatilityTradingStrategy:
    """
    Volatility-based trading strategy using straddles and strangles
    """

    def __init__(self, vol_threshold_low: float = 0.15, vol_threshold_high: float = 0.35,
                 target_dte: int = 30):
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.target_dte = target_dte
        self.logger = logging.getLogger(__name__)

    def generate_orders(self, market_data: MarketData, positions: Dict, capital: float) -> List[Order]:
        """Generate orders based on volatility analysis"""
        orders = []

        # Estimate current implied volatility (simplified)
        current_iv = self._estimate_implied_volatility(market_data)

        # Volatility-based strategy logic
        if current_iv < self.vol_threshold_low:
            # Low volatility - buy volatility (long straddle)
            orders.extend(self._buy_volatility_orders(market_data, capital))
        elif current_iv > self.vol_threshold_high:
            # High volatility - sell volatility (short strangle)
            orders.extend(self._sell_volatility_orders(market_data, capital))

        return orders

    def _estimate_implied_volatility(self, market_data: MarketData) -> float:
        """Estimate implied volatility from market data"""
        # Simplified IV estimation
        base_vol = 0.25
        random_component = np.random.normal(0, 0.05)
        return max(0.1, base_vol + random_component)

    def _buy_volatility_orders(self, market_data: MarketData, capital: float) -> List[Order]:
        """Generate orders to buy volatility"""
        orders = []

        strike = market_data.underlying_price
        expiry = market_data.timestamp + timedelta(days=self.target_dte)
        quantity = max(1, int(capital * 0.02 / 10000))

        # Long straddle
        orders.extend([
            Order(OptionContract("SPY", strike, expiry, OptionType.CALL),
                  quantity, OrderType.BUY),
            Order(OptionContract("SPY", strike, expiry, OptionType.PUT),
                  quantity, OrderType.BUY)
        ])

        return orders

    def _sell_volatility_orders(self, market_data: MarketData, capital: float) -> List[Order]:
        """Generate orders to sell volatility"""
        orders = []

        current_price = market_data.underlying_price
        expiry = market_data.timestamp + timedelta(days=self.target_dte)
        quantity = max(1, int(capital * 0.03 / 10000))

        # Short strangle
        call_strike = current_price + 10
        put_strike = current_price - 10

        orders.extend([
            Order(OptionContract("SPY", call_strike, expiry, OptionType.CALL),
                  -quantity, OrderType.SELL),
            Order(OptionContract("SPY", put_strike, expiry, OptionType.PUT),
                  -quantity, OrderType.SELL)
        ])

        return orders

class MomentumStrategy:
    """
    Momentum-based options strategy using directional bets
    """

    def __init__(self, momentum_threshold: float = 0.02, lookback_days: int = 5):
        self.momentum_threshold = momentum_threshold
        self.lookback_days = lookback_days
        self.price_history = []
        self.logger = logging.getLogger(__name__)

    def generate_orders(self, market_data: MarketData, positions: Dict, capital: float) -> List[Order]:
        """Generate momentum-based orders"""
        orders = []

        # Update price history
        self.price_history.append(market_data.underlying_price)
        if len(self.price_history) > self.lookback_days:
            self.price_history.pop(0)

        # Calculate momentum
        if len(self.price_history) >= self.lookback_days:
            momentum = self._calculate_momentum()

            if abs(momentum) > self.momentum_threshold:
                orders.extend(self._generate_momentum_orders(
                    market_data, momentum, capital
                ))

        return orders

    def _calculate_momentum(self) -> float:
        """Calculate price momentum"""
        if len(self.price_history) < 2:
            return 0

        return (self.price_history[-1] - self.price_history[0]) / self.price_history[0]

    def _generate_momentum_orders(self, market_data: MarketData,
                                momentum: float, capital: float) -> List[Order]:
        """Generate orders based on momentum"""
        orders = []

        expiry = market_data.timestamp + timedelta(days=21)
        quantity = max(1, int(capital * 0.04 / 10000))

        if momentum > self.momentum_threshold:
            # Bullish momentum - buy calls
            strike = market_data.underlying_price + 2
            orders.append(Order(
                OptionContract("SPY", strike, expiry, OptionType.CALL),
                quantity, OrderType.BUY
            ))
        elif momentum < -self.momentum_threshold:
            # Bearish momentum - buy puts
            strike = market_data.underlying_price - 2
            orders.append(Order(
                OptionContract("SPY", strike, expiry, OptionType.PUT),
                quantity, OrderType.BUY
            ))

        return orders

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'delta_neutral': DeltaNeutralStrategy,
    'iron_condor': IronCondorStrategy,
    'volatility_trading': VolatilityTradingStrategy,
    'momentum': MomentumStrategy
}

def get_strategy(strategy_name: str, **kwargs):
    """Get strategy instance by name"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return STRATEGY_REGISTRY[strategy_name](**kwargs)

def list_available_strategies() -> List[str]:
    """List all available strategies"""
    return list(STRATEGY_REGISTRY.keys())