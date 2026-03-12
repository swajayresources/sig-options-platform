"""
Sophisticated Market Making Strategy Framework for Options Trading

This module implements advanced market making strategies including delta-neutral
market making, volatility arbitrage, statistical arbitrage, and various spread
strategies with comprehensive risk management and automated execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings
from enum import Enum
import threading
import time
from collections import defaultdict, deque


class OrderSide(Enum):
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OptionContract:
    """Option contract specification"""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    multiplier: int = 100

    def __hash__(self):
        return hash((self.symbol, self.strike, self.expiry, self.option_type))


@dataclass
class MarketData:
    """Real-time market data container"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    volume: int
    open_interest: int
    implied_vol: Optional[float] = None
    underlying_price: Optional[float] = None


@dataclass
class Greeks:
    """Option Greeks container"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float
    timestamp: datetime


@dataclass
class Position:
    """Position tracking container"""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    greeks: Optional[Greeks] = None


@dataclass
class Quote:
    """Market making quote"""
    symbol: str
    side: OrderSide
    price: float
    size: int
    timestamp: datetime
    strategy_id: str
    confidence: float = 1.0
    max_position: Optional[int] = None


@dataclass
class Trade:
    """Executed trade record"""
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    strategy_id: str
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class StrategyConfig:
    """Configuration for market making strategies"""
    max_position_size: int = 100
    max_portfolio_delta: float = 1000.0
    max_portfolio_gamma: float = 100.0
    max_portfolio_vega: float = 500.0
    min_spread_width: float = 0.01
    max_spread_width: float = 0.50
    inventory_target: float = 0.0
    risk_free_rate: float = 0.05
    hedge_frequency_seconds: int = 30
    quote_refresh_seconds: int = 5
    max_adverse_selection: float = 0.02
    profit_target_bps: int = 10
    stop_loss_bps: int = 50


class MarketMakingStrategy(ABC):
    """Abstract base class for market making strategies"""

    def __init__(self, strategy_id: str, config: StrategyConfig):
        self.strategy_id = strategy_id
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.quotes: Dict[str, List[Quote]] = defaultdict(list)
        self.trades: List[Trade] = []
        self.portfolio_greeks = Greeks(0, 0, 0, 0, 0, 0, datetime.now())
        self.is_active = False
        self.pnl_history: List[Tuple[datetime, float]] = []

    @abstractmethod
    def generate_quotes(self, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate bid/ask quotes based on market data"""
        pass

    @abstractmethod
    def should_hedge(self, portfolio_greeks: Greeks) -> bool:
        """Determine if hedging is needed"""
        pass

    @abstractmethod
    def calculate_hedge_orders(self, portfolio_greeks: Greeks) -> List[Tuple[str, OrderSide, int, float]]:
        """Calculate required hedge orders"""
        pass

    def update_position(self, trade: Trade):
        """Update position after trade execution"""
        symbol = trade.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )

        position = self.positions[symbol]
        old_quantity = position.quantity

        # Update quantity based on trade side
        if trade.side == OrderSide.BID:  # We bought
            new_quantity = position.quantity + trade.quantity
        else:  # We sold
            new_quantity = position.quantity - trade.quantity

        # Calculate realized P&L for closing trades
        if old_quantity != 0 and np.sign(old_quantity) != np.sign(new_quantity):
            closing_quantity = min(abs(old_quantity), trade.quantity)
            if trade.side == OrderSide.ASK:  # Selling from long position
                realized_pnl = closing_quantity * (trade.price - position.avg_price)
            else:  # Buying to cover short position
                realized_pnl = closing_quantity * (position.avg_price - trade.price)
            position.realized_pnl += realized_pnl

        # Update average price for remaining position
        if new_quantity != 0:
            if old_quantity == 0 or np.sign(old_quantity) != np.sign(new_quantity):
                position.avg_price = trade.price
            else:
                total_cost = old_quantity * position.avg_price + trade.quantity * trade.price * (1 if trade.side == OrderSide.BID else -1)
                position.avg_price = total_cost / new_quantity

        position.quantity = new_quantity
        self.trades.append(trade)

    def calculate_portfolio_greeks(self, market_data: Dict[str, MarketData]) -> Greeks:
        """Calculate aggregate portfolio Greeks"""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        for symbol, position in self.positions.items():
            if position.quantity != 0 and position.greeks:
                total_delta += position.quantity * position.greeks.delta
                total_gamma += position.quantity * position.greeks.gamma
                total_theta += position.quantity * position.greeks.theta
                total_vega += position.quantity * position.greeks.vega
                total_rho += position.quantity * position.greeks.rho

        underlying_price = 0.0
        if market_data:
            # Use first available underlying price
            for data in market_data.values():
                if data.underlying_price:
                    underlying_price = data.underlying_price
                    break

        self.portfolio_greeks = Greeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho,
            underlying_price=underlying_price,
            timestamp=datetime.now()
        )

        return self.portfolio_greeks


class DeltaNeutralMarketMaker(MarketMakingStrategy):
    """Delta-neutral market making strategy with continuous hedging"""

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self.delta_tolerance = 10.0  # Maximum portfolio delta before hedging
        self.gamma_target = 0.0  # Target gamma exposure
        self.hedge_ratio = 0.8  # Proportion of delta to hedge

    def generate_quotes(self, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate delta-neutral market making quotes"""
        quotes = []

        for symbol, data in market_data.items():
            if not self._should_quote(symbol, data):
                continue

            # Calculate theoretical value and Greeks
            theo_value, greeks = self._calculate_theoretical_value(symbol, data)

            if theo_value is None:
                continue

            # Adjust quotes based on inventory and Greeks
            inventory_skew = self._calculate_inventory_skew(symbol)
            greeks_skew = self._calculate_greeks_skew(greeks)

            # Calculate spreads
            base_spread = self._calculate_base_spread(data)
            adjusted_spread = base_spread * (1 + abs(inventory_skew) + abs(greeks_skew))

            # Generate bid/ask prices
            bid_price = theo_value - adjusted_spread / 2 + inventory_skew
            ask_price = theo_value + adjusted_spread / 2 + inventory_skew

            # Ensure minimum tick size and spread
            bid_price = self._round_to_tick(bid_price)
            ask_price = self._round_to_tick(ask_price)

            if ask_price - bid_price < self.config.min_spread_width:
                mid = (bid_price + ask_price) / 2
                bid_price = mid - self.config.min_spread_width / 2
                ask_price = mid + self.config.min_spread_width / 2

            # Calculate position sizes
            bid_size, ask_size = self._calculate_quote_sizes(symbol, greeks)

            if bid_size > 0:
                quotes.append(Quote(
                    symbol=symbol,
                    side=OrderSide.BID,
                    price=bid_price,
                    size=bid_size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=self._calculate_confidence(data, greeks)
                ))

            if ask_size > 0:
                quotes.append(Quote(
                    symbol=symbol,
                    side=OrderSide.ASK,
                    price=ask_price,
                    size=ask_size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=self._calculate_confidence(data, greeks)
                ))

        return quotes

    def should_hedge(self, portfolio_greeks: Greeks) -> bool:
        """Determine if delta hedging is needed"""
        return abs(portfolio_greeks.delta) > self.delta_tolerance

    def calculate_hedge_orders(self, portfolio_greeks: Greeks) -> List[Tuple[str, OrderSide, int, float]]:
        """Calculate delta hedge orders"""
        hedge_orders = []

        if not self.should_hedge(portfolio_greeks):
            return hedge_orders

        # Calculate required hedge quantity
        hedge_quantity = int(portfolio_greeks.delta * self.hedge_ratio)

        if hedge_quantity == 0:
            return hedge_orders

        # Create hedge order for underlying
        underlying_symbol = self._get_underlying_symbol()
        side = OrderSide.ASK if hedge_quantity > 0 else OrderSide.BID
        quantity = abs(hedge_quantity)

        hedge_orders.append((underlying_symbol, side, quantity, 0.0))  # Market order

        return hedge_orders

    def _should_quote(self, symbol: str, data: MarketData) -> bool:
        """Determine if we should quote this option"""
        # Check basic market data quality
        if data.bid <= 0 or data.ask <= 0 or data.ask <= data.bid:
            return False

        # Check position limits
        current_position = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0)).quantity
        if abs(current_position) >= self.config.max_position_size:
            return False

        # Check portfolio risk limits
        if abs(self.portfolio_greeks.delta) > self.config.max_portfolio_delta:
            return False

        return True

    def _calculate_theoretical_value(self, symbol: str, data: MarketData) -> Tuple[Optional[float], Optional[Greeks]]:
        """Calculate theoretical option value and Greeks"""
        # Simplified Black-Scholes calculation
        # In practice, this would use a sophisticated pricing model

        if not data.implied_vol or not data.underlying_price:
            return None, None

        # Parse option details from symbol (simplified)
        contract = self._parse_option_symbol(symbol)
        if not contract:
            return None, None

        # Calculate time to expiry
        time_to_expiry = (contract.expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)

        if time_to_expiry <= 0:
            return None, None

        # Black-Scholes calculation (simplified)
        S = data.underlying_price
        K = contract.strike
        r = self.config.risk_free_rate
        vol = data.implied_vol
        T = time_to_expiry

        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)

        from scipy.stats import norm

        if contract.option_type.lower() == 'call':
            theo_value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            theo_value = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)

        # Calculate Greeks
        gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
        theta = (-S*norm.pdf(d1)*vol/(2*np.sqrt(T)) -
                r*K*np.exp(-r*T)*norm.cdf(d2 if contract.option_type.lower() == 'call' else -d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = (K*T*np.exp(-r*T)*norm.cdf(d2 if contract.option_type.lower() == 'call' else -d2)) / 100

        greeks = Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            underlying_price=S,
            timestamp=datetime.now()
        )

        return theo_value, greeks

    def _calculate_inventory_skew(self, symbol: str) -> float:
        """Calculate inventory-based quote skew"""
        position = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0))

        # Skew quotes away from large positions
        max_pos = self.config.max_position_size
        skew_factor = position.quantity / max_pos if max_pos > 0 else 0

        # Scale skew (positive skew moves quotes up, negative down)
        max_skew = 0.05  # 5% maximum skew
        return np.clip(skew_factor * max_skew, -max_skew, max_skew)

    def _calculate_greeks_skew(self, greeks: Greeks) -> float:
        """Calculate Greeks-based quote skew"""
        # Skew based on portfolio Greeks vs limits
        delta_skew = self.portfolio_greeks.delta / self.config.max_portfolio_delta
        gamma_skew = self.portfolio_greeks.gamma / self.config.max_portfolio_gamma

        total_skew = (delta_skew + gamma_skew) / 2
        max_skew = 0.03  # 3% maximum Greeks skew

        return np.clip(total_skew * max_skew, -max_skew, max_skew)

    def _calculate_base_spread(self, data: MarketData) -> float:
        """Calculate base spread based on market conditions"""
        # Use current market spread as baseline
        market_spread = data.ask - data.bid

        # Add minimum spread requirement
        base_spread = max(market_spread * 1.1, self.config.min_spread_width)

        # Cap at maximum spread
        return min(base_spread, self.config.max_spread_width)

    def _calculate_quote_sizes(self, symbol: str, greeks: Optional[Greeks]) -> Tuple[int, int]:
        """Calculate bid and ask sizes"""
        base_size = 10  # Base quote size

        # Adjust size based on Greeks
        if greeks:
            # Reduce size for high gamma options
            gamma_factor = max(0.1, 1.0 - abs(greeks.gamma) * 10)
            base_size = int(base_size * gamma_factor)

        # Adjust based on current position
        position = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0))
        remaining_capacity = self.config.max_position_size - abs(position.quantity)

        if remaining_capacity <= 0:
            return 0, 0

        size = min(base_size, remaining_capacity)

        # Reduce size if we're already long/short
        bid_size = size if position.quantity <= 0 else max(1, size // 2)
        ask_size = size if position.quantity >= 0 else max(1, size // 2)

        return bid_size, ask_size

    def _calculate_confidence(self, data: MarketData, greeks: Optional[Greeks]) -> float:
        """Calculate quote confidence score"""
        confidence = 1.0

        # Reduce confidence for wide spreads
        if data.ask > 0 and data.bid > 0:
            spread_ratio = (data.ask - data.bid) / ((data.ask + data.bid) / 2)
            if spread_ratio > 0.1:  # 10% spread
                confidence *= 0.5

        # Reduce confidence for high gamma
        if greeks and abs(greeks.gamma) > 0.1:
            confidence *= 0.7

        return max(0.1, confidence)

    def _round_to_tick(self, price: float) -> float:
        """Round price to minimum tick size"""
        tick_size = 0.01  # $0.01 minimum tick
        return round(price / tick_size) * tick_size

    def _parse_option_symbol(self, symbol: str) -> Optional[OptionContract]:
        """Parse option symbol to extract contract details"""
        # Simplified parser - in practice would handle various symbol formats
        # Format: AAPL_230120C150 (underlying_expiryPutCall_strike)

        try:
            parts = symbol.split('_')
            if len(parts) != 2:
                return None

            underlying = parts[0]
            option_part = parts[1]

            # Extract expiry, option type, and strike
            if 'C' in option_part:
                expiry_str, strike_str = option_part.split('C')
                option_type = 'call'
            elif 'P' in option_part:
                expiry_str, strike_str = option_part.split('P')
                option_type = 'put'
            else:
                return None

            # Parse expiry (YYMMDD format)
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

    def _get_underlying_symbol(self) -> str:
        """Get underlying symbol for hedging"""
        # Extract from first option position
        for symbol in self.positions:
            contract = self._parse_option_symbol(symbol)
            if contract:
                return contract.underlying
        return "SPY"  # Default


class VolatilityArbitrageStrategy(MarketMakingStrategy):
    """Volatility arbitrage strategy trading implied vs realized volatility"""

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self.realized_vol_window = 20  # Days for realized vol calculation
        self.vol_threshold = 0.05  # 5% vol difference threshold
        self.max_vega_exposure = 1000  # Maximum vega exposure
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 1 year of daily prices

    def generate_quotes(self, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate quotes based on volatility arbitrage opportunities"""
        quotes = []

        for symbol, data in market_data.items():
            if not data.implied_vol or not data.underlying_price:
                continue

            # Update price history
            underlying = self._get_underlying_from_option(symbol)
            if underlying:
                self.price_history[underlying].append(data.underlying_price)

            # Calculate realized volatility
            realized_vol = self._calculate_realized_volatility(underlying)
            if realized_vol is None:
                continue

            # Check for volatility arbitrage opportunity
            vol_diff = data.implied_vol - realized_vol

            if abs(vol_diff) < self.vol_threshold:
                continue  # No significant arbitrage opportunity

            # Generate arbitrage quotes
            theo_value, greeks = self._calculate_theoretical_value(symbol, data)
            if theo_value is None or greeks is None:
                continue

            # Adjust theoretical value based on volatility edge
            vol_edge = vol_diff * greeks.vega * 0.01  # Convert to dollar edge
            adjusted_theo = theo_value + vol_edge

            # Calculate spreads
            base_spread = min(0.05, data.ask - data.bid) if data.ask > data.bid else 0.05

            # Determine quote direction based on volatility arbitrage
            if vol_diff > self.vol_threshold:  # Implied > Realized, sell volatility
                ask_price = min(data.ask - 0.01, adjusted_theo + base_spread/2)
                quotes.append(Quote(
                    symbol=symbol,
                    side=OrderSide.ASK,
                    price=self._round_to_tick(ask_price),
                    size=self._calculate_arb_size(greeks),
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(vol_diff) / 0.1)
                ))

            elif vol_diff < -self.vol_threshold:  # Implied < Realized, buy volatility
                bid_price = max(data.bid + 0.01, adjusted_theo - base_spread/2)
                quotes.append(Quote(
                    symbol=symbol,
                    side=OrderSide.BID,
                    price=self._round_to_tick(bid_price),
                    size=self._calculate_arb_size(greeks),
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(vol_diff) / 0.1)
                ))

        return quotes

    def should_hedge(self, portfolio_greeks: Greeks) -> bool:
        """Hedge when delta or vega exposure is too high"""
        return (abs(portfolio_greeks.delta) > self.config.max_portfolio_delta or
                abs(portfolio_greeks.vega) > self.max_vega_exposure)

    def calculate_hedge_orders(self, portfolio_greeks: Greeks) -> List[Tuple[str, OrderSide, int, float]]:
        """Calculate hedge orders for delta and vega"""
        hedge_orders = []

        # Delta hedge
        if abs(portfolio_greeks.delta) > self.config.max_portfolio_delta:
            hedge_quantity = int(portfolio_greeks.delta * 0.8)
            if hedge_quantity != 0:
                underlying_symbol = self._get_underlying_symbol()
                side = OrderSide.ASK if hedge_quantity > 0 else OrderSide.BID
                hedge_orders.append((underlying_symbol, side, abs(hedge_quantity), 0.0))

        # Vega hedge (would require options with different expiries)
        # This is simplified - real implementation would find best hedge options

        return hedge_orders

    def _calculate_realized_volatility(self, underlying: str) -> Optional[float]:
        """Calculate realized volatility from price history"""
        if underlying not in self.price_history or len(self.price_history[underlying]) < self.realized_vol_window:
            return None

        prices = np.array(list(self.price_history[underlying]))
        if len(prices) < 2:
            return None

        # Calculate daily returns
        returns = np.diff(np.log(prices))

        # Annualized realized volatility
        realized_vol = np.std(returns) * np.sqrt(252)

        return realized_vol

    def _calculate_arb_size(self, greeks: Greeks) -> int:
        """Calculate position size for arbitrage opportunity"""
        # Base size on vega exposure
        target_vega = 100  # Target vega per trade
        size = max(1, int(target_vega / abs(greeks.vega))) if greeks.vega != 0 else 1

        # Limit by remaining vega capacity
        current_vega = abs(self.portfolio_greeks.vega)
        remaining_capacity = max(0, self.max_vega_exposure - current_vega)

        if greeks.vega != 0:
            max_size = int(remaining_capacity / abs(greeks.vega))
            size = min(size, max_size)

        return max(1, min(size, 50))  # Cap at 50 contracts

    def _get_underlying_from_option(self, option_symbol: str) -> Optional[str]:
        """Extract underlying symbol from option symbol"""
        contract = self._parse_option_symbol(option_symbol)
        return contract.underlying if contract else None

    def _calculate_theoretical_value(self, symbol: str, data: MarketData) -> Tuple[Optional[float], Optional[Greeks]]:
        """Use same Black-Scholes calculation as base class"""
        # This would typically use the same calculation as DeltaNeutralMarketMaker
        # For brevity, reusing the method
        return DeltaNeutralMarketMaker._calculate_theoretical_value(self, symbol, data)

    def _parse_option_symbol(self, symbol: str) -> Optional[OptionContract]:
        """Same parsing logic as base class"""
        return DeltaNeutralMarketMaker._parse_option_symbol(self, symbol)

    def _round_to_tick(self, price: float) -> float:
        """Same tick rounding as base class"""
        return DeltaNeutralMarketMaker._round_to_tick(self, price)

    def _get_underlying_symbol(self) -> str:
        """Same underlying symbol logic as base class"""
        return DeltaNeutralMarketMaker._get_underlying_symbol(self)


class StatisticalArbitrageStrategy(MarketMakingStrategy):
    """Statistical arbitrage across related options (e.g., calendar spreads, cross-strikes)"""

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self.correlation_window = 60  # Minutes for correlation calculation
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.correlation_window))
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.correlation_window))
        self.z_score_threshold = 2.0  # Z-score threshold for trades

    def generate_quotes(self, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate quotes based on statistical arbitrage opportunities"""
        quotes = []

        # Update price histories
        self._update_price_histories(market_data)

        # Find arbitrage opportunities
        spread_opportunities = self._find_spread_opportunities(market_data)

        for opportunity in spread_opportunities:
            spread_quotes = self._generate_spread_quotes(opportunity, market_data)
            quotes.extend(spread_quotes)

        return quotes

    def should_hedge(self, portfolio_greeks: Greeks) -> bool:
        """Hedge based on overall portfolio risk"""
        return (abs(portfolio_greeks.delta) > self.config.max_portfolio_delta or
                abs(portfolio_greeks.gamma) > self.config.max_portfolio_gamma)

    def calculate_hedge_orders(self, portfolio_greeks: Greeks) -> List[Tuple[str, OrderSide, int, float]]:
        """Calculate hedge orders"""
        # Simple delta hedge for now
        hedge_orders = []

        if abs(portfolio_greeks.delta) > self.config.max_portfolio_delta:
            hedge_quantity = int(portfolio_greeks.delta * 0.8)
            if hedge_quantity != 0:
                underlying_symbol = self._get_underlying_symbol()
                side = OrderSide.ASK if hedge_quantity > 0 else OrderSide.BID
                hedge_orders.append((underlying_symbol, side, abs(hedge_quantity), 0.0))

        return hedge_orders

    def _update_price_histories(self, market_data: Dict[str, MarketData]):
        """Update price histories for all symbols"""
        for symbol, data in market_data.items():
            mid_price = (data.bid + data.ask) / 2 if data.bid > 0 and data.ask > 0 else data.last
            self.price_history[symbol].append(mid_price)

    def _find_spread_opportunities(self, market_data: Dict[str, MarketData]) -> List[Dict]:
        """Find statistical arbitrage opportunities in spreads"""
        opportunities = []

        # Group options by underlying and expiry for calendar spreads
        options_by_underlying = defaultdict(list)
        for symbol in market_data:
            contract = self._parse_option_symbol(symbol)
            if contract:
                options_by_underlying[contract.underlying].append(symbol)

        # Check calendar spread opportunities
        for underlying, symbols in options_by_underlying.items():
            calendar_opps = self._find_calendar_spread_opportunities(symbols, market_data)
            opportunities.extend(calendar_opps)

            # Check strike spread opportunities
            strike_opps = self._find_strike_spread_opportunities(symbols, market_data)
            opportunities.extend(strike_opps)

        return opportunities

    def _find_calendar_spread_opportunities(self, symbols: List[str], market_data: Dict[str, MarketData]) -> List[Dict]:
        """Find calendar spread arbitrage opportunities"""
        opportunities = []

        # Group by strike and option type
        by_strike_type = defaultdict(list)
        for symbol in symbols:
            contract = self._parse_option_symbol(symbol)
            if contract:
                key = (contract.strike, contract.option_type)
                by_strike_type[key].append((symbol, contract))

        # Look for calendar spread opportunities
        for (strike, option_type), contracts in by_strike_type.items():
            if len(contracts) < 2:
                continue

            # Sort by expiry
            contracts.sort(key=lambda x: x[1].expiry)

            for i in range(len(contracts) - 1):
                near_symbol, near_contract = contracts[i]
                far_symbol, far_contract = contracts[i + 1]

                if near_symbol not in market_data or far_symbol not in market_data:
                    continue

                # Calculate calendar spread value
                spread_key = f"{near_symbol}_{far_symbol}_calendar"
                current_spread = self._calculate_calendar_spread_value(
                    market_data[near_symbol], market_data[far_symbol]
                )

                if current_spread is None:
                    continue

                self.spread_history[spread_key].append(current_spread)

                if len(self.spread_history[spread_key]) < 20:  # Need history
                    continue

                # Calculate z-score
                spread_values = np.array(list(self.spread_history[spread_key]))
                z_score = (current_spread - np.mean(spread_values)) / np.std(spread_values)

                if abs(z_score) > self.z_score_threshold:
                    opportunities.append({
                        'type': 'calendar',
                        'near_symbol': near_symbol,
                        'far_symbol': far_symbol,
                        'z_score': z_score,
                        'current_spread': current_spread,
                        'mean_spread': np.mean(spread_values),
                        'direction': 'sell' if z_score > 0 else 'buy'
                    })

        return opportunities

    def _find_strike_spread_opportunities(self, symbols: List[str], market_data: Dict[str, MarketData]) -> List[Dict]:
        """Find strike spread arbitrage opportunities"""
        opportunities = []

        # Group by expiry and option type
        by_expiry_type = defaultdict(list)
        for symbol in symbols:
            contract = self._parse_option_symbol(symbol)
            if contract:
                key = (contract.expiry, contract.option_type)
                by_expiry_type[key].append((symbol, contract))

        # Look for strike spread opportunities
        for (expiry, option_type), contracts in by_expiry_type.items():
            if len(contracts) < 2:
                continue

            # Sort by strike
            contracts.sort(key=lambda x: x[1].strike)

            for i in range(len(contracts) - 1):
                low_symbol, low_contract = contracts[i]
                high_symbol, high_contract = contracts[i + 1]

                if low_symbol not in market_data or high_symbol not in market_data:
                    continue

                # Calculate strike spread value
                spread_key = f"{low_symbol}_{high_symbol}_strike"
                current_spread = self._calculate_strike_spread_value(
                    market_data[low_symbol], market_data[high_symbol],
                    low_contract, high_contract
                )

                if current_spread is None:
                    continue

                self.spread_history[spread_key].append(current_spread)

                if len(self.spread_history[spread_key]) < 20:
                    continue

                # Calculate z-score
                spread_values = np.array(list(self.spread_history[spread_key]))
                z_score = (current_spread - np.mean(spread_values)) / np.std(spread_values)

                if abs(z_score) > self.z_score_threshold:
                    opportunities.append({
                        'type': 'strike',
                        'low_symbol': low_symbol,
                        'high_symbol': high_symbol,
                        'z_score': z_score,
                        'current_spread': current_spread,
                        'mean_spread': np.mean(spread_values),
                        'direction': 'sell' if z_score > 0 else 'buy'
                    })

        return opportunities

    def _calculate_calendar_spread_value(self, near_data: MarketData, far_data: MarketData) -> Optional[float]:
        """Calculate calendar spread value (far - near)"""
        if near_data.bid <= 0 or near_data.ask <= 0 or far_data.bid <= 0 or far_data.ask <= 0:
            return None

        near_mid = (near_data.bid + near_data.ask) / 2
        far_mid = (far_data.bid + far_data.ask) / 2

        return far_mid - near_mid

    def _calculate_strike_spread_value(self, low_data: MarketData, high_data: MarketData,
                                     low_contract: OptionContract, high_contract: OptionContract) -> Optional[float]:
        """Calculate strike spread value"""
        if low_data.bid <= 0 or low_data.ask <= 0 or high_data.bid <= 0 or high_data.ask <= 0:
            return None

        low_mid = (low_data.bid + low_data.ask) / 2
        high_mid = (high_data.bid + high_data.ask) / 2

        # For calls: high strike - low strike spread value
        # For puts: low strike - high strike spread value
        if low_contract.option_type.lower() == 'call':
            return high_mid - low_mid
        else:
            return low_mid - high_mid

    def _generate_spread_quotes(self, opportunity: Dict, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate quotes for spread opportunities"""
        quotes = []

        if opportunity['type'] == 'calendar':
            quotes.extend(self._generate_calendar_quotes(opportunity, market_data))
        elif opportunity['type'] == 'strike':
            quotes.extend(self._generate_strike_quotes(opportunity, market_data))

        return quotes

    def _generate_calendar_quotes(self, opportunity: Dict, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate calendar spread quotes"""
        quotes = []

        near_symbol = opportunity['near_symbol']
        far_symbol = opportunity['far_symbol']
        direction = opportunity['direction']

        # Position size based on z-score strength
        size = min(10, max(1, int(abs(opportunity['z_score']))))

        if direction == 'sell':  # Sell calendar (sell far, buy near)
            # Sell far month
            if far_symbol in market_data:
                quotes.append(Quote(
                    symbol=far_symbol,
                    side=OrderSide.ASK,
                    price=market_data[far_symbol].bid + 0.01,  # Aggressive sell
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

            # Buy near month
            if near_symbol in market_data:
                quotes.append(Quote(
                    symbol=near_symbol,
                    side=OrderSide.BID,
                    price=market_data[near_symbol].ask - 0.01,  # Aggressive buy
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

        else:  # Buy calendar (buy far, sell near)
            # Buy far month
            if far_symbol in market_data:
                quotes.append(Quote(
                    symbol=far_symbol,
                    side=OrderSide.BID,
                    price=market_data[far_symbol].ask - 0.01,
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

            # Sell near month
            if near_symbol in market_data:
                quotes.append(Quote(
                    symbol=near_symbol,
                    side=OrderSide.ASK,
                    price=market_data[near_symbol].bid + 0.01,
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

        return quotes

    def _generate_strike_quotes(self, opportunity: Dict, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate strike spread quotes"""
        quotes = []

        low_symbol = opportunity['low_symbol']
        high_symbol = opportunity['high_symbol']
        direction = opportunity['direction']

        # Position size based on z-score strength
        size = min(10, max(1, int(abs(opportunity['z_score']))))

        if direction == 'sell':  # Sell spread
            # Sell low strike, buy high strike (for calls)
            if low_symbol in market_data:
                quotes.append(Quote(
                    symbol=low_symbol,
                    side=OrderSide.ASK,
                    price=market_data[low_symbol].bid + 0.01,
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

            if high_symbol in market_data:
                quotes.append(Quote(
                    symbol=high_symbol,
                    side=OrderSide.BID,
                    price=market_data[high_symbol].ask - 0.01,
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

        else:  # Buy spread
            # Buy low strike, sell high strike (for calls)
            if low_symbol in market_data:
                quotes.append(Quote(
                    symbol=low_symbol,
                    side=OrderSide.BID,
                    price=market_data[low_symbol].ask - 0.01,
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

            if high_symbol in market_data:
                quotes.append(Quote(
                    symbol=high_symbol,
                    side=OrderSide.ASK,
                    price=market_data[high_symbol].bid + 0.01,
                    size=size,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    confidence=min(1.0, abs(opportunity['z_score']) / 3.0)
                ))

        return quotes

    def _parse_option_symbol(self, symbol: str) -> Optional[OptionContract]:
        """Same parsing logic as base class"""
        return DeltaNeutralMarketMaker._parse_option_symbol(self, symbol)

    def _get_underlying_symbol(self) -> str:
        """Same underlying symbol logic as base class"""
        return DeltaNeutralMarketMaker._get_underlying_symbol(self)


class PinRiskManagementStrategy(MarketMakingStrategy):
    """Pin risk management strategy for options near expiration"""

    def __init__(self, strategy_id: str, config: StrategyConfig):
        super().__init__(strategy_id, config)
        self.pin_risk_window_hours = 24  # Hours before expiry to manage pin risk
        self.pin_tolerance = 0.02  # 2% around strike for pin risk
        self.max_gamma_exposure = 500  # Maximum gamma exposure near expiry

    def generate_quotes(self, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate quotes with pin risk management"""
        quotes = []
        current_time = datetime.now()

        for symbol, data in market_data.items():
            contract = self._parse_option_symbol(symbol)
            if not contract:
                continue

            # Check if option is near expiration
            time_to_expiry = (contract.expiry - current_time).total_seconds() / 3600  # Hours

            if time_to_expiry > self.pin_risk_window_hours:
                continue  # Not in pin risk window

            # Check if underlying is near strike (pin risk)
            if not data.underlying_price:
                continue

            pin_distance = abs(data.underlying_price - contract.strike) / contract.strike

            if pin_distance > self.pin_tolerance:
                continue  # Not close enough to strike for pin risk

            # Calculate position risk
            position = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0))

            if position.quantity == 0:
                continue  # No position to manage

            # Generate quotes to reduce pin risk exposure
            pin_quotes = self._generate_pin_risk_quotes(symbol, data, contract, position, time_to_expiry)
            quotes.extend(pin_quotes)

        return quotes

    def should_hedge(self, portfolio_greeks: Greeks) -> bool:
        """More aggressive hedging near expiration"""
        # Lower thresholds for hedging near expiry
        delta_threshold = self.config.max_portfolio_delta * 0.5
        gamma_threshold = self.max_gamma_exposure

        return (abs(portfolio_greeks.delta) > delta_threshold or
                abs(portfolio_greeks.gamma) > gamma_threshold)

    def calculate_hedge_orders(self, portfolio_greeks: Greeks) -> List[Tuple[str, OrderSide, int, float]]:
        """Calculate hedge orders with enhanced pin risk management"""
        hedge_orders = []

        # Aggressive delta hedge near expiry
        if abs(portfolio_greeks.delta) > self.config.max_portfolio_delta * 0.5:
            hedge_quantity = int(portfolio_greeks.delta * 0.9)  # More aggressive hedging
            if hedge_quantity != 0:
                underlying_symbol = self._get_underlying_symbol()
                side = OrderSide.ASK if hedge_quantity > 0 else OrderSide.BID
                hedge_orders.append((underlying_symbol, side, abs(hedge_quantity), 0.0))

        # Gamma hedge by reducing positions
        if abs(portfolio_greeks.gamma) > self.max_gamma_exposure:
            # Would generate orders to reduce high-gamma positions
            # Implementation would identify highest gamma positions and create closing orders
            pass

        return hedge_orders

    def _generate_pin_risk_quotes(self, symbol: str, data: MarketData, contract: OptionContract,
                                position: Position, time_to_expiry: float) -> List[Quote]:
        """Generate quotes to manage pin risk"""
        quotes = []

        # Calculate urgency based on time to expiry
        urgency = max(0.1, 1.0 - time_to_expiry / self.pin_risk_window_hours)

        # Determine if we want to reduce position
        should_reduce = abs(position.quantity) > 0

        if not should_reduce:
            return quotes

        # Calculate aggressive price to close position
        if position.quantity > 0:  # Long position, need to sell
            # Offer at bid price or slightly below for quick execution
            price = max(data.bid - 0.01, data.bid * 0.95) if data.bid > 0 else data.last * 0.95

            quotes.append(Quote(
                symbol=symbol,
                side=OrderSide.ASK,
                price=self._round_to_tick(price),
                size=min(abs(position.quantity), 20),  # Limit size for liquidity
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                confidence=urgency
            ))

        else:  # Short position, need to buy
            # Bid at ask price or slightly above for quick execution
            price = min(data.ask + 0.01, data.ask * 1.05) if data.ask > 0 else data.last * 1.05

            quotes.append(Quote(
                symbol=symbol,
                side=OrderSide.BID,
                price=self._round_to_tick(price),
                size=min(abs(position.quantity), 20),
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                confidence=urgency
            ))

        return quotes

    def _parse_option_symbol(self, symbol: str) -> Optional[OptionContract]:
        """Same parsing logic as base class"""
        return DeltaNeutralMarketMaker._parse_option_symbol(self, symbol)

    def _round_to_tick(self, price: float) -> float:
        """Same tick rounding as base class"""
        return DeltaNeutralMarketMaker._round_to_tick(self, price)

    def _get_underlying_symbol(self) -> str:
        """Same underlying symbol logic as base class"""
        return DeltaNeutralMarketMaker._get_underlying_symbol(self)


# Strategy factory and management
class StrategyManager:
    """Manages multiple market making strategies"""

    def __init__(self):
        self.strategies: Dict[str, MarketMakingStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.total_portfolio_greeks = Greeks(0, 0, 0, 0, 0, 0, datetime.now())

    def add_strategy(self, strategy: MarketMakingStrategy, weight: float = 1.0):
        """Add a strategy to the manager"""
        self.strategies[strategy.strategy_id] = strategy
        self.strategy_weights[strategy.strategy_id] = weight

    def remove_strategy(self, strategy_id: str):
        """Remove a strategy from the manager"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            del self.strategy_weights[strategy_id]

    def generate_consolidated_quotes(self, market_data: Dict[str, MarketData]) -> List[Quote]:
        """Generate consolidated quotes from all strategies"""
        all_quotes = []

        for strategy_id, strategy in self.strategies.items():
            if strategy.is_active:
                try:
                    strategy_quotes = strategy.generate_quotes(market_data)
                    # Apply strategy weight to quote sizes
                    weight = self.strategy_weights.get(strategy_id, 1.0)
                    for quote in strategy_quotes:
                        quote.size = max(1, int(quote.size * weight))
                    all_quotes.extend(strategy_quotes)
                except Exception as e:
                    warnings.warn(f"Strategy {strategy_id} failed to generate quotes: {e}")

        # Consolidate overlapping quotes
        consolidated_quotes = self._consolidate_quotes(all_quotes)

        return consolidated_quotes

    def update_portfolio_greeks(self, market_data: Dict[str, MarketData]):
        """Update portfolio Greeks across all strategies"""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        for strategy in self.strategies.values():
            strategy.calculate_portfolio_greeks(market_data)
            total_delta += strategy.portfolio_greeks.delta
            total_gamma += strategy.portfolio_greeks.gamma
            total_theta += strategy.portfolio_greeks.theta
            total_vega += strategy.portfolio_greeks.vega
            total_rho += strategy.portfolio_greeks.rho

        self.total_portfolio_greeks = Greeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho,
            underlying_price=market_data.get(list(market_data.keys())[0]).underlying_price if market_data else 0,
            timestamp=datetime.now()
        )

    def _consolidate_quotes(self, quotes: List[Quote]) -> List[Quote]:
        """Consolidate overlapping quotes from multiple strategies"""
        # Group quotes by symbol and side
        quote_groups = defaultdict(list)

        for quote in quotes:
            key = (quote.symbol, quote.side)
            quote_groups[key].append(quote)

        consolidated = []

        for (symbol, side), group in quote_groups.items():
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # Consolidate multiple quotes for same symbol/side
                # Use best price and sum sizes
                if side == OrderSide.BID:
                    best_quote = max(group, key=lambda q: q.price)
                else:
                    best_quote = min(group, key=lambda q: q.price)

                total_size = sum(q.size for q in group)
                avg_confidence = sum(q.confidence for q in group) / len(group)

                consolidated_quote = Quote(
                    symbol=symbol,
                    side=side,
                    price=best_quote.price,
                    size=total_size,
                    timestamp=datetime.now(),
                    strategy_id="CONSOLIDATED",
                    confidence=avg_confidence
                )

                consolidated.append(consolidated_quote)

        return consolidated


# Example factory functions
def create_delta_neutral_strategy(underlying: str = "SPY") -> DeltaNeutralMarketMaker:
    """Create a delta-neutral market making strategy"""
    config = StrategyConfig(
        max_position_size=100,
        max_portfolio_delta=500.0,
        min_spread_width=0.01,
        max_spread_width=0.25
    )

    return DeltaNeutralMarketMaker(f"delta_neutral_{underlying}", config)


def create_volatility_arbitrage_strategy(underlying: str = "SPY") -> VolatilityArbitrageStrategy:
    """Create a volatility arbitrage strategy"""
    config = StrategyConfig(
        max_position_size=50,
        max_portfolio_vega=1000.0,
        min_spread_width=0.02,
        max_spread_width=0.30
    )

    return VolatilityArbitrageStrategy(f"vol_arb_{underlying}", config)


def create_statistical_arbitrage_strategy(underlying: str = "SPY") -> StatisticalArbitrageStrategy:
    """Create a statistical arbitrage strategy"""
    config = StrategyConfig(
        max_position_size=25,
        max_portfolio_delta=300.0,
        min_spread_width=0.01,
        max_spread_width=0.20
    )

    return StatisticalArbitrageStrategy(f"stat_arb_{underlying}", config)


def create_pin_risk_strategy(underlying: str = "SPY") -> PinRiskManagementStrategy:
    """Create a pin risk management strategy"""
    config = StrategyConfig(
        max_position_size=10,
        max_portfolio_gamma=200.0,
        min_spread_width=0.01,
        max_spread_width=0.15
    )

    return PinRiskManagementStrategy(f"pin_risk_{underlying}", config)