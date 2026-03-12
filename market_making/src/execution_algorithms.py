"""
Execution Algorithm Library for Options Market Making

This module implements sophisticated execution algorithms including smart order routing,
VWAP execution, implementation shortfall minimization, hidden liquidity detection,
cross-trading, and latency arbitrage protection for optimal trade execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import threading
import queue
import warnings
from enum import Enum
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from market_making_strategies import OrderSide, OrderType, MarketData, Trade


class ExecutionAlgorithm(Enum):
    """Types of execution algorithms"""
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    VOLUME_PARTICIPATION = "volume_participation"
    LIQUIDITY_SEEKING = "liquidity_seeking"
    HIDDEN_LIQUIDITY = "hidden_liquidity"
    ICEBERG = "iceberg"
    SMART_ORDER_ROUTING = "smart_order_routing"


class ExecutionVenue(Enum):
    """Execution venues"""
    PRIMARY_EXCHANGE = "primary"
    DARK_POOL_1 = "dark_pool_1"
    DARK_POOL_2 = "dark_pool_2"
    ECN_1 = "ecn_1"
    ECN_2 = "ecn_2"
    INTERNALIZATION = "internalization"


@dataclass
class ExecutionOrder:
    """Order for execution"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    time_in_force: str = "DAY"  # DAY, IOC, FOK, GTC

    # Algorithm-specific parameters
    participation_rate: Optional[float] = None  # For volume participation
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_percentage_of_volume: float = 0.1  # 10% max of volume
    urgency: float = 0.5  # 0 = patient, 1 = aggressive

    # Venue preferences
    venue_preferences: List[ExecutionVenue] = field(default_factory=list)
    allow_dark_pools: bool = True
    allow_internalization: bool = True

    # Risk controls
    max_price_deviation: float = 0.02  # 2% max price deviation
    max_execution_time_minutes: int = 30

    order_id: str = field(default_factory=lambda: f"order_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of order execution"""
    order_id: str
    symbol: str
    total_quantity: int
    executed_quantity: int
    remaining_quantity: int
    average_price: float
    total_cost: float

    # Performance metrics
    implementation_shortfall: float
    volume_weighted_price: float
    time_weighted_price: float
    market_impact: float
    timing_cost: float
    commission_cost: float

    # Execution details
    fills: List[Trade]
    venues_used: List[ExecutionVenue]
    algorithm_used: ExecutionAlgorithm
    execution_time_seconds: float

    # Quality metrics
    fill_rate: float
    price_improvement: float
    adverse_selection: float

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VenueData:
    """Market data for a specific venue"""
    venue: ExecutionVenue
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    latency_ms: float
    fill_probability: float  # Historical fill probability
    price_improvement_bps: float  # Average price improvement
    timestamp: datetime


@dataclass
class MarketMicrostructure:
    """Market microstructure data"""
    symbol: str
    bid_ask_spread: float
    effective_spread: float
    quoted_spread: float
    depth_at_touch: int
    depth_beyond_touch: int
    order_flow_imbalance: float
    volatility_estimate: float
    recent_trade_direction: int  # +1 buy, -1 sell, 0 neutral
    hidden_liquidity_estimate: float
    timestamp: datetime


class VWAPAlgorithm:
    """Volume Weighted Average Price execution algorithm"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'participation_rate': 0.1,        # 10% of volume
            'max_participation_rate': 0.3,    # 30% max
            'min_participation_rate': 0.05,   # 5% min
            'volume_lookback_minutes': 30,    # 30 minutes volume lookback
            'price_limit_buffer': 0.01,       # 1% price buffer
            'slice_size_multiplier': 2.0,     # Slice size vs average
            'urgency_adjustment': True         # Adjust for urgency
        }

        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def execute_order(self, order: ExecutionOrder, market_data: MarketData,
                     venue_data: List[VenueData]) -> ExecutionResult:
        """Execute order using VWAP algorithm"""

        start_time = time.time()
        fills = []
        total_executed = 0
        total_cost = 0.0
        venues_used = []

        # Calculate target VWAP
        target_vwap = self._calculate_target_vwap(order.symbol, market_data)

        # Calculate execution schedule
        execution_schedule = self._calculate_execution_schedule(order, market_data)

        # Execute according to schedule
        for time_slice, target_quantity in execution_schedule:
            if total_executed >= order.quantity:
                break

            remaining_quantity = order.quantity - total_executed
            slice_quantity = min(target_quantity, remaining_quantity)

            # Execute slice
            slice_fills, slice_cost = self._execute_slice(
                order, slice_quantity, market_data, venue_data, target_vwap
            )

            fills.extend(slice_fills)
            total_cost += slice_cost
            total_executed += sum(fill.quantity for fill in slice_fills)

            # Update venues used
            for fill in slice_fills:
                if hasattr(fill, 'venue') and fill.venue not in venues_used:
                    venues_used.append(fill.venue)

            # Wait for next slice (in real implementation)
            time.sleep(0.1)  # Simulate slice timing

        # Calculate performance metrics
        avg_price = total_cost / total_executed if total_executed > 0 else 0
        implementation_shortfall = self._calculate_implementation_shortfall(
            order, avg_price, target_vwap, market_data
        )

        execution_time = time.time() - start_time

        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            total_quantity=order.quantity,
            executed_quantity=total_executed,
            remaining_quantity=order.quantity - total_executed,
            average_price=avg_price,
            total_cost=total_cost,
            implementation_shortfall=implementation_shortfall,
            volume_weighted_price=target_vwap,
            time_weighted_price=avg_price,
            market_impact=self._calculate_market_impact(order, fills, market_data),
            timing_cost=0.0,  # Would calculate based on market movement
            commission_cost=len(fills) * 0.5,  # $0.50 per fill
            fills=fills,
            venues_used=venues_used,
            algorithm_used=ExecutionAlgorithm.VWAP,
            execution_time_seconds=execution_time,
            fill_rate=total_executed / order.quantity,
            price_improvement=0.0,  # Would calculate vs benchmark
            adverse_selection=0.0   # Would calculate based on post-trade analysis
        )

    def _calculate_target_vwap(self, symbol: str, market_data: MarketData) -> float:
        """Calculate target VWAP based on historical data"""
        # Simplified - would use sophisticated volume forecasting
        return (market_data.bid + market_data.ask) / 2

    def _calculate_execution_schedule(self, order: ExecutionOrder,
                                    market_data: MarketData) -> List[Tuple[datetime, int]]:
        """Calculate execution schedule for VWAP"""

        # Estimate execution window
        if order.end_time:
            execution_window = (order.end_time - datetime.now()).total_seconds() / 60  # minutes
        else:
            execution_window = 30  # Default 30 minutes

        # Estimate volume rate
        volume_rate = market_data.volume / 60  # per minute (simplified)

        # Calculate participation rate
        participation_rate = order.participation_rate or self.config['participation_rate']

        # Adjust for urgency
        if self.config['urgency_adjustment']:
            urgency_multiplier = 1.0 + order.urgency
            participation_rate *= urgency_multiplier
            participation_rate = min(participation_rate, self.config['max_participation_rate'])

        # Create schedule
        schedule = []
        minutes_per_slice = max(1, execution_window / 10)  # 10 slices
        target_quantity_per_minute = volume_rate * participation_rate

        for i in range(int(execution_window / minutes_per_slice)):
            slice_time = datetime.now() + timedelta(minutes=i * minutes_per_slice)
            slice_quantity = int(target_quantity_per_minute * minutes_per_slice)
            schedule.append((slice_time, slice_quantity))

        return schedule

    def _execute_slice(self, order: ExecutionOrder, quantity: int,
                      market_data: MarketData, venue_data: List[VenueData],
                      target_vwap: float) -> Tuple[List[Trade], float]:
        """Execute a single slice"""

        fills = []
        total_cost = 0.0

        # Sort venues by attractiveness
        best_venues = self._rank_venues(order, venue_data, target_vwap)

        remaining_quantity = quantity

        for venue_info in best_venues:
            if remaining_quantity <= 0:
                break

            venue = venue_info['venue']
            available_size = venue_info['size']
            price = venue_info['price']

            # Calculate fill quantity
            fill_quantity = min(remaining_quantity, available_size)

            # Create fill
            fill = Trade(
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=price,
                timestamp=datetime.now(),
                strategy_id="vwap_execution",
                commission=0.5,
                slippage=0.0
            )

            # Add venue information (would be in extended Trade class)
            setattr(fill, 'venue', venue)

            fills.append(fill)
            total_cost += fill_quantity * price
            remaining_quantity -= fill_quantity

        return fills, total_cost

    def _rank_venues(self, order: ExecutionOrder, venue_data: List[VenueData],
                    target_price: float) -> List[Dict]:
        """Rank venues by attractiveness"""

        venue_scores = []

        for venue in venue_data:
            if order.side == OrderSide.BID:
                price = venue.ask
                size = venue.ask_size
            else:
                price = venue.bid
                size = venue.bid_size

            # Score based on price, size, and venue quality
            price_score = 1.0 - abs(price - target_price) / target_price
            size_score = min(1.0, size / 100)  # Normalize by typical size
            latency_score = 1.0 - venue.latency_ms / 100  # Normalize by 100ms
            fill_prob_score = venue.fill_probability

            total_score = (price_score * 0.4 + size_score * 0.2 +
                          latency_score * 0.2 + fill_prob_score * 0.2)

            venue_scores.append({
                'venue': venue.venue,
                'price': price,
                'size': size,
                'score': total_score
            })

        # Sort by score (descending)
        venue_scores.sort(key=lambda x: x['score'], reverse=True)

        return venue_scores

    def _calculate_implementation_shortfall(self, order: ExecutionOrder,
                                          avg_price: float, target_vwap: float,
                                          market_data: MarketData) -> float:
        """Calculate implementation shortfall"""

        # Benchmark price (decision price)
        benchmark_price = (market_data.bid + market_data.ask) / 2

        # Implementation shortfall = (execution_price - benchmark_price) / benchmark_price
        if order.side == OrderSide.BID:
            shortfall = (avg_price - benchmark_price) / benchmark_price
        else:
            shortfall = (benchmark_price - avg_price) / benchmark_price

        return shortfall

    def _calculate_market_impact(self, order: ExecutionOrder, fills: List[Trade],
                               market_data: MarketData) -> float:
        """Calculate market impact"""

        # Simplified market impact calculation
        # Would use more sophisticated models in practice

        if not fills:
            return 0.0

        # Estimate impact based on order size vs average volume
        order_size_ratio = order.quantity / max(market_data.volume, 1)

        # Square root impact model
        impact = np.sqrt(order_size_ratio) * 0.01  # 1% per sqrt(participation)

        return impact


class ImplementationShortfallAlgorithm:
    """Implementation Shortfall minimization algorithm"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'risk_aversion': 0.5,           # Risk aversion parameter
            'volatility_multiplier': 2.0,   # Volatility impact on timing
            'market_impact_model': 'sqrt',   # 'linear', 'sqrt', 'power'
            'temporary_impact_decay': 0.5,   # Temporary impact decay rate
            'permanent_impact_factor': 0.3,  # Permanent impact coefficient
            'max_order_rate': 0.5           # Maximum order rate
        }

    def execute_order(self, order: ExecutionOrder, market_data: MarketData,
                     microstructure: MarketMicrostructure) -> ExecutionResult:
        """Execute order using Implementation Shortfall algorithm"""

        start_time = time.time()

        # Calculate optimal trading trajectory
        optimal_trajectory = self._calculate_optimal_trajectory(
            order, market_data, microstructure
        )

        # Execute according to trajectory
        fills = []
        total_executed = 0
        total_cost = 0.0

        for time_point, target_rate in optimal_trajectory:
            if total_executed >= order.quantity:
                break

            # Calculate slice size
            time_remaining = (order.end_time or (datetime.now() + timedelta(minutes=30))) - datetime.now()
            slice_duration = min(60, time_remaining.total_seconds())  # 1 minute slices
            slice_quantity = int(target_rate * slice_duration / 60)

            remaining_quantity = order.quantity - total_executed
            slice_quantity = min(slice_quantity, remaining_quantity)

            if slice_quantity > 0:
                # Execute slice with adaptive pricing
                slice_fills, slice_cost = self._execute_adaptive_slice(
                    order, slice_quantity, market_data, microstructure
                )

                fills.extend(slice_fills)
                total_cost += slice_cost
                total_executed += sum(fill.quantity for fill in slice_fills)

            # Update market microstructure (would come from feed)
            time.sleep(0.1)  # Simulate time progression

        # Calculate performance metrics
        avg_price = total_cost / total_executed if total_executed > 0 else 0
        implementation_shortfall = self._calculate_is_components(
            order, fills, market_data, microstructure
        )

        execution_time = time.time() - start_time

        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            total_quantity=order.quantity,
            executed_quantity=total_executed,
            remaining_quantity=order.quantity - total_executed,
            average_price=avg_price,
            total_cost=total_cost,
            implementation_shortfall=implementation_shortfall['total'],
            volume_weighted_price=avg_price,
            time_weighted_price=avg_price,
            market_impact=implementation_shortfall['market_impact'],
            timing_cost=implementation_shortfall['timing_cost'],
            commission_cost=len(fills) * 0.5,
            fills=fills,
            venues_used=[ExecutionVenue.PRIMARY_EXCHANGE],
            algorithm_used=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
            execution_time_seconds=execution_time,
            fill_rate=total_executed / order.quantity,
            price_improvement=0.0,
            adverse_selection=0.0
        )

    def _calculate_optimal_trajectory(self, order: ExecutionOrder,
                                    market_data: MarketData,
                                    microstructure: MarketMicrostructure) -> List[Tuple[datetime, float]]:
        """Calculate optimal trading trajectory to minimize implementation shortfall"""

        # Estimate parameters
        volatility = microstructure.volatility_estimate
        T = 30 * 60  # 30 minutes in seconds
        Q = order.quantity

        # Risk aversion and market impact parameters
        lambda_risk = self.config['risk_aversion']
        eta = self._estimate_temporary_impact_parameter(market_data, microstructure)
        gamma = self._estimate_permanent_impact_parameter(market_data, microstructure)

        # Optimal trading rate (simplified Almgren-Chriss model)
        kappa = np.sqrt(lambda_risk * volatility**2 / eta)

        # Calculate optimal trajectory
        trajectory = []
        dt = 60  # 1 minute intervals

        for t in range(0, int(T), dt):
            time_remaining = T - t

            # Optimal trading rate
            if time_remaining > 0:
                sinh_term = np.sinh(kappa * time_remaining)
                cosh_term = np.cosh(kappa * T)
                trading_rate = (Q * kappa * sinh_term) / (np.sinh(kappa * T))
                trading_rate = min(trading_rate, Q * self.config['max_order_rate'])
            else:
                trading_rate = 0

            trajectory.append((datetime.now() + timedelta(seconds=t), trading_rate))

        return trajectory

    def _estimate_temporary_impact_parameter(self, market_data: MarketData,
                                           microstructure: MarketMicrostructure) -> float:
        """Estimate temporary market impact parameter"""

        # Based on bid-ask spread and volatility
        spread = microstructure.bid_ask_spread
        volatility = microstructure.volatility_estimate

        # Temporary impact ~ spread / sqrt(volume)
        daily_volume = market_data.volume * 6.5 * 60  # Annualize hourly volume
        eta = spread / np.sqrt(daily_volume) * 10000  # Scale factor

        return max(eta, 0.0001)  # Minimum value

    def _estimate_permanent_impact_parameter(self, market_data: MarketData,
                                           microstructure: MarketMicrostructure) -> float:
        """Estimate permanent market impact parameter"""

        # Based on price volatility and volume
        volatility = microstructure.volatility_estimate
        daily_volume = market_data.volume * 6.5 * 60

        # Permanent impact ~ volatility / volume
        gamma = volatility / np.sqrt(daily_volume) * self.config['permanent_impact_factor']

        return max(gamma, 0.00001)  # Minimum value

    def _execute_adaptive_slice(self, order: ExecutionOrder, quantity: int,
                              market_data: MarketData,
                              microstructure: MarketMicrostructure) -> Tuple[List[Trade], float]:
        """Execute slice with adaptive pricing based on market conditions"""

        fills = []
        total_cost = 0.0

        # Adaptive pricing based on urgency and market conditions
        if order.side == OrderSide.BID:
            base_price = market_data.ask
            # Adjust for market conditions
            if microstructure.order_flow_imbalance > 0.5:  # Heavy buying
                price_adjustment = microstructure.bid_ask_spread * 0.2
            else:
                price_adjustment = 0
            exec_price = base_price + price_adjustment
        else:
            base_price = market_data.bid
            if microstructure.order_flow_imbalance < -0.5:  # Heavy selling
                price_adjustment = microstructure.bid_ask_spread * 0.2
            else:
                price_adjustment = 0
            exec_price = base_price - price_adjustment

        # Create fill (simplified)
        fill = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=exec_price,
            timestamp=datetime.now(),
            strategy_id="implementation_shortfall",
            commission=0.5,
            slippage=0.0
        )

        fills.append(fill)
        total_cost = quantity * exec_price

        return fills, total_cost

    def _calculate_is_components(self, order: ExecutionOrder, fills: List[Trade],
                               market_data: MarketData,
                               microstructure: MarketMicrostructure) -> Dict[str, float]:
        """Calculate implementation shortfall components"""

        if not fills:
            return {'total': 0, 'market_impact': 0, 'timing_cost': 0}

        # Decision price (benchmark)
        decision_price = (market_data.bid + market_data.ask) / 2

        # Calculate average execution price
        total_quantity = sum(fill.quantity for fill in fills)
        avg_exec_price = sum(fill.quantity * fill.price for fill in fills) / total_quantity

        # Market impact cost
        if order.side == OrderSide.BID:
            market_impact = (avg_exec_price - decision_price) / decision_price
        else:
            market_impact = (decision_price - avg_exec_price) / decision_price

        # Timing cost (simplified - would need price evolution)
        timing_cost = 0.0  # Would calculate based on price movement during execution

        total_shortfall = market_impact + timing_cost

        return {
            'total': total_shortfall,
            'market_impact': market_impact,
            'timing_cost': timing_cost
        }


class SmartOrderRouter:
    """Smart Order Routing (SOR) system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'venue_selection_method': 'best_price',  # 'best_price', 'pro_rata', 'random'
            'dark_pool_preference': 0.3,             # 30% preference for dark pools
            'latency_penalty_ms': 10,                # Latency penalty per ms
            'fill_probability_weight': 0.4,          # Weight for fill probability
            'price_improvement_weight': 0.6,         # Weight for price improvement
            'venue_diversification': True,           # Diversify across venues
            'max_venues_per_order': 3               # Maximum venues per order
        }

        self.venue_performance: Dict[ExecutionVenue, Dict] = defaultdict(dict)
        self.order_history: deque = deque(maxlen=10000)

    def route_order(self, order: ExecutionOrder, venue_data: List[VenueData]) -> ExecutionResult:
        """Route order optimally across venues"""

        start_time = time.time()

        # Analyze venue landscape
        venue_analysis = self._analyze_venues(order, venue_data)

        # Generate routing strategy
        routing_strategy = self._generate_routing_strategy(order, venue_analysis)

        # Execute across venues
        fills = []
        total_cost = 0.0
        venues_used = []

        for venue_allocation in routing_strategy:
            venue = venue_allocation['venue']
            quantity = venue_allocation['quantity']
            price = venue_allocation['price']

            if quantity > 0:
                # Execute on venue
                venue_fills, venue_cost = self._execute_on_venue(
                    order, venue, quantity, price, venue_data
                )

                fills.extend(venue_fills)
                total_cost += venue_cost

                if venue not in venues_used:
                    venues_used.append(venue)

        # Calculate performance metrics
        total_executed = sum(fill.quantity for fill in fills)
        avg_price = total_cost / total_executed if total_executed > 0 else 0

        # Update venue performance tracking
        self._update_venue_performance(order, fills, venue_data)

        execution_time = time.time() - start_time

        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            total_quantity=order.quantity,
            executed_quantity=total_executed,
            remaining_quantity=order.quantity - total_executed,
            average_price=avg_price,
            total_cost=total_cost,
            implementation_shortfall=0.0,  # Would calculate
            volume_weighted_price=avg_price,
            time_weighted_price=avg_price,
            market_impact=0.0,  # Would calculate
            timing_cost=0.0,
            commission_cost=len(fills) * 0.5,
            fills=fills,
            venues_used=venues_used,
            algorithm_used=ExecutionAlgorithm.SMART_ORDER_ROUTING,
            execution_time_seconds=execution_time,
            fill_rate=total_executed / order.quantity,
            price_improvement=self._calculate_price_improvement(order, fills, venue_data),
            adverse_selection=0.0
        )

    def _analyze_venues(self, order: ExecutionOrder, venue_data: List[VenueData]) -> List[Dict]:
        """Analyze available venues for order routing"""

        venue_analysis = []

        for venue in venue_data:
            if order.side == OrderSide.BID:
                available_price = venue.ask
                available_size = venue.ask_size
            else:
                available_price = venue.bid
                available_size = venue.bid_size

            # Calculate venue score
            score_components = self._calculate_venue_score(order, venue)

            venue_analysis.append({
                'venue': venue.venue,
                'price': available_price,
                'size': available_size,
                'latency': venue.latency_ms,
                'fill_probability': venue.fill_probability,
                'price_improvement': venue.price_improvement_bps,
                'score': score_components['total_score'],
                'score_components': score_components
            })

        # Sort by score
        venue_analysis.sort(key=lambda x: x['score'], reverse=True)

        return venue_analysis

    def _calculate_venue_score(self, order: ExecutionOrder, venue: VenueData) -> Dict:
        """Calculate venue attractiveness score"""

        # Price competitiveness
        if order.side == OrderSide.BID:
            price_score = 1.0 / (venue.ask + 0.01)  # Lower ask is better
        else:
            price_score = venue.bid  # Higher bid is better

        # Normalize price score
        price_score = min(price_score, 1.0)

        # Size availability
        size_score = min(1.0, venue.ask_size / max(order.quantity, 1))

        # Latency score
        latency_score = max(0, 1.0 - venue.latency_ms / 100.0)

        # Fill probability
        fill_prob_score = venue.fill_probability

        # Price improvement
        price_imp_score = venue.price_improvement_bps / 10.0  # Normalize by 10bps

        # Venue preference based on type
        venue_pref_score = 1.0
        if venue.venue in [ExecutionVenue.DARK_POOL_1, ExecutionVenue.DARK_POOL_2]:
            venue_pref_score = 1.0 + self.config['dark_pool_preference']

        # Historical performance
        historical_score = self._get_historical_venue_performance(venue.venue)

        # Combine scores
        total_score = (
            price_score * self.config['price_improvement_weight'] +
            fill_prob_score * self.config['fill_probability_weight'] +
            size_score * 0.2 +
            latency_score * 0.1 +
            price_imp_score * 0.1 +
            venue_pref_score * 0.1 +
            historical_score * 0.1
        )

        return {
            'total_score': total_score,
            'price_score': price_score,
            'size_score': size_score,
            'latency_score': latency_score,
            'fill_prob_score': fill_prob_score,
            'price_imp_score': price_imp_score,
            'venue_pref_score': venue_pref_score,
            'historical_score': historical_score
        }

    def _generate_routing_strategy(self, order: ExecutionOrder, venue_analysis: List[Dict]) -> List[Dict]:
        """Generate optimal routing strategy"""

        routing_strategy = []
        remaining_quantity = order.quantity

        # Select top venues
        selected_venues = venue_analysis[:self.config['max_venues_per_order']]

        if self.config['venue_selection_method'] == 'best_price':
            # Route to best venue first
            best_venue = selected_venues[0]
            allocated_quantity = min(remaining_quantity, best_venue['size'])

            routing_strategy.append({
                'venue': best_venue['venue'],
                'quantity': allocated_quantity,
                'price': best_venue['price']
            })

        elif self.config['venue_selection_method'] == 'pro_rata':
            # Allocate proportionally based on scores
            total_score = sum(venue['score'] for venue in selected_venues)

            for venue in selected_venues:
                if remaining_quantity <= 0:
                    break

                allocation_ratio = venue['score'] / total_score
                allocated_quantity = min(
                    int(order.quantity * allocation_ratio),
                    remaining_quantity,
                    venue['size']
                )

                if allocated_quantity > 0:
                    routing_strategy.append({
                        'venue': venue['venue'],
                        'quantity': allocated_quantity,
                        'price': venue['price']
                    })
                    remaining_quantity -= allocated_quantity

        elif self.config['venue_selection_method'] == 'random':
            # Random allocation with bias towards better venues
            weights = [venue['score'] for venue in selected_venues]

            while remaining_quantity > 0 and selected_venues:
                # Weighted random selection
                chosen_venue = np.random.choice(selected_venues, p=np.array(weights)/sum(weights))
                allocated_quantity = min(remaining_quantity, chosen_venue['size'])

                routing_strategy.append({
                    'venue': chosen_venue['venue'],
                    'quantity': allocated_quantity,
                    'price': chosen_venue['price']
                })

                remaining_quantity -= allocated_quantity
                break  # Simplified - would continue allocation

        return routing_strategy

    def _execute_on_venue(self, order: ExecutionOrder, venue: ExecutionVenue,
                         quantity: int, price: float,
                         venue_data: List[VenueData]) -> Tuple[List[Trade], float]:
        """Execute order on specific venue"""

        fills = []
        total_cost = 0.0

        # Simulate venue-specific execution
        # In practice, would send orders to actual venues

        # Apply venue-specific characteristics
        venue_info = next((v for v in venue_data if v.venue == venue), None)
        if venue_info:
            # Simulate fill based on venue characteristics
            fill_probability = venue_info.fill_probability

            if np.random.random() < fill_probability:
                # Successful fill
                actual_price = price + np.random.normal(0, 0.01)  # Add noise

                fill = Trade(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=quantity,
                    price=actual_price,
                    timestamp=datetime.now(),
                    strategy_id="smart_order_routing",
                    commission=0.5,
                    slippage=abs(actual_price - price)
                )

                setattr(fill, 'venue', venue)
                fills.append(fill)
                total_cost = quantity * actual_price

        return fills, total_cost

    def _update_venue_performance(self, order: ExecutionOrder, fills: List[Trade],
                                 venue_data: List[VenueData]):
        """Update venue performance tracking"""

        for fill in fills:
            venue = getattr(fill, 'venue', ExecutionVenue.PRIMARY_EXCHANGE)

            if venue not in self.venue_performance:
                self.venue_performance[venue] = {
                    'total_fills': 0,
                    'total_quantity': 0,
                    'avg_fill_rate': 0.0,
                    'avg_price_improvement': 0.0,
                    'avg_latency': 0.0
                }

            perf = self.venue_performance[venue]
            perf['total_fills'] += 1
            perf['total_quantity'] += fill.quantity

            # Update running averages (simplified)
            perf['avg_fill_rate'] = 0.9 * perf['avg_fill_rate'] + 0.1 * 1.0  # Successful fill
            perf['avg_price_improvement'] = 0.9 * perf['avg_price_improvement'] + 0.1 * 0.0  # Would calculate

        # Store order in history
        self.order_history.append({
            'order': order,
            'fills': fills,
            'timestamp': datetime.now()
        })

    def _get_historical_venue_performance(self, venue: ExecutionVenue) -> float:
        """Get historical performance score for venue"""

        if venue in self.venue_performance:
            perf = self.venue_performance[venue]
            # Combine metrics into single score
            return (perf['avg_fill_rate'] * 0.5 +
                   min(perf['avg_price_improvement'] / 5.0, 1.0) * 0.3 +
                   max(0, 1.0 - perf['avg_latency'] / 100.0) * 0.2)

        return 0.5  # Default neutral score

    def _calculate_price_improvement(self, order: ExecutionOrder, fills: List[Trade],
                                   venue_data: List[VenueData]) -> float:
        """Calculate price improvement vs benchmark"""

        if not fills:
            return 0.0

        # Use NBBO as benchmark
        best_bid = max(v.bid for v in venue_data if v.bid > 0)
        best_ask = min(v.ask for v in venue_data if v.ask > 0)

        total_improvement = 0.0
        total_quantity = 0

        for fill in fills:
            if order.side == OrderSide.BID:
                benchmark = best_ask
                improvement = (benchmark - fill.price) / benchmark
            else:
                benchmark = best_bid
                improvement = (fill.price - benchmark) / benchmark

            total_improvement += improvement * fill.quantity
            total_quantity += fill.quantity

        return total_improvement / total_quantity if total_quantity > 0 else 0.0


class HiddenLiquidityDetector:
    """Detects and accesses hidden liquidity"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'iceberg_detection_threshold': 0.1,    # 10% size refresh threshold
            'hidden_size_multiplier': 2.0,         # Estimate hidden size
            'probe_order_size': 10,                # Size for probing orders
            'dark_pool_participation': 0.3,        # 30% to dark pools
            'patience_factor': 1.5                 # Patience multiplier
        }

        self.iceberg_detection: Dict[str, Dict] = defaultdict(dict)
        self.hidden_liquidity_estimates: Dict[str, float] = defaultdict(float)

    def detect_hidden_liquidity(self, symbol: str, market_data: MarketData,
                              venue_data: List[VenueData]) -> Dict[str, float]:
        """Detect hidden liquidity across venues"""

        hidden_liquidity = {}

        for venue in venue_data:
            # Iceberg detection
            iceberg_estimate = self._detect_iceberg_orders(symbol, venue)

            # Dark pool liquidity estimation
            dark_pool_estimate = self._estimate_dark_pool_liquidity(symbol, venue)

            # Reserve order detection
            reserve_estimate = self._detect_reserve_orders(symbol, venue)

            total_hidden = iceberg_estimate + dark_pool_estimate + reserve_estimate
            hidden_liquidity[venue.venue] = total_hidden

        return hidden_liquidity

    def access_hidden_liquidity(self, order: ExecutionOrder, hidden_liquidity: Dict[str, float],
                              venue_data: List[VenueData]) -> ExecutionResult:
        """Access hidden liquidity using appropriate strategies"""

        start_time = time.time()
        fills = []
        total_cost = 0.0
        venues_used = []

        # Strategy 1: Use iceberg orders
        iceberg_fills, iceberg_cost = self._execute_iceberg_strategy(order, venue_data)
        fills.extend(iceberg_fills)
        total_cost += iceberg_cost

        # Strategy 2: Probe for hidden liquidity
        probe_fills, probe_cost = self._execute_probing_strategy(order, venue_data, hidden_liquidity)
        fills.extend(probe_fills)
        total_cost += probe_cost

        # Strategy 3: Dark pool routing
        dark_fills, dark_cost = self._execute_dark_pool_strategy(order, venue_data)
        fills.extend(dark_fills)
        total_cost += dark_cost

        # Calculate metrics
        total_executed = sum(fill.quantity for fill in fills)
        avg_price = total_cost / total_executed if total_executed > 0 else 0

        execution_time = time.time() - start_time

        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            total_quantity=order.quantity,
            executed_quantity=total_executed,
            remaining_quantity=order.quantity - total_executed,
            average_price=avg_price,
            total_cost=total_cost,
            implementation_shortfall=0.0,
            volume_weighted_price=avg_price,
            time_weighted_price=avg_price,
            market_impact=0.0,
            timing_cost=0.0,
            commission_cost=len(fills) * 0.5,
            fills=fills,
            venues_used=venues_used,
            algorithm_used=ExecutionAlgorithm.HIDDEN_LIQUIDITY,
            execution_time_seconds=execution_time,
            fill_rate=total_executed / order.quantity,
            price_improvement=0.0,
            adverse_selection=0.0
        )

    def _detect_iceberg_orders(self, symbol: str, venue: VenueData) -> float:
        """Detect iceberg orders by monitoring size refreshes"""

        venue_key = f"{symbol}_{venue.venue}"

        if venue_key not in self.iceberg_detection:
            self.iceberg_detection[venue_key] = {
                'size_history': deque(maxlen=20),
                'refresh_count': 0,
                'estimated_size': 0
            }

        detection_data = self.iceberg_detection[venue_key]

        # Monitor bid/ask size changes
        current_size = venue.bid_size + venue.ask_size
        detection_data['size_history'].append(current_size)

        if len(detection_data['size_history']) >= 2:
            # Check for size refreshes
            prev_size = detection_data['size_history'][-2]
            size_change_ratio = abs(current_size - prev_size) / max(prev_size, 1)

            if size_change_ratio > self.config['iceberg_detection_threshold']:
                detection_data['refresh_count'] += 1

                # Estimate hidden size
                estimated_hidden = current_size * self.config['hidden_size_multiplier']
                detection_data['estimated_size'] = estimated_hidden

                return estimated_hidden

        return detection_data.get('estimated_size', 0)

    def _estimate_dark_pool_liquidity(self, symbol: str, venue: VenueData) -> float:
        """Estimate dark pool liquidity"""

        if venue.venue in [ExecutionVenue.DARK_POOL_1, ExecutionVenue.DARK_POOL_2]:
            # Estimate based on overall market volume
            # Dark pools typically have 10-15% of total volume
            estimated_dark_volume = venue.volume * 0.12

            # Convert volume to available liquidity (simplified)
            return estimated_dark_volume * 0.1  # 10% of volume available

        return 0

    def _detect_reserve_orders(self, symbol: str, venue: VenueData) -> float:
        """Detect reserve (hidden) orders"""

        # Look for patterns indicating reserve orders
        # Simplified implementation

        if venue.bid_size < 50 and venue.volume > 1000:
            # Small displayed size but high volume suggests reserves
            estimated_reserve = venue.volume * 0.05  # 5% of volume
            return estimated_reserve

        return 0

    def _execute_iceberg_strategy(self, order: ExecutionOrder,
                                venue_data: List[VenueData]) -> Tuple[List[Trade], float]:
        """Execute using iceberg order strategy"""

        fills = []
        total_cost = 0.0

        # Use small displayed size to avoid detection
        iceberg_size = min(order.quantity // 10, 100)  # 10% or 100 max

        if iceberg_size > 0:
            # Execute iceberg slices
            remaining = order.quantity

            while remaining > 0 and len(fills) < 5:  # Limit iterations
                slice_size = min(iceberg_size, remaining)

                # Execute slice
                if order.side == OrderSide.BID:
                    price = venue_data[0].ask  # Use primary venue
                else:
                    price = venue_data[0].bid

                fill = Trade(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_size,
                    price=price,
                    timestamp=datetime.now(),
                    strategy_id="iceberg",
                    commission=0.5
                )

                fills.append(fill)
                total_cost += slice_size * price
                remaining -= slice_size

                # Wait between slices
                time.sleep(0.1)

        return fills, total_cost

    def _execute_probing_strategy(self, order: ExecutionOrder, venue_data: List[VenueData],
                                hidden_liquidity: Dict[str, float]) -> Tuple[List[Trade], float]:
        """Execute probing strategy to find hidden liquidity"""

        fills = []
        total_cost = 0.0

        # Send small probe orders to detect hidden liquidity
        probe_size = self.config['probe_order_size']

        for venue_data_item in venue_data[:2]:  # Probe top 2 venues
            if hidden_liquidity.get(venue_data_item.venue, 0) > probe_size:
                # Send probe order
                if order.side == OrderSide.BID:
                    probe_price = venue_data_item.ask
                else:
                    probe_price = venue_data_item.bid

                # Simulate probe execution
                if np.random.random() < 0.7:  # 70% probe success rate
                    fill = Trade(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=probe_size,
                        price=probe_price,
                        timestamp=datetime.now(),
                        strategy_id="probe",
                        commission=0.5
                    )

                    fills.append(fill)
                    total_cost += probe_size * probe_price

        return fills, total_cost

    def _execute_dark_pool_strategy(self, order: ExecutionOrder,
                                  venue_data: List[VenueData]) -> Tuple[List[Trade], float]:
        """Execute in dark pools"""

        fills = []
        total_cost = 0.0

        # Route portion to dark pools
        dark_allocation = int(order.quantity * self.config['dark_pool_participation'])

        if dark_allocation > 0:
            dark_venues = [v for v in venue_data
                          if v.venue in [ExecutionVenue.DARK_POOL_1, ExecutionVenue.DARK_POOL_2]]

            if dark_venues:
                venue = dark_venues[0]

                # Midpoint pricing in dark pools
                midpoint_price = (venue.bid + venue.ask) / 2

                # Simulate dark pool execution
                if np.random.random() < 0.5:  # 50% fill rate in dark pools
                    fill = Trade(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=dark_allocation,
                        price=midpoint_price,
                        timestamp=datetime.now(),
                        strategy_id="dark_pool",
                        commission=0.5
                    )

                    fills.append(fill)
                    total_cost += dark_allocation * midpoint_price

        return fills, total_cost


# Factory functions and main execution engine
class ExecutionEngine:
    """Main execution engine coordinating all algorithms"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize algorithms
        self.vwap_algo = VWAPAlgorithm()
        self.is_algo = ImplementationShortfallAlgorithm()
        self.smart_router = SmartOrderRouter()
        self.hidden_liquidity = HiddenLiquidityDetector()

        # Execution history
        self.execution_history: deque = deque(maxlen=10000)

    def execute_order(self, order: ExecutionOrder, market_data: MarketData,
                     venue_data: List[VenueData],
                     microstructure: Optional[MarketMicrostructure] = None) -> ExecutionResult:
        """Execute order using appropriate algorithm"""

        # Select algorithm based on order parameters
        if order.algorithm == ExecutionAlgorithm.VWAP:
            result = self.vwap_algo.execute_order(order, market_data, venue_data)
        elif order.algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
            if microstructure is None:
                microstructure = self._create_default_microstructure(market_data)
            result = self.is_algo.execute_order(order, market_data, microstructure)
        elif order.algorithm == ExecutionAlgorithm.SMART_ORDER_ROUTING:
            result = self.smart_router.route_order(order, venue_data)
        elif order.algorithm == ExecutionAlgorithm.HIDDEN_LIQUIDITY:
            hidden_liq = self.hidden_liquidity.detect_hidden_liquidity(order.symbol, market_data, venue_data)
            result = self.hidden_liquidity.access_hidden_liquidity(order, hidden_liq, venue_data)
        else:
            # Default to VWAP
            result = self.vwap_algo.execute_order(order, market_data, venue_data)

        # Store execution history
        self.execution_history.append(result)

        return result

    def _create_default_microstructure(self, market_data: MarketData) -> MarketMicrostructure:
        """Create default market microstructure data"""

        return MarketMicrostructure(
            symbol=market_data.symbol,
            bid_ask_spread=market_data.ask - market_data.bid,
            effective_spread=(market_data.ask - market_data.bid) * 0.8,
            quoted_spread=market_data.ask - market_data.bid,
            depth_at_touch=market_data.bid_size + market_data.ask_size,
            depth_beyond_touch=0,
            order_flow_imbalance=0.0,
            volatility_estimate=0.02,  # 2% daily volatility
            recent_trade_direction=0,
            hidden_liquidity_estimate=market_data.volume * 0.1,
            timestamp=datetime.now()
        )

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics"""

        if not self.execution_history:
            return {}

        recent_executions = list(self.execution_history)[-100:]  # Last 100 executions

        avg_fill_rate = np.mean([ex.fill_rate for ex in recent_executions])
        avg_implementation_shortfall = np.mean([ex.implementation_shortfall for ex in recent_executions])
        avg_execution_time = np.mean([ex.execution_time_seconds for ex in recent_executions])

        algorithm_performance = defaultdict(list)
        for ex in recent_executions:
            algorithm_performance[ex.algorithm_used].append(ex.implementation_shortfall)

        algo_stats = {algo.value: {
            'count': len(shortfalls),
            'avg_shortfall': np.mean(shortfalls),
            'std_shortfall': np.std(shortfalls)
        } for algo, shortfalls in algorithm_performance.items()}

        return {
            'total_executions': len(recent_executions),
            'avg_fill_rate': avg_fill_rate,
            'avg_implementation_shortfall': avg_implementation_shortfall,
            'avg_execution_time': avg_execution_time,
            'algorithm_performance': algo_stats
        }


# Example usage
if __name__ == "__main__":
    # Create execution engine
    engine = ExecutionEngine()

    # Example order
    order = ExecutionOrder(
        symbol="AAPL_231215C150",
        side=OrderSide.BID,
        quantity=1000,
        order_type=OrderType.LIMIT,
        limit_price=5.00,
        algorithm=ExecutionAlgorithm.VWAP,
        participation_rate=0.15,
        urgency=0.7
    )

    # Example market data
    market_data = MarketData(
        symbol="AAPL_231215C150",
        timestamp=datetime.now(),
        bid=4.90,
        ask=5.10,
        last=5.00,
        bid_size=50,
        ask_size=60,
        volume=5000,
        open_interest=10000
    )

    # Example venue data
    venue_data = [
        VenueData(ExecutionVenue.PRIMARY_EXCHANGE, "AAPL_231215C150", 4.90, 5.10, 50, 60, 5.00, 5000, 5.0, 0.85, 2.0, datetime.now()),
        VenueData(ExecutionVenue.DARK_POOL_1, "AAPL_231215C150", 4.95, 5.05, 30, 40, 5.00, 2000, 8.0, 0.60, 5.0, datetime.now()),
        VenueData(ExecutionVenue.ECN_1, "AAPL_231215C150", 4.92, 5.08, 25, 35, 5.00, 1500, 3.0, 0.90, 1.5, datetime.now())
    ]

    # Execute order
    result = engine.execute_order(order, market_data, venue_data)

    print(f"Execution Results:")
    print(f"Executed: {result.executed_quantity}/{result.total_quantity}")
    print(f"Average Price: ${result.average_price:.2f}")
    print(f"Implementation Shortfall: {result.implementation_shortfall:.4f}")
    print(f"Fill Rate: {result.fill_rate:.2%}")
    print(f"Execution Time: {result.execution_time_seconds:.2f}s")
    print(f"Venues Used: {[v.value for v in result.venues_used]}")

    # Get performance statistics
    stats = engine.get_execution_statistics()
    print(f"\nExecution Statistics: {stats}")