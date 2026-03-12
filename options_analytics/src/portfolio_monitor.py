"""
Real-time Portfolio Monitoring and Visualization
Advanced portfolio analytics with live Greeks tracking and P&L attribution
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import asyncio
import json
import logging
from abc import ABC, abstractmethod

from analytics_framework import (
    MarketData, Position, PortfolioGreeks, OptionsAnalyticsFramework,
    AnalyticsType
)

@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    total_value: float
    total_pnl: float
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    var_95: float
    max_drawdown: float
    sharpe_ratio: float
    positions_count: int

@dataclass
class GreeksHeatMap:
    symbols: List[str]
    strikes: List[float]
    expiries: List[datetime]
    delta_matrix: np.ndarray
    gamma_matrix: np.ndarray
    theta_matrix: np.ndarray
    vega_matrix: np.ndarray
    risk_levels: np.ndarray

@dataclass
class PnLAttribution:
    timestamp: datetime
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    trading_pnl: float
    total_pnl: float
    strategy_attribution: Dict[str, float]
    symbol_attribution: Dict[str, float]

class PortfolioMonitor:
    """Real-time portfolio monitoring and analytics"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analytics_framework = OptionsAnalyticsFramework(config)

        # Portfolio state
        self.current_positions: Dict[str, Position] = {}
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.pnl_history: List[PnLAttribution] = []
        self.greeks_history: List[PortfolioGreeks] = []

        # Monitoring settings
        self.update_frequency = config.get('update_frequency_ms', 1000)
        self.history_length = config.get('history_length', 10000)

        # Risk limits
        self.risk_limits = config.get('risk_limits', {
            'max_delta': 1000,
            'max_gamma': 500,
            'max_theta': -200,
            'max_vega': 2000,
            'max_var': 100000
        })

        # Event handlers
        self.event_handlers: Dict[str, List[callable]] = {}

        # Performance trackers
        self.performance_calculator = PerformanceCalculator(config)
        self.greeks_visualizer = GreeksVisualizer(config)
        self.risk_monitor = RiskMonitor(config)

    async def initialize(self):
        """Initialize portfolio monitor"""
        self.logger.info("Initializing Portfolio Monitor")
        await self.analytics_framework.initialize()
        await self.performance_calculator.initialize()

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    async def update_positions(self, positions: Dict[str, Position]):
        """Update portfolio positions"""
        self.current_positions = positions
        await self.analytics_framework.update_positions(positions)

        # Calculate current portfolio metrics
        await self._calculate_portfolio_snapshot()

        # Check risk limits
        await self._check_risk_limits()

        # Trigger position update events
        await self._trigger_event('position_update', positions)

    async def update_market_data(self, market_data: Dict[str, MarketData]):
        """Update market data"""
        await self.analytics_framework.update_market_data(market_data)

        # Recalculate P&L attribution
        await self._calculate_pnl_attribution(market_data)

        # Update Greeks
        await self._update_portfolio_greeks()

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        if not self.current_positions:
            return {}

        portfolio_greeks = self.analytics_framework.analytics_engine.portfolio_analyzer.calculate_portfolio_greeks(
            self.current_positions
        )

        total_value = sum(pos.market_value for pos in self.current_positions.values())
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.current_positions.values())

        performance_metrics = await self.performance_calculator.calculate_performance_metrics()
        risk_metrics = await self.risk_monitor.calculate_risk_metrics(self.current_positions)

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': total_value,
            'total_pnl': total_pnl,
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.current_positions.values()),
            'realized_pnl': sum(pos.realized_pnl for pos in self.current_positions.values()),
            'greeks': {
                'delta': portfolio_greeks.total_delta,
                'gamma': portfolio_greeks.total_gamma,
                'theta': portfolio_greeks.total_theta,
                'vega': portfolio_greeks.total_vega,
                'rho': portfolio_greeks.total_rho
            },
            'positions_count': len(self.current_positions),
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'risk_status': await self._get_risk_status()
        }

    async def get_greeks_heatmap(self) -> GreeksHeatMap:
        """Generate Greeks heatmap for visualization"""
        return await self.greeks_visualizer.generate_heatmap(self.current_positions)

    async def get_pnl_attribution(self, time_period: str = '1d') -> List[PnLAttribution]:
        """Get P&L attribution for specified time period"""
        cutoff_time = datetime.now()

        if time_period == '1h':
            cutoff_time -= timedelta(hours=1)
        elif time_period == '1d':
            cutoff_time -= timedelta(days=1)
        elif time_period == '1w':
            cutoff_time -= timedelta(weeks=1)
        elif time_period == '1m':
            cutoff_time -= timedelta(days=30)

        return [attr for attr in self.pnl_history if attr.timestamp >= cutoff_time]

    async def get_position_details(self, symbol: str = None) -> Dict[str, Any]:
        """Get detailed position information"""
        if symbol and symbol in self.current_positions:
            positions = {symbol: self.current_positions[symbol]}
        else:
            positions = self.current_positions

        position_details = {}

        for sym, pos in positions.items():
            position_details[sym] = {
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'average_price': pos.average_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'total_pnl': pos.unrealized_pnl + pos.realized_pnl,
                'greeks': {
                    'delta': pos.delta,
                    'gamma': pos.gamma,
                    'theta': pos.theta,
                    'vega': pos.vega,
                    'rho': pos.rho
                },
                'risk_contribution': await self._calculate_position_risk_contribution(pos),
                'pnl_contribution': await self._calculate_position_pnl_contribution(pos)
            }

        return position_details

    async def get_scenario_analysis(self, scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform scenario analysis on current portfolio"""
        scenario_results = self.analytics_framework.analytics_engine.portfolio_analyzer.calculate_scenario_analysis(
            self.current_positions, scenarios
        )

        return {
            'scenarios': scenario_results,
            'worst_case': min(scenario_results.values(), key=lambda x: x['total_pnl']),
            'best_case': max(scenario_results.values(), key=lambda x: x['total_pnl']),
            'expected_case': {
                'total_pnl': np.mean([s['total_pnl'] for s in scenario_results.values()])
            }
        }

    async def add_event_handler(self, event_type: str, handler: callable):
        """Add event handler for portfolio events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                if self.current_positions:
                    await self._calculate_portfolio_snapshot()
                    await self._check_risk_limits()
                    await self._cleanup_history()

                await asyncio.sleep(self.update_frequency / 1000.0)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)

    async def _calculate_portfolio_snapshot(self):
        """Calculate current portfolio snapshot"""
        if not self.current_positions:
            return

        portfolio_greeks = self.analytics_framework.analytics_engine.portfolio_analyzer.calculate_portfolio_greeks(
            self.current_positions
        )

        total_value = sum(pos.market_value for pos in self.current_positions.values())
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.current_positions.values())

        var_95 = await self.risk_monitor.calculate_var(self.current_positions, 0.95)
        max_drawdown = await self.performance_calculator.calculate_max_drawdown()
        sharpe_ratio = await self.performance_calculator.calculate_sharpe_ratio()

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=total_value,
            total_pnl=total_pnl,
            total_delta=portfolio_greeks.total_delta,
            total_gamma=portfolio_greeks.total_gamma,
            total_theta=portfolio_greeks.total_theta,
            total_vega=portfolio_greeks.total_vega,
            total_rho=portfolio_greeks.total_rho,
            var_95=var_95.get('var', 0.0),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            positions_count=len(self.current_positions)
        )

        self.portfolio_history.append(snapshot)
        self.greeks_history.append(portfolio_greeks)

    async def _calculate_pnl_attribution(self, market_data: Dict[str, MarketData]):
        """Calculate detailed P&L attribution"""
        if not self.pnl_history:
            # First calculation, initialize with zeros
            attribution = PnLAttribution(
                timestamp=datetime.now(),
                delta_pnl=0.0,
                gamma_pnl=0.0,
                theta_pnl=0.0,
                vega_pnl=0.0,
                rho_pnl=0.0,
                trading_pnl=0.0,
                total_pnl=0.0,
                strategy_attribution={},
                symbol_attribution={}
            )
        else:
            # Calculate attribution since last update
            prev_positions = {pos.symbol: pos for pos in self.current_positions.values()}
            time_elapsed = 1.0 / 24.0  # Assume 1 hour elapsed

            attribution_data = self.analytics_framework.analytics_engine.portfolio_analyzer.calculate_pnl_attribution(
                self.current_positions, prev_positions, market_data, time_elapsed
            )

            # Calculate strategy and symbol attribution
            strategy_attribution = await self._calculate_strategy_attribution()
            symbol_attribution = await self._calculate_symbol_attribution()

            attribution = PnLAttribution(
                timestamp=datetime.now(),
                delta_pnl=attribution_data['delta_pnl'],
                gamma_pnl=attribution_data['gamma_pnl'],
                theta_pnl=attribution_data['theta_pnl'],
                vega_pnl=attribution_data['vega_pnl'],
                rho_pnl=attribution_data['rho_pnl'],
                trading_pnl=attribution_data['trading_pnl'],
                total_pnl=attribution_data['total_pnl'],
                strategy_attribution=strategy_attribution,
                symbol_attribution=symbol_attribution
            )

        self.pnl_history.append(attribution)

    async def _update_portfolio_greeks(self):
        """Update portfolio Greeks"""
        portfolio_greeks = self.analytics_framework.analytics_engine.portfolio_analyzer.calculate_portfolio_greeks(
            self.current_positions
        )
        self.greeks_history.append(portfolio_greeks)

    async def _check_risk_limits(self):
        """Check portfolio against risk limits"""
        if not self.current_positions:
            return

        portfolio_greeks = self.analytics_framework.analytics_engine.portfolio_analyzer.calculate_portfolio_greeks(
            self.current_positions
        )

        violations = []

        if abs(portfolio_greeks.total_delta) > self.risk_limits['max_delta']:
            violations.append({
                'type': 'delta_limit',
                'current': portfolio_greeks.total_delta,
                'limit': self.risk_limits['max_delta']
            })

        if abs(portfolio_greeks.total_gamma) > self.risk_limits['max_gamma']:
            violations.append({
                'type': 'gamma_limit',
                'current': portfolio_greeks.total_gamma,
                'limit': self.risk_limits['max_gamma']
            })

        if portfolio_greeks.total_theta < self.risk_limits['max_theta']:
            violations.append({
                'type': 'theta_limit',
                'current': portfolio_greeks.total_theta,
                'limit': self.risk_limits['max_theta']
            })

        if abs(portfolio_greeks.total_vega) > self.risk_limits['max_vega']:
            violations.append({
                'type': 'vega_limit',
                'current': portfolio_greeks.total_vega,
                'limit': self.risk_limits['max_vega']
            })

        if violations:
            await self._trigger_event('risk_limit_violation', violations)

    async def _get_risk_status(self) -> str:
        """Get current risk status"""
        if not self.current_positions:
            return 'no_positions'

        portfolio_greeks = self.analytics_framework.analytics_engine.portfolio_analyzer.calculate_portfolio_greeks(
            self.current_positions
        )

        delta_util = abs(portfolio_greeks.total_delta) / self.risk_limits['max_delta']
        gamma_util = abs(portfolio_greeks.total_gamma) / self.risk_limits['max_gamma']
        vega_util = abs(portfolio_greeks.total_vega) / self.risk_limits['max_vega']

        max_util = max(delta_util, gamma_util, vega_util)

        if max_util > 1.0:
            return 'high_risk'
        elif max_util > 0.8:
            return 'medium_risk'
        else:
            return 'low_risk'

    async def _calculate_strategy_attribution(self) -> Dict[str, float]:
        """Calculate P&L attribution by strategy"""
        # Placeholder implementation
        return {
            'delta_neutral': 1000.0,
            'volatility_arbitrage': 500.0,
            'calendar_spreads': -200.0
        }

    async def _calculate_symbol_attribution(self) -> Dict[str, float]:
        """Calculate P&L attribution by symbol"""
        symbol_attribution = {}

        for symbol, pos in self.current_positions.items():
            symbol_attribution[symbol] = pos.unrealized_pnl + pos.realized_pnl

        return symbol_attribution

    async def _calculate_position_risk_contribution(self, position: Position) -> float:
        """Calculate position's contribution to portfolio risk"""
        position_var = abs(position.delta * position.quantity * 0.02 * position.current_price)
        return position_var

    async def _calculate_position_pnl_contribution(self, position: Position) -> float:
        """Calculate position's contribution to total P&L"""
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.current_positions.values())

        if total_pnl == 0:
            return 0.0

        position_pnl = position.unrealized_pnl + position.realized_pnl
        return position_pnl / total_pnl

    async def _trigger_event(self, event_type: str, data: Any):
        """Trigger event handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler {event_type}: {e}")

    async def _cleanup_history(self):
        """Cleanup old history data"""
        if len(self.portfolio_history) > self.history_length:
            self.portfolio_history = self.portfolio_history[-self.history_length:]

        if len(self.pnl_history) > self.history_length:
            self.pnl_history = self.pnl_history[-self.history_length:]

        if len(self.greeks_history) > self.history_length:
            self.greeks_history = self.greeks_history[-self.history_length:]

class PerformanceCalculator:
    """Portfolio performance calculation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.returns_history: List[float] = []

    async def initialize(self):
        """Initialize performance calculator"""
        self.logger.info("Initializing Performance Calculator")

    async def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(self.returns_history) < 2:
            return {}

        returns = np.array(self.returns_history)

        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / max(volatility, 0.01)

        max_drawdown = await self.calculate_max_drawdown()
        calmar_ratio = annualized_return / max(abs(max_drawdown), 0.01)

        win_rate = len([r for r in returns if r > 0]) / len(returns)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'periods': len(returns)
        }

    async def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.returns_history) < 2:
            return 0.0

        cumulative = np.cumprod(1 + np.array(self.returns_history))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return np.min(drawdown)

    async def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(self.returns_history)
        mean_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)

        return mean_return / max(volatility, 0.01)

    def add_return(self, return_value: float):
        """Add return to history"""
        self.returns_history.append(return_value)

        if len(self.returns_history) > 10000:
            self.returns_history = self.returns_history[-10000:]

class GreeksVisualizer:
    """Greeks visualization and heatmap generation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def generate_heatmap(self, positions: Dict[str, Position]) -> GreeksHeatMap:
        """Generate Greeks heatmap"""
        if not positions:
            return GreeksHeatMap([], [], [], np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

        symbols = list(positions.keys())
        strikes = [100, 105, 110, 115, 120]  # Example strikes
        expiries = [datetime.now() + timedelta(days=x) for x in [7, 14, 30, 60]]

        # Create matrices for Greeks
        n_symbols = len(symbols)
        n_strikes = len(strikes)

        delta_matrix = np.zeros((n_symbols, n_strikes))
        gamma_matrix = np.zeros((n_symbols, n_strikes))
        theta_matrix = np.zeros((n_symbols, n_strikes))
        vega_matrix = np.zeros((n_symbols, n_strikes))
        risk_levels = np.zeros((n_symbols, n_strikes))

        for i, symbol in enumerate(symbols):
            position = positions[symbol]
            for j, strike in enumerate(strikes):
                # Populate with position Greeks (simplified)
                delta_matrix[i, j] = position.delta
                gamma_matrix[i, j] = position.gamma
                theta_matrix[i, j] = position.theta
                vega_matrix[i, j] = position.vega

                # Calculate risk level (0=low, 1=medium, 2=high)
                risk_score = abs(position.delta) + abs(position.gamma) + abs(position.vega)
                if risk_score > 100:
                    risk_levels[i, j] = 2
                elif risk_score > 50:
                    risk_levels[i, j] = 1
                else:
                    risk_levels[i, j] = 0

        return GreeksHeatMap(
            symbols=symbols,
            strikes=strikes,
            expiries=expiries,
            delta_matrix=delta_matrix,
            gamma_matrix=gamma_matrix,
            theta_matrix=theta_matrix,
            vega_matrix=vega_matrix,
            risk_levels=risk_levels
        )

class RiskMonitor:
    """Portfolio risk monitoring"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def calculate_risk_metrics(self, positions: Dict[str, Position]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        if not positions:
            return {}

        var_metrics = await self.calculate_var(positions, 0.95)
        stress_test_results = await self.calculate_stress_tests(positions)
        concentration_risk = await self.calculate_concentration_risk(positions)

        return {
            'var_metrics': var_metrics,
            'stress_tests': stress_test_results,
            'concentration_risk': concentration_risk,
            'leverage_ratio': await self.calculate_leverage_ratio(positions)
        }

    async def calculate_var(self, positions: Dict[str, Position], confidence_level: float) -> Dict[str, float]:
        """Calculate Value at Risk using Monte Carlo simulation"""
        return self.analytics_framework.analytics_engine.risk_analyzer.calculate_var(
            positions, confidence_level
        )

    async def calculate_stress_tests(self, positions: Dict[str, Position]) -> Dict[str, Any]:
        """Perform stress tests"""
        return self.analytics_framework.analytics_engine.risk_analyzer.stress_test_portfolio(positions)

    async def calculate_concentration_risk(self, positions: Dict[str, Position]) -> Dict[str, float]:
        """Calculate concentration risk metrics"""
        if not positions:
            return {}

        total_exposure = sum(abs(pos.market_value) for pos in positions.values())

        # Calculate largest position exposure
        max_position = max(abs(pos.market_value) for pos in positions.values())
        max_concentration = max_position / total_exposure

        # Calculate Herfindahl index
        exposures = [abs(pos.market_value) / total_exposure for pos in positions.values()]
        herfindahl_index = sum(exp ** 2 for exp in exposures)

        return {
            'max_position_concentration': max_concentration,
            'herfindahl_index': herfindahl_index,
            'effective_positions': 1.0 / herfindahl_index if herfindahl_index > 0 else 0
        }

    async def calculate_leverage_ratio(self, positions: Dict[str, Position]) -> float:
        """Calculate portfolio leverage ratio"""
        total_exposure = sum(abs(pos.market_value) for pos in positions.values())
        net_value = sum(pos.market_value for pos in positions.values())

        return total_exposure / max(abs(net_value), 1.0)