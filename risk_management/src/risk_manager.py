"""
Advanced Risk Management and Hedging System

Comprehensive risk management framework for options trading including
VaR calculation, scenario analysis, dynamic hedging, and limit monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Import trading system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python_api', 'src'))

from pricing_engine import OptionContract, MarketData, Greeks, PricingEngine, OptionType

logger = logging.getLogger(__name__)

class RiskMetricType(Enum):
    VAR_95 = "VAR_95"
    VAR_99 = "VAR_99"
    EXPECTED_SHORTFALL = "EXPECTED_SHORTFALL"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    SHARPE_RATIO = "SHARPE_RATIO"
    SORTINO_RATIO = "SORTINO_RATIO"

class HedgeType(Enum):
    DELTA_HEDGE = "DELTA_HEDGE"
    GAMMA_HEDGE = "GAMMA_HEDGE"
    VEGA_HEDGE = "VEGA_HEDGE"
    DYNAMIC_HEDGE = "DYNAMIC_HEDGE"

@dataclass
class RiskLimit:
    """Risk limit definition"""
    name: str
    limit_type: str  # 'absolute', 'percentage', 'volatility'
    threshold: float
    warning_threshold: float
    current_value: float = 0.0
    breach_count: int = 0
    last_breach: Optional[datetime] = None

@dataclass
class Position:
    """Trading position with risk metrics"""
    symbol: str
    option_type: str
    strike: float
    expiry: datetime
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class HedgeRecommendation:
    """Hedge recommendation from risk system"""
    hedge_type: HedgeType
    instrument: str
    quantity: int
    target_exposure: float
    current_exposure: float
    urgency: str  # 'low', 'medium', 'high', 'critical'
    expected_cost: float
    reasoning: str

@dataclass
class RiskReport:
    """Comprehensive risk report"""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    portfolio_greeks: Greeks
    limit_breaches: List[RiskLimit]
    hedge_recommendations: List[HedgeRecommendation]
    stress_test_results: Dict[str, float]

class VaRCalculator:
    """Value at Risk calculation using multiple methods"""

    @staticmethod
    def parametric_var(returns: np.ndarray, confidence_level: float = 0.95,
                      holding_period: int = 1) -> float:
        """Parametric VaR assuming normal distribution"""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Scale for holding period
        scaled_std = std_return * np.sqrt(holding_period)

        # VaR calculation
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mean_return * holding_period + z_score * scaled_std)

        return var

    @staticmethod
    def historical_var(returns: np.ndarray, confidence_level: float = 0.95,
                      holding_period: int = 1) -> float:
        """Historical VaR using empirical distribution"""
        if len(returns) < 10:
            return VaRCalculator.parametric_var(returns, confidence_level, holding_period)

        # Scale returns for holding period
        if holding_period > 1:
            scaled_returns = returns * np.sqrt(holding_period)
        else:
            scaled_returns = returns

        # Calculate percentile
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(scaled_returns, percentile)

        return var

    @staticmethod
    def monte_carlo_var(portfolio_value: float, vol: float, confidence_level: float = 0.95,
                       holding_period: int = 1, num_simulations: int = 10000) -> float:
        """Monte Carlo VaR simulation"""
        # Generate random price changes
        dt = holding_period / 252  # Convert to years
        random_returns = np.random.normal(0, vol * np.sqrt(dt), num_simulations)

        # Calculate portfolio values
        portfolio_changes = portfolio_value * (np.exp(random_returns) - 1)

        # Calculate VaR
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(portfolio_changes, percentile)

        return var

    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Expected Shortfall (Conditional VaR)"""
        if len(returns) < 10:
            return 0.0

        var = VaRCalculator.historical_var(returns, confidence_level)
        tail_losses = returns[returns <= -var]

        if len(tail_losses) == 0:
            return var

        return -np.mean(tail_losses)

class GreeksAggregator:
    """Aggregate and analyze portfolio Greeks"""

    @staticmethod
    def aggregate_portfolio_greeks(positions: List[Position]) -> Greeks:
        """Aggregate Greeks across all positions"""
        total_greeks = Greeks()

        for position in positions:
            total_greeks.delta += position.delta * position.quantity
            total_greeks.gamma += position.gamma * position.quantity
            total_greeks.theta += position.theta * position.quantity
            total_greeks.vega += position.vega * position.quantity
            total_greeks.rho += position.rho * position.quantity

        return total_greeks

    @staticmethod
    def calculate_greeks_exposure(positions: List[Position],
                                underlying_price: float) -> Dict[str, Dict]:
        """Calculate Greeks exposure by underlying and maturity"""
        exposure_by_underlying = {}

        for position in positions:
            underlying = position.symbol.split('_')[0] if '_' in position.symbol else 'UNKNOWN'

            if underlying not in exposure_by_underlying:
                exposure_by_underlying[underlying] = {
                    'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0,
                    'positions': []
                }

            exp = exposure_by_underlying[underlying]
            exp['delta'] += position.delta * position.quantity
            exp['gamma'] += position.gamma * position.quantity
            exp['theta'] += position.theta * position.quantity
            exp['vega'] += position.vega * position.quantity
            exp['rho'] += position.rho * position.quantity
            exp['positions'].append(position)

        return exposure_by_underlying

class StressTester:
    """Comprehensive stress testing framework"""

    def __init__(self, pricing_engine: PricingEngine):
        self.pricing_engine = pricing_engine

    def parallel_shift_test(self, positions: List[Position], shift_amount: float) -> float:
        """Test parallel shift in underlying prices"""
        total_pnl = 0.0

        for position in positions:
            # Create stressed market data
            stressed_market_data = MarketData(
                spot_price=100.0 * (1 + shift_amount),  # Mock base price
                risk_free_rate=0.02,
                dividend_yield=0.0,
                volatility=0.25,
                time_to_expiry=30/365.0
            )

            # Price under stress
            option = OptionContract(
                symbol=position.symbol,
                option_type=OptionType.CALL if 'C' in position.option_type else OptionType.PUT,
                exercise_type=ExerciseType.EUROPEAN,
                strike=position.strike,
                expiry=30/365.0
            )

            result = self.pricing_engine.price_option(option, stressed_market_data)
            if result.success:
                stressed_value = result.price * position.quantity
                original_value = position.current_price * position.quantity
                total_pnl += stressed_value - original_value

        return total_pnl

    def volatility_shock_test(self, positions: List[Position], vol_shift: float) -> float:
        """Test volatility shock scenarios"""
        total_pnl = 0.0

        for position in positions:
            # Calculate vega P&L
            vega_pnl = position.vega * position.quantity * vol_shift
            total_pnl += vega_pnl

        return total_pnl

    def time_decay_test(self, positions: List[Position], days_forward: int) -> float:
        """Test time decay impact"""
        total_pnl = 0.0

        for position in positions:
            # Calculate theta P&L
            theta_pnl = position.theta * position.quantity * days_forward
            total_pnl += theta_pnl

        return total_pnl

    def interest_rate_shock(self, positions: List[Position], rate_shift: float) -> float:
        """Test interest rate shock"""
        total_pnl = 0.0

        for position in positions:
            # Calculate rho P&L
            rho_pnl = position.rho * position.quantity * rate_shift
            total_pnl += rho_pnl

        return total_pnl

    def extreme_scenarios(self, positions: List[Position]) -> Dict[str, float]:
        """Run extreme scenario tests"""
        scenarios = {
            'market_crash_20': self.parallel_shift_test(positions, -0.20),
            'market_rally_15': self.parallel_shift_test(positions, 0.15),
            'volatility_spike_50': self.volatility_shock_test(positions, 0.50),
            'volatility_collapse_30': self.volatility_shock_test(positions, -0.30),
            'time_decay_7d': self.time_decay_test(positions, 7),
            'rate_hike_200bp': self.interest_rate_shock(positions, 0.02),
            'rate_cut_150bp': self.interest_rate_shock(positions, -0.015)
        }

        return scenarios

class HedgeCalculator:
    """Calculate optimal hedges for portfolio risks"""

    def __init__(self, pricing_engine: PricingEngine):
        self.pricing_engine = pricing_engine

    def calculate_delta_hedge(self, portfolio_delta: float,
                            underlying_price: float) -> HedgeRecommendation:
        """Calculate delta hedge using underlying"""
        hedge_quantity = -int(round(portfolio_delta))

        return HedgeRecommendation(
            hedge_type=HedgeType.DELTA_HEDGE,
            instrument="SPY",  # Mock underlying
            quantity=hedge_quantity,
            target_exposure=0.0,
            current_exposure=portfolio_delta,
            urgency="medium" if abs(portfolio_delta) > 100 else "low",
            expected_cost=abs(hedge_quantity) * underlying_price * 0.001,  # Mock cost
            reasoning=f"Hedge {portfolio_delta:.1f} delta exposure with {hedge_quantity} shares"
        )

    def calculate_gamma_hedge(self, portfolio_gamma: float,
                            portfolio_delta: float) -> List[HedgeRecommendation]:
        """Calculate gamma hedge using options"""
        recommendations = []

        if abs(portfolio_gamma) > 10:  # Threshold for gamma hedging
            # Use ATM options for gamma hedge
            hedge_quantity = -int(portfolio_gamma / 0.05)  # Assume 0.05 gamma per option

            recommendations.append(HedgeRecommendation(
                hedge_type=HedgeType.GAMMA_HEDGE,
                instrument="SPY_ATM_CALL",
                quantity=hedge_quantity,
                target_exposure=0.0,
                current_exposure=portfolio_gamma,
                urgency="high" if abs(portfolio_gamma) > 50 else "medium",
                expected_cost=abs(hedge_quantity) * 2.0,  # Mock option cost
                reasoning=f"Hedge {portfolio_gamma:.1f} gamma exposure"
            ))

            # Recalculate delta after gamma hedge
            new_delta = portfolio_delta + hedge_quantity * 0.5  # Assume 0.5 delta
            if abs(new_delta) > 50:
                delta_hedge = self.calculate_delta_hedge(new_delta, 400.0)
                recommendations.append(delta_hedge)

        return recommendations

    def calculate_vega_hedge(self, portfolio_vega: float) -> HedgeRecommendation:
        """Calculate vega hedge using options"""
        # Use longer-dated options for vega hedge
        hedge_quantity = -int(portfolio_vega / 10.0)  # Assume 10 vega per option

        return HedgeRecommendation(
            hedge_type=HedgeType.VEGA_HEDGE,
            instrument="SPY_60D_ATM_CALL",
            quantity=hedge_quantity,
            target_exposure=0.0,
            current_exposure=portfolio_vega,
            urgency="medium" if abs(portfolio_vega) > 1000 else "low",
            expected_cost=abs(hedge_quantity) * 3.0,  # Mock longer-dated option cost
            reasoning=f"Hedge {portfolio_vega:.1f} vega exposure"
        )

    def calculate_optimal_hedge(self, positions: List[Position]) -> List[HedgeRecommendation]:
        """Calculate optimal hedge combination"""
        portfolio_greeks = GreeksAggregator.aggregate_portfolio_greeks(positions)
        recommendations = []

        # Delta hedge (always needed if significant)
        if abs(portfolio_greeks.delta) > 25:
            delta_hedge = self.calculate_delta_hedge(portfolio_greeks.delta, 400.0)
            recommendations.append(delta_hedge)

        # Gamma hedge (for large gamma exposures)
        if abs(portfolio_greeks.gamma) > 10:
            gamma_hedges = self.calculate_gamma_hedge(
                portfolio_greeks.gamma, portfolio_greeks.delta
            )
            recommendations.extend(gamma_hedges)

        # Vega hedge (for large vega exposures)
        if abs(portfolio_greeks.vega) > 500:
            vega_hedge = self.calculate_vega_hedge(portfolio_greeks.vega)
            recommendations.append(vega_hedge)

        return recommendations

class LimitMonitor:
    """Monitor and enforce risk limits"""

    def __init__(self, limits: Dict[str, RiskLimit]):
        self.limits = limits
        self.violation_history: List[Dict] = []

    def check_limits(self, current_values: Dict[str, float]) -> List[RiskLimit]:
        """Check current values against limits"""
        violations = []

        for limit_name, limit in self.limits.items():
            if limit_name in current_values:
                current_value = current_values[limit_name]
                limit.current_value = current_value

                # Check for violation
                if abs(current_value) > limit.threshold:
                    limit.breach_count += 1
                    limit.last_breach = datetime.now()
                    violations.append(limit)

                    # Log violation
                    self.violation_history.append({
                        'timestamp': datetime.now(),
                        'limit_name': limit_name,
                        'threshold': limit.threshold,
                        'current_value': current_value,
                        'severity': 'critical' if abs(current_value) > limit.threshold * 1.5 else 'warning'
                    })

        return violations

    def update_limit(self, limit_name: str, new_threshold: float):
        """Update limit threshold"""
        if limit_name in self.limits:
            self.limits[limit_name].threshold = new_threshold
            self.limits[limit_name].warning_threshold = new_threshold * 0.8

class RiskManager:
    """Main risk management system"""

    def __init__(self, pricing_engine: PricingEngine):
        self.pricing_engine = pricing_engine
        self.stress_tester = StressTester(pricing_engine)
        self.hedge_calculator = HedgeCalculator(pricing_engine)

        # Initialize default limits
        self.limits = {
            'portfolio_delta': RiskLimit('Portfolio Delta', 'absolute', 1000.0, 800.0),
            'portfolio_gamma': RiskLimit('Portfolio Gamma', 'absolute', 500.0, 400.0),
            'portfolio_vega': RiskLimit('Portfolio Vega', 'absolute', 10000.0, 8000.0),
            'max_position_size': RiskLimit('Max Position Size', 'absolute', 100.0, 80.0),
            'portfolio_value': RiskLimit('Portfolio Value', 'absolute', 1000000.0, 800000.0),
            'daily_var_95': RiskLimit('Daily VaR 95%', 'absolute', 50000.0, 40000.0)
        }

        self.limit_monitor = LimitMonitor(self.limits)
        self.pnl_history: List[float] = []
        self.returns_history: List[float] = []

    def update_pnl_history(self, current_pnl: float):
        """Update P&L history for risk calculations"""
        self.pnl_history.append(current_pnl)

        # Calculate returns if we have previous P&L
        if len(self.pnl_history) > 1:
            prev_pnl = self.pnl_history[-2]
            if abs(prev_pnl) > 1e-6:  # Avoid division by zero
                return_pct = (current_pnl - prev_pnl) / abs(prev_pnl)
                self.returns_history.append(return_pct)

        # Keep only recent history (252 trading days)
        if len(self.pnl_history) > 252:
            self.pnl_history = self.pnl_history[-252:]
        if len(self.returns_history) > 252:
            self.returns_history = self.returns_history[-252:]

    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Calculate portfolio VaR"""
        if len(self.returns_history) < 10:
            return 0.0

        returns_array = np.array(self.returns_history)
        return VaRCalculator.historical_var(returns_array, confidence_level)

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.pnl_history) < 2:
            return 0.0

        pnl_array = np.array(self.pnl_history)
        running_max = np.maximum.accumulate(pnl_array)
        drawdown = (pnl_array - running_max) / (running_max + 1e-8)

        return float(np.min(drawdown))

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 10:
            return 0.0

        returns_array = np.array(self.returns_history)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def generate_risk_report(self, positions: List[Position]) -> RiskReport:
        """Generate comprehensive risk report"""
        # Calculate portfolio metrics
        portfolio_value = sum(pos.market_value for pos in positions)
        total_pnl = sum(pos.unrealized_pnl for pos in positions)
        portfolio_greeks = GreeksAggregator.aggregate_portfolio_greeks(positions)

        # Calculate risk metrics
        var_95 = self.calculate_portfolio_var(0.95) * portfolio_value
        var_99 = self.calculate_portfolio_var(0.99) * portfolio_value
        expected_shortfall = VaRCalculator.expected_shortfall(
            np.array(self.returns_history), 0.95
        ) * portfolio_value
        max_drawdown = self.calculate_max_drawdown()
        sharpe_ratio = self.calculate_sharpe_ratio()

        # Check limits
        current_values = {
            'portfolio_delta': abs(portfolio_greeks.delta),
            'portfolio_gamma': abs(portfolio_greeks.gamma),
            'portfolio_vega': abs(portfolio_greeks.vega),
            'portfolio_value': portfolio_value,
            'daily_var_95': var_95
        }

        limit_breaches = self.limit_monitor.check_limits(current_values)

        # Generate hedge recommendations
        hedge_recommendations = self.hedge_calculator.calculate_optimal_hedge(positions)

        # Run stress tests
        stress_test_results = self.stress_tester.extreme_scenarios(positions)

        return RiskReport(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            total_pnl=total_pnl,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            portfolio_greeks=portfolio_greeks,
            limit_breaches=limit_breaches,
            hedge_recommendations=hedge_recommendations,
            stress_test_results=stress_test_results
        )

    def should_halt_trading(self, positions: List[Position]) -> Tuple[bool, str]:
        """Determine if trading should be halted due to risk"""
        # Check for critical limit breaches
        portfolio_greeks = GreeksAggregator.aggregate_portfolio_greeks(positions)
        portfolio_value = sum(pos.market_value for pos in positions)

        # Critical thresholds
        if abs(portfolio_greeks.delta) > self.limits['portfolio_delta'].threshold * 1.5:
            return True, f"Critical delta exposure: {portfolio_greeks.delta}"

        if abs(portfolio_greeks.gamma) > self.limits['portfolio_gamma'].threshold * 1.5:
            return True, f"Critical gamma exposure: {portfolio_greeks.gamma}"

        if portfolio_value > self.limits['portfolio_value'].threshold:
            return True, f"Portfolio value limit exceeded: {portfolio_value}"

        # Check drawdown
        max_drawdown = self.calculate_max_drawdown()
        if max_drawdown < -0.15:  # 15% drawdown threshold
            return True, f"Maximum drawdown exceeded: {max_drawdown:.2%}"

        return False, ""

    def get_risk_dashboard_data(self, positions: List[Position]) -> Dict:
        """Get risk data for dashboard display"""
        risk_report = self.generate_risk_report(positions)

        return {
            'timestamp': risk_report.timestamp.isoformat(),
            'portfolio_value': risk_report.portfolio_value,
            'total_pnl': risk_report.total_pnl,
            'var_95': risk_report.var_95,
            'var_99': risk_report.var_99,
            'expected_shortfall': risk_report.expected_shortfall,
            'max_drawdown': risk_report.max_drawdown,
            'sharpe_ratio': risk_report.sharpe_ratio,
            'portfolio_greeks': risk_report.portfolio_greeks.to_dict(),
            'limit_utilization': {
                name: {
                    'current': limit.current_value,
                    'threshold': limit.threshold,
                    'utilization_pct': (limit.current_value / limit.threshold) * 100
                    if limit.threshold != 0 else 0
                }
                for name, limit in self.limits.items()
            },
            'hedge_recommendations': [
                {
                    'type': rec.hedge_type.value,
                    'instrument': rec.instrument,
                    'quantity': rec.quantity,
                    'urgency': rec.urgency,
                    'reasoning': rec.reasoning
                }
                for rec in risk_report.hedge_recommendations
            ],
            'stress_tests': risk_report.stress_test_results
        }