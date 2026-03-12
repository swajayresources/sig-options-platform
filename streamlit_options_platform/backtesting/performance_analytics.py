"""
Performance Analytics and Attribution Analysis
Comprehensive performance measurement and analysis tools
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import logging

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    value_at_risk_95: float
    expected_shortfall_95: float

@dataclass
class GreeksAttribution:
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    residual_pnl: float

class PerformanceAnalyzer:
    """
    Advanced performance analytics for options trading strategies
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_comprehensive_metrics(self, equity_curve: List[Tuple[datetime, float]],
                                      trades: List = None,
                                      risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        """
        if len(equity_curve) < 2:
            return self._empty_metrics()

        # Extract data
        dates = [point[0] for point in equity_curve]
        values = [point[1] for point in equity_curve]
        initial_value = values[0]

        # Calculate returns
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

        # Basic metrics
        total_return = (values[-1] - initial_value) / initial_value

        # Annualized return
        days = (dates[-1] - dates[0]).days
        years = days / 365.25
        annualized_return = (values[-1] / initial_value) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if returns else 0

        # Sharpe ratio
        excess_returns = [r - risk_free_rate/252 for r in returns]
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0

        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else np.std(returns)
        sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Maximum drawdown
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(values)

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Trade-based metrics
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0

        if trades:
            profits = [t.price * t.quantity for t in trades if t.price * t.quantity > 0]
            losses = [t.price * t.quantity for t in trades if t.price * t.quantity < 0]

            win_rate = len(profits) / len(trades) * 100 if trades else 0
            total_profit = sum(profits) if profits else 0
            total_loss = abs(sum(losses)) if losses else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0

        # Risk metrics
        var_95 = np.percentile(returns, 5) if returns else 0
        es_95 = np.mean([r for r in returns if r <= var_95]) if returns else 0

        return PerformanceMetrics(
            total_return=total_return * 100,
            annualized_return=annualized_return * 100,
            volatility=volatility * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown * 100,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=avg_win,
            average_loss=avg_loss,
            value_at_risk_95=var_95 * 100,
            expected_shortfall_95=es_95 * 100
        )

    def _calculate_max_drawdown(self, values: List[float]) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        peak = values[0]
        max_dd = 0
        max_dd_duration = 0
        current_dd_duration = 0

        for value in values:
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
                max_dd_duration = max(max_dd_duration, current_dd_duration)

        return max_dd, max_dd_duration

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for edge cases"""
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def analyze_greeks_attribution(self, greeks_history: List[Dict[str, float]],
                                 underlying_prices: List[float]) -> GreeksAttribution:
        """
        Detailed Greeks-based P&L attribution analysis
        """
        if len(greeks_history) < 2 or len(underlying_prices) < 2:
            return GreeksAttribution(0, 0, 0, 0, 0, 0)

        total_pnl_change = greeks_history[-1]['pnl'] - greeks_history[0]['pnl']

        # Calculate P&L attribution
        delta_pnl = 0
        gamma_pnl = 0
        theta_pnl = 0
        vega_pnl = 0
        rho_pnl = 0

        for i in range(1, len(greeks_history)):
            prev_greeks = greeks_history[i-1]
            curr_greeks = greeks_history[i]

            # Price movement
            price_change = underlying_prices[i] - underlying_prices[i-1]

            # Delta P&L
            delta_pnl += prev_greeks.get('delta', 0) * price_change

            # Gamma P&L (convexity adjustment)
            gamma_pnl += 0.5 * prev_greeks.get('gamma', 0) * (price_change ** 2)

            # Theta P&L (time decay)
            theta_pnl += prev_greeks.get('theta', 0)

            # Vega P&L (volatility change - simplified)
            # In practice, would need actual volatility changes
            vega_pnl += prev_greeks.get('vega', 0) * 0.01  # Assume 1% vol change

            # Rho P&L (interest rate change - simplified)
            rho_pnl += prev_greeks.get('rho', 0) * 0.001  # Assume 0.1% rate change

        # Residual P&L (unexplained)
        explained_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
        residual_pnl = total_pnl_change - explained_pnl

        return GreeksAttribution(
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            theta_pnl=theta_pnl,
            vega_pnl=vega_pnl,
            rho_pnl=rho_pnl,
            residual_pnl=residual_pnl
        )

    def create_performance_dashboard(self, metrics: PerformanceMetrics,
                                   equity_curve: List[Tuple[datetime, float]],
                                   attribution: GreeksAttribution = None) -> go.Figure:
        """
        Create comprehensive performance dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Equity Curve', 'Drawdown Analysis',
                'Return Distribution', 'Greeks Attribution',
                'Rolling Sharpe Ratio', 'Risk Metrics'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )

        if equity_curve:
            dates = [point[0] for point in equity_curve]
            values = [point[1] for point in equity_curve]

            # 1. Equity Curve
            fig.add_trace(
                go.Scatter(x=dates, y=values, name='Portfolio Value', line=dict(color='blue')),
                row=1, col=1
            )

            # 2. Drawdown Analysis
            drawdowns = self._calculate_drawdown_series(values)
            fig.add_trace(
                go.Scatter(x=dates, y=drawdowns, fill='tonexty', name='Drawdown',
                          fillcolor='rgba(255,0,0,0.3)', line=dict(color='red')),
                row=1, col=2
            )

            # 3. Return Distribution
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            fig.add_trace(
                go.Histogram(x=returns, nbinsx=30, name='Daily Returns', opacity=0.7),
                row=2, col=1
            )

        # 4. Greeks Attribution
        if attribution:
            greeks_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Residual']
            greeks_values = [
                attribution.delta_pnl, attribution.gamma_pnl, attribution.theta_pnl,
                attribution.vega_pnl, attribution.rho_pnl, attribution.residual_pnl
            ]

            fig.add_trace(
                go.Bar(x=greeks_names, y=greeks_values, name='P&L Attribution'),
                row=2, col=2
            )

        # 5. Rolling Sharpe Ratio
        if equity_curve and len(equity_curve) > 60:
            rolling_sharpe = self._calculate_rolling_sharpe(equity_curve, window=60)
            fig.add_trace(
                go.Scatter(x=dates[60:], y=rolling_sharpe, name='60-Day Rolling Sharpe'),
                row=3, col=1
            )

        # 6. Risk Metrics Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics.sharpe_ratio,
                domain={'x': [0.55, 1], 'y': [0, 0.33]},
                title={'text': "Sharpe Ratio"},
                gauge={
                    'axis': {'range': [None, 3]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 2
                    }
                }
            ),
            row=3, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Strategy Performance Dashboard",
            showlegend=True
        )

        return fig

    def _calculate_drawdown_series(self, values: List[float]) -> List[float]:
        """Calculate drawdown series"""
        peak = values[0]
        drawdowns = []

        for value in values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak
            drawdowns.append(drawdown)

        return drawdowns

    def _calculate_rolling_sharpe(self, equity_curve: List[Tuple[datetime, float]],
                                window: int = 60) -> List[float]:
        """Calculate rolling Sharpe ratio"""
        values = [point[1] for point in equity_curve]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

        rolling_sharpe = []
        for i in range(window, len(returns)):
            period_returns = returns[i-window:i]
            if len(period_returns) > 1:
                mean_return = np.mean(period_returns)
                std_return = np.std(period_returns)
                sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
                rolling_sharpe.append(sharpe)

        return rolling_sharpe

    def benchmark_comparison(self, strategy_metrics: PerformanceMetrics,
                           benchmark_data: List[float]) -> Dict[str, float]:
        """
        Compare strategy performance against benchmark
        """
        if not benchmark_data:
            return {}

        # Calculate benchmark metrics
        benchmark_returns = [(benchmark_data[i] - benchmark_data[i-1]) / benchmark_data[i-1]
                           for i in range(1, len(benchmark_data))]

        benchmark_total_return = (benchmark_data[-1] - benchmark_data[0]) / benchmark_data[0] * 100
        benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252) * 100
        benchmark_sharpe = (np.mean(benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252)) if np.std(benchmark_returns) > 0 else 0

        # Calculate comparison metrics
        excess_return = strategy_metrics.total_return - benchmark_total_return
        tracking_error = strategy_metrics.volatility - benchmark_volatility
        information_ratio = excess_return / abs(tracking_error) if tracking_error != 0 else 0

        return {
            'benchmark_return': benchmark_total_return,
            'benchmark_volatility': benchmark_volatility,
            'benchmark_sharpe': benchmark_sharpe,
            'excess_return': excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }

    def sector_attribution_analysis(self, trades: List, sector_mapping: Dict[str, str]) -> Dict[str, Dict]:
        """
        Analyze performance attribution by sector
        """
        sector_metrics = {}

        for sector in set(sector_mapping.values()):
            sector_trades = [t for t in trades if sector_mapping.get(t.contract.symbol, '') == sector]

            if sector_trades:
                sector_pnl = sum(t.price * t.quantity for t in sector_trades)
                sector_volume = sum(abs(t.quantity) for t in sector_trades)

                sector_metrics[sector] = {
                    'total_pnl': sector_pnl,
                    'trade_count': len(sector_trades),
                    'total_volume': sector_volume,
                    'avg_pnl_per_trade': sector_pnl / len(sector_trades)
                }

        return sector_metrics

    def create_risk_report(self, equity_curve: List[Tuple[datetime, float]],
                          greeks_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        """
        if not equity_curve or not greeks_history:
            return {}

        values = [point[1] for point in equity_curve]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

        # Value at Risk calculations
        var_95 = np.percentile(returns, 5) * values[-1] if returns else 0
        var_99 = np.percentile(returns, 1) * values[-1] if returns else 0

        # Expected Shortfall
        es_95 = np.mean([r for r in returns if r <= np.percentile(returns, 5)]) * values[-1] if returns else 0

        # Greeks risk metrics
        delta_history = [g.get('delta', 0) for g in greeks_history]
        gamma_history = [g.get('gamma', 0) for g in greeks_history]
        vega_history = [g.get('vega', 0) for g in greeks_history]

        max_delta_exposure = max(delta_history) if delta_history else 0
        max_gamma_exposure = max(gamma_history) if gamma_history else 0
        max_vega_exposure = max(vega_history) if vega_history else 0

        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'max_delta_exposure': max_delta_exposure,
            'max_gamma_exposure': max_gamma_exposure,
            'max_vega_exposure': max_vega_exposure,
            'avg_delta': np.mean(delta_history) if delta_history else 0,
            'avg_gamma': np.mean(gamma_history) if gamma_history else 0,
            'avg_vega': np.mean(vega_history) if vega_history else 0,
            'delta_volatility': np.std(delta_history) if delta_history else 0,
            'gamma_volatility': np.std(gamma_history) if gamma_history else 0,
            'vega_volatility': np.std(vega_history) if vega_history else 0
        }