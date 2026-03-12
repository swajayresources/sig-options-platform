"""
Monte Carlo Simulation Engine
Advanced Monte Carlo simulation for options strategy validation and risk analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm, t, skewnorm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from .backtesting_engine import BacktestingEngine, Order, MarketData, OptionType, OptionContract

@dataclass
class MonteCarloResult:
    simulation_id: int
    final_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    total_return: float
    volatility: float
    var_95: float
    var_99: float

@dataclass
class MonteCarloSummary:
    num_simulations: int
    mean_return: float
    std_return: float
    percentiles: Dict[str, float]
    sharpe_statistics: Dict[str, float]
    drawdown_statistics: Dict[str, float]
    risk_metrics: Dict[str, float]
    success_probability: float
    tail_risk_analysis: Dict[str, float]

class MonteCarloEngine:
    """
    Advanced Monte Carlo simulation engine for options strategies
    """

    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
        self.logger = logging.getLogger(__name__)

    def run_monte_carlo_backtest(self, strategy_function: Callable,
                                start_date: datetime, end_date: datetime,
                                market_scenarios: Optional[Dict[str, Any]] = None) -> MonteCarloSummary:
        """
        Run Monte Carlo simulation with multiple market scenarios
        """
        self.logger.info(f"Starting Monte Carlo simulation with {self.num_simulations} runs")

        results = []
        scenarios = market_scenarios or self._get_default_scenarios()

        for sim_id in range(self.num_simulations):
            # Generate scenario-based market data
            scenario_data = self._generate_scenario_market_data(
                start_date, end_date, scenarios, sim_id
            )

            # Run backtest with simulated data
            engine = BacktestingEngine()
            backtest_results = self._run_simulation_backtest(
                engine, strategy_function, scenario_data
            )

            # Calculate simulation metrics
            sim_result = self._calculate_simulation_metrics(backtest_results, sim_id)
            results.append(sim_result)

            if (sim_id + 1) % 100 == 0:
                self.logger.info(f"Completed {sim_id + 1}/{self.num_simulations} simulations")

        # Aggregate results
        summary = self._aggregate_simulation_results(results)

        self.logger.info("Monte Carlo simulation completed")
        return summary

    def _get_default_scenarios(self) -> Dict[str, Any]:
        """
        Default market scenarios for Monte Carlo simulation
        """
        return {
            'volatility_regime': {
                'low_vol': {'prob': 0.3, 'vol_multiplier': 0.6},
                'normal_vol': {'prob': 0.5, 'vol_multiplier': 1.0},
                'high_vol': {'prob': 0.2, 'vol_multiplier': 2.0}
            },
            'trend_regime': {
                'bear': {'prob': 0.2, 'drift': -0.05},
                'sideways': {'prob': 0.6, 'drift': 0.02},
                'bull': {'prob': 0.2, 'drift': 0.08}
            },
            'jump_risk': {
                'prob_jump': 0.02,  # 2% chance of jump per day
                'jump_mean': -0.02,  # Negative jump on average
                'jump_std': 0.05
            },
            'correlation_breakdown': {
                'prob_breakdown': 0.05,  # 5% chance of correlation breakdown
                'breakdown_duration': 20,  # Days
                'correlation_shift': 0.5  # How much correlations shift
            }
        }

    def _generate_scenario_market_data(self, start_date: datetime, end_date: datetime,
                                     scenarios: Dict[str, Any], sim_id: int) -> List[MarketData]:
        """
        Generate market data based on scenario parameters
        """
        market_data = []
        current_date = start_date
        base_price = 100.0

        # Sample scenario parameters
        vol_regime = np.random.choice(
            list(scenarios['volatility_regime'].keys()),
            p=[scenarios['volatility_regime'][k]['prob'] for k in scenarios['volatility_regime']]
        )
        trend_regime = np.random.choice(
            list(scenarios['trend_regime'].keys()),
            p=[scenarios['trend_regime'][k]['prob'] for k in scenarios['trend_regime']]
        )

        base_volatility = 0.25 * scenarios['volatility_regime'][vol_regime]['vol_multiplier']
        drift = scenarios['trend_regime'][trend_regime]['drift']

        # Generate correlated underlying assets
        num_assets = 5
        correlation_matrix = self._generate_correlation_matrix(num_assets)
        asset_prices = [base_price] * num_assets

        day_count = 0
        while current_date <= end_date:
            # Generate correlated returns
            independent_returns = np.random.normal(0, base_volatility / np.sqrt(252), num_assets)
            correlated_returns = np.dot(np.linalg.cholesky(correlation_matrix), independent_returns)

            # Add trend
            correlated_returns += drift / 252

            # Add jump risk
            if np.random.random() < scenarios['jump_risk']['prob_jump']:
                jump_size = np.random.normal(
                    scenarios['jump_risk']['jump_mean'],
                    scenarios['jump_risk']['jump_std']
                )
                correlated_returns[0] += jump_size  # Apply jump to main asset

            # Update asset prices
            for i in range(num_assets):
                asset_prices[i] *= (1 + correlated_returns[i])

            # Generate options data for main asset
            main_price = asset_prices[0]
            options_data = self._generate_options_data(main_price, current_date, base_volatility)

            market_data.append(options_data)
            current_date += timedelta(days=1)
            day_count += 1

        return market_data

    def _generate_correlation_matrix(self, num_assets: int) -> np.ndarray:
        """
        Generate realistic correlation matrix
        """
        # Create base correlations
        base_corr = 0.3
        correlations = np.full((num_assets, num_assets), base_corr)
        np.fill_diagonal(correlations, 1.0)

        # Add some randomness while keeping matrix positive definite
        noise = np.random.normal(0, 0.1, (num_assets, num_assets))
        noise = (noise + noise.T) / 2  # Make symmetric
        correlations += noise

        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(correlations)
        eigenvals = np.maximum(eigenvals, 0.1)  # Ensure positive eigenvalues
        correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Normalize diagonal to 1
        sqrt_diag = np.sqrt(np.diag(correlations))
        correlations = correlations / np.outer(sqrt_diag, sqrt_diag)

        return correlations

    def _generate_options_data(self, underlying_price: float, current_date: datetime,
                             volatility: float) -> MarketData:
        """
        Generate realistic options data for Monte Carlo simulation
        """
        strikes = [underlying_price * k for k in [0.9, 0.95, 1.0, 1.05, 1.1]]
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

                    # Calculate theoretical price with volatility smile
                    time_to_expiry = (expiry - current_date).days / 365.0
                    if time_to_expiry > 0:
                        # Add volatility smile effect
                        moneyness = strike / underlying_price
                        smile_adjustment = self._calculate_vol_smile(moneyness)
                        adjusted_vol = volatility * smile_adjustment

                        price = self._black_scholes_price(
                            underlying_price, strike, adjusted_vol, time_to_expiry, opt_type
                        )

                        # Add realistic bid-ask spread
                        spread_pct = 0.02 + 0.01 * abs(moneyness - 1.0)  # Wider spreads for OTM
                        spread = price * spread_pct
                        bid = price - spread / 2
                        ask = price + spread / 2

                        option_prices[option_id] = price
                        bid_ask_spreads[option_id] = (max(0.01, bid), ask)
                        volatilities[option_id] = adjusted_vol
                        volumes[option_id] = np.random.poisson(200)  # Poisson-distributed volume
                        open_interest[option_id] = np.random.randint(100, 5000)

        return MarketData(
            timestamp=current_date,
            underlying_price=underlying_price,
            option_prices=option_prices,
            bid_ask_spreads=bid_ask_spreads,
            volatilities=volatilities,
            volumes=volumes,
            open_interest=open_interest
        )

    def _calculate_vol_smile(self, moneyness: float) -> float:
        """
        Calculate volatility smile adjustment
        """
        # Simplified volatility smile model
        skew_factor = -0.5  # Negative skew (puts more expensive)
        smile_factor = 0.2  # Convexity

        log_moneyness = np.log(moneyness)
        smile_adjustment = 1 + skew_factor * log_moneyness + smile_factor * (log_moneyness ** 2)

        return max(0.5, smile_adjustment)  # Minimum 50% of base vol

    def _black_scholes_price(self, spot: float, strike: float, vol: float,
                           time_to_expiry: float, option_type: OptionType,
                           risk_free_rate: float = 0.05) -> float:
        """Black-Scholes pricing with Monte Carlo adjustments"""
        if time_to_expiry <= 0:
            if option_type == OptionType.CALL:
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)

        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol ** 2) * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
        d2 = d1 - vol * np.sqrt(time_to_expiry)

        if option_type == OptionType.CALL:
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return max(0.01, price)  # Minimum price

    def _run_simulation_backtest(self, engine: BacktestingEngine,
                               strategy_function: Callable,
                               market_data: List[MarketData]) -> Dict[str, Any]:
        """
        Run backtest for a single simulation
        """
        trades = []
        equity_curve = []
        greeks_history = []

        for data in market_data:
            # Generate orders from strategy
            orders = strategy_function(data, engine.positions, engine.current_capital)

            # Execute orders
            for order in orders:
                trade = engine.execute_order(order, data)
                if trade:
                    trades.append(trade)

            # Calculate portfolio metrics
            portfolio_metrics = engine.calculate_portfolio_value(data)
            equity_curve.append((data.timestamp, portfolio_metrics['total_value']))
            greeks_history.append(portfolio_metrics)

        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'greeks_history': greeks_history,
            'final_capital': engine.current_capital
        }

    def _calculate_simulation_metrics(self, backtest_results: Dict[str, Any],
                                    sim_id: int) -> MonteCarloResult:
        """
        Calculate metrics for a single simulation
        """
        equity_curve = backtest_results['equity_curve']

        if len(equity_curve) < 2:
            return MonteCarloResult(sim_id, 0, 0, 0, 0, 0, 0, 0)

        values = [point[1] for point in equity_curve]
        initial_value = values[0]
        final_value = values[-1]

        # Calculate returns
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

        # Basic metrics
        final_pnl = final_value - initial_value
        total_return = (final_value - initial_value) / initial_value * 100

        # Volatility
        volatility = np.std(returns) * np.sqrt(252) * 100 if returns else 0

        # Sharpe ratio
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

        # Maximum drawdown
        peak = initial_value
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Value at Risk
        var_95 = np.percentile(returns, 5) * 100 if returns else 0
        var_99 = np.percentile(returns, 1) * 100 if returns else 0

        return MonteCarloResult(
            simulation_id=sim_id,
            final_pnl=final_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            volatility=volatility,
            var_95=var_95,
            var_99=var_99
        )

    def _aggregate_simulation_results(self, results: List[MonteCarloResult]) -> MonteCarloSummary:
        """
        Aggregate results from all simulations
        """
        if not results:
            return MonteCarloSummary(0, 0, 0, {}, {}, {}, {}, 0, {})

        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        drawdowns = [r.max_drawdown for r in results]
        vars_95 = [r.var_95 for r in results]

        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Percentiles
        percentiles = {
            '5th': np.percentile(returns, 5),
            '10th': np.percentile(returns, 10),
            '25th': np.percentile(returns, 25),
            '50th': np.percentile(returns, 50),
            '75th': np.percentile(returns, 75),
            '90th': np.percentile(returns, 90),
            '95th': np.percentile(returns, 95)
        }

        # Sharpe statistics
        sharpe_statistics = {
            'mean': np.mean(sharpe_ratios),
            'std': np.std(sharpe_ratios),
            'min': np.min(sharpe_ratios),
            'max': np.max(sharpe_ratios),
            'positive_sharpe_pct': len([s for s in sharpe_ratios if s > 0]) / len(sharpe_ratios) * 100
        }

        # Drawdown statistics
        drawdown_statistics = {
            'mean': np.mean(drawdowns),
            'std': np.std(drawdowns),
            'max': np.max(drawdowns),
            'percentile_95': np.percentile(drawdowns, 95)
        }

        # Risk metrics
        risk_metrics = {
            'var_95_mean': np.mean(vars_95),
            'var_95_worst': np.min(vars_95),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }

        # Success probability (positive returns)
        success_probability = len([r for r in returns if r > 0]) / len(returns) * 100

        # Tail risk analysis
        tail_returns = [r for r in returns if r <= np.percentile(returns, 5)]
        tail_risk_analysis = {
            'tail_expectation': np.mean(tail_returns) if tail_returns else 0,
            'tail_volatility': np.std(tail_returns) if tail_returns else 0,
            'worst_case': np.min(returns),
            'tail_ratio': abs(np.mean(tail_returns)) / np.mean([r for r in returns if r > 0]) if tail_returns else 0
        }

        return MonteCarloSummary(
            num_simulations=len(results),
            mean_return=mean_return,
            std_return=std_return,
            percentiles=percentiles,
            sharpe_statistics=sharpe_statistics,
            drawdown_statistics=drawdown_statistics,
            risk_metrics=risk_metrics,
            success_probability=success_probability,
            tail_risk_analysis=tail_risk_analysis
        )

    def create_monte_carlo_visualization(self, summary: MonteCarloSummary,
                                       results: List[MonteCarloResult]) -> go.Figure:
        """
        Create comprehensive Monte Carlo visualization
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Return Distribution', 'Sharpe Ratio Distribution',
                'Drawdown Analysis', 'Risk-Return Scatter',
                'Percentile Analysis', 'Tail Risk Analysis'
            ]
        )

        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        drawdowns = [r.max_drawdown for r in results]

        # 1. Return Distribution
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns', opacity=0.7),
            row=1, col=1
        )

        # 2. Sharpe Ratio Distribution
        fig.add_trace(
            go.Histogram(x=sharpe_ratios, nbinsx=30, name='Sharpe Ratios', opacity=0.7),
            row=1, col=2
        )

        # 3. Drawdown Analysis
        fig.add_trace(
            go.Histogram(x=drawdowns, nbinsx=30, name='Max Drawdown', opacity=0.7),
            row=1, col=3
        )

        # 4. Risk-Return Scatter
        volatilities = [r.volatility for r in results]
        fig.add_trace(
            go.Scatter(x=volatilities, y=returns, mode='markers',
                      name='Risk-Return', opacity=0.6),
            row=2, col=1
        )

        # 5. Percentile Analysis
        percentile_values = list(summary.percentiles.values())
        percentile_labels = list(summary.percentiles.keys())
        fig.add_trace(
            go.Bar(x=percentile_labels, y=percentile_values, name='Percentiles'),
            row=2, col=2
        )

        # 6. Tail Risk Analysis
        tail_metrics = ['Worst Case', 'Tail Expectation', 'Tail Volatility']
        tail_values = [
            summary.tail_risk_analysis['worst_case'],
            summary.tail_risk_analysis['tail_expectation'],
            summary.tail_risk_analysis['tail_volatility']
        ]
        fig.add_trace(
            go.Bar(x=tail_metrics, y=tail_values, name='Tail Risk'),
            row=2, col=3
        )

        fig.update_layout(
            height=800,
            title_text=f"Monte Carlo Analysis - {summary.num_simulations} Simulations",
            showlegend=False
        )

        return fig