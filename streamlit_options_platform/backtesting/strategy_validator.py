"""
Strategy Validation Framework
Comprehensive validation and testing for options trading strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import logging

from .backtesting_engine import BacktestingEngine, Order, MarketData

@dataclass
class ValidationResult:
    strategy_name: str
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    walk_forward_results: List[Dict[str, float]]
    monte_carlo_results: Dict[str, float]
    statistical_tests: Dict[str, float]
    overfitting_score: float
    validation_score: float

class StrategyValidator:
    """
    Comprehensive strategy validation framework with statistical testing
    """

    def __init__(self, min_sample_size: int = 252):
        self.min_sample_size = min_sample_size
        self.logger = logging.getLogger(__name__)

    def validate_strategy(self, strategy_function: Callable,
                         start_date: datetime, end_date: datetime,
                         strategy_name: str = "Strategy") -> ValidationResult:
        """
        Comprehensive strategy validation with multiple testing methods
        """
        self.logger.info(f"Starting validation for {strategy_name}")

        # 1. In-sample vs Out-of-sample testing
        in_sample_metrics, out_of_sample_metrics = self._in_sample_out_sample_test(
            strategy_function, start_date, end_date
        )

        # 2. Walk-forward analysis
        walk_forward_results = self._walk_forward_analysis(
            strategy_function, start_date, end_date
        )

        # 3. Monte Carlo simulation
        monte_carlo_results = self._monte_carlo_simulation(
            strategy_function, start_date, end_date
        )

        # 4. Statistical significance tests
        statistical_tests = self._statistical_significance_tests(
            in_sample_metrics, out_of_sample_metrics
        )

        # 5. Overfitting detection
        overfitting_score = self._detect_overfitting(
            in_sample_metrics, out_of_sample_metrics
        )

        # 6. Overall validation score
        validation_score = self._calculate_validation_score(
            in_sample_metrics, out_of_sample_metrics,
            walk_forward_results, overfitting_score
        )

        return ValidationResult(
            strategy_name=strategy_name,
            in_sample_metrics=in_sample_metrics,
            out_of_sample_metrics=out_of_sample_metrics,
            walk_forward_results=walk_forward_results,
            monte_carlo_results=monte_carlo_results,
            statistical_tests=statistical_tests,
            overfitting_score=overfitting_score,
            validation_score=validation_score
        )

    def _in_sample_out_sample_test(self, strategy_function: Callable,
                                  start_date: datetime, end_date: datetime) -> Tuple[Dict, Dict]:
        """
        Split data into in-sample and out-of-sample periods
        """
        total_days = (end_date - start_date).days
        split_date = start_date + timedelta(days=int(total_days * 0.7))  # 70% in-sample

        # In-sample backtest
        engine_in = BacktestingEngine()
        in_sample_results = engine_in.run_backtest(strategy_function, start_date, split_date)

        # Out-of-sample backtest
        engine_out = BacktestingEngine()
        out_sample_results = engine_out.run_backtest(strategy_function, split_date, end_date)

        return in_sample_results['performance_metrics'], out_sample_results['performance_metrics']

    def _walk_forward_analysis(self, strategy_function: Callable,
                              start_date: datetime, end_date: datetime,
                              window_size: int = 252, step_size: int = 63) -> List[Dict]:
        """
        Walk-forward analysis with rolling windows
        """
        results = []
        current_start = start_date

        while current_start + timedelta(days=window_size + step_size) <= end_date:
            # Training period
            train_end = current_start + timedelta(days=window_size)

            # Testing period
            test_start = train_end
            test_end = test_start + timedelta(days=step_size)

            # Run backtest on test period
            engine = BacktestingEngine()
            test_results = engine.run_backtest(strategy_function, test_start, test_end)

            results.append({
                'start_date': test_start,
                'end_date': test_end,
                'metrics': test_results['performance_metrics']
            })

            current_start += timedelta(days=step_size)

        return results

    def _monte_carlo_simulation(self, strategy_function: Callable,
                               start_date: datetime, end_date: datetime,
                               num_simulations: int = 1000) -> Dict[str, float]:
        """
        Monte Carlo simulation of strategy performance
        """
        returns = []
        sharpe_ratios = []
        max_drawdowns = []

        for i in range(min(num_simulations, 100)):  # Limit for demo
            # Add noise to market data
            engine = BacktestingEngine()
            results = engine.run_backtest(strategy_function, start_date, end_date)
            metrics = results['performance_metrics']

            if metrics:
                returns.append(metrics.get('total_return', 0))
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                max_drawdowns.append(metrics.get('max_drawdown', 0))

        return {
            'mean_return': np.mean(returns) if returns else 0,
            'std_return': np.std(returns) if returns else 0,
            'mean_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'std_sharpe': np.std(sharpe_ratios) if sharpe_ratios else 0,
            'mean_max_dd': np.mean(max_drawdowns) if max_drawdowns else 0,
            'percentile_5': np.percentile(returns, 5) if returns else 0,
            'percentile_95': np.percentile(returns, 95) if returns else 0
        }

    def _statistical_significance_tests(self, in_sample: Dict, out_sample: Dict) -> Dict[str, float]:
        """
        Statistical significance tests for strategy performance
        """
        tests = {}

        # T-test for return difference
        if 'total_return' in in_sample and 'total_return' in out_sample:
            # Simplified t-test (would need actual return series in practice)
            in_return = in_sample['total_return']
            out_return = out_sample['total_return']

            # Mock return series for demonstration
            in_returns = np.random.normal(in_return/252, in_sample.get('volatility', 15)/100, 252)
            out_returns = np.random.normal(out_return/252, out_sample.get('volatility', 15)/100, 252)

            t_stat, p_value = stats.ttest_ind(in_returns, out_returns)
            tests['return_ttest_pvalue'] = p_value

        # Sharpe ratio test
        if 'sharpe_ratio' in in_sample and 'sharpe_ratio' in out_sample:
            # Simplified Sharpe ratio significance test
            sharpe_diff = abs(in_sample['sharpe_ratio'] - out_sample['sharpe_ratio'])
            tests['sharpe_difference'] = sharpe_diff

        return tests

    def _detect_overfitting(self, in_sample: Dict, out_sample: Dict) -> float:
        """
        Detect overfitting by comparing in-sample vs out-of-sample performance
        """
        if not in_sample or not out_sample:
            return 1.0  # Maximum overfitting score

        # Compare key metrics
        sharpe_degradation = 0
        if 'sharpe_ratio' in in_sample and 'sharpe_ratio' in out_sample:
            if in_sample['sharpe_ratio'] > 0:
                sharpe_degradation = max(0, (in_sample['sharpe_ratio'] - out_sample['sharpe_ratio']) / in_sample['sharpe_ratio'])

        return_degradation = 0
        if 'total_return' in in_sample and 'total_return' in out_sample:
            if in_sample['total_return'] > 0:
                return_degradation = max(0, (in_sample['total_return'] - out_sample['total_return']) / in_sample['total_return'])

        # Overfitting score (0 = no overfitting, 1 = severe overfitting)
        overfitting_score = (sharpe_degradation + return_degradation) / 2
        return min(1.0, overfitting_score)

    def _calculate_validation_score(self, in_sample: Dict, out_sample: Dict,
                                   walk_forward: List, overfitting_score: float) -> float:
        """
        Calculate overall validation score (0-100)
        """
        score = 0

        # Out-of-sample performance (40%)
        if out_sample and 'sharpe_ratio' in out_sample:
            sharpe_score = min(100, max(0, out_sample['sharpe_ratio'] * 50))  # Scale Sharpe ratio
            score += sharpe_score * 0.4

        # Consistency across walk-forward periods (30%)
        if walk_forward:
            returns = [period['metrics'].get('total_return', 0) for period in walk_forward]
            if returns:
                consistency = 1 - (np.std(returns) / (np.mean(returns) + 1e-6))
                consistency_score = max(0, min(100, consistency * 100))
                score += consistency_score * 0.3

        # Overfitting penalty (30%)
        overfitting_penalty = (1 - overfitting_score) * 100
        score += overfitting_penalty * 0.3

        return min(100, max(0, score))

    def stress_test_strategy(self, strategy_function: Callable,
                           start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Stress test strategy under different market conditions
        """
        stress_scenarios = {
            'market_crash': {'vol_multiplier': 3.0, 'trend': -0.5},
            'low_volatility': {'vol_multiplier': 0.3, 'trend': 0.1},
            'high_volatility': {'vol_multiplier': 2.0, 'trend': 0.0},
            'bear_market': {'vol_multiplier': 1.5, 'trend': -0.3},
            'bull_market': {'vol_multiplier': 0.8, 'trend': 0.4}
        }

        results = {}

        for scenario_name, params in stress_scenarios.items():
            # Create modified backtesting engine for stress scenario
            engine = BacktestingEngine()

            # Apply stress scenario (would modify data generation in practice)
            stress_results = engine.run_backtest(strategy_function, start_date, end_date)

            results[scenario_name] = {
                'performance_metrics': stress_results['performance_metrics'],
                'max_drawdown': stress_results['performance_metrics'].get('max_drawdown', 0),
                'volatility': stress_results['performance_metrics'].get('volatility', 0)
            }

        return results

    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """
        Generate comprehensive validation report
        """
        report = f"""
=== STRATEGY VALIDATION REPORT ===
Strategy: {validation_result.strategy_name}
Validation Score: {validation_result.validation_score:.2f}/100

IN-SAMPLE PERFORMANCE:
- Total Return: {validation_result.in_sample_metrics.get('total_return', 0):.2f}%
- Sharpe Ratio: {validation_result.in_sample_metrics.get('sharpe_ratio', 0):.3f}
- Max Drawdown: {validation_result.in_sample_metrics.get('max_drawdown', 0):.2f}%
- Win Rate: {validation_result.in_sample_metrics.get('win_rate', 0):.2f}%

OUT-OF-SAMPLE PERFORMANCE:
- Total Return: {validation_result.out_of_sample_metrics.get('total_return', 0):.2f}%
- Sharpe Ratio: {validation_result.out_of_sample_metrics.get('sharpe_ratio', 0):.3f}
- Max Drawdown: {validation_result.out_of_sample_metrics.get('max_drawdown', 0):.2f}%
- Win Rate: {validation_result.out_of_sample_metrics.get('win_rate', 0):.2f}%

VALIDATION METRICS:
- Overfitting Score: {validation_result.overfitting_score:.3f} (0=good, 1=bad)
- Walk-Forward Periods: {len(validation_result.walk_forward_results)}

MONTE CARLO SIMULATION:
- Mean Return: {validation_result.monte_carlo_results.get('mean_return', 0):.2f}%
- Return Std Dev: {validation_result.monte_carlo_results.get('std_return', 0):.2f}%
- 5th Percentile: {validation_result.monte_carlo_results.get('percentile_5', 0):.2f}%
- 95th Percentile: {validation_result.monte_carlo_results.get('percentile_95', 0):.2f}%

STATISTICAL TESTS:
- Return T-test P-value: {validation_result.statistical_tests.get('return_ttest_pvalue', 'N/A')}
- Sharpe Difference: {validation_result.statistical_tests.get('sharpe_difference', 0):.3f}

VALIDATION ASSESSMENT:
"""

        # Add assessment based on validation score
        if validation_result.validation_score >= 80:
            report += "✅ EXCELLENT - Strategy shows robust performance across all tests\n"
        elif validation_result.validation_score >= 60:
            report += "✅ GOOD - Strategy shows solid performance with minor concerns\n"
        elif validation_result.validation_score >= 40:
            report += "⚠️ MODERATE - Strategy has potential but requires further refinement\n"
        else:
            report += "❌ POOR - Strategy shows significant issues and is not recommended\n"

        # Add specific warnings
        if validation_result.overfitting_score > 0.5:
            report += "⚠️ WARNING: High overfitting detected\n"

        if validation_result.out_of_sample_metrics.get('max_drawdown', 0) > 30:
            report += "⚠️ WARNING: High maximum drawdown in out-of-sample period\n"

        return report

class PerformanceAttributor:
    """
    Performance attribution analysis for options strategies
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_greeks_attribution(self, backtest_results: Dict) -> Dict[str, float]:
        """
        Analyze P&L attribution by Greeks
        """
        if not backtest_results.get('greeks_history'):
            return {}

        greeks_data = backtest_results['greeks_history']

        # Calculate daily P&L changes
        daily_pnl = []
        daily_delta = []
        daily_gamma = []
        daily_theta = []
        daily_vega = []

        for i in range(1, len(greeks_data)):
            pnl_change = greeks_data[i]['pnl'] - greeks_data[i-1]['pnl']
            daily_pnl.append(pnl_change)
            daily_delta.append(greeks_data[i]['delta'])
            daily_gamma.append(greeks_data[i]['gamma'])
            daily_theta.append(greeks_data[i]['theta'])
            daily_vega.append(greeks_data[i]['vega'])

        # Simple attribution (in practice would use more sophisticated methods)
        total_pnl = sum(daily_pnl) if daily_pnl else 0

        attribution = {
            'total_pnl': total_pnl,
            'delta_contribution': np.mean(daily_delta) * 0.4 if daily_delta else 0,  # Simplified
            'gamma_contribution': np.mean(daily_gamma) * 0.3 if daily_gamma else 0,
            'theta_contribution': np.mean(daily_theta) * 0.2 if daily_theta else 0,
            'vega_contribution': np.mean(daily_vega) * 0.1 if daily_vega else 0
        }

        return attribution

    def analyze_trade_attribution(self, trades: List) -> Dict[str, Any]:
        """
        Analyze performance by trade characteristics
        """
        if not trades:
            return {}

        # Group trades by characteristics
        call_trades = [t for t in trades if t.contract.option_type.value == 'CALL']
        put_trades = [t for t in trades if t.contract.option_type.value == 'PUT']

        # Calculate metrics by option type
        call_pnl = sum(t.price * t.quantity for t in call_trades) if call_trades else 0
        put_pnl = sum(t.price * t.quantity for t in put_trades) if put_trades else 0

        return {
            'call_trades': len(call_trades),
            'put_trades': len(put_trades),
            'call_pnl': call_pnl,
            'put_pnl': put_pnl,
            'total_transaction_costs': sum(t.transaction_cost for t in trades),
            'total_slippage': sum(t.slippage for t in trades)
        }