"""
Comprehensive Test Suite for Backtesting Framework
Unit tests, integration tests, and validation tests
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtesting.backtesting_engine import (
    BacktestingEngine, Order, MarketData, OptionContract,
    OptionType, OrderType, Trade
)
from backtesting.strategy_validator import StrategyValidator, ValidationResult
from backtesting.performance_analytics import PerformanceAnalyzer, PerformanceMetrics
from backtesting.monte_carlo_engine import MonteCarloEngine

class TestBlackScholesCalculations(unittest.TestCase):
    """Test Black-Scholes pricing and Greeks calculations"""

    def setUp(self):
        self.engine = BacktestingEngine()

    def test_black_scholes_call_price(self):
        """Test Black-Scholes call option pricing"""
        spot = 100.0
        strike = 100.0
        vol = 0.2
        time_to_expiry = 0.25  # 3 months
        option_type = OptionType.CALL

        price = self.engine._black_scholes_price(spot, strike, vol, time_to_expiry, option_type)

        # Expected price should be positive and reasonable
        self.assertGreater(price, 0)
        self.assertLess(price, spot)  # Call price should be less than spot for ATM

    def test_black_scholes_put_price(self):
        """Test Black-Scholes put option pricing"""
        spot = 100.0
        strike = 100.0
        vol = 0.2
        time_to_expiry = 0.25
        option_type = OptionType.PUT

        price = self.engine._black_scholes_price(spot, strike, vol, time_to_expiry, option_type)

        # Expected price should be positive and reasonable
        self.assertGreater(price, 0)
        self.assertLess(price, strike)  # Put price should be less than strike for ATM

    def test_greeks_calculation(self):
        """Test Greeks calculations"""
        spot = 100.0
        strike = 100.0
        vol = 0.2
        time_to_expiry = 0.25
        option_type = OptionType.CALL

        greeks = self.engine.calculate_greeks(spot, strike, vol, time_to_expiry, option_type)

        # Test that all Greeks are present
        required_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in required_greeks:
            self.assertIn(greek, greeks)

        # Test reasonable ranges for ATM call
        self.assertGreater(greeks['delta'], 0.4)  # ATM call delta should be around 0.5
        self.assertLess(greeks['delta'], 0.6)
        self.assertGreater(greeks['gamma'], 0)  # Gamma should be positive
        self.assertLess(greeks['theta'], 0)  # Theta should be negative (time decay)
        self.assertGreater(greeks['vega'], 0)  # Vega should be positive

    def test_option_expiry_handling(self):
        """Test handling of expired options"""
        spot = 100.0
        strike = 95.0
        vol = 0.2
        time_to_expiry = 0.0  # Expired
        option_type = OptionType.CALL

        price = self.engine._black_scholes_price(spot, strike, vol, time_to_expiry, option_type)
        intrinsic_value = max(0, spot - strike)

        self.assertEqual(price, intrinsic_value)

class TestBacktestingEngine(unittest.TestCase):
    """Test backtesting engine functionality"""

    def setUp(self):
        self.engine = BacktestingEngine(initial_capital=100000.0)

    def test_initial_capital(self):
        """Test initial capital setup"""
        self.assertEqual(self.engine.initial_capital, 100000.0)
        self.assertEqual(self.engine.current_capital, 100000.0)
        self.assertEqual(len(self.engine.positions), 0)

    def test_order_execution(self):
        """Test order execution"""
        # Create test market data
        market_data = self._create_test_market_data()

        # Create test order
        contract = OptionContract(
            symbol="SPY",
            strike=100.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL
        )

        order = Order(
            contract=contract,
            quantity=10,
            order_type=OrderType.BUY
        )

        # Execute order
        trade = self.engine.execute_order(order, market_data)

        # Verify trade execution
        self.assertIsNotNone(trade)
        self.assertEqual(trade.quantity, 10)
        self.assertGreater(trade.price, 0)

    def test_position_tracking(self):
        """Test position tracking after trades"""
        market_data = self._create_test_market_data()

        contract = OptionContract(
            symbol="SPY",
            strike=100.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL
        )

        # Buy 10 contracts
        buy_order = Order(contract=contract, quantity=10, order_type=OrderType.BUY)
        self.engine.execute_order(buy_order, market_data)

        # Sell 5 contracts
        sell_order = Order(contract=contract, quantity=-5, order_type=OrderType.SELL)
        self.engine.execute_order(sell_order, market_data)

        # Check final position
        self.assertEqual(len(self.engine.positions), 1)
        position = list(self.engine.positions.values())[0]
        self.assertEqual(position.quantity, 5)

    def test_portfolio_value_calculation(self):
        """Test portfolio value and Greeks calculation"""
        market_data = self._create_test_market_data()

        # Add some positions
        contract = OptionContract(
            symbol="SPY",
            strike=100.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL
        )

        order = Order(contract=contract, quantity=10, order_type=OrderType.BUY)
        self.engine.execute_order(order, market_data)

        # Calculate portfolio metrics
        portfolio_metrics = self.engine.calculate_portfolio_value(market_data)

        # Verify metrics
        required_metrics = ['total_value', 'cash', 'delta', 'gamma', 'theta', 'vega', 'rho', 'pnl']
        for metric in required_metrics:
            self.assertIn(metric, portfolio_metrics)

        self.assertGreater(portfolio_metrics['total_value'], 0)

    def test_risk_limit_checking(self):
        """Test risk limit checking"""
        portfolio_metrics = {
            'delta': 2000,  # Exceeds max_portfolio_delta (1000)
            'gamma': 300,
            'vega': 8000
        }

        violations = self.engine.check_risk_limits(portfolio_metrics)

        # Should have delta violation
        self.assertGreater(len(violations), 0)
        self.assertTrue(any('Delta' in violation for violation in violations))

    def _create_test_market_data(self) -> MarketData:
        """Create test market data"""
        expiry = datetime.now() + timedelta(days=30)
        option_id = f"SPY_{expiry.strftime('%y%m%d')}C100"

        return MarketData(
            timestamp=datetime.now(),
            underlying_price=100.0,
            option_prices={option_id: 5.0},
            bid_ask_spreads={option_id: (4.8, 5.2)},
            volatilities={option_id: 0.25},
            volumes={option_id: 1000},
            open_interest={option_id: 5000}
        )

class TestStrategyValidator(unittest.TestCase):
    """Test strategy validation framework"""

    def setUp(self):
        self.validator = StrategyValidator()

    def test_validation_result_structure(self):
        """Test validation result structure"""
        # Create a simple test strategy
        def test_strategy(market_data, positions, capital):
            return []  # No trades

        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()

        result = self.validator.validate_strategy(test_strategy, start_date, end_date)

        # Check result structure
        self.assertIsInstance(result, ValidationResult)
        self.assertIsInstance(result.in_sample_metrics, dict)
        self.assertIsInstance(result.out_of_sample_metrics, dict)
        self.assertIsInstance(result.walk_forward_results, list)
        self.assertIsInstance(result.monte_carlo_results, dict)
        self.assertIsInstance(result.overfitting_score, float)
        self.assertIsInstance(result.validation_score, float)

    def test_overfitting_detection(self):
        """Test overfitting detection"""
        # Mock metrics with obvious overfitting
        in_sample = {'sharpe_ratio': 2.0, 'total_return': 50.0}
        out_sample = {'sharpe_ratio': -0.5, 'total_return': -20.0}

        overfitting_score = self.validator._detect_overfitting(in_sample, out_sample)

        # Should detect high overfitting
        self.assertGreater(overfitting_score, 0.5)

    def test_validation_score_calculation(self):
        """Test validation score calculation"""
        in_sample = {'sharpe_ratio': 1.5, 'total_return': 20.0}
        out_sample = {'sharpe_ratio': 1.2, 'total_return': 15.0}
        walk_forward = [{'metrics': {'total_return': 10.0}}, {'metrics': {'total_return': 12.0}}]
        overfitting_score = 0.2

        validation_score = self.validator._calculate_validation_score(
            in_sample, out_sample, walk_forward, overfitting_score
        )

        # Should be a reasonable score
        self.assertGreaterEqual(validation_score, 0)
        self.assertLessEqual(validation_score, 100)

class TestPerformanceAnalytics(unittest.TestCase):
    """Test performance analytics"""

    def setUp(self):
        self.analyzer = PerformanceAnalyzer()

    def test_performance_metrics_calculation(self):
        """Test comprehensive performance metrics calculation"""
        # Create test equity curve
        base_value = 100000
        equity_curve = [
            (datetime.now() - timedelta(days=i), base_value * (1 + 0.001 * np.random.randn()))
            for i in range(252, 0, -1)
        ]

        metrics = self.analyzer.calculate_comprehensive_metrics(equity_curve)

        # Check that all metrics are calculated
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)

    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Create values with known drawdown
        values = [100, 110, 105, 95, 90, 100, 120]
        max_dd, duration = self.analyzer._calculate_max_drawdown(values)

        # Maximum drawdown should be from 110 to 90 = 18.18%
        expected_dd = (110 - 90) / 110
        self.assertAlmostEqual(max_dd, expected_dd, places=2)

    def test_greeks_attribution(self):
        """Test Greeks-based P&L attribution"""
        # Create test Greeks history
        greeks_history = [
            {'pnl': 0, 'delta': 100, 'gamma': 10, 'theta': -5, 'vega': 20, 'rho': 1},
            {'pnl': 1000, 'delta': 110, 'gamma': 12, 'theta': -6, 'vega': 22, 'rho': 1.1}
        ]
        underlying_prices = [100, 102]

        attribution = self.analyzer.analyze_greeks_attribution(greeks_history, underlying_prices)

        # Check attribution structure
        self.assertIsInstance(attribution.delta_pnl, float)
        self.assertIsInstance(attribution.gamma_pnl, float)
        self.assertIsInstance(attribution.theta_pnl, float)

class TestMonteCarloEngine(unittest.TestCase):
    """Test Monte Carlo simulation engine"""

    def setUp(self):
        self.mc_engine = MonteCarloEngine(num_simulations=10)  # Small number for testing

    def test_scenario_generation(self):
        """Test market scenario generation"""
        scenarios = self.mc_engine._get_default_scenarios()

        # Check scenario structure
        self.assertIn('volatility_regime', scenarios)
        self.assertIn('trend_regime', scenarios)
        self.assertIn('jump_risk', scenarios)

        # Check probabilities sum to 1
        vol_probs = sum(scenarios['volatility_regime'][k]['prob'] for k in scenarios['volatility_regime'])
        self.assertAlmostEqual(vol_probs, 1.0, places=2)

    def test_correlation_matrix_generation(self):
        """Test correlation matrix generation"""
        num_assets = 5
        corr_matrix = self.mc_engine._generate_correlation_matrix(num_assets)

        # Check matrix properties
        self.assertEqual(corr_matrix.shape, (num_assets, num_assets))

        # Check diagonal elements are 1
        for i in range(num_assets):
            self.assertAlmostEqual(corr_matrix[i, i], 1.0, places=6)

        # Check symmetry
        self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))

        # Check positive semi-definite
        eigenvals = np.linalg.eigvals(corr_matrix)
        self.assertTrue(np.all(eigenvals >= -1e-8))

    def test_volatility_smile_calculation(self):
        """Test volatility smile calculation"""
        # Test different moneyness levels
        moneyness_levels = [0.9, 0.95, 1.0, 1.05, 1.1]

        for moneyness in moneyness_levels:
            smile_adj = self.mc_engine._calculate_vol_smile(moneyness)
            self.assertGreater(smile_adj, 0)
            self.assertIsInstance(smile_adj, float)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete backtesting framework"""

    def test_complete_backtest_workflow(self):
        """Test complete backtesting workflow"""
        # Create a simple strategy
        def simple_strategy(market_data, positions, capital):
            orders = []

            # Simple buy and hold strategy
            if len(positions) == 0 and capital > 50000:
                contract = OptionContract(
                    symbol="SPY",
                    strike=market_data.underlying_price,
                    expiry=market_data.timestamp + timedelta(days=30),
                    option_type=OptionType.CALL
                )

                order = Order(
                    contract=contract,
                    quantity=1,
                    order_type=OrderType.BUY
                )
                orders.append(order)

            return orders

        # Run backtest
        engine = BacktestingEngine()
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)

        results = engine.run_backtest(simple_strategy, start_date, end_date)

        # Check results structure
        self.assertIn('trades', results)
        self.assertIn('equity_curve', results)
        self.assertIn('performance_metrics', results)

    def test_validation_workflow(self):
        """Test complete validation workflow"""
        def test_strategy(market_data, positions, capital):
            return []  # No trades for simplicity

        validator = StrategyValidator()
        start_date = datetime.now() - timedelta(days=100)
        end_date = datetime.now() - timedelta(days=30)

        # This should complete without errors
        try:
            result = validator.validate_strategy(test_strategy, start_date, end_date)
            self.assertIsInstance(result, ValidationResult)
        except Exception as e:
            self.fail(f"Validation workflow failed: {e}")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_empty_market_data(self):
        """Test handling of empty market data"""
        engine = BacktestingEngine()

        # Test with empty market data list
        def dummy_strategy(market_data, positions, capital):
            return []

        results = engine.run_backtest(dummy_strategy, datetime.now(), datetime.now())

        # Should handle gracefully
        self.assertIsInstance(results, dict)

    def test_invalid_order_execution(self):
        """Test handling of invalid orders"""
        engine = BacktestingEngine()

        # Create market data without the option we're trying to trade
        market_data = MarketData(
            timestamp=datetime.now(),
            underlying_price=100.0,
            option_prices={},  # Empty
            bid_ask_spreads={},
            volatilities={},
            volumes={},
            open_interest={}
        )

        contract = OptionContract(
            symbol="INVALID",
            strike=100.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL
        )

        order = Order(contract=contract, quantity=10, order_type=OrderType.BUY)

        # Should return None for invalid order
        trade = engine.execute_order(order, market_data)
        self.assertIsNone(trade)

    def test_extreme_market_conditions(self):
        """Test handling of extreme market conditions"""
        engine = BacktestingEngine()

        # Test with extreme prices
        market_data = MarketData(
            timestamp=datetime.now(),
            underlying_price=0.01,  # Very low price
            option_prices={'test_option': 1000.0},  # Very high option price
            bid_ask_spreads={'test_option': (999.0, 1001.0)},
            volatilities={'test_option': 5.0},  # 500% volatility
            volumes={'test_option': 1},
            open_interest={'test_option': 1}
        )

        # Should handle without crashing
        portfolio_metrics = engine.calculate_portfolio_value(market_data)
        self.assertIsInstance(portfolio_metrics, dict)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestBlackScholesCalculations,
        TestBacktestingEngine,
        TestStrategyValidator,
        TestPerformanceAnalytics,
        TestMonteCarloEngine,
        TestIntegration,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"BACKTESTING FRAMEWORK TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")