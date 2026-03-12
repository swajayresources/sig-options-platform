"""
Comprehensive Test Suite for Options Pricing Engine

Tests for Black-Scholes, Binomial, Monte Carlo models and Greeks calculation
with validation against known analytical solutions and market benchmarks.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_api', 'src'))

from pricing_engine import (
    PricingEngine, OptionContract, MarketData, Greeks, OptionType, ExerciseType,
    BlackScholesCalculator, BinomialTreeCalculator, MonteCarloCalculator
)

class TestBlackScholesModel(unittest.TestCase):
    """Test Black-Scholes pricing model"""

    def setUp(self):
        self.calculator = BlackScholesCalculator()
        self.tolerance = 1e-6

    def test_call_option_pricing(self):
        """Test call option pricing against known values"""
        # Known test case: S=100, K=100, T=0.25, r=0.05, vol=0.2
        S, K, T, r, q, vol = 100, 100, 0.25, 0.05, 0.0, 0.2

        call_price = self.calculator.price_call(S, K, r, q, vol, T)
        expected_price = 5.875  # Known analytical value

        self.assertAlmostEqual(call_price, expected_price, places=2,
                              msg="Call option price mismatch")

    def test_put_option_pricing(self):
        """Test put option pricing against known values"""
        S, K, T, r, q, vol = 100, 100, 0.25, 0.05, 0.0, 0.2

        put_price = self.calculator.price_put(S, K, r, q, vol, T)
        expected_price = 4.635  # Known analytical value

        self.assertAlmostEqual(put_price, expected_price, places=2,
                              msg="Put option price mismatch")

    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        S, K, T, r, q, vol = 100, 105, 0.5, 0.03, 0.01, 0.25

        call_price = self.calculator.price_call(S, K, r, q, vol, T)
        put_price = self.calculator.price_put(S, K, r, q, vol, T)

        # Put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
        lhs = call_price - put_price
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

        self.assertAlmostEqual(lhs, rhs, places=6,
                              msg="Put-call parity violation")

    def test_greeks_calculation(self):
        """Test Greeks calculation accuracy"""
        option = OptionContract(
            symbol="TEST_CALL",
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            expiry=30/365.0,
            underlying="SPY"
        )

        market_data = MarketData(
            spot_price=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
            time_to_expiry=30/365.0
        )

        greeks = self.calculator.calculate_greeks(option, market_data)

        # Test delta range for ATM call
        self.assertGreater(greeks.delta, 0.4, "Delta too low for ATM call")
        self.assertLess(greeks.delta, 0.7, "Delta too high for ATM call")

        # Test gamma positivity
        self.assertGreater(greeks.gamma, 0, "Gamma should be positive")

        # Test vega positivity
        self.assertGreater(greeks.vega, 0, "Vega should be positive")

        # Test theta negativity for long option
        self.assertLess(greeks.theta, 0, "Theta should be negative for long option")

    def test_implied_volatility(self):
        """Test implied volatility calculation"""
        engine = PricingEngine()

        option = OptionContract(
            symbol="TEST_CALL",
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            expiry=30/365.0,
            underlying="SPY"
        )

        market_data = MarketData(
            spot_price=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            volatility=0.25,
            time_to_expiry=30/365.0
        )

        # Price option with known volatility
        result = engine.price_option(option, market_data)
        theoretical_price = result.price

        # Calculate implied volatility from theoretical price
        implied_vol = engine.calculate_implied_volatility(
            option, market_data, theoretical_price
        )

        self.assertAlmostEqual(implied_vol, 0.25, places=3,
                              msg="Implied volatility calculation failed")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Zero time to expiry
        option_expired = OptionContract(
            symbol="EXPIRED_CALL",
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            expiry=0.0,
            underlying="SPY"
        )

        market_data = MarketData(
            spot_price=105.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
            time_to_expiry=0.0
        )

        result = PricingEngine().price_option(option_expired, market_data)
        expected_intrinsic = max(105.0 - 100.0, 0)

        self.assertAlmostEqual(result.price, expected_intrinsic, places=6,
                              msg="Expired option should equal intrinsic value")

class TestBinomialModel(unittest.TestCase):
    """Test Binomial Tree pricing model"""

    def setUp(self):
        self.model = BinomialTreeCalculator(steps=100)
        self.tolerance = 0.05  # 5% tolerance for numerical methods

    def test_european_convergence(self):
        """Test convergence to Black-Scholes for European options"""
        option = OptionContract(
            symbol="TEST_CALL",
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            expiry=90/365.0,
            underlying="SPY"
        )

        market_data = MarketData(
            spot_price=100.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
            time_to_expiry=90/365.0
        )

        # Calculate using both models
        bs_calculator = BlackScholesCalculator()
        bs_price = bs_calculator.price_call(
            market_data.spot_price, option.strike,
            market_data.risk_free_rate, market_data.dividend_yield,
            market_data.volatility, market_data.time_to_expiry
        )

        binomial_result = self.model.price_option(option, market_data)

        relative_error = abs(binomial_result.price - bs_price) / bs_price

        self.assertLess(relative_error, self.tolerance,
                       f"Binomial model should converge to Black-Scholes: "
                       f"BS={bs_price:.4f}, Binomial={binomial_result.price:.4f}")

    def test_american_option_premium(self):
        """Test that American options are worth at least as much as European"""
        option_european = OptionContract(
            symbol="EUR_PUT",
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=110.0,
            expiry=180/365.0,
            underlying="SPY"
        )

        option_american = OptionContract(
            symbol="AMR_PUT",
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.AMERICAN,
            strike=110.0,
            expiry=180/365.0,
            underlying="SPY"
        )

        market_data = MarketData(
            spot_price=100.0,
            risk_free_rate=0.06,
            dividend_yield=0.0,
            volatility=0.3,
            time_to_expiry=180/365.0
        )

        european_result = self.model.price_option(option_european, market_data)
        american_result = self.model.price_option(option_american, market_data)

        self.assertGreaterEqual(american_result.price, european_result.price,
                               "American option should be worth at least as much as European")

class TestMonteCarloModel(unittest.TestCase):
    """Test Monte Carlo pricing model"""

    def setUp(self):
        self.model = MonteCarloCalculator(num_simulations=50000, random_seed=42)
        self.tolerance = 0.1  # 10% tolerance for Monte Carlo

    def test_european_option_convergence(self):
        """Test Monte Carlo convergence to Black-Scholes"""
        option = OptionContract(
            symbol="MC_CALL",
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=100.0,
            expiry=60/365.0,
            underlying="SPY"
        )

        market_data = MarketData(
            spot_price=105.0,
            risk_free_rate=0.04,
            dividend_yield=0.01,
            volatility=0.25,
            time_to_expiry=60/365.0
        )

        # Calculate using both models
        bs_calculator = BlackScholesCalculator()
        bs_price = bs_calculator.price_call(
            market_data.spot_price, option.strike,
            market_data.risk_free_rate, market_data.dividend_yield,
            market_data.volatility, market_data.time_to_expiry
        )

        mc_result = self.model.price_european_option(option, market_data)

        relative_error = abs(mc_result.price - bs_price) / bs_price

        self.assertLess(relative_error, self.tolerance,
                       f"Monte Carlo should approximate Black-Scholes: "
                       f"BS={bs_price:.4f}, MC={mc_result.price:.4f}")

    def test_path_generation(self):
        """Test that generated paths follow geometric Brownian motion properties"""
        # Generate multiple paths
        S0, r, q, vol, T = 100.0, 0.05, 0.02, 0.2, 1.0
        num_paths = 1000
        steps = 252

        final_prices = []
        for _ in range(num_paths):
            path = self.model._generate_price_path(
                MarketData(S0, r, q, vol, T), steps
            )
            final_prices.append(path[-1])

        final_prices = np.array(final_prices)

        # Test expected final price (geometric Brownian motion)
        expected_final = S0 * np.exp((r - q) * T)
        actual_mean = np.mean(final_prices)

        relative_error = abs(actual_mean - expected_final) / expected_final
        self.assertLess(relative_error, 0.1,
                       f"Final price distribution mean should match theory: "
                       f"Expected={expected_final:.2f}, Actual={actual_mean:.2f}")

class TestPricingEngineIntegration(unittest.TestCase):
    """Integration tests for the complete pricing engine"""

    def setUp(self):
        self.engine = PricingEngine()

    def test_portfolio_pricing(self):
        """Test portfolio pricing functionality"""
        # Create a portfolio of options
        options = [
            (OptionContract("SPY_CALL_400", OptionType.CALL, ExerciseType.EUROPEAN,
                           400.0, 30/365.0, "SPY"),
             MarketData(405.0, 0.03, 0.01, 0.2, 30/365.0),
             10),  # Long 10 calls

            (OptionContract("SPY_PUT_395", OptionType.PUT, ExerciseType.EUROPEAN,
                           395.0, 30/365.0, "SPY"),
             MarketData(405.0, 0.03, 0.01, 0.2, 30/365.0),
             -5),  # Short 5 puts

            (OptionContract("SPY_CALL_410", OptionType.CALL, ExerciseType.EUROPEAN,
                           410.0, 30/365.0, "SPY"),
             MarketData(405.0, 0.03, 0.01, 0.2, 30/365.0),
             -10),  # Short 10 calls
        ]

        portfolio_result = self.engine.price_portfolio(options)

        # Verify portfolio structure
        self.assertIn('total_value', portfolio_result)
        self.assertIn('total_greeks', portfolio_result)
        self.assertIn('positions', portfolio_result)

        # Verify all positions were priced
        self.assertEqual(len(portfolio_result['positions']), 3)

        # Verify Greeks aggregation
        total_greeks = portfolio_result['total_greeks']
        self.assertIsInstance(total_greeks.delta, (int, float))
        self.assertIsInstance(total_greeks.gamma, (int, float))

    def test_greeks_symmetry(self):
        """Test Greeks symmetry properties"""
        base_option = OptionContract(
            "SYM_TEST", OptionType.CALL, ExerciseType.EUROPEAN,
            100.0, 45/365.0, "SPY"
        )

        base_market = MarketData(100.0, 0.05, 0.0, 0.25, 45/365.0)

        base_result = self.engine.price_option(base_option, base_market)

        # Test spot symmetry for ATM options
        market_up = MarketData(101.0, 0.05, 0.0, 0.25, 45/365.0)
        market_down = MarketData(99.0, 0.05, 0.0, 0.25, 45/365.0)

        result_up = self.engine.price_option(base_option, market_up)
        result_down = self.engine.price_option(base_option, market_down)

        # Gamma should be approximately equal for small moves
        gamma_diff = abs(result_up.greeks.gamma - result_down.greeks.gamma)
        self.assertLess(gamma_diff / base_result.greeks.gamma, 0.05,
                       "Gamma should be symmetric for small moves")

class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and extreme scenarios"""

    def setUp(self):
        self.engine = PricingEngine()

    def test_extreme_volatilities(self):
        """Test pricing with extreme volatility values"""
        option = OptionContract(
            "EXTREME_VOL", OptionType.CALL, ExerciseType.EUROPEAN,
            100.0, 30/365.0, "SPY"
        )

        # Test very low volatility
        low_vol_market = MarketData(100.0, 0.05, 0.0, 0.001, 30/365.0)
        low_vol_result = self.engine.price_option(option, low_vol_market)

        self.assertTrue(low_vol_result.success, "Low volatility pricing should succeed")
        self.assertGreater(low_vol_result.price, 0, "Price should be positive")

        # Test high volatility
        high_vol_market = MarketData(100.0, 0.05, 0.0, 2.0, 30/365.0)
        high_vol_result = self.engine.price_option(option, high_vol_market)

        self.assertTrue(high_vol_result.success, "High volatility pricing should succeed")
        self.assertGreater(high_vol_result.price, low_vol_result.price,
                          "Higher volatility should increase option value")

    def test_extreme_moneyness(self):
        """Test pricing for deep ITM and OTM options"""
        market_data = MarketData(100.0, 0.05, 0.0, 0.2, 30/365.0)

        # Deep ITM call
        deep_itm_call = OptionContract(
            "DEEP_ITM", OptionType.CALL, ExerciseType.EUROPEAN,
            50.0, 30/365.0, "SPY"
        )

        itm_result = self.engine.price_option(deep_itm_call, market_data)
        intrinsic_value = max(100.0 - 50.0, 0)

        self.assertGreater(itm_result.price, intrinsic_value,
                          "Deep ITM option should be worth more than intrinsic")

        # Deep OTM call
        deep_otm_call = OptionContract(
            "DEEP_OTM", OptionType.CALL, ExerciseType.EUROPEAN,
            200.0, 30/365.0, "SPY"
        )

        otm_result = self.engine.price_option(deep_otm_call, market_data)

        self.assertGreater(otm_result.price, 0, "Deep OTM option should have some value")
        self.assertLess(otm_result.price, 1.0, "Deep OTM option should have small value")

    def test_near_expiry(self):
        """Test pricing behavior near expiration"""
        market_data = MarketData(105.0, 0.05, 0.0, 0.2, 1/365.0)  # 1 day to expiry

        # ITM call near expiry
        itm_call = OptionContract(
            "NEAR_EXPIRY_ITM", OptionType.CALL, ExerciseType.EUROPEAN,
            100.0, 1/365.0, "SPY"
        )

        result = self.engine.price_option(itm_call, market_data)
        intrinsic = max(105.0 - 100.0, 0)

        # Should be very close to intrinsic value
        self.assertAlmostEqual(result.price, intrinsic, delta=0.5,
                              msg="Near expiry ITM option should approach intrinsic value")

def run_comprehensive_tests():
    """Run all tests and generate report"""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestBlackScholesModel,
        TestBinomialModel,
        TestMonteCarloModel,
        TestPricingEngineIntegration,
        TestNumericalStability
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)