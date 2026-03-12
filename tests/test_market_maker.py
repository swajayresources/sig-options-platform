"""
Test Suite for Market Making System

Comprehensive tests for market making strategies, risk management,
and automated trading functionality.
"""

import unittest
import numpy as np
import asyncio
from unittest.mock import Mock, patch
import sys
import os
from datetime import datetime, timedelta

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_api', 'src'))

from pricing_engine import OptionContract, MarketData, OptionType, ExerciseType
from market_maker import (
 MarketMaker, MarketMakerConfig, InventoryManager, VolatilityOracle,
 EdgeCalculator, RiskManager, Quote
)

class TestMarketMakerConfig(unittest.TestCase):
 """Test market maker configuration"""

 def test_default_config(self):
 """Test default configuration values"""
 config = MarketMakerConfig()

 self.assertEqual(config.max_position_size, 100)
 self.assertEqual(config.max_portfolio_delta, 1000.0)
 self.assertEqual(config.min_spread_bps, 10.0)
 self.assertTrue(0 < config.edge_target < 1.0)

 def test_config_validation(self):
 """Test configuration parameter validation"""
 # Test invalid position size
 with self.assertRaises(ValueError):
 config = MarketMakerConfig(max_position_size=-10)

 # Test invalid spread
 config = MarketMakerConfig(min_spread_bps=0.5)
 self.assertGreater(config.min_spread_bps, 0)

class TestInventoryManager(unittest.TestCase):
 """Test inventory management functionality"""

 def setUp(self):
 self.config = MarketMakerConfig()
 self.inventory = InventoryManager(self.config)

 def test_position_tracking(self):
 """Test position addition and tracking"""
 # Add initial position
 greeks = Mock()
 greeks.delta = 0.5
 greeks.gamma = 0.05
 greeks.theta = -0.02
 greeks.vega = 0.1

 self.inventory.add_position("SPY_400C", 10, 5.50, greeks)

 # Verify position exists
 self.assertIn("SPY_400C", self.inventory.positions)
 position = self.inventory.positions["SPY_400C"]
 self.assertEqual(position.quantity, 10)
 self.assertEqual(position.average_price, 5.50)

 def test_position_aggregation(self):
 """Test position aggregation for same symbol"""
 greeks = Mock()
 greeks.delta = 0.5
 greeks.gamma = 0.05
 greeks.theta = -0.02
 greeks.vega = 0.1

 # Add multiple positions in same symbol
 self.inventory.add_position("SPY_400C", 10, 5.50, greeks)
 self.inventory.add_position("SPY_400C", 5, 6.00, greeks)

 position = self.inventory.positions["SPY_400C"]
 expected_avg_price = (10 * 5.50 + 5 * 6.00) / 15

 self.assertEqual(position.quantity, 15)
 self.assertAlmostEqual(position.average_price, expected_avg_price, places=2)

 def test_portfolio_greeks_aggregation(self):
 """Test portfolio Greeks aggregation"""
 # Create mock Greeks
 greeks1 = Mock()
 greeks1.delta, greeks1.gamma, greeks1.theta, greeks1.vega = 0.5, 0.05, -0.02, 0.1

 greeks2 = Mock()
 greeks2.delta, greeks2.gamma, greeks2.theta, greeks2.vega = -0.3, 0.08, -0.03, 0.15

 # Add positions
 self.inventory.add_position("SPY_400C", 10, 5.50, greeks1)
 self.inventory.add_position("SPY_395P", -5, 3.20, greeks2)

 # Get portfolio Greeks
 portfolio_greeks = self.inventory.get_portfolio_greeks()

 expected_delta = 0.5 * 10 + (-0.3) * (-5)
 self.assertAlmostEqual(portfolio_greeks.delta, expected_delta, places=2)

 def test_inventory_skew_calculation(self):
 """Test inventory skew calculation"""
 greeks = Mock()
 greeks.delta = 0.5
 greeks.gamma = 0.05
 greeks.theta = -0.02
 greeks.vega = 0.1

 # Add position at 50% of max size
 max_size = self.config.max_position_size
 self.inventory.add_position("SPY_400C", max_size // 2, 5.50, greeks)

 skew = self.inventory.get_inventory_skew("SPY_400C")
 expected_skew = 0.5 # 50% of max position

 self.assertAlmostEqual(skew, expected_skew, places=2)

 def test_hedging_signals(self):
 """Test hedging requirement detection"""
 # Create large delta exposure
 greeks = Mock()
 greeks.delta = 50.0 # Large delta per contract
 greeks.gamma = 0.05
 greeks.theta = -0.02
 greeks.vega = 0.1

 # Add position that exceeds delta threshold
 contracts_needed = int(self.config.hedge_threshold_delta / 50.0) + 1
 self.inventory.add_position("SPY_400C", contracts_needed, 5.50, greeks)

 should_hedge, reason = self.inventory.should_hedge()

 self.assertTrue(should_hedge)
 self.assertIn("Delta", reason)

class TestVolatilityOracle(unittest.TestCase):
 """Test volatility forecasting and surface management"""

 def setUp(self):
 self.oracle = VolatilityOracle()

 def test_volatility_update(self):
 """Test volatility data updates"""
 symbol = "SPY"
 vol = 0.25

 self.oracle.update_realized_vol(symbol, vol)

 self.assertEqual(self.oracle.realized_vols[symbol], vol)
 self.assertEqual(len(self.oracle.historical_vols[symbol]), 1)

 def test_volatility_forecast(self):
 """Test volatility forecasting"""
 symbol = "SPY"

 # Add historical data
 for i in range(50):
 vol = 0.2 + 0.02 * np.sin(i / 10) # Sinusoidal volatility
 self.oracle.update_realized_vol(symbol, vol)

 forecast = self.oracle.get_vol_forecast(symbol, 1.0)

 self.assertGreater(forecast, 0.15)
 self.assertLess(forecast, 0.3)

 def test_term_structure(self):
 """Test volatility term structure generation"""
 symbol = "SPY"
 self.oracle.update_realized_vol(symbol, 0.25)

 term_structure = self.oracle.get_vol_term_structure(symbol)

 self.assertIn(0.25, term_structure) # 3M
 self.assertIn(1.0, term_structure) # 1Y

 # Term structure should be reasonable
 for maturity, vol in term_structure.items():
 self.assertGreater(vol, 0.1)
 self.assertLess(vol, 0.5)

class TestEdgeCalculator(unittest.TestCase):
 """Test edge calculation and confidence scoring"""

 def setUp(self):
 self.oracle = VolatilityOracle()
 self.edge_calculator = EdgeCalculator(self.oracle)

 def test_edge_calculation(self):
 """Test edge calculation for options"""
 # Setup volatility data
 self.oracle.update_realized_vol("SPY", 0.22)

 option = OptionContract(
 "SPY_400C", OptionType.CALL, ExerciseType.EUROPEAN,
 400.0, 30/365.0, "SPY"
 )

 market_data = MarketData(405.0, 0.03, 0.01, 0.25, 30/365.0)

 edge, confidence = self.edge_calculator.calculate_edge(
 option, market_data, 8.50, 8.80 # Mock market bid/ask
 )

 self.assertIsInstance(edge, float)
 self.assertIsInstance(confidence, float)
 self.assertGreaterEqual(confidence, 0.0)
 self.assertLessEqual(confidence, 1.0)

 def test_confidence_scoring(self):
 """Test confidence scoring based on market conditions"""
 # Test tight spread (high liquidity)
 tight_score = self.edge_calculator._calculate_liquidity_score(9.90, 10.10)

 # Test wide spread (low liquidity)
 wide_score = self.edge_calculator._calculate_liquidity_score(9.50, 10.50)

 self.assertGreater(tight_score, wide_score,
 "Tighter spreads should have higher liquidity scores")

class TestMarketMaker(unittest.TestCase):
 """Test main market making functionality"""

 def setUp(self):
 self.config = MarketMakerConfig(
 max_position_size=50,
 min_spread_bps=5.0,
 edge_target=0.01
 )
 self.market_maker = MarketMaker(self.config)
 self.market_maker.start()

 def tearDown(self):
 self.market_maker.stop()

 def test_quote_generation(self):
 """Test market maker quote generation"""
 option = OptionContract(
 "SPY_400C", OptionType.CALL, ExerciseType.EUROPEAN,
 400.0, 30/365.0, "SPY"
 )

 market_data = MarketData(405.0, 0.03, 0.01, 0.25, 30/365.0)

 quote = self.market_maker.generate_quote(option, market_data)

 self.assertIsNotNone(quote)
 self.assertGreater(quote.ask_price, quote.bid_price)
 self.assertGreater(quote.spread_bps, self.config.min_spread_bps)
 self.assertGreater(quote.bid_size, 0)
 self.assertGreater(quote.ask_size, 0)

 def test_quote_adjustment_for_inventory(self):
 """Test quote adjustment based on inventory position"""
 option = OptionContract(
 "SPY_400C", OptionType.CALL, ExerciseType.EUROPEAN,
 400.0, 30/365.0, "SPY"
 )

 market_data = MarketData(405.0, 0.03, 0.01, 0.25, 30/365.0)

 # Generate initial quote (no inventory)
 initial_quote = self.market_maker.generate_quote(option, market_data)
 initial_mid = (initial_quote.bid_price + initial_quote.ask_price) / 2

 # Simulate large long position
 greeks = Mock()
 greeks.delta, greeks.gamma, greeks.theta, greeks.vega = 0.5, 0.05, -0.02, 0.1

 self.market_maker.inventory.add_position(
 option.symbol, 40, 6.0, greeks # Large long position
 )

 # Generate quote with inventory
 inventory_quote = self.market_maker.generate_quote(option, market_data)
 inventory_mid = (inventory_quote.bid_price + inventory_quote.ask_price) / 2

 # With large long position, quotes should be skewed lower
 self.assertLess(inventory_mid, initial_mid,
 "Large long position should skew quotes lower")

 def test_fill_handling(self):
 """Test order fill processing"""
 option = OptionContract(
 "SPY_400C", OptionType.CALL, ExerciseType.EUROPEAN,
 400.0, 30/365.0, "SPY"
 )

 market_data = MarketData(405.0, 0.03, 0.01, 0.25, 30/365.0)

 # Simulate fill
 initial_pnl = self.market_maker.inventory.total_pnl

 self.market_maker.handle_fill(
 "SPY_400C", 10, 6.50, option, market_data
 )

 # Verify position was added
 self.assertIn("SPY_400C", self.market_maker.inventory.positions)

 # Verify P&L was updated
 self.assertNotEqual(self.market_maker.inventory.total_pnl, initial_pnl)

 def test_risk_limits_enforcement(self):
 """Test risk limits enforcement"""
 # Test position size limit
 large_quantity = self.config.max_position_size + 10

 can_trade = self.market_maker.risk_manager.check_position_limits(
 "SPY_400C", large_quantity
 )

 self.assertFalse(can_trade, "Should reject trades exceeding position limits")

 def test_performance_metrics(self):
 """Test performance metrics calculation"""
 # Add some P&L history
 for i in range(10):
 self.market_maker.pnl_history.append(i * 100.0)

 metrics = self.market_maker.get_performance_metrics()

 self.assertIn('total_pnl', metrics)
 self.assertIn('sharpe_ratio', metrics)
 self.assertIn('max_drawdown', metrics)
 self.assertIn('hit_ratio', metrics)

class TestIntegrationScenarios(unittest.TestCase):
 """Integration tests for realistic trading scenarios"""

 def setUp(self):
 self.config = MarketMakerConfig()
 self.market_maker = MarketMaker(self.config)
 self.market_maker.start()

 def tearDown(self):
 self.market_maker.stop()

 def test_straddle_trading_scenario(self):
 """Test trading a straddle strategy"""
 # Create ATM call and put
 call_option = OptionContract(
 "SPY_400C", OptionType.CALL, ExerciseType.EUROPEAN,
 400.0, 30/365.0, "SPY"
 )

 put_option = OptionContract(
 "SPY_400P", OptionType.PUT, ExerciseType.EUROPEAN,
 400.0, 30/365.0, "SPY"
 )

 market_data = MarketData(400.0, 0.03, 0.01, 0.25, 30/365.0)

 # Generate quotes for both legs
 call_quote = self.market_maker.generate_quote(call_option, market_data)
 put_quote = self.market_maker.generate_quote(put_option, market_data)

 self.assertIsNotNone(call_quote)
 self.assertIsNotNone(put_quote)

 # Simulate selling straddle (collect premium)
 self.market_maker.handle_fill("SPY_400C", -10, call_quote.ask_price,
 call_option, market_data)
 self.market_maker.handle_fill("SPY_400P", -10, put_quote.ask_price,
 put_option, market_data)

 # Check portfolio Greeks (should be approximately delta neutral)
 portfolio_greeks = self.market_maker.inventory.get_portfolio_greeks()
 self.assertLess(abs(portfolio_greeks.delta), 5.0,
 "Straddle should be approximately delta neutral")

 def test_volatility_spike_scenario(self):
 """Test market making during volatility spike"""
 option = OptionContract(
 "SPY_400C", OptionType.CALL, ExerciseType.EUROPEAN,
 400.0, 30/365.0, "SPY"
 )

 # Normal volatility environment
 normal_market = MarketData(400.0, 0.03, 0.01, 0.20, 30/365.0)
 normal_quote = self.market_maker.generate_quote(option, normal_market)

 # High volatility environment
 high_vol_market = MarketData(400.0, 0.03, 0.01, 0.40, 30/365.0)
 high_vol_quote = self.market_maker.generate_quote(option, high_vol_market)

 # Spreads should widen during high volatility
 self.assertGreater(high_vol_quote.spread, normal_quote.spread,
 "Spreads should widen during high volatility")

def run_market_maker_tests():
 """Run all market maker tests"""
 test_suite = unittest.TestSuite()

 test_classes = [
 TestMarketMakerConfig,
 TestInventoryManager,
 TestVolatilityOracle,
 TestEdgeCalculator,
 TestMarketMaker,
 TestIntegrationScenarios
 ]

 for test_class in test_classes:
 tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
 test_suite.addTests(tests)

 runner = unittest.TextTestRunner(verbosity=2)
 result = runner.run(test_suite)

 print(f"\n{'='*60}")
 print("MARKET MAKER TEST SUMMARY")
 print(f"{'='*60}")
 print(f"Tests run: {result.testsRun}")
 print(f"Failures: {len(result.failures)}")
 print(f"Errors: {len(result.errors)}")
 print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

 return result.wasSuccessful()

if __name__ == "__main__":
 success = run_market_maker_tests()
 exit(0 if success else 1)