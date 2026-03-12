"""
Comprehensive Testing and Benchmarking Framework for Volatility Surface Models

This module provides extensive testing capabilities including unit tests, integration tests,
performance benchmarks, numerical accuracy validation, and stress testing for all
components of the volatility surface modeling system.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import unittest
from datetime import datetime, timedelta
import concurrent.futures
from contextlib import contextmanager
import traceback

from surface_models import VolatilitySurfaceModel, SVIModel, SABRModel
from calibration_engine import VolatilityQuote, RealTimeCalibrationEngine, CalibrationConfig
from market_microstructure import MarketMicrostructureModel
from interpolation_methods import AdaptiveInterpolationFramework
from arbitrage_detection import ArbitrageMonitoringSystem
from model_validation import ModelSelectionFramework, ValidationMetrics
from surface_visualization import VolatilitySurfaceVisualizer


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    operation_name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    throughput: Optional[float] = None
    memory_usage: Optional[float] = None


@dataclass
class StressTestResult:
    """Container for stress test results"""
    scenario_name: str
    max_data_size: int
    max_execution_time: float
    memory_peak: float
    success_rate: float
    breaking_point: Optional[int] = None
    error_details: List[str] = field(default_factory=list)


class TestDataGenerator:
    """Generate synthetic test data for volatility surface testing"""

    @staticmethod
    def generate_synthetic_quotes(n_strikes: int = 20, n_expiries: int = 10,
                                 spot: float = 100.0, rate: float = 0.05,
                                 dividend: float = 0.0, base_vol: float = 0.2,
                                 vol_of_vol: float = 0.1, skew_param: float = -0.1,
                                 seed: Optional[int] = None) -> List[VolatilityQuote]:
        """Generate synthetic volatility quotes with realistic patterns"""

        if seed is not None:
            np.random.seed(seed)

        quotes = []
        expiries = np.linspace(0.08, 2.0, n_expiries)  # 1 month to 2 years

        for expiry in expiries:
            forward = spot * np.exp((rate - dividend) * expiry)

            # Generate strikes around forward with realistic distribution
            log_strikes = np.linspace(
                np.log(forward * 0.7), np.log(forward * 1.3), n_strikes
            )
            strikes = np.exp(log_strikes)

            for strike in strikes:
                # Generate realistic implied volatility with smile
                log_moneyness = np.log(strike / forward)

                # SVI-like smile pattern
                vol_smile = base_vol + skew_param * log_moneyness + \
                           0.5 * vol_of_vol * (log_moneyness ** 2)

                # Add term structure effect
                vol_term = base_vol * (1 + 0.1 * np.exp(-expiry))

                # Combine and add noise
                implied_vol = max(0.05, vol_smile + vol_term +
                                np.random.normal(0, 0.005))

                # Generate bid-ask spread
                vol_spread = 0.002 + 0.001 * abs(log_moneyness)
                bid_vol = max(0.01, implied_vol - vol_spread/2)
                ask_vol = implied_vol + vol_spread/2

                quote = VolatilityQuote(
                    strike=strike,
                    expiry=expiry,
                    implied_vol=implied_vol,
                    forward=forward,
                    timestamp=datetime.now() + timedelta(seconds=np.random.randint(3600)),
                    bid_vol=bid_vol,
                    ask_vol=ask_vol,
                    volume=np.random.randint(1, 100),
                    open_interest=np.random.randint(10, 1000)
                )
                quotes.append(quote)

        return quotes

    @staticmethod
    def generate_market_stress_scenarios() -> Dict[str, List[VolatilityQuote]]:
        """Generate market stress scenarios for testing"""

        scenarios = {}
        base_quotes = TestDataGenerator.generate_synthetic_quotes()

        # High volatility scenario
        high_vol_quotes = []
        for quote in base_quotes:
            stressed_quote = VolatilityQuote(
                strike=quote.strike,
                expiry=quote.expiry,
                implied_vol=quote.implied_vol * 2.0,  # Double volatility
                forward=quote.forward,
                timestamp=quote.timestamp,
                bid_vol=quote.bid_vol * 2.0 if quote.bid_vol else None,
                ask_vol=quote.ask_vol * 2.0 if quote.ask_vol else None,
                volume=quote.volume,
                open_interest=quote.open_interest
            )
            high_vol_quotes.append(stressed_quote)
        scenarios['high_volatility'] = high_vol_quotes

        # Market crash scenario (steep skew)
        crash_quotes = []
        for quote in base_quotes:
            log_moneyness = np.log(quote.strike / quote.forward)
            skew_adjustment = -0.3 * log_moneyness  # Strong negative skew
            stressed_vol = max(0.05, quote.implied_vol + skew_adjustment)

            crash_quote = VolatilityQuote(
                strike=quote.strike,
                expiry=quote.expiry,
                implied_vol=stressed_vol,
                forward=quote.forward * 0.8,  # Market down 20%
                timestamp=quote.timestamp,
                bid_vol=quote.bid_vol,
                ask_vol=quote.ask_vol,
                volume=quote.volume * 5,  # High volume
                open_interest=quote.open_interest
            )
            crash_quotes.append(crash_quote)
        scenarios['market_crash'] = crash_quotes

        # Low liquidity scenario
        illiquid_quotes = []
        for quote in base_quotes:
            # Wide bid-ask spreads
            wide_spread = 0.05
            illiquid_quote = VolatilityQuote(
                strike=quote.strike,
                expiry=quote.expiry,
                implied_vol=quote.implied_vol,
                forward=quote.forward,
                timestamp=quote.timestamp,
                bid_vol=max(0.01, quote.implied_vol - wide_spread/2),
                ask_vol=quote.implied_vol + wide_spread/2,
                volume=1,  # Very low volume
                open_interest=quote.open_interest
            )
            illiquid_quotes.append(illiquid_quote)
        scenarios['low_liquidity'] = illiquid_quotes

        return scenarios


class UnitTestSuite(unittest.TestCase):
    """Comprehensive unit tests for all volatility surface components"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_quotes = TestDataGenerator.generate_synthetic_quotes(seed=42)
        self.svi_model = SVIModel()
        self.sabr_model = SABRModel()

    def test_svi_model_calibration(self):
        """Test SVI model calibration"""
        try:
            self.svi_model.calibrate(self.test_quotes)

            # Check that parameters are reasonable
            for expiry, params in self.svi_model.slice_parameters.items():
                self.assertIsInstance(params['a'], float)
                self.assertIsInstance(params['b'], float)
                self.assertGreaterEqual(params['b'], 0)  # b >= 0 for no calendar arbitrage
                self.assertGreaterEqual(abs(params['rho']), 0)
                self.assertLessEqual(abs(params['rho']), 1)  # |rho| <= 1

        except Exception as e:
            self.fail(f"SVI calibration failed: {e}")

    def test_sabr_model_calibration(self):
        """Test SABR model calibration"""
        try:
            self.sabr_model.calibrate(self.test_quotes)

            # Check parameter bounds
            for expiry, params in self.sabr_model.slice_parameters.items():
                self.assertGreater(params['alpha'], 0)  # alpha > 0
                self.assertGreaterEqual(params['beta'], 0)  # beta >= 0
                self.assertLessEqual(params['beta'], 1)  # beta <= 1
                self.assertGreaterEqual(params['nu'], 0)  # nu >= 0
                self.assertGreaterEqual(abs(params['rho']), 0)
                self.assertLessEqual(abs(params['rho']), 1)  # |rho| <= 1

        except Exception as e:
            self.fail(f"SABR calibration failed: {e}")

    def test_volatility_calculation_consistency(self):
        """Test that volatility calculations are consistent"""
        self.svi_model.calibrate(self.test_quotes)

        for quote in self.test_quotes[:10]:  # Test first 10 quotes
            log_moneyness = np.log(quote.strike / quote.forward)
            calculated_vol = self.svi_model.calculate_volatility(log_moneyness, quote.expiry)

            # Should be close to market quote (within reasonable tolerance)
            self.assertGreater(calculated_vol, 0)
            self.assertLess(abs(calculated_vol - quote.implied_vol), 0.1)

    def test_arbitrage_free_constraints(self):
        """Test that calibrated models satisfy arbitrage-free constraints"""
        self.svi_model.calibrate(self.test_quotes)

        # Test calendar spread constraint (simplified)
        expiries = sorted(self.svi_model.slice_parameters.keys())
        if len(expiries) >= 2:
            t1, t2 = expiries[0], expiries[1]
            log_k = 0.0  # ATM

            vol1 = self.svi_model.calculate_volatility(log_k, t1)
            vol2 = self.svi_model.calculate_volatility(log_k, t2)

            total_var1 = vol1 ** 2 * t1
            total_var2 = vol2 ** 2 * t2

            # Total variance should be non-decreasing
            self.assertLessEqual(total_var1, total_var2)


class IntegrationTestSuite:
    """Integration tests for full system workflows"""

    def __init__(self):
        self.test_quotes = TestDataGenerator.generate_synthetic_quotes(seed=42)
        self.results = []

    def test_full_calibration_pipeline(self) -> TestResult:
        """Test complete calibration pipeline"""
        start_time = time.time()

        try:
            # Initialize components
            config = CalibrationConfig()
            engine = RealTimeCalibrationEngine(config)

            # Run calibration
            calibration_result = engine.calibrate_model(
                'svi', self.test_quotes, expiry_slice=None
            )

            # Validate results
            self.assertTrue(calibration_result.success)
            self.assertIsNotNone(calibration_result.calibrated_model)

            execution_time = time.time() - start_time

            return TestResult(
                test_name='full_calibration_pipeline',
                passed=True,
                execution_time=execution_time,
                metrics={'num_quotes': len(self.test_quotes)}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name='full_calibration_pipeline',
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_real_time_monitoring(self) -> TestResult:
        """Test real-time monitoring system"""
        start_time = time.time()

        try:
            # Set up monitoring system
            from arbitrage_detection import ArbitrageMonitoringSystem, MonitoringConfig

            config = MonitoringConfig(
                calendar_spread_tolerance=0.01,
                butterfly_spread_tolerance=0.005
            )
            monitor = ArbitrageMonitoringSystem(config)

            # Run monitoring
            violations = monitor.monitor_surface(self.test_quotes)

            # Should return list (may be empty if no violations)
            self.assertIsInstance(violations, list)

            execution_time = time.time() - start_time

            return TestResult(
                test_name='real_time_monitoring',
                passed=True,
                execution_time=execution_time,
                metrics={'violations_found': len(violations)}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name='real_time_monitoring',
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_model_selection_workflow(self) -> TestResult:
        """Test model selection framework"""
        start_time = time.time()

        try:
            selector = ModelSelectionFramework()

            candidate_models = {
                'svi': SVIModel(),
                'sabr': SABRModel()
            }

            best_name, best_model, comparison = selector.select_best_model(
                candidate_models, self.test_quotes
            )

            # Validate selection results
            self.assertIn(best_name, candidate_models.keys())
            self.assertIsNotNone(best_model)
            self.assertIsNotNone(comparison)

            execution_time = time.time() - start_time

            return TestResult(
                test_name='model_selection_workflow',
                passed=True,
                execution_time=execution_time,
                metrics={'best_model': best_name}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name='model_selection_workflow',
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def assertTrue(self, condition):
        """Simple assertion helper"""
        if not condition:
            raise AssertionError("Condition was False")

    def assertIsNotNone(self, value):
        """Assert value is not None"""
        if value is None:
            raise AssertionError("Value was None")

    def assertIsInstance(self, obj, class_type):
        """Assert object is instance of class"""
        if not isinstance(obj, class_type):
            raise AssertionError(f"Object is not instance of {class_type}")

    def assertIn(self, item, container):
        """Assert item is in container"""
        if item not in container:
            raise AssertionError(f"Item {item} not in container")


class PerformanceBenchmark:
    """Performance benchmarking for all system components"""

    def __init__(self, iterations: int = 10):
        self.iterations = iterations
        self.results = []

    def benchmark_model_calibration(self, model: VolatilitySurfaceModel,
                                   quotes: List[VolatilityQuote]) -> BenchmarkResult:
        """Benchmark model calibration performance"""

        times = []

        for _ in range(self.iterations):
            # Fresh model instance for each iteration
            model_copy = type(model)()

            start_time = time.time()
            try:
                model_copy.calibrate(quotes)
                times.append(time.time() - start_time)
            except Exception:
                continue

        if not times:
            raise ValueError("All calibration attempts failed")

        times = np.array(times)

        return BenchmarkResult(
            operation_name=f'{type(model).__name__}_calibration',
            mean_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            throughput=len(quotes) / np.mean(times)  # quotes per second
        )

    def benchmark_volatility_calculation(self, model: VolatilitySurfaceModel,
                                       n_calculations: int = 1000) -> BenchmarkResult:
        """Benchmark volatility calculation performance"""

        # Calibrate model first
        test_quotes = TestDataGenerator.generate_synthetic_quotes(seed=42)
        model.calibrate(test_quotes)

        # Generate random calculation points
        log_moneyness_points = np.random.normal(0, 0.3, n_calculations)
        expiry_points = np.random.uniform(0.1, 2.0, n_calculations)

        times = []

        for _ in range(self.iterations):
            start_time = time.time()

            for log_k, expiry in zip(log_moneyness_points, expiry_points):
                try:
                    model.calculate_volatility(log_k, expiry)
                except Exception:
                    continue

            times.append(time.time() - start_time)

        times = np.array(times)

        return BenchmarkResult(
            operation_name=f'{type(model).__name__}_calculation',
            mean_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            throughput=n_calculations / np.mean(times)  # calculations per second
        )

    def benchmark_arbitrage_detection(self, quotes: List[VolatilityQuote]) -> BenchmarkResult:
        """Benchmark arbitrage detection performance"""

        from arbitrage_detection import ArbitrageMonitoringSystem, MonitoringConfig

        config = MonitoringConfig()
        monitor = ArbitrageMonitoringSystem(config)

        times = []

        for _ in range(self.iterations):
            start_time = time.time()
            try:
                monitor.monitor_surface(quotes)
                times.append(time.time() - start_time)
            except Exception:
                continue

        if not times:
            raise ValueError("All arbitrage detection attempts failed")

        times = np.array(times)

        return BenchmarkResult(
            operation_name='arbitrage_detection',
            mean_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            iterations=len(times),
            throughput=len(quotes) / np.mean(times)
        )


class StressTesting:
    """Stress testing framework for system limits and robustness"""

    def stress_test_data_volume(self, model_class, max_quotes: int = 10000,
                              step_size: int = 500) -> StressTestResult:
        """Test system with increasing data volumes"""

        successful_sizes = []
        error_details = []
        execution_times = []

        for n_quotes in range(step_size, max_quotes + 1, step_size):
            try:
                # Generate large dataset
                quotes = TestDataGenerator.generate_synthetic_quotes(
                    n_strikes=int(np.sqrt(n_quotes)),
                    n_expiries=int(np.sqrt(n_quotes)),
                    seed=42
                )[:n_quotes]

                model = model_class()

                start_time = time.time()
                model.calibrate(quotes)
                execution_time = time.time() - start_time

                successful_sizes.append(n_quotes)
                execution_times.append(execution_time)

            except Exception as e:
                error_details.append(f"Failed at {n_quotes} quotes: {str(e)}")
                breaking_point = n_quotes - step_size if successful_sizes else 0
                break
        else:
            breaking_point = None

        if not successful_sizes:
            raise ValueError("Failed even with smallest dataset")

        return StressTestResult(
            scenario_name=f'{model_class.__name__}_volume_stress',
            max_data_size=max(successful_sizes),
            max_execution_time=max(execution_times),
            memory_peak=0,  # Would implement memory monitoring
            success_rate=len(successful_sizes) / (max_quotes // step_size),
            breaking_point=breaking_point,
            error_details=error_details
        )

    def stress_test_extreme_parameters(self, model_class) -> StressTestResult:
        """Test system with extreme market conditions"""

        stress_scenarios = TestDataGenerator.generate_market_stress_scenarios()
        successful_scenarios = []
        error_details = []
        execution_times = []

        for scenario_name, quotes in stress_scenarios.items():
            try:
                model = model_class()

                start_time = time.time()
                model.calibrate(quotes)
                execution_time = time.time() - start_time

                successful_scenarios.append(scenario_name)
                execution_times.append(execution_time)

            except Exception as e:
                error_details.append(f"Failed scenario {scenario_name}: {str(e)}")

        return StressTestResult(
            scenario_name=f'{model_class.__name__}_extreme_parameters',
            max_data_size=len(stress_scenarios),
            max_execution_time=max(execution_times) if execution_times else 0,
            memory_peak=0,
            success_rate=len(successful_scenarios) / len(stress_scenarios),
            error_details=error_details
        )


class TestRunner:
    """Main test runner that orchestrates all testing"""

    def __init__(self):
        self.unit_tests = UnitTestSuite()
        self.integration_tests = IntegrationTestSuite()
        self.benchmark = PerformanceBenchmark()
        self.stress_tester = StressTesting()
        self.results = {
            'unit_tests': [],
            'integration_tests': [],
            'benchmarks': [],
            'stress_tests': []
        }

    def run_all_tests(self, verbose: bool = True) -> Dict[str, List]:
        """Run complete test suite"""

        if verbose:
            print("Starting comprehensive test suite...")

        # Unit tests
        if verbose:
            print("\nRunning unit tests...")

        unit_test_loader = unittest.TestLoader()
        unit_test_suite = unit_test_loader.loadTestsFromTestCase(UnitTestSuite)
        unittest.TextTestRunner(verbosity=2 if verbose else 0).run(unit_test_suite)

        # Integration tests
        if verbose:
            print("\nRunning integration tests...")

        integration_methods = [
            self.integration_tests.test_full_calibration_pipeline,
            self.integration_tests.test_real_time_monitoring,
            self.integration_tests.test_model_selection_workflow
        ]

        for test_method in integration_methods:
            result = test_method()
            self.results['integration_tests'].append(result)
            if verbose:
                status = "PASSED" if result.passed else "FAILED"
                print(f"  {result.test_name}: {status} ({result.execution_time:.3f}s)")
                if not result.passed and result.error_message:
                    print(f"    Error: {result.error_message}")

        # Performance benchmarks
        if verbose:
            print("\nRunning performance benchmarks...")

        test_quotes = TestDataGenerator.generate_synthetic_quotes(seed=42)

        models_to_benchmark = [SVIModel, SABRModel]

        for model_class in models_to_benchmark:
            try:
                model = model_class()

                # Calibration benchmark
                calibration_result = self.benchmark.benchmark_model_calibration(
                    model, test_quotes
                )
                self.results['benchmarks'].append(calibration_result)

                if verbose:
                    print(f"  {calibration_result.operation_name}: "
                          f"{calibration_result.mean_time:.4f}s ± {calibration_result.std_time:.4f}s")

                # Calculation benchmark
                calculation_result = self.benchmark.benchmark_volatility_calculation(model)
                self.results['benchmarks'].append(calculation_result)

                if verbose:
                    print(f"  {calculation_result.operation_name}: "
                          f"{calculation_result.throughput:.0f} calc/s")

            except Exception as e:
                if verbose:
                    print(f"  Benchmark failed for {model_class.__name__}: {e}")

        # Arbitrage detection benchmark
        try:
            arb_result = self.benchmark.benchmark_arbitrage_detection(test_quotes)
            self.results['benchmarks'].append(arb_result)
            if verbose:
                print(f"  {arb_result.operation_name}: "
                      f"{arb_result.mean_time:.4f}s ({arb_result.throughput:.0f} quotes/s)")
        except Exception as e:
            if verbose:
                print(f"  Arbitrage detection benchmark failed: {e}")

        # Stress tests
        if verbose:
            print("\nRunning stress tests...")

        for model_class in [SVIModel, SABRModel]:
            try:
                # Volume stress test
                volume_result = self.stress_tester.stress_test_data_volume(model_class)
                self.results['stress_tests'].append(volume_result)

                if verbose:
                    print(f"  {volume_result.scenario_name}: "
                          f"Max {volume_result.max_data_size} quotes "
                          f"({volume_result.success_rate:.1%} success)")

                # Parameter stress test
                param_result = self.stress_tester.stress_test_extreme_parameters(model_class)
                self.results['stress_tests'].append(param_result)

                if verbose:
                    print(f"  {param_result.scenario_name}: "
                          f"{param_result.success_rate:.1%} success rate")

            except Exception as e:
                if verbose:
                    print(f"  Stress test failed for {model_class.__name__}: {e}")

        if verbose:
            print("\nTest suite completed!")
            self._print_summary()

        return self.results

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)

        # Integration test summary
        integration_passed = sum(1 for r in self.results['integration_tests'] if r.passed)
        integration_total = len(self.results['integration_tests'])
        print(f"Integration Tests: {integration_passed}/{integration_total} passed")

        # Benchmark summary
        benchmark_count = len(self.results['benchmarks'])
        print(f"Performance Benchmarks: {benchmark_count} completed")

        # Stress test summary
        stress_passed = sum(1 for r in self.results['stress_tests'] if r.success_rate > 0.8)
        stress_total = len(self.results['stress_tests'])
        print(f"Stress Tests: {stress_passed}/{stress_total} passed (>80% success)")


# Convenience functions
def run_quick_test():
    """Run a quick test of core functionality"""
    runner = TestRunner()

    # Just run integration tests for quick validation
    print("Running quick integration tests...")

    results = []
    test_methods = [
        runner.integration_tests.test_full_calibration_pipeline,
        runner.integration_tests.test_model_selection_workflow
    ]

    for test_method in test_methods:
        result = test_method()
        results.append(result)
        status = "PASSED" if result.passed else "FAILED"
        print(f"{result.test_name}: {status}")

    return results


def run_performance_analysis():
    """Run comprehensive performance analysis"""
    benchmark = PerformanceBenchmark(iterations=20)
    test_quotes = TestDataGenerator.generate_synthetic_quotes(
        n_strikes=50, n_expiries=20, seed=42
    )

    print("Performance Analysis")
    print("="*40)

    models = [SVIModel(), SABRModel()]

    for model in models:
        print(f"\n{type(model).__name__} Performance:")

        try:
            # Calibration performance
            calib_result = benchmark.benchmark_model_calibration(model, test_quotes)
            print(f"  Calibration: {calib_result.mean_time:.4f}s ± {calib_result.std_time:.4f}s")
            print(f"  Throughput: {calib_result.throughput:.1f} quotes/s")

            # Calculation performance
            calc_result = benchmark.benchmark_volatility_calculation(model)
            print(f"  Calculation: {calc_result.throughput:.0f} calculations/s")

        except Exception as e:
            print(f"  Benchmark failed: {e}")


# Example usage
if __name__ == "__main__":
    # Run comprehensive test suite
    runner = TestRunner()
    results = runner.run_all_tests(verbose=True)

    # Or run quick test
    # quick_results = run_quick_test()

    # Or run performance analysis
    # run_performance_analysis()