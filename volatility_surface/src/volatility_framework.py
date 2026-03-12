"""
Comprehensive Volatility Surface Framework - Main Integration Module

This module provides a unified interface to all components of the sophisticated
volatility surface modeling and calibration system, including SVI/SABR models,
real-time calibration, market microstructure modeling, and advanced analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# Import all framework components
from surface_models import VolatilitySurfaceModel, SVIModel, SABRModel
from calibration_engine import (
    RealTimeCalibrationEngine, CalibrationConfig, VolatilityQuote,
    CalibrationTrigger, MarketDataTrigger, TimeBasedTrigger, VolatilityChangeTrigger
)
from market_microstructure import (
    MarketMicrostructureModel, VolumetricQuote,
    BidAskSpreadModel, VolumeWeightedVolatility, EventDrivenVolatility
)
from interpolation_methods import (
    AdaptiveInterpolationFramework, VolatilityInterpolator,
    CubicSplineInterpolator, RBFInterpolator, KrigingInterpolator
)
from arbitrage_detection import (
    ArbitrageMonitoringSystem, MonitoringConfig, ArbitrageViolation
)
from model_validation import (
    ModelSelectionFramework, ModelComparator, ValidationMetrics,
    CrossValidationValidator, ModelBacktester
)
from surface_visualization import (
    VolatilitySurfaceVisualizer, RiskVisualization, PlotConfig,
    SurfaceVisualizationData, prepare_surface_data
)
from testing_framework import TestRunner, TestDataGenerator


@dataclass
class FrameworkConfig:
    """Configuration for the entire volatility surface framework"""
    # Model preferences
    preferred_models: List[str] = None
    default_model: str = 'svi'

    # Calibration settings
    calibration_frequency: str = 'realtime'  # 'realtime', 'hourly', 'daily'
    recalibration_threshold: float = 0.05    # 5% volatility change threshold
    min_quotes_for_calibration: int = 10

    # Market microstructure
    enable_microstructure_modeling: bool = True
    bid_ask_modeling: bool = True
    volume_weighting: bool = True

    # Risk management
    enable_arbitrage_monitoring: bool = True
    arbitrage_tolerance: float = 0.01        # 1% tolerance for arbitrage detection
    alert_on_violations: bool = True

    # Interpolation
    adaptive_interpolation: bool = True
    default_interpolation_method: str = 'cubic_spline'

    # Validation
    cross_validation_folds: int = 5
    backtest_lookback_days: int = 30

    # Visualization
    enable_visualization: bool = True
    plot_style: str = 'professional'
    interactive_plots: bool = True

    def __post_init__(self):
        if self.preferred_models is None:
            self.preferred_models = ['svi', 'sabr']


class VolatilitySurfaceFramework:
    """
    Main framework class that integrates all volatility surface modeling components

    This class provides a unified interface for:
    - Model calibration and selection
    - Real-time monitoring and updates
    - Market microstructure analysis
    - Arbitrage detection
    - Visualization and reporting
    - Performance validation
    """

    def __init__(self, config: FrameworkConfig = None):
        self.config = config or FrameworkConfig()

        # Initialize core components
        self._initialize_models()
        self._initialize_calibration_engine()
        self._initialize_microstructure_model()
        self._initialize_interpolation_framework()
        self._initialize_arbitrage_monitoring()
        self._initialize_validation_framework()
        self._initialize_visualization()

        # State management
        self.current_surface_data: Optional[SurfaceVisualizationData] = None
        self.calibrated_models: Dict[str, VolatilitySurfaceModel] = {}
        self.last_calibration_time: Optional[datetime] = None
        self.monitoring_alerts: List[ArbitrageViolation] = []

    def _initialize_models(self):
        """Initialize volatility surface models"""
        self.available_models = {
            'svi': SVIModel,
            'sabr': SABRModel
        }

        self.model_instances = {}
        for model_name in self.config.preferred_models:
            if model_name in self.available_models:
                self.model_instances[model_name] = self.available_models[model_name]()

    def _initialize_calibration_engine(self):
        """Initialize calibration engine"""
        calibration_config = CalibrationConfig(
            optimization_method='differential_evolution',
            cross_validation_folds=self.config.cross_validation_folds,
            regularization_strength=0.01,
            max_iterations=1000
        )

        self.calibration_engine = RealTimeCalibrationEngine(calibration_config)

        # Set up calibration triggers
        triggers = []

        if self.config.calibration_frequency == 'realtime':
            triggers.append(MarketDataTrigger(min_quote_change_threshold=0.001))
            triggers.append(VolatilityChangeTrigger(threshold=self.config.recalibration_threshold))
        elif self.config.calibration_frequency == 'hourly':
            triggers.append(TimeBasedTrigger(frequency_minutes=60))
        elif self.config.calibration_frequency == 'daily':
            triggers.append(TimeBasedTrigger(frequency_minutes=1440))

        self.calibration_engine.triggers = triggers

    def _initialize_microstructure_model(self):
        """Initialize market microstructure modeling"""
        if self.config.enable_microstructure_modeling:
            self.microstructure_model = MarketMicrostructureModel()

            if self.config.bid_ask_modeling:
                self.bid_ask_model = BidAskSpreadModel()
            if self.config.volume_weighting:
                self.volume_weighted_model = VolumeWeightedVolatility()
        else:
            self.microstructure_model = None

    def _initialize_interpolation_framework(self):
        """Initialize interpolation framework"""
        if self.config.adaptive_interpolation:
            self.interpolation_framework = AdaptiveInterpolationFramework()
        else:
            # Use default method
            interpolator_map = {
                'cubic_spline': CubicSplineInterpolator,
                'rbf': RBFInterpolator,
                'kriging': KrigingInterpolator
            }
            method = self.config.default_interpolation_method
            if method in interpolator_map:
                self.interpolator = interpolator_map[method]()
            else:
                self.interpolator = CubicSplineInterpolator()

    def _initialize_arbitrage_monitoring(self):
        """Initialize arbitrage monitoring system"""
        if self.config.enable_arbitrage_monitoring:
            monitoring_config = MonitoringConfig(
                calendar_spread_tolerance=self.config.arbitrage_tolerance,
                butterfly_spread_tolerance=self.config.arbitrage_tolerance,
                enable_alerts=self.config.alert_on_violations
            )
            self.arbitrage_monitor = ArbitrageMonitoringSystem(monitoring_config)
        else:
            self.arbitrage_monitor = None

    def _initialize_validation_framework(self):
        """Initialize model validation framework"""
        self.validation_framework = ModelSelectionFramework()
        self.model_comparator = ModelComparator()
        self.backtester = ModelBacktester(
            lookback_window=self.config.backtest_lookback_days
        )

    def _initialize_visualization(self):
        """Initialize visualization components"""
        if self.config.enable_visualization:
            plot_config = PlotConfig(
                style='seaborn-v0_8' if self.config.plot_style == 'professional' else 'default',
                figure_size=(12, 8),
                dpi=300
            )
            self.visualizer = VolatilitySurfaceVisualizer(plot_config)
            self.risk_visualizer = RiskVisualization(plot_config)
        else:
            self.visualizer = None

    def calibrate_surface(self, quotes: List[VolatilityQuote],
                         model_name: Optional[str] = None,
                         auto_select: bool = False) -> Dict[str, Any]:
        """
        Calibrate volatility surface with given quotes

        Args:
            quotes: List of volatility quotes
            model_name: Specific model to calibrate (or None for default)
            auto_select: Whether to automatically select best model

        Returns:
            Dictionary containing calibration results
        """

        if len(quotes) < self.config.min_quotes_for_calibration:
            raise ValueError(f"Insufficient quotes for calibration. Got {len(quotes)}, need {self.config.min_quotes_for_calibration}")

        # Apply market microstructure adjustments if enabled
        if self.microstructure_model:
            adjusted_quotes = self._apply_microstructure_adjustments(quotes)
        else:
            adjusted_quotes = quotes

        calibration_results = {}

        if auto_select:
            # Use model selection framework
            candidate_models = {name: self.available_models[name]()
                              for name in self.config.preferred_models}

            best_name, best_model, comparison = self.validation_framework.select_best_model(
                candidate_models, adjusted_quotes
            )

            self.calibrated_models[best_name] = best_model

            calibration_results = {
                'selected_model': best_name,
                'calibrated_model': best_model,
                'model_comparison': comparison,
                'calibration_time': datetime.now()
            }

        else:
            # Calibrate specific model or default
            target_model = model_name or self.config.default_model

            if target_model not in self.model_instances:
                raise ValueError(f"Model '{target_model}' not available")

            model = self.model_instances[target_model]
            calibration_result = self.calibration_engine.calibrate_model(
                target_model, adjusted_quotes
            )

            if calibration_result.success:
                self.calibrated_models[target_model] = calibration_result.calibrated_model
                calibration_results = {
                    'selected_model': target_model,
                    'calibrated_model': calibration_result.calibrated_model,
                    'calibration_metrics': calibration_result.metrics,
                    'calibration_time': datetime.now()
                }
            else:
                raise RuntimeError(f"Calibration failed: {calibration_result.error_message}")

        # Update internal state
        self.last_calibration_time = datetime.now()
        self.current_surface_data = prepare_surface_data(
            adjusted_quotes,
            calibration_results['calibrated_model']
        )

        # Run arbitrage monitoring
        if self.arbitrage_monitor:
            violations = self.arbitrage_monitor.monitor_surface(adjusted_quotes)
            if violations:
                self.monitoring_alerts.extend(violations)
                calibration_results['arbitrage_violations'] = violations

        return calibration_results

    def _apply_microstructure_adjustments(self, quotes: List[VolatilityQuote]) -> List[VolatilityQuote]:
        """Apply market microstructure adjustments to quotes"""

        if not self.microstructure_model:
            return quotes

        # Convert to volumetric quotes if needed
        volumetric_quotes = []
        for quote in quotes:
            vol_quote = VolumetricQuote(
                base_quote=quote,
                volume=getattr(quote, 'volume', 0),
                open_interest=getattr(quote, 'open_interest', 0),
                time_to_settlement=quote.expiry,
                session_info={'session': 'regular', 'time_of_day': 'mid'}
            )
            volumetric_quotes.append(vol_quote)

        # Apply microstructure modeling
        adjusted_quotes = []

        for vol_quote in volumetric_quotes:
            try:
                # Volume weighting adjustment
                if self.config.volume_weighting and hasattr(self, 'volume_weighted_model'):
                    vol_adjustment = self.volume_weighted_model.calculate_volume_weighted_volatility(
                        [vol_quote], vol_quote.base_quote.expiry
                    )
                    adjusted_vol = vol_quote.base_quote.implied_vol * vol_adjustment
                else:
                    adjusted_vol = vol_quote.base_quote.implied_vol

                # Create adjusted quote
                adjusted_quote = VolatilityQuote(
                    strike=vol_quote.base_quote.strike,
                    expiry=vol_quote.base_quote.expiry,
                    implied_vol=max(0.001, adjusted_vol),  # Ensure positive volatility
                    forward=vol_quote.base_quote.forward,
                    timestamp=vol_quote.base_quote.timestamp,
                    bid_vol=vol_quote.base_quote.bid_vol,
                    ask_vol=vol_quote.base_quote.ask_vol,
                    volume=getattr(vol_quote.base_quote, 'volume', 0),
                    open_interest=getattr(vol_quote.base_quote, 'open_interest', 0)
                )

                adjusted_quotes.append(adjusted_quote)

            except Exception as e:
                warnings.warn(f"Failed to apply microstructure adjustment: {e}")
                adjusted_quotes.append(vol_quote.base_quote)

        return adjusted_quotes

    def get_surface_volatility(self, strike: float, expiry: float,
                              forward: float, model_name: Optional[str] = None) -> float:
        """
        Get volatility from calibrated surface

        Args:
            strike: Option strike price
            expiry: Time to expiry
            forward: Forward price
            model_name: Specific model to use (or None for default)

        Returns:
            Implied volatility
        """

        target_model = model_name or self.config.default_model

        if target_model not in self.calibrated_models:
            raise ValueError(f"Model '{target_model}' not calibrated")

        model = self.calibrated_models[target_model]
        log_moneyness = np.log(strike / forward)

        return model.calculate_volatility(log_moneyness, expiry)

    def generate_surface_report(self, quotes: List[VolatilityQuote]) -> Dict[str, Any]:
        """Generate comprehensive surface analysis report"""

        if not self.calibrated_models:
            raise ValueError("No models calibrated. Run calibrate_surface() first.")

        report = {
            'timestamp': datetime.now(),
            'quote_count': len(quotes),
            'calibrated_models': list(self.calibrated_models.keys()),
            'surface_analytics': {},
            'risk_metrics': {},
            'arbitrage_status': {}
        }

        # Surface analytics
        for model_name, model in self.calibrated_models.items():
            try:
                # Calculate surface statistics
                vol_matrix = self.current_surface_data.volatilities if self.current_surface_data else None

                if vol_matrix is not None:
                    analytics = {
                        'mean_volatility': float(np.nanmean(vol_matrix)),
                        'vol_of_vol': float(np.nanstd(vol_matrix)),
                        'min_volatility': float(np.nanmin(vol_matrix)),
                        'max_volatility': float(np.nanmax(vol_matrix)),
                        'surface_curvature': self._calculate_surface_curvature(vol_matrix)
                    }
                    report['surface_analytics'][model_name] = analytics

            except Exception as e:
                warnings.warn(f"Failed to calculate analytics for {model_name}: {e}")

        # Arbitrage monitoring results
        if self.arbitrage_monitor:
            violations = self.monitoring_alerts
            report['arbitrage_status'] = {
                'violation_count': len(violations),
                'violations': [
                    {
                        'type': v.violation_type,
                        'severity': v.severity,
                        'description': v.description,
                        'strike': v.strike,
                        'expiry': v.expiry
                    } for v in violations
                ],
                'surface_health': 'HEALTHY' if not violations else 'VIOLATIONS_DETECTED'
            }

        return report

    def _calculate_surface_curvature(self, vol_matrix: np.ndarray) -> float:
        """Calculate approximate surface curvature metric"""
        try:
            # Simple curvature approximation using second derivatives
            grad_x = np.gradient(vol_matrix, axis=1)
            grad_y = np.gradient(vol_matrix, axis=0)
            grad_xx = np.gradient(grad_x, axis=1)
            grad_yy = np.gradient(grad_y, axis=0)

            curvature = np.nanmean(np.abs(grad_xx) + np.abs(grad_yy))
            return float(curvature)
        except:
            return 0.0

    def create_visualization_suite(self, save_directory: Optional[str] = None) -> Dict[str, Any]:
        """Create comprehensive visualization suite"""

        if not self.visualizer or not self.current_surface_data:
            raise ValueError("Visualization not enabled or no surface data available")

        visualizations = {}

        try:
            # 3D Surface plot
            surface_fig = self.visualizer.plot_surface_3d(
                self.current_surface_data,
                title="Calibrated Volatility Surface"
            )
            visualizations['surface_3d'] = surface_fig

            if save_directory:
                surface_fig.savefig(f"{save_directory}/surface_3d.png")

            # Volatility heatmap
            heatmap_fig = self.visualizer.plot_volatility_heatmap(
                self.current_surface_data,
                title="Volatility Surface Heatmap"
            )
            visualizations['heatmap'] = heatmap_fig

            if save_directory:
                heatmap_fig.savefig(f"{save_directory}/heatmap.png")

            # Interactive surface (if Plotly available)
            if self.config.interactive_plots:
                try:
                    interactive_surface = self.visualizer.create_interactive_surface(
                        self.current_surface_data,
                        title="Interactive Volatility Surface"
                    )
                    if interactive_surface:
                        visualizations['interactive_surface'] = interactive_surface
                        if save_directory:
                            interactive_surface.write_html(f"{save_directory}/interactive_surface.html")
                except ImportError:
                    warnings.warn("Plotly not available for interactive visualizations")

        except Exception as e:
            warnings.warn(f"Visualization creation failed: {e}")

        return visualizations

    def run_validation_suite(self) -> Dict[str, Any]:
        """Run comprehensive validation suite"""

        if not self.calibrated_models:
            raise ValueError("No models calibrated for validation")

        validation_results = {}

        # Performance validation
        for model_name, model in self.calibrated_models.items():
            try:
                # Generate test data
                test_quotes = TestDataGenerator.generate_synthetic_quotes(seed=42)

                # Cross-validation
                validator = CrossValidationValidator(n_folds=self.config.cross_validation_folds)
                cv_metrics = validator.validate(model, test_quotes, test_quotes[:len(test_quotes)//2])

                validation_results[model_name] = {
                    'cross_validation': cv_metrics.to_dict(),
                    'model_params': model.get_parameters() if hasattr(model, 'get_parameters') else None
                }

            except Exception as e:
                warnings.warn(f"Validation failed for {model_name}: {e}")
                validation_results[model_name] = {'error': str(e)}

        return validation_results

    def get_model_parameters(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get parameters from calibrated models"""

        if model_name:
            if model_name not in self.calibrated_models:
                raise ValueError(f"Model '{model_name}' not calibrated")

            model = self.calibrated_models[model_name]
            return {
                'model_type': type(model).__name__,
                'parameters': getattr(model, 'slice_parameters', {}),
                'calibration_time': self.last_calibration_time
            }
        else:
            # Return all model parameters
            all_params = {}
            for name, model in self.calibrated_models.items():
                all_params[name] = {
                    'model_type': type(model).__name__,
                    'parameters': getattr(model, 'slice_parameters', {}),
                    'calibration_time': self.last_calibration_time
                }
            return all_params

    def reset_framework(self):
        """Reset framework state"""
        self.calibrated_models.clear()
        self.monitoring_alerts.clear()
        self.current_surface_data = None
        self.last_calibration_time = None


# Convenience functions for common workflows
def quick_surface_calibration(quotes: List[VolatilityQuote],
                            model: str = 'svi') -> VolatilitySurfaceFramework:
    """Quick setup and calibration workflow"""

    config = FrameworkConfig(
        default_model=model,
        calibration_frequency='manual',
        enable_arbitrage_monitoring=True
    )

    framework = VolatilitySurfaceFramework(config)
    framework.calibrate_surface(quotes, model_name=model)

    return framework


def create_production_framework() -> VolatilitySurfaceFramework:
    """Create production-ready framework configuration"""

    config = FrameworkConfig(
        preferred_models=['svi', 'sabr'],
        calibration_frequency='realtime',
        enable_microstructure_modeling=True,
        enable_arbitrage_monitoring=True,
        adaptive_interpolation=True,
        cross_validation_folds=10,
        backtest_lookback_days=60,
        enable_visualization=True,
        interactive_plots=True
    )

    return VolatilitySurfaceFramework(config)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    test_quotes = TestDataGenerator.generate_synthetic_quotes(
        n_strikes=25, n_expiries=10, seed=42
    )

    print("Volatility Surface Framework Demo")
    print("=" * 50)

    # Create and configure framework
    framework = create_production_framework()

    # Calibrate surface with automatic model selection
    print("Calibrating volatility surface...")
    calibration_results = framework.calibrate_surface(
        test_quotes, auto_select=True
    )

    print(f"Selected model: {calibration_results['selected_model']}")
    print(f"Calibration completed at: {calibration_results['calibration_time']}")

    # Generate comprehensive report
    print("\nGenerating surface report...")
    report = framework.generate_surface_report(test_quotes)
    print(f"Surface health: {report['arbitrage_status']['surface_health']}")

    # Example volatility lookup
    sample_vol = framework.get_surface_volatility(
        strike=100.0, expiry=0.25, forward=100.0
    )
    print(f"Sample volatility (K=100, T=0.25): {sample_vol:.4f}")

    # Run validation
    print("\nRunning validation suite...")
    validation_results = framework.run_validation_suite()

    for model_name, results in validation_results.items():
        if 'cross_validation' in results:
            rmse = results['cross_validation']['rmse']
            print(f"{model_name} RMSE: {rmse:.6f}")

    print("\nFramework demonstration completed!")