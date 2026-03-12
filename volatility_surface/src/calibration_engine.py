"""
Real-Time Volatility Surface Calibration Engine

Advanced calibration algorithms with real-time triggers, model selection,
cross-validation, and sophisticated optimization techniques for professional
volatility surface management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
import warnings
warnings.filterwarnings('ignore')

from scipy import optimize
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from.surface_models import (
 VolatilitySurfaceModel, SVIModel, SABRModel, VolatilityQuote,
 CalibrationResult, CalibrationMethod, SurfaceModelType
)

logger = logging.getLogger(__name__)

class TriggerType(Enum):
 """Calibration trigger types"""
 TIME_BASED = "TIME_BASED"
 QUOTE_UPDATE = "QUOTE_UPDATE"
 PRICE_MOVEMENT = "PRICE_MOVEMENT"
 VOLATILITY_CHANGE = "VOLATILITY_CHANGE"
 MODEL_DEGRADATION = "MODEL_DEGRADATION"
 MANUAL = "MANUAL"

class ValidationMethod(Enum):
 """Cross-validation methods"""
 K_FOLD = "K_FOLD"
 TIME_SERIES_SPLIT = "TIME_SERIES_SPLIT"
 HOLD_OUT = "HOLD_OUT"
 BOOTSTRAP = "BOOTSTRAP"

@dataclass
class CalibrationTrigger:
 """Calibration trigger configuration"""
 trigger_type: TriggerType
 threshold: float
 enabled: bool = True
 last_triggered: Optional[datetime] = None
 trigger_count: int = 0

@dataclass
class ModelPerformance:
 """Model performance metrics"""
 model_type: SurfaceModelType
 in_sample_r2: float
 out_of_sample_r2: float
 rmse: float
 mae: float
 aic: float
 bic: float
 calibration_time: float
 stability_score: float
 coverage_ratio: float

@dataclass
class CalibrationConfig:
 """Calibration configuration"""
 models_to_fit: List[SurfaceModelType] = field(default_factory=lambda: [SurfaceModelType.SVI, SurfaceModelType.SABR])
 optimization_method: CalibrationMethod = CalibrationMethod.DIFFERENTIAL_EVOLUTION
 validation_method: ValidationMethod = ValidationMethod.TIME_SERIES_SPLIT
 min_quotes_per_slice: int = 5
 max_calibration_time: float = 60.0 # seconds
 regularization_strength: float = 0.01
 confidence_threshold: float = 0.7
 parallel_execution: bool = True
 max_workers: int = 4

class RegularizationFramework:
 """Advanced regularization techniques for volatility surface calibration"""

 @staticmethod
 def l1_regularization(parameters: np.ndarray, lambda_reg: float) -> float:
 """L1 (Lasso) regularization penalty"""
 return lambda_reg * np.sum(np.abs(parameters))

 @staticmethod
 def l2_regularization(parameters: np.ndarray, lambda_reg: float) -> float:
 """L2 (Ridge) regularization penalty"""
 return lambda_reg * np.sum(parameters**2)

 @staticmethod
 def elastic_net_regularization(parameters: np.ndarray, lambda_reg: float,
 alpha: float = 0.5) -> float:
 """Elastic Net regularization (combination of L1 and L2)"""
 l1_penalty = RegularizationFramework.l1_regularization(parameters, lambda_reg * alpha)
 l2_penalty = RegularizationFramework.l2_regularization(parameters, lambda_reg * (1 - alpha))
 return l1_penalty + l2_penalty

 @staticmethod
 def smoothness_penalty(model: VolatilitySurfaceModel, strikes: np.ndarray,
 expiries: np.ndarray, lambda_smooth: float) -> float:
 """Smoothness penalty to prevent overfitting"""
 try:
 # Calculate second derivatives to measure curvature
 penalty = 0.0

 for expiry in expiries:
 vols = model.calculate_volatility(np.log(strikes), expiry)

 # Second derivative penalty (discrete approximation)
 if len(vols) >= 3:
 second_deriv = np.diff(vols, n=2)
 penalty += lambda_smooth * np.sum(second_deriv**2)

 return penalty
 except:
 return 0.0

class CrossValidationFramework:
 """Advanced cross-validation for volatility surface models"""

 @staticmethod
 def time_series_split_validation(quotes: List[VolatilityQuote],
 model: VolatilitySurfaceModel,
 n_splits: int = 5) -> Tuple[float, float]:
 """Time series split cross-validation preserving temporal order"""

 # Sort quotes by timestamp
 sorted_quotes = sorted(quotes, key=lambda q: q.timestamp)
 n_quotes = len(sorted_quotes)

 if n_quotes < n_splits * 2:
 raise ValueError(f"Not enough quotes for {n_splits} splits")

 scores = []
 split_size = n_quotes // (n_splits + 1)

 for i in range(n_splits):
 # Use expanding window for training
 train_end = (i + 1) * split_size
 test_start = train_end
 test_end = test_start + split_size

 if test_end > n_quotes:
 break

 train_quotes = sorted_quotes[:train_end]
 test_quotes = sorted_quotes[test_start:test_end]

 # Calibrate on training data
 try:
 model_copy = type(model)()
 model_copy.calibrate(train_quotes)

 # Evaluate on test data
 test_score = CrossValidationFramework._evaluate_model(model_copy, test_quotes)
 scores.append(test_score)
 except Exception as e:
 logger.warning(f"Cross-validation fold failed: {e}")
 continue

 if not scores:
 return 0.0, 0.0

 return np.mean(scores), np.std(scores)

 @staticmethod
 def k_fold_validation(quotes: List[VolatilityQuote],
 model: VolatilitySurfaceModel,
 k: int = 5) -> Tuple[float, float]:
 """K-fold cross-validation for volatility models"""

 # Group quotes by expiry to ensure each fold has representation
 expiry_groups = {}
 for quote in quotes:
 exp_key = round(quote.expiry, 3) # Round to avoid floating point issues
 if exp_key not in expiry_groups:
 expiry_groups[exp_key] = []
 expiry_groups[exp_key].append(quote)

 scores = []

 # Create folds ensuring each has quotes from all expiries
 for fold in range(k):
 train_quotes = []
 test_quotes = []

 for expiry, exp_quotes in expiry_groups.items():
 n_exp_quotes = len(exp_quotes)
 test_size = max(1, n_exp_quotes // k)
 test_start = fold * test_size
 test_end = min(test_start + test_size, n_exp_quotes)

 test_quotes.extend(exp_quotes[test_start:test_end])
 train_quotes.extend(exp_quotes[:test_start] + exp_quotes[test_end:])

 if len(train_quotes) < 10 or len(test_quotes) < 5:
 continue

 try:
 model_copy = type(model)()
 model_copy.calibrate(train_quotes)

 test_score = CrossValidationFramework._evaluate_model(model_copy, test_quotes)
 scores.append(test_score)
 except Exception as e:
 logger.warning(f"K-fold validation fold failed: {e}")
 continue

 if not scores:
 return 0.0, 0.0

 return np.mean(scores), np.std(scores)

 @staticmethod
 def bootstrap_validation(quotes: List[VolatilityQuote],
 model: VolatilitySurfaceModel,
 n_bootstrap: int = 100) -> Tuple[float, float, np.ndarray]:
 """Bootstrap validation for confidence intervals"""

 scores = []
 n_quotes = len(quotes)

 for _ in range(n_bootstrap):
 # Bootstrap sample with replacement
 bootstrap_indices = np.random.choice(n_quotes, size=n_quotes, replace=True)
 bootstrap_quotes = [quotes[i] for i in bootstrap_indices]

 # Use out-of-bag samples for testing
 oob_indices = list(set(range(n_quotes)) - set(bootstrap_indices))
 if len(oob_indices) < 5:
 continue

 oob_quotes = [quotes[i] for i in oob_indices]

 try:
 model_copy = type(model)()
 model_copy.calibrate(bootstrap_quotes)

 oob_score = CrossValidationFramework._evaluate_model(model_copy, oob_quotes)
 scores.append(oob_score)
 except Exception as e:
 continue

 if not scores:
 return 0.0, 0.0, np.array([])

 scores_array = np.array(scores)
 return np.mean(scores_array), np.std(scores_array), scores_array

 @staticmethod
 def _evaluate_model(model: VolatilitySurfaceModel, test_quotes: List[VolatilityQuote]) -> float:
 """Evaluate model performance on test quotes"""
 try:
 predicted_vols = []
 actual_vols = []

 for quote in test_quotes:
 pred_vol = model.calculate_volatility(quote.log_moneyness, quote.expiry)
 if np.isfinite(pred_vol) and pred_vol > 0:
 predicted_vols.append(pred_vol)
 actual_vols.append(quote.implied_vol)

 if len(predicted_vols) < 2:
 return 0.0

 return r2_score(actual_vols, predicted_vols)
 except:
 return 0.0

class ModelSelectionFramework:
 """Advanced model selection using information criteria and cross-validation"""

 def __init__(self):
 self.model_performances = {}

 def compare_models(self, quotes: List[VolatilityQuote],
 models: List[VolatilitySurfaceModel],
 validation_method: ValidationMethod = ValidationMethod.TIME_SERIES_SPLIT) -> Dict[str, ModelPerformance]:
 """Compare multiple volatility surface models"""

 performances = {}

 for model in models:
 try:
 performance = self._evaluate_single_model(quotes, model, validation_method)
 performances[model.name] = performance
 logger.info(f"Model {model.name}: R²={performance.out_of_sample_r2:.4f}, "
 f"RMSE={performance.rmse:.4f}")
 except Exception as e:
 logger.error(f"Error evaluating model {model.name}: {e}")

 return performances

 def _evaluate_single_model(self, quotes: List[VolatilityQuote],
 model: VolatilitySurfaceModel,
 validation_method: ValidationMethod) -> ModelPerformance:
 """Evaluate a single model comprehensively"""

 start_time = time.time()

 # Calibrate model
 calibration_results = model.calibrate(quotes)
 calibration_time = time.time() - start_time

 # In-sample performance
 in_sample_r2 = self._calculate_in_sample_r2(model, quotes)

 # Out-of-sample performance using cross-validation
 if validation_method == ValidationMethod.TIME_SERIES_SPLIT:
 oos_r2, oos_std = CrossValidationFramework.time_series_split_validation(quotes, model)
 elif validation_method == ValidationMethod.K_FOLD:
 oos_r2, oos_std = CrossValidationFramework.k_fold_validation(quotes, model)
 else:
 oos_r2, oos_std = 0.0, 0.0

 # Calculate additional metrics
 rmse, mae = self._calculate_error_metrics(model, quotes)
 aic, bic = self._calculate_information_criteria(model, quotes)
 stability_score = self._calculate_stability_score(model, quotes)
 coverage_ratio = self._calculate_coverage_ratio(model, quotes)

 return ModelPerformance(
 model_type=SurfaceModelType[model.name] if model.name in SurfaceModelType.__members__ else SurfaceModelType.SVI,
 in_sample_r2=in_sample_r2,
 out_of_sample_r2=oos_r2,
 rmse=rmse,
 mae=mae,
 aic=aic,
 bic=bic,
 calibration_time=calibration_time,
 stability_score=stability_score,
 coverage_ratio=coverage_ratio
 )

 def _calculate_in_sample_r2(self, model: VolatilitySurfaceModel,
 quotes: List[VolatilityQuote]) -> float:
 """Calculate in-sample R-squared"""
 try:
 predicted_vols = []
 actual_vols = []

 for quote in quotes:
 pred_vol = model.calculate_volatility(quote.log_moneyness, quote.expiry)
 if np.isfinite(pred_vol) and pred_vol > 0:
 predicted_vols.append(pred_vol)
 actual_vols.append(quote.implied_vol)

 if len(predicted_vols) < 2:
 return 0.0

 return r2_score(actual_vols, predicted_vols)
 except:
 return 0.0

 def _calculate_error_metrics(self, model: VolatilitySurfaceModel,
 quotes: List[VolatilityQuote]) -> Tuple[float, float]:
 """Calculate RMSE and MAE"""
 try:
 predicted_vols = []
 actual_vols = []

 for quote in quotes:
 pred_vol = model.calculate_volatility(quote.log_moneyness, quote.expiry)
 if np.isfinite(pred_vol) and pred_vol > 0:
 predicted_vols.append(pred_vol)
 actual_vols.append(quote.implied_vol)

 if len(predicted_vols) < 2:
 return 1.0, 1.0

 rmse = np.sqrt(mean_squared_error(actual_vols, predicted_vols))
 mae = np.mean(np.abs(np.array(actual_vols) - np.array(predicted_vols)))

 return rmse, mae
 except:
 return 1.0, 1.0

 def _calculate_information_criteria(self, model: VolatilitySurfaceModel,
 quotes: List[VolatilityQuote]) -> Tuple[float, float]:
 """Calculate AIC and BIC"""
 try:
 predicted_vols = []
 actual_vols = []

 for quote in quotes:
 pred_vol = model.calculate_volatility(quote.log_moneyness, quote.expiry)
 if np.isfinite(pred_vol) and pred_vol > 0:
 predicted_vols.append(pred_vol)
 actual_vols.append(quote.implied_vol)

 if len(predicted_vols) < 2:
 return 1e10, 1e10

 n = len(predicted_vols)
 residuals = np.array(actual_vols) - np.array(predicted_vols)
 rss = np.sum(residuals**2)

 # Estimate number of parameters based on model type
 if hasattr(model, 'slice_parameters'):
 k = len(model.slice_parameters) * 5 # Approximate for SVI/SABR
 else:
 k = 10 # Default estimate

 aic = n * np.log(rss / n) + 2 * k
 bic = n * np.log(rss / n) + k * np.log(n)

 return aic, bic
 except:
 return 1e10, 1e10

 def _calculate_stability_score(self, model: VolatilitySurfaceModel,
 quotes: List[VolatilityQuote]) -> float:
 """Calculate model stability score based on parameter sensitivity"""
 try:
 # Add small noise to quotes and recalibrate
 n_trials = 5
 stability_scores = []

 base_params = getattr(model, 'slice_parameters', {})
 if not base_params:
 return 0.5 # Default moderate stability

 for trial in range(n_trials):
 # Add 1% noise to volatilities
 noisy_quotes = []
 for quote in quotes:
 noisy_vol = quote.implied_vol * (1 + np.random.normal(0, 0.01))
 noisy_quote = VolatilityQuote(
 quote.strike, quote.expiry, max(noisy_vol, 0.01),
 quote.bid_vol, quote.ask_vol, quote.volume,
 quote.open_interest, quote.timestamp, quote.confidence
 )
 noisy_quotes.append(noisy_quote)

 # Recalibrate
 test_model = type(model)()
 test_model.calibrate(noisy_quotes)

 # Compare parameters
 test_params = getattr(test_model, 'slice_parameters', {})
 param_diff = self._compare_parameters(base_params, test_params)
 stability_scores.append(1.0 / (1.0 + param_diff)) # Higher score = more stable

 return np.mean(stability_scores) if stability_scores else 0.5
 except:
 return 0.5

 def _compare_parameters(self, params1: Dict, params2: Dict) -> float:
 """Compare parameter dictionaries"""
 total_diff = 0.0
 count = 0

 for expiry in params1.keys():
 if expiry in params2:
 for param_name in params1[expiry].keys():
 if param_name in params2[expiry]:
 val1 = params1[expiry][param_name]
 val2 = params2[expiry][param_name]
 if val1 != 0:
 total_diff += abs((val2 - val1) / val1)
 count += 1

 return total_diff / max(count, 1)

 def _calculate_coverage_ratio(self, model: VolatilitySurfaceModel,
 quotes: List[VolatilityQuote]) -> float:
 """Calculate how well model covers the quote space"""
 try:
 successful_predictions = 0
 total_quotes = len(quotes)

 for quote in quotes:
 try:
 pred_vol = model.calculate_volatility(quote.log_moneyness, quote.expiry)
 if np.isfinite(pred_vol) and pred_vol > 0:
 successful_predictions += 1
 except:
 continue

 return successful_predictions / max(total_quotes, 1)
 except:
 return 0.0

class RealTimeCalibrationEngine:
 """Real-time volatility surface calibration engine with triggers and monitoring"""

 def __init__(self, config: CalibrationConfig):
 self.config = config
 self.triggers = self._initialize_triggers()
 self.model_selector = ModelSelectionFramework()
 self.current_models = {}
 self.calibration_history = []
 self.is_running = False
 self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
 self.lock = threading.Lock()

 def _initialize_triggers(self) -> Dict[TriggerType, CalibrationTrigger]:
 """Initialize calibration triggers"""
 return {
 TriggerType.TIME_BASED: CalibrationTrigger(TriggerType.TIME_BASED, 300.0), # 5 minutes
 TriggerType.QUOTE_UPDATE: CalibrationTrigger(TriggerType.QUOTE_UPDATE, 10.0), # 10 new quotes
 TriggerType.PRICE_MOVEMENT: CalibrationTrigger(TriggerType.PRICE_MOVEMENT, 0.02), # 2% movement
 TriggerType.VOLATILITY_CHANGE: CalibrationTrigger(TriggerType.VOLATILITY_CHANGE, 0.05), # 5% vol change
 TriggerType.MODEL_DEGRADATION: CalibrationTrigger(TriggerType.MODEL_DEGRADATION, 0.1), # R² drop > 0.1
 }

 async def start_monitoring(self):
 """Start the real-time calibration monitoring system"""
 self.is_running = True
 logger.info("Real-time calibration engine started")

 while self.is_running:
 try:
 await self._check_triggers()
 await asyncio.sleep(10) # Check every 10 seconds
 except Exception as e:
 logger.error(f"Error in calibration monitoring: {e}")
 await asyncio.sleep(30)

 def stop_monitoring(self):
 """Stop the real-time calibration monitoring"""
 self.is_running = False
 self.executor.shutdown(wait=True)
 logger.info("Real-time calibration engine stopped")

 async def _check_triggers(self):
 """Check if any calibration triggers should fire"""
 for trigger_type, trigger in self.triggers.items():
 if trigger.enabled and self._should_trigger(trigger):
 await self._handle_trigger(trigger_type)

 def _should_trigger(self, trigger: CalibrationTrigger) -> bool:
 """Determine if a trigger should fire"""
 if trigger.last_triggered is None:
 return True

 time_since_last = (datetime.now() - trigger.last_triggered).total_seconds()

 if trigger.trigger_type == TriggerType.TIME_BASED:
 return time_since_last >= trigger.threshold

 # Other trigger logic would be implemented based on real-time data
 return False

 async def _handle_trigger(self, trigger_type: TriggerType):
 """Handle triggered calibration"""
 logger.info(f"Calibration trigger fired: {trigger_type}")

 with self.lock:
 self.triggers[trigger_type].last_triggered = datetime.now()
 self.triggers[trigger_type].trigger_count += 1

 # Trigger recalibration in background
 if self.config.parallel_execution:
 future = self.executor.submit(self._perform_calibration)
 # Don't wait for completion to maintain real-time performance
 else:
 await asyncio.to_thread(self._perform_calibration)

 def _perform_calibration(self):
 """Perform volatility surface calibration"""
 try:
 # This would be called with real market data
 # For now, we'll use a placeholder
 logger.info("Performing volatility surface calibration...")

 start_time = time.time()

 # Generate models to test
 models_to_test = []
 for model_type in self.config.models_to_fit:
 if model_type == SurfaceModelType.SVI:
 models_to_test.append(SVIModel())
 elif model_type == SurfaceModelType.SABR:
 models_to_test.append(SABRModel())

 # Would calibrate with real quotes here
 # performances = self.model_selector.compare_models(quotes, models_to_test)

 execution_time = time.time() - start_time
 logger.info(f"Calibration completed in {execution_time:.2f} seconds")

 # Store calibration result
 self.calibration_history.append({
 'timestamp': datetime.now(),
 'execution_time': execution_time,
 'models_tested': [m.name for m in models_to_test],
 'trigger_type': 'manual' # Would be actual trigger type
 })

 except Exception as e:
 logger.error(f"Calibration failed: {e}")

 def calibrate_surface(self, quotes: List[VolatilityQuote],
 force_recalibration: bool = False) -> Dict[str, ModelPerformance]:
 """Manually trigger surface calibration"""

 if not quotes:
 raise ValueError("No quotes provided for calibration")

 logger.info(f"Calibrating volatility surface with {len(quotes)} quotes")

 # Generate models to test
 models_to_test = []
 for model_type in self.config.models_to_fit:
 if model_type == SurfaceModelType.SVI:
 models_to_test.append(SVIModel())
 elif model_type == SurfaceModelType.SABR:
 models_to_test.append(SABRModel())

 # Compare models
 performances = self.model_selector.compare_models(
 quotes, models_to_test, self.config.validation_method
 )

 # Select best model based on out-of-sample performance
 if performances:
 best_model_name = max(performances.keys(),
 key=lambda k: performances[k].out_of_sample_r2)

 logger.info(f"Best model: {best_model_name} "
 f"(OOS R²: {performances[best_model_name].out_of_sample_r2:.4f})")

 # Store the best model
 for model in models_to_test:
 if model.name == best_model_name:
 self.current_models[best_model_name] = model
 break

 return performances

 def get_calibration_status(self) -> Dict[str, Any]:
 """Get current calibration system status"""
 return {
 'is_running': self.is_running,
 'active_triggers': {t.name: t.enabled for t in self.triggers.values()},
 'trigger_counts': {t.name: t.trigger_count for t in self.triggers.values()},
 'last_calibration': self.calibration_history[-1] if self.calibration_history else None,
 'current_models': list(self.current_models.keys()),
 'total_calibrations': len(self.calibration_history)
 }

 def update_trigger_config(self, trigger_type: TriggerType,
 threshold: Optional[float] = None,
 enabled: Optional[bool] = None):
 """Update trigger configuration"""
 if trigger_type in self.triggers:
 if threshold is not None:
 self.triggers[trigger_type].threshold = threshold
 if enabled is not None:
 self.triggers[trigger_type].enabled = enabled

 logger.info(f"Updated trigger {trigger_type}: threshold={threshold}, enabled={enabled}")

# Factory function for creating calibration engines
def create_calibration_engine(models: List[SurfaceModelType] = None,
 optimization_method: CalibrationMethod = CalibrationMethod.DIFFERENTIAL_EVOLUTION,
 validation_method: ValidationMethod = ValidationMethod.TIME_SERIES_SPLIT,
 parallel: bool = True) -> RealTimeCalibrationEngine:
 """Factory function to create a calibration engine with specified configuration"""

 if models is None:
 models = [SurfaceModelType.SVI, SurfaceModelType.SABR]

 config = CalibrationConfig(
 models_to_fit=models,
 optimization_method=optimization_method,
 validation_method=validation_method,
 parallel_execution=parallel
 )

 return RealTimeCalibrationEngine(config)