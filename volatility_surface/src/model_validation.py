"""
Advanced Model Validation and Selection Framework for Volatility Surfaces

This module provides comprehensive validation, comparison, and selection capabilities
for volatility surface models, including statistical tests, performance metrics,
and backtesting frameworks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from datetime import datetime, timedelta
from surface_models import VolatilitySurfaceModel, SVIModel, SABRModel
from calibration_engine import VolatilityQuote


@dataclass
class ValidationMetrics:
 """Container for model validation metrics"""
 rmse: float
 mae: float
 mape: float
 r_squared: float
 max_error: float
 mean_error: float
 std_error: float
 calibration_time: float
 prediction_time: float
 likelihood: Optional[float] = None
 aic: Optional[float] = None
 bic: Optional[float] = None

 def to_dict(self) -> Dict[str, float]:
 return {
 'rmse': self.rmse,
 'mae': self.mae,
 'mape': self.mape,
 'r_squared': self.r_squared,
 'max_error': self.max_error,
 'mean_error': self.mean_error,
 'std_error': self.std_error,
 'calibration_time': self.calibration_time,
 'prediction_time': self.prediction_time,
 'likelihood': self.likelihood,
 'aic': self.aic,
 'bic': self.bic
 }


@dataclass
class BacktestResult:
 """Results from backtesting analysis"""
 start_date: datetime
 end_date: datetime
 total_periods: int
 successful_calibrations: int
 failed_calibrations: int
 average_metrics: ValidationMetrics
 period_metrics: List[ValidationMetrics]
 sharpe_ratio: Optional[float] = None
 max_drawdown: Optional[float] = None
 hit_ratio: Optional[float] = None


@dataclass
class ModelComparisonResult:
 """Results from comparing multiple models"""
 best_model: str
 model_rankings: List[Tuple[str, float]]
 detailed_metrics: Dict[str, ValidationMetrics]
 statistical_tests: Dict[str, Dict[str, float]]
 confidence_intervals: Dict[str, Tuple[float, float]]


class ModelValidator(ABC):
 """Abstract base class for model validation"""

 @abstractmethod
 def validate(self, model: VolatilitySurfaceModel,
 train_data: List[VolatilityQuote],
 test_data: List[VolatilityQuote]) -> ValidationMetrics:
 """Validate a model on test data"""
 pass


class CrossValidationValidator(ModelValidator):
 """Cross-validation based model validator"""

 def __init__(self, n_folds: int = 5, cv_type: str = 'kfold',
 gap_days: int = 0, test_size: float = 0.2):
 self.n_folds = n_folds
 self.cv_type = cv_type
 self.gap_days = gap_days
 self.test_size = test_size

 def validate(self, model: VolatilitySurfaceModel,
 train_data: List[VolatilityQuote],
 test_data: List[VolatilityQuote]) -> ValidationMetrics:
 """Perform cross-validation"""

 if self.cv_type == 'kfold':
 cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
 data_indices = range(len(train_data))
 elif self.cv_type == 'timeseries':
 cv = TimeSeriesSplit(n_splits=self.n_folds, gap=self.gap_days,
 test_size=int(len(train_data) * self.test_size))
 data_indices = range(len(train_data))
 else:
 raise ValueError(f"Unknown CV type: {self.cv_type}")

 fold_metrics = []

 for train_idx, val_idx in cv.split(data_indices):
 train_fold = [train_data[i] for i in train_idx]
 val_fold = [train_data[i] for i in val_idx]

 try:
 start_time = time.time()
 model.calibrate(train_fold)
 calibration_time = time.time() - start_time

 start_time = time.time()
 predictions = self._predict_volatilities(model, val_fold)
 prediction_time = time.time() - start_time

 actuals = [q.implied_vol for q in val_fold]
 metrics = self._calculate_metrics(actuals, predictions,
 calibration_time, prediction_time,
 model)
 fold_metrics.append(metrics)

 except Exception as e:
 warnings.warn(f"Fold failed: {e}")
 continue

 if not fold_metrics:
 raise ValueError("All CV folds failed")

 return self._aggregate_metrics(fold_metrics)

 def _predict_volatilities(self, model: VolatilitySurfaceModel,
 quotes: List[VolatilityQuote]) -> List[float]:
 """Predict volatilities for quotes"""
 predictions = []
 for quote in quotes:
 try:
 log_moneyness = np.log(quote.strike / quote.forward)
 vol = model.calculate_volatility(log_moneyness, quote.expiry)
 predictions.append(vol)
 except:
 predictions.append(np.nan)
 return predictions

 def _calculate_metrics(self, actuals: List[float], predictions: List[float],
 calibration_time: float, prediction_time: float,
 model: VolatilitySurfaceModel) -> ValidationMetrics:
 """Calculate validation metrics"""

 # Remove NaN values
 valid_pairs = [(a, p) for a, p in zip(actuals, predictions)
 if not (np.isnan(a) or np.isnan(p))]

 if not valid_pairs:
 raise ValueError("No valid predictions")

 actuals_clean, predictions_clean = zip(*valid_pairs)
 actuals_clean = np.array(actuals_clean)
 predictions_clean = np.array(predictions_clean)

 errors = predictions_clean - actuals_clean

 rmse = np.sqrt(mean_squared_error(actuals_clean, predictions_clean))
 mae = mean_absolute_error(actuals_clean, predictions_clean)
 mape = np.mean(np.abs(errors / actuals_clean)) * 100
 r_squared = r2_score(actuals_clean, predictions_clean)
 max_error = np.max(np.abs(errors))
 mean_error = np.mean(errors)
 std_error = np.std(errors)

 # Calculate likelihood-based metrics if possible
 likelihood = None
 aic = None
 bic = None

 try:
 if hasattr(model, 'calculate_likelihood'):
 likelihood = model.calculate_likelihood(valid_pairs)
 n_params = len(model.get_parameters())
 n_obs = len(valid_pairs)
 aic = 2 * n_params - 2 * likelihood
 bic = n_params * np.log(n_obs) - 2 * likelihood
 except:
 pass

 return ValidationMetrics(
 rmse=rmse, mae=mae, mape=mape, r_squared=r_squared,
 max_error=max_error, mean_error=mean_error, std_error=std_error,
 calibration_time=calibration_time, prediction_time=prediction_time,
 likelihood=likelihood, aic=aic, bic=bic
 )

 def _aggregate_metrics(self, fold_metrics: List[ValidationMetrics]) -> ValidationMetrics:
 """Aggregate metrics across folds"""

 avg_metrics = {}
 for metric_name in fold_metrics[0].to_dict().keys():
 values = [getattr(m, metric_name) for m in fold_metrics
 if getattr(m, metric_name) is not None]
 if values:
 avg_metrics[metric_name] = np.mean(values)
 else:
 avg_metrics[metric_name] = None

 return ValidationMetrics(**avg_metrics)


class OutOfSampleValidator(ModelValidator):
 """Out-of-sample validation for time series data"""

 def __init__(self, lookback_days: int = 30, forward_days: int = 1):
 self.lookback_days = lookback_days
 self.forward_days = forward_days

 def validate(self, model: VolatilitySurfaceModel,
 train_data: List[VolatilityQuote],
 test_data: List[VolatilityQuote]) -> ValidationMetrics:
 """Perform out-of-sample validation"""

 start_time = time.time()
 model.calibrate(train_data)
 calibration_time = time.time() - start_time

 start_time = time.time()
 predictions = self._predict_volatilities(model, test_data)
 prediction_time = time.time() - start_time

 actuals = [q.implied_vol for q in test_data]

 return self._calculate_metrics(actuals, predictions, calibration_time,
 prediction_time, model)

 def _predict_volatilities(self, model: VolatilitySurfaceModel,
 quotes: List[VolatilityQuote]) -> List[float]:
 """Predict volatilities for quotes"""
 predictions = []
 for quote in quotes:
 try:
 log_moneyness = np.log(quote.strike / quote.forward)
 vol = model.calculate_volatility(log_moneyness, quote.expiry)
 predictions.append(vol)
 except:
 predictions.append(np.nan)
 return predictions

 def _calculate_metrics(self, actuals: List[float], predictions: List[float],
 calibration_time: float, prediction_time: float,
 model: VolatilitySurfaceModel) -> ValidationMetrics:
 """Calculate validation metrics"""

 valid_pairs = [(a, p) for a, p in zip(actuals, predictions)
 if not (np.isnan(a) or np.isnan(p))]

 if not valid_pairs:
 raise ValueError("No valid predictions")

 actuals_clean, predictions_clean = zip(*valid_pairs)
 actuals_clean = np.array(actuals_clean)
 predictions_clean = np.array(predictions_clean)

 errors = predictions_clean - actuals_clean

 rmse = np.sqrt(mean_squared_error(actuals_clean, predictions_clean))
 mae = mean_absolute_error(actuals_clean, predictions_clean)
 mape = np.mean(np.abs(errors / actuals_clean)) * 100
 r_squared = r2_score(actuals_clean, predictions_clean)
 max_error = np.max(np.abs(errors))
 mean_error = np.mean(errors)
 std_error = np.std(errors)

 return ValidationMetrics(
 rmse=rmse, mae=mae, mape=mape, r_squared=r_squared,
 max_error=max_error, mean_error=mean_error, std_error=std_error,
 calibration_time=calibration_time, prediction_time=prediction_time
 )


class ModelBacktester:
 """Backtesting framework for volatility surface models"""

 def __init__(self, rebalance_frequency: str = 'daily',
 lookback_window: int = 30, min_data_points: int = 10):
 self.rebalance_frequency = rebalance_frequency
 self.lookback_window = lookback_window
 self.min_data_points = min_data_points
 self.validator = OutOfSampleValidator()

 def backtest(self, model: VolatilitySurfaceModel,
 historical_data: List[VolatilityQuote],
 start_date: datetime, end_date: datetime) -> BacktestResult:
 """Run backtesting analysis"""

 # Group data by date
 data_by_date = self._group_by_date(historical_data)

 # Generate rebalance dates
 rebalance_dates = self._generate_rebalance_dates(start_date, end_date)

 period_metrics = []
 successful_calibrations = 0
 failed_calibrations = 0

 for i, rebal_date in enumerate(rebalance_dates[:-1]):
 next_date = rebalance_dates[i + 1]

 # Get training data
 train_data = self._get_training_data(data_by_date, rebal_date)

 if len(train_data) < self.min_data_points:
 failed_calibrations += 1
 continue

 # Get test data
 test_data = self._get_test_data(data_by_date, next_date)

 if not test_data:
 continue

 try:
 metrics = self.validator.validate(model, train_data, test_data)
 period_metrics.append(metrics)
 successful_calibrations += 1
 except Exception as e:
 warnings.warn(f"Calibration failed for {rebal_date}: {e}")
 failed_calibrations += 1

 if not period_metrics:
 raise ValueError("All backtesting periods failed")

 average_metrics = self._average_metrics(period_metrics)

 return BacktestResult(
 start_date=start_date,
 end_date=end_date,
 total_periods=len(rebalance_dates) - 1,
 successful_calibrations=successful_calibrations,
 failed_calibrations=failed_calibrations,
 average_metrics=average_metrics,
 period_metrics=period_metrics
 )

 def _group_by_date(self, data: List[VolatilityQuote]) -> Dict[datetime, List[VolatilityQuote]]:
 """Group quotes by date"""
 grouped = {}
 for quote in data:
 date = quote.timestamp.date()
 if date not in grouped:
 grouped[date] = []
 grouped[date].append(quote)
 return grouped

 def _generate_rebalance_dates(self, start_date: datetime,
 end_date: datetime) -> List[datetime]:
 """Generate rebalancing dates"""
 dates = []
 current = start_date

 if self.rebalance_frequency == 'daily':
 delta = timedelta(days=1)
 elif self.rebalance_frequency == 'weekly':
 delta = timedelta(weeks=1)
 elif self.rebalance_frequency == 'monthly':
 delta = timedelta(days=30)
 else:
 raise ValueError(f"Unknown frequency: {self.rebalance_frequency}")

 while current <= end_date:
 dates.append(current)
 current += delta

 return dates

 def _get_training_data(self, data_by_date: Dict[datetime, List[VolatilityQuote]],
 as_of_date: datetime) -> List[VolatilityQuote]:
 """Get training data up to as_of_date"""
 train_data = []
 lookback_date = as_of_date - timedelta(days=self.lookback_window)

 for date, quotes in data_by_date.items():
 if lookback_date <= date <= as_of_date:
 train_data.extend(quotes)

 return train_data

 def _get_test_data(self, data_by_date: Dict[datetime, List[VolatilityQuote]],
 test_date: datetime) -> List[VolatilityQuote]:
 """Get test data for specific date"""
 return data_by_date.get(test_date, [])

 def _average_metrics(self, metrics_list: List[ValidationMetrics]) -> ValidationMetrics:
 """Calculate average metrics"""
 avg_metrics = {}
 for metric_name in metrics_list[0].to_dict().keys():
 values = [getattr(m, metric_name) for m in metrics_list
 if getattr(m, metric_name) is not None]
 if values:
 avg_metrics[metric_name] = np.mean(values)
 else:
 avg_metrics[metric_name] = None

 return ValidationMetrics(**avg_metrics)


class ModelComparator:
 """Framework for comparing multiple volatility surface models"""

 def __init__(self, validator: ModelValidator = None):
 self.validator = validator or CrossValidationValidator()

 def compare_models(self, models: Dict[str, VolatilitySurfaceModel],
 train_data: List[VolatilityQuote],
 test_data: List[VolatilityQuote],
 scoring_metric: str = 'rmse') -> ModelComparisonResult:
 """Compare multiple models"""

 model_metrics = {}
 model_scores = {}

 # Validate each model
 for name, model in models.items():
 try:
 metrics = self.validator.validate(model, train_data, test_data)
 model_metrics[name] = metrics
 model_scores[name] = getattr(metrics, scoring_metric)
 except Exception as e:
 warnings.warn(f"Model {name} validation failed: {e}")
 continue

 if not model_metrics:
 raise ValueError("All model validations failed")

 # Rank models (lower is better for error metrics)
 if scoring_metric in ['rmse', 'mae', 'mape', 'max_error']:
 model_rankings = sorted(model_scores.items(), key=lambda x: x[1])
 else:
 model_rankings = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

 best_model = model_rankings[0][0]

 # Perform statistical tests
 statistical_tests = self._perform_statistical_tests(model_metrics, scoring_metric)

 # Calculate confidence intervals
 confidence_intervals = self._calculate_confidence_intervals(model_metrics, scoring_metric)

 return ModelComparisonResult(
 best_model=best_model,
 model_rankings=model_rankings,
 detailed_metrics=model_metrics,
 statistical_tests=statistical_tests,
 confidence_intervals=confidence_intervals
 )

 def _perform_statistical_tests(self, model_metrics: Dict[str, ValidationMetrics],
 scoring_metric: str) -> Dict[str, Dict[str, float]]:
 """Perform statistical tests between models"""

 tests = {}
 model_names = list(model_metrics.keys())

 for i, model1 in enumerate(model_names):
 for model2 in model_names[i+1:]:
 score1 = getattr(model_metrics[model1], scoring_metric)
 score2 = getattr(model_metrics[model2], scoring_metric)

 # Perform t-test (simplified - in practice would use cross-validation errors)
 try:
 t_stat, p_value = stats.ttest_rel([score1], [score2])
 test_key = f"{model1}_vs_{model2}"
 tests[test_key] = {
 't_statistic': t_stat,
 'p_value': p_value,
 'significant': p_value < 0.05
 }
 except:
 pass

 return tests

 def _calculate_confidence_intervals(self, model_metrics: Dict[str, ValidationMetrics],
 scoring_metric: str, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
 """Calculate confidence intervals for metrics"""

 intervals = {}
 alpha = 1 - confidence

 for name, metrics in model_metrics.items():
 score = getattr(metrics, scoring_metric)
 # Simplified CI calculation - in practice would use bootstrap or CV folds
 stderr = score * 0.1 # Placeholder
 margin = stats.norm.ppf(1 - alpha/2) * stderr
 intervals[name] = (score - margin, score + margin)

 return intervals


class ModelSelectionFramework:
 """Comprehensive framework for automated model selection"""

 def __init__(self, validation_config: Dict[str, Any] = None):
 self.validation_config = validation_config or {
 'cv_folds': 5,
 'scoring_metrics': ['rmse', 'mae', 'r_squared'],
 'primary_metric': 'rmse',
 'confidence_level': 0.95
 }

 self.validator = CrossValidationValidator(
 n_folds=self.validation_config['cv_folds']
 )
 self.comparator = ModelComparator(self.validator)
 self.backtester = ModelBacktester()

 def select_best_model(self, candidate_models: Dict[str, VolatilitySurfaceModel],
 historical_data: List[VolatilityQuote],
 validation_start: datetime = None,
 validation_end: datetime = None) -> Tuple[str, VolatilitySurfaceModel, ModelComparisonResult]:
 """Select the best model using comprehensive validation"""

 # Split data into train/test
 if validation_start and validation_end:
 train_data, test_data = self._split_by_date(historical_data, validation_start)
 else:
 split_idx = int(len(historical_data) * 0.8)
 train_data = historical_data[:split_idx]
 test_data = historical_data[split_idx:]

 # Compare models
 comparison_result = self.comparator.compare_models(
 candidate_models, train_data, test_data,
 self.validation_config['primary_metric']
 )

 best_model_name = comparison_result.best_model
 best_model = candidate_models[best_model_name]

 return best_model_name, best_model, comparison_result

 def _split_by_date(self, data: List[VolatilityQuote],
 split_date: datetime) -> Tuple[List[VolatilityQuote], List[VolatilityQuote]]:
 """Split data by date"""
 train_data = [q for q in data if q.timestamp < split_date]
 test_data = [q for q in data if q.timestamp >= split_date]
 return train_data, test_data


# Example usage functions
def create_validation_suite() -> Dict[str, ModelValidator]:
 """Create a comprehensive validation suite"""
 return {
 'cross_validation': CrossValidationValidator(n_folds=5),
 'time_series_cv': CrossValidationValidator(cv_type='timeseries'),
 'out_of_sample': OutOfSampleValidator(lookback_days=30)
 }


def run_model_selection_pipeline(historical_data: List[VolatilityQuote]) -> Tuple[str, VolatilitySurfaceModel]:
 """Run complete model selection pipeline"""

 # Create candidate models
 candidate_models = {
 'svi': SVIModel(),
 'sabr': SABRModel()
 }

 # Initialize selection framework
 selector = ModelSelectionFramework()

 # Select best model
 best_name, best_model, comparison = selector.select_best_model(
 candidate_models, historical_data
 )

 print(f"Best model: {best_name}")
 print(f"Performance: {comparison.detailed_metrics[best_name].rmse:.4f} RMSE")

 return best_name, best_model