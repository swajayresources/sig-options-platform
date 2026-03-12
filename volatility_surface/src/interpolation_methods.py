"""
Advanced Interpolation and Extrapolation Methods for Volatility Surfaces

Sophisticated interpolation techniques including cubic splines, radial basis functions,
Kriging, thin-plate splines, and model-free methods for professional volatility
surface construction and Greeks calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from scipy import interpolate, optimize, spatial
from scipy.linalg import solve, LinAlgError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from.surface_models import VolatilityQuote

logger = logging.getLogger(__name__)

class InterpolationMethod(Enum):
 """Available interpolation methods"""
 BILINEAR = "BILINEAR"
 BICUBIC = "BICUBIC"
 CUBIC_SPLINE = "CUBIC_SPLINE"
 RBF_GAUSSIAN = "RBF_GAUSSIAN"
 RBF_MULTIQUADRIC = "RBF_MULTIQUADRIC"
 RBF_INVERSE_MULTIQUADRIC = "RBF_INVERSE_MULTIQUADRIC"
 THIN_PLATE_SPLINE = "THIN_PLATE_SPLINE"
 KRIGING = "KRIGING"
 TENSION_SPLINE = "TENSION_SPLINE"
 NATURAL_NEIGHBOR = "NATURAL_NEIGHBOR"

class ExtrapolationBehavior(Enum):
 """Extrapolation behavior options"""
 CONSTANT = "CONSTANT"
 LINEAR = "LINEAR"
 NATURAL = "NATURAL"
 BOUNDED = "BOUNDED"
 MODEL_BASED = "MODEL_BASED"

@dataclass
class InterpolationResult:
 """Result of volatility interpolation"""
 volatility: Union[float, np.ndarray]
 confidence: Union[float, np.ndarray]
 gradient: Optional[np.ndarray] = None
 hessian: Optional[np.ndarray] = None
 method_used: Optional[str] = None
 extrapolated: bool = False

class VolatilityInterpolator(ABC):
 """Abstract base class for volatility interpolators"""

 def __init__(self, name: str):
 self.name = name
 self.is_fitted = False
 self.quotes = []
 self.bounds = None

 @abstractmethod
 def fit(self, quotes: List[VolatilityQuote]) -> bool:
 """Fit the interpolator to volatility quotes"""
 pass

 @abstractmethod
 def interpolate(self, log_moneyness: Union[float, np.ndarray],
 time_to_expiry: Union[float, np.ndarray]) -> InterpolationResult:
 """Interpolate volatility at given points"""
 pass

 @abstractmethod
 def gradient(self, log_moneyness: float, time_to_expiry: float) -> np.ndarray:
 """Calculate gradient (partial derivatives) at given point"""
 pass

 def _validate_input(self, log_moneyness: Union[float, np.ndarray],
 time_to_expiry: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
 """Validate and prepare input arrays"""
 k = np.atleast_1d(log_moneyness)
 T = np.atleast_1d(time_to_expiry)

 if len(k) != len(T) and len(k) > 1 and len(T) > 1:
 raise ValueError("log_moneyness and time_to_expiry must have same length or one must be scalar")

 return k, T

 def _check_bounds(self, log_moneyness: np.ndarray, time_to_expiry: np.ndarray) -> np.ndarray:
 """Check if points are within interpolation bounds"""
 if self.bounds is None:
 return np.zeros(len(log_moneyness), dtype=bool)

 k_min, k_max, T_min, T_max = self.bounds

 out_of_bounds = ((log_moneyness < k_min) | (log_moneyness > k_max) |
 (time_to_expiry < T_min) | (time_to_expiry > T_max))

 return out_of_bounds

class CubicSplineInterpolator(VolatilityInterpolator):
 """Cubic spline interpolation across strikes and maturities"""

 def __init__(self, smoothing_factor: float = 0.0, boundary_condition: str = 'natural'):
 super().__init__("CubicSpline")
 self.smoothing_factor = smoothing_factor
 self.boundary_condition = boundary_condition
 self.spline_functions = {}

 def fit(self, quotes: List[VolatilityQuote]) -> bool:
 """Fit cubic splines for each maturity slice"""
 try:
 self.quotes = quotes
 if len(quotes) < 4:
 logger.warning("Need at least 4 quotes for cubic spline interpolation")
 return False

 # Group quotes by expiry
 expiry_groups = {}
 for quote in quotes:
 expiry_key = round(quote.expiry, 4) # Round to avoid floating point issues
 if expiry_key not in expiry_groups:
 expiry_groups[expiry_key] = []
 expiry_groups[expiry_key].append(quote)

 # Fit spline for each expiry with sufficient quotes
 for expiry, exp_quotes in expiry_groups.items():
 if len(exp_quotes) >= 3: # Minimum for cubic spline
 # Sort by strike
 sorted_quotes = sorted(exp_quotes, key=lambda q: q.strike)

 strikes = np.array([q.strike for q in sorted_quotes])
 log_moneyness = np.array([q.log_moneyness for q in sorted_quotes])
 volatilities = np.array([q.implied_vol for q in sorted_quotes])

 # Remove duplicates
 unique_mask = np.diff(np.concatenate(([False], log_moneyness[1:] == log_moneyness[:-1]))) == 0
 if np.any(~unique_mask):
 log_moneyness = log_moneyness[unique_mask]
 volatilities = volatilities[unique_mask]

 if len(log_moneyness) >= 3:
 # Fit cubic spline
 if self.boundary_condition == 'natural':
 bc_type = 'natural'
 elif self.boundary_condition == 'clamped':
 bc_type = 'clamped'
 else:
 bc_type = 'not-a-knot'

 spline = interpolate.CubicSpline(
 log_moneyness, volatilities,
 bc_type=bc_type,
 extrapolate=False
 )

 self.spline_functions[expiry] = spline

 # Set bounds
 if quotes:
 all_k = [q.log_moneyness for q in quotes]
 all_T = [q.expiry for q in quotes]
 self.bounds = (min(all_k), max(all_k), min(all_T), max(all_T))

 self.is_fitted = len(self.spline_functions) > 0
 return self.is_fitted

 except Exception as e:
 logger.error(f"Error fitting cubic spline: {e}")
 return False

 def interpolate(self, log_moneyness: Union[float, np.ndarray],
 time_to_expiry: Union[float, np.ndarray]) -> InterpolationResult:
 """Interpolate volatility using cubic splines"""
 if not self.is_fitted:
 raise ValueError("Interpolator not fitted")

 k, T = self._validate_input(log_moneyness, time_to_expiry)
 out_of_bounds = self._check_bounds(k, T)

 result_vols = np.zeros_like(k)
 confidences = np.zeros_like(k)

 for i, (ki, Ti) in enumerate(zip(k, T)):
 if out_of_bounds[i]:
 result_vols[i] = self._extrapolate(ki, Ti)
 confidences[i] = 0.5 # Lower confidence for extrapolation
 else:
 result_vols[i] = self._interpolate_single_point(ki, Ti)
 confidences[i] = 1.0

 return InterpolationResult(
 volatility=result_vols.squeeze(),
 confidence=confidences.squeeze(),
 method_used=self.name,
 extrapolated=np.any(out_of_bounds)
 )

 def _interpolate_single_point(self, log_moneyness: float, time_to_expiry: float) -> float:
 """Interpolate single point"""
 available_expiries = sorted(self.spline_functions.keys())

 if not available_expiries:
 return 0.2 # Default volatility

 # Find surrounding expiries
 if time_to_expiry <= available_expiries[0]:
 # Use shortest expiry
 spline = self.spline_functions[available_expiries[0]]
 return float(spline(log_moneyness))

 elif time_to_expiry >= available_expiries[-1]:
 # Use longest expiry
 spline = self.spline_functions[available_expiries[-1]]
 return float(spline(log_moneyness))

 else:
 # Interpolate between expiries
 for i in range(len(available_expiries) - 1):
 T1, T2 = available_expiries[i], available_expiries[i + 1]
 if T1 <= time_to_expiry <= T2:
 spline1 = self.spline_functions[T1]
 spline2 = self.spline_functions[T2]

 vol1 = float(spline1(log_moneyness))
 vol2 = float(spline2(log_moneyness))

 # Linear interpolation in time
 weight = (time_to_expiry - T1) / (T2 - T1)
 return vol1 * (1 - weight) + vol2 * weight

 return 0.2 # Fallback

 def _extrapolate(self, log_moneyness: float, time_to_expiry: float) -> float:
 """Extrapolate beyond fitted region"""
 available_expiries = sorted(self.spline_functions.keys())

 if not available_expiries:
 return 0.2

 if time_to_expiry < available_expiries[0]:
 # Extrapolate to shorter expiry
 spline = self.spline_functions[available_expiries[0]]
 return max(float(spline(log_moneyness)), 0.01)
 else:
 # Extrapolate to longer expiry
 spline = self.spline_functions[available_expiries[-1]]
 return max(float(spline(log_moneyness)), 0.01)

 def gradient(self, log_moneyness: float, time_to_expiry: float) -> np.ndarray:
 """Calculate gradient using finite differences"""
 h = 1e-6

 # Partial derivative w.r.t. log-moneyness
 vol_plus_k = self._interpolate_single_point(log_moneyness + h, time_to_expiry)
 vol_minus_k = self._interpolate_single_point(log_moneyness - h, time_to_expiry)
 dvol_dk = (vol_plus_k - vol_minus_k) / (2 * h)

 # Partial derivative w.r.t. time to expiry
 vol_plus_T = self._interpolate_single_point(log_moneyness, time_to_expiry + h)
 vol_minus_T = self._interpolate_single_point(log_moneyness, time_to_expiry - h)
 dvol_dT = (vol_plus_T - vol_minus_T) / (2 * h)

 return np.array([dvol_dk, dvol_dT])

class RBFInterpolator(VolatilityInterpolator):
 """Radial Basis Function interpolation"""

 def __init__(self, function: str = 'gaussian', smoothing: float = 0.0):
 super().__init__(f"RBF_{function}")
 self.function = function
 self.smoothing = smoothing
 self.rbf = None
 self.scaler_k = StandardScaler()
 self.scaler_T = StandardScaler()

 def fit(self, quotes: List[VolatilityQuote]) -> bool:
 """Fit RBF interpolator"""
 try:
 self.quotes = quotes
 if len(quotes) < 3:
 logger.warning("Need at least 3 quotes for RBF interpolation")
 return False

 # Prepare data
 log_moneyness = np.array([q.log_moneyness for q in quotes])
 time_to_expiry = np.array([q.expiry for q in quotes])
 volatilities = np.array([q.implied_vol for q in quotes])

 # Scale inputs for better numerical stability
 log_moneyness_scaled = self.scaler_k.fit_transform(log_moneyness.reshape(-1, 1)).flatten()
 time_to_expiry_scaled = self.scaler_T.fit_transform(time_to_expiry.reshape(-1, 1)).flatten()

 # Create coordinate arrays
 coords = np.column_stack([log_moneyness_scaled, time_to_expiry_scaled])

 # Fit RBF
 self.rbf = interpolate.RBFInterpolator(
 coords, volatilities,
 kernel=self.function,
 smoothing=self.smoothing
 )

 # Set bounds
 self.bounds = (log_moneyness.min(), log_moneyness.max(),
 time_to_expiry.min(), time_to_expiry.max())

 self.is_fitted = True
 return True

 except Exception as e:
 logger.error(f"Error fitting RBF interpolator: {e}")
 return False

 def interpolate(self, log_moneyness: Union[float, np.ndarray],
 time_to_expiry: Union[float, np.ndarray]) -> InterpolationResult:
 """Interpolate using RBF"""
 if not self.is_fitted:
 raise ValueError("Interpolator not fitted")

 k, T = self._validate_input(log_moneyness, time_to_expiry)
 out_of_bounds = self._check_bounds(k, T)

 # Scale inputs
 k_scaled = self.scaler_k.transform(k.reshape(-1, 1)).flatten()
 T_scaled = self.scaler_T.transform(T.reshape(-1, 1)).flatten()

 coords = np.column_stack([k_scaled, T_scaled])

 try:
 result_vols = self.rbf(coords)
 result_vols = np.maximum(result_vols, 0.01) # Ensure positive volatility

 confidences = np.where(out_of_bounds, 0.5, 1.0)

 return InterpolationResult(
 volatility=result_vols.squeeze(),
 confidence=confidences.squeeze(),
 method_used=self.name,
 extrapolated=np.any(out_of_bounds)
 )

 except Exception as e:
 logger.error(f"RBF interpolation failed: {e}")
 return InterpolationResult(
 volatility=np.full_like(k, 0.2),
 confidence=np.full_like(k, 0.1),
 method_used=self.name,
 extrapolated=True
 )

 def gradient(self, log_moneyness: float, time_to_expiry: float) -> np.ndarray:
 """Calculate gradient using finite differences"""
 if not self.is_fitted:
 return np.array([0.0, 0.0])

 h = 1e-6

 # Scale inputs
 k_scaled = self.scaler_k.transform([[log_moneyness]])[0, 0]
 T_scaled = self.scaler_T.transform([[time_to_expiry]])[0, 0]

 # Finite difference approximation
 coords_base = np.array([[k_scaled, T_scaled]])
 coords_k_plus = np.array([[k_scaled + h, T_scaled]])
 coords_k_minus = np.array([[k_scaled - h, T_scaled]])
 coords_T_plus = np.array([[k_scaled, T_scaled + h]])
 coords_T_minus = np.array([[k_scaled, T_scaled - h]])

 try:
 vol_base = self.rbf(coords_base)[0]
 vol_k_plus = self.rbf(coords_k_plus)[0]
 vol_k_minus = self.rbf(coords_k_minus)[0]
 vol_T_plus = self.rbf(coords_T_plus)[0]
 vol_T_minus = self.rbf(coords_T_minus)[0]

 # Account for scaling
 k_scale = self.scaler_k.scale_[0]
 T_scale = self.scaler_T.scale_[0]

 dvol_dk = (vol_k_plus - vol_k_minus) / (2 * h) * k_scale
 dvol_dT = (vol_T_plus - vol_T_minus) / (2 * h) * T_scale

 return np.array([dvol_dk, dvol_dT])

 except Exception:
 return np.array([0.0, 0.0])

class KrigingInterpolator(VolatilityInterpolator):
 """Kriging (Gaussian Process) interpolation with uncertainty quantification"""

 def __init__(self, kernel_type: str = 'matern', length_scale: float = 1.0,
 nugget: float = 1e-6):
 super().__init__("Kriging")
 self.kernel_type = kernel_type
 self.length_scale = length_scale
 self.nugget = nugget
 self.gp = None
 self.scaler_X = StandardScaler()
 self.scaler_y = StandardScaler()

 def fit(self, quotes: List[VolatilityQuote]) -> bool:
 """Fit Gaussian Process"""
 try:
 self.quotes = quotes
 if len(quotes) < 5:
 logger.warning("Need at least 5 quotes for Kriging interpolation")
 return False

 # Prepare data
 X = np.array([[q.log_moneyness, q.expiry] for q in quotes])
 y = np.array([q.implied_vol for q in quotes])

 # Scale features and targets
 X_scaled = self.scaler_X.fit_transform(X)
 y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

 # Define kernel
 if self.kernel_type == 'rbf':
 kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=self.nugget)
 elif self.kernel_type == 'matern':
 kernel = ConstantKernel(1.0) * Matern(length_scale=self.length_scale, nu=2.5) + WhiteKernel(noise_level=self.nugget)
 else:
 kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=self.nugget)

 # Fit Gaussian Process
 self.gp = GaussianProcessRegressor(
 kernel=kernel,
 alpha=1e-6,
 normalize_y=False,
 n_restarts_optimizer=3,
 random_state=42
 )

 self.gp.fit(X_scaled, y_scaled)

 # Set bounds
 self.bounds = (X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max())

 self.is_fitted = True
 return True

 except Exception as e:
 logger.error(f"Error fitting Kriging interpolator: {e}")
 return False

 def interpolate(self, log_moneyness: Union[float, np.ndarray],
 time_to_expiry: Union[float, np.ndarray]) -> InterpolationResult:
 """Interpolate using Kriging with uncertainty quantification"""
 if not self.is_fitted:
 raise ValueError("Interpolator not fitted")

 k, T = self._validate_input(log_moneyness, time_to_expiry)
 out_of_bounds = self._check_bounds(k, T)

 # Prepare input
 X_pred = np.column_stack([k, T])
 X_pred_scaled = self.scaler_X.transform(X_pred)

 try:
 # Predict with uncertainty
 y_pred_scaled, y_std_scaled = self.gp.predict(X_pred_scaled, return_std=True)

 # Inverse transform
 y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
 y_pred = np.maximum(y_pred, 0.01) # Ensure positive volatility

 # Calculate confidence based on prediction uncertainty
 confidence = 1.0 / (1.0 + y_std_scaled) # Higher uncertainty = lower confidence
 confidence = np.where(out_of_bounds, confidence * 0.5, confidence)

 return InterpolationResult(
 volatility=y_pred.squeeze(),
 confidence=confidence.squeeze(),
 method_used=self.name,
 extrapolated=np.any(out_of_bounds)
 )

 except Exception as e:
 logger.error(f"Kriging interpolation failed: {e}")
 return InterpolationResult(
 volatility=np.full_like(k, 0.2),
 confidence=np.full_like(k, 0.1),
 method_used=self.name,
 extrapolated=True
 )

 def gradient(self, log_moneyness: float, time_to_expiry: float) -> np.ndarray:
 """Calculate gradient with uncertainty"""
 if not self.is_fitted:
 return np.array([0.0, 0.0])

 h = 1e-6
 X_base = np.array([[log_moneyness, time_to_expiry]])
 X_k_plus = np.array([[log_moneyness + h, time_to_expiry]])
 X_k_minus = np.array([[log_moneyness - h, time_to_expiry]])
 X_T_plus = np.array([[log_moneyness, time_to_expiry + h]])
 X_T_minus = np.array([[log_moneyness, time_to_expiry - h]])

 try:
 def predict_single(X):
 X_scaled = self.scaler_X.transform(X)
 y_scaled = self.gp.predict(X_scaled)
 return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()[0]

 vol_k_plus = predict_single(X_k_plus)
 vol_k_minus = predict_single(X_k_minus)
 vol_T_plus = predict_single(X_T_plus)
 vol_T_minus = predict_single(X_T_minus)

 dvol_dk = (vol_k_plus - vol_k_minus) / (2 * h)
 dvol_dT = (vol_T_plus - vol_T_minus) / (2 * h)

 return np.array([dvol_dk, dvol_dT])

 except Exception:
 return np.array([0.0, 0.0])

class ThinPlateSplineInterpolator(VolatilityInterpolator):
 """Thin plate spline interpolation with smoothing"""

 def __init__(self, smoothing: float = 0.0):
 super().__init__("ThinPlateSpline")
 self.smoothing = smoothing
 self.coefficients = None
 self.points = None
 self.values = None

 def fit(self, quotes: List[VolatilityQuote]) -> bool:
 """Fit thin plate spline"""
 try:
 self.quotes = quotes
 if len(quotes) < 3:
 logger.warning("Need at least 3 quotes for thin plate spline")
 return False

 # Prepare data
 self.points = np.array([[q.log_moneyness, q.expiry] for q in quotes])
 self.values = np.array([q.implied_vol for q in quotes])

 # Build thin plate spline system
 n = len(quotes)
 A = np.zeros((n + 3, n + 3))

 # Fill distance matrix with TPS kernel
 for i in range(n):
 for j in range(n):
 if i != j:
 r = np.linalg.norm(self.points[i] - self.points[j])
 A[i, j] = self._tps_kernel(r)
 else:
 A[i, j] = self.smoothing # Smoothing parameter

 # Polynomial terms
 A[n, :n] = 1
 A[:n, n] = 1
 A[n+1, :n] = self.points[:, 0] # x coordinates
 A[:n, n+1] = self.points[:, 0]
 A[n+2, :n] = self.points[:, 1] # y coordinates
 A[:n, n+2] = self.points[:, 1]

 # Right hand side
 b = np.zeros(n + 3)
 b[:n] = self.values

 # Solve for coefficients
 try:
 self.coefficients = solve(A, b)
 except LinAlgError:
 logger.warning("TPS system is singular, using least squares")
 self.coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

 # Set bounds
 self.bounds = (self.points[:, 0].min(), self.points[:, 0].max(),
 self.points[:, 1].min(), self.points[:, 1].max())

 self.is_fitted = True
 return True

 except Exception as e:
 logger.error(f"Error fitting thin plate spline: {e}")
 return False

 def _tps_kernel(self, r: float) -> float:
 """Thin plate spline radial basis function"""
 if r == 0:
 return 0
 return r * r * np.log(r)

 def interpolate(self, log_moneyness: Union[float, np.ndarray],
 time_to_expiry: Union[float, np.ndarray]) -> InterpolationResult:
 """Interpolate using thin plate spline"""
 if not self.is_fitted:
 raise ValueError("Interpolator not fitted")

 k, T = self._validate_input(log_moneyness, time_to_expiry)
 out_of_bounds = self._check_bounds(k, T)

 result_vols = np.zeros_like(k)

 for i, (ki, Ti) in enumerate(zip(k, T)):
 query_point = np.array([ki, Ti])

 # Evaluate TPS
 vol = 0.0

 # Radial basis function terms
 for j in range(len(self.points)):
 r = np.linalg.norm(query_point - self.points[j])
 vol += self.coefficients[j] * self._tps_kernel(r)

 # Polynomial terms
 vol += self.coefficients[-3] # constant
 vol += self.coefficients[-2] * ki # linear in x
 vol += self.coefficients[-1] * Ti # linear in y

 result_vols[i] = max(vol, 0.01) # Ensure positive

 confidences = np.where(out_of_bounds, 0.5, 1.0)

 return InterpolationResult(
 volatility=result_vols.squeeze(),
 confidence=confidences.squeeze(),
 method_used=self.name,
 extrapolated=np.any(out_of_bounds)
 )

 def gradient(self, log_moneyness: float, time_to_expiry: float) -> np.ndarray:
 """Calculate analytical gradient of thin plate spline"""
 if not self.is_fitted:
 return np.array([0.0, 0.0])

 query_point = np.array([log_moneyness, time_to_expiry])

 dvol_dk = self.coefficients[-2] # Linear term
 dvol_dT = self.coefficients[-1] # Linear term

 # Add contributions from RBF terms
 for j in range(len(self.points)):
 diff = query_point - self.points[j]
 r = np.linalg.norm(diff)

 if r > 1e-10: # Avoid division by zero
 # Derivative of r²log(r) w.r.t. position
 kernel_deriv = 2 * r * np.log(r) + r
 gradient_contribution = self.coefficients[j] * kernel_deriv * diff / r
 dvol_dk += gradient_contribution[0]
 dvol_dT += gradient_contribution[1]

 return np.array([dvol_dk, dvol_dT])

class AdaptiveInterpolationFramework:
 """Adaptive framework that selects best interpolation method based on data characteristics"""

 def __init__(self):
 self.available_methods = {
 InterpolationMethod.CUBIC_SPLINE: CubicSplineInterpolator,
 InterpolationMethod.RBF_GAUSSIAN: lambda: RBFInterpolator('gaussian'),
 InterpolationMethod.KRIGING: KrigingInterpolator,
 InterpolationMethod.THIN_PLATE_SPLINE: ThinPlateSplineInterpolator
 }
 self.method_scores = {}
 self.best_method = None
 self.best_interpolator = None

 def select_best_method(self, quotes: List[VolatilityQuote]) -> VolatilityInterpolator:
 """Select best interpolation method based on cross-validation"""
 if len(quotes) < 10:
 # Default to cubic spline for small datasets
 interpolator = CubicSplineInterpolator()
 interpolator.fit(quotes)
 return interpolator

 method_scores = {}

 for method, interpolator_class in self.available_methods.items():
 try:
 score = self._cross_validate_method(quotes, interpolator_class)
 method_scores[method] = score
 logger.info(f"Method {method.value}: CV score = {score:.4f}")
 except Exception as e:
 logger.warning(f"Failed to evaluate method {method.value}: {e}")
 method_scores[method] = -1.0

 # Select best method
 best_method = max(method_scores.keys(), key=lambda m: method_scores[m])
 self.best_method = best_method
 self.method_scores = method_scores

 # Fit best interpolator on full dataset
 best_interpolator_class = self.available_methods[best_method]
 self.best_interpolator = best_interpolator_class()
 self.best_interpolator.fit(quotes)

 logger.info(f"Selected best method: {best_method.value} (score: {method_scores[best_method]:.4f})")

 return self.best_interpolator

 def _cross_validate_method(self, quotes: List[VolatilityQuote],
 interpolator_class: Callable) -> float:
 """Cross-validate interpolation method"""
 n_folds = min(5, len(quotes) // 3)
 if n_folds < 2:
 return -1.0

 scores = []
 fold_size = len(quotes) // n_folds

 for fold in range(n_folds):
 # Create train/test split
 test_start = fold * fold_size
 test_end = test_start + fold_size if fold < n_folds - 1 else len(quotes)

 test_quotes = quotes[test_start:test_end]
 train_quotes = quotes[:test_start] + quotes[test_end:]

 if len(train_quotes) < 3 or len(test_quotes) < 1:
 continue

 try:
 # Fit on training data
 interpolator = interpolator_class()
 if not interpolator.fit(train_quotes):
 continue

 # Evaluate on test data
 fold_score = 0.0
 for test_quote in test_quotes:
 result = interpolator.interpolate(test_quote.log_moneyness, test_quote.expiry)
 predicted_vol = result.volatility
 actual_vol = test_quote.implied_vol

 # Use relative error
 error = abs(predicted_vol - actual_vol) / actual_vol
 fold_score += (1.0 - min(error, 1.0)) # Convert to score (higher is better)

 scores.append(fold_score / len(test_quotes))

 except Exception:
 continue

 return np.mean(scores) if scores else -1.0

# Factory function for creating interpolators
def create_interpolator(method: InterpolationMethod, **kwargs) -> VolatilityInterpolator:
 """Factory function to create interpolators"""
 if method == InterpolationMethod.CUBIC_SPLINE:
 return CubicSplineInterpolator(**kwargs)
 elif method == InterpolationMethod.RBF_GAUSSIAN:
 return RBFInterpolator('gaussian', **kwargs)
 elif method == InterpolationMethod.RBF_MULTIQUADRIC:
 return RBFInterpolator('multiquadric', **kwargs)
 elif method == InterpolationMethod.KRIGING:
 return KrigingInterpolator(**kwargs)
 elif method == InterpolationMethod.THIN_PLATE_SPLINE:
 return ThinPlateSplineInterpolator(**kwargs)
 else:
 raise ValueError(f"Unknown interpolation method: {method}")

# Utility functions for Greeks calculation using interpolated surfaces
def calculate_greeks_from_surface(interpolator: VolatilityInterpolator,
 log_moneyness: float, time_to_expiry: float,
 spot_price: float, strike: float) -> Dict[str, float]:
 """Calculate Greeks from volatility surface using finite differences"""
 h_k = 0.01 # 1% bump in moneyness
 h_T = 1/365 # 1 day bump in time

 try:
 # Base volatility
 vol_base = interpolator.interpolate(log_moneyness, time_to_expiry).volatility

 # Vega (sensitivity to volatility)
 vol_up = interpolator.interpolate(log_moneyness, time_to_expiry).volatility * 1.01
 vol_down = interpolator.interpolate(log_moneyness, time_to_expiry).volatility * 0.99

 # Volga (vomma) - second derivative w.r.t. volatility
 volga = (vol_up - 2*vol_base + vol_down) / (0.01 * vol_base)**2

 # Vanna - mixed derivative w.r.t. spot and volatility
 vol_k_up = interpolator.interpolate(log_moneyness + h_k, time_to_expiry).volatility
 vol_k_down = interpolator.interpolate(log_moneyness - h_k, time_to_expiry).volatility

 # This is a simplified calculation - in practice would use option pricing model
 vanna_approx = (vol_k_up - vol_k_down) / (2 * h_k * spot_price)

 # Time decay of volatility
 if time_to_expiry > h_T:
 vol_T_down = interpolator.interpolate(log_moneyness, time_to_expiry - h_T).volatility
 vol_time_decay = (vol_T_down - vol_base) / h_T
 else:
 vol_time_decay = 0.0

 return {
 'volatility': float(vol_base),
 'volga': float(volga),
 'vanna': float(vanna_approx),
 'vol_time_decay': float(vol_time_decay)
 }

 except Exception as e:
 logger.error(f"Error calculating Greeks from surface: {e}")
 return {
 'volatility': 0.2,
 'volga': 0.0,
 'vanna': 0.0,
 'vol_time_decay': 0.0
 }