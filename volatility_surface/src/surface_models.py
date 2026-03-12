"""
Advanced Volatility Surface Modeling Framework

Comprehensive implementation of sophisticated volatility surface models including
SVI, SABR, Local Volatility, and advanced calibration algorithms for professional
options trading and market making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from scipy import optimize, interpolate
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

class SurfaceModelType(Enum):
    """Volatility surface model types"""
    SVI = "SVI"
    SABR = "SABR"
    LOCAL_VOL = "LOCAL_VOL"
    CUBIC_SPLINE = "CUBIC_SPLINE"
    RBFN = "RBFN"
    HESTON = "HESTON"

class CalibrationMethod(Enum):
    """Calibration optimization methods"""
    LEVENBERG_MARQUARDT = "LM"
    DIFFERENTIAL_EVOLUTION = "DE"
    PARTICLE_SWARM = "PSO"
    SIMULATED_ANNEALING = "SA"
    BASINHOPPING = "BH"
    DUAL_ANNEALING = "DA"

@dataclass
class VolatilityQuote:
    """Individual volatility quote with metadata"""
    strike: float
    expiry: float  # Time to expiry in years
    implied_vol: float
    bid_vol: Optional[float] = None
    ask_vol: Optional[float] = None
    volume: int = 0
    open_interest: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0

    @property
    def log_moneyness(self) -> float:
        """Calculate log-moneyness assuming spot = 1 (normalized)"""
        return np.log(self.strike)

    @property
    def total_variance(self) -> float:
        """Calculate total implied variance"""
        return self.implied_vol ** 2 * self.expiry

@dataclass
class CalibrationResult:
    """Result of volatility surface calibration"""
    success: bool
    parameters: Dict[str, float]
    objective_value: float
    iterations: int
    execution_time: float
    convergence_message: str
    fitted_vols: np.ndarray
    residuals: np.ndarray
    r_squared: float
    aic: float
    bic: float

class VolatilitySurfaceModel(ABC):
    """Abstract base class for volatility surface models"""

    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.is_calibrated = False
        self.calibration_history = []

    @abstractmethod
    def calculate_volatility(self, log_moneyness: Union[float, np.ndarray],
                           time_to_expiry: Union[float, np.ndarray]) -> np.ndarray:
        """Calculate implied volatility for given log-moneyness and time to expiry"""
        pass

    @abstractmethod
    def calibrate(self, quotes: List[VolatilityQuote], **kwargs) -> CalibrationResult:
        """Calibrate model parameters to market quotes"""
        pass

    @abstractmethod
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate parameter constraints"""
        pass

class SVIModel(VolatilitySurfaceModel):
    """
    SVI (Stochastic Volatility Inspired) Model

    Total variance parameterization:
    w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))

    Where:
    - k = log(K/F) is log-moneyness
    - w = σ²T is total variance
    - Parameters: a, b, ρ, m, σ
    """

    def __init__(self):
        super().__init__("SVI")
        self.slice_parameters = {}  # Parameters by expiry

    def calculate_total_variance(self, log_moneyness: Union[float, np.ndarray],
                               expiry: float) -> np.ndarray:
        """Calculate total variance using SVI parameterization"""
        if expiry not in self.slice_parameters:
            raise ValueError(f"No calibrated parameters for expiry {expiry}")

        params = self.slice_parameters[expiry]
        a, b, rho, m, sigma = params['a'], params['b'], params['rho'], params['m'], params['sigma']

        k = np.atleast_1d(log_moneyness)
        w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        return w

    def calculate_volatility(self, log_moneyness: Union[float, np.ndarray],
                           time_to_expiry: Union[float, np.ndarray]) -> np.ndarray:
        """Calculate implied volatility from SVI parameters"""
        k = np.atleast_1d(log_moneyness)
        T = np.atleast_1d(time_to_expiry)

        result = np.zeros((len(k), len(T)))

        for i, expiry in enumerate(T):
            if expiry in self.slice_parameters:
                w = self.calculate_total_variance(k, expiry)
                result[:, i] = np.sqrt(np.maximum(w / expiry, 1e-8))
            else:
                # Interpolate between available expiries
                result[:, i] = self._interpolate_across_time(k, expiry)

        return result.squeeze()

    def calibrate_slice(self, quotes: List[VolatilityQuote], expiry: float,
                       method: CalibrationMethod = CalibrationMethod.LEVENBERG_MARQUARDT) -> CalibrationResult:
        """Calibrate SVI parameters for a single expiry slice"""

        # Extract data for this expiry
        slice_quotes = [q for q in quotes if abs(q.expiry - expiry) < 1e-6]
        if len(slice_quotes) < 5:
            raise ValueError(f"Need at least 5 quotes for SVI calibration, got {len(slice_quotes)}")

        k_market = np.array([q.log_moneyness for q in slice_quotes])
        w_market = np.array([q.total_variance for q in slice_quotes])
        weights = np.array([q.confidence for q in slice_quotes])

        # Objective function
        def objective(params):
            try:
                a, b, rho, m, sigma = params

                # Parameter constraints
                if not self._validate_svi_parameters(a, b, rho, m, sigma):
                    return 1e10

                # Calculate model total variance
                w_model = a + b * (rho * (k_market - m) + np.sqrt((k_market - m)**2 + sigma**2))

                # Weighted mean squared error
                residuals = (w_model - w_market) * weights
                return np.sum(residuals**2)

            except Exception:
                return 1e10

        # Initial guess
        initial_guess = self._generate_svi_initial_guess(k_market, w_market)

        # Parameter bounds
        bounds = [
            (0.001, 1.0),      # a: minimum variance level
            (0.001, 2.0),      # b: volatility of variance
            (-0.999, 0.999),   # ρ: correlation
            (k_market.min()-1, k_market.max()+1),  # m: location
            (0.001, 2.0)       # σ: scale
        ]

        start_time = datetime.now()

        try:
            if method == CalibrationMethod.LEVENBERG_MARQUARDT:
                result = optimize.least_squares(
                    lambda p: np.sqrt(objective(p)),
                    initial_guess,
                    bounds=tuple(zip(*bounds)),
                    method='lm',
                    max_nfev=1000
                )

            elif method == CalibrationMethod.DIFFERENTIAL_EVOLUTION:
                result = optimize.differential_evolution(
                    objective,
                    bounds,
                    seed=42,
                    maxiter=500,
                    atol=1e-8
                )

            else:
                result = optimize.minimize(
                    objective,
                    initial_guess,
                    bounds=bounds,
                    method='L-BFGS-B'
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            if result.success:
                params = dict(zip(['a', 'b', 'rho', 'm', 'sigma'], result.x))
                self.slice_parameters[expiry] = params

                # Calculate fitted values and metrics
                fitted_w = self.calculate_total_variance(k_market, expiry)
                fitted_vols = np.sqrt(fitted_w / expiry)
                market_vols = np.array([q.implied_vol for q in slice_quotes])

                residuals = fitted_vols - market_vols
                r_squared = 1 - np.sum(residuals**2) / np.sum((market_vols - np.mean(market_vols))**2)

                # Information criteria
                n = len(slice_quotes)
                k_params = 5
                mse = np.mean(residuals**2)
                aic = n * np.log(mse) + 2 * k_params
                bic = n * np.log(mse) + k_params * np.log(n)

                return CalibrationResult(
                    success=True,
                    parameters=params,
                    objective_value=result.fun,
                    iterations=getattr(result, 'nit', 0),
                    execution_time=execution_time,
                    convergence_message=getattr(result, 'message', 'Success'),
                    fitted_vols=fitted_vols,
                    residuals=residuals,
                    r_squared=r_squared,
                    aic=aic,
                    bic=bic
                )
            else:
                return CalibrationResult(
                    success=False,
                    parameters={},
                    objective_value=1e10,
                    iterations=0,
                    execution_time=execution_time,
                    convergence_message=getattr(result, 'message', 'Failed'),
                    fitted_vols=np.array([]),
                    residuals=np.array([]),
                    r_squared=0.0,
                    aic=1e10,
                    bic=1e10
                )

        except Exception as e:
            logger.error(f"SVI calibration failed: {e}")
            return CalibrationResult(
                success=False,
                parameters={},
                objective_value=1e10,
                iterations=0,
                execution_time=0.0,
                convergence_message=str(e),
                fitted_vols=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
                aic=1e10,
                bic=1e10
            )

    def calibrate(self, quotes: List[VolatilityQuote], **kwargs) -> Dict[float, CalibrationResult]:
        """Calibrate SVI parameters for all expiry slices"""
        expiries = sorted(list(set(q.expiry for q in quotes)))
        results = {}

        for expiry in expiries:
            try:
                result = self.calibrate_slice(quotes, expiry, **kwargs)
                results[expiry] = result
                if result.success:
                    logger.info(f"SVI calibration successful for T={expiry:.3f}, R²={result.r_squared:.4f}")
                else:
                    logger.warning(f"SVI calibration failed for T={expiry:.3f}")
            except Exception as e:
                logger.error(f"Error calibrating SVI for T={expiry:.3f}: {e}")

        self.is_calibrated = len(results) > 0 and any(r.success for r in results.values())
        return results

    def _validate_svi_parameters(self, a: float, b: float, rho: float, m: float, sigma: float) -> bool:
        """Validate SVI parameter constraints for no-arbitrage"""
        try:
            # Basic constraints
            if a < 0 or b < 0 or abs(rho) >= 1 or sigma <= 0:
                return False

            # Gatheral no-arbitrage conditions
            if b * (1 + abs(rho)) >= 4:
                return False

            # Additional stability constraints
            if a + b * sigma * np.sqrt(1 - rho**2) < 0:
                return False

            return True
        except:
            return False

    def _generate_svi_initial_guess(self, k: np.ndarray, w: np.ndarray) -> List[float]:
        """Generate intelligent initial guess for SVI parameters"""
        try:
            # Simple moments-based initial guess
            a = np.min(w) * 0.8  # Minimum variance level
            b = (np.max(w) - np.min(w)) * 0.5  # Scale of variance variation
            rho = -0.3  # Typical negative correlation
            m = np.mean(k)  # Center around mean log-moneyness
            sigma = np.std(k) if np.std(k) > 0.1 else 0.2  # Scale parameter

            return [a, b, rho, m, sigma]
        except:
            return [0.04, 0.4, -0.3, 0.0, 0.2]  # Fallback defaults

    def _interpolate_across_time(self, k: np.ndarray, target_expiry: float) -> np.ndarray:
        """Interpolate volatility across time for missing expiries"""
        available_expiries = sorted(self.slice_parameters.keys())

        if not available_expiries:
            return np.full_like(k, 0.2)  # Default 20% volatility

        if target_expiry <= available_expiries[0]:
            return self.calculate_volatility(k, available_expiries[0])
        elif target_expiry >= available_expiries[-1]:
            return self.calculate_volatility(k, available_expiries[-1])
        else:
            # Linear interpolation between adjacent expiries
            for i in range(len(available_expiries) - 1):
                if available_expiries[i] <= target_expiry <= available_expiries[i + 1]:
                    T1, T2 = available_expiries[i], available_expiries[i + 1]
                    vol1 = self.calculate_volatility(k, T1)
                    vol2 = self.calculate_volatility(k, T2)

                    weight = (target_expiry - T1) / (T2 - T1)
                    return vol1 * (1 - weight) + vol2 * weight

            return np.full_like(k, 0.2)  # Fallback

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate SVI parameters"""
        required_params = ['a', 'b', 'rho', 'm', 'sigma']
        if not all(p in params for p in required_params):
            return False

        return self._validate_svi_parameters(
            params['a'], params['b'], params['rho'], params['m'], params['sigma']
        )

class SABRModel(VolatilitySurfaceModel):
    """
    SABR (Stochastic Alpha Beta Rho) Model

    Hagan's formula for implied volatility:
    σ_BS(K,F,T) = (α/((FK)^((1-β)/2) * (1 + ((1-β)/24)ln²(F/K) + ...))) *
                  (z/x(z)) * (1 + ((1-β)²/24 * α²/(FK)^(1-β) + ρβνα/4/(FK)^((1-β)/2) + (2-3ρ²)/24 * ν²) * T)

    Parameters:
    - α: ATM volatility scaling
    - β: CEV parameter (0 ≤ β ≤ 1)
    - ρ: Correlation between asset and volatility (-1 ≤ ρ ≤ 1)
    - ν: Volatility of volatility (vol-of-vol)
    """

    def __init__(self):
        super().__init__("SABR")
        self.slice_parameters = {}

    def calculate_volatility(self, log_moneyness: Union[float, np.ndarray],
                           time_to_expiry: Union[float, np.ndarray],
                           forward: float = 1.0) -> np.ndarray:
        """Calculate implied volatility using SABR formula"""
        k = np.atleast_1d(log_moneyness)
        T = np.atleast_1d(time_to_expiry)
        strikes = forward * np.exp(k)  # Convert log-moneyness to strikes

        result = np.zeros((len(k), len(T)))

        for i, expiry in enumerate(T):
            if expiry in self.slice_parameters:
                params = self.slice_parameters[expiry]
                result[:, i] = self._sabr_volatility(forward, strikes, expiry, params)
            else:
                # Use interpolation or default
                result[:, i] = 0.2  # Default 20% volatility

        return result.squeeze()

    def _sabr_volatility(self, forward: float, strikes: np.ndarray,
                        time_to_expiry: float, params: Dict[str, float]) -> np.ndarray:
        """Calculate SABR implied volatility using Hagan's formula"""
        alpha, beta, rho, nu = params['alpha'], params['beta'], params['rho'], params['nu']

        F, K, T = forward, strikes, time_to_expiry

        # Handle ATM case separately
        atm_mask = np.abs(K - F) < 1e-8
        vol = np.zeros_like(K)

        # ATM volatility
        if np.any(atm_mask):
            vol_atm = alpha / (F ** (1 - beta)) * (
                1 + ((1 - beta)**2 / 24 * alpha**2 / F**(2 - 2*beta) +
                     rho * beta * nu * alpha / (4 * F**(1 - beta)) +
                     (2 - 3 * rho**2) / 24 * nu**2) * T
            )
            vol[atm_mask] = vol_atm

        # Non-ATM volatility
        non_atm_mask = ~atm_mask
        if np.any(non_atm_mask):
            F_nonATM = F
            K_nonATM = K[non_atm_mask]

            # Calculate z and x(z)
            sqrt_FK = np.sqrt(F_nonATM * K_nonATM)
            log_FK = np.log(F_nonATM / K_nonATM)

            z = nu / alpha * sqrt_FK * log_FK

            # Calculate x(z) with numerical stability
            x_z = np.where(
                np.abs(z) < 1e-7,
                1.0,  # Limit as z -> 0
                z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
            )

            # Main SABR formula
            vol_non_atm = (
                alpha / (sqrt_FK**(1 - beta) *
                        (1 + (1 - beta)**2 / 24 * log_FK**2 +
                         (1 - beta)**4 / 1920 * log_FK**4)) *
                x_z *
                (1 + ((1 - beta)**2 / 24 * alpha**2 / sqrt_FK**(2 - 2*beta) +
                      rho * beta * nu * alpha / (4 * sqrt_FK**(1 - beta)) +
                      (2 - 3 * rho**2) / 24 * nu**2) * T)
            )

            vol[non_atm_mask] = vol_non_atm

        return np.maximum(vol, 1e-8)  # Ensure positive volatility

    def calibrate_slice(self, quotes: List[VolatilityQuote], expiry: float,
                       forward: float = 1.0,
                       method: CalibrationMethod = CalibrationMethod.DIFFERENTIAL_EVOLUTION) -> CalibrationResult:
        """Calibrate SABR parameters for a single expiry slice"""

        slice_quotes = [q for q in quotes if abs(q.expiry - expiry) < 1e-6]
        if len(slice_quotes) < 4:
            raise ValueError(f"Need at least 4 quotes for SABR calibration, got {len(slice_quotes)}")

        strikes = forward * np.exp(np.array([q.log_moneyness for q in slice_quotes]))
        market_vols = np.array([q.implied_vol for q in slice_quotes])
        weights = np.array([q.confidence for q in slice_quotes])

        def objective(params):
            try:
                alpha, beta, rho, nu = params

                if not self._validate_sabr_parameters(alpha, beta, rho, nu):
                    return 1e10

                model_vols = self._sabr_volatility(forward, strikes, expiry,
                                                 {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu})

                residuals = (model_vols - market_vols) * weights
                return np.sum(residuals**2)

            except Exception:
                return 1e10

        # Parameter bounds
        bounds = [
            (0.001, 2.0),     # alpha
            (0.0, 1.0),       # beta
            (-0.99, 0.99),    # rho
            (0.001, 2.0)      # nu
        ]

        # Initial guess
        initial_guess = [
            np.mean(market_vols),  # alpha ~ ATM vol
            0.5,                   # beta = 0.5 (common choice)
            -0.3,                  # rho (typically negative)
            0.3                    # nu (moderate vol-of-vol)
        ]

        start_time = datetime.now()

        try:
            if method == CalibrationMethod.DIFFERENTIAL_EVOLUTION:
                result = optimize.differential_evolution(
                    objective, bounds, seed=42, maxiter=500, atol=1e-8
                )
            else:
                result = optimize.minimize(
                    objective, initial_guess, bounds=bounds, method='L-BFGS-B'
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            if result.success:
                params = dict(zip(['alpha', 'beta', 'rho', 'nu'], result.x))
                self.slice_parameters[expiry] = params

                # Calculate metrics
                fitted_vols = self._sabr_volatility(forward, strikes, expiry, params)
                residuals = fitted_vols - market_vols
                r_squared = 1 - np.sum(residuals**2) / np.sum((market_vols - np.mean(market_vols))**2)

                n = len(slice_quotes)
                k_params = 4
                mse = np.mean(residuals**2)
                aic = n * np.log(mse) + 2 * k_params
                bic = n * np.log(mse) + k_params * np.log(n)

                return CalibrationResult(
                    success=True,
                    parameters=params,
                    objective_value=result.fun,
                    iterations=getattr(result, 'nit', 0),
                    execution_time=execution_time,
                    convergence_message=getattr(result, 'message', 'Success'),
                    fitted_vols=fitted_vols,
                    residuals=residuals,
                    r_squared=r_squared,
                    aic=aic,
                    bic=bic
                )
            else:
                return CalibrationResult(
                    success=False,
                    parameters={},
                    objective_value=1e10,
                    iterations=0,
                    execution_time=execution_time,
                    convergence_message=getattr(result, 'message', 'Failed'),
                    fitted_vols=np.array([]),
                    residuals=np.array([]),
                    r_squared=0.0,
                    aic=1e10,
                    bic=1e10
                )

        except Exception as e:
            logger.error(f"SABR calibration failed: {e}")
            return CalibrationResult(
                success=False,
                parameters={},
                objective_value=1e10,
                iterations=0,
                execution_time=0.0,
                convergence_message=str(e),
                fitted_vols=np.array([]),
                residuals=np.array([]),
                r_squared=0.0,
                aic=1e10,
                bic=1e10
            )

    def calibrate(self, quotes: List[VolatilityQuote], **kwargs) -> Dict[float, CalibrationResult]:
        """Calibrate SABR parameters for all expiry slices"""
        expiries = sorted(list(set(q.expiry for q in quotes)))
        results = {}

        for expiry in expiries:
            try:
                result = self.calibrate_slice(quotes, expiry, **kwargs)
                results[expiry] = result
                if result.success:
                    logger.info(f"SABR calibration successful for T={expiry:.3f}, R²={result.r_squared:.4f}")
                else:
                    logger.warning(f"SABR calibration failed for T={expiry:.3f}")
            except Exception as e:
                logger.error(f"Error calibrating SABR for T={expiry:.3f}: {e}")

        self.is_calibrated = len(results) > 0 and any(r.success for r in results.values())
        return results

    def _validate_sabr_parameters(self, alpha: float, beta: float, rho: float, nu: float) -> bool:
        """Validate SABR parameter constraints"""
        return (alpha > 0 and 0 <= beta <= 1 and -1 <= rho <= 1 and nu > 0)

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate SABR parameters"""
        required_params = ['alpha', 'beta', 'rho', 'nu']
        if not all(p in params for p in required_params):
            return False

        return self._validate_sabr_parameters(
            params['alpha'], params['beta'], params['rho'], params['nu']
        )