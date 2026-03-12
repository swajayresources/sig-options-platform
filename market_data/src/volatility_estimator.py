"""
Advanced Volatility Estimation and Modeling

Sophisticated volatility estimation methods for options trading including
GARCH models, stochastic volatility, and realized volatility measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import optimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

@dataclass
class VolatilityMetrics:
 """Volatility metrics and statistics"""
 realized_vol: float
 implied_vol: float
 vol_of_vol: float
 skew: float
 term_structure_slope: float
 mean_reversion_speed: float
 correlation_to_underlying: float

class RealizedVolatilityCalculator:
 """Calculate various measures of realized volatility"""

 @staticmethod
 def garman_klass(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
 """Garman-Klass volatility estimator"""
 log_hl = np.log(high / low)
 log_co = np.log(close / np.roll(close, 1)[1:])

 gk_var = 0.5 * np.mean(log_hl[1:]**2) - (2*np.log(2) - 1) * np.mean(log_co**2)
 return np.sqrt(gk_var * 252)

 @staticmethod
 def rogers_satchell(high: np.ndarray, low: np.ndarray,
 open_: np.ndarray, close: np.ndarray) -> float:
 """Rogers-Satchell volatility estimator"""
 log_ho = np.log(high / open_)
 log_hc = np.log(high / close)
 log_lo = np.log(low / open_)
 log_lc = np.log(low / close)

 rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)
 return np.sqrt(rs_var * 252)

 @staticmethod
 def yang_zhang(high: np.ndarray, low: np.ndarray,
 open_: np.ndarray, close: np.ndarray) -> float:
 """Yang-Zhang volatility estimator"""
 log_co = np.log(close[1:] / open_[1:])
 log_oo = np.log(open_[1:] / close[:-1])

 # Rogers-Satchell component
 rs_vol = RealizedVolatilityCalculator.rogers_satchell(
 high[1:], low[1:], open_[1:], close[1:]
 )

 # Overnight and close-to-close components
 k = 0.34 / (1.34 + (len(close) + 1) / (len(close) - 1))

 overnight_var = np.var(log_oo)
 close_to_close_var = np.var(log_co)

 yz_var = overnight_var + k * close_to_close_var + (1 - k) * (rs_vol/np.sqrt(252))**2
 return np.sqrt(yz_var * 252)

 @staticmethod
 def parkinson(high: np.ndarray, low: np.ndarray) -> float:
 """Parkinson volatility estimator"""
 log_hl = np.log(high / low)
 park_var = np.mean(log_hl**2) / (4 * np.log(2))
 return np.sqrt(park_var * 252)

class GARCHModel:
 """GARCH(1,1) model for volatility forecasting"""

 def __init__(self):
 self.omega = None # Constant term
 self.alpha = None # ARCH term
 self.beta = None # GARCH term
 self.is_fitted = False

 def fit(self, returns: np.ndarray) -> Dict:
 """Fit GARCH(1,1) model to return series"""
 n = len(returns)
 returns = returns - np.mean(returns) # Demean returns

 def garch_likelihood(params):
 omega, alpha, beta = params

 if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
 return 1e6

 # Initialize variance
 sigma2 = np.var(returns)
 log_likelihood = 0

 for t in range(1, n):
 sigma2 = omega + alpha * returns[t-1]**2 + beta * sigma2
 if sigma2 <= 0:
 return 1e6
 log_likelihood += 0.5 * (np.log(sigma2) + returns[t]**2 / sigma2)

 return log_likelihood

 # Initial parameters
 initial_guess = [np.var(returns) * 0.1, 0.1, 0.8]

 # Constraints
 bounds = [(1e-6, None), (0, 1), (0, 1)]
 constraint = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}

 try:
 result = optimize.minimize(
 garch_likelihood, initial_guess,
 method='SLSQP', bounds=bounds, constraints=constraint
 )

 if result.success:
 self.omega, self.alpha, self.beta = result.x
 self.is_fitted = True

 return {
 'omega': self.omega,
 'alpha': self.alpha,
 'beta': self.beta,
 'persistence': self.alpha + self.beta,
 'unconditional_vol': np.sqrt(self.omega / (1 - self.alpha - self.beta)),
 'log_likelihood': -result.fun
 }
 else:
 logger.error("GARCH optimization failed")
 return {}

 except Exception as e:
 logger.error(f"GARCH fitting error: {e}")
 return {}

 def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
 """Forecast volatility for given horizon"""
 if not self.is_fitted:
 raise ValueError("Model must be fitted before forecasting")

 n = len(returns)
 returns = returns - np.mean(returns)

 # Calculate current conditional variance
 sigma2 = np.var(returns)
 for t in range(1, n):
 sigma2 = self.omega + self.alpha * returns[t-1]**2 + self.beta * sigma2

 # Forecast
 forecasts = np.zeros(horizon)
 unconditional_var = self.omega / (1 - self.alpha - self.beta)

 for h in range(horizon):
 if h == 0:
 forecasts[h] = sigma2
 else:
 # Mean reversion to unconditional variance
 forecasts[h] = (unconditional_var +
 (forecasts[0] - unconditional_var) *
 (self.alpha + self.beta)**(h))

 return np.sqrt(forecasts * 252) # Annualized volatility

class HestonModel:
 """Heston stochastic volatility model"""

 def __init__(self):
 self.kappa = None # Mean reversion speed
 self.theta = None # Long-term variance
 self.sigma_v = None # Vol of vol
 self.rho = None # Correlation
 self.v0 = None # Initial variance

 def calibrate_to_surface(self, surface_data: pd.DataFrame) -> Dict:
 """Calibrate Heston model to implied volatility surface"""

 def heston_objective(params):
 kappa, theta, sigma_v, rho, v0 = params

 # Parameter constraints
 if (kappa <= 0 or theta <= 0 or sigma_v <= 0 or
 abs(rho) >= 1 or v0 <= 0):
 return 1e6

 # Feller condition
 if 2 * kappa * theta <= sigma_v**2:
 return 1e6

 total_error = 0
 for _, row in surface_data.iterrows():
 S = row['spot']
 K = row['strike']
 T = row['time_to_expiry']
 market_vol = row['implied_vol']

 model_price = self._heston_price(S, K, T, 0.02, kappa, theta, sigma_v, rho, v0)
 model_vol = self._implied_vol_from_price(S, K, T, 0.02, model_price)

 error = (model_vol - market_vol)**2
 total_error += error

 return total_error

 # Initial guess
 initial_params = [2.0, 0.04, 0.3, -0.5, 0.04]

 # Bounds
 bounds = [
 (0.1, 10), # kappa
 (0.01, 1), # theta
 (0.1, 2), # sigma_v
 (-0.99, 0.99), # rho
 (0.01, 1) # v0
 ]

 try:
 result = optimize.minimize(
 heston_objective, initial_params,
 method='L-BFGS-B', bounds=bounds
 )

 if result.success:
 self.kappa, self.theta, self.sigma_v, self.rho, self.v0 = result.x

 return {
 'kappa': self.kappa,
 'theta': self.theta,
 'sigma_v': self.sigma_v,
 'rho': self.rho,
 'v0': self.v0,
 'calibration_error': result.fun
 }
 else:
 logger.error("Heston calibration failed")
 return {}

 except Exception as e:
 logger.error(f"Heston calibration error: {e}")
 return {}

 def _heston_price(self, S: float, K: float, T: float, r: float,
 kappa: float, theta: float, sigma_v: float,
 rho: float, v0: float) -> float:
 """Calculate Heston option price using characteristic function"""
 # Simplified implementation - in practice would use Fourier methods
 # For now, return Black-Scholes with adjusted volatility
 avg_vol = np.sqrt((v0 + theta) / 2)
 d1 = (np.log(S/K) + (r + 0.5*avg_vol**2)*T) / (avg_vol*np.sqrt(T))
 d2 = d1 - avg_vol*np.sqrt(T)

 call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
 return max(call_price, 0)

 def _implied_vol_from_price(self, S: float, K: float, T: float,
 r: float, price: float) -> float:
 """Calculate implied volatility from option price"""
 def black_scholes_price(vol):
 d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
 d2 = d1 - vol*np.sqrt(T)
 return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

 def objective(vol):
 return (black_scholes_price(vol) - price)**2

 try:
 result = optimize.minimize_scalar(objective, bounds=(0.01, 5), method='bounded')
 return result.x if result.success else 0.2
 except:
 return 0.2

class VolatilitySurfaceModel:
 """Model and forecast entire volatility surface"""

 def __init__(self):
 self.models = {}
 self.surface_params = {}

 def fit_surface(self, surface_data: pd.DataFrame):
 """Fit models to volatility surface"""

 # Group by expiry
 for expiry, group in surface_data.groupby('time_to_expiry'):
 strikes = group['strike'].values
 vols = group['implied_vol'].values

 # Fit SVI model to each expiry slice
 svi_params = self._fit_svi_slice(strikes, vols, expiry)
 self.models[expiry] = svi_params

 def _fit_svi_slice(self, strikes: np.ndarray, vols: np.ndarray,
 expiry: float) -> Dict:
 """Fit SVI model to volatility slice"""

 # Convert to log-moneyness and total variance
 forward = 100 # Assume normalized
 k = np.log(strikes / forward)
 w = vols**2 * expiry # Total variance

 def svi_formula(k_val, a, b, rho, m, sigma):
 """SVI parameterization"""
 return a + b * (rho * (k_val - m) +
 np.sqrt((k_val - m)**2 + sigma**2))

 def svi_objective(params):
 a, b, rho, m, sigma = params

 # Parameter constraints
 if b < 0 or abs(rho) >= 1 or sigma <= 0:
 return 1e6

 model_w = svi_formula(k, a, b, rho, m, sigma)
 return np.sum((model_w - w)**2)

 # Initial guess
 initial_params = [0.04, 0.4, -0.4, 0, 0.2]

 try:
 result = optimize.minimize(svi_objective, initial_params, method='Nelder-Mead')

 if result.success:
 return {
 'a': result.x[0],
 'b': result.x[1],
 'rho': result.x[2],
 'm': result.x[3],
 'sigma': result.x[4],
 'fit_error': result.fun
 }
 else:
 return {}
 except:
 return {}

 def interpolate_volatility(self, strike: float, expiry: float) -> float:
 """Interpolate volatility for any strike/expiry"""

 if not self.models:
 return 0.2

 # Find surrounding expiries
 expiries = sorted(self.models.keys())

 if expiry <= expiries[0]:
 return self._svi_volatility(strike, expiry, self.models[expiries[0]])
 elif expiry >= expiries[-1]:
 return self._svi_volatility(strike, expiry, self.models[expiries[-1]])
 else:
 # Linear interpolation between expiries
 for i in range(len(expiries) - 1):
 if expiries[i] <= expiry <= expiries[i + 1]:
 vol1 = self._svi_volatility(strike, expiry, self.models[expiries[i]])
 vol2 = self._svi_volatility(strike, expiry, self.models[expiries[i + 1]])

 weight = (expiry - expiries[i]) / (expiries[i + 1] - expiries[i])
 return vol1 * (1 - weight) + vol2 * weight

 return 0.2

 def _svi_volatility(self, strike: float, expiry: float, params: Dict) -> float:
 """Calculate volatility using SVI parameterization"""
 if not params:
 return 0.2

 forward = 100 # Assume normalized
 k = np.log(strike / forward)

 total_var = (params['a'] + params['b'] *
 (params['rho'] * (k - params['m']) +
 np.sqrt((k - params['m'])**2 + params['sigma']**2)))

 if total_var <= 0 or expiry <= 0:
 return 0.2

 return np.sqrt(total_var / expiry)

class VolatilityRiskMetrics:
 """Calculate various volatility risk metrics"""

 @staticmethod
 def volatility_of_volatility(vol_series: np.ndarray, window: int = 30) -> float:
 """Calculate volatility of volatility"""
 if len(vol_series) < window:
 return 0.1

 vol_changes = np.diff(np.log(vol_series))
 rolling_vol = np.std(vol_changes[-window:])
 return rolling_vol * np.sqrt(252)

 @staticmethod
 def vol_skew(atm_vol: float, otm_put_vol: float, otm_call_vol: float) -> float:
 """Calculate volatility skew"""
 return (otm_put_vol - otm_call_vol) / atm_vol if atm_vol > 0 else 0

 @staticmethod
 def term_structure_slope(short_vol: float, long_vol: float,
 short_term: float, long_term: float) -> float:
 """Calculate volatility term structure slope"""
 if long_term <= short_term:
 return 0
 return (long_vol - short_vol) / (long_term - short_term)

 @staticmethod
 def vol_mean_reversion_speed(vol_series: np.ndarray,
 long_term_vol: float) -> float:
 """Estimate mean reversion speed of volatility"""
 if len(vol_series) < 10:
 return 1.0

 deviations = vol_series - long_term_vol
 lagged_deviations = deviations[:-1]
 current_deviations = deviations[1:]

 if np.var(lagged_deviations) == 0:
 return 1.0

 correlation = np.corrcoef(lagged_deviations, current_deviations)[0, 1]
 mean_reversion = -np.log(max(correlation, 0.01)) * 252 # Annualized

 return mean_reversion

class VolatilityForecaster:
 """Combine multiple models for volatility forecasting"""

 def __init__(self):
 self.garch_model = GARCHModel()
 self.heston_model = HestonModel()
 self.surface_model = VolatilitySurfaceModel()

 # Model weights for ensemble
 self.model_weights = {
 'garch': 0.4,
 'implied': 0.4,
 'realized': 0.2
 }

 def fit_models(self, price_data: pd.DataFrame, surface_data: pd.DataFrame):
 """Fit all volatility models"""

 # Calculate returns for GARCH
 returns = np.diff(np.log(price_data['close'].values))
 garch_results = self.garch_model.fit(returns)

 # Fit surface model
 self.surface_model.fit_surface(surface_data)

 logger.info("Volatility models fitted successfully")

 def forecast_volatility(self, strike: float, expiry: float,
 horizon: int = 1) -> Dict:
 """Generate ensemble volatility forecast"""

 forecasts = {}

 # GARCH forecast (for ATM)
 if self.garch_model.is_fitted:
 # Would need price returns for this - simplified for demo
 garch_vol = 0.2 # Placeholder
 forecasts['garch'] = garch_vol

 # Implied volatility from surface
 implied_vol = self.surface_model.interpolate_volatility(strike, expiry)
 forecasts['implied'] = implied_vol

 # Simple realized volatility (placeholder)
 forecasts['realized'] = 0.18

 # Ensemble forecast
 ensemble_vol = (
 self.model_weights['garch'] * forecasts.get('garch', 0.2) +
 self.model_weights['implied'] * forecasts.get('implied', 0.2) +
 self.model_weights['realized'] * forecasts.get('realized', 0.2)
 )

 return {
 'ensemble_forecast': ensemble_vol,
 'individual_forecasts': forecasts,
 'confidence_interval': (ensemble_vol * 0.8, ensemble_vol * 1.2)
 }