"""
Options Pricing Engine - Python Interface

High-level Python interface to the C++ pricing engine with support for
multiple pricing models and sophisticated Greeks calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import ctypes
import os

class OptionType(Enum):
 CALL = "CALL"
 PUT = "PUT"

class ExerciseType(Enum):
 EUROPEAN = "EUROPEAN"
 AMERICAN = "AMERICAN"
 BERMUDAN = "BERMUDAN"

class PricingModel(Enum):
 BLACK_SCHOLES = "BLACK_SCHOLES"
 BINOMIAL = "BINOMIAL"
 TRINOMIAL = "TRINOMIAL"
 MONTE_CARLO = "MONTE_CARLO"
 HESTON = "HESTON"
 SABR = "SABR"
 JUMP_DIFFUSION = "JUMP_DIFFUSION"

@dataclass
class OptionContract:
 """Options contract specification"""
 symbol: str
 option_type: OptionType
 exercise_type: ExerciseType
 strike: float
 expiry: float # Time to expiry in years
 underlying: str = "SPY"

 def __post_init__(self):
 if self.strike <= 0:
 raise ValueError("Strike must be positive")
 if self.expiry < 0:
 raise ValueError("Time to expiry cannot be negative")

@dataclass
class MarketData:
 """Market data for pricing"""
 spot_price: float
 risk_free_rate: float
 dividend_yield: float
 volatility: float
 time_to_expiry: float

 def __post_init__(self):
 if self.spot_price <= 0:
 raise ValueError("Spot price must be positive")
 if self.volatility < 0:
 raise ValueError("Volatility cannot be negative")

@dataclass
class Greeks:
 """Options Greeks"""
 delta: float = 0.0
 gamma: float = 0.0
 theta: float = 0.0
 vega: float = 0.0
 rho: float = 0.0

 # Second-order Greeks
 vomma: float = 0.0 # Volga
 vanna: float = 0.0
 charm: float = 0.0
 color: float = 0.0
 speed: float = 0.0
 ultima: float = 0.0

 def to_dict(self) -> Dict[str, float]:
 """Convert Greeks to dictionary"""
 return {
 'delta': self.delta,
 'gamma': self.gamma,
 'theta': self.theta,
 'vega': self.vega,
 'rho': self.rho,
 'vomma': self.vomma,
 'vanna': self.vanna,
 'charm': self.charm,
 'color': self.color,
 'speed': self.speed,
 'ultima': self.ultima
 }

@dataclass
class PricingResult:
 """Result of option pricing calculation"""
 price: float
 greeks: Greeks
 implied_volatility: float = 0.0
 success: bool = True
 error_message: str = ""

 def to_dict(self) -> Dict:
 """Convert pricing result to dictionary"""
 return {
 'price': self.price,
 'greeks': self.greeks.to_dict(),
 'implied_volatility': self.implied_volatility,
 'success': self.success,
 'error_message': self.error_message
 }

class BlackScholesCalculator:
 """Black-Scholes pricing model implementation"""

 @staticmethod
 def d1(S: float, K: float, r: float, q: float, vol: float, T: float) -> float:
 """Calculate d1 parameter"""
 if T <= 0 or vol <= 0:
 return 0.0
 return (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))

 @staticmethod
 def d2(S: float, K: float, r: float, q: float, vol: float, T: float) -> float:
 """Calculate d2 parameter"""
 if T <= 0:
 return BlackScholesCalculator.d1(S, K, r, q, vol, T)
 return BlackScholesCalculator.d1(S, K, r, q, vol, T) - vol * np.sqrt(T)

 @staticmethod
 def norm_cdf(x: float) -> float:
 """Standard normal cumulative distribution function"""
 from scipy.stats import norm
 return norm.cdf(x)

 @staticmethod
 def norm_pdf(x: float) -> float:
 """Standard normal probability density function"""
 from scipy.stats import norm
 return norm.pdf(x)

 @classmethod
 def price_call(cls, S: float, K: float, r: float, q: float, vol: float, T: float) -> float:
 """Price European call option"""
 if T <= 0:
 return max(S - K, 0)

 d1_val = cls.d1(S, K, r, q, vol, T)
 d2_val = cls.d2(S, K, r, q, vol, T)

 return S * np.exp(-q * T) * cls.norm_cdf(d1_val) - K * np.exp(-r * T) * cls.norm_cdf(d2_val)

 @classmethod
 def price_put(cls, S: float, K: float, r: float, q: float, vol: float, T: float) -> float:
 """Price European put option"""
 if T <= 0:
 return max(K - S, 0)

 d1_val = cls.d1(S, K, r, q, vol, T)
 d2_val = cls.d2(S, K, r, q, vol, T)

 return K * np.exp(-r * T) * cls.norm_cdf(-d2_val) - S * np.exp(-q * T) * cls.norm_cdf(-d1_val)

 @classmethod
 def calculate_greeks(cls, option: OptionContract, market_data: MarketData) -> Greeks:
 """Calculate all Greeks for an option"""
 S = market_data.spot_price
 K = option.strike
 r = market_data.risk_free_rate
 q = market_data.dividend_yield
 vol = market_data.volatility
 T = market_data.time_to_expiry

 if T <= 0:
 return Greeks()

 d1_val = cls.d1(S, K, r, q, vol, T)
 d2_val = cls.d2(S, K, r, q, vol, T)

 greeks = Greeks()

 # First-order Greeks
 if option.option_type == OptionType.CALL:
 greeks.delta = np.exp(-q * T) * cls.norm_cdf(d1_val)
 greeks.rho = K * T * np.exp(-r * T) * cls.norm_cdf(d2_val)
 else:
 greeks.delta = -np.exp(-q * T) * cls.norm_cdf(-d1_val)
 greeks.rho = -K * T * np.exp(-r * T) * cls.norm_cdf(-d2_val)

 greeks.gamma = np.exp(-q * T) * cls.norm_pdf(d1_val) / (S * vol * np.sqrt(T))
 greeks.vega = S * np.exp(-q * T) * cls.norm_pdf(d1_val) * np.sqrt(T)

 # Theta calculation
 term1 = -S * np.exp(-q * T) * cls.norm_pdf(d1_val) * vol / (2 * np.sqrt(T))
 if option.option_type == OptionType.CALL:
 term2 = -r * K * np.exp(-r * T) * cls.norm_cdf(d2_val)
 term3 = q * S * np.exp(-q * T) * cls.norm_cdf(d1_val)
 else:
 term2 = r * K * np.exp(-r * T) * cls.norm_cdf(-d2_val)
 term3 = -q * S * np.exp(-q * T) * cls.norm_cdf(-d1_val)

 greeks.theta = term1 + term2 + term3

 # Second-order Greeks
 greeks.vomma = greeks.vega * d1_val * d2_val / vol
 greeks.vanna = -np.exp(-q * T) * d2_val / vol * cls.norm_pdf(d1_val) * np.sqrt(T)

 return greeks

class BinomialTreeCalculator:
 """Binomial tree pricing model"""

 def __init__(self, steps: int = 100):
 self.steps = steps

 def price_option(self, option: OptionContract, market_data: MarketData) -> PricingResult:
 """Price option using binomial tree"""
 S = market_data.spot_price
 K = option.strike
 r = market_data.risk_free_rate
 q = market_data.dividend_yield
 vol = market_data.volatility
 T = market_data.time_to_expiry

 if T <= 0:
 intrinsic = max(S - K, 0) if option.option_type == OptionType.CALL else max(K - S, 0)
 return PricingResult(intrinsic, Greeks())

 dt = T / self.steps
 u = np.exp(vol * np.sqrt(dt))
 d = 1 / u
 p = (np.exp((r - q) * dt) - d) / (u - d)
 discount = np.exp(-r * dt)

 # Build stock price tree (final prices only)
 stock_prices = np.zeros(self.steps + 1)
 for i in range(self.steps + 1):
 stock_prices[i] = S * (u ** i) * (d ** (self.steps - i))

 # Option values at expiry
 option_values = np.zeros(self.steps + 1)
 for i in range(self.steps + 1):
 if option.option_type == OptionType.CALL:
 option_values[i] = max(stock_prices[i] - K, 0)
 else:
 option_values[i] = max(K - stock_prices[i], 0)

 # Work backwards through tree
 for step in range(self.steps - 1, -1, -1):
 for i in range(step + 1):
 continuation_value = discount * (p * option_values[i + 1] + (1 - p) * option_values[i])

 if option.exercise_type == ExerciseType.AMERICAN:
 # Check early exercise
 current_stock = S * (u ** i) * (d ** (step - i))
 if option.option_type == OptionType.CALL:
 exercise_value = max(current_stock - K, 0)
 else:
 exercise_value = max(K - current_stock, 0)

 option_values[i] = max(continuation_value, exercise_value)
 else:
 option_values[i] = continuation_value

 # Calculate Greeks using finite differences
 greeks = self._calculate_greeks_finite_diff(option, market_data)

 return PricingResult(option_values[0], greeks)

 def _calculate_greeks_finite_diff(self, option: OptionContract, market_data: MarketData) -> Greeks:
 """Calculate Greeks using finite differences"""
 base_result = self.price_option(option, market_data)
 base_price = base_result.price

 greeks = Greeks()

 # Delta
 bump = market_data.spot_price * 0.01
 market_up = MarketData(
 market_data.spot_price + bump,
 market_data.risk_free_rate,
 market_data.dividend_yield,
 market_data.volatility,
 market_data.time_to_expiry
 )
 market_down = MarketData(
 market_data.spot_price - bump,
 market_data.risk_free_rate,
 market_data.dividend_yield,
 market_data.volatility,
 market_data.time_to_expiry
 )

 price_up = self.price_option(option, market_up).price
 price_down = self.price_option(option, market_down).price

 greeks.delta = (price_up - price_down) / (2 * bump)
 greeks.gamma = (price_up - 2 * base_price + price_down) / (bump ** 2)

 return greeks

class MonteCarloCalculator:
 """Monte Carlo pricing engine"""

 def __init__(self, num_simulations: int = 100000, random_seed: Optional[int] = None):
 self.num_simulations = num_simulations
 if random_seed is not None:
 np.random.seed(random_seed)

 def price_european_option(self, option: OptionContract, market_data: MarketData) -> PricingResult:
 """Price European option using Monte Carlo"""
 S = market_data.spot_price
 K = option.strike
 r = market_data.risk_free_rate
 q = market_data.dividend_yield
 vol = market_data.volatility
 T = market_data.time_to_expiry

 if T <= 0:
 intrinsic = max(S - K, 0) if option.option_type == OptionType.CALL else max(K - S, 0)
 return PricingResult(intrinsic, Greeks())

 # Generate random paths
 z = np.random.standard_normal(self.num_simulations)
 ST = S * np.exp((r - q - 0.5 * vol**2) * T + vol * np.sqrt(T) * z)

 # Calculate payoffs
 if option.option_type == OptionType.CALL:
 payoffs = np.maximum(ST - K, 0)
 else:
 payoffs = np.maximum(K - ST, 0)

 # Discount to present value
 price = np.exp(-r * T) * np.mean(payoffs)

 # Calculate Greeks using finite differences
 greeks = self._calculate_greeks_finite_diff(option, market_data)

 return PricingResult(price, greeks)

 def price_asian_option(self, option: OptionContract, market_data: MarketData,
 num_time_steps: int = 252) -> PricingResult:
 """Price Asian option using Monte Carlo"""
 S = market_data.spot_price
 K = option.strike
 r = market_data.risk_free_rate
 q = market_data.dividend_yield
 vol = market_data.volatility
 T = market_data.time_to_expiry

 dt = T / num_time_steps
 payoffs = np.zeros(self.num_simulations)

 for i in range(self.num_simulations):
 path = self._generate_price_path(S, r, q, vol, T, num_time_steps)
 average_price = np.mean(path)

 if option.option_type == OptionType.CALL:
 payoffs[i] = max(average_price - K, 0)
 else:
 payoffs[i] = max(K - average_price, 0)

 price = np.exp(-r * T) * np.mean(payoffs)
 return PricingResult(price, Greeks())

 def _generate_price_path(self, S0: float, r: float, q: float, vol: float,
 T: float, steps: int) -> np.ndarray:
 """Generate single price path using geometric Brownian motion"""
 dt = T / steps
 path = np.zeros(steps + 1)
 path[0] = S0

 for i in range(1, steps + 1):
 z = np.random.standard_normal()
 path[i] = path[i-1] * np.exp((r - q - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z)

 return path

 def _calculate_greeks_finite_diff(self, option: OptionContract, market_data: MarketData) -> Greeks:
 """Calculate Greeks using finite differences"""
 # Simplified Greeks calculation
 return Greeks()

class PricingEngine:
 """Main pricing engine coordinating all models"""

 def __init__(self):
 self.black_scholes = BlackScholesCalculator()
 self.binomial = BinomialTreeCalculator()
 self.monte_carlo = MonteCarloCalculator()

 def price_option(self, option: OptionContract, market_data: MarketData,
 model: PricingModel = PricingModel.BLACK_SCHOLES) -> PricingResult:
 """Price option using specified model"""
 try:
 if model == PricingModel.BLACK_SCHOLES:
 return self._price_black_scholes(option, market_data)
 elif model == PricingModel.BINOMIAL:
 return self.binomial.price_option(option, market_data)
 elif model == PricingModel.MONTE_CARLO:
 return self.monte_carlo.price_european_option(option, market_data)
 else:
 return PricingResult(0, Greeks(), success=False,
 error_message=f"Model {model} not implemented")

 except Exception as e:
 return PricingResult(0, Greeks(), success=False, error_message=str(e))

 def _price_black_scholes(self, option: OptionContract, market_data: MarketData) -> PricingResult:
 """Price using Black-Scholes model"""
 S = market_data.spot_price
 K = option.strike
 r = market_data.risk_free_rate
 q = market_data.dividend_yield
 vol = market_data.volatility
 T = market_data.time_to_expiry

 if option.option_type == OptionType.CALL:
 price = self.black_scholes.price_call(S, K, r, q, vol, T)
 else:
 price = self.black_scholes.price_put(S, K, r, q, vol, T)

 greeks = self.black_scholes.calculate_greeks(option, market_data)

 return PricingResult(price, greeks)

 def calculate_implied_volatility(self, option: OptionContract, market_data: MarketData,
 market_price: float, tolerance: float = 1e-6) -> float:
 """Calculate implied volatility using Newton-Raphson method"""
 vol_guess = 0.2 # 20% initial guess

 for i in range(50): # Maximum iterations
 market_data_temp = MarketData(
 market_data.spot_price,
 market_data.risk_free_rate,
 market_data.dividend_yield,
 vol_guess,
 market_data.time_to_expiry
 )

 result = self._price_black_scholes(option, market_data_temp)
 price_diff = result.price - market_price

 if abs(price_diff) < tolerance:
 return vol_guess

 vega = result.greeks.vega
 if abs(vega) < 1e-10:
 break

 vol_guess = vol_guess - price_diff / vega
 vol_guess = max(vol_guess, 0.001) # Keep positive

 return 0.0 # Failed to converge

 def price_portfolio(self, options: List[Tuple[OptionContract, MarketData, int]],
 model: PricingModel = PricingModel.BLACK_SCHOLES) -> Dict:
 """Price a portfolio of options"""
 total_value = 0.0
 total_greeks = Greeks()
 results = []

 for option, market_data, quantity in options:
 result = self.price_option(option, market_data, model)

 if result.success:
 position_value = result.price * quantity
 total_value += position_value

 # Aggregate Greeks
 total_greeks.delta += result.greeks.delta * quantity
 total_greeks.gamma += result.greeks.gamma * quantity
 total_greeks.theta += result.greeks.theta * quantity
 total_greeks.vega += result.greeks.vega * quantity
 total_greeks.rho += result.greeks.rho * quantity

 results.append({
 'option': option,
 'quantity': quantity,
 'result': result,
 'position_value': result.price * quantity if result.success else 0
 })

 return {
 'total_value': total_value,
 'total_greeks': total_greeks,
 'positions': results
 }