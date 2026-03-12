"""
Options Pricing Engine
Advanced options pricing and Greeks calculations using QuantLib
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
import math

try:
 import QuantLib as ql
 QUANTLIB_AVAILABLE = True
except ImportError:
 QUANTLIB_AVAILABLE = False
 print("QuantLib not available. Using simplified Black-Scholes implementation.")

class OptionsPricingEngine:
 """Advanced options pricing engine with multiple models"""

 def __init__(self):
 self.risk_free_rate = 0.05
 self.dividend_yield = 0.0
 self.calculation_date = datetime.now()

 if QUANTLIB_AVAILABLE:
 self._setup_quantlib()

 def _setup_quantlib(self):
 """Setup QuantLib environment"""
 if not QUANTLIB_AVAILABLE:
 return

 # Set evaluation date
 ql_date = ql.Date(
 self.calculation_date.day,
 self.calculation_date.month,
 self.calculation_date.year
 )
 ql.Settings.instance().evaluationDate = ql_date

 # Setup curves
 self.risk_free_curve = ql.FlatForward(ql_date, self.risk_free_rate, ql.Actual365Fixed())
 self.dividend_curve = ql.FlatForward(ql_date, self.dividend_yield, ql.Actual365Fixed())

 def calculate_option_price(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float, option_type: str = 'call',
 model: str = 'black_scholes') -> Dict[str, float]:
 """Calculate option price using specified model"""

 if model == 'black_scholes':
 return self._black_scholes_price(spot, strike, volatility, time_to_expiry, option_type)
 elif model == 'binomial' and QUANTLIB_AVAILABLE:
 return self._binomial_price(spot, strike, volatility, time_to_expiry, option_type)
 elif model == 'monte_carlo':
 return self._monte_carlo_price(spot, strike, volatility, time_to_expiry, option_type)
 else:
 # Fallback to Black-Scholes
 return self._black_scholes_price(spot, strike, volatility, time_to_expiry, option_type)

 def calculate_greeks(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float, option_type: str = 'call') -> Dict[str, float]:
 """Calculate option Greeks"""

 if QUANTLIB_AVAILABLE:
 return self._quantlib_greeks(spot, strike, volatility, time_to_expiry, option_type)
 else:
 return self._analytical_greeks(spot, strike, volatility, time_to_expiry, option_type)

 def calculate_implied_volatility(self, spot: float, strike: float, time_to_expiry: float,
 market_price: float, option_type: str = 'call') -> float:
 """Calculate implied volatility from market price"""

 def objective_function(vol):
 theoretical_price = self._black_scholes_price(
 spot, strike, vol, time_to_expiry, option_type
 )['price']
 return abs(theoretical_price - market_price)

 # Use bisection method to find implied volatility
 vol_low, vol_high = 0.01, 3.0
 tolerance = 1e-6
 max_iterations = 100

 for _ in range(max_iterations):
 vol_mid = (vol_low + vol_high) / 2
 price_diff = objective_function(vol_mid)

 if price_diff < tolerance:
 return vol_mid

 if objective_function(vol_low) < objective_function(vol_high):
 vol_high = vol_mid
 else:
 vol_low = vol_mid

 return vol_mid

 def _black_scholes_price(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float, option_type: str) -> Dict[str, float]:
 """Black-Scholes option pricing"""

 if time_to_expiry <= 0:
 # Handle expiration
 if option_type.lower() == 'call':
 return {'price': max(0, spot - strike)}
 else:
 return {'price': max(0, strike - spot)}

 d1 = (math.log(spot / strike) + (self.risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
 (volatility * math.sqrt(time_to_expiry))
 d2 = d1 - volatility * math.sqrt(time_to_expiry)

 if option_type.lower() == 'call':
 price = (spot * norm.cdf(d1) -
 strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
 else: # put
 price = (strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
 spot * norm.cdf(-d1))

 return {'price': max(0, price), 'd1': d1, 'd2': d2}

 def _analytical_greeks(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float, option_type: str) -> Dict[str, float]:
 """Calculate Greeks analytically"""

 if time_to_expiry <= 0:
 return {
 'delta': 1.0 if (option_type.lower() == 'call' and spot > strike) or
 (option_type.lower() == 'put' and spot < strike) else 0.0,
 'gamma': 0.0,
 'theta': 0.0,
 'vega': 0.0,
 'rho': 0.0
 }

 bs_result = self._black_scholes_price(spot, strike, volatility, time_to_expiry, option_type)
 d1, d2 = bs_result['d1'], bs_result['d2']

 # Delta
 if option_type.lower() == 'call':
 delta = norm.cdf(d1)
 else:
 delta = norm.cdf(d1) - 1

 # Gamma
 gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(time_to_expiry))

 # Theta
 term1 = -(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))
 if option_type.lower() == 'call':
 term2 = -self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
 theta = (term1 + term2) / 365 # Daily theta
 else:
 term2 = self.risk_free_rate * strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
 theta = (term1 + term2) / 365 # Daily theta

 # Vega
 vega = spot * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100 # 1% vol change

 # Rho
 if option_type.lower() == 'call':
 rho = strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
 else:
 rho = -strike * time_to_expiry * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100

 return {
 'delta': round(delta, 4),
 'gamma': round(gamma, 6),
 'theta': round(theta, 2),
 'vega': round(vega, 2),
 'rho': round(rho, 2)
 }

 def _quantlib_greeks(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float, option_type: str) -> Dict[str, float]:
 """Calculate Greeks using QuantLib"""
 if not QUANTLIB_AVAILABLE:
 return self._analytical_greeks(spot, strike, volatility, time_to_expiry, option_type)

 try:
 # Setup option
 expiry_date = self.calculation_date + timedelta(days=int(time_to_expiry * 365))
 ql_expiry = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)

 exercise = ql.EuropeanExercise(ql_expiry)
 payoff = ql.PlainVanillaPayoff(
 ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put,
 strike
 )

 option = ql.VanillaOption(payoff, exercise)

 # Setup market data
 spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
 vol_handle = ql.BlackVolTermStructureHandle(
 ql.BlackConstantVol(
 ql.Settings.instance().evaluationDate,
 ql.TARGET(),
 volatility,
 ql.Actual365Fixed()
 )
 )

 # Setup process
 process = ql.BlackScholesMertonProcess(
 spot_handle,
 ql.YieldTermStructureHandle(self.dividend_curve),
 ql.YieldTermStructureHandle(self.risk_free_curve),
 vol_handle
 )

 # Setup engine
 engine = ql.AnalyticEuropeanEngine(process)
 option.setPricingEngine(engine)

 # Calculate Greeks
 return {
 'delta': round(option.delta(), 4),
 'gamma': round(option.gamma(), 6),
 'theta': round(option.theta() / 365, 2), # Daily theta
 'vega': round(option.vega() / 100, 2), # 1% vol change
 'rho': round(option.rho() / 100, 2) # 1% rate change
 }

 except Exception as e:
 # Fallback to analytical
 return self._analytical_greeks(spot, strike, volatility, time_to_expiry, option_type)

 def _binomial_price(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float, option_type: str, steps: int = 100) -> Dict[str, float]:
 """Binomial tree pricing (QuantLib)"""
 if not QUANTLIB_AVAILABLE:
 return self._black_scholes_price(spot, strike, volatility, time_to_expiry, option_type)

 try:
 # Setup option
 expiry_date = self.calculation_date + timedelta(days=int(time_to_expiry * 365))
 ql_expiry = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)

 exercise = ql.AmericanExercise(
 ql.Settings.instance().evaluationDate,
 ql_expiry
 )
 payoff = ql.PlainVanillaPayoff(
 ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put,
 strike
 )

 option = ql.VanillaOption(payoff, exercise)

 # Setup market data
 spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
 vol_handle = ql.BlackVolTermStructureHandle(
 ql.BlackConstantVol(
 ql.Settings.instance().evaluationDate,
 ql.TARGET(),
 volatility,
 ql.Actual365Fixed()
 )
 )

 # Setup process
 process = ql.BlackScholesMertonProcess(
 spot_handle,
 ql.YieldTermStructureHandle(self.dividend_curve),
 ql.YieldTermStructureHandle(self.risk_free_curve),
 vol_handle
 )

 # Setup binomial engine
 engine = ql.BinomialVanillaEngine(process, "crr", steps)
 option.setPricingEngine(engine)

 return {'price': option.NPV()}

 except Exception as e:
 return self._black_scholes_price(spot, strike, volatility, time_to_expiry, option_type)

 def _monte_carlo_price(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float, option_type: str, num_simulations: int = 10000) -> Dict[str, float]:
 """Monte Carlo option pricing"""

 dt = time_to_expiry
 discount_factor = math.exp(-self.risk_free_rate * time_to_expiry)

 # Generate random stock price paths
 z = np.random.standard_normal(num_simulations)
 final_prices = spot * np.exp(
 (self.risk_free_rate - 0.5 * volatility ** 2) * dt + volatility * math.sqrt(dt) * z
 )

 # Calculate payoffs
 if option_type.lower() == 'call':
 payoffs = np.maximum(final_prices - strike, 0)
 else:
 payoffs = np.maximum(strike - final_prices, 0)

 # Calculate option price
 option_price = discount_factor * np.mean(payoffs)

 return {'price': option_price}

 def calculate_portfolio_greeks(self, positions: List[Dict]) -> Dict[str, float]:
 """Calculate portfolio-level Greeks"""

 portfolio_greeks = {
 'delta': 0.0,
 'gamma': 0.0,
 'theta': 0.0,
 'vega': 0.0,
 'rho': 0.0
 }

 for position in positions:
 quantity = position['quantity']
 greeks = position.get('greeks', {})

 for greek in portfolio_greeks:
 portfolio_greeks[greek] += quantity * greeks.get(greek, 0)

 return portfolio_greeks

 def calculate_option_strategies(self, strategy_type: str, spot: float,
 **kwargs) -> Dict[str, any]:
 """Calculate prices and Greeks for common option strategies"""

 if strategy_type == 'straddle':
 return self._calculate_straddle(spot, **kwargs)
 elif strategy_type == 'strangle':
 return self._calculate_strangle(spot, **kwargs)
 elif strategy_type == 'iron_condor':
 return self._calculate_iron_condor(spot, **kwargs)
 elif strategy_type == 'butterfly':
 return self._calculate_butterfly(spot, **kwargs)
 elif strategy_type == 'calendar_spread':
 return self._calculate_calendar_spread(spot, **kwargs)
 else:
 raise ValueError(f"Unknown strategy type: {strategy_type}")

 def _calculate_straddle(self, spot: float, strike: float, volatility: float,
 time_to_expiry: float) -> Dict[str, any]:
 """Calculate long straddle"""

 call_result = self.calculate_option_price(spot, strike, volatility, time_to_expiry, 'call')
 put_result = self.calculate_option_price(spot, strike, volatility, time_to_expiry, 'put')

 call_greeks = self.calculate_greeks(spot, strike, volatility, time_to_expiry, 'call')
 put_greeks = self.calculate_greeks(spot, strike, volatility, time_to_expiry, 'put')

 total_price = call_result['price'] + put_result['price']
 total_greeks = {
 'delta': call_greeks['delta'] + put_greeks['delta'],
 'gamma': call_greeks['gamma'] + put_greeks['gamma'],
 'theta': call_greeks['theta'] + put_greeks['theta'],
 'vega': call_greeks['vega'] + put_greeks['vega'],
 'rho': call_greeks['rho'] + put_greeks['rho']
 }

 return {
 'strategy': 'Long Straddle',
 'total_price': total_price,
 'greeks': total_greeks,
 'max_profit': float('inf'),
 'max_loss': total_price,
 'breakeven_points': [strike - total_price, strike + total_price]
 }

 def _calculate_strangle(self, spot: float, call_strike: float, put_strike: float,
 volatility: float, time_to_expiry: float) -> Dict[str, any]:
 """Calculate long strangle"""

 call_result = self.calculate_option_price(spot, call_strike, volatility, time_to_expiry, 'call')
 put_result = self.calculate_option_price(spot, put_strike, volatility, time_to_expiry, 'put')

 call_greeks = self.calculate_greeks(spot, call_strike, volatility, time_to_expiry, 'call')
 put_greeks = self.calculate_greeks(spot, put_strike, volatility, time_to_expiry, 'put')

 total_price = call_result['price'] + put_result['price']
 total_greeks = {
 'delta': call_greeks['delta'] + put_greeks['delta'],
 'gamma': call_greeks['gamma'] + put_greeks['gamma'],
 'theta': call_greeks['theta'] + put_greeks['theta'],
 'vega': call_greeks['vega'] + put_greeks['vega'],
 'rho': call_greeks['rho'] + put_greeks['rho']
 }

 return {
 'strategy': 'Long Strangle',
 'total_price': total_price,
 'greeks': total_greeks,
 'max_profit': float('inf'),
 'max_loss': total_price,
 'breakeven_points': [put_strike - total_price, call_strike + total_price]
 }

 def _calculate_iron_condor(self, spot: float, strikes: List[float], volatility: float,
 time_to_expiry: float) -> Dict[str, any]:
 """Calculate iron condor (short put spread + short call spread)"""

 if len(strikes) != 4:
 raise ValueError("Iron condor requires 4 strikes")

 strikes = sorted(strikes)
 put_strike_low, put_strike_high, call_strike_low, call_strike_high = strikes

 # Short put spread
 put_low = self.calculate_option_price(spot, put_strike_low, volatility, time_to_expiry, 'put')
 put_high = self.calculate_option_price(spot, put_strike_high, volatility, time_to_expiry, 'put')

 # Short call spread
 call_low = self.calculate_option_price(spot, call_strike_low, volatility, time_to_expiry, 'call')
 call_high = self.calculate_option_price(spot, call_strike_high, volatility, time_to_expiry, 'call')

 # Net credit received
 net_credit = (put_high['price'] - put_low['price']) + (call_low['price'] - call_high['price'])

 # Greeks
 put_low_greeks = self.calculate_greeks(spot, put_strike_low, volatility, time_to_expiry, 'put')
 put_high_greeks = self.calculate_greeks(spot, put_strike_high, volatility, time_to_expiry, 'put')
 call_low_greeks = self.calculate_greeks(spot, call_strike_low, volatility, time_to_expiry, 'call')
 call_high_greeks = self.calculate_greeks(spot, call_strike_high, volatility, time_to_expiry, 'call')

 total_greeks = {}
 for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
 total_greeks[greek] = (
 -put_low_greeks[greek] + put_high_greeks[greek] +
 call_low_greeks[greek] - call_high_greeks[greek]
 )

 max_profit = net_credit
 max_loss = max(put_strike_high - put_strike_low, call_strike_high - call_strike_low) - net_credit

 return {
 'strategy': 'Iron Condor',
 'net_credit': net_credit,
 'greeks': total_greeks,
 'max_profit': max_profit,
 'max_loss': max_loss,
 'profit_zone': [put_strike_high, call_strike_low]
 }

 def _calculate_butterfly(self, spot: float, center_strike: float, wing_width: float,
 volatility: float, time_to_expiry: float, option_type: str = 'call') -> Dict[str, any]:
 """Calculate butterfly spread"""

 lower_strike = center_strike - wing_width
 upper_strike = center_strike + wing_width

 # Long 1 lower strike, short 2 center strikes, long 1 upper strike
 lower_option = self.calculate_option_price(spot, lower_strike, volatility, time_to_expiry, option_type)
 center_option = self.calculate_option_price(spot, center_strike, volatility, time_to_expiry, option_type)
 upper_option = self.calculate_option_price(spot, upper_strike, volatility, time_to_expiry, option_type)

 net_debit = lower_option['price'] - 2 * center_option['price'] + upper_option['price']

 # Greeks
 lower_greeks = self.calculate_greeks(spot, lower_strike, volatility, time_to_expiry, option_type)
 center_greeks = self.calculate_greeks(spot, center_strike, volatility, time_to_expiry, option_type)
 upper_greeks = self.calculate_greeks(spot, upper_strike, volatility, time_to_expiry, option_type)

 total_greeks = {}
 for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
 total_greeks[greek] = (
 lower_greeks[greek] - 2 * center_greeks[greek] + upper_greeks[greek]
 )

 return {
 'strategy': f'{option_type.title()} Butterfly',
 'net_debit': net_debit,
 'greeks': total_greeks,
 'max_profit': wing_width - net_debit,
 'max_loss': net_debit,
 'breakeven_points': [center_strike - (wing_width - net_debit), center_strike + (wing_width - net_debit)]
 }

 def _calculate_calendar_spread(self, spot: float, strike: float,
 short_vol: float, long_vol: float,
 short_tte: float, long_tte: float,
 option_type: str = 'call') -> Dict[str, any]:
 """Calculate calendar spread"""

 short_option = self.calculate_option_price(spot, strike, short_vol, short_tte, option_type)
 long_option = self.calculate_option_price(spot, strike, long_vol, long_tte, option_type)

 net_debit = long_option['price'] - short_option['price']

 # Greeks
 short_greeks = self.calculate_greeks(spot, strike, short_vol, short_tte, option_type)
 long_greeks = self.calculate_greeks(spot, strike, long_vol, long_tte, option_type)

 total_greeks = {}
 for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
 total_greeks[greek] = long_greeks[greek] - short_greeks[greek]

 return {
 'strategy': f'{option_type.title()} Calendar Spread',
 'net_debit': net_debit,
 'greeks': total_greeks,
 'max_loss': net_debit,
 'profit_target': strike # Maximum profit typically near the strike at short expiration
 }