"""
Options Data Provider
Real-time market data and options chain provider
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
from typing import Dict, List, Tuple, Optional
import asyncio
import streamlit as st

class OptionsDataProvider:
 """Provides real-time options and market data"""

 def __init__(self):
 self.api_key = st.secrets.get("api_key", "demo")
 self.cache_duration = 60 # seconds
 self.data_cache = {}

 def get_market_status(self) -> str:
 """Get current market status"""
 now = datetime.now()
 market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
 market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

 # Check if it's a weekday and within market hours
 if now.weekday() < 5 and market_open <= now <= market_close:
 return "OPEN"
 else:
 return "CLOSED"

 def get_current_price(self, symbol: str) -> float:
 """Get current stock price"""
 try:
 ticker = yf.Ticker(symbol)
 data = ticker.history(period="1d", interval="1m")
 if not data.empty:
 return float(data['Close'].iloc[-1])
 else:
 # Fallback to demo data
 return self._get_demo_price(symbol)
 except Exception:
 return self._get_demo_price(symbol)

 def get_expiration_dates(self, symbol: str) -> List[str]:
 """Get available expiration dates for options"""
 try:
 ticker = yf.Ticker(symbol)
 expirations = ticker.options
 if expirations:
 return list(expirations)
 else:
 return self._get_demo_expirations()
 except Exception:
 return self._get_demo_expirations()

 def get_options_chain(self, symbol: str, expiration: str, chain_type: str = "Both") -> pd.DataFrame:
 """Get options chain data"""
 cache_key = f"{symbol}_{expiration}_{chain_type}"

 # Check cache first
 if cache_key in self.data_cache:
 cached_time, cached_data = self.data_cache[cache_key]
 if (datetime.now() - cached_time).seconds < self.cache_duration:
 return cached_data

 try:
 ticker = yf.Ticker(symbol)
 options_data = ticker.option_chain(expiration)

 # Process calls and puts
 calls_df = self._process_options_data(options_data.calls, "CALL", symbol)
 puts_df = self._process_options_data(options_data.puts, "PUT", symbol)

 # Combine based on chain_type
 if chain_type == "Calls Only":
 result_df = calls_df
 elif chain_type == "Puts Only":
 result_df = puts_df
 else: # Both
 result_df = pd.concat([calls_df, puts_df], ignore_index=True)

 # Cache the result
 self.data_cache[cache_key] = (datetime.now(), result_df)
 return result_df

 except Exception as e:
 # Return demo data on error
 return self._get_demo_options_chain(symbol, expiration, chain_type)

 def _process_options_data(self, options_df: pd.DataFrame, option_type: str, symbol: str) -> pd.DataFrame:
 """Process raw options data and calculate Greeks"""
 if options_df.empty:
 return pd.DataFrame()

 processed_data = []
 current_price = self.get_current_price(symbol)

 for _, row in options_df.iterrows():
 strike = row['strike']
 last_price = row.get('lastPrice', 0)
 bid = row.get('bid', 0)
 ask = row.get('ask', 0)
 volume = row.get('volume', 0)
 open_interest = row.get('openInterest', 0)
 implied_vol = row.get('impliedVolatility', 0.2)

 # Calculate Greeks (simplified)
 greeks = self._calculate_greeks(current_price, strike, implied_vol, option_type)

 processed_data.append({
 'Symbol': symbol,
 'Type': option_type,
 'Strike': strike,
 'Last': last_price,
 'Bid': bid,
 'Ask': ask,
 'Volume': volume,
 'Open_Interest': open_interest,
 'IV': implied_vol * 100, # Convert to percentage
 'Delta': greeks['delta'],
 'Gamma': greeks['gamma'],
 'Theta': greeks['theta'],
 'Vega': greeks['vega'],
 'Intrinsic': max(0, current_price - strike if option_type == 'CALL' else strike - current_price),
 'Time_Value': max(0, last_price - max(0, current_price - strike if option_type == 'CALL' else strike - current_price))
 })

 return pd.DataFrame(processed_data)

 def _calculate_greeks(self, spot: float, strike: float, vol: float, option_type: str,
 risk_free_rate: float = 0.05, time_to_expiry: float = 0.25) -> Dict[str, float]:
 """Calculate option Greeks using Black-Scholes"""
 from scipy.stats import norm
 import math

 # Prevent division by zero
 if vol <= 0:
 vol = 0.01
 if time_to_expiry <= 0:
 time_to_expiry = 1/365

 d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * vol ** 2) * time_to_expiry) / (vol * math.sqrt(time_to_expiry))
 d2 = d1 - vol * math.sqrt(time_to_expiry)

 if option_type == 'CALL':
 delta = norm.cdf(d1)
 theta = -(spot * norm.pdf(d1) * vol) / (2 * math.sqrt(time_to_expiry)) - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
 else: # PUT
 delta = norm.cdf(d1) - 1
 theta = -(spot * norm.pdf(d1) * vol) / (2 * math.sqrt(time_to_expiry)) + risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)

 gamma = norm.pdf(d1) / (spot * vol * math.sqrt(time_to_expiry))
 vega = spot * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100 # Divide by 100 for 1% vol change

 return {
 'delta': round(delta, 4),
 'gamma': round(gamma, 4),
 'theta': round(theta / 365, 2), # Daily theta
 'vega': round(vega, 2)
 }

 def get_volatility_surface(self, symbol: str) -> Optional[Tuple[List[float], List[int], np.ndarray]]:
 """Get volatility surface data"""
 try:
 # Get multiple expirations
 expirations = self.get_expiration_dates(symbol)[:5] # First 5 expirations
 current_price = self.get_current_price(symbol)

 # Define strike range around current price
 strikes = [current_price * mult for mult in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]]
 strikes = [round(strike, 0) for strike in strikes]

 # Calculate days to expiration
 days_to_expiry = []
 vol_matrix = []

 for exp_date_str in expirations:
 exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
 days = (exp_date - datetime.now()).days
 days_to_expiry.append(days)

 # Get implied volatilities for this expiration
 vol_row = []
 for strike in strikes:
 # Try to get actual IV data, fallback to model
 iv = self._get_implied_volatility(symbol, strike, exp_date_str)
 vol_row.append(iv)

 vol_matrix.append(vol_row)

 return strikes, days_to_expiry, np.array(vol_matrix)

 except Exception:
 return self._get_demo_volatility_surface(symbol)

 def _get_implied_volatility(self, symbol: str, strike: float, expiration: str) -> float:
 """Get implied volatility for specific option"""
 try:
 ticker = yf.Ticker(symbol)
 options_data = ticker.option_chain(expiration)

 # Look for the strike in calls data
 calls = options_data.calls
 if not calls.empty:
 matching_strike = calls[calls['strike'] == strike]
 if not matching_strike.empty:
 iv = matching_strike.iloc[0].get('impliedVolatility', 0.2)
 return iv * 100 # Convert to percentage

 # Fallback to model-based IV
 return self._model_implied_volatility(symbol, strike, expiration)

 except Exception:
 return self._model_implied_volatility(symbol, strike, expiration)

 def _model_implied_volatility(self, symbol: str, strike: float, expiration: str) -> float:
 """Model-based implied volatility calculation"""
 current_price = self.get_current_price(symbol)

 # Simple volatility smile model
 moneyness = strike / current_price
 base_vol = 25.0 # Base volatility in %

 # Add skew (puts more expensive)
 if moneyness < 1.0: # ITM puts / OTM calls
 skew_adjustment = (1.0 - moneyness) * 20
 else: # OTM puts / ITM calls
 skew_adjustment = (moneyness - 1.0) * 5

 # Add term structure effect
 exp_date = datetime.strptime(expiration, '%Y-%m-%d')
 days_to_expiry = (exp_date - datetime.now()).days
 term_adjustment = max(0, (30 - days_to_expiry) / 30 * 5)

 total_vol = base_vol + skew_adjustment + term_adjustment + np.random.normal(0, 2)
 return max(5.0, min(80.0, total_vol)) # Clamp between 5% and 80%

 def get_options_flow(self, timeframe: str, min_premium: float, flow_type: str) -> pd.DataFrame:
 """Get options flow data"""
 # Generate demo flow data for now
 num_flows = {'5m': 20, '15m': 50, '1h': 100, '1d': 300}.get(timeframe, 50)

 symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'META']
 option_types = ['CALL', 'PUT']
 sentiments = ['bullish', 'bearish', 'neutral']

 flow_data = []
 base_time = datetime.now()

 for i in range(num_flows):
 symbol = np.random.choice(symbols)
 current_price = self.get_current_price(symbol)

 # Generate flow data
 flow = {
 'time': base_time - timedelta(minutes=np.random.randint(0, {'5m': 5, '15m': 15, '1h': 60, '1d': 1440}[timeframe])),
 'symbol': symbol,
 'type': np.random.choice(option_types),
 'strike': current_price * np.random.uniform(0.9, 1.1),
 'premium': np.random.uniform(min_premium, min_premium * 5),
 'volume': np.random.randint(10, 1000),
 'sentiment': np.random.choice(sentiments)
 }

 # Filter by flow type
 if flow_type != "All":
 if flow_type == "Calls" and flow['type'] != 'CALL':
 continue
 if flow_type == "Puts" and flow['type'] != 'PUT':
 continue
 if flow_type == "Unusual" and flow['premium'] < min_premium * 2:
 continue

 flow_data.append(flow)

 return pd.DataFrame(flow_data)

 # Demo data methods for fallback
 def _get_demo_price(self, symbol: str) -> float:
 """Get demo price for symbol"""
 demo_prices = {
 'AAPL': 175.50,
 'MSFT': 335.20,
 'GOOGL': 135.80,
 'TSLA': 245.60,
 'SPY': 445.30,
 'QQQ': 365.40,
 'NVDA': 485.70,
 'META': 325.90
 }
 return demo_prices.get(symbol, 100.0) * (1 + np.random.uniform(-0.02, 0.02))

 def _get_demo_expirations(self) -> List[str]:
 """Get demo expiration dates"""
 base_date = datetime.now()
 expirations = []

 # Weekly expirations for next 8 weeks
 for i in range(1, 9):
 # Find next Friday
 days_ahead = (4 - base_date.weekday()) % 7 # 4 = Friday
 if days_ahead == 0:
 days_ahead = 7
 next_friday = base_date + timedelta(days=days_ahead + (i-1)*7)
 expirations.append(next_friday.strftime('%Y-%m-%d'))

 # Monthly expirations
 for i in range(2, 7):
 # Third Friday of month
 next_month = base_date.replace(month=base_date.month + i if base_date.month + i <= 12 else base_date.month + i - 12,
 year=base_date.year if base_date.month + i <= 12 else base_date.year + 1)
 third_friday = self._get_third_friday(next_month.year, next_month.month)
 expirations.append(third_friday.strftime('%Y-%m-%d'))

 return sorted(list(set(expirations)))

 def _get_third_friday(self, year: int, month: int) -> datetime:
 """Get third Friday of given month"""
 first_day = datetime(year, month, 1)
 first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
 return first_friday + timedelta(days=14)

 def _get_demo_options_chain(self, symbol: str, expiration: str, chain_type: str) -> pd.DataFrame:
 """Generate demo options chain"""
 current_price = self.get_current_price(symbol)

 # Generate strikes around current price
 strikes = []
 for mult in np.arange(0.85, 1.16, 0.05):
 strikes.append(round(current_price * mult, 0))

 data = []

 for strike in strikes:
 for option_type in (['CALL', 'PUT'] if chain_type == "Both"
 else ['CALL'] if chain_type == "Calls Only"
 else ['PUT']):

 # Calculate theoretical price and Greeks
 vol = np.random.uniform(0.15, 0.45)
 greeks = self._calculate_greeks(current_price, strike, vol, option_type)

 # Generate realistic bid/ask/last
 intrinsic = max(0, current_price - strike if option_type == 'CALL' else strike - current_price)
 time_value = max(0.05, vol * current_price * 0.1 * np.random.uniform(0.5, 1.5))
 theoretical_price = intrinsic + time_value

 bid = max(0.01, theoretical_price - np.random.uniform(0.05, 0.25))
 ask = theoretical_price + np.random.uniform(0.05, 0.25)
 last = np.random.uniform(bid, ask)

 data.append({
 'Symbol': symbol,
 'Type': option_type,
 'Strike': strike,
 'Last': round(last, 2),
 'Bid': round(bid, 2),
 'Ask': round(ask, 2),
 'Volume': np.random.randint(0, 1000),
 'Open_Interest': np.random.randint(0, 5000),
 'IV': round(vol * 100, 1),
 'Delta': greeks['delta'],
 'Gamma': greeks['gamma'],
 'Theta': greeks['theta'],
 'Vega': greeks['vega'],
 'Intrinsic': round(intrinsic, 2),
 'Time_Value': round(time_value, 2)
 })

 return pd.DataFrame(data)

 def _get_demo_volatility_surface(self, symbol: str) -> Tuple[List[float], List[int], np.ndarray]:
 """Generate demo volatility surface"""
 current_price = self.get_current_price(symbol)

 # Define strikes and expirations
 strikes = [current_price * mult for mult in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]]
 strikes = [round(strike, 0) for strike in strikes]
 days_to_expiry = [7, 14, 21, 30, 60, 90]

 # Generate volatility surface with realistic patterns
 vol_matrix = []
 for days in days_to_expiry:
 vol_row = []
 for strike in strikes:
 moneyness = strike / current_price

 # Base volatility with term structure
 base_vol = 0.25 + (30 - days) / 365 * 0.05

 # Add volatility skew
 if moneyness < 1.0: # ITM puts / OTM calls
 skew = (1.0 - moneyness) * 0.3
 else: # OTM puts / ITM calls
 skew = (moneyness - 1.0) * 0.1

 total_vol = (base_vol + skew + np.random.normal(0, 0.02)) * 100
 vol_row.append(max(10, min(60, total_vol)))

 vol_matrix.append(vol_row)

 return strikes, days_to_expiry, np.array(vol_matrix)