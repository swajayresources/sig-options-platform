"""
Real-Time Market Data Integration System

Sophisticated market data handling for options trading with support for
multiple data sources, real-time streaming, and options chain management.
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import queue
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class OptionQuote:
 """Individual option quote"""
 symbol: str
 underlying: str
 option_type: str # 'C' or 'P'
 strike: float
 expiry: datetime
 bid: float
 ask: float
 bid_size: int
 ask_size: int
 last: float
 volume: int
 open_interest: int
 implied_vol: float
 delta: float
 gamma: float
 theta: float
 vega: float
 timestamp: datetime

 @property
 def mid_price(self) -> float:
 return (self.bid + self.ask) / 2.0 if self.bid > 0 and self.ask > 0 else self.last

 @property
 def spread(self) -> float:
 return self.ask - self.bid if self.bid > 0 and self.ask > 0 else 0.0

 @property
 def spread_pct(self) -> float:
 mid = self.mid_price
 return (self.spread / mid * 100) if mid > 0 else 0.0

@dataclass
class UnderlyingQuote:
 """Underlying asset quote"""
 symbol: str
 bid: float
 ask: float
 last: float
 volume: int
 high: float
 low: float
 open: float
 prev_close: float
 timestamp: datetime

 @property
 def mid_price(self) -> float:
 return (self.bid + self.ask) / 2.0 if self.bid > 0 and self.ask > 0 else self.last

@dataclass
class OptionsChain:
 """Complete options chain for an underlying"""
 underlying: str
 expiries: List[datetime]
 strikes: List[float]
 calls: Dict[tuple, OptionQuote] # (expiry, strike) -> quote
 puts: Dict[tuple, OptionQuote] # (expiry, strike) -> quote
 underlying_quote: Optional[UnderlyingQuote] = None
 timestamp: datetime = field(default_factory=datetime.now)

 def get_option(self, expiry: datetime, strike: float, option_type: str) -> Optional[OptionQuote]:
 """Get specific option quote"""
 key = (expiry, strike)
 if option_type.upper() == 'C':
 return self.calls.get(key)
 elif option_type.upper() == 'P':
 return self.puts.get(key)
 return None

 def get_atm_options(self, expiry: datetime) -> Tuple[Optional[OptionQuote], Optional[OptionQuote]]:
 """Get at-the-money call and put for given expiry"""
 if not self.underlying_quote:
 return None, None

 spot = self.underlying_quote.mid_price
 closest_strike = min(self.strikes, key=lambda x: abs(x - spot))

 call = self.get_option(expiry, closest_strike, 'C')
 put = self.get_option(expiry, closest_strike, 'P')

 return call, put

 def get_strikes_for_expiry(self, expiry: datetime) -> List[float]:
 """Get all strikes available for given expiry"""
 available_strikes = []
 for exp, strike in self.calls.keys():
 if exp == expiry:
 available_strikes.append(strike)
 return sorted(list(set(available_strikes)))

class DataProvider(ABC):
 """Abstract base class for market data providers"""

 @abstractmethod
 async def connect(self):
 """Connect to data source"""
 pass

 @abstractmethod
 async def disconnect(self):
 """Disconnect from data source"""
 pass

 @abstractmethod
 async def subscribe_options_chain(self, symbol: str, callback: Callable):
 """Subscribe to options chain updates"""
 pass

 @abstractmethod
 async def subscribe_underlying(self, symbol: str, callback: Callable):
 """Subscribe to underlying quotes"""
 pass

 @abstractmethod
 async def get_historical_data(self, symbol: str, start_date: datetime,
 end_date: datetime) -> pd.DataFrame:
 """Get historical data"""
 pass

class SimulatedDataProvider(DataProvider):
 """Simulated market data for testing and development"""

 def __init__(self, volatility: float = 0.2, update_interval: float = 0.1):
 self.volatility = volatility
 self.update_interval = update_interval
 self.callbacks: Dict[str, List[Callable]] = {}
 self.underlying_prices: Dict[str, float] = {}
 self.is_connected = False
 self.simulation_task = None

 async def connect(self):
 """Connect to simulated data"""
 self.is_connected = True
 self.simulation_task = asyncio.create_task(self._simulate_market_data())
 logger.info("Connected to simulated data provider")

 async def disconnect(self):
 """Disconnect from simulated data"""
 self.is_connected = False
 if self.simulation_task:
 self.simulation_task.cancel()
 logger.info("Disconnected from simulated data provider")

 async def subscribe_options_chain(self, symbol: str, callback: Callable):
 """Subscribe to simulated options chain"""
 if symbol not in self.callbacks:
 self.callbacks[symbol] = []
 self.callbacks[symbol].append(callback)

 # Initialize underlying price
 if symbol not in self.underlying_prices:
 self.underlying_prices[symbol] = 100.0 # Default price

 async def subscribe_underlying(self, symbol: str, callback: Callable):
 """Subscribe to simulated underlying quotes"""
 await self.subscribe_options_chain(symbol, callback)

 async def _simulate_market_data(self):
 """Simulate real-time market data"""
 try:
 while self.is_connected:
 for symbol in self.callbacks.keys():
 # Update underlying price using GBM
 dt = self.update_interval / (365 * 24 * 3600) # Convert to years
 dW = np.random.normal(0, np.sqrt(dt))

 current_price = self.underlying_prices[symbol]
 new_price = current_price * np.exp(-0.5 * self.volatility**2 * dt +
 self.volatility * dW)
 self.underlying_prices[symbol] = new_price

 # Generate options chain
 chain = self._generate_options_chain(symbol, new_price)

 # Call callbacks
 for callback in self.callbacks[symbol]:
 try:
 await callback(chain)
 except Exception as e:
 logger.error(f"Error in callback for {symbol}: {e}")

 await asyncio.sleep(self.update_interval)

 except asyncio.CancelledError:
 logger.info("Market data simulation cancelled")

 def _generate_options_chain(self, symbol: str, spot_price: float) -> OptionsChain:
 """Generate simulated options chain"""
 now = datetime.now()

 # Generate expiries (weekly and monthly)
 expiries = []
 current_date = now
 for i in range(8): # Next 8 weeks
 friday = current_date + timedelta(days=(4 - current_date.weekday()) % 7)
 expiries.append(friday)
 current_date = friday + timedelta(days=1)

 # Generate strikes around current price
 strikes = []
 for i in range(-10, 11): # 21 strikes
 strike = spot_price * (1 + i * 0.05) # 5% intervals
 strikes.append(round(strike, 2))

 # Create underlying quote
 underlying_quote = UnderlyingQuote(
 symbol=symbol,
 bid=spot_price * 0.9999,
 ask=spot_price * 1.0001,
 last=spot_price,
 volume=1000000,
 high=spot_price * 1.02,
 low=spot_price * 0.98,
 open=spot_price * 1.01,
 prev_close=spot_price * 0.995,
 timestamp=now
 )

 calls = {}
 puts = {}

 for expiry in expiries:
 time_to_expiry = (expiry - now).days / 365.0
 if time_to_expiry <= 0:
 continue

 for strike in strikes:
 # Black-Scholes pricing for simulation
 call_price, put_price = self._black_scholes_price(
 spot_price, strike, time_to_expiry, 0.02, self.volatility
 )

 # Add noise to prices
 noise_factor = 0.02
 call_price *= (1 + np.random.normal(0, noise_factor))
 put_price *= (1 + np.random.normal(0, noise_factor))

 # Calculate Greeks (simplified)
 d1 = (np.log(spot_price / strike) + (0.02 + 0.5 * self.volatility**2) * time_to_expiry) / (self.volatility * np.sqrt(time_to_expiry))
 delta_call = self._norm_cdf(d1)
 delta_put = delta_call - 1.0
 gamma = self._norm_pdf(d1) / (spot_price * self.volatility * np.sqrt(time_to_expiry))
 theta_call = -(spot_price * self._norm_pdf(d1) * self.volatility / (2 * np.sqrt(time_to_expiry)) +
 0.02 * strike * np.exp(-0.02 * time_to_expiry) * self._norm_cdf(d1 - self.volatility * np.sqrt(time_to_expiry)))
 vega = spot_price * self._norm_pdf(d1) * np.sqrt(time_to_expiry)

 # Create call quote
 call_quote = OptionQuote(
 symbol=f"{symbol}_{expiry.strftime('%Y%m%d')}_C_{strike}",
 underlying=symbol,
 option_type='C',
 strike=strike,
 expiry=expiry,
 bid=call_price * 0.98,
 ask=call_price * 1.02,
 bid_size=10,
 ask_size=10,
 last=call_price,
 volume=100,
 open_interest=1000,
 implied_vol=self.volatility,
 delta=delta_call,
 gamma=gamma,
 theta=theta_call / 365,
 vega=vega / 100,
 timestamp=now
 )

 # Create put quote
 put_quote = OptionQuote(
 symbol=f"{symbol}_{expiry.strftime('%Y%m%d')}_P_{strike}",
 underlying=symbol,
 option_type='P',
 strike=strike,
 expiry=expiry,
 bid=put_price * 0.98,
 ask=put_price * 1.02,
 bid_size=10,
 ask_size=10,
 last=put_price,
 volume=100,
 open_interest=1000,
 implied_vol=self.volatility,
 delta=delta_put,
 gamma=gamma,
 theta=(theta_call + 0.02 * strike * np.exp(-0.02 * time_to_expiry)) / 365,
 vega=vega / 100,
 timestamp=now
 )

 calls[(expiry, strike)] = call_quote
 puts[(expiry, strike)] = put_quote

 return OptionsChain(
 underlying=symbol,
 expiries=expiries,
 strikes=strikes,
 calls=calls,
 puts=puts,
 underlying_quote=underlying_quote,
 timestamp=now
 )

 def _black_scholes_price(self, S: float, K: float, T: float, r: float, vol: float) -> Tuple[float, float]:
 """Calculate Black-Scholes call and put prices"""
 if T <= 0:
 return max(S - K, 0), max(K - S, 0)

 d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
 d2 = d1 - vol * np.sqrt(T)

 call_price = S * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)
 put_price = K * np.exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

 return max(call_price, 0), max(put_price, 0)

 def _norm_cdf(self, x: float) -> float:
 """Standard normal CDF approximation"""
 return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

 def _norm_pdf(self, x: float) -> float:
 """Standard normal PDF"""
 return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

 async def get_historical_data(self, symbol: str, start_date: datetime,
 end_date: datetime) -> pd.DataFrame:
 """Generate simulated historical data"""
 dates = pd.date_range(start_date, end_date, freq='D')
 n_days = len(dates)

 # Generate price series using GBM
 returns = np.random.normal(0.0001, self.volatility / np.sqrt(252), n_days)
 prices = [100.0] # Starting price

 for ret in returns[1:]:
 prices.append(prices[-1] * np.exp(ret))

 return pd.DataFrame({
 'date': dates,
 'symbol': symbol,
 'open': prices,
 'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
 'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
 'close': prices,
 'volume': np.random.randint(100000, 10000000, n_days)
 })

class RealTimeDataManager:
 """Manages real-time market data feeds and subscriptions"""

 def __init__(self, provider: DataProvider):
 self.provider = provider
 self.subscriptions: Dict[str, List[Callable]] = {}
 self.latest_chains: Dict[str, OptionsChain] = {}
 self.data_quality_metrics: Dict[str, Dict] = {}
 self.is_running = False

 async def start(self):
 """Start the data manager"""
 await self.provider.connect()
 self.is_running = True
 logger.info("Real-time data manager started")

 async def stop(self):
 """Stop the data manager"""
 await self.provider.disconnect()
 self.is_running = False
 logger.info("Real-time data manager stopped")

 async def subscribe_to_options_chain(self, symbol: str, callback: Callable):
 """Subscribe to options chain updates"""
 if symbol not in self.subscriptions:
 self.subscriptions[symbol] = []
 await self.provider.subscribe_options_chain(symbol, self._handle_chain_update)

 self.subscriptions[symbol].append(callback)
 logger.info(f"Subscribed to options chain for {symbol}")

 async def _handle_chain_update(self, chain: OptionsChain):
 """Handle incoming options chain update"""
 symbol = chain.underlying
 self.latest_chains[symbol] = chain

 # Update data quality metrics
 self._update_data_quality(symbol, chain)

 # Notify subscribers
 if symbol in self.subscriptions:
 for callback in self.subscriptions[symbol]:
 try:
 await callback(chain)
 except Exception as e:
 logger.error(f"Error in chain update callback for {symbol}: {e}")

 def _update_data_quality(self, symbol: str, chain: OptionsChain):
 """Update data quality metrics"""
 if symbol not in self.data_quality_metrics:
 self.data_quality_metrics[symbol] = {
 'last_update': None,
 'update_count': 0,
 'avg_spread': 0.0,
 'missing_quotes': 0
 }

 metrics = self.data_quality_metrics[symbol]
 metrics['last_update'] = datetime.now()
 metrics['update_count'] += 1

 # Calculate average spread
 spreads = []
 missing_count = 0

 for quote in list(chain.calls.values()) + list(chain.puts.values()):
 if quote.bid > 0 and quote.ask > 0:
 spreads.append(quote.spread_pct)
 else:
 missing_count += 1

 if spreads:
 metrics['avg_spread'] = np.mean(spreads)
 metrics['missing_quotes'] = missing_count

 def get_latest_chain(self, symbol: str) -> Optional[OptionsChain]:
 """Get latest options chain for symbol"""
 return self.latest_chains.get(symbol)

 def get_data_quality_report(self) -> Dict:
 """Get data quality report for all symbols"""
 return self.data_quality_metrics.copy()

class HistoricalDataManager:
 """Manages historical data storage and retrieval"""

 def __init__(self, provider: DataProvider):
 self.provider = provider
 self.cache: Dict[str, pd.DataFrame] = {}

 async def get_historical_prices(self, symbol: str, start_date: datetime,
 end_date: datetime, force_refresh: bool = False) -> pd.DataFrame:
 """Get historical price data"""
 cache_key = f"{symbol}_{start_date}_{end_date}"

 if not force_refresh and cache_key in self.cache:
 return self.cache[cache_key]

 data = await self.provider.get_historical_data(symbol, start_date, end_date)
 self.cache[cache_key] = data

 return data

 def calculate_realized_volatility(self, symbol: str, days: int = 30) -> float:
 """Calculate realized volatility from historical data"""
 end_date = datetime.now()
 start_date = end_date - timedelta(days=days)

 try:
 # Get cached data if available
 cache_key = f"{symbol}_{start_date}_{end_date}"
 if cache_key in self.cache:
 data = self.cache[cache_key]
 else:
 return 0.2 # Default volatility

 if len(data) < 2:
 return 0.2

 # Calculate returns
 prices = data['close'].values
 returns = np.diff(np.log(prices))

 # Annualized volatility
 vol = np.std(returns) * np.sqrt(252)
 return float(vol)

 except Exception as e:
 logger.error(f"Error calculating realized volatility for {symbol}: {e}")
 return 0.2

class MarketDataAggregator:
 """Aggregates and normalizes data from multiple sources"""

 def __init__(self, providers: List[DataProvider]):
 self.providers = providers
 self.weights = [1.0 / len(providers)] * len(providers) # Equal weighting
 self.aggregated_chains: Dict[str, OptionsChain] = {}

 def set_provider_weights(self, weights: List[float]):
 """Set weights for provider data aggregation"""
 if len(weights) != len(self.providers):
 raise ValueError("Number of weights must match number of providers")
 if abs(sum(weights) - 1.0) > 1e-6:
 raise ValueError("Weights must sum to 1.0")

 self.weights = weights

 async def get_aggregated_chain(self, symbol: str) -> Optional[OptionsChain]:
 """Get aggregated options chain from all providers"""
 chains = []

 for provider in self.providers:
 try:
 # This would need to be implemented based on provider API
 # For now, return None as placeholder
 pass
 except Exception as e:
 logger.error(f"Error getting chain from provider: {e}")

 if not chains:
 return None

 # Aggregate the chains (simplified implementation)
 return self._aggregate_chains(chains)

 def _aggregate_chains(self, chains: List[OptionsChain]) -> OptionsChain:
 """Aggregate multiple options chains into one"""
 if not chains:
 return None

 # Use first chain as base
 base_chain = chains[0]

 # For simplicity, just return the first chain
 # In practice, would aggregate quotes with weighted averages
 return base_chain