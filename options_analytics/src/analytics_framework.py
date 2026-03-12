"""
Comprehensive Options Trading Analytics Framework
Core analytics engine for options portfolio monitoring and analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketData:
 symbol: str
 underlying_price: float
 bid: float
 ask: float
 last: float
 volume: int
 open_interest: int
 implied_volatility: float
 delta: float
 gamma: float
 theta: float
 vega: float
 rho: float
 time_to_expiry: float
 strike: float
 option_type: str
 timestamp: datetime

@dataclass
class Position:
 symbol: str
 quantity: int
 average_price: float
 current_price: float
 market_value: float
 unrealized_pnl: float
 realized_pnl: float
 delta: float
 gamma: float
 theta: float
 vega: float
 rho: float

@dataclass
class PortfolioGreeks:
 total_delta: float
 total_gamma: float
 total_theta: float
 total_vega: float
 total_rho: float
 net_exposure: float
 leverage_ratio: float
 timestamp: datetime

@dataclass
class VolatilitySurface:
 underlying: str
 strikes: List[float]
 expiries: List[datetime]
 implied_vols: np.ndarray
 bid_iv: np.ndarray
 ask_iv: np.ndarray
 timestamp: datetime

@dataclass
class OptionsFlow:
 symbol: str
 strike: float
 expiry: datetime
 option_type: str
 volume: int
 open_interest: int
 premium: float
 direction: str
 size_category: str
 unusual_activity: bool
 timestamp: datetime

class AnalyticsType(Enum):
 PORTFOLIO_ANALYTICS = "portfolio_analytics"
 MARKET_ANALYSIS = "market_analysis"
 TRADING_SIGNALS = "trading_signals"
 RISK_METRICS = "risk_metrics"

class OptionsAnalyticsFramework:
 """Core analytics framework for comprehensive options analysis"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)
 self.positions: Dict[str, Position] = {}
 self.market_data: Dict[str, MarketData] = {}
 self.vol_surfaces: Dict[str, VolatilitySurface] = {}
 self.options_flows: List[OptionsFlow] = []
 self.analytics_cache: Dict[str, Any] = {}
 self.analytics_engine = AnalyticsEngine(config)

 async def initialize(self):
 """Initialize analytics framework"""
 self.logger.info("Initializing Options Analytics Framework")
 await self.analytics_engine.initialize()

 def update_positions(self, positions: Dict[str, Position]):
 """Update portfolio positions"""
 self.positions = positions
 self._invalidate_cache(['portfolio_analytics'])

 def update_market_data(self, market_data: Dict[str, MarketData]):
 """Update market data"""
 self.market_data = market_data
 self._invalidate_cache(['market_analysis', 'trading_signals'])

 def update_volatility_surfaces(self, vol_surfaces: Dict[str, VolatilitySurface]):
 """Update volatility surfaces"""
 self.vol_surfaces = vol_surfaces
 self._invalidate_cache(['vol_analysis'])

 def add_options_flow(self, flow: OptionsFlow):
 """Add options flow data"""
 self.options_flows.append(flow)
 if len(self.options_flows) > self.config.get('max_flow_history', 10000):
 self.options_flows = self.options_flows[-self.config.get('max_flow_history', 10000):]

 def _invalidate_cache(self, cache_keys: List[str]):
 """Invalidate specific cache entries"""
 for key in cache_keys:
 if key in self.analytics_cache:
 del self.analytics_cache[key]

class AnalyticsEngine:
 """Core analytics computation engine"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)
 self.portfolio_analyzer = PortfolioAnalyzer(config)
 self.market_analyzer = MarketAnalyzer(config)
 self.signal_generator = SignalGenerator(config)
 self.risk_analyzer = RiskAnalyzer(config)

 async def initialize(self):
 """Initialize analytics components"""
 await asyncio.gather(
 self.portfolio_analyzer.initialize(),
 self.market_analyzer.initialize(),
 self.signal_generator.initialize(),
 self.risk_analyzer.initialize()
 )

class PortfolioAnalyzer:
 """Portfolio analytics and monitoring"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize portfolio analyzer"""
 self.logger.info("Initializing Portfolio Analyzer")

 def calculate_portfolio_greeks(self, positions: Dict[str, Position]) -> PortfolioGreeks:
 """Calculate aggregated portfolio Greeks"""
 total_delta = sum(pos.delta * pos.quantity for pos in positions.values())
 total_gamma = sum(pos.gamma * pos.quantity for pos in positions.values())
 total_theta = sum(pos.theta * pos.quantity for pos in positions.values())
 total_vega = sum(pos.vega * pos.quantity for pos in positions.values())
 total_rho = sum(pos.rho * pos.quantity for pos in positions.values())

 net_exposure = sum(abs(pos.market_value) for pos in positions.values())
 total_value = sum(pos.market_value for pos in positions.values())
 leverage_ratio = net_exposure / max(abs(total_value), 1.0)

 return PortfolioGreeks(
 total_delta=total_delta,
 total_gamma=total_gamma,
 total_theta=total_theta,
 total_vega=total_vega,
 total_rho=total_rho,
 net_exposure=net_exposure,
 leverage_ratio=leverage_ratio,
 timestamp=datetime.now()
 )

 def calculate_pnl_attribution(self, positions: Dict[str, Position],
 prev_positions: Dict[str, Position],
 market_data: Dict[str, MarketData],
 time_elapsed: float) -> Dict[str, float]:
 """Calculate detailed P&L attribution"""
 attribution = {
 'delta_pnl': 0.0,
 'gamma_pnl': 0.0,
 'theta_pnl': 0.0,
 'vega_pnl': 0.0,
 'rho_pnl': 0.0,
 'trading_pnl': 0.0,
 'total_pnl': 0.0
 }

 for symbol, position in positions.items():
 if symbol not in prev_positions or symbol not in market_data:
 continue

 prev_pos = prev_positions[symbol]
 md = market_data[symbol]

 price_change = md.last - prev_pos.current_price

 delta_pnl = position.quantity * prev_pos.delta * price_change
 gamma_pnl = 0.5 * position.quantity * prev_pos.gamma * (price_change ** 2)
 theta_pnl = position.quantity * prev_pos.theta * time_elapsed
 vega_pnl = position.quantity * prev_pos.vega * (md.implied_volatility -
 self._get_prev_iv(symbol, prev_positions))

 attribution['delta_pnl'] += delta_pnl
 attribution['gamma_pnl'] += gamma_pnl
 attribution['theta_pnl'] += theta_pnl
 attribution['vega_pnl'] += vega_pnl

 attribution['total_pnl'] = sum(attribution.values())
 return attribution

 def analyze_theta_decay(self, positions: Dict[str, Position]) -> Dict[str, Any]:
 """Analyze time decay across portfolio"""
 theta_analysis = {
 'daily_theta': 0.0,
 'weekly_theta': 0.0,
 'monthly_theta': 0.0,
 'theta_by_expiry': {},
 'theta_by_strike': {},
 'theta_efficiency': 0.0
 }

 daily_theta = sum(pos.theta * pos.quantity for pos in positions.values())
 theta_analysis['daily_theta'] = daily_theta
 theta_analysis['weekly_theta'] = daily_theta * 7
 theta_analysis['monthly_theta'] = daily_theta * 30

 total_premium = sum(abs(pos.market_value) for pos in positions.values())
 theta_analysis['theta_efficiency'] = abs(daily_theta) / max(total_premium, 1.0)

 return theta_analysis

 def calculate_scenario_analysis(self, positions: Dict[str, Position],
 scenarios: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
 """Perform scenario analysis on portfolio"""
 scenario_results = {}

 for i, scenario in enumerate(scenarios):
 scenario_name = f"scenario_{i+1}"
 scenario_pnl = 0.0

 for symbol, position in positions.items():
 underlying_change = scenario.get('underlying_change', 0.0)
 vol_change = scenario.get('vol_change', 0.0)
 time_change = scenario.get('time_change', 0.0)

 delta_impact = position.delta * position.quantity * underlying_change
 gamma_impact = 0.5 * position.gamma * position.quantity * (underlying_change ** 2)
 vega_impact = position.vega * position.quantity * vol_change
 theta_impact = position.theta * position.quantity * time_change

 scenario_pnl += delta_impact + gamma_impact + vega_impact + theta_impact

 scenario_results[scenario_name] = {
 'total_pnl': scenario_pnl,
 'scenario_params': scenario
 }

 return scenario_results

 def _get_prev_iv(self, symbol: str, prev_positions: Dict[str, Position]) -> float:
 """Get previous implied volatility for symbol"""
 return 0.2

class MarketAnalyzer:
 """Market analysis and sentiment tools"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize market analyzer"""
 self.logger.info("Initializing Market Analyzer")

 def analyze_options_flow(self, flows: List[OptionsFlow]) -> Dict[str, Any]:
 """Analyze options flow and sentiment"""
 if not flows:
 return {}

 recent_flows = [f for f in flows if
 (datetime.now() - f.timestamp).total_seconds() < 3600]

 total_volume = sum(f.volume for f in recent_flows)
 call_volume = sum(f.volume for f in recent_flows if f.option_type == 'call')
 put_volume = sum(f.volume for f in recent_flows if f.option_type == 'put')

 put_call_ratio = put_volume / max(call_volume, 1.0)

 unusual_activity = [f for f in recent_flows if f.unusual_activity]
 large_trades = [f for f in recent_flows if f.size_category == 'large']

 sentiment_score = self._calculate_sentiment_score(recent_flows)

 return {
 'total_volume': total_volume,
 'call_volume': call_volume,
 'put_volume': put_volume,
 'put_call_ratio': put_call_ratio,
 'unusual_activity_count': len(unusual_activity),
 'large_trades_count': len(large_trades),
 'sentiment_score': sentiment_score,
 'dominant_strikes': self._get_dominant_strikes(recent_flows),
 'flow_direction': self._analyze_flow_direction(recent_flows)
 }

 def analyze_volatility_surface(self, vol_surface: VolatilitySurface) -> Dict[str, Any]:
 """Analyze volatility surface characteristics"""
 analysis = {
 'atm_vol': 0.0,
 'vol_skew': 0.0,
 'term_structure_slope': 0.0,
 'surface_curvature': 0.0,
 'arbitrage_opportunities': []
 }

 if vol_surface.implied_vols.size == 0:
 return analysis

 atm_index = np.argmin(np.abs(np.array(vol_surface.strikes) - 100))
 analysis['atm_vol'] = vol_surface.implied_vols[0, atm_index] if vol_surface.implied_vols.ndim > 1 else vol_surface.implied_vols[atm_index]

 if len(vol_surface.strikes) > 2:
 high_strike_vol = vol_surface.implied_vols[0, -1] if vol_surface.implied_vols.ndim > 1 else vol_surface.implied_vols[-1]
 low_strike_vol = vol_surface.implied_vols[0, 0] if vol_surface.implied_vols.ndim > 1 else vol_surface.implied_vols[0]
 analysis['vol_skew'] = high_strike_vol - low_strike_vol

 analysis['arbitrage_opportunities'] = self._detect_vol_arbitrage(vol_surface)

 return analysis

 def calculate_iv_percentiles(self, symbol: str, current_iv: float,
 historical_ivs: List[float]) -> Dict[str, float]:
 """Calculate implied volatility percentiles"""
 if not historical_ivs:
 return {'percentile': 50.0, 'rank': 'medium'}

 percentile = (sum(1 for iv in historical_ivs if iv < current_iv) / len(historical_ivs)) * 100

 rank = 'low' if percentile < 25 else 'high' if percentile > 75 else 'medium'

 return {
 'percentile': percentile,
 'rank': rank,
 'historical_mean': np.mean(historical_ivs),
 'historical_std': np.std(historical_ivs),
 'z_score': (current_iv - np.mean(historical_ivs)) / max(np.std(historical_ivs), 0.01)
 }

 def detect_unusual_activity(self, flows: List[OptionsFlow]) -> List[Dict[str, Any]]:
 """Detect unusual options activity"""
 unusual_activities = []

 for flow in flows:
 if not flow.unusual_activity:
 continue

 activity = {
 'symbol': flow.symbol,
 'type': 'large_volume' if flow.size_category == 'large' else 'unusual_oi',
 'details': {
 'volume': flow.volume,
 'open_interest': flow.open_interest,
 'premium': flow.premium,
 'direction': flow.direction
 },
 'significance': self._calculate_activity_significance(flow),
 'timestamp': flow.timestamp
 }
 unusual_activities.append(activity)

 return sorted(unusual_activities, key=lambda x: x['significance'], reverse=True)

 def _calculate_sentiment_score(self, flows: List[OptionsFlow]) -> float:
 """Calculate market sentiment score from flows"""
 if not flows:
 return 0.0

 bullish_flows = sum(f.volume for f in flows if f.direction == 'bullish')
 bearish_flows = sum(f.volume for f in flows if f.direction == 'bearish')
 total_flows = bullish_flows + bearish_flows

 if total_flows == 0:
 return 0.0

 return (bullish_flows - bearish_flows) / total_flows

 def _get_dominant_strikes(self, flows: List[OptionsFlow]) -> List[float]:
 """Get strikes with highest activity"""
 strike_volumes = {}
 for flow in flows:
 strike_volumes[flow.strike] = strike_volumes.get(flow.strike, 0) + flow.volume

 return sorted(strike_volumes.keys(), key=lambda x: strike_volumes[x], reverse=True)[:5]

 def _analyze_flow_direction(self, flows: List[OptionsFlow]) -> str:
 """Analyze overall flow direction"""
 buy_volume = sum(f.volume for f in flows if f.direction in ['buy', 'bullish'])
 sell_volume = sum(f.volume for f in flows if f.direction in ['sell', 'bearish'])

 if buy_volume > sell_volume * 1.2:
 return 'bullish'
 elif sell_volume > buy_volume * 1.2:
 return 'bearish'
 else:
 return 'neutral'

 def _detect_vol_arbitrage(self, vol_surface: VolatilitySurface) -> List[Dict[str, Any]]:
 """Detect volatility arbitrage opportunities"""
 arbitrage_ops = []

 if vol_surface.implied_vols.ndim < 2 or vol_surface.implied_vols.shape[0] < 2:
 return arbitrage_ops

 for i in range(len(vol_surface.strikes) - 2):
 left_vol = vol_surface.implied_vols[0, i]
 center_vol = vol_surface.implied_vols[0, i + 1]
 right_vol = vol_surface.implied_vols[0, i + 2]

 expected_center = (left_vol + right_vol) / 2
 vol_deviation = abs(center_vol - expected_center)

 if vol_deviation > 0.02:
 arbitrage_ops.append({
 'type': 'butterfly_arbitrage',
 'strikes': [vol_surface.strikes[i], vol_surface.strikes[i+1], vol_surface.strikes[i+2]],
 'vol_deviation': vol_deviation,
 'profit_potential': vol_deviation * 100
 })

 return arbitrage_ops

 def _calculate_activity_significance(self, flow: OptionsFlow) -> float:
 """Calculate significance score for unusual activity"""
 volume_score = min(flow.volume / 1000.0, 10.0)
 premium_score = min(flow.premium / 100000.0, 10.0)
 oi_score = min(flow.open_interest / 5000.0, 5.0)

 return volume_score + premium_score + oi_score

class SignalGenerator:
 """Trading signal generation"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize signal generator"""
 self.logger.info("Initializing Signal Generator")

 def generate_volatility_signals(self, market_data: Dict[str, MarketData],
 historical_data: Dict[str, List[float]]) -> List[Dict[str, Any]]:
 """Generate volatility-based trading signals"""
 signals = []

 for symbol, md in market_data.items():
 if symbol not in historical_data:
 continue

 hist_vols = historical_data[symbol]
 if len(hist_vols) < 20:
 continue

 current_iv = md.implied_volatility
 mean_iv = np.mean(hist_vols[-20:])
 std_iv = np.std(hist_vols[-20:])

 z_score = (current_iv - mean_iv) / max(std_iv, 0.01)

 if z_score > 2.0:
 signals.append({
 'symbol': symbol,
 'signal_type': 'volatility_mean_reversion',
 'direction': 'sell_vol',
 'strength': min(abs(z_score) / 2.0, 1.0),
 'entry_iv': current_iv,
 'target_iv': mean_iv,
 'timestamp': datetime.now()
 })
 elif z_score < -2.0:
 signals.append({
 'symbol': symbol,
 'signal_type': 'volatility_mean_reversion',
 'direction': 'buy_vol',
 'strength': min(abs(z_score) / 2.0, 1.0),
 'entry_iv': current_iv,
 'target_iv': mean_iv,
 'timestamp': datetime.now()
 })

 return signals

 def detect_mispricing(self, market_data: Dict[str, MarketData]) -> List[Dict[str, Any]]:
 """Detect options mispricing opportunities"""
 mispricing_signals = []

 for symbol, md in market_data.items():
 theoretical_price = self._calculate_theoretical_price(md)
 market_price = (md.bid + md.ask) / 2

 price_deviation = (market_price - theoretical_price) / theoretical_price

 if abs(price_deviation) > 0.05:
 mispricing_signals.append({
 'symbol': symbol,
 'signal_type': 'mispricing',
 'direction': 'sell' if price_deviation > 0 else 'buy',
 'market_price': market_price,
 'theoretical_price': theoretical_price,
 'deviation': price_deviation,
 'profit_potential': abs(price_deviation) * 100,
 'timestamp': datetime.now()
 })

 return mispricing_signals

 def generate_arbitrage_signals(self, vol_surfaces: Dict[str, VolatilitySurface]) -> List[Dict[str, Any]]:
 """Generate cross-asset arbitrage signals"""
 arbitrage_signals = []

 symbols = list(vol_surfaces.keys())
 for i in range(len(symbols)):
 for j in range(i + 1, len(symbols)):
 symbol1, symbol2 = symbols[i], symbols[j]

 correlation = self._calculate_correlation(symbol1, symbol2)
 if correlation < 0.7:
 continue

 vol_spread = self._calculate_vol_spread(vol_surfaces[symbol1], vol_surfaces[symbol2])

 if abs(vol_spread) > 0.1:
 arbitrage_signals.append({
 'signal_type': 'cross_asset_arbitrage',
 'asset_pair': [symbol1, symbol2],
 'vol_spread': vol_spread,
 'correlation': correlation,
 'direction': 'long_spread' if vol_spread < 0 else 'short_spread',
 'profit_potential': abs(vol_spread) * 50,
 'timestamp': datetime.now()
 })

 return arbitrage_signals

 def detect_earnings_opportunities(self, market_data: Dict[str, MarketData],
 earnings_dates: Dict[str, datetime]) -> List[Dict[str, Any]]:
 """Detect earnings volatility trading opportunities"""
 earnings_signals = []

 for symbol, md in market_data.items():
 if symbol not in earnings_dates:
 continue

 earnings_date = earnings_dates[symbol]
 days_to_earnings = (earnings_date - datetime.now()).days

 if 0 <= days_to_earnings <= 5:
 historical_moves = self._get_historical_earnings_moves(symbol)
 implied_move = self._calculate_implied_move(md)

 if historical_moves:
 avg_historical_move = np.mean(historical_moves)
 move_ratio = implied_move / avg_historical_move

 if move_ratio > 1.3:
 earnings_signals.append({
 'symbol': symbol,
 'signal_type': 'earnings_vol_crush',
 'direction': 'sell_straddle',
 'implied_move': implied_move,
 'historical_move': avg_historical_move,
 'overpricing': move_ratio - 1.0,
 'days_to_earnings': days_to_earnings,
 'timestamp': datetime.now()
 })
 elif move_ratio < 0.7:
 earnings_signals.append({
 'symbol': symbol,
 'signal_type': 'earnings_vol_expansion',
 'direction': 'buy_straddle',
 'implied_move': implied_move,
 'historical_move': avg_historical_move,
 'underpricing': 1.0 - move_ratio,
 'days_to_earnings': days_to_earnings,
 'timestamp': datetime.now()
 })

 return earnings_signals

 def _calculate_theoretical_price(self, md: MarketData) -> float:
 """Calculate theoretical option price using Black-Scholes"""
 from scipy.stats import norm
 import math

 S = md.underlying_price
 K = md.strike
 T = md.time_to_expiry
 r = 0.05
 sigma = md.implied_volatility

 if T <= 0:
 return max(S - K, 0) if md.option_type == 'call' else max(K - S, 0)

 d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
 d2 = d1 - sigma * math.sqrt(T)

 if md.option_type == 'call':
 price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
 else:
 price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

 return max(price, 0.01)

 def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
 """Calculate correlation between two assets"""
 return 0.8

 def _calculate_vol_spread(self, vol_surface1: VolatilitySurface,
 vol_surface2: VolatilitySurface) -> float:
 """Calculate volatility spread between surfaces"""
 if vol_surface1.implied_vols.size == 0 or vol_surface2.implied_vols.size == 0:
 return 0.0

 avg_vol1 = np.mean(vol_surface1.implied_vols)
 avg_vol2 = np.mean(vol_surface2.implied_vols)

 return avg_vol1 - avg_vol2

 def _get_historical_earnings_moves(self, symbol: str) -> List[float]:
 """Get historical earnings moves for symbol"""
 return [0.05, 0.08, 0.06, 0.12, 0.04]

 def _calculate_implied_move(self, md: MarketData) -> float:
 """Calculate implied earnings move from straddle price"""
 straddle_price = md.last * 2
 return straddle_price / md.underlying_price

class RiskAnalyzer:
 """Risk analysis and monitoring"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize risk analyzer"""
 self.logger.info("Initializing Risk Analyzer")

 def calculate_var(self, positions: Dict[str, Position],
 confidence_level: float = 0.95,
 time_horizon: int = 1) -> Dict[str, float]:
 """Calculate Value at Risk"""
 portfolio_values = []

 for _ in range(10000):
 scenario_value = 0.0
 for position in positions.values():
 random_return = np.random.normal(0, 0.02)
 new_price = position.current_price * (1 + random_return)
 scenario_value += position.quantity * new_price

 portfolio_values.append(scenario_value)

 portfolio_values = np.array(portfolio_values)
 current_value = sum(pos.market_value for pos in positions.values())

 var_value = np.percentile(portfolio_values - current_value, (1 - confidence_level) * 100)
 expected_shortfall = np.mean(portfolio_values[portfolio_values <= np.percentile(portfolio_values, (1 - confidence_level) * 100)]) - current_value

 return {
 'var': abs(var_value),
 'expected_shortfall': abs(expected_shortfall),
 'confidence_level': confidence_level,
 'time_horizon': time_horizon
 }

 def stress_test_portfolio(self, positions: Dict[str, Position]) -> Dict[str, Dict[str, float]]:
 """Perform stress tests on portfolio"""
 stress_scenarios = {
 'market_crash': {'underlying_change': -0.20, 'vol_change': 0.10},
 'vol_spike': {'underlying_change': 0.0, 'vol_change': 0.15},
 'interest_rate_shock': {'underlying_change': 0.0, 'rate_change': 0.02},
 'time_decay': {'underlying_change': 0.0, 'time_change': 7},
 'black_swan': {'underlying_change': -0.35, 'vol_change': 0.25}
 }

 stress_results = {}

 for scenario_name, scenario in stress_scenarios.items():
 total_pnl = 0.0

 for position in positions.values():
 underlying_change = scenario.get('underlying_change', 0.0)
 vol_change = scenario.get('vol_change', 0.0)
 time_change = scenario.get('time_change', 0.0)

 delta_impact = position.delta * position.quantity * underlying_change * position.current_price
 gamma_impact = 0.5 * position.gamma * position.quantity * (underlying_change * position.current_price) ** 2
 vega_impact = position.vega * position.quantity * vol_change
 theta_impact = position.theta * position.quantity * time_change

 total_pnl += delta_impact + gamma_impact + vega_impact + theta_impact

 stress_results[scenario_name] = {
 'total_pnl': total_pnl,
 'scenario_params': scenario
 }

 return stress_results