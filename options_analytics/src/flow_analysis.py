"""
Options Flow and Sentiment Analysis
Advanced options flow tracking and market sentiment analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import warnings

from analytics_framework import OptionsFlow, MarketData

@dataclass
class FlowPattern:
 pattern_id: str
 pattern_type: str
 symbol: str
 timeframe: str
 strength: float
 confidence: float
 flow_data: List[OptionsFlow]
 characteristics: Dict[str, Any]
 detected_at: datetime

@dataclass
class UnusualActivity:
 activity_id: str
 symbol: str
 activity_type: str
 magnitude: float
 significance_score: float
 flow_details: OptionsFlow
 market_context: Dict[str, Any]
 alert_level: str
 timestamp: datetime

@dataclass
class SentimentIndicator:
 indicator_name: str
 value: float
 normalized_value: float
 confidence: float
 timeframe: str
 components: Dict[str, float]
 trend: str
 timestamp: datetime

@dataclass
class FlowAnalytics:
 symbol: str
 timeframe: str
 total_volume: int
 call_volume: int
 put_volume: int
 put_call_ratio: float
 net_premium: float
 dominant_strikes: List[float]
 flow_direction: str
 unusual_activity_score: float
 sentiment_score: float
 institutional_activity: float
 retail_activity: float

class FlowAnalysisEngine:
 """Comprehensive options flow and sentiment analysis"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 # Flow data storage
 self.flow_history: deque = deque(maxlen=50000)
 self.processed_flows: Dict[str, List[OptionsFlow]] = defaultdict(list)

 # Analysis components
 self.pattern_detector = FlowPatternDetector(config)
 self.unusual_activity_detector = UnusualActivityDetector(config)
 self.sentiment_calculator = SentimentCalculator(config)
 self.flow_classifier = FlowClassifier(config)

 # Analytics storage
 self.flow_analytics: Dict[str, FlowAnalytics] = {}
 self.sentiment_indicators: Dict[str, SentimentIndicator] = {}
 self.unusual_activities: List[UnusualActivity] = []
 self.detected_patterns: List[FlowPattern] = []

 # Configuration
 self.analysis_intervals = config.get('analysis_intervals', ['1m', '5m', '15m', '1h', '1d'])
 self.unusual_activity_threshold = config.get('unusual_activity_threshold', 2.0)

 async def initialize(self):
 """Initialize flow analysis engine"""
 self.logger.info("Initializing Options Flow Analysis Engine")
 await asyncio.gather(
 self.pattern_detector.initialize(),
 self.unusual_activity_detector.initialize(),
 self.sentiment_calculator.initialize(),
 self.flow_classifier.initialize()
 )

 async def process_options_flow(self, flows: List[OptionsFlow]):
 """Process new options flow data"""
 for flow in flows:
 self.flow_history.append(flow)
 self.processed_flows[flow.symbol].append(flow)

 # Trigger analysis
 await self._analyze_flows(flows)

 async def get_flow_analytics(self, symbol: str = None, timeframe: str = '1h') -> Dict[str, FlowAnalytics]:
 """Get flow analytics for symbol or all symbols"""
 if symbol:
 return {symbol: await self._calculate_flow_analytics(symbol, timeframe)}
 else:
 analytics = {}
 symbols = set(flow.symbol for flow in self.flow_history)
 for sym in symbols:
 analytics[sym] = await self._calculate_flow_analytics(sym, timeframe)
 return analytics

 async def get_unusual_activities(self, symbol: str = None,
 min_significance: float = 0.0) -> List[UnusualActivity]:
 """Get unusual activities"""
 activities = self.unusual_activities

 if symbol:
 activities = [a for a in activities if a.symbol == symbol]

 if min_significance > 0:
 activities = [a for a in activities if a.significance_score >= min_significance]

 return sorted(activities, key=lambda x: x.significance_score, reverse=True)

 async def get_sentiment_indicators(self, timeframe: str = '1h') -> Dict[str, SentimentIndicator]:
 """Get market sentiment indicators"""
 return await self.sentiment_calculator.calculate_sentiment_indicators(
 list(self.flow_history), timeframe
 )

 async def get_flow_patterns(self, symbol: str = None) -> List[FlowPattern]:
 """Get detected flow patterns"""
 patterns = self.detected_patterns

 if symbol:
 patterns = [p for p in patterns if p.symbol == symbol]

 return sorted(patterns, key=lambda x: x.confidence, reverse=True)

 async def get_put_call_analysis(self, symbol: str = None) -> Dict[str, Any]:
 """Get put/call ratio analysis"""
 recent_flows = [f for f in self.flow_history if
 (datetime.now() - f.timestamp).total_seconds() < 3600]

 if symbol:
 recent_flows = [f for f in recent_flows if f.symbol == symbol]

 if not recent_flows:
 return {}

 call_volume = sum(f.volume for f in recent_flows if f.option_type == 'call')
 put_volume = sum(f.volume for f in recent_flows if f.option_type == 'put')

 put_call_ratio = put_volume / max(call_volume, 1)

 # Calculate rolling averages
 pc_ratios_1d = await self._calculate_historical_pc_ratios(symbol, 24)
 pc_ratios_5d = await self._calculate_historical_pc_ratios(symbol, 120)

 return {
 'current_ratio': put_call_ratio,
 'call_volume': call_volume,
 'put_volume': put_volume,
 'total_volume': call_volume + put_volume,
 'ratio_1d_avg': np.mean(pc_ratios_1d) if pc_ratios_1d else put_call_ratio,
 'ratio_5d_avg': np.mean(pc_ratios_5d) if pc_ratios_5d else put_call_ratio,
 'ratio_percentile': await self._calculate_pc_ratio_percentile(put_call_ratio, symbol),
 'interpretation': await self._interpret_pc_ratio(put_call_ratio)
 }

 async def get_volume_analysis(self, symbol: str = None) -> Dict[str, Any]:
 """Get detailed volume analysis"""
 recent_flows = [f for f in self.flow_history if
 (datetime.now() - f.timestamp).total_seconds() < 3600]

 if symbol:
 recent_flows = [f for f in recent_flows if f.symbol == symbol]

 if not recent_flows:
 return {}

 total_volume = sum(f.volume for f in recent_flows)
 avg_trade_size = total_volume / len(recent_flows)

 # Volume by strike analysis
 strike_volumes = defaultdict(int)
 for flow in recent_flows:
 strike_volumes[flow.strike] += flow.volume

 top_strikes = sorted(strike_volumes.items(), key=lambda x: x[1], reverse=True)[:5]

 # Volume by expiry
 expiry_volumes = defaultdict(int)
 for flow in recent_flows:
 expiry_volumes[flow.expiry] += flow.volume

 # Large trade analysis
 large_trades = [f for f in recent_flows if f.size_category == 'large']
 large_trade_volume = sum(f.volume for f in large_trades)

 return {
 'total_volume': total_volume,
 'trade_count': len(recent_flows),
 'avg_trade_size': avg_trade_size,
 'large_trade_volume': large_trade_volume,
 'large_trade_ratio': large_trade_volume / max(total_volume, 1),
 'top_strikes': top_strikes,
 'expiry_distribution': dict(expiry_volumes),
 'volume_trend': await self._calculate_volume_trend(symbol)
 }

 async def _analyze_flows(self, flows: List[OptionsFlow]):
 """Analyze new flows"""
 # Detect patterns
 patterns = await self.pattern_detector.detect_patterns(flows, list(self.flow_history))
 self.detected_patterns.extend(patterns)

 # Detect unusual activity
 unusual_activities = await self.unusual_activity_detector.detect_unusual_activity(
 flows, list(self.flow_history)
 )
 self.unusual_activities.extend(unusual_activities)

 # Update sentiment indicators
 for interval in self.analysis_intervals:
 sentiment = await self.sentiment_calculator.calculate_sentiment_indicators(
 list(self.flow_history), interval
 )
 self.sentiment_indicators.update(sentiment)

 # Cleanup old data
 await self._cleanup_old_data()

 async def _calculate_flow_analytics(self, symbol: str, timeframe: str) -> FlowAnalytics:
 """Calculate comprehensive flow analytics for symbol"""
 # Get flows for timeframe
 cutoff_time = self._get_cutoff_time(timeframe)
 symbol_flows = [f for f in self.processed_flows[symbol]
 if f.timestamp >= cutoff_time]

 if not symbol_flows:
 return FlowAnalytics(
 symbol=symbol,
 timeframe=timeframe,
 total_volume=0,
 call_volume=0,
 put_volume=0,
 put_call_ratio=1.0,
 net_premium=0.0,
 dominant_strikes=[],
 flow_direction='neutral',
 unusual_activity_score=0.0,
 sentiment_score=0.0,
 institutional_activity=0.0,
 retail_activity=0.0
 )

 # Calculate basic metrics
 total_volume = sum(f.volume for f in symbol_flows)
 call_volume = sum(f.volume for f in symbol_flows if f.option_type == 'call')
 put_volume = sum(f.volume for f in symbol_flows if f.option_type == 'put')
 put_call_ratio = put_volume / max(call_volume, 1)

 net_premium = sum(f.premium for f in symbol_flows)

 # Dominant strikes
 strike_volumes = defaultdict(int)
 for flow in symbol_flows:
 strike_volumes[flow.strike] += flow.volume

 dominant_strikes = sorted(strike_volumes.keys(),
 key=lambda x: strike_volumes[x], reverse=True)[:3]

 # Flow direction
 bullish_volume = sum(f.volume for f in symbol_flows if f.direction == 'bullish')
 bearish_volume = sum(f.volume for f in symbol_flows if f.direction == 'bearish')

 if bullish_volume > bearish_volume * 1.2:
 flow_direction = 'bullish'
 elif bearish_volume > bullish_volume * 1.2:
 flow_direction = 'bearish'
 else:
 flow_direction = 'neutral'

 # Unusual activity score
 unusual_flows = [f for f in symbol_flows if f.unusual_activity]
 unusual_activity_score = len(unusual_flows) / max(len(symbol_flows), 1)

 # Sentiment score
 sentiment_score = (bullish_volume - bearish_volume) / max(total_volume, 1)

 # Institutional vs retail activity
 large_flows = [f for f in symbol_flows if f.size_category == 'large']
 institutional_activity = sum(f.volume for f in large_flows) / max(total_volume, 1)
 retail_activity = 1.0 - institutional_activity

 return FlowAnalytics(
 symbol=symbol,
 timeframe=timeframe,
 total_volume=total_volume,
 call_volume=call_volume,
 put_volume=put_volume,
 put_call_ratio=put_call_ratio,
 net_premium=net_premium,
 dominant_strikes=dominant_strikes,
 flow_direction=flow_direction,
 unusual_activity_score=unusual_activity_score,
 sentiment_score=sentiment_score,
 institutional_activity=institutional_activity,
 retail_activity=retail_activity
 )

 async def _calculate_historical_pc_ratios(self, symbol: str, hours: int) -> List[float]:
 """Calculate historical put/call ratios"""
 cutoff_time = datetime.now() - timedelta(hours=hours)

 if symbol:
 flows = [f for f in self.processed_flows[symbol] if f.timestamp >= cutoff_time]
 else:
 flows = [f for f in self.flow_history if f.timestamp >= cutoff_time]

 # Group by hour and calculate ratios
 hourly_ratios = []
 for hour in range(hours):
 hour_start = cutoff_time + timedelta(hours=hour)
 hour_end = hour_start + timedelta(hours=1)

 hour_flows = [f for f in flows if hour_start <= f.timestamp < hour_end]

 if hour_flows:
 call_vol = sum(f.volume for f in hour_flows if f.option_type == 'call')
 put_vol = sum(f.volume for f in hour_flows if f.option_type == 'put')
 ratio = put_vol / max(call_vol, 1)
 hourly_ratios.append(ratio)

 return hourly_ratios

 async def _calculate_pc_ratio_percentile(self, current_ratio: float, symbol: str) -> float:
 """Calculate percentile of current P/C ratio"""
 historical_ratios = await self._calculate_historical_pc_ratios(symbol, 24 * 30) # 30 days

 if not historical_ratios:
 return 50.0

 percentile = (sum(1 for ratio in historical_ratios if ratio < current_ratio) /
 len(historical_ratios)) * 100

 return percentile

 async def _interpret_pc_ratio(self, ratio: float) -> str:
 """Interpret put/call ratio"""
 if ratio > 1.5:
 return 'extremely_bearish'
 elif ratio > 1.2:
 return 'bearish'
 elif ratio > 0.8:
 return 'neutral'
 elif ratio > 0.6:
 return 'bullish'
 else:
 return 'extremely_bullish'

 async def _calculate_volume_trend(self, symbol: str) -> str:
 """Calculate volume trend"""
 recent_volume = sum(f.volume for f in self.processed_flows[symbol][-10:])
 historical_volume = sum(f.volume for f in self.processed_flows[symbol][-20:-10])

 if not historical_volume:
 return 'insufficient_data'

 volume_change = (recent_volume - historical_volume) / historical_volume

 if volume_change > 0.2:
 return 'increasing'
 elif volume_change < -0.2:
 return 'decreasing'
 else:
 return 'stable'

 def _get_cutoff_time(self, timeframe: str) -> datetime:
 """Get cutoff time for timeframe"""
 now = datetime.now()

 if timeframe == '1m':
 return now - timedelta(minutes=1)
 elif timeframe == '5m':
 return now - timedelta(minutes=5)
 elif timeframe == '15m':
 return now - timedelta(minutes=15)
 elif timeframe == '1h':
 return now - timedelta(hours=1)
 elif timeframe == '1d':
 return now - timedelta(days=1)
 else:
 return now - timedelta(hours=1)

 async def _cleanup_old_data(self):
 """Cleanup old analysis data"""
 cutoff_time = datetime.now() - timedelta(hours=24)

 # Cleanup unusual activities
 self.unusual_activities = [a for a in self.unusual_activities
 if a.timestamp >= cutoff_time]

 # Cleanup patterns
 self.detected_patterns = [p for p in self.detected_patterns
 if p.detected_at >= cutoff_time]

 # Cleanup processed flows
 for symbol in self.processed_flows:
 self.processed_flows[symbol] = [f for f in self.processed_flows[symbol]
 if f.timestamp >= cutoff_time]

class FlowPatternDetector:
 """Detect patterns in options flow"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize pattern detector"""
 self.logger.info("Initializing Flow Pattern Detector")

 async def detect_patterns(self, new_flows: List[OptionsFlow],
 historical_flows: List[OptionsFlow]) -> List[FlowPattern]:
 """Detect flow patterns"""
 patterns = []

 # Group flows by symbol
 symbol_flows = defaultdict(list)
 for flow in historical_flows[-1000:]: # Recent flows
 symbol_flows[flow.symbol].append(flow)

 for symbol, flows in symbol_flows.items():
 if len(flows) < 5:
 continue

 # Detect sweep patterns
 sweep_patterns = await self._detect_sweep_patterns(symbol, flows)
 patterns.extend(sweep_patterns)

 # Detect accumulation patterns
 accumulation_patterns = await self._detect_accumulation_patterns(symbol, flows)
 patterns.extend(accumulation_patterns)

 # Detect hedge patterns
 hedge_patterns = await self._detect_hedge_patterns(symbol, flows)
 patterns.extend(hedge_patterns)

 return patterns

 async def _detect_sweep_patterns(self, symbol: str, flows: List[OptionsFlow]) -> List[FlowPattern]:
 """Detect options sweep patterns"""
 patterns = []

 # Look for rapid succession of trades across strikes
 recent_flows = [f for f in flows if
 (datetime.now() - f.timestamp).total_seconds() < 300] # 5 minutes

 if len(recent_flows) < 3:
 return patterns

 # Group by expiry and option type
 grouped_flows = defaultdict(list)
 for flow in recent_flows:
 key = (flow.expiry, flow.option_type)
 grouped_flows[key].append(flow)

 for (expiry, option_type), group_flows in grouped_flows.items():
 if len(group_flows) >= 3:
 # Check if strikes are consecutive or close
 strikes = sorted([f.strike for f in group_flows])

 # Check for sweep pattern (multiple strikes hit quickly)
 time_span = max(f.timestamp for f in group_flows) - min(f.timestamp for f in group_flows)

 if time_span.total_seconds() <= 60 and len(set(strikes)) >= 3:
 total_volume = sum(f.volume for f in group_flows)
 avg_premium = sum(f.premium for f in group_flows) / len(group_flows)

 pattern = FlowPattern(
 pattern_id=f"sweep_{symbol}_{datetime.now().strftime('%H%M%S')}",
 pattern_type="sweep",
 symbol=symbol,
 timeframe="5m",
 strength=min(len(group_flows) / 5.0, 1.0),
 confidence=0.8,
 flow_data=group_flows,
 characteristics={
 'strikes': strikes,
 'option_type': option_type,
 'total_volume': total_volume,
 'avg_premium': avg_premium,
 'time_span_seconds': time_span.total_seconds()
 },
 detected_at=datetime.now()
 )
 patterns.append(pattern)

 return patterns

 async def _detect_accumulation_patterns(self, symbol: str, flows: List[OptionsFlow]) -> List[FlowPattern]:
 """Detect accumulation patterns"""
 patterns = []

 # Look for sustained buying/selling pressure
 recent_flows = [f for f in flows if
 (datetime.now() - f.timestamp).total_seconds() < 3600] # 1 hour

 if len(recent_flows) < 10:
 return patterns

 # Group by strike and expiry
 strike_activity = defaultdict(list)
 for flow in recent_flows:
 key = (flow.strike, flow.expiry, flow.option_type)
 strike_activity[key].append(flow)

 for (strike, expiry, option_type), strike_flows in strike_activity.items():
 if len(strike_flows) >= 5:
 # Check for consistent direction
 directions = [f.direction for f in strike_flows]
 dominant_direction = max(set(directions), key=directions.count)
 direction_consistency = directions.count(dominant_direction) / len(directions)

 if direction_consistency >= 0.7:
 total_volume = sum(f.volume for f in strike_flows)

 pattern = FlowPattern(
 pattern_id=f"accumulation_{symbol}_{datetime.now().strftime('%H%M%S')}",
 pattern_type="accumulation",
 symbol=symbol,
 timeframe="1h",
 strength=min(total_volume / 5000.0, 1.0),
 confidence=direction_consistency,
 flow_data=strike_flows,
 characteristics={
 'strike': strike,
 'expiry': expiry.isoformat(),
 'option_type': option_type,
 'direction': dominant_direction,
 'consistency': direction_consistency,
 'total_volume': total_volume
 },
 detected_at=datetime.now()
 )
 patterns.append(pattern)

 return patterns

 async def _detect_hedge_patterns(self, symbol: str, flows: List[OptionsFlow]) -> List[FlowPattern]:
 """Detect hedging patterns"""
 patterns = []

 recent_flows = [f for f in flows if
 (datetime.now() - f.timestamp).total_seconds() < 1800] # 30 minutes

 if len(recent_flows) < 4:
 return patterns

 # Look for offsetting positions (e.g., call spreads, put spreads)
 call_flows = [f for f in recent_flows if f.option_type == 'call']
 put_flows = [f for f in recent_flows if f.option_type == 'put']

 # Detect potential spreads
 if len(call_flows) >= 2:
 hedge_pattern = await self._detect_spread_pattern(symbol, call_flows, 'call_spread')
 if hedge_pattern:
 patterns.append(hedge_pattern)

 if len(put_flows) >= 2:
 hedge_pattern = await self._detect_spread_pattern(symbol, put_flows, 'put_spread')
 if hedge_pattern:
 patterns.append(hedge_pattern)

 # Detect collar patterns (calls + puts)
 if len(call_flows) >= 1 and len(put_flows) >= 1:
 collar_pattern = await self._detect_collar_pattern(symbol, call_flows, put_flows)
 if collar_pattern:
 patterns.append(collar_pattern)

 return patterns

 async def _detect_spread_pattern(self, symbol: str, flows: List[OptionsFlow],
 pattern_type: str) -> Optional[FlowPattern]:
 """Detect spread patterns in flows"""
 if len(flows) < 2:
 return None

 # Look for opposing directions at different strikes
 by_direction = defaultdict(list)
 for flow in flows:
 by_direction[flow.direction].append(flow)

 if len(by_direction) < 2:
 return None

 # Check for buy and sell at different strikes
 buy_flows = by_direction.get('buy', []) + by_direction.get('bullish', [])
 sell_flows = by_direction.get('sell', []) + by_direction.get('bearish', [])

 if not buy_flows or not sell_flows:
 return None

 return FlowPattern(
 pattern_id=f"spread_{symbol}_{datetime.now().strftime('%H%M%S')}",
 pattern_type=pattern_type,
 symbol=symbol,
 timeframe="30m",
 strength=0.7,
 confidence=0.6,
 flow_data=flows,
 characteristics={
 'buy_strikes': [f.strike for f in buy_flows],
 'sell_strikes': [f.strike for f in sell_flows],
 'net_premium': sum(f.premium for f in buy_flows) - sum(f.premium for f in sell_flows)
 },
 detected_at=datetime.now()
 )

 async def _detect_collar_pattern(self, symbol: str, call_flows: List[OptionsFlow],
 put_flows: List[OptionsFlow]) -> Optional[FlowPattern]:
 """Detect collar patterns"""
 # Simplified collar detection
 if not call_flows or not put_flows:
 return None

 return FlowPattern(
 pattern_id=f"collar_{symbol}_{datetime.now().strftime('%H%M%S')}",
 pattern_type="collar",
 symbol=symbol,
 timeframe="30m",
 strength=0.6,
 confidence=0.5,
 flow_data=call_flows + put_flows,
 characteristics={
 'call_strikes': [f.strike for f in call_flows],
 'put_strikes': [f.strike for f in put_flows],
 'total_flows': len(call_flows) + len(put_flows)
 },
 detected_at=datetime.now()
 )

class UnusualActivityDetector:
 """Detect unusual options activity"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)
 self.volume_thresholds = config.get('volume_thresholds', {
 'small': 100,
 'medium': 500,
 'large': 2000,
 'block': 10000
 })

 async def initialize(self):
 """Initialize unusual activity detector"""
 self.logger.info("Initializing Unusual Activity Detector")

 async def detect_unusual_activity(self, new_flows: List[OptionsFlow],
 historical_flows: List[OptionsFlow]) -> List[UnusualActivity]:
 """Detect unusual activity in options flows"""
 unusual_activities = []

 for flow in new_flows:
 # Check volume-based unusual activity
 volume_activity = await self._check_volume_activity(flow, historical_flows)
 if volume_activity:
 unusual_activities.append(volume_activity)

 # Check open interest activity
 oi_activity = await self._check_open_interest_activity(flow, historical_flows)
 if oi_activity:
 unusual_activities.append(oi_activity)

 # Check premium-based activity
 premium_activity = await self._check_premium_activity(flow, historical_flows)
 if premium_activity:
 unusual_activities.append(premium_activity)

 return unusual_activities

 async def _check_volume_activity(self, flow: OptionsFlow,
 historical_flows: List[OptionsFlow]) -> Optional[UnusualActivity]:
 """Check for unusual volume activity"""
 # Get historical volumes for same symbol and similar strikes
 symbol_flows = [f for f in historical_flows if f.symbol == flow.symbol]

 if len(symbol_flows) < 10:
 return None

 # Calculate average volume for this symbol
 avg_volume = np.mean([f.volume for f in symbol_flows[-50:]])
 std_volume = np.std([f.volume for f in symbol_flows[-50:]])

 # Check if current volume is unusual
 if std_volume > 0:
 z_score = (flow.volume - avg_volume) / std_volume
 else:
 z_score = 0

 if abs(z_score) > 3.0 or flow.volume > self.volume_thresholds['block']:
 significance_score = min(abs(z_score) / 3.0, 2.0)

 alert_level = 'critical' if z_score > 5 else 'high' if z_score > 3 else 'medium'

 return UnusualActivity(
 activity_id=f"volume_{flow.symbol}_{datetime.now().strftime('%H%M%S')}",
 symbol=flow.symbol,
 activity_type="unusual_volume",
 magnitude=flow.volume,
 significance_score=significance_score,
 flow_details=flow,
 market_context={
 'avg_volume': avg_volume,
 'z_score': z_score,
 'volume_category': self._categorize_volume(flow.volume)
 },
 alert_level=alert_level,
 timestamp=datetime.now()
 )

 return None

 async def _check_open_interest_activity(self, flow: OptionsFlow,
 historical_flows: List[OptionsFlow]) -> Optional[UnusualActivity]:
 """Check for unusual open interest activity"""
 if flow.open_interest < 1000:
 return None

 # Compare with recent flows for same strike/expiry
 similar_flows = [f for f in historical_flows if
 f.symbol == flow.symbol and
 f.strike == flow.strike and
 f.expiry == flow.expiry]

 if not similar_flows:
 return None

 avg_oi = np.mean([f.open_interest for f in similar_flows[-20:]])

 if flow.open_interest > avg_oi * 2:
 return UnusualActivity(
 activity_id=f"oi_{flow.symbol}_{datetime.now().strftime('%H%M%S')}",
 symbol=flow.symbol,
 activity_type="unusual_open_interest",
 magnitude=flow.open_interest,
 significance_score=flow.open_interest / max(avg_oi, 1),
 flow_details=flow,
 market_context={
 'avg_open_interest': avg_oi,
 'oi_ratio': flow.open_interest / max(avg_oi, 1)
 },
 alert_level='medium',
 timestamp=datetime.now()
 )

 return None

 async def _check_premium_activity(self, flow: OptionsFlow,
 historical_flows: List[OptionsFlow]) -> Optional[UnusualActivity]:
 """Check for unusual premium activity"""
 if flow.premium < 10000: # Minimum threshold
 return None

 # Get average premium for similar flows
 symbol_flows = [f for f in historical_flows if f.symbol == flow.symbol]

 if len(symbol_flows) < 5:
 return None

 avg_premium = np.mean([f.premium for f in symbol_flows[-20:]])

 if flow.premium > avg_premium * 3:
 return UnusualActivity(
 activity_id=f"premium_{flow.symbol}_{datetime.now().strftime('%H%M%S')}",
 symbol=flow.symbol,
 activity_type="large_premium",
 magnitude=flow.premium,
 significance_score=flow.premium / max(avg_premium, 1),
 flow_details=flow,
 market_context={
 'avg_premium': avg_premium,
 'premium_ratio': flow.premium / max(avg_premium, 1)
 },
 alert_level='high',
 timestamp=datetime.now()
 )

 return None

 def _categorize_volume(self, volume: int) -> str:
 """Categorize volume size"""
 if volume >= self.volume_thresholds['block']:
 return 'block'
 elif volume >= self.volume_thresholds['large']:
 return 'large'
 elif volume >= self.volume_thresholds['medium']:
 return 'medium'
 else:
 return 'small'

class SentimentCalculator:
 """Calculate market sentiment indicators"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize sentiment calculator"""
 self.logger.info("Initializing Sentiment Calculator")

 async def calculate_sentiment_indicators(self, flows: List[OptionsFlow],
 timeframe: str) -> Dict[str, SentimentIndicator]:
 """Calculate comprehensive sentiment indicators"""
 cutoff_time = self._get_cutoff_time(timeframe)
 relevant_flows = [f for f in flows if f.timestamp >= cutoff_time]

 if not relevant_flows:
 return {}

 indicators = {}

 # Put/Call Ratio
 indicators['put_call_ratio'] = await self._calculate_pc_ratio_indicator(relevant_flows, timeframe)

 # Flow Direction Sentiment
 indicators['flow_sentiment'] = await self._calculate_flow_sentiment(relevant_flows, timeframe)

 # Smart Money Indicator
 indicators['smart_money'] = await self._calculate_smart_money_indicator(relevant_flows, timeframe)

 # Fear/Greed Index
 indicators['fear_greed'] = await self._calculate_fear_greed_indicator(relevant_flows, timeframe)

 # Volatility Sentiment
 indicators['volatility_sentiment'] = await self._calculate_volatility_sentiment(relevant_flows, timeframe)

 return indicators

 async def _calculate_pc_ratio_indicator(self, flows: List[OptionsFlow],
 timeframe: str) -> SentimentIndicator:
 """Calculate put/call ratio sentiment indicator"""
 call_volume = sum(f.volume for f in flows if f.option_type == 'call')
 put_volume = sum(f.volume for f in flows if f.option_type == 'put')

 pc_ratio = put_volume / max(call_volume, 1)

 # Normalize (typical range 0.5 - 2.0)
 normalized = max(0, min(1, (pc_ratio - 0.5) / 1.5))

 # Determine trend
 if pc_ratio > 1.2:
 trend = 'bearish'
 elif pc_ratio < 0.8:
 trend = 'bullish'
 else:
 trend = 'neutral'

 return SentimentIndicator(
 indicator_name="put_call_ratio",
 value=pc_ratio,
 normalized_value=normalized,
 confidence=min(len(flows) / 100.0, 1.0),
 timeframe=timeframe,
 components={
 'call_volume': call_volume,
 'put_volume': put_volume,
 'total_volume': call_volume + put_volume
 },
 trend=trend,
 timestamp=datetime.now()
 )

 async def _calculate_flow_sentiment(self, flows: List[OptionsFlow],
 timeframe: str) -> SentimentIndicator:
 """Calculate flow direction sentiment"""
 bullish_volume = sum(f.volume for f in flows if f.direction in ['bullish', 'buy'])
 bearish_volume = sum(f.volume for f in flows if f.direction in ['bearish', 'sell'])
 total_volume = bullish_volume + bearish_volume

 if total_volume == 0:
 sentiment_value = 0.0
 else:
 sentiment_value = (bullish_volume - bearish_volume) / total_volume

 normalized = (sentiment_value + 1) / 2 # Convert from [-1,1] to [0,1]

 trend = 'bullish' if sentiment_value > 0.1 else 'bearish' if sentiment_value < -0.1 else 'neutral'

 return SentimentIndicator(
 indicator_name="flow_sentiment",
 value=sentiment_value,
 normalized_value=normalized,
 confidence=min(total_volume / 5000.0, 1.0),
 timeframe=timeframe,
 components={
 'bullish_volume': bullish_volume,
 'bearish_volume': bearish_volume,
 'net_sentiment': sentiment_value
 },
 trend=trend,
 timestamp=datetime.now()
 )

 async def _calculate_smart_money_indicator(self, flows: List[OptionsFlow],
 timeframe: str) -> SentimentIndicator:
 """Calculate smart money activity indicator"""
 large_flows = [f for f in flows if f.size_category in ['large', 'block']]
 total_large_volume = sum(f.volume for f in large_flows)
 total_volume = sum(f.volume for f in flows)

 smart_money_ratio = total_large_volume / max(total_volume, 1)

 # Smart money flow direction
 large_bullish = sum(f.volume for f in large_flows if f.direction in ['bullish', 'buy'])
 large_bearish = sum(f.volume for f in large_flows if f.direction in ['bearish', 'sell'])

 if total_large_volume > 0:
 smart_direction = (large_bullish - large_bearish) / total_large_volume
 else:
 smart_direction = 0.0

 normalized = smart_money_ratio # Already between 0 and 1

 trend = 'bullish' if smart_direction > 0.1 else 'bearish' if smart_direction < -0.1 else 'neutral'

 return SentimentIndicator(
 indicator_name="smart_money",
 value=smart_money_ratio,
 normalized_value=normalized,
 confidence=min(len(large_flows) / 10.0, 1.0),
 timeframe=timeframe,
 components={
 'large_volume': total_large_volume,
 'smart_direction': smart_direction,
 'large_flow_count': len(large_flows)
 },
 trend=trend,
 timestamp=datetime.now()
 )

 async def _calculate_fear_greed_indicator(self, flows: List[OptionsFlow],
 timeframe: str) -> SentimentIndicator:
 """Calculate fear/greed indicator"""
 # Combine multiple fear/greed signals

 # 1. Put/call ratio component
 call_volume = sum(f.volume for f in flows if f.option_type == 'call')
 put_volume = sum(f.volume for f in flows if f.option_type == 'put')
 pc_ratio = put_volume / max(call_volume, 1)
 fear_component = min(pc_ratio / 2.0, 1.0) # High puts = fear

 # 2. Volatility component (simplified)
 vol_flows = [f for f in flows if hasattr(f, 'implied_vol')]
 if vol_flows:
 avg_vol = np.mean([getattr(f, 'implied_vol', 0.2) for f in vol_flows])
 vol_component = min(avg_vol / 0.5, 1.0) # High vol = fear
 else:
 vol_component = 0.5

 # 3. Premium component
 avg_premium = np.mean([f.premium for f in flows]) if flows else 0
 premium_component = min(avg_premium / 50000, 1.0) # High premium = greed

 # Combine components (fear vs greed)
 fear_greed_value = 1 - ((fear_component + vol_component) / 2 - premium_component / 2)
 fear_greed_value = max(0, min(1, fear_greed_value))

 if fear_greed_value > 0.7:
 trend = 'greed'
 elif fear_greed_value < 0.3:
 trend = 'fear'
 else:
 trend = 'neutral'

 return SentimentIndicator(
 indicator_name="fear_greed",
 value=fear_greed_value,
 normalized_value=fear_greed_value,
 confidence=min(len(flows) / 50.0, 1.0),
 timeframe=timeframe,
 components={
 'fear_component': fear_component,
 'vol_component': vol_component,
 'premium_component': premium_component
 },
 trend=trend,
 timestamp=datetime.now()
 )

 async def _calculate_volatility_sentiment(self, flows: List[OptionsFlow],
 timeframe: str) -> SentimentIndicator:
 """Calculate volatility-based sentiment"""
 # Analyze implied volatility from flows (simplified)
 vol_estimates = []
 for flow in flows:
 # Estimate volatility from premium (simplified)
 if flow.premium > 0:
 vol_estimate = min(flow.premium / 10000, 1.0)
 vol_estimates.append(vol_estimate)

 if vol_estimates:
 avg_vol_sentiment = np.mean(vol_estimates)
 else:
 avg_vol_sentiment = 0.5

 trend = 'high' if avg_vol_sentiment > 0.7 else 'low' if avg_vol_sentiment < 0.3 else 'medium'

 return SentimentIndicator(
 indicator_name="volatility_sentiment",
 value=avg_vol_sentiment,
 normalized_value=avg_vol_sentiment,
 confidence=min(len(vol_estimates) / 20.0, 1.0),
 timeframe=timeframe,
 components={
 'vol_estimates_count': len(vol_estimates),
 'avg_vol_estimate': avg_vol_sentiment
 },
 trend=trend,
 timestamp=datetime.now()
 )

 def _get_cutoff_time(self, timeframe: str) -> datetime:
 """Get cutoff time for timeframe"""
 now = datetime.now()

 if timeframe == '1m':
 return now - timedelta(minutes=1)
 elif timeframe == '5m':
 return now - timedelta(minutes=5)
 elif timeframe == '15m':
 return now - timedelta(minutes=15)
 elif timeframe == '1h':
 return now - timedelta(hours=1)
 elif timeframe == '1d':
 return now - timedelta(days=1)
 else:
 return now - timedelta(hours=1)

class FlowClassifier:
 """Classify options flows by type and characteristics"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize flow classifier"""
 self.logger.info("Initializing Flow Classifier")

 async def classify_flow(self, flow: OptionsFlow) -> Dict[str, Any]:
 """Classify an options flow"""
 classification = {
 'flow_type': await self._determine_flow_type(flow),
 'size_category': self._determine_size_category(flow),
 'directional_bias': await self._determine_directional_bias(flow),
 'institutional_probability': await self._estimate_institutional_probability(flow),
 'strategy_type': await self._infer_strategy_type(flow)
 }

 return classification

 async def _determine_flow_type(self, flow: OptionsFlow) -> str:
 """Determine the type of flow"""
 if flow.volume > 5000:
 return 'block'
 elif flow.volume > 1000:
 return 'institutional'
 elif flow.volume > 100:
 return 'retail_large'
 else:
 return 'retail_small'

 def _determine_size_category(self, flow: OptionsFlow) -> str:
 """Determine size category"""
 if flow.volume >= 10000:
 return 'block'
 elif flow.volume >= 2000:
 return 'large'
 elif flow.volume >= 500:
 return 'medium'
 else:
 return 'small'

 async def _determine_directional_bias(self, flow: OptionsFlow) -> str:
 """Determine directional bias"""
 # Simplified logic based on option type and direction
 if flow.option_type == 'call':
 return 'bullish' if flow.direction in ['buy', 'bullish'] else 'bearish'
 else: # put
 return 'bearish' if flow.direction in ['buy', 'bullish'] else 'bullish'

 async def _estimate_institutional_probability(self, flow: OptionsFlow) -> float:
 """Estimate probability this is institutional flow"""
 score = 0.0

 # Volume factor
 if flow.volume > 5000:
 score += 0.4
 elif flow.volume > 1000:
 score += 0.2

 # Premium factor
 if flow.premium > 100000:
 score += 0.3
 elif flow.premium > 50000:
 score += 0.15

 # Time factor (institutions often trade during market hours)
 trade_hour = flow.timestamp.hour
 if 9 <= trade_hour <= 16: # Market hours
 score += 0.1

 # Unusual activity factor
 if flow.unusual_activity:
 score += 0.2

 return min(score, 1.0)

 async def _infer_strategy_type(self, flow: OptionsFlow) -> str:
 """Infer the likely strategy type"""
 # This would be more sophisticated in practice
 if flow.volume > 5000:
 return 'institutional_strategy'
 elif flow.premium > 50000:
 return 'high_premium_strategy'
 elif flow.option_type == 'call' and flow.direction == 'buy':
 return 'long_call'
 elif flow.option_type == 'put' and flow.direction == 'buy':
 return 'long_put'
 else:
 return 'unknown'