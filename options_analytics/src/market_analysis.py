"""
Advanced Market Analysis and Signal Generation
Comprehensive market intelligence and trading signal systems
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
from collections import defaultdict, deque
from scipy import stats
from scipy.optimize import minimize
import math

from analytics_framework import MarketData, OptionsFlow, VolatilitySurface

@dataclass
class TradingSignal:
    signal_id: str
    symbol: str
    signal_type: str
    direction: str
    strength: float
    confidence: float
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    expiry_time: Optional[datetime]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class MarketRegime:
    regime_type: str
    confidence: float
    characteristics: Dict[str, float]
    start_time: datetime
    duration: timedelta
    volatility_level: str
    trend_direction: str

@dataclass
class VolatilityAnalysis:
    symbol: str
    current_iv: float
    historical_iv_mean: float
    historical_iv_std: float
    iv_percentile: float
    iv_rank: float
    vol_regime: str
    mean_reversion_signal: float
    term_structure_slope: float
    skew_analysis: Dict[str, float]

@dataclass
class ArbitrageOpportunity:
    opportunity_id: str
    opportunity_type: str
    symbols: List[str]
    expected_profit: float
    max_risk: float
    profit_probability: float
    time_to_expiration: float
    execution_complexity: str
    metadata: Dict[str, Any]

class SignalType(Enum):
    VOLATILITY_MEAN_REVERSION = "vol_mean_reversion"
    VOLATILITY_BREAKOUT = "vol_breakout"
    MISPRICING = "mispricing"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    EARNINGS = "earnings"
    EVENT_DRIVEN = "event_driven"
    REGIME_CHANGE = "regime_change"

class MarketAnalysisEngine:
    """Advanced market analysis and signal generation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Market data storage
        self.market_data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.vol_surface_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.options_flow_history: deque = deque(maxlen=10000)

        # Analysis components
        self.volatility_analyzer = VolatilityAnalyzer(config)
        self.signal_generator = AdvancedSignalGenerator(config)
        self.arbitrage_detector = ArbitrageDetector(config)
        self.regime_detector = RegimeDetector(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)

        # Signal storage
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: deque = deque(maxlen=10000)

        # Performance tracking
        self.signal_performance: Dict[str, List[float]] = defaultdict(list)

    async def initialize(self):
        """Initialize market analysis engine"""
        self.logger.info("Initializing Market Analysis Engine")
        await asyncio.gather(
            self.volatility_analyzer.initialize(),
            self.signal_generator.initialize(),
            self.arbitrage_detector.initialize(),
            self.regime_detector.initialize(),
            self.sentiment_analyzer.initialize()
        )

    async def update_market_data(self, market_data: Dict[str, MarketData]):
        """Update market data and trigger analysis"""
        # Store market data
        for symbol, data in market_data.items():
            self.market_data_history[symbol].append(data)

        # Trigger analysis
        await self._analyze_market_conditions(market_data)
        await self._generate_signals(market_data)

    async def update_volatility_surfaces(self, vol_surfaces: Dict[str, VolatilitySurface]):
        """Update volatility surfaces"""
        for symbol, surface in vol_surfaces.items():
            self.vol_surface_history[symbol].append(surface)

        await self._analyze_volatility_surfaces(vol_surfaces)

    async def update_options_flow(self, flows: List[OptionsFlow]):
        """Update options flow data"""
        for flow in flows:
            self.options_flow_history.append(flow)

        await self._analyze_options_flow(flows)

    async def get_active_signals(self, signal_types: List[str] = None,
                               min_strength: float = 0.0) -> List[TradingSignal]:
        """Get active trading signals"""
        signals = list(self.active_signals.values())

        if signal_types:
            signals = [s for s in signals if s.signal_type in signal_types]

        if min_strength > 0:
            signals = [s for s in signals if s.strength >= min_strength]

        return sorted(signals, key=lambda x: x.strength, reverse=True)

    async def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        current_regime = await self.regime_detector.detect_current_regime(self.market_data_history)
        sentiment = await self.sentiment_analyzer.analyze_market_sentiment(self.options_flow_history)

        vol_analysis = {}
        for symbol in self.market_data_history.keys():
            vol_analysis[symbol] = await self.volatility_analyzer.analyze_volatility(
                symbol, list(self.market_data_history[symbol])
            )

        arbitrage_ops = await self.arbitrage_detector.scan_arbitrage_opportunities(
            dict(self.vol_surface_history)
        )

        return {
            'timestamp': datetime.now().isoformat(),
            'market_regime': current_regime,
            'market_sentiment': sentiment,
            'volatility_analysis': vol_analysis,
            'active_signals_count': len(self.active_signals),
            'arbitrage_opportunities': len(arbitrage_ops),
            'signal_performance': await self._calculate_signal_performance()
        }

    async def get_volatility_analysis(self, symbol: str) -> VolatilityAnalysis:
        """Get detailed volatility analysis for symbol"""
        if symbol not in self.market_data_history:
            raise ValueError(f"No market data available for {symbol}")

        return await self.volatility_analyzer.analyze_volatility(
            symbol, list(self.market_data_history[symbol])
        )

    async def get_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get current arbitrage opportunities"""
        return await self.arbitrage_detector.scan_arbitrage_opportunities(
            dict(self.vol_surface_history)
        )

    async def _analyze_market_conditions(self, market_data: Dict[str, MarketData]):
        """Analyze current market conditions"""
        # Detect regime changes
        await self.regime_detector.update_market_data(market_data)

        # Update volatility analysis
        for symbol, data in market_data.items():
            await self.volatility_analyzer.update_analysis(symbol, data)

    async def _generate_signals(self, market_data: Dict[str, MarketData]):
        """Generate trading signals"""
        new_signals = await self.signal_generator.generate_signals(
            market_data, dict(self.market_data_history)
        )

        for signal in new_signals:
            self.active_signals[signal.signal_id] = signal
            self.signal_history.append(signal)

        # Clean up expired signals
        await self._cleanup_expired_signals()

    async def _analyze_volatility_surfaces(self, vol_surfaces: Dict[str, VolatilitySurface]):
        """Analyze volatility surfaces for opportunities"""
        arbitrage_signals = await self.arbitrage_detector.analyze_vol_surfaces(vol_surfaces)

        for signal in arbitrage_signals:
            self.active_signals[signal.signal_id] = signal

    async def _analyze_options_flow(self, flows: List[OptionsFlow]):
        """Analyze options flow for sentiment and signals"""
        sentiment_signals = await self.sentiment_analyzer.generate_flow_signals(flows)

        for signal in sentiment_signals:
            self.active_signals[signal.signal_id] = signal

    async def _cleanup_expired_signals(self):
        """Remove expired signals"""
        current_time = datetime.now()
        expired_signals = []

        for signal_id, signal in self.active_signals.items():
            if signal.expiry_time and current_time > signal.expiry_time:
                expired_signals.append(signal_id)

        for signal_id in expired_signals:
            del self.active_signals[signal_id]

    async def _calculate_signal_performance(self) -> Dict[str, Any]:
        """Calculate signal performance metrics"""
        if not self.signal_performance:
            return {}

        performance = {}
        for signal_type, returns in self.signal_performance.items():
            if returns:
                performance[signal_type] = {
                    'win_rate': len([r for r in returns if r > 0]) / len(returns),
                    'avg_return': np.mean(returns),
                    'total_signals': len(returns),
                    'sharpe_ratio': np.mean(returns) / max(np.std(returns), 0.01)
                }

        return performance

class VolatilityAnalyzer:
    """Advanced volatility analysis"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.volatility_models = {}

    async def initialize(self):
        """Initialize volatility analyzer"""
        self.logger.info("Initializing Volatility Analyzer")

    async def analyze_volatility(self, symbol: str, market_data: List[MarketData]) -> VolatilityAnalysis:
        """Comprehensive volatility analysis"""
        if len(market_data) < 20:
            return self._empty_vol_analysis(symbol)

        current_data = market_data[-1]
        historical_ivs = [md.implied_volatility for md in market_data[-100:]]

        iv_mean = np.mean(historical_ivs)
        iv_std = np.std(historical_ivs)
        iv_percentile = (sum(1 for iv in historical_ivs if iv < current_data.implied_volatility) /
                        len(historical_ivs)) * 100

        # Calculate IV rank (0-100)
        iv_min = min(historical_ivs)
        iv_max = max(historical_ivs)
        iv_rank = ((current_data.implied_volatility - iv_min) /
                  max(iv_max - iv_min, 0.01)) * 100

        # Determine volatility regime
        vol_regime = self._determine_vol_regime(current_data.implied_volatility, iv_mean, iv_std)

        # Calculate mean reversion signal
        z_score = (current_data.implied_volatility - iv_mean) / max(iv_std, 0.01)
        mean_reversion_signal = -np.tanh(z_score)  # Strong signal when far from mean

        # Analyze term structure (simplified)
        term_structure_slope = self._calculate_term_structure_slope(market_data)

        # Skew analysis
        skew_analysis = await self._analyze_volatility_skew(symbol, market_data)

        return VolatilityAnalysis(
            symbol=symbol,
            current_iv=current_data.implied_volatility,
            historical_iv_mean=iv_mean,
            historical_iv_std=iv_std,
            iv_percentile=iv_percentile,
            iv_rank=iv_rank,
            vol_regime=vol_regime,
            mean_reversion_signal=mean_reversion_signal,
            term_structure_slope=term_structure_slope,
            skew_analysis=skew_analysis
        )

    async def update_analysis(self, symbol: str, market_data: MarketData):
        """Update real-time volatility analysis"""
        # Update volatility models
        if symbol not in self.volatility_models:
            self.volatility_models[symbol] = GARCHModel()

        self.volatility_models[symbol].update(market_data.implied_volatility)

    def _empty_vol_analysis(self, symbol: str) -> VolatilityAnalysis:
        """Return empty volatility analysis"""
        return VolatilityAnalysis(
            symbol=symbol,
            current_iv=0.0,
            historical_iv_mean=0.0,
            historical_iv_std=0.0,
            iv_percentile=50.0,
            iv_rank=50.0,
            vol_regime='unknown',
            mean_reversion_signal=0.0,
            term_structure_slope=0.0,
            skew_analysis={}
        )

    def _determine_vol_regime(self, current_iv: float, mean_iv: float, std_iv: float) -> str:
        """Determine current volatility regime"""
        z_score = (current_iv - mean_iv) / max(std_iv, 0.01)

        if z_score > 1.5:
            return 'high'
        elif z_score < -1.5:
            return 'low'
        else:
            return 'normal'

    def _calculate_term_structure_slope(self, market_data: List[MarketData]) -> float:
        """Calculate term structure slope"""
        if len(market_data) < 2:
            return 0.0

        # Simplified calculation using recent vs longer-term
        recent_vol = np.mean([md.implied_volatility for md in market_data[-5:]])
        longer_term_vol = np.mean([md.implied_volatility for md in market_data[-20:-5]])

        return recent_vol - longer_term_vol

    async def _analyze_volatility_skew(self, symbol: str, market_data: List[MarketData]) -> Dict[str, float]:
        """Analyze volatility skew characteristics"""
        # Simplified skew analysis
        return {
            'put_call_skew': 0.02,
            'skew_slope': -0.01,
            'skew_convexity': 0.001
        }

class GARCHModel:
    """Simple GARCH model for volatility forecasting"""

    def __init__(self):
        self.volatilities = deque(maxlen=100)
        self.returns = deque(maxlen=100)

    def update(self, volatility: float):
        """Update model with new volatility"""
        self.volatilities.append(volatility)

        if len(self.volatilities) > 1:
            ret = (volatility - self.volatilities[-2]) / self.volatilities[-2]
            self.returns.append(ret)

    def forecast(self, periods: int = 1) -> float:
        """Forecast future volatility"""
        if len(self.volatilities) < 10:
            return self.volatilities[-1] if self.volatilities else 0.2

        # Simple EWMA forecast
        alpha = 0.94
        forecast_var = np.var(list(self.returns))

        for _ in range(periods):
            forecast_var = alpha * forecast_var + (1 - alpha) * (list(self.returns)[-1] ** 2)

        return math.sqrt(forecast_var * 252)

class AdvancedSignalGenerator:
    """Advanced trading signal generation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.signal_counter = 0

    async def initialize(self):
        """Initialize signal generator"""
        self.logger.info("Initializing Advanced Signal Generator")

    async def generate_signals(self, current_data: Dict[str, MarketData],
                             historical_data: Dict[str, deque]) -> List[TradingSignal]:
        """Generate comprehensive trading signals"""
        signals = []

        # Volatility mean reversion signals
        signals.extend(await self._generate_vol_mean_reversion_signals(current_data, historical_data))

        # Mispricing signals
        signals.extend(await self._generate_mispricing_signals(current_data))

        # Momentum signals
        signals.extend(await self._generate_momentum_signals(current_data, historical_data))

        # Event-driven signals
        signals.extend(await self._generate_event_signals(current_data))

        return signals

    async def _generate_vol_mean_reversion_signals(self, current_data: Dict[str, MarketData],
                                                  historical_data: Dict[str, deque]) -> List[TradingSignal]:
        """Generate volatility mean reversion signals"""
        signals = []

        for symbol, data in current_data.items():
            if symbol not in historical_data or len(historical_data[symbol]) < 20:
                continue

            hist_data = list(historical_data[symbol])
            hist_ivs = [md.implied_volatility for md in hist_data[-50:]]

            mean_iv = np.mean(hist_ivs)
            std_iv = np.std(hist_ivs)
            z_score = (data.implied_volatility - mean_iv) / max(std_iv, 0.01)

            if abs(z_score) > 2.0:
                direction = 'sell' if z_score > 0 else 'buy'
                strength = min(abs(z_score) / 3.0, 1.0)
                confidence = min(abs(z_score) / 2.0, 0.9)

                signal = TradingSignal(
                    signal_id=f"vol_mr_{self._get_next_id()}",
                    symbol=symbol,
                    signal_type=SignalType.VOLATILITY_MEAN_REVERSION.value,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    entry_price=data.last,
                    target_price=None,
                    stop_loss=None,
                    expiry_time=datetime.now() + timedelta(hours=24),
                    metadata={
                        'z_score': z_score,
                        'current_iv': data.implied_volatility,
                        'mean_iv': mean_iv,
                        'std_iv': std_iv
                    },
                    timestamp=datetime.now()
                )
                signals.append(signal)

        return signals

    async def _generate_mispricing_signals(self, current_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """Generate option mispricing signals"""
        signals = []

        for symbol, data in current_data.items():
            theoretical_price = self._calculate_black_scholes_price(data)
            market_price = (data.bid + data.ask) / 2

            if market_price > 0:
                mispricing = (market_price - theoretical_price) / market_price

                if abs(mispricing) > 0.05:
                    direction = 'sell' if mispricing > 0 else 'buy'
                    strength = min(abs(mispricing) * 10, 1.0)
                    confidence = min(abs(mispricing) * 5, 0.9)

                    signal = TradingSignal(
                        signal_id=f"mispricing_{self._get_next_id()}",
                        symbol=symbol,
                        signal_type=SignalType.MISPRICING.value,
                        direction=direction,
                        strength=strength,
                        confidence=confidence,
                        entry_price=market_price,
                        target_price=theoretical_price,
                        stop_loss=None,
                        expiry_time=datetime.now() + timedelta(hours=6),
                        metadata={
                            'theoretical_price': theoretical_price,
                            'market_price': market_price,
                            'mispricing_pct': mispricing * 100
                        },
                        timestamp=datetime.now()
                    )
                    signals.append(signal)

        return signals

    async def _generate_momentum_signals(self, current_data: Dict[str, MarketData],
                                       historical_data: Dict[str, deque]) -> List[TradingSignal]:
        """Generate momentum-based signals"""
        signals = []

        for symbol, data in current_data.items():
            if symbol not in historical_data or len(historical_data[symbol]) < 10:
                continue

            hist_data = list(historical_data[symbol])
            prices = [md.last for md in hist_data[-10:]]

            if len(prices) < 5:
                continue

            # Calculate momentum
            short_ma = np.mean(prices[-3:])
            long_ma = np.mean(prices[-5:])
            momentum = (short_ma - long_ma) / long_ma

            # Calculate RSI-like momentum indicator
            price_changes = np.diff(prices)
            gains = [change for change in price_changes if change > 0]
            losses = [-change for change in price_changes if change < 0]

            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if avg_gain > 0 else 50

            # Generate momentum signal
            if abs(momentum) > 0.02 and (rsi > 70 or rsi < 30):
                direction = 'buy' if momentum > 0 and rsi < 30 else 'sell'
                strength = min(abs(momentum) * 20, 1.0)
                confidence = 0.6

                signal = TradingSignal(
                    signal_id=f"momentum_{self._get_next_id()}",
                    symbol=symbol,
                    signal_type=SignalType.MOMENTUM.value,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    entry_price=data.last,
                    target_price=None,
                    stop_loss=None,
                    expiry_time=datetime.now() + timedelta(hours=12),
                    metadata={
                        'momentum': momentum,
                        'rsi': rsi,
                        'short_ma': short_ma,
                        'long_ma': long_ma
                    },
                    timestamp=datetime.now()
                )
                signals.append(signal)

        return signals

    async def _generate_event_signals(self, current_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """Generate event-driven signals"""
        signals = []

        # Simplified event detection based on volume and volatility spikes
        for symbol, data in current_data.items():
            if data.volume > 10000 and data.implied_volatility > 0.3:
                signal = TradingSignal(
                    signal_id=f"event_{self._get_next_id()}",
                    symbol=symbol,
                    signal_type=SignalType.EVENT_DRIVEN.value,
                    direction='neutral',
                    strength=0.7,
                    confidence=0.5,
                    entry_price=data.last,
                    target_price=None,
                    stop_loss=None,
                    expiry_time=datetime.now() + timedelta(hours=4),
                    metadata={
                        'trigger': 'high_volume_volatility',
                        'volume': data.volume,
                        'implied_vol': data.implied_volatility
                    },
                    timestamp=datetime.now()
                )
                signals.append(signal)

        return signals

    def _calculate_black_scholes_price(self, data: MarketData) -> float:
        """Calculate Black-Scholes theoretical price"""
        from scipy.stats import norm

        S = data.underlying_price
        K = data.strike
        T = data.time_to_expiry
        r = 0.05
        sigma = data.implied_volatility

        if T <= 0:
            return max(S - K, 0) if data.option_type == 'call' else max(K - S, 0)

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if data.option_type == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0.01)

    def _get_next_id(self) -> int:
        """Get next signal ID"""
        self.signal_counter += 1
        return self.signal_counter

class ArbitrageDetector:
    """Arbitrage opportunity detection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize arbitrage detector"""
        self.logger.info("Initializing Arbitrage Detector")

    async def scan_arbitrage_opportunities(self, vol_surfaces: Dict[str, deque]) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities"""
        opportunities = []

        # Box spread arbitrage
        opportunities.extend(await self._detect_box_spread_arbitrage(vol_surfaces))

        # Calendar spread arbitrage
        opportunities.extend(await self._detect_calendar_arbitrage(vol_surfaces))

        # Volatility arbitrage
        opportunities.extend(await self._detect_volatility_arbitrage(vol_surfaces))

        return opportunities

    async def analyze_vol_surfaces(self, vol_surfaces: Dict[str, VolatilitySurface]) -> List[TradingSignal]:
        """Analyze volatility surfaces for trading signals"""
        signals = []

        for symbol, surface in vol_surfaces.items():
            # Detect surface arbitrage
            arbitrage_signals = await self._detect_surface_arbitrage(symbol, surface)
            signals.extend(arbitrage_signals)

        return signals

    async def _detect_box_spread_arbitrage(self, vol_surfaces: Dict[str, deque]) -> List[ArbitrageOpportunity]:
        """Detect box spread arbitrage opportunities"""
        opportunities = []

        for symbol, surfaces in vol_surfaces.items():
            if not surfaces:
                continue

            current_surface = surfaces[-1]

            # Simplified box spread detection
            if len(current_surface.strikes) >= 4:
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"box_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    opportunity_type="box_spread",
                    symbols=[symbol],
                    expected_profit=100.0,
                    max_risk=500.0,
                    profit_probability=0.8,
                    time_to_expiration=30.0,
                    execution_complexity="medium",
                    metadata={
                        'strikes': current_surface.strikes[:4],
                        'detection_method': 'box_spread_analysis'
                    }
                )
                opportunities.append(opportunity)

        return opportunities

    async def _detect_calendar_arbitrage(self, vol_surfaces: Dict[str, deque]) -> List[ArbitrageOpportunity]:
        """Detect calendar spread arbitrage"""
        opportunities = []

        for symbol, surfaces in vol_surfaces.items():
            if len(surfaces) < 2:
                continue

            # Compare different expiration surfaces
            surface1 = surfaces[-1]
            surface2 = surfaces[-2] if len(surfaces) > 1 else surface1

            vol_diff = np.mean(surface1.implied_vols) - np.mean(surface2.implied_vols)

            if abs(vol_diff) > 0.05:
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"calendar_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    opportunity_type="calendar_spread",
                    symbols=[symbol],
                    expected_profit=abs(vol_diff) * 1000,
                    max_risk=500.0,
                    profit_probability=0.7,
                    time_to_expiration=15.0,
                    execution_complexity="low",
                    metadata={
                        'vol_difference': vol_diff,
                        'expiry1': surface1.timestamp,
                        'expiry2': surface2.timestamp
                    }
                )
                opportunities.append(opportunity)

        return opportunities

    async def _detect_volatility_arbitrage(self, vol_surfaces: Dict[str, deque]) -> List[ArbitrageOpportunity]:
        """Detect volatility arbitrage opportunities"""
        opportunities = []

        symbols = list(vol_surfaces.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]

                if not vol_surfaces[symbol1] or not vol_surfaces[symbol2]:
                    continue

                surface1 = vol_surfaces[symbol1][-1]
                surface2 = vol_surfaces[symbol2][-1]

                # Calculate volatility spread
                vol1 = np.mean(surface1.implied_vols)
                vol2 = np.mean(surface2.implied_vols)
                vol_spread = vol1 - vol2

                if abs(vol_spread) > 0.1:
                    opportunity = ArbitrageOpportunity(
                        opportunity_id=f"vol_arb_{symbol1}_{symbol2}_{datetime.now().strftime('%H%M%S')}",
                        opportunity_type="volatility_arbitrage",
                        symbols=[symbol1, symbol2],
                        expected_profit=abs(vol_spread) * 500,
                        max_risk=1000.0,
                        profit_probability=0.6,
                        time_to_expiration=7.0,
                        execution_complexity="high",
                        metadata={
                            'vol_spread': vol_spread,
                            'vol1': vol1,
                            'vol2': vol2
                        }
                    )
                    opportunities.append(opportunity)

        return opportunities

    async def _detect_surface_arbitrage(self, symbol: str, surface: VolatilitySurface) -> List[TradingSignal]:
        """Detect arbitrage opportunities within a volatility surface"""
        signals = []

        if surface.implied_vols.size == 0:
            return signals

        # Butterfly arbitrage detection
        if len(surface.strikes) >= 3:
            for i in range(len(surface.strikes) - 2):
                if surface.implied_vols.ndim > 1:
                    left_vol = surface.implied_vols[0, i]
                    center_vol = surface.implied_vols[0, i + 1]
                    right_vol = surface.implied_vols[0, i + 2]
                else:
                    left_vol = surface.implied_vols[i]
                    center_vol = surface.implied_vols[i + 1]
                    right_vol = surface.implied_vols[i + 2]

                expected_center = (left_vol + right_vol) / 2
                vol_deviation = abs(center_vol - expected_center)

                if vol_deviation > 0.02:
                    direction = 'sell' if center_vol > expected_center else 'buy'

                    signal = TradingSignal(
                        signal_id=f"butterfly_{symbol}_{i}",
                        symbol=symbol,
                        signal_type=SignalType.ARBITRAGE.value,
                        direction=direction,
                        strength=min(vol_deviation * 20, 1.0),
                        confidence=0.8,
                        entry_price=None,
                        target_price=None,
                        stop_loss=None,
                        expiry_time=datetime.now() + timedelta(hours=12),
                        metadata={
                            'arbitrage_type': 'butterfly',
                            'strikes': [surface.strikes[i], surface.strikes[i+1], surface.strikes[i+2]],
                            'vol_deviation': vol_deviation
                        },
                        timestamp=datetime.now()
                    )
                    signals.append(signal)

        return signals

class RegimeDetector:
    """Market regime detection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: List[MarketRegime] = []

    async def initialize(self):
        """Initialize regime detector"""
        self.logger.info("Initializing Regime Detector")

    async def update_market_data(self, market_data: Dict[str, MarketData]):
        """Update regime detection with new market data"""
        regime = await self._detect_regime(market_data)

        if not self.current_regime or regime.regime_type != self.current_regime.regime_type:
            if self.current_regime:
                self.current_regime.duration = datetime.now() - self.current_regime.start_time
                self.regime_history.append(self.current_regime)

            self.current_regime = regime

    async def detect_current_regime(self, historical_data: Dict[str, deque]) -> MarketRegime:
        """Detect current market regime"""
        if self.current_regime:
            return self.current_regime

        # Default regime if no data
        return MarketRegime(
            regime_type='normal',
            confidence=0.5,
            characteristics={},
            start_time=datetime.now(),
            duration=timedelta(0),
            volatility_level='medium',
            trend_direction='neutral'
        )

    async def _detect_regime(self, market_data: Dict[str, MarketData]) -> MarketRegime:
        """Detect market regime from current data"""
        if not market_data:
            return await self.detect_current_regime({})

        # Calculate market-wide volatility
        avg_volatility = np.mean([data.implied_volatility for data in market_data.values()])

        # Determine volatility level
        if avg_volatility > 0.4:
            vol_level = 'high'
            regime_type = 'high_volatility'
        elif avg_volatility < 0.15:
            vol_level = 'low'
            regime_type = 'low_volatility'
        else:
            vol_level = 'medium'
            regime_type = 'normal'

        # Calculate trend direction
        price_changes = []
        for data in market_data.values():
            if hasattr(data, 'price_change'):
                price_changes.append(data.price_change)

        if price_changes:
            avg_change = np.mean(price_changes)
            if avg_change > 0.01:
                trend_direction = 'bullish'
            elif avg_change < -0.01:
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'
        else:
            trend_direction = 'neutral'

        return MarketRegime(
            regime_type=regime_type,
            confidence=0.7,
            characteristics={
                'avg_volatility': avg_volatility,
                'vol_dispersion': np.std([data.implied_volatility for data in market_data.values()]),
                'market_breadth': len(market_data)
            },
            start_time=datetime.now(),
            duration=timedelta(0),
            volatility_level=vol_level,
            trend_direction=trend_direction
        )

class SentimentAnalyzer:
    """Market sentiment analysis from options flow"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize sentiment analyzer"""
        self.logger.info("Initializing Sentiment Analyzer")

    async def analyze_market_sentiment(self, options_flows: deque) -> Dict[str, Any]:
        """Analyze overall market sentiment"""
        if not options_flows:
            return {'sentiment_score': 0.0, 'confidence': 0.0}

        recent_flows = [f for f in options_flows if
                       (datetime.now() - f.timestamp).total_seconds() < 3600]

        if not recent_flows:
            return {'sentiment_score': 0.0, 'confidence': 0.0}

        # Calculate sentiment metrics
        total_volume = sum(f.volume for f in recent_flows)
        call_volume = sum(f.volume for f in recent_flows if f.option_type == 'call')
        put_volume = sum(f.volume for f in recent_flows if f.option_type == 'put')

        put_call_ratio = put_volume / max(call_volume, 1)

        # Bullish vs bearish flow
        bullish_flow = sum(f.volume for f in recent_flows if f.direction == 'bullish')
        bearish_flow = sum(f.volume for f in recent_flows if f.direction == 'bearish')

        sentiment_score = (bullish_flow - bearish_flow) / max(total_volume, 1)

        # Unusual activity indicator
        unusual_activity = len([f for f in recent_flows if f.unusual_activity])
        total_trades = len(recent_flows)
        unusual_ratio = unusual_activity / max(total_trades, 1)

        confidence = min(total_volume / 10000, 1.0)

        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'put_call_ratio': put_call_ratio,
            'unusual_activity_ratio': unusual_ratio,
            'total_volume': total_volume,
            'dominant_direction': 'bullish' if sentiment_score > 0.1 else 'bearish' if sentiment_score < -0.1 else 'neutral'
        }

    async def generate_flow_signals(self, flows: List[OptionsFlow]) -> List[TradingSignal]:
        """Generate trading signals from options flow"""
        signals = []

        # Group flows by symbol
        symbol_flows = defaultdict(list)
        for flow in flows:
            symbol_flows[flow.symbol].append(flow)

        for symbol, symbol_flow_list in symbol_flows.items():
            if len(symbol_flow_list) < 3:
                continue

            # Analyze flow for this symbol
            total_volume = sum(f.volume for f in symbol_flow_list)
            unusual_flows = [f for f in symbol_flow_list if f.unusual_activity]

            if unusual_flows and total_volume > 1000:
                # Generate unusual activity signal
                dominant_direction = max(set(f.direction for f in unusual_flows),
                                       key=lambda x: sum(f.volume for f in unusual_flows if f.direction == x))

                signal = TradingSignal(
                    signal_id=f"flow_{symbol}_{datetime.now().strftime('%H%M%S')}",
                    symbol=symbol,
                    signal_type="unusual_flow",
                    direction=dominant_direction,
                    strength=min(len(unusual_flows) / 5.0, 1.0),
                    confidence=min(total_volume / 5000.0, 0.9),
                    entry_price=None,
                    target_price=None,
                    stop_loss=None,
                    expiry_time=datetime.now() + timedelta(hours=8),
                    metadata={
                        'unusual_flows': len(unusual_flows),
                        'total_volume': total_volume,
                        'avg_premium': np.mean([f.premium for f in unusual_flows])
                    },
                    timestamp=datetime.now()
                )
                signals.append(signal)

        return signals