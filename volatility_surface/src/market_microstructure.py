"""
Market Microstructure Volatility Modeling

Advanced modeling of bid-ask spreads in volatility space, volume-weighted
volatility, time-of-day patterns, event-driven spikes, and cross-asset correlations
for sophisticated volatility surface construction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import logging
from scipy import stats, signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

from .surface_models import VolatilityQuote

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Market event types that affect volatility"""
    EARNINGS = "EARNINGS"
    FOMC = "FOMC"
    EXPIRATION = "EXPIRATION"
    DIVIDEND = "DIVIDEND"
    ECONOMIC_RELEASE = "ECONOMIC_RELEASE"
    GEOPOLITICAL = "GEOPOLITICAL"
    TECHNICAL = "TECHNICAL"

class TradingSession(Enum):
    """Trading session periods"""
    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    MID_MORNING = "MID_MORNING"
    LUNCH = "LUNCH"
    AFTERNOON = "AFTERNOON"
    MARKET_CLOSE = "MARKET_CLOSE"
    AFTER_HOURS = "AFTER_HOURS"

@dataclass
class VolumetricQuote:
    """Volatility quote with volume and microstructure data"""
    base_quote: VolatilityQuote
    bid_size: int
    ask_size: int
    trade_volume: int
    trade_count: int
    vwap_vol: float  # Volume-weighted average volatility
    time_of_day: time
    session: TradingSession
    bid_ask_spread_bps: float
    effective_spread_bps: float
    price_impact: float

@dataclass
class MarketEvent:
    """Market event affecting volatility"""
    event_type: EventType
    event_time: datetime
    underlying: str
    impact_magnitude: float  # Expected volatility impact
    duration_hours: float
    confidence: float

class BidAskSpreadModel:
    """Model bid-ask spreads in volatility space"""

    def __init__(self):
        self.spread_models = {}
        self.microstructure_factors = {}

    def model_volatility_spreads(self, quotes: List[VolumetricQuote]) -> Dict[str, float]:
        """Model bid-ask spreads in volatility space"""

        spread_data = []
        for quote in quotes:
            if quote.base_quote.bid_vol and quote.base_quote.ask_vol:
                vol_spread = quote.base_quote.ask_vol - quote.base_quote.bid_vol
                vol_mid = (quote.base_quote.ask_vol + quote.base_quote.bid_vol) / 2.0

                spread_data.append({
                    'vol_spread_bps': (vol_spread / vol_mid) * 10000,
                    'moneyness': abs(quote.base_quote.log_moneyness),
                    'time_to_expiry': quote.base_quote.expiry,
                    'volume': quote.trade_volume,
                    'session': quote.session.value,
                    'trade_count': quote.trade_count
                })

        if not spread_data:
            return {}

        df = pd.DataFrame(spread_data)

        # Model spread as function of moneyness, time to expiry, and volume
        spread_factors = {
            'moneyness_effect': self._model_moneyness_effect(df),
            'time_decay_effect': self._model_time_decay_effect(df),
            'volume_effect': self._model_volume_effect(df),
            'session_effect': self._model_session_effect(df),
            'base_spread': df['vol_spread_bps'].median()
        }

        return spread_factors

    def _model_moneyness_effect(self, df: pd.DataFrame) -> float:
        """Model how moneyness affects volatility spreads"""
        try:
            # Spreads typically widen for OTM options
            correlation = df['vol_spread_bps'].corr(df['moneyness'])
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _model_time_decay_effect(self, df: pd.DataFrame) -> float:
        """Model how time to expiry affects volatility spreads"""
        try:
            # Spreads typically widen as expiration approaches
            correlation = df['vol_spread_bps'].corr(1.0 / (df['time_to_expiry'] + 0.01))
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _model_volume_effect(self, df: pd.DataFrame) -> float:
        """Model how volume affects volatility spreads"""
        try:
            # Higher volume typically means tighter spreads
            df_log_vol = df.copy()
            df_log_vol['log_volume'] = np.log(df['volume'] + 1)
            correlation = df_log_vol['vol_spread_bps'].corr(df_log_vol['log_volume'])
            return -abs(correlation) if not np.isnan(correlation) else 0.0  # Negative = tighter with volume
        except:
            return 0.0

    def _model_session_effect(self, df: pd.DataFrame) -> Dict[str, float]:
        """Model session-specific spread effects"""
        try:
            session_effects = {}
            for session in df['session'].unique():
                session_spreads = df[df['session'] == session]['vol_spread_bps']
                if len(session_spreads) > 0:
                    session_effects[session] = session_spreads.mean()

            return session_effects
        except:
            return {}

    def predict_spread(self, moneyness: float, time_to_expiry: float,
                      volume: int, session: TradingSession) -> float:
        """Predict volatility spread for given characteristics"""
        if not self.microstructure_factors:
            return 50.0  # Default 50 bps spread

        factors = self.microstructure_factors
        base_spread = factors.get('base_spread', 50.0)

        # Adjust for moneyness
        moneyness_adjustment = factors.get('moneyness_effect', 0.0) * abs(moneyness) * 20

        # Adjust for time decay
        time_adjustment = factors.get('time_decay_effect', 0.0) * (1.0 / (time_to_expiry + 0.01)) * 10

        # Adjust for volume
        volume_adjustment = factors.get('volume_effect', 0.0) * np.log(volume + 1) * 5

        # Adjust for session
        session_effects = factors.get('session_effect', {})
        session_adjustment = session_effects.get(session.value, 0.0) - base_spread

        predicted_spread = (base_spread + moneyness_adjustment + time_adjustment +
                          volume_adjustment + session_adjustment)

        return max(predicted_spread, 5.0)  # Minimum 5 bps spread

class VolumeWeightedVolatilityCalculator:
    """Calculate volume-weighted volatility measures"""

    @staticmethod
    def calculate_vwap_volatility(quotes: List[VolumetricQuote], window_minutes: int = 30) -> float:
        """Calculate volume-weighted average price volatility"""
        if not quotes:
            return 0.0

        # Sort by timestamp
        sorted_quotes = sorted(quotes, key=lambda q: q.base_quote.timestamp)

        # Filter to window
        cutoff_time = sorted_quotes[-1].base_quote.timestamp - timedelta(minutes=window_minutes)
        recent_quotes = [q for q in sorted_quotes if q.base_quote.timestamp >= cutoff_time]

        if not recent_quotes:
            return 0.0

        # Calculate VWAP volatility
        total_volume = sum(q.trade_volume for q in recent_quotes)
        if total_volume == 0:
            return np.mean([q.base_quote.implied_vol for q in recent_quotes])

        vwap_vol = sum(q.base_quote.implied_vol * q.trade_volume for q in recent_quotes) / total_volume
        return vwap_vol

    @staticmethod
    def calculate_volume_profile_volatility(quotes: List[VolumetricQuote],
                                          volume_buckets: int = 10) -> Dict[str, float]:
        """Calculate volatility by volume profile buckets"""
        if not quotes:
            return {}

        # Sort by volume
        volume_sorted = sorted(quotes, key=lambda q: q.trade_volume)
        bucket_size = len(volume_sorted) // volume_buckets

        volume_profile = {}
        for i in range(volume_buckets):
            start_idx = i * bucket_size
            end_idx = start_idx + bucket_size if i < volume_buckets - 1 else len(volume_sorted)

            bucket_quotes = volume_sorted[start_idx:end_idx]
            if bucket_quotes:
                avg_volume = np.mean([q.trade_volume for q in bucket_quotes])
                avg_volatility = np.mean([q.base_quote.implied_vol for q in bucket_quotes])
                volume_profile[f"bucket_{i+1}_{int(avg_volume)}"] = avg_volatility

        return volume_profile

    @staticmethod
    def calculate_flow_weighted_volatility(quotes: List[VolumetricQuote],
                                         decay_factor: float = 0.95) -> float:
        """Calculate exponentially weighted volatility based on order flow"""
        if not quotes:
            return 0.0

        # Sort by timestamp
        sorted_quotes = sorted(quotes, key=lambda q: q.base_quote.timestamp)

        weighted_vol = 0.0
        total_weight = 0.0

        for i, quote in enumerate(sorted_quotes):
            # Weight decays exponentially with time
            weight = (decay_factor ** (len(sorted_quotes) - i - 1)) * quote.trade_volume
            weighted_vol += quote.base_quote.implied_vol * weight
            total_weight += weight

        return weighted_vol / total_weight if total_weight > 0 else 0.0

class TimeOfDayVolatilityModel:
    """Model intraday volatility patterns"""

    def __init__(self):
        self.intraday_patterns = {}
        self.session_volatilities = {}

    def fit_intraday_patterns(self, quotes: List[VolumetricQuote], underlying: str):
        """Fit intraday volatility patterns"""
        if not quotes:
            return

        # Group quotes by time of day (rounded to nearest 15 minutes)
        time_buckets = {}
        for quote in quotes:
            time_bucket = self._round_to_quarter_hour(quote.time_of_day)
            if time_bucket not in time_buckets:
                time_buckets[time_bucket] = []
            time_buckets[time_bucket].append(quote.base_quote.implied_vol)

        # Calculate average volatility for each time bucket
        pattern = {}
        for time_bucket, vols in time_buckets.items():
            pattern[time_bucket] = {
                'mean_vol': np.mean(vols),
                'std_vol': np.std(vols),
                'count': len(vols)
            }

        self.intraday_patterns[underlying] = pattern

        # Fit session-specific patterns
        self._fit_session_patterns(quotes, underlying)

    def _round_to_quarter_hour(self, time_obj: time) -> str:
        """Round time to nearest 15-minute interval"""
        minutes = time_obj.minute
        rounded_minutes = (minutes // 15) * 15
        return f"{time_obj.hour:02d}:{rounded_minutes:02d}"

    def _fit_session_patterns(self, quotes: List[VolumetricQuote], underlying: str):
        """Fit session-specific volatility patterns"""
        session_data = {}
        for quote in quotes:
            session = quote.session
            if session not in session_data:
                session_data[session] = []
            session_data[session].append(quote.base_quote.implied_vol)

        session_stats = {}
        for session, vols in session_data.items():
            session_stats[session] = {
                'mean_vol': np.mean(vols),
                'std_vol': np.std(vols),
                'min_vol': np.min(vols),
                'max_vol': np.max(vols),
                'count': len(vols)
            }

        self.session_volatilities[underlying] = session_stats

    def get_time_adjustment_factor(self, underlying: str, current_time: time) -> float:
        """Get time-of-day adjustment factor for volatility"""
        if underlying not in self.intraday_patterns:
            return 1.0

        pattern = self.intraday_patterns[underlying]
        time_bucket = self._round_to_quarter_hour(current_time)

        if time_bucket in pattern:
            bucket_vol = pattern[time_bucket]['mean_vol']
            daily_avg_vol = np.mean([p['mean_vol'] for p in pattern.values()])
            return bucket_vol / daily_avg_vol if daily_avg_vol > 0 else 1.0

        return 1.0

    def get_session_adjustment_factor(self, underlying: str, session: TradingSession) -> float:
        """Get session-specific adjustment factor"""
        if underlying not in self.session_volatilities:
            return 1.0

        session_stats = self.session_volatilities[underlying]
        if session not in session_stats:
            return 1.0

        session_vol = session_stats[session]['mean_vol']
        daily_avg_vol = np.mean([s['mean_vol'] for s in session_stats.values()])
        return session_vol / daily_avg_vol if daily_avg_vol > 0 else 1.0

    def predict_intraday_volatility(self, underlying: str, target_time: time,
                                   base_volatility: float) -> float:
        """Predict volatility at specific time of day"""
        time_factor = self.get_time_adjustment_factor(underlying, target_time)
        return base_volatility * time_factor

class EventDrivenVolatilityModel:
    """Model volatility spikes around market events"""

    def __init__(self):
        self.event_impacts = {}
        self.event_decay_rates = {}

    def register_event(self, event: MarketEvent):
        """Register a market event for volatility modeling"""
        event_key = f"{event.underlying}_{event.event_type.value}"

        if event_key not in self.event_impacts:
            self.event_impacts[event_key] = []

        self.event_impacts[event_key].append({
            'event_time': event.event_time,
            'impact_magnitude': event.impact_magnitude,
            'duration_hours': event.duration_hours,
            'confidence': event.confidence
        })

    def calculate_event_impact(self, underlying: str, current_time: datetime,
                             base_volatility: float) -> float:
        """Calculate volatility impact from nearby events"""
        total_impact = 0.0

        for event_key, events in self.event_impacts.items():
            if not event_key.startswith(underlying):
                continue

            for event_data in events:
                event_time = event_data['event_time']
                hours_since_event = (current_time - event_time).total_seconds() / 3600

                # Only consider events within their impact duration
                if 0 <= hours_since_event <= event_data['duration_hours']:
                    # Exponential decay of impact
                    decay_rate = self._get_decay_rate(event_key)
                    impact_factor = event_data['impact_magnitude'] * np.exp(-decay_rate * hours_since_event)
                    total_impact += impact_factor * event_data['confidence']

        return base_volatility * (1.0 + total_impact)

    def _get_decay_rate(self, event_key: str) -> float:
        """Get decay rate for event type"""
        # Default decay rates by event type
        default_rates = {
            'EARNINGS': 0.5,    # Half-life of ~1.4 hours
            'FOMC': 0.1,        # Half-life of ~7 hours
            'EXPIRATION': 2.0,  # Half-life of ~20 minutes
            'DIVIDEND': 0.2,    # Half-life of ~3.5 hours
            'ECONOMIC_RELEASE': 0.3,  # Half-life of ~2.3 hours
        }

        for event_type, rate in default_rates.items():
            if event_type in event_key:
                return rate

        return 0.3  # Default decay rate

    def fit_event_impacts(self, quotes: List[VolumetricQuote], events: List[MarketEvent]):
        """Fit event impact models from historical data"""
        # Group quotes by underlying
        quotes_by_underlying = {}
        for quote in quotes:
            underlying = quote.base_quote.strike  # Simplified - would use actual underlying
            if underlying not in quotes_by_underlying:
                quotes_by_underlying[underlying] = []
            quotes_by_underlying[underlying].append(quote)

        # For each event, measure impact on surrounding volatilities
        for event in events:
            event_time = event.event_time
            pre_event_window = timedelta(hours=2)
            post_event_window = timedelta(hours=event.duration_hours)

            if event.underlying in quotes_by_underlying:
                quotes = quotes_by_underlying[event.underlying]

                # Get pre-event baseline
                pre_event_quotes = [
                    q for q in quotes
                    if event_time - pre_event_window <= q.base_quote.timestamp < event_time
                ]

                # Get post-event quotes
                post_event_quotes = [
                    q for q in quotes
                    if event_time <= q.base_quote.timestamp <= event_time + post_event_window
                ]

                if pre_event_quotes and post_event_quotes:
                    baseline_vol = np.mean([q.base_quote.implied_vol for q in pre_event_quotes])
                    post_event_vol = np.mean([q.base_quote.implied_vol for q in post_event_quotes])

                    observed_impact = (post_event_vol - baseline_vol) / baseline_vol
                    event.impact_magnitude = observed_impact

class VolatilityTermStructureModel:
    """Model and detect volatility term structure inversions"""

    def __init__(self):
        self.term_structure_history = []
        self.inversion_alerts = []

    def analyze_term_structure(self, quotes: List[VolatilityQuote],
                             underlying: str) -> Dict[str, Any]:
        """Analyze volatility term structure for given underlying"""

        # Group quotes by expiry, use ATM options
        expiry_groups = {}
        for quote in quotes:
            expiry_key = round(quote.expiry, 3)
            if expiry_key not in expiry_groups:
                expiry_groups[expiry_key] = []
            expiry_groups[expiry_key].append(quote)

        # Find ATM volatility for each expiry
        atm_vols = []
        expiries = []

        for expiry, exp_quotes in expiry_groups.items():
            if len(exp_quotes) > 0:
                # Find closest to ATM (log-moneyness = 0)
                atm_quote = min(exp_quotes, key=lambda q: abs(q.log_moneyness))
                atm_vols.append(atm_quote.implied_vol)
                expiries.append(expiry)

        if len(atm_vols) < 2:
            return {}

        # Sort by expiry
        sorted_data = sorted(zip(expiries, atm_vols))
        expiries = [x[0] for x in sorted_data]
        atm_vols = [x[1] for x in sorted_data]

        # Calculate term structure metrics
        analysis = {
            'expiries': expiries,
            'atm_volatilities': atm_vols,
            'slope': self._calculate_term_structure_slope(expiries, atm_vols),
            'curvature': self._calculate_term_structure_curvature(atm_vols),
            'inversions': self._detect_inversions(expiries, atm_vols),
            'forward_volatilities': self._calculate_forward_volatilities(expiries, atm_vols)
        }

        # Store in history
        self.term_structure_history.append({
            'timestamp': datetime.now(),
            'underlying': underlying,
            'analysis': analysis
        })

        return analysis

    def _calculate_term_structure_slope(self, expiries: List[float],
                                      atm_vols: List[float]) -> float:
        """Calculate overall slope of term structure"""
        if len(expiries) < 2:
            return 0.0

        # Linear regression slope
        correlation_matrix = np.corrcoef(expiries, atm_vols)
        if correlation_matrix.shape == (2, 2):
            slope = correlation_matrix[0, 1] * (np.std(atm_vols) / np.std(expiries))
            return slope
        return 0.0

    def _calculate_term_structure_curvature(self, atm_vols: List[float]) -> float:
        """Calculate curvature (second derivative) of term structure"""
        if len(atm_vols) < 3:
            return 0.0

        # Calculate discrete second derivative
        second_derivatives = []
        for i in range(1, len(atm_vols) - 1):
            second_deriv = atm_vols[i+1] - 2*atm_vols[i] + atm_vols[i-1]
            second_derivatives.append(abs(second_deriv))

        return np.mean(second_derivatives) if second_derivatives else 0.0

    def _detect_inversions(self, expiries: List[float], atm_vols: List[float]) -> List[Dict]:
        """Detect term structure inversions"""
        inversions = []

        for i in range(len(atm_vols) - 1):
            if atm_vols[i] > atm_vols[i + 1]:  # Shorter expiry has higher vol
                inversion_magnitude = atm_vols[i] - atm_vols[i + 1]
                inversions.append({
                    'start_expiry': expiries[i],
                    'end_expiry': expiries[i + 1],
                    'magnitude': inversion_magnitude,
                    'relative_magnitude': inversion_magnitude / atm_vols[i]
                })

        return inversions

    def _calculate_forward_volatilities(self, expiries: List[float],
                                      atm_vols: List[float]) -> List[float]:
        """Calculate forward implied volatilities"""
        if len(expiries) < 2:
            return []

        forward_vols = []
        for i in range(len(expiries) - 1):
            T1, T2 = expiries[i], expiries[i + 1]
            vol1, vol2 = atm_vols[i], atm_vols[i + 1]

            # Forward variance
            if T2 > T1:
                forward_var = (vol2**2 * T2 - vol1**2 * T1) / (T2 - T1)
                forward_vol = np.sqrt(max(forward_var, 0.01))  # Ensure positive
                forward_vols.append(forward_vol)

        return forward_vols

class CrossAssetVolatilityModel:
    """Model cross-asset volatility correlations and spillover effects"""

    def __init__(self):
        self.correlation_matrix = {}
        self.spillover_effects = {}

    def calculate_volatility_correlations(self, quotes_by_asset: Dict[str, List[VolatilityQuote]],
                                        window_days: int = 30) -> np.ndarray:
        """Calculate cross-asset volatility correlations"""
        assets = list(quotes_by_asset.keys())
        if len(assets) < 2:
            return np.array([])

        # Align volatility time series
        vol_series = {}
        for asset, quotes in quotes_by_asset.items():
            # Sort by timestamp and calculate daily average volatility
            sorted_quotes = sorted(quotes, key=lambda q: q.timestamp)

            daily_vols = {}
            for quote in sorted_quotes:
                date_key = quote.timestamp.date()
                if date_key not in daily_vols:
                    daily_vols[date_key] = []
                daily_vols[date_key].append(quote.implied_vol)

            # Average volatility per day
            vol_series[asset] = {
                date: np.mean(vols) for date, vols in daily_vols.items()
            }

        # Find common dates
        common_dates = set(vol_series[assets[0]].keys())
        for asset in assets[1:]:
            common_dates = common_dates.intersection(set(vol_series[asset].keys()))

        if len(common_dates) < 10:
            return np.eye(len(assets))  # Return identity matrix

        common_dates = sorted(list(common_dates))[-window_days:]  # Use recent data

        # Build correlation matrix
        vol_matrix = []
        for asset in assets:
            asset_vols = [vol_series[asset][date] for date in common_dates]
            vol_matrix.append(asset_vols)

        vol_matrix = np.array(vol_matrix)

        try:
            correlation_matrix = np.corrcoef(vol_matrix)
            self.correlation_matrix = dict(zip(assets, correlation_matrix))
            return correlation_matrix
        except:
            return np.eye(len(assets))

    def estimate_spillover_effects(self, source_asset: str, target_asset: str,
                                 volatility_shock: float) -> float:
        """Estimate spillover effect from one asset to another"""
        if source_asset not in self.correlation_matrix:
            return 0.0

        assets = list(self.correlation_matrix.keys())
        if source_asset not in assets or target_asset not in assets:
            return 0.0

        source_idx = assets.index(source_asset)
        target_idx = assets.index(target_asset)

        correlation = self.correlation_matrix[source_asset][target_idx]

        # Estimate spillover as correlation * shock * decay factor
        spillover_effect = correlation * volatility_shock * 0.5  # 50% transmission

        return spillover_effect

    def detect_volatility_regime_changes(self, quotes: List[VolatilityQuote],
                                       window_size: int = 20) -> List[Dict]:
        """Detect regime changes in volatility using rolling statistics"""
        if len(quotes) < window_size * 2:
            return []

        sorted_quotes = sorted(quotes, key=lambda q: q.timestamp)
        volatilities = [q.implied_vol for q in sorted_quotes]

        regime_changes = []

        # Rolling mean and standard deviation
        for i in range(window_size, len(volatilities) - window_size):
            window1 = volatilities[i-window_size:i]
            window2 = volatilities[i:i+window_size]

            mean1, std1 = np.mean(window1), np.std(window1)
            mean2, std2 = np.mean(window2), np.std(window2)

            # Statistical test for mean change
            if std1 > 0 and std2 > 0:
                t_stat = (mean2 - mean1) / np.sqrt((std1**2 + std2**2) / window_size)

                # Significant change threshold
                if abs(t_stat) > 2.0:  # Roughly 95% confidence
                    regime_changes.append({
                        'timestamp': sorted_quotes[i].timestamp,
                        'change_magnitude': mean2 - mean1,
                        'relative_change': (mean2 - mean1) / mean1 if mean1 > 0 else 0,
                        't_statistic': t_stat,
                        'confidence': min(abs(t_stat) / 2.0, 1.0)
                    })

        return regime_changes