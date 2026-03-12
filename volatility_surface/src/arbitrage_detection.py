"""
Arbitrage Detection and Surface Monitoring System

Advanced arbitrage detection algorithms for volatility surfaces including
calendar spreads, butterfly arbitrage, forward volatility constraints,
and real-time surface monitoring with alerting capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from scipy import optimize, interpolate
from scipy.stats import norm

from.surface_models import VolatilityQuote, VolatilitySurfaceModel
from.interpolation_methods import VolatilityInterpolator

logger = logging.getLogger(__name__)

class ArbitrageType(Enum):
 """Types of arbitrage violations"""
 CALENDAR_SPREAD = "CALENDAR_SPREAD"
 BUTTERFLY_SPREAD = "BUTTERFLY_SPREAD"
 FORWARD_VOLATILITY = "FORWARD_VOLATILITY"
 PUT_CALL_PARITY = "PUT_CALL_PARITY"
 CONVEXITY = "CONVEXITY"
 SMILE_ASYMMETRY = "SMILE_ASYMMETRY"
 TERM_STRUCTURE_INVERSION = "TERM_STRUCTURE_INVERSION"

class SeverityLevel(Enum):
 """Arbitrage violation severity levels"""
 LOW = "LOW"
 MEDIUM = "MEDIUM"
 HIGH = "HIGH"
 CRITICAL = "CRITICAL"

@dataclass
class ArbitrageViolation:
 """Arbitrage violation detection result"""
 violation_type: ArbitrageType
 severity: SeverityLevel
 location: Tuple[float, float] # (log_moneyness, time_to_expiry)
 magnitude: float
 description: str
 timestamp: datetime
 confidence: float
 affected_strikes: List[float] = field(default_factory=list)
 affected_expiries: List[float] = field(default_factory=list)
 suggested_correction: Optional[str] = None

@dataclass
class SurfaceHealth:
 """Overall surface health metrics"""
 arbitrage_score: float # 0 = many violations, 1 = arbitrage-free
 smoothness_score: float
 coverage_score: float
 stability_score: float
 overall_score: float
 violation_count: int
 critical_violations: int
 last_updated: datetime

class ArbitrageDetector:
 """Base class for arbitrage detection algorithms"""

 def __init__(self, name: str, tolerance: float = 1e-6):
 self.name = name
 self.tolerance = tolerance
 self.violation_history = []

 def detect(self, quotes: List[VolatilityQuote],
 interpolator: Optional[VolatilityInterpolator] = None) -> List[ArbitrageViolation]:
 """Detect arbitrage violations"""
 raise NotImplementedError

 def _classify_severity(self, magnitude: float, threshold_low: float = 0.01,
 threshold_medium: float = 0.05, threshold_high: float = 0.1) -> SeverityLevel:
 """Classify violation severity based on magnitude"""
 if magnitude >= threshold_high:
 return SeverityLevel.CRITICAL
 elif magnitude >= threshold_medium:
 return SeverityLevel.HIGH
 elif magnitude >= threshold_low:
 return SeverityLevel.MEDIUM
 else:
 return SeverityLevel.LOW

class CalendarSpreadArbitrageDetector(ArbitrageDetector):
 """Detect calendar spread arbitrage violations"""

 def __init__(self, tolerance: float = 1e-6):
 super().__init__("CalendarSpread", tolerance)

 def detect(self, quotes: List[VolatilityQuote],
 interpolator: Optional[VolatilityInterpolator] = None) -> List[ArbitrageViolation]:
 """Detect calendar spread arbitrage"""
 violations = []

 # Group quotes by strike
 strike_groups = {}
 for quote in quotes:
 strike_key = round(quote.strike, 2)
 if strike_key not in strike_groups:
 strike_groups[strike_key] = []
 strike_groups[strike_key].append(quote)

 # Check each strike for calendar arbitrage
 for strike, strike_quotes in strike_groups.items():
 if len(strike_quotes) < 2:
 continue

 # Sort by expiry
 sorted_quotes = sorted(strike_quotes, key=lambda q: q.expiry)

 # Check total variance non-decreasing condition
 for i in range(len(sorted_quotes) - 1):
 quote1, quote2 = sorted_quotes[i], sorted_quotes[i + 1]

 # Total variance = vol² × time
 total_var1 = quote1.implied_vol ** 2 * quote1.expiry
 total_var2 = quote2.implied_vol ** 2 * quote2.expiry

 # Calendar arbitrage: total variance must be non-decreasing
 if total_var2 < total_var1:
 violation_magnitude = (total_var1 - total_var2) / total_var1

 violation = ArbitrageViolation(
 violation_type=ArbitrageType.CALENDAR_SPREAD,
 severity=self._classify_severity(violation_magnitude, 0.005, 0.02, 0.05),
 location=(quote1.log_moneyness, quote1.expiry),
 magnitude=violation_magnitude,
 description=f"Calendar arbitrage at strike {strike:.2f}: "
 f"shorter expiry ({quote1.expiry:.3f}) has higher total variance "
 f"than longer expiry ({quote2.expiry:.3f})",
 timestamp=datetime.now(),
 confidence=0.95,
 affected_strikes=[strike],
 affected_expiries=[quote1.expiry, quote2.expiry],
 suggested_correction="Increase longer-dated volatility or decrease shorter-dated volatility"
 )

 violations.append(violation)

 return violations

class ButterflyArbitrageDetector(ArbitrageDetector):
 """Detect butterfly spread arbitrage violations"""

 def __init__(self, tolerance: float = 1e-6):
 super().__init__("ButterflySpread", tolerance)

 def detect(self, quotes: List[VolatilityQuote],
 interpolator: Optional[VolatilityInterpolator] = None) -> List[ArbitrageViolation]:
 """Detect butterfly arbitrage using density constraints"""
 violations = []

 # Group quotes by expiry
 expiry_groups = {}
 for quote in quotes:
 expiry_key = round(quote.expiry, 4)
 if expiry_key not in expiry_groups:
 expiry_groups[expiry_key] = []
 expiry_groups[expiry_key].append(quote)

 # Check each expiry for butterfly arbitrage
 for expiry, exp_quotes in expiry_groups.items():
 if len(exp_quotes) < 3:
 continue

 # Sort by strike
 sorted_quotes = sorted(exp_quotes, key=lambda q: q.strike)

 # Check convexity condition using three-point stencil
 for i in range(1, len(sorted_quotes) - 1):
 quote_low = sorted_quotes[i - 1]
 quote_mid = sorted_quotes[i]
 quote_high = sorted_quotes[i + 1]

 # Calculate implied densities using Breeden-Litzenberger formula
 density_violation = self._check_density_constraint(
 quote_low, quote_mid, quote_high
 )

 if density_violation > self.tolerance:
 violation = ArbitrageViolation(
 violation_type=ArbitrageType.BUTTERFLY_SPREAD,
 severity=self._classify_severity(density_violation, 0.01, 0.05, 0.1),
 location=(quote_mid.log_moneyness, expiry),
 magnitude=density_violation,
 description=f"Butterfly arbitrage at strikes "
 f"[{quote_low.strike:.2f}, {quote_mid.strike:.2f}, {quote_high.strike:.2f}]: "
 f"negative implied density detected",
 timestamp=datetime.now(),
 confidence=0.9,
 affected_strikes=[quote_low.strike, quote_mid.strike, quote_high.strike],
 affected_expiries=[expiry],
 suggested_correction="Adjust middle strike volatility to ensure positive density"
 )

 violations.append(violation)

 return violations

 def _check_density_constraint(self, quote_low: VolatilityQuote,
 quote_mid: VolatilityQuote,
 quote_high: VolatilityQuote) -> float:
 """Check risk-neutral density constraint using simplified Breeden-Litzenberger"""
 try:
 K_low, K_mid, K_high = quote_low.strike, quote_mid.strike, quote_high.strike
 vol_low, vol_mid, vol_high = quote_low.implied_vol, quote_mid.implied_vol, quote_high.implied_vol
 T = quote_mid.expiry

 # Simplified call price calculation (assuming r=0, q=0, S=K_mid for ATM)
 S = K_mid # Assuming ATM
 r, q = 0.02, 0.0 # Typical values

 # Black-Scholes call prices
 def bs_call_price(S, K, T, r, q, vol):
 if T <= 0 or vol <= 0:
 return max(S - K, 0)
 d1 = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
 d2 = d1 - vol*np.sqrt(T)
 return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

 C_low = bs_call_price(S, K_low, T, r, q, vol_low)
 C_mid = bs_call_price(S, K_mid, T, r, q, vol_mid)
 C_high = bs_call_price(S, K_high, T, r, q, vol_high)

 # Second derivative approximation
 h1 = K_mid - K_low
 h2 = K_high - K_mid

 if h1 <= 0 or h2 <= 0:
 return 0.0

 # Uneven grid second derivative
 second_deriv = (2 * C_low / (h1 * (h1 + h2)) -
 2 * C_mid / (h1 * h2) +
 2 * C_high / (h2 * (h1 + h2)))

 # Risk-neutral density
 density = np.exp(r * T) * second_deriv

 # Return violation magnitude if density is negative
 return max(-density, 0.0)

 except Exception as e:
 logger.warning(f"Error checking density constraint: {e}")
 return 0.0

class ForwardVolatilityArbitrageDetector(ArbitrageDetector):
 """Detect forward volatility arbitrage violations"""

 def __init__(self, tolerance: float = 1e-6):
 super().__init__("ForwardVolatility", tolerance)

 def detect(self, quotes: List[VolatilityQuote],
 interpolator: Optional[VolatilityInterpolator] = None) -> List[ArbitrageViolation]:
 """Detect forward volatility arbitrage"""
 violations = []

 # Group quotes by strike
 strike_groups = {}
 for quote in quotes:
 strike_key = round(quote.strike, 2)
 if strike_key not in strike_groups:
 strike_groups[strike_key] = []
 strike_groups[strike_key].append(quote)

 # Check forward volatilities for each strike
 for strike, strike_quotes in strike_groups.items():
 if len(strike_quotes) < 3:
 continue

 # Sort by expiry
 sorted_quotes = sorted(strike_quotes, key=lambda q: q.expiry)

 # Calculate forward volatilities
 for i in range(len(sorted_quotes) - 2):
 quote1 = sorted_quotes[i]
 quote2 = sorted_quotes[i + 1]
 quote3 = sorted_quotes[i + 2]

 # Forward volatility from T1 to T2
 fwd_vol_12 = self._calculate_forward_volatility(quote1, quote2)
 # Forward volatility from T2 to T3
 fwd_vol_23 = self._calculate_forward_volatility(quote2, quote3)

 # Check for unrealistic forward volatility patterns
 if fwd_vol_12 > 0 and fwd_vol_23 > 0:
 # Forward volatilities shouldn't vary too dramatically
 ratio = max(fwd_vol_12, fwd_vol_23) / min(fwd_vol_12, fwd_vol_23)

 if ratio > 3.0: # More than 3x difference suggests arbitrage
 violation_magnitude = (ratio - 1.0) / ratio

 violation = ArbitrageViolation(
 violation_type=ArbitrageType.FORWARD_VOLATILITY,
 severity=self._classify_severity(violation_magnitude, 0.5, 1.0, 2.0),
 location=(quote2.log_moneyness, quote2.expiry),
 magnitude=violation_magnitude,
 description=f"Forward volatility arbitrage at strike {strike:.2f}: "
 f"forward vol {fwd_vol_12:.1%} to {fwd_vol_23:.1%} "
 f"(ratio: {ratio:.2f})",
 timestamp=datetime.now(),
 confidence=0.8,
 affected_strikes=[strike],
 affected_expiries=[quote1.expiry, quote2.expiry, quote3.expiry],
 suggested_correction="Smooth forward volatility term structure"
 )

 violations.append(violation)

 return violations

 def _calculate_forward_volatility(self, quote1: VolatilityQuote, quote2: VolatilityQuote) -> float:
 """Calculate forward volatility between two expiries"""
 try:
 T1, T2 = quote1.expiry, quote2.expiry
 vol1, vol2 = quote1.implied_vol, quote2.implied_vol

 if T2 <= T1:
 return 0.0

 # Forward variance
 forward_variance = (vol2**2 * T2 - vol1**2 * T1) / (T2 - T1)

 if forward_variance <= 0:
 return 0.0

 return np.sqrt(forward_variance)

 except Exception:
 return 0.0

class ConvexityArbitrageDetector(ArbitrageDetector):
 """Detect convexity violations in volatility smile"""

 def __init__(self, tolerance: float = 1e-6):
 super().__init__("Convexity", tolerance)

 def detect(self, quotes: List[VolatilityQuote],
 interpolator: Optional[VolatilityInterpolator] = None) -> List[ArbitrageViolation]:
 """Detect convexity violations"""
 violations = []

 if not interpolator:
 return violations

 # Group by expiry
 expiry_groups = {}
 for quote in quotes:
 expiry_key = round(quote.expiry, 4)
 if expiry_key not in expiry_groups:
 expiry_groups[expiry_key] = []
 expiry_groups[expiry_key].append(quote)

 # Check convexity for each expiry
 for expiry, exp_quotes in expiry_groups.items():
 if len(exp_quotes) < 5:
 continue

 # Create dense grid for convexity checking
 k_values = np.array([q.log_moneyness for q in exp_quotes])
 k_min, k_max = k_values.min(), k_values.max()
 k_grid = np.linspace(k_min, k_max, 50)

 try:
 # Interpolate volatilities on grid
 vol_results = interpolator.interpolate(k_grid, np.full_like(k_grid, expiry))
 vol_grid = vol_results.volatility

 # Calculate second derivative (convexity)
 second_derivatives = np.gradient(np.gradient(vol_grid))

 # Find points with negative convexity (concave regions)
 concave_points = second_derivatives < -self.tolerance

 if np.any(concave_points):
 # Find clusters of convexity violations
 violation_indices = np.where(concave_points)[0]

 for idx in violation_indices:
 violation_magnitude = abs(second_derivatives[idx])

 violation = ArbitrageViolation(
 violation_type=ArbitrageType.CONVEXITY,
 severity=self._classify_severity(violation_magnitude, 0.1, 0.5, 1.0),
 location=(k_grid[idx], expiry),
 magnitude=violation_magnitude,
 description=f"Convexity violation at log-moneyness {k_grid[idx]:.3f}, "
 f"expiry {expiry:.3f}: negative curvature detected",
 timestamp=datetime.now(),
 confidence=0.85,
 affected_expiries=[expiry],
 suggested_correction="Smooth volatility smile to ensure convexity"
 )

 violations.append(violation)

 except Exception as e:
 logger.warning(f"Error checking convexity for expiry {expiry}: {e}")

 return violations

class SurfaceMonitor:
 """Real-time volatility surface monitoring system"""

 def __init__(self, update_interval: float = 60.0):
 self.update_interval = update_interval
 self.detectors = [
 CalendarSpreadArbitrageDetector(),
 ButterflyArbitrageDetector(),
 ForwardVolatilityArbitrageDetector(),
 ConvexityArbitrageDetector()
 ]
 self.violation_history = []
 self.health_history = []
 self.is_monitoring = False
 self.alert_callbacks = []

 def add_alert_callback(self, callback: Callable[[ArbitrageViolation], None]):
 """Add callback function for alerts"""
 self.alert_callbacks.append(callback)

 async def start_monitoring(self, quote_provider: Callable[[], List[VolatilityQuote]],
 interpolator_provider: Callable[[], VolatilityInterpolator]):
 """Start real-time surface monitoring"""
 self.is_monitoring = True
 logger.info("Surface monitoring started")

 while self.is_monitoring:
 try:
 await self._monitor_cycle(quote_provider, interpolator_provider)
 await asyncio.sleep(self.update_interval)
 except Exception as e:
 logger.error(f"Error in monitoring cycle: {e}")
 await asyncio.sleep(30) # Wait before retrying

 def stop_monitoring(self):
 """Stop surface monitoring"""
 self.is_monitoring = False
 logger.info("Surface monitoring stopped")

 async def _monitor_cycle(self, quote_provider: Callable[[], List[VolatilityQuote]],
 interpolator_provider: Callable[[], VolatilityInterpolator]):
 """Single monitoring cycle"""
 try:
 # Get current quotes and interpolator
 quotes = quote_provider()
 interpolator = interpolator_provider()

 if not quotes:
 return

 # Run all detectors
 all_violations = []
 for detector in self.detectors:
 violations = detector.detect(quotes, interpolator)
 all_violations.extend(violations)

 # Calculate surface health
 health = self._calculate_surface_health(quotes, all_violations)

 # Store results
 self.violation_history.extend(all_violations)
 self.health_history.append(health)

 # Trigger alerts for critical violations
 for violation in all_violations:
 if violation.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
 await self._trigger_alert(violation)

 # Log summary
 if all_violations:
 critical_count = sum(1 for v in all_violations if v.severity == SeverityLevel.CRITICAL)
 high_count = sum(1 for v in all_violations if v.severity == SeverityLevel.HIGH)

 logger.warning(f"Surface health check: {len(all_violations)} violations "
 f"({critical_count} critical, {high_count} high)")

 except Exception as e:
 logger.error(f"Error in monitoring cycle: {e}")

 def _calculate_surface_health(self, quotes: List[VolatilityQuote],
 violations: List[ArbitrageViolation]) -> SurfaceHealth:
 """Calculate overall surface health metrics"""
 # Arbitrage score (1 = no violations, 0 = many violations)
 critical_violations = sum(1 for v in violations if v.severity == SeverityLevel.CRITICAL)
 high_violations = sum(1 for v in violations if v.severity == SeverityLevel.HIGH)

 arbitrage_penalty = critical_violations * 0.2 + high_violations * 0.1
 arbitrage_score = max(0.0, 1.0 - arbitrage_penalty)

 # Smoothness score (based on volatility gradient consistency)
 smoothness_score = self._calculate_smoothness_score(quotes)

 # Coverage score (based on quote density)
 coverage_score = self._calculate_coverage_score(quotes)

 # Stability score (based on recent volatility changes)
 stability_score = self._calculate_stability_score(quotes)

 # Overall score (weighted average)
 overall_score = (0.4 * arbitrage_score +
 0.2 * smoothness_score +
 0.2 * coverage_score +
 0.2 * stability_score)

 return SurfaceHealth(
 arbitrage_score=arbitrage_score,
 smoothness_score=smoothness_score,
 coverage_score=coverage_score,
 stability_score=stability_score,
 overall_score=overall_score,
 violation_count=len(violations),
 critical_violations=critical_violations,
 last_updated=datetime.now()
 )

 def _calculate_smoothness_score(self, quotes: List[VolatilityQuote]) -> float:
 """Calculate smoothness score based on volatility gradients"""
 try:
 if len(quotes) < 3:
 return 0.5

 # Group by expiry and calculate smile smoothness
 expiry_groups = {}
 for quote in quotes:
 expiry_key = round(quote.expiry, 4)
 if expiry_key not in expiry_groups:
 expiry_groups[expiry_key] = []
 expiry_groups[expiry_key].append(quote)

 smoothness_scores = []
 for expiry, exp_quotes in expiry_groups.items():
 if len(exp_quotes) >= 3:
 # Sort by strike
 sorted_quotes = sorted(exp_quotes, key=lambda q: q.strike)
 vols = [q.implied_vol for q in sorted_quotes]

 # Calculate volatility gradients
 gradients = np.diff(vols)
 gradient_changes = np.diff(gradients)

 # Smoothness = 1 / (1 + std of gradient changes)
 if len(gradient_changes) > 0:
 smoothness = 1.0 / (1.0 + np.std(gradient_changes))
 smoothness_scores.append(smoothness)

 return np.mean(smoothness_scores) if smoothness_scores else 0.5

 except Exception:
 return 0.5

 def _calculate_coverage_score(self, quotes: List[VolatilityQuote]) -> float:
 """Calculate coverage score based on quote density"""
 try:
 if not quotes:
 return 0.0

 # Calculate coverage in strike-expiry space
 strikes = [q.strike for q in quotes]
 expiries = [q.expiry for q in quotes]

 strike_range = max(strikes) - min(strikes) if len(set(strikes)) > 1 else 0
 expiry_range = max(expiries) - min(expiries) if len(set(expiries)) > 1 else 0

 # Normalize by typical ranges
 strike_coverage = min(strike_range / 100.0, 1.0) # Assume 100 strike range is good
 expiry_coverage = min(expiry_range / 2.0, 1.0) # Assume 2 year range is good

 return (strike_coverage + expiry_coverage) / 2.0

 except Exception:
 return 0.5

 def _calculate_stability_score(self, quotes: List[VolatilityQuote]) -> float:
 """Calculate stability score based on recent changes"""
 # This would compare with historical data
 # For now, return neutral score
 return 0.8

 async def _trigger_alert(self, violation: ArbitrageViolation):
 """Trigger alert for violation"""
 for callback in self.alert_callbacks:
 try:
 if asyncio.iscoroutinefunction(callback):
 await callback(violation)
 else:
 callback(violation)
 except Exception as e:
 logger.error(f"Error in alert callback: {e}")

 def get_surface_report(self, lookback_hours: int = 24) -> Dict[str, Any]:
 """Get comprehensive surface health report"""
 cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

 recent_violations = [
 v for v in self.violation_history
 if v.timestamp >= cutoff_time
 ]

 recent_health = [
 h for h in self.health_history
 if h.last_updated >= cutoff_time
 ]

 # Violation summary by type and severity
 violation_summary = {}
 for v_type in ArbitrageType:
 violation_summary[v_type.value] = {
 'count': sum(1 for v in recent_violations if v.violation_type == v_type),
 'critical': sum(1 for v in recent_violations
 if v.violation_type == v_type and v.severity == SeverityLevel.CRITICAL),
 'high': sum(1 for v in recent_violations
 if v.violation_type == v_type and v.severity == SeverityLevel.HIGH)
 }

 # Health trend
 if recent_health:
 current_health = recent_health[-1]
 health_trend = {
 'current_score': current_health.overall_score,
 'arbitrage_score': current_health.arbitrage_score,
 'smoothness_score': current_health.smoothness_score,
 'coverage_score': current_health.coverage_score,
 'stability_score': current_health.stability_score
 }
 else:
 health_trend = {
 'current_score': 0.0,
 'arbitrage_score': 0.0,
 'smoothness_score': 0.0,
 'coverage_score': 0.0,
 'stability_score': 0.0
 }

 return {
 'report_timestamp': datetime.now(),
 'lookback_hours': lookback_hours,
 'total_violations': len(recent_violations),
 'violation_summary': violation_summary,
 'health_trend': health_trend,
 'monitoring_status': self.is_monitoring,
 'recent_violations': [
 {
 'type': v.violation_type.value,
 'severity': v.severity.value,
 'magnitude': v.magnitude,
 'description': v.description,
 'timestamp': v.timestamp.isoformat()
 }
 for v in recent_violations[-10:] # Last 10 violations
 ]
 }

# Factory function for creating monitoring systems
def create_surface_monitor(update_interval: float = 60.0,
 detectors: Optional[List[ArbitrageDetector]] = None) -> SurfaceMonitor:
 """Create surface monitoring system with specified configuration"""
 monitor = SurfaceMonitor(update_interval)

 if detectors:
 monitor.detectors = detectors

 return monitor