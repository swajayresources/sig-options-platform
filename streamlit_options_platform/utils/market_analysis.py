"""
Market Analysis Module
Advanced market analysis, correlation monitoring, and sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class MarketAnalyzer:
    """Advanced market analysis and monitoring"""

    def __init__(self):
        self.assets = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer', 'Industrial']

    def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        return {
            'vix': np.random.uniform(15, 35),
            'vix_change': np.random.uniform(-3, 3),
            'put_call_ratio': np.random.uniform(0.7, 1.4),
            'iv_rank': np.random.uniform(20, 80),
            'volume_surge': np.random.uniform(0.8, 2.5),
            'market_breadth': np.random.uniform(0.3, 0.8),
            'fear_greed_index': np.random.uniform(20, 80)
        }

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Get cross-asset correlation matrix"""
        n_assets = len(self.assets)

        # Generate realistic correlation matrix
        correlations = np.random.uniform(0.3, 0.9, (n_assets, n_assets))

        # Make matrix symmetric and set diagonal to 1
        for i in range(n_assets):
            correlations[i, i] = 1.0
            for j in range(i + 1, n_assets):
                correlations[j, i] = correlations[i, j]

        # Add some negative correlations for realism
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if np.random.random() < 0.1:  # 10% chance of negative correlation
                    correlations[i, j] = correlations[j, i] = -np.random.uniform(0.1, 0.4)

        return {
            'assets': self.assets,
            'matrix': correlations.tolist()
        }

    def get_term_structure_comparison(self) -> Dict[str, Any]:
        """Get volatility term structure comparison across assets"""
        days_to_expiry = [7, 14, 21, 30, 45, 60, 90]
        volatilities = {}

        for asset in self.assets[:5]:  # First 5 assets for clarity
            # Generate realistic term structure
            base_vol = np.random.uniform(0.15, 0.45)
            term_structure = []

            for days in days_to_expiry:
                # Add term structure effect (typically upward sloping)
                vol = base_vol + (days / 365) * np.random.uniform(0.02, 0.08)
                # Add some noise
                vol += np.random.normal(0, 0.02)
                term_structure.append(max(0.05, vol) * 100)  # Convert to percentage

            volatilities[asset] = term_structure

        return {
            'assets': list(volatilities.keys()),
            'days_to_expiry': days_to_expiry,
            'volatilities': volatilities
        }

    def get_sector_analysis(self) -> pd.DataFrame:
        """Get sector rotation analysis"""
        sector_data = []

        for sector in self.sectors:
            sector_data.append({
                'sector': sector,
                'momentum': np.random.uniform(-0.1, 0.15),
                'volatility': np.random.uniform(0.15, 0.35),
                'volume': np.random.uniform(0.5, 2.0),
                'performance': np.random.uniform(-0.05, 0.12)
            })

        return pd.DataFrame(sector_data)

    def get_iv_rank_distribution(self) -> pd.DataFrame:
        """Get IV rank distribution across market"""
        # Generate IV rank data for multiple symbols
        iv_ranks = []

        for _ in range(100):  # Sample of 100 symbols
            iv_ranks.append({
                'symbol': f'STOCK_{np.random.randint(1, 1000)}',
                'iv_rank': np.random.beta(2, 2) * 100  # Beta distribution for realistic IV ranks
            })

        return pd.DataFrame(iv_ranks)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active market alerts"""
        alerts = []

        # VIX spike alert
        if np.random.random() < 0.3:
            alerts.append({
                'title': 'VIX Spike Alert',
                'message': f'VIX increased to {np.random.uniform(25, 40):.1f}, indicating elevated market stress',
                'severity': 'HIGH',
                'timestamp': datetime.now(),
                'category': 'volatility'
            })

        # Unusual volume alert
        if np.random.random() < 0.4:
            symbol = np.random.choice(self.assets)
            alerts.append({
                'title': 'Unusual Volume Alert',
                'message': f'{symbol} showing {np.random.uniform(2, 5):.1f}x normal volume',
                'severity': 'MEDIUM',
                'timestamp': datetime.now(),
                'category': 'volume'
            })

        # Correlation breakdown alert
        if np.random.random() < 0.2:
            alerts.append({
                'title': 'Correlation Breakdown',
                'message': 'Significant change in cross-asset correlations detected',
                'severity': 'MEDIUM',
                'timestamp': datetime.now(),
                'category': 'correlation'
            })

        # Put/Call ratio alert
        if np.random.random() < 0.25:
            pc_ratio = np.random.uniform(1.2, 2.0)
            alerts.append({
                'title': 'High Put/Call Ratio',
                'message': f'Put/Call ratio at {pc_ratio:.2f}, indicating bearish sentiment',
                'severity': 'MEDIUM',
                'timestamp': datetime.now(),
                'category': 'sentiment'
            })

        return alerts

    def detect_unusual_activity(self, flow_data: pd.DataFrame) -> pd.DataFrame:
        """Detect unusual options activity"""
        if flow_data.empty:
            return pd.DataFrame()

        unusual_activities = []

        # Group by symbol and analyze
        for symbol in flow_data['symbol'].unique():
            symbol_data = flow_data[flow_data['symbol'] == symbol]

            # Check for volume spikes
            avg_volume = symbol_data['volume'].mean()
            max_volume = symbol_data['volume'].max()

            if max_volume > avg_volume * 3:
                unusual_activities.append({
                    'symbol': symbol,
                    'description': f'Volume spike: {max_volume:,.0f} contracts ({max_volume/avg_volume:.1f}x average)',
                    'severity': 'HIGH' if max_volume > avg_volume * 5 else 'MEDIUM',
                    'category': 'volume'
                })

            # Check for large premium trades
            large_premium = symbol_data[symbol_data['premium'] > symbol_data['premium'].quantile(0.9)]
            if not large_premium.empty:
                max_premium = large_premium['premium'].max()
                unusual_activities.append({
                    'symbol': symbol,
                    'description': f'Large premium trade: ${max_premium:,.0f}',
                    'severity': 'MEDIUM',
                    'category': 'premium'
                })

            # Check for directional flow
            call_volume = symbol_data[symbol_data['type'] == 'CALL']['volume'].sum()
            put_volume = symbol_data[symbol_data['type'] == 'PUT']['volume'].sum()

            if call_volume > 0 or put_volume > 0:
                ratio = put_volume / max(call_volume, 1)
                if ratio > 3 or ratio < 0.3:
                    direction = 'Bearish' if ratio > 3 else 'Bullish'
                    unusual_activities.append({
                        'symbol': symbol,
                        'description': f'{direction} flow detected (P/C ratio: {ratio:.2f})',
                        'severity': 'MEDIUM',
                        'category': 'directional'
                    })

        return pd.DataFrame(unusual_activities)

    def get_sentiment_metrics(self) -> Dict[str, float]:
        """Get market sentiment metrics"""
        return {
            'put_call_sentiment': np.random.uniform(20, 80),
            'volume_sentiment': np.random.uniform(30, 85),
            'fear_greed_index': np.random.uniform(15, 85),
            'vix_sentiment': np.random.uniform(25, 75),
            'momentum_sentiment': np.random.uniform(35, 80)
        }

    def calculate_volatility_rank(self, current_vol: float, historical_vols: List[float]) -> float:
        """Calculate volatility rank (percentile)"""
        if not historical_vols:
            return 50.0

        return (sum(1 for vol in historical_vols if vol < current_vol) / len(historical_vols)) * 100

    def analyze_options_skew(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze options volatility skew"""
        if options_data.empty:
            return {}

        calls = options_data[options_data['Type'] == 'CALL']
        puts = options_data[options_data['Type'] == 'PUT']

        if calls.empty or puts.empty:
            return {}

        # Calculate skew metrics
        call_iv_mean = calls['IV'].mean()
        put_iv_mean = puts['IV'].mean()

        skew = put_iv_mean - call_iv_mean

        # ATM volatility
        current_price = 100  # Placeholder
        atm_calls = calls.iloc[(calls['Strike'] - current_price).abs().argsort()[:1]]
        atm_puts = puts.iloc[(puts['Strike'] - current_price).abs().argsort()[:1]]

        atm_vol = (atm_calls['IV'].iloc[0] + atm_puts['IV'].iloc[0]) / 2 if not atm_calls.empty and not atm_puts.empty else 0

        return {
            'put_call_skew': skew,
            'atm_volatility': atm_vol,
            'call_vol_mean': call_iv_mean,
            'put_vol_mean': put_iv_mean,
            'vol_smile_curvature': self._calculate_smile_curvature(options_data)
        }

    def _calculate_smile_curvature(self, options_data: pd.DataFrame) -> float:
        """Calculate volatility smile curvature"""
        if len(options_data) < 3:
            return 0.0

        # Simple curvature approximation
        sorted_data = options_data.sort_values('Strike')
        if len(sorted_data) >= 3:
            vols = sorted_data['IV'].values
            # Second derivative approximation
            curvature = np.mean(np.diff(vols, 2))
            return curvature
        return 0.0

    def calculate_term_structure_slope(self, vol_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate volatility term structure slope"""
        slopes = {}

        for asset, vols in vol_data.items():
            if len(vols) >= 2:
                # Simple slope calculation (long-term - short-term)
                slope = vols[-1] - vols[0]
                slopes[asset] = slope

        return slopes

    def detect_regime_change(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market regime changes"""
        vix_level = market_data.get('vix', 20)
        correlation_avg = market_data.get('correlation_avg', 0.5)
        volume_surge = market_data.get('volume_surge', 1.0)

        # Simple regime classification
        if vix_level > 30 and correlation_avg > 0.7:
            regime = 'crisis'
            confidence = 0.8
        elif vix_level > 25 and volume_surge > 1.5:
            regime = 'stress'
            confidence = 0.7
        elif vix_level < 15 and correlation_avg < 0.4:
            regime = 'complacency'
            confidence = 0.6
        else:
            regime = 'normal'
            confidence = 0.5

        return {
            'current_regime': regime,
            'confidence': confidence,
            'indicators': {
                'vix_level': vix_level,
                'correlation': correlation_avg,
                'volume': volume_surge
            }
        }

    def calculate_risk_on_off(self) -> Dict[str, Any]:
        """Calculate risk-on/risk-off sentiment"""
        # Risk-on assets performance
        risk_on_performance = np.random.uniform(-0.02, 0.03)

        # Risk-off assets performance
        risk_off_performance = np.random.uniform(-0.01, 0.02)

        risk_sentiment = risk_on_performance - risk_off_performance

        if risk_sentiment > 0.01:
            sentiment = 'risk_on'
        elif risk_sentiment < -0.01:
            sentiment = 'risk_off'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'score': risk_sentiment,
            'risk_on_performance': risk_on_performance,
            'risk_off_performance': risk_off_performance,
            'confidence': min(abs(risk_sentiment) * 10, 1.0)
        }

    def analyze_cross_asset_momentum(self) -> Dict[str, Any]:
        """Analyze momentum across asset classes"""
        asset_classes = {
            'equities': ['SPY', 'QQQ', 'IWM'],
            'bonds': ['TLT', 'IEF', 'SHY'],
            'commodities': ['GLD', 'SLV', 'USO'],
            'currencies': ['UUP', 'FXE', 'FXY']
        }

        momentum_data = {}

        for asset_class, symbols in asset_classes.items():
            # Simulate momentum scores
            momentum_scores = [np.random.uniform(-0.1, 0.1) for _ in symbols]
            momentum_data[asset_class] = {
                'avg_momentum': np.mean(momentum_scores),
                'momentum_breadth': len([s for s in momentum_scores if s > 0]) / len(momentum_scores),
                'max_momentum': max(momentum_scores),
                'min_momentum': min(momentum_scores)
            }

        return momentum_data

    def get_earnings_calendar_impact(self) -> Dict[str, Any]:
        """Analyze upcoming earnings impact on volatility"""
        # Simulate earnings calendar
        earnings_dates = []
        base_date = datetime.now()

        for i in range(7):  # Next 7 days
            date = base_date + timedelta(days=i)
            symbols_count = np.random.poisson(3)  # Average 3 earnings per day

            for _ in range(symbols_count):
                symbol = np.random.choice(self.assets)
                earnings_dates.append({
                    'date': date,
                    'symbol': symbol,
                    'expected_move': np.random.uniform(0.03, 0.12),
                    'iv_rank': np.random.uniform(30, 80)
                })

        return {
            'upcoming_earnings': earnings_dates,
            'high_impact_count': len([e for e in earnings_dates if e['expected_move'] > 0.08]),
            'total_earnings': len(earnings_dates)
        }