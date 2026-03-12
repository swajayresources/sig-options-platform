"""
Interactive Charting and Visualization Components
Professional-grade visualization tools for options analytics
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import asyncio
import logging
import json

from analytics_framework import MarketData, Position, PortfolioGreeks, VolatilitySurface
from portfolio_monitor import PortfolioSnapshot, PnLAttribution, GreeksHeatMap
from flow_analysis import FlowAnalytics, UnusualActivity, SentimentIndicator

@dataclass
class ChartConfig:
 chart_type: str
 title: str
 width: int = 800
 height: int = 600
 theme: str = 'plotly_dark'
 show_legend: bool = True
 interactive: bool = True
 auto_refresh: bool = False
 refresh_interval: int = 5

@dataclass
class VisualizationData:
 chart_id: str
 chart_config: ChartConfig
 data: Dict[str, Any]
 figure: Optional[go.Figure] = None
 last_updated: Optional[datetime] = None

class VisualizationEngine:
 """Core visualization engine for options analytics"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 # Chart generators
 self.portfolio_charts = PortfolioChartGenerator(config)
 self.market_charts = MarketChartGenerator(config)
 self.flow_charts = FlowChartGenerator(config)
 self.greeks_charts = GreeksChartGenerator(config)
 self.volatility_charts = VolatilityChartGenerator(config)

 # Chart cache
 self.chart_cache: Dict[str, VisualizationData] = {}

 # Default theme settings
 self.default_theme = config.get('chart_theme', 'plotly_dark')
 self.color_palette = config.get('color_palette', {
 'primary': '#00D4FF',
 'secondary': '#FF6B35',
 'success': '#4CAF50',
 'warning': '#FF9800',
 'danger': '#F44336',
 'info': '#2196F3'
 })

 async def initialize(self):
 """Initialize visualization engine"""
 self.logger.info("Initializing Visualization Engine")
 await asyncio.gather(
 self.portfolio_charts.initialize(),
 self.market_charts.initialize(),
 self.flow_charts.initialize(),
 self.greeks_charts.initialize(),
 self.volatility_charts.initialize()
 )

 async def generate_portfolio_dashboard(self, portfolio_data: Dict[str, Any]) -> Dict[str, go.Figure]:
 """Generate comprehensive portfolio dashboard"""
 dashboard_charts = {}

 # Portfolio overview chart
 dashboard_charts['portfolio_overview'] = await self.portfolio_charts.create_portfolio_overview(
 portfolio_data.get('positions', {}),
 portfolio_data.get('summary', {})
 )

 # P&L attribution chart
 dashboard_charts['pnl_attribution'] = await self.portfolio_charts.create_pnl_attribution_chart(
 portfolio_data.get('pnl_attribution', [])
 )

 # Greeks evolution chart
 dashboard_charts['greeks_evolution'] = await self.greeks_charts.create_greeks_timeline(
 portfolio_data.get('greeks_history', [])
 )

 # Risk metrics chart
 dashboard_charts['risk_metrics'] = await self.portfolio_charts.create_risk_dashboard(
 portfolio_data.get('risk_metrics', {})
 )

 return dashboard_charts

 async def generate_market_analysis_dashboard(self, market_data: Dict[str, Any]) -> Dict[str, go.Figure]:
 """Generate market analysis dashboard"""
 dashboard_charts = {}

 # Volatility surface chart
 dashboard_charts['vol_surface'] = await self.volatility_charts.create_volatility_surface_3d(
 market_data.get('vol_surfaces', {})
 )

 # Options flow analysis
 dashboard_charts['flow_analysis'] = await self.flow_charts.create_flow_heatmap(
 market_data.get('flow_analytics', {})
 )

 # Sentiment indicators
 dashboard_charts['sentiment'] = await self.flow_charts.create_sentiment_dashboard(
 market_data.get('sentiment_indicators', {})
 )

 # Put/call analysis
 dashboard_charts['put_call'] = await self.flow_charts.create_put_call_analysis(
 market_data.get('put_call_data', {})
 )

 return dashboard_charts

 async def generate_trading_signals_dashboard(self, signals_data: Dict[str, Any]) -> Dict[str, go.Figure]:
 """Generate trading signals dashboard"""
 dashboard_charts = {}

 # Signal strength chart
 dashboard_charts['signal_strength'] = await self.market_charts.create_signal_strength_chart(
 signals_data.get('active_signals', [])
 )

 # Signal performance
 dashboard_charts['signal_performance'] = await self.market_charts.create_signal_performance_chart(
 signals_data.get('signal_performance', {})
 )

 # Arbitrage opportunities
 dashboard_charts['arbitrage'] = await self.market_charts.create_arbitrage_chart(
 signals_data.get('arbitrage_opportunities', [])
 )

 return dashboard_charts

 async def create_custom_chart(self, chart_config: ChartConfig, data: Dict[str, Any]) -> go.Figure:
 """Create custom chart based on configuration"""
 if chart_config.chart_type == 'portfolio_overview':
 return await self.portfolio_charts.create_portfolio_overview(
 data.get('positions', {}), data.get('summary', {})
 )
 elif chart_config.chart_type == 'volatility_surface':
 return await self.volatility_charts.create_volatility_surface_3d(
 data.get('vol_surfaces', {})
 )
 elif chart_config.chart_type == 'options_chain':
 return await self.market_charts.create_options_chain_viz(
 data.get('options_chain', {})
 )
 elif chart_config.chart_type == 'greeks_heatmap':
 return await self.greeks_charts.create_greeks_heatmap(
 data.get('greeks_data', {})
 )
 else:
 raise ValueError(f"Unknown chart type: {chart_config.chart_type}")

 def get_chart_html(self, figure: go.Figure, include_plotlyjs: str = 'cdn') -> str:
 """Convert figure to HTML string"""
 return figure.to_html(include_plotlyjs=include_plotlyjs, div_id="chart")

 def get_chart_json(self, figure: go.Figure) -> str:
 """Convert figure to JSON string"""
 return figure.to_json()

class PortfolioChartGenerator:
 """Portfolio-specific chart generation"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize portfolio chart generator"""
 self.logger.info("Initializing Portfolio Chart Generator")

 async def create_portfolio_overview(self, positions: Dict[str, Position],
 summary: Dict[str, Any]) -> go.Figure:
 """Create portfolio overview chart"""
 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('Position Distribution', 'P&L by Symbol', 'Greeks Distribution', 'Risk Metrics'),
 specs=[[{"type": "pie"}, {"type": "bar"}],
 [{"type": "bar"}, {"type": "indicator"}]]
 )

 if positions:
 # Position distribution pie chart
 symbols = list(positions.keys())
 values = [abs(pos.market_value) for pos in positions.values()]
 colors = px.colors.qualitative.Set3[:len(symbols)]

 fig.add_trace(
 go.Pie(labels=symbols, values=values, name="Positions", marker=dict(colors=colors)),
 row=1, col=1
 )

 # P&L by symbol bar chart
 pnl_values = [pos.unrealized_pnl + pos.realized_pnl for pos in positions.values()]
 colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]

 fig.add_trace(
 go.Bar(x=symbols, y=pnl_values, name="P&L", marker=dict(color=colors)),
 row=1, col=2
 )

 # Greeks distribution
 greeks_names = ['Delta', 'Gamma', 'Theta', 'Vega']
 greeks_values = [
 sum(pos.delta * pos.quantity for pos in positions.values()),
 sum(pos.gamma * pos.quantity for pos in positions.values()),
 sum(pos.theta * pos.quantity for pos in positions.values()),
 sum(pos.vega * pos.quantity for pos in positions.values())
 ]

 fig.add_trace(
 go.Bar(x=greeks_names, y=greeks_values, name="Greeks",
 marker=dict(color=['#FF6B35', '#00D4FF', '#4CAF50', '#FF9800'])),
 row=2, col=1
 )

 # Risk metrics indicator
 total_value = summary.get('portfolio_value', 0)
 var_95 = summary.get('risk_metrics', {}).get('var_metrics', {}).get('var', 0)

 fig.add_trace(
 go.Indicator(
 mode="gauge+number+delta",
 value=var_95,
 title={'text': "VaR 95%"},
 gauge={'axis': {'range': [None, total_value * 0.1]},
 'bar': {'color': "red"},
 'steps': [{'range': [0, total_value * 0.03], 'color': "lightgray"},
 {'range': [total_value * 0.03, total_value * 0.05], 'color': "gray"}],
 'threshold': {'line': {'color': "red", 'width': 4},
 'thickness': 0.75, 'value': total_value * 0.05}}
 ),
 row=2, col=2
 )

 fig.update_layout(
 title_text="Portfolio Overview Dashboard",
 showlegend=True,
 template="plotly_dark",
 height=800
 )

 return fig

 async def create_pnl_attribution_chart(self, pnl_attribution: List[PnLAttribution]) -> go.Figure:
 """Create P&L attribution chart"""
 if not pnl_attribution:
 return go.Figure().add_annotation(text="No P&L data available", showarrow=False)

 df = pd.DataFrame([
 {
 'timestamp': attr.timestamp,
 'delta_pnl': attr.delta_pnl,
 'gamma_pnl': attr.gamma_pnl,
 'theta_pnl': attr.theta_pnl,
 'vega_pnl': attr.vega_pnl,
 'total_pnl': attr.total_pnl
 }
 for attr in pnl_attribution
 ])

 fig = make_subplots(
 rows=2, cols=1,
 subplot_titles=('P&L Attribution by Greeks', 'Cumulative P&L'),
 shared_xaxes=True
 )

 # Stacked bar chart for Greeks attribution
 greeks = ['delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl']
 colors = ['#FF6B35', '#00D4FF', '#4CAF50', '#FF9800']

 for greek, color in zip(greeks, colors):
 fig.add_trace(
 go.Bar(x=df['timestamp'], y=df[greek], name=greek.replace('_pnl', '').title(),
 marker=dict(color=color)),
 row=1, col=1
 )

 # Cumulative P&L line chart
 df['cumulative_pnl'] = df['total_pnl'].cumsum()
 fig.add_trace(
 go.Scatter(x=df['timestamp'], y=df['cumulative_pnl'], mode='lines+markers',
 name='Cumulative P&L', line=dict(color='white', width=2)),
 row=2, col=1
 )

 fig.update_layout(
 title_text="P&L Attribution Analysis",
 template="plotly_dark",
 height=600,
 barmode='relative'
 )

 return fig

 async def create_risk_dashboard(self, risk_metrics: Dict[str, Any]) -> go.Figure:
 """Create risk metrics dashboard"""
 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('VaR Distribution', 'Stress Test Results', 'Concentration Risk', 'Leverage Metrics'),
 specs=[[{"type": "histogram"}, {"type": "bar"}],
 [{"type": "pie"}, {"type": "indicator"}]]
 )

 # VaR distribution (mock data for visualization)
 if 'var_metrics' in risk_metrics:
 var_data = np.random.normal(0, risk_metrics['var_metrics'].get('var', 1000), 1000)
 fig.add_trace(
 go.Histogram(x=var_data, name="VaR Distribution", nbinsx=50,
 marker=dict(color='red', opacity=0.7)),
 row=1, col=1
 )

 # Stress test results
 if 'stress_tests' in risk_metrics:
 stress_tests = risk_metrics['stress_tests']
 scenarios = list(stress_tests.keys())
 pnl_values = [stress_tests[scenario]['total_pnl'] for scenario in scenarios]

 colors = ['red' if pnl < 0 else 'green' for pnl in pnl_values]
 fig.add_trace(
 go.Bar(x=scenarios, y=pnl_values, name="Stress Test P&L",
 marker=dict(color=colors)),
 row=1, col=2
 )

 # Concentration risk pie chart
 if 'concentration_risk' in risk_metrics:
 conc_risk = risk_metrics['concentration_risk']
 fig.add_trace(
 go.Pie(labels=['Max Position', 'Other Positions'],
 values=[conc_risk.get('max_position_concentration', 0.3),
 1 - conc_risk.get('max_position_concentration', 0.3)],
 name="Concentration"),
 row=2, col=1
 )

 # Leverage indicator
 leverage_ratio = risk_metrics.get('leverage_ratio', 1.0)
 fig.add_trace(
 go.Indicator(
 mode="gauge+number",
 value=leverage_ratio,
 title={'text': "Leverage Ratio"},
 gauge={'axis': {'range': [None, 5]},
 'bar': {'color': "orange"},
 'steps': [{'range': [0, 2], 'color': "lightgray"},
 {'range': [2, 3], 'color': "gray"}],
 'threshold': {'line': {'color': "red", 'width': 4},
 'thickness': 0.75, 'value': 3}}
 ),
 row=2, col=2
 )

 fig.update_layout(
 title_text="Risk Metrics Dashboard",
 template="plotly_dark",
 height=800
 )

 return fig

class GreeksChartGenerator:
 """Greeks-specific chart generation"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize Greeks chart generator"""
 self.logger.info("Initializing Greeks Chart Generator")

 async def create_greeks_timeline(self, greeks_history: List[PortfolioGreeks]) -> go.Figure:
 """Create Greeks evolution timeline"""
 if not greeks_history:
 return go.Figure().add_annotation(text="No Greeks history available", showarrow=False)

 df = pd.DataFrame([
 {
 'timestamp': greeks.timestamp,
 'delta': greeks.total_delta,
 'gamma': greeks.total_gamma,
 'theta': greeks.total_theta,
 'vega': greeks.total_vega
 }
 for greeks in greeks_history
 ])

 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega'),
 shared_xaxes=True
 )

 # Delta timeline
 fig.add_trace(
 go.Scatter(x=df['timestamp'], y=df['delta'], mode='lines+markers',
 name='Delta', line=dict(color='#FF6B35', width=2)),
 row=1, col=1
 )

 # Gamma timeline
 fig.add_trace(
 go.Scatter(x=df['timestamp'], y=df['gamma'], mode='lines+markers',
 name='Gamma', line=dict(color='#00D4FF', width=2)),
 row=1, col=2
 )

 # Theta timeline
 fig.add_trace(
 go.Scatter(x=df['timestamp'], y=df['theta'], mode='lines+markers',
 name='Theta', line=dict(color='#4CAF50', width=2)),
 row=2, col=1
 )

 # Vega timeline
 fig.add_trace(
 go.Scatter(x=df['timestamp'], y=df['vega'], mode='lines+markers',
 name='Vega', line=dict(color='#FF9800', width=2)),
 row=2, col=2
 )

 fig.update_layout(
 title_text="Portfolio Greeks Evolution",
 template="plotly_dark",
 height=600,
 showlegend=False
 )

 return fig

 async def create_greeks_heatmap(self, greeks_data: Dict[str, Any]) -> go.Figure:
 """Create Greeks heatmap visualization"""
 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('Delta Heatmap', 'Gamma Heatmap', 'Theta Heatmap', 'Vega Heatmap'),
 specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
 [{"type": "heatmap"}, {"type": "heatmap"}]]
 )

 # Generate sample heatmap data
 symbols = greeks_data.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
 strikes = greeks_data.get('strikes', [90, 95, 100, 105, 110])

 # Create sample matrices
 n_symbols, n_strikes = len(symbols), len(strikes)
 delta_matrix = np.random.normal(0.5, 0.2, (n_symbols, n_strikes))
 gamma_matrix = np.random.normal(0.1, 0.05, (n_symbols, n_strikes))
 theta_matrix = np.random.normal(-0.02, 0.01, (n_symbols, n_strikes))
 vega_matrix = np.random.normal(0.15, 0.05, (n_symbols, n_strikes))

 # Delta heatmap
 fig.add_trace(
 go.Heatmap(z=delta_matrix, x=strikes, y=symbols, colorscale='RdYlBu_r',
 name='Delta', showscale=True),
 row=1, col=1
 )

 # Gamma heatmap
 fig.add_trace(
 go.Heatmap(z=gamma_matrix, x=strikes, y=symbols, colorscale='Viridis',
 name='Gamma', showscale=True),
 row=1, col=2
 )

 # Theta heatmap
 fig.add_trace(
 go.Heatmap(z=theta_matrix, x=strikes, y=symbols, colorscale='Reds',
 name='Theta', showscale=True),
 row=2, col=1
 )

 # Vega heatmap
 fig.add_trace(
 go.Heatmap(z=vega_matrix, x=strikes, y=symbols, colorscale='Blues',
 name='Vega', showscale=True),
 row=2, col=2
 )

 fig.update_layout(
 title_text="Greeks Heatmap Analysis",
 template="plotly_dark",
 height=800
 )

 return fig

class VolatilityChartGenerator:
 """Volatility-specific chart generation"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize volatility chart generator"""
 self.logger.info("Initializing Volatility Chart Generator")

 async def create_volatility_surface_3d(self, vol_surfaces: Dict[str, VolatilitySurface]) -> go.Figure:
 """Create 3D volatility surface"""
 if not vol_surfaces:
 return go.Figure().add_annotation(text="No volatility surface data available", showarrow=False)

 # Use first available surface
 symbol, surface = next(iter(vol_surfaces.items()))

 # Create meshgrid for 3D surface
 strikes = np.array(surface.strikes)
 expiries_numeric = np.array([(exp - datetime.now()).days for exp in surface.expiries])

 if surface.implied_vols.ndim == 1:
 # Create a 2D surface from 1D data
 X, Y = np.meshgrid(strikes, expiries_numeric)
 Z = np.tile(surface.implied_vols, (len(expiries_numeric), 1))
 else:
 X, Y = np.meshgrid(strikes, expiries_numeric)
 Z = surface.implied_vols

 fig = go.Figure(data=[
 go.Surface(
 x=X, y=Y, z=Z,
 colorscale='Viridis',
 name=f'{symbol} Vol Surface'
 )
 ])

 fig.update_layout(
 title=f'{symbol} Implied Volatility Surface',
 scene=dict(
 xaxis_title='Strike',
 yaxis_title='Days to Expiry',
 zaxis_title='Implied Volatility',
 camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
 ),
 template="plotly_dark",
 height=600
 )

 return fig

 async def create_volatility_smile(self, vol_surface: VolatilitySurface) -> go.Figure:
 """Create volatility smile chart"""
 fig = go.Figure()

 if vol_surface.implied_vols.ndim > 1:
 for i, expiry in enumerate(vol_surface.expiries):
 expiry_vols = vol_surface.implied_vols[i, :]
 days_to_exp = (expiry - datetime.now()).days

 fig.add_trace(
 go.Scatter(
 x=vol_surface.strikes,
 y=expiry_vols,
 mode='lines+markers',
 name=f'{days_to_exp}D',
 line=dict(width=2)
 )
 )
 else:
 fig.add_trace(
 go.Scatter(
 x=vol_surface.strikes,
 y=vol_surface.implied_vols,
 mode='lines+markers',
 name='Implied Vol',
 line=dict(color='#00D4FF', width=3)
 )
 )

 fig.update_layout(
 title=f'{vol_surface.underlying} Volatility Smile',
 xaxis_title='Strike Price',
 yaxis_title='Implied Volatility',
 template="plotly_dark",
 height=500
 )

 return fig

class FlowChartGenerator:
 """Options flow chart generation"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize flow chart generator"""
 self.logger.info("Initializing Flow Chart Generator")

 async def create_flow_heatmap(self, flow_analytics: Dict[str, FlowAnalytics]) -> go.Figure:
 """Create options flow heatmap"""
 if not flow_analytics:
 return go.Figure().add_annotation(text="No flow data available", showarrow=False)

 symbols = list(flow_analytics.keys())
 metrics = ['total_volume', 'put_call_ratio', 'sentiment_score', 'unusual_activity_score']

 # Create data matrix
 data_matrix = []
 for metric in metrics:
 metric_values = []
 for symbol in symbols:
 analytics = flow_analytics[symbol]
 if metric == 'total_volume':
 value = analytics.total_volume / 10000 # Normalize
 elif metric == 'put_call_ratio':
 value = min(analytics.put_call_ratio, 3.0) # Cap at 3
 elif metric == 'sentiment_score':
 value = (analytics.sentiment_score + 1) / 2 # Normalize to 0-1
 else:
 value = analytics.unusual_activity_score
 metric_values.append(value)
 data_matrix.append(metric_values)

 fig = go.Figure(data=go.Heatmap(
 z=data_matrix,
 x=symbols,
 y=metrics,
 colorscale='RdYlBu_r',
 text=[[f"{val:.2f}" for val in row] for row in data_matrix],
 texttemplate="%{text}",
 textfont={"size": 10},
 ))

 fig.update_layout(
 title="Options Flow Analysis Heatmap",
 template="plotly_dark",
 height=400
 )

 return fig

 async def create_sentiment_dashboard(self, sentiment_indicators: Dict[str, SentimentIndicator]) -> go.Figure:
 """Create sentiment indicators dashboard"""
 if not sentiment_indicators:
 return go.Figure().add_annotation(text="No sentiment data available", showarrow=False)

 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=list(sentiment_indicators.keys())[:4],
 specs=[[{"type": "indicator"}, {"type": "indicator"}],
 [{"type": "indicator"}, {"type": "indicator"}]]
 )

 indicators = list(sentiment_indicators.values())[:4]
 positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

 for i, (indicator, pos) in enumerate(zip(indicators, positions)):
 # Determine color based on value
 if indicator.normalized_value > 0.7:
 color = "green"
 elif indicator.normalized_value < 0.3:
 color = "red"
 else:
 color = "yellow"

 fig.add_trace(
 go.Indicator(
 mode="gauge+number",
 value=indicator.normalized_value * 100,
 title={'text': indicator.indicator_name.replace('_', ' ').title()},
 gauge={
 'axis': {'range': [None, 100]},
 'bar': {'color': color},
 'steps': [
 {'range': [0, 30], 'color': "lightgray"},
 {'range': [30, 70], 'color': "gray"}
 ],
 'threshold': {
 'line': {'color': "red", 'width': 4},
 'thickness': 0.75,
 'value': 90
 }
 }
 ),
 row=pos[0], col=pos[1]
 )

 fig.update_layout(
 title_text="Market Sentiment Dashboard",
 template="plotly_dark",
 height=600
 )

 return fig

 async def create_put_call_analysis(self, put_call_data: Dict[str, Any]) -> go.Figure:
 """Create put/call ratio analysis chart"""
 if not put_call_data:
 return go.Figure().add_annotation(text="No put/call data available", showarrow=False)

 fig = make_subplots(
 rows=2, cols=1,
 subplot_titles=('Put/Call Ratio Trend', 'Volume Distribution'),
 shared_xaxes=True
 )

 # Generate sample time series data
 dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
 pc_ratios = np.random.normal(put_call_data.get('current_ratio', 1.0), 0.2, len(dates))

 fig.add_trace(
 go.Scatter(
 x=dates, y=pc_ratios,
 mode='lines+markers',
 name='P/C Ratio',
 line=dict(color='#FF6B35', width=2)
 ),
 row=1, col=1
 )

 # Add horizontal reference lines
 fig.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="Neutral", row=1, col=1)
 fig.add_hline(y=1.2, line_dash="dash", line_color="red", annotation_text="Bearish", row=1, col=1)
 fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Bullish", row=1, col=1)

 # Volume distribution
 call_volume = put_call_data.get('call_volume', 50000)
 put_volume = put_call_data.get('put_volume', 60000)

 fig.add_trace(
 go.Bar(
 x=['Call Volume', 'Put Volume'],
 y=[call_volume, put_volume],
 name='Volume',
 marker=dict(color=['green', 'red'])
 ),
 row=2, col=1
 )

 fig.update_layout(
 title_text="Put/Call Ratio Analysis",
 template="plotly_dark",
 height=600
 )

 return fig

class MarketChartGenerator:
 """Market analysis chart generation"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 async def initialize(self):
 """Initialize market chart generator"""
 self.logger.info("Initializing Market Chart Generator")

 async def create_options_chain_viz(self, options_chain: Dict[str, Any]) -> go.Figure:
 """Create options chain visualization"""
 if not options_chain:
 return go.Figure().add_annotation(text="No options chain data available", showarrow=False)

 # Sample options chain data
 strikes = list(range(90, 111, 5))
 call_volumes = np.random.randint(100, 10000, len(strikes))
 put_volumes = np.random.randint(100, 10000, len(strikes))
 call_oi = np.random.randint(1000, 50000, len(strikes))
 put_oi = np.random.randint(1000, 50000, len(strikes))

 fig = make_subplots(
 rows=2, cols=1,
 subplot_titles=('Volume', 'Open Interest'),
 shared_xaxes=True
 )

 # Volume bars
 fig.add_trace(
 go.Bar(x=strikes, y=call_volumes, name='Call Volume',
 marker=dict(color='green'), offsetgroup=1),
 row=1, col=1
 )
 fig.add_trace(
 go.Bar(x=strikes, y=[-v for v in put_volumes], name='Put Volume',
 marker=dict(color='red'), offsetgroup=1),
 row=1, col=1
 )

 # Open Interest bars
 fig.add_trace(
 go.Bar(x=strikes, y=call_oi, name='Call OI',
 marker=dict(color='lightgreen'), offsetgroup=2),
 row=2, col=1
 )
 fig.add_trace(
 go.Bar(x=strikes, y=[-oi for oi in put_oi], name='Put OI',
 marker=dict(color='lightcoral'), offsetgroup=2),
 row=2, col=1
 )

 fig.update_layout(
 title_text="Options Chain Visualization",
 xaxis_title="Strike Price",
 template="plotly_dark",
 height=600,
 barmode='overlay'
 )

 return fig

 async def create_signal_strength_chart(self, active_signals: List[Dict[str, Any]]) -> go.Figure:
 """Create signal strength chart"""
 if not active_signals:
 return go.Figure().add_annotation(text="No active signals", showarrow=False)

 df = pd.DataFrame(active_signals)

 fig = go.Figure()

 # Bubble chart with signal strength
 fig.add_trace(
 go.Scatter(
 x=df.get('confidence', [0.5] * len(active_signals)),
 y=df.get('strength', [0.5] * len(active_signals)),
 mode='markers+text',
 marker=dict(
 size=[s * 50 for s in df.get('strength', [0.5] * len(active_signals))],
 color=df.get('strength', [0.5] * len(active_signals)),
 colorscale='Viridis',
 showscale=True,
 sizemode='diameter'
 ),
 text=df.get('symbol', ['Signal'] * len(active_signals)),
 textposition="middle center",
 name='Signals'
 )
 )

 fig.update_layout(
 title="Trading Signals Strength Analysis",
 xaxis_title="Confidence",
 yaxis_title="Strength",
 template="plotly_dark",
 height=500
 )

 return fig

 async def create_signal_performance_chart(self, signal_performance: Dict[str, Any]) -> go.Figure:
 """Create signal performance chart"""
 if not signal_performance:
 return go.Figure().add_annotation(text="No signal performance data", showarrow=False)

 signal_types = list(signal_performance.keys())
 win_rates = [signal_performance[st].get('win_rate', 0.5) for st in signal_types]
 avg_returns = [signal_performance[st].get('avg_return', 0.0) for st in signal_types]

 fig = go.Figure()

 fig.add_trace(
 go.Bar(
 x=signal_types,
 y=win_rates,
 name='Win Rate',
 marker=dict(color='green'),
 yaxis='y1'
 )
 )

 fig.add_trace(
 go.Scatter(
 x=signal_types,
 y=avg_returns,
 mode='lines+markers',
 name='Avg Return',
 line=dict(color='orange', width=3),
 yaxis='y2'
 )
 )

 fig.update_layout(
 title="Signal Performance Analysis",
 xaxis_title="Signal Type",
 yaxis=dict(title="Win Rate", side="left"),
 yaxis2=dict(title="Average Return", side="right", overlaying="y"),
 template="plotly_dark",
 height=500
 )

 return fig

 async def create_arbitrage_chart(self, arbitrage_opportunities: List[Dict[str, Any]]) -> go.Figure:
 """Create arbitrage opportunities chart"""
 if not arbitrage_opportunities:
 return go.Figure().add_annotation(text="No arbitrage opportunities", showarrow=False)

 # Create sample data structure
 opportunities = []
 for i, opp in enumerate(arbitrage_opportunities[:10]): # Limit to 10 for visualization
 opportunities.append({
 'id': f"Opportunity {i+1}",
 'profit': opp.get('expected_profit', np.random.uniform(100, 1000)),
 'risk': opp.get('max_risk', np.random.uniform(500, 2000)),
 'probability': opp.get('profit_probability', np.random.uniform(0.5, 0.9))
 })

 df = pd.DataFrame(opportunities)

 fig = go.Figure()

 fig.add_trace(
 go.Scatter(
 x=df['risk'],
 y=df['profit'],
 mode='markers+text',
 marker=dict(
 size=[p * 50 for p in df['probability']],
 color=df['probability'],
 colorscale='RdYlGn',
 showscale=True,
 colorbar=dict(title="Probability"),
 sizemode='diameter'
 ),
 text=df['id'],
 textposition="top center",
 name='Arbitrage Opportunities'
 )
 )

 fig.update_layout(
 title="Arbitrage Opportunities Analysis",
 xaxis_title="Max Risk ($)",
 yaxis_title="Expected Profit ($)",
 template="plotly_dark",
 height=500
 )

 return fig