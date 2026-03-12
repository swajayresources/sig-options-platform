"""
Professional Options Trading Platform - Streamlit Interface
Institutional-grade options trading dashboard with real-time analytics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time
import json
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from utils.data_provider import OptionsDataProvider
from utils.pricing_engine import OptionsPricingEngine
from utils.portfolio_manager import PortfolioManager
from utils.market_analysis import MarketAnalyzer
from utils.visualization import ChartGenerator

# Page configuration
st.set_page_config(
 page_title="Options Trading Platform",
 page_icon="📈",
 layout="wide",
 initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>.main-header {
 font-size: 2.5rem;
 font-weight: bold;
 color: #1f77b4;
 text-align: center;
 margin-bottom: 1rem;
 }.metric-container {
 background-color: #f0f2f6;
 padding: 1rem;
 border-radius: 0.5rem;
 border-left: 4px solid #1f77b4;
 }.profit {
 color: #00ff00 !important;
 font-weight: bold;
 }.loss {
 color: #ff0000 !important;
 font-weight: bold;
 }.neutral {
 color: #ffa500 !important;
 font-weight: bold;
 }.alert-high {
 background-color: #ffebee;
 border-left: 4px solid #f44336;
 padding: 1rem;
 margin: 1rem 0;
 }.alert-medium {
 background-color: #fff3e0;
 border-left: 4px solid #ff9800;
 padding: 1rem;
 margin: 1rem 0;
 }.stDataFrame {
 border: 1px solid #e0e0e0;
 border-radius: 0.25rem;
 }.sidebar.sidebar-content {
 background-color: #f8f9fa;
 }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
 """Initialize session state variables"""
 if 'portfolio_data' not in st.session_state:
 st.session_state.portfolio_data = {}

 if 'market_data' not in st.session_state:
 st.session_state.market_data = {}

 if 'last_update' not in st.session_state:
 st.session_state.last_update = datetime.now()

 if 'alert_count' not in st.session_state:
 st.session_state.alert_count = 0

 if 'auto_refresh' not in st.session_state:
 st.session_state.auto_refresh = True

def main():
 """Main application function"""
 initialize_session_state()

 # Initialize data providers and engines
 data_provider = OptionsDataProvider()
 pricing_engine = OptionsPricingEngine()
 portfolio_manager = PortfolioManager()
 market_analyzer = MarketAnalyzer()
 chart_generator = ChartGenerator()

 # Header
 st.markdown('<h1 class="main-header">📈 Professional Options Trading Platform</h1>',
 unsafe_allow_html=True)

 # Sidebar navigation
 with st.sidebar:
 st.title("🎯 Trading Dashboard")

 # Auto-refresh toggle
 st.session_state.auto_refresh = st.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)

 if st.session_state.auto_refresh:
 refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)

 st.divider()

 # Navigation
 page = st.selectbox(
 "Select Trading View",
 [
 "📊 Portfolio Dashboard",
 "🔗 Options Chain",
 "🌊 Volatility Surface",
 "📈 Strategy Performance",
 "🔍 Market Analysis",
 "⚡ Flow & Alerts",
 "📊 Risk Management"
 ]
 )

 st.divider()

 # Market status
 st.subheader("📡 Market Status")
 market_status = data_provider.get_market_status()
 status_color = "🟢" if market_status == "OPEN" else "🔴"
 st.write(f"{status_color} Market: **{market_status}**")
 st.write(f"🕐 Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

 # Quick portfolio stats
 st.subheader("⚡ Quick Stats")
 portfolio_summary = portfolio_manager.get_portfolio_summary()

 total_pnl = portfolio_summary.get('total_pnl', 0)
 pnl_color = "profit" if total_pnl >= 0 else "loss"
 st.markdown(f'<p class="{pnl_color}">Total P&L: ${total_pnl:,.2f}</p>',
 unsafe_allow_html=True)

 st.write(f"📊 Positions: {portfolio_summary.get('position_count', 0)}")
 st.write(f"🎯 Greeks Delta: {portfolio_summary.get('total_delta', 0):.2f}")

 # Main content area
 if page == "📊 Portfolio Dashboard":
 render_portfolio_dashboard(portfolio_manager, chart_generator)
 elif page == "🔗 Options Chain":
 render_options_chain(data_provider, pricing_engine)
 elif page == "🌊 Volatility Surface":
 render_volatility_surface(data_provider, chart_generator)
 elif page == "📈 Strategy Performance":
 render_strategy_performance(portfolio_manager, chart_generator)
 elif page == "🔍 Market Analysis":
 render_market_analysis(market_analyzer, chart_generator)
 elif page == "⚡ Flow & Alerts":
 render_flow_alerts(market_analyzer, data_provider)
 elif page == "📊 Risk Management":
 render_risk_management(portfolio_manager, chart_generator)

 # Auto-refresh mechanism
 if st.session_state.auto_refresh and market_status == "OPEN":
 time.sleep(refresh_interval)
 st.rerun()

def render_portfolio_dashboard(portfolio_manager, chart_generator):
 """Render the main portfolio dashboard"""
 st.header("📊 Portfolio Dashboard")

 # Real-time portfolio metrics
 col1, col2, col3, col4 = st.columns(4)

 portfolio_greeks = portfolio_manager.get_portfolio_greeks()

 with col1:
 delta_change = np.random.uniform(-50, 50)
 st.metric(
 "Portfolio Delta",
 f"{portfolio_greeks['delta']:,.0f}",
 f"{delta_change:+.0f}",
 delta_color="normal"
 )

 with col2:
 gamma_change = np.random.uniform(-25, 25)
 st.metric(
 "Portfolio Gamma",
 f"{portfolio_greeks['gamma']:,.0f}",
 f"{gamma_change:+.0f}",
 delta_color="normal"
 )

 with col3:
 theta_value = portfolio_greeks['theta']
 theta_change = np.random.uniform(-150, 50)
 st.metric(
 "Portfolio Theta",
 f"${theta_value:,.0f}",
 f"${theta_change:+.0f}",
 delta_color="inverse"
 )

 with col4:
 vega_change = np.random.uniform(-500, 500)
 st.metric(
 "Portfolio Vega",
 f"${portfolio_greeks['vega']:,.0f}",
 f"${vega_change:+.0f}",
 delta_color="normal"
 )

 st.divider()

 # Portfolio composition and P&L charts
 col1, col2 = st.columns(2)

 with col1:
 st.subheader("📈 Portfolio Composition")
 portfolio_composition = portfolio_manager.get_portfolio_composition()

 fig_pie = px.pie(
 values=list(portfolio_composition.values()),
 names=list(portfolio_composition.keys()),
 title="Position Distribution by Symbol",
 color_discrete_sequence=px.colors.qualitative.Set3
 )
 fig_pie.update_layout(height=400)
 st.plotly_chart(fig_pie, use_container_width=True)

 with col2:
 st.subheader("💰 P&L Attribution")
 pnl_data = portfolio_manager.get_pnl_attribution()

 fig_bar = px.bar(
 x=list(pnl_data.keys()),
 y=list(pnl_data.values()),
 title="P&L by Greeks",
 color=list(pnl_data.values()),
 color_continuous_scale="RdYlGn"
 )
 fig_bar.update_layout(height=400)
 st.plotly_chart(fig_bar, use_container_width=True)

 # Portfolio positions table
 st.subheader("📋 Current Positions")
 positions_df = portfolio_manager.get_positions_dataframe()

 # Color-code the P&L column
 def highlight_pnl(val):
 color = 'lightgreen' if val > 0 else 'lightcoral' if val < 0 else 'lightyellow'
 return f'background-color: {color}'

 styled_df = positions_df.style.applymap(highlight_pnl, subset=['Unrealized_PnL'])
 st.dataframe(styled_df, use_container_width=True, height=300)

 # Greeks evolution chart
 st.subheader("📊 Greeks Evolution")
 greeks_history = portfolio_manager.get_greeks_history()

 fig_greeks = go.Figure()

 for greek in ['delta', 'gamma', 'theta', 'vega']:
 fig_greeks.add_trace(
 go.Scatter(
 x=greeks_history['timestamp'],
 y=greeks_history[greek],
 mode='lines+markers',
 name=greek.title(),
 line=dict(width=2)
 )
 )

 fig_greeks.update_layout(
 title="Portfolio Greeks Evolution Over Time",
 xaxis_title="Time",
 yaxis_title="Greeks Value",
 height=400,
 hovermode='x unified'
 )

 st.plotly_chart(fig_greeks, use_container_width=True)

def render_options_chain(data_provider, pricing_engine):
 """Render the options chain view"""
 st.header("🔗 Live Options Chain")

 # Symbol selection and expiration
 col1, col2, col3 = st.columns(3)

 with col1:
 symbol = st.selectbox("Select Symbol", ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"])

 with col2:
 expirations = data_provider.get_expiration_dates(symbol)
 expiration = st.selectbox("Expiration Date", expirations)

 with col3:
 chain_type = st.selectbox("Chain Type", ["Both", "Calls Only", "Puts Only"])

 # Get options chain data
 options_chain = data_provider.get_options_chain(symbol, expiration, chain_type)

 if not options_chain.empty:
 # Current stock price
 current_price = data_provider.get_current_price(symbol)
 st.metric(f"{symbol} Current Price", f"${current_price:.2f}")

 # Options chain display
 st.subheader("📊 Options Chain Data")

 # Color-code ITM/OTM options
 def color_options(row):
 if row['Type'] == 'CALL':
 color = 'lightgreen' if row['Strike'] < current_price else 'lightgray'
 else: # PUT
 color = 'lightcoral' if row['Strike'] > current_price else 'lightgray'
 return [f'background-color: {color}'] * len(row)

 styled_chain = options_chain.style.apply(color_options, axis=1)
 st.dataframe(styled_chain, use_container_width=True, height=500)

 # Greeks heatmap
 st.subheader("🔥 Greeks Heatmap")

 # Create heatmap data
 strikes = sorted(options_chain['Strike'].unique())
 greeks_data = {}

 for greek in ['Delta', 'Gamma', 'Theta', 'Vega']:
 calls_data = []
 puts_data = []

 for strike in strikes:
 call_row = options_chain[(options_chain['Strike'] == strike) &
 (options_chain['Type'] == 'CALL')]
 put_row = options_chain[(options_chain['Strike'] == strike) &
 (options_chain['Type'] == 'PUT')]

 calls_data.append(call_row[greek].iloc[0] if not call_row.empty else 0)
 puts_data.append(put_row[greek].iloc[0] if not put_row.empty else 0)

 greeks_data[f'{greek}_CALL'] = calls_data
 greeks_data[f'{greek}_PUT'] = puts_data

 # Display Greeks heatmaps
 col1, col2 = st.columns(2)

 with col1:
 fig_delta = px.imshow(
 [greeks_data['Delta_CALL'], greeks_data['Delta_PUT']],
 labels=dict(x="Strike", y="Type", color="Delta"),
 x=strikes,
 y=['CALL', 'PUT'],
 color_continuous_scale='RdYlBu_r',
 title="Delta Heatmap"
 )
 st.plotly_chart(fig_delta, use_container_width=True)

 with col2:
 fig_gamma = px.imshow(
 [greeks_data['Gamma_CALL'], greeks_data['Gamma_PUT']],
 labels=dict(x="Strike", y="Type", color="Gamma"),
 x=strikes,
 y=['CALL', 'PUT'],
 color_continuous_scale='Viridis',
 title="Gamma Heatmap"
 )
 st.plotly_chart(fig_gamma, use_container_width=True)

def render_volatility_surface(data_provider, chart_generator):
 """Render the volatility surface view"""
 st.header("🌊 Volatility Surface Analysis")

 # Symbol selection
 col1, col2 = st.columns(2)

 with col1:
 symbol = st.selectbox("Select Symbol", ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"], key="vol_symbol")

 with col2:
 surface_type = st.selectbox("Surface Type", ["Implied Volatility", "Bid-Ask Spread"])

 # Get volatility surface data
 vol_surface_data = data_provider.get_volatility_surface(symbol)

 if vol_surface_data is not None:
 strikes, expiries, vol_matrix = vol_surface_data

 # 3D Volatility Surface
 st.subheader("📈 3D Volatility Surface")

 fig_3d = go.Figure(data=[
 go.Surface(
 x=strikes,
 y=expiries,
 z=vol_matrix,
 colorscale='Viridis',
 name=f'{symbol} Volatility Surface'
 )
 ])

 fig_3d.update_layout(
 title=f'{symbol} Implied Volatility Surface',
 scene=dict(
 xaxis_title='Strike Price',
 yaxis_title='Days to Expiration',
 zaxis_title='Implied Volatility',
 camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
 ),
 height=600
 )

 st.plotly_chart(fig_3d, use_container_width=True)

 # Volatility smile for different expirations
 st.subheader("😊 Volatility Smile Evolution")

 fig_smile = go.Figure()

 colors = px.colors.qualitative.Set1
 for i, expiry in enumerate(expiries[::3]): # Show every 3rd expiry to avoid clutter
 idx = list(expiries).index(expiry)
 fig_smile.add_trace(
 go.Scatter(
 x=strikes,
 y=vol_matrix[idx],
 mode='lines+markers',
 name=f'{expiry} DTE',
 line=dict(color=colors[i % len(colors)], width=2)
 )
 )

 fig_smile.update_layout(
 title=f'{symbol} Volatility Smile by Expiration',
 xaxis_title='Strike Price',
 yaxis_title='Implied Volatility',
 height=400,
 hovermode='x unified'
 )

 st.plotly_chart(fig_smile, use_container_width=True)

 # Term structure analysis
 st.subheader("📊 Volatility Term Structure")

 # ATM volatility across expirations
 current_price = data_provider.get_current_price(symbol)
 atm_strikes_idx = [np.argmin(np.abs(np.array(strikes) - current_price)) for _ in expiries]
 atm_vols = [vol_matrix[i][atm_strikes_idx[i]] for i in range(len(expiries))]

 fig_term = go.Figure()
 fig_term.add_trace(
 go.Scatter(
 x=expiries,
 y=atm_vols,
 mode='lines+markers',
 name='ATM Implied Volatility',
 line=dict(color='blue', width=3),
 marker=dict(size=8)
 )
 )

 fig_term.update_layout(
 title=f'{symbol} ATM Volatility Term Structure',
 xaxis_title='Days to Expiration',
 yaxis_title='Implied Volatility',
 height=400
 )

 st.plotly_chart(fig_term, use_container_width=True)

 # Volatility statistics
 col1, col2, col3 = st.columns(3)

 with col1:
 st.metric("Average IV", f"{np.mean(vol_matrix):.2%}")

 with col2:
 st.metric("IV Range", f"{np.ptp(vol_matrix):.2%}")

 with col3:
 st.metric("Skew", f"{np.mean(vol_matrix[:, -1]) - np.mean(vol_matrix[:, 0]):.2%}")

def render_strategy_performance(portfolio_manager, chart_generator):
 """Render strategy performance tracking"""
 st.header("📈 Strategy Performance Tracking")

 # Strategy selection
 strategies = portfolio_manager.get_available_strategies()
 selected_strategy = st.selectbox("Select Strategy", strategies)

 # Performance metrics
 performance_data = portfolio_manager.get_strategy_performance(selected_strategy)

 # Key metrics display
 col1, col2, col3, col4 = st.columns(4)

 with col1:
 st.metric(
 "Total Return",
 f"{performance_data['total_return']:.2%}",
 f"{performance_data['period_return']:+.2%}"
 )

 with col2:
 st.metric(
 "Sharpe Ratio",
 f"{performance_data['sharpe_ratio']:.2f}",
 f"{performance_data['sharpe_change']:+.2f}"
 )

 with col3:
 st.metric(
 "Max Drawdown",
 f"{performance_data['max_drawdown']:.2%}",
 f"{performance_data['dd_change']:+.2%}",
 delta_color="inverse"
 )

 with col4:
 st.metric(
 "Win Rate",
 f"{performance_data['win_rate']:.1%}",
 f"{performance_data['win_rate_change']:+.1%}"
 )

 # Performance charts
 col1, col2 = st.columns(2)

 with col1:
 st.subheader("💰 Cumulative P&L")

 pnl_history = performance_data['pnl_history']

 fig_pnl = go.Figure()
 fig_pnl.add_trace(
 go.Scatter(
 x=pnl_history['date'],
 y=pnl_history['cumulative_pnl'],
 mode='lines',
 name='Cumulative P&L',
 line=dict(color='green', width=2),
 fill='tonexty' if any(pnl_history['cumulative_pnl'] < 0) else 'tozeroy'
 )
 )

 if any(pnl_history['cumulative_pnl'] < 0):
 fig_pnl.add_hline(y=0, line_dash="dash", line_color="red")

 fig_pnl.update_layout(
 title=f"{selected_strategy} Cumulative P&L",
 xaxis_title="Date",
 yaxis_title="P&L ($)",
 height=400
 )

 st.plotly_chart(fig_pnl, use_container_width=True)

 with col2:
 st.subheader("📊 Monthly Returns")

 monthly_returns = performance_data['monthly_returns']

 colors = ['green' if x >= 0 else 'red' for x in monthly_returns['return']]

 fig_monthly = go.Figure()
 fig_monthly.add_trace(
 go.Bar(
 x=monthly_returns['month'],
 y=monthly_returns['return'],
 marker_color=colors,
 name='Monthly Return'
 )
 )

 fig_monthly.update_layout(
 title=f"{selected_strategy} Monthly Returns",
 xaxis_title="Month",
 yaxis_title="Return (%)",
 height=400
 )

 st.plotly_chart(fig_monthly, use_container_width=True)

 # Strategy-specific metrics
 st.subheader("📋 Strategy Details")

 strategy_details = portfolio_manager.get_strategy_details(selected_strategy)

 col1, col2 = st.columns(2)

 with col1:
 st.write("**Strategy Parameters:**")
 for param, value in strategy_details['parameters'].items():
 st.write(f"• {param}: {value}")

 with col2:
 st.write("**Performance Statistics:**")
 stats = strategy_details['statistics']
 for stat, value in stats.items():
 st.write(f"• {stat}: {value}")

 # Risk attribution waterfall chart
 st.subheader("🌊 Risk Attribution Waterfall")

 risk_attribution = performance_data['risk_attribution']

 fig_waterfall = go.Figure(go.Waterfall(
 name="Risk Attribution",
 orientation="v",
 measure=["relative"] * (len(risk_attribution) - 1) + ["total"],
 x=list(risk_attribution.keys()),
 y=list(risk_attribution.values()),
 connector={"line": {"color": "rgb(63, 63, 63)"}},
 decreasing={"marker": {"color": "red"}},
 increasing={"marker": {"color": "green"}},
 totals={"marker": {"color": "blue"}}
 ))

 fig_waterfall.update_layout(
 title=f"{selected_strategy} Risk Attribution",
 height=400
 )

 st.plotly_chart(fig_waterfall, use_container_width=True)

def render_market_analysis(market_analyzer, chart_generator):
 """Render market analysis view"""
 st.header("🔍 Market Analysis")

 # Market overview metrics
 market_overview = market_analyzer.get_market_overview()

 col1, col2, col3, col4 = st.columns(4)

 with col1:
 vix_level = market_overview['vix']
 vix_color = "🔴" if vix_level > 30 else "🟡" if vix_level > 20 else "🟢"
 st.metric("VIX Level", f"{vix_level:.2f}", delta=f"{market_overview['vix_change']:+.2f}")
 st.write(f"{vix_color} Fear Level")

 with col2:
 put_call_ratio = market_overview['put_call_ratio']
 pc_sentiment = "Bearish" if put_call_ratio > 1.1 else "Bullish" if put_call_ratio < 0.9 else "Neutral"
 st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
 st.write(f"📊 Sentiment: {pc_sentiment}")

 with col3:
 iv_rank = market_overview['iv_rank']
 st.metric("IV Rank", f"{iv_rank:.0f}")
 st.write("📈 Volatility Percentile")

 with col4:
 volume_surge = market_overview['volume_surge']
 st.metric("Volume Surge", f"{volume_surge:.1f}x")
 st.write("📊 vs 20-day avg")

 # Cross-asset correlation heatmap
 st.subheader("🔗 Cross-Asset Correlation Matrix")

 correlation_data = market_analyzer.get_correlation_matrix()

 fig_corr = px.imshow(
 correlation_data['matrix'],
 labels=dict(x="Asset", y="Asset", color="Correlation"),
 x=correlation_data['assets'],
 y=correlation_data['assets'],
 color_continuous_scale='RdBu_r',
 title="Asset Correlation Heatmap",
 zmin=-1,
 zmax=1
 )

 # Add correlation values as text
 for i in range(len(correlation_data['assets'])):
 for j in range(len(correlation_data['assets'])):
 fig_corr.add_annotation(
 x=j, y=i,
 text=f"{correlation_data['matrix'][i][j]:.2f}",
 showarrow=False,
 font=dict(color="white" if abs(correlation_data['matrix'][i][j]) > 0.5 else "black")
 )

 fig_corr.update_layout(height=500)
 st.plotly_chart(fig_corr, use_container_width=True)

 # Volatility term structure comparison
 st.subheader("📊 Volatility Term Structure Comparison")

 term_structure_data = market_analyzer.get_term_structure_comparison()

 fig_term_comp = go.Figure()

 for asset in term_structure_data['assets']:
 fig_term_comp.add_trace(
 go.Scatter(
 x=term_structure_data['days_to_expiry'],
 y=term_structure_data['volatilities'][asset],
 mode='lines+markers',
 name=asset,
 line=dict(width=2)
 )
 )

 fig_term_comp.update_layout(
 title="Volatility Term Structure Comparison",
 xaxis_title="Days to Expiration",
 yaxis_title="Implied Volatility",
 height=400,
 hovermode='x unified'
 )

 st.plotly_chart(fig_term_comp, use_container_width=True)

 # Sector rotation analysis
 col1, col2 = st.columns(2)

 with col1:
 st.subheader("🔄 Sector Rotation")

 sector_data = market_analyzer.get_sector_analysis()

 fig_sector = px.scatter(
 sector_data,
 x='momentum',
 y='volatility',
 size='volume',
 color='performance',
 hover_name='sector',
 title="Sector Momentum vs Volatility",
 color_continuous_scale='RdYlGn'
 )

 fig_sector.update_layout(height=400)
 st.plotly_chart(fig_sector, use_container_width=True)

 with col2:
 st.subheader("📈 IV Rank Distribution")

 iv_distribution = market_analyzer.get_iv_rank_distribution()

 fig_iv_dist = px.histogram(
 iv_distribution,
 x='iv_rank',
 nbins=20,
 title="IV Rank Distribution Across Market",
 color_discrete_sequence=['skyblue']
 )

 fig_iv_dist.update_layout(height=400)
 st.plotly_chart(fig_iv_dist, use_container_width=True)

def render_flow_alerts(market_analyzer, data_provider):
 """Render options flow and alerts"""
 st.header("⚡ Options Flow & Alerts")

 # Real-time alerts section
 st.subheader("🚨 Active Alerts")

 alerts = market_analyzer.get_active_alerts()

 # Display alerts with appropriate styling
 for alert in alerts:
 if alert['severity'] == 'HIGH':
 st.error(f"🔴 **{alert['title']}**: {alert['message']}")
 elif alert['severity'] == 'MEDIUM':
 st.warning(f"🟡 **{alert['title']}**: {alert['message']}")
 else:
 st.info(f"🔵 **{alert['title']}**: {alert['message']}")

 if not alerts:
 st.success("✅ No active alerts - All systems normal")

 # Options flow analysis
 st.subheader("🌊 Real-time Options Flow")

 # Flow filter controls
 col1, col2, col3 = st.columns(3)

 with col1:
 flow_timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "1d"])

 with col2:
 min_premium = st.number_input("Min Premium ($)", value=10000, step=5000)

 with col3:
 flow_type = st.selectbox("Flow Type", ["All", "Calls", "Puts", "Unusual"])

 # Get options flow data
 flow_data = data_provider.get_options_flow(flow_timeframe, min_premium, flow_type)

 if not flow_data.empty:
 # Flow bubble chart
 fig_flow = px.scatter(
 flow_data,
 x='time',
 y='strike',
 size='premium',
 color='sentiment',
 hover_data=['symbol', 'volume', 'type'],
 title="Options Flow Bubble Chart",
 color_discrete_map={'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}
 )

 fig_flow.update_layout(height=500)
 st.plotly_chart(fig_flow, use_container_width=True)

 # Flow summary table
 st.subheader("📋 Flow Summary")

 # Aggregate flow data
 flow_summary = flow_data.groupby(['symbol', 'type']).agg({
 'premium': 'sum',
 'volume': 'sum',
 'sentiment': lambda x: x.mode()[0] if not x.empty else 'neutral'
 }).reset_index()

 # Style the flow summary
 def highlight_flow(row):
 if row['sentiment'] == 'bullish':
 return ['background-color: lightgreen'] * len(row)
 elif row['sentiment'] == 'bearish':
 return ['background-color: lightcoral'] * len(row)
 else:
 return ['background-color: lightyellow'] * len(row)

 styled_flow = flow_summary.style.apply(highlight_flow, axis=1)
 st.dataframe(styled_flow, use_container_width=True)

 # Unusual activity detection
 st.subheader("🎯 Unusual Activity Detection")

 unusual_activity = market_analyzer.detect_unusual_activity(flow_data)

 if not unusual_activity.empty:
 for _, activity in unusual_activity.iterrows():
 severity = activity['severity']
 if severity == 'HIGH':
 st.error(f"🔥 **{activity['symbol']}**: {activity['description']}")
 else:
 st.warning(f"⚠️ **{activity['symbol']}**: {activity['description']}")
 else:
 st.info("🔍 No unusual activity detected in current timeframe")

 # Sentiment gauges
 st.subheader("📊 Market Sentiment Gauges")

 col1, col2, col3 = st.columns(3)

 sentiment_data = market_analyzer.get_sentiment_metrics()

 with col1:
 # Put/Call sentiment gauge
 pc_sentiment = sentiment_data['put_call_sentiment']

 fig_pc_gauge = go.Figure(go.Indicator(
 mode="gauge+number",
 value=pc_sentiment,
 domain={'x': [0, 1], 'y': [0, 1]},
 title={'text': "Put/Call Sentiment"},
 gauge={
 'axis': {'range': [None, 100]},
 'bar': {'color': "darkblue"},
 'steps': [
 {'range': [0, 30], 'color': "lightgreen"},
 {'range': [30, 70], 'color': "yellow"},
 {'range': [70, 100], 'color': "lightcoral"}
 ],
 'threshold': {
 'line': {'color': "red", 'width': 4},
 'thickness': 0.75,
 'value': 80
 }
 }
 ))

 fig_pc_gauge.update_layout(height=300)
 st.plotly_chart(fig_pc_gauge, use_container_width=True)

 with col2:
 # Volume sentiment gauge
 vol_sentiment = sentiment_data['volume_sentiment']

 fig_vol_gauge = go.Figure(go.Indicator(
 mode="gauge+number",
 value=vol_sentiment,
 domain={'x': [0, 1], 'y': [0, 1]},
 title={'text': "Volume Sentiment"},
 gauge={
 'axis': {'range': [None, 100]},
 'bar': {'color': "darkgreen"},
 'steps': [
 {'range': [0, 30], 'color': "lightcoral"},
 {'range': [30, 70], 'color': "yellow"},
 {'range': [70, 100], 'color': "lightgreen"}
 ]
 }
 ))

 fig_vol_gauge.update_layout(height=300)
 st.plotly_chart(fig_vol_gauge, use_container_width=True)

 with col3:
 # Fear/Greed gauge
 fear_greed = sentiment_data['fear_greed_index']

 fig_fg_gauge = go.Figure(go.Indicator(
 mode="gauge+number",
 value=fear_greed,
 domain={'x': [0, 1], 'y': [0, 1]},
 title={'text': "Fear & Greed Index"},
 gauge={
 'axis': {'range': [None, 100]},
 'bar': {'color': "purple"},
 'steps': [
 {'range': [0, 25], 'color': "red"},
 {'range': [25, 45], 'color': "orange"},
 {'range': [45, 55], 'color': "yellow"},
 {'range': [55, 75], 'color': "lightgreen"},
 {'range': [75, 100], 'color': "green"}
 ]
 }
 ))

 fig_fg_gauge.update_layout(height=300)
 st.plotly_chart(fig_fg_gauge, use_container_width=True)

def render_risk_management(portfolio_manager, chart_generator):
 """Render risk management dashboard"""
 st.header("📊 Risk Management Dashboard")

 # Risk metrics overview
 risk_metrics = portfolio_manager.get_risk_metrics()

 col1, col2, col3, col4 = st.columns(4)

 with col1:
 var_95 = risk_metrics['var_95']
 var_color = "🔴" if var_95 > 50000 else "🟡" if var_95 > 25000 else "🟢"
 st.metric("VaR (95%)", f"${var_95:,.0f}")
 st.write(f"{var_color} Risk Level")

 with col2:
 max_drawdown = risk_metrics['max_drawdown']
 st.metric("Max Drawdown", f"{max_drawdown:.2%}", delta_color="inverse")

 with col3:
 leverage_ratio = risk_metrics['leverage_ratio']
 st.metric("Leverage Ratio", f"{leverage_ratio:.2f}x")

 with col4:
 concentration_risk = risk_metrics['concentration_risk']
 st.metric("Concentration Risk", f"{concentration_risk:.1%}")

 # Risk limit monitoring
 st.subheader("⚠️ Risk Limit Monitoring")

 risk_limits = portfolio_manager.get_risk_limits()
 current_exposure = portfolio_manager.get_current_exposure()

 # Create risk limit gauges
 col1, col2 = st.columns(2)

 with col1:
 # Delta limit gauge
 delta_utilization = abs(current_exposure['delta']) / risk_limits['max_delta'] * 100

 fig_delta_limit = go.Figure(go.Indicator(
 mode="gauge+number+delta",
 value=delta_utilization,
 domain={'x': [0, 1], 'y': [0, 1]},
 title={'text': "Delta Limit Utilization (%)"},
 delta={'reference': 80},
 gauge={
 'axis': {'range': [None, 100]},
 'bar': {'color': "darkblue"},
 'steps': [
 {'range': [0, 60], 'color': "lightgreen"},
 {'range': [60, 80], 'color': "yellow"},
 {'range': [80, 100], 'color': "lightcoral"}
 ],
 'threshold': {
 'line': {'color': "red", 'width': 4},
 'thickness': 0.75,
 'value': 90
 }
 }
 ))

 fig_delta_limit.update_layout(height=300)
 st.plotly_chart(fig_delta_limit, use_container_width=True)

 with col2:
 # Gamma limit gauge
 gamma_utilization = abs(current_exposure['gamma']) / risk_limits['max_gamma'] * 100

 fig_gamma_limit = go.Figure(go.Indicator(
 mode="gauge+number+delta",
 value=gamma_utilization,
 domain={'x': [0, 1], 'y': [0, 1]},
 title={'text': "Gamma Limit Utilization (%)"},
 delta={'reference': 80},
 gauge={
 'axis': {'range': [None, 100]},
 'bar': {'color': "darkgreen"},
 'steps': [
 {'range': [0, 60], 'color': "lightgreen"},
 {'range': [60, 80], 'color': "yellow"},
 {'range': [80, 100], 'color': "lightcoral"}
 ],
 'threshold': {
 'line': {'color': "red", 'width': 4},
 'thickness': 0.75,
 'value': 90
 }
 }
 ))

 fig_gamma_limit.update_layout(height=300)
 st.plotly_chart(fig_gamma_limit, use_container_width=True)

 # Stress testing results
 st.subheader("🧪 Stress Testing Results")

 stress_scenarios = portfolio_manager.get_stress_test_results()

 scenario_names = list(stress_scenarios.keys())
 scenario_pnl = list(stress_scenarios.values())

 colors = ['red' if pnl < 0 else 'green' for pnl in scenario_pnl]

 fig_stress = go.Figure()
 fig_stress.add_trace(
 go.Bar(
 x=scenario_names,
 y=scenario_pnl,
 marker_color=colors,
 name='Scenario P&L'
 )
 )

 fig_stress.add_hline(y=0, line_dash="dash", line_color="black")

 fig_stress.update_layout(
 title="Stress Test Scenario Results",
 xaxis_title="Scenario",
 yaxis_title="P&L Impact ($)",
 height=400
 )

 st.plotly_chart(fig_stress, use_container_width=True)

 # Risk decomposition
 col1, col2 = st.columns(2)

 with col1:
 st.subheader("🥧 Risk Decomposition")

 risk_breakdown = portfolio_manager.get_risk_decomposition()

 fig_risk_pie = px.pie(
 values=list(risk_breakdown.values()),
 names=list(risk_breakdown.keys()),
 title="Portfolio Risk by Component",
 color_discrete_sequence=px.colors.qualitative.Set3
 )

 fig_risk_pie.update_layout(height=400)
 st.plotly_chart(fig_risk_pie, use_container_width=True)

 with col2:
 st.subheader("📈 Risk Evolution")

 risk_history = portfolio_manager.get_risk_history()

 fig_risk_evolution = go.Figure()

 fig_risk_evolution.add_trace(
 go.Scatter(
 x=risk_history['date'],
 y=risk_history['var_95'],
 mode='lines+markers',
 name='VaR 95%',
 line=dict(color='red', width=2)
 )
 )

 fig_risk_evolution.add_trace(
 go.Scatter(
 x=risk_history['date'],
 y=risk_history['expected_shortfall'],
 mode='lines+markers',
 name='Expected Shortfall',
 line=dict(color='orange', width=2)
 )
 )

 fig_risk_evolution.update_layout(
 title="Risk Metrics Evolution",
 xaxis_title="Date",
 yaxis_title="Risk ($)",
 height=400
 )

 st.plotly_chart(fig_risk_evolution, use_container_width=True)

if __name__ == "__main__":
 main()