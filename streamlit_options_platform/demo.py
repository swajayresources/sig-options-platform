"""
Professional Options Trading Platform Demo
Demonstration script showcasing all platform features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Configure the demo page
st.set_page_config(
 page_title="Options Trading Platform Demo",
 page_icon="📈",
 layout="wide",
 initial_sidebar_state="expanded"
)

# Custom CSS for the demo
st.markdown("""
<style>.demo-header {
 background: linear-gradient(135deg, #1f77b4, #ff7f0e);
 color: white;
 padding: 2rem;
 border-radius: 10px;
 text-align: center;
 margin-bottom: 2rem;
 }.feature-card {
 background-color: #f8f9fa;
 padding: 1.5rem;
 border-radius: 8px;
 border-left: 4px solid #1f77b4;
 margin-bottom: 1rem;
 }.metric-highlight {
 background-color: #e3f2fd;
 padding: 1rem;
 border-radius: 5px;
 border: 1px solid #1976d2;
 }.demo-section {
 margin: 2rem 0;
 padding: 1rem;
 background-color: #fafafa;
 border-radius: 8px;
 }
</style>
""", unsafe_allow_html=True)

def main():
 """Main demo function"""

 # Demo header
 st.markdown("""
 <div class="demo-header">
 <h1>🚀 Professional Options Trading Platform</h1>
 <h3>Institutional-Grade Analytics with Streamlit</h3>
 <p>Real-time portfolio monitoring • Advanced visualizations • Risk management</p>
 </div>
 """, unsafe_allow_html=True)

 # Feature overview
 st.markdown("## 🎯 Platform Features Overview")

 col1, col2, col3 = st.columns(3)

 with col1:
 st.markdown("""
 <div class="feature-card">
 <h4>📊 Portfolio Analytics</h4>
 <ul>
 <li>Real-time Greeks tracking</li>
 <li>P&L attribution analysis</li>
 <li>Risk decomposition</li>
 <li>Performance metrics</li>
 </ul>
 </div>
 """, unsafe_allow_html=True)

 with col2:
 st.markdown("""
 <div class="feature-card">
 <h4>📈 Market Analysis</h4>
 <ul>
 <li>3D volatility surfaces</li>
 <li>Options chain visualization</li>
 <li>Flow analysis & alerts</li>
 <li>Correlation monitoring</li>
 </ul>
 </div>
 """, unsafe_allow_html=True)

 with col3:
 st.markdown("""
 <div class="feature-card">
 <h4>⚡ Real-time Features</h4>
 <ul>
 <li>Live data updates</li>
 <li>Interactive dashboards</li>
 <li>Automated alerts</li>
 <li>Mobile responsive</li>
 </ul>
 </div>
 """, unsafe_allow_html=True)

 # Live Demo Section
 st.markdown("## 🔴 Live Demo")

 demo_tab1, demo_tab2, demo_tab3, demo_tab4 = st.tabs([
 "📊 Portfolio Dashboard",
 "🌊 Volatility Surface",
 "📋 Options Chain",
 "⚡ Real-time Alerts"
 ])

 with demo_tab1:
 demo_portfolio_dashboard()

 with demo_tab2:
 demo_volatility_surface()

 with demo_tab3:
 demo_options_chain()

 with demo_tab4:
 demo_real_time_alerts()

 # Technical Features
 st.markdown("## 🛠️ Technical Implementation")

 tech_col1, tech_col2 = st.columns(2)

 with tech_col1:
 st.markdown("""
 ### Streamlit Advantages
 ```python
 # Real-time updates
 if st.session_state.auto_refresh:
 st.rerun()

 # Interactive widgets
 delta_limit = st.slider("Delta Limit", 0, 2000, 1000)

 # Professional styling
 st.markdown(custom_css, unsafe_allow_html=True)

 # File upload & export
 uploaded_file = st.file_uploader("Portfolio Data")
 st.download_button("Download Report", data)
 ```
 """)

 with tech_col2:
 st.markdown("""
 ### Advanced Visualizations
 ```python
 # 3D Volatility Surface
 fig = go.Figure(data=[go.Surface(
 x=strikes, y=expiries, z=vol_matrix,
 colorscale='Viridis'
 )])

 # Greeks Heatmap
 fig = px.imshow(greeks_matrix,
 color_continuous_scale='RdYlBu_r')

 # Real-time Charts
 st.plotly_chart(fig, use_container_width=True)
 ```
 """)

def demo_portfolio_dashboard():
 """Demo portfolio dashboard functionality"""
 st.subheader("📊 Real-time Portfolio Monitoring")

 # Simulated real-time metrics
 col1, col2, col3, col4 = st.columns(4)

 # Generate random but realistic data
 delta = np.random.randint(800, 1200)
 gamma = np.random.randint(400, 600)
 theta = np.random.randint(-350, -250)
 vega = np.random.randint(1200, 1800)

 with col1:
 st.metric("Portfolio Delta", f"{delta:,}", delta=f"{np.random.randint(-50, 50):+}")
 with col2:
 st.metric("Portfolio Gamma", f"{gamma:,}", delta=f"{np.random.randint(-25, 25):+}")
 with col3:
 st.metric("Portfolio Theta", f"${theta:,}", delta=f"${np.random.randint(-100, 50):+}")
 with col4:
 st.metric("Portfolio Vega", f"${vega:,}", delta=f"${np.random.randint(-200, 200):+}")

 # Portfolio composition chart
 col1, col2 = st.columns(2)

 with col1:
 # Portfolio composition pie chart
 symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
 values = np.random.dirichlet(np.ones(5)) * 100

 fig_pie = px.pie(values=values, names=symbols, title="Portfolio Composition (%)")
 fig_pie.update_layout(height=400)
 st.plotly_chart(fig_pie, use_container_width=True)

 with col2:
 # P&L attribution bar chart
 greeks = ['Delta P&L', 'Gamma P&L', 'Theta P&L', 'Vega P&L', 'Other']
 pnl_values = [np.random.uniform(-1000, 2000) for _ in greeks]
 colors = ['green' if x >= 0 else 'red' for x in pnl_values]

 fig_bar = go.Figure(data=[go.Bar(x=greeks, y=pnl_values, marker_color=colors)])
 fig_bar.update_layout(title="P&L Attribution by Greeks", height=400)
 st.plotly_chart(fig_bar, use_container_width=True)

 # Greeks evolution timeline
 st.subheader("📈 Greeks Evolution")
 dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')

 fig_timeline = go.Figure()

 for greek, color in zip(['Delta', 'Gamma', 'Theta', 'Vega'],
 ['blue', 'green', 'red', 'orange']):
 base_value = {'Delta': 1000, 'Gamma': 500, 'Theta': -300, 'Vega': 1500}[greek]
 values = [base_value + np.random.normal(0, abs(base_value) * 0.1) for _ in dates]

 fig_timeline.add_trace(go.Scatter(
 x=dates, y=values, mode='lines', name=greek, line=dict(color=color, width=2)
 ))

 fig_timeline.update_layout(
 title="Portfolio Greeks Evolution (30 Days)",
 xaxis_title="Date",
 yaxis_title="Greeks Value",
 height=400
 )
 st.plotly_chart(fig_timeline, use_container_width=True)

def demo_volatility_surface():
 """Demo 3D volatility surface"""
 st.subheader("🌊 Interactive 3D Volatility Surface")

 # Symbol selection
 symbol = st.selectbox("Select Symbol for Volatility Analysis",
 ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'])

 # Generate realistic volatility surface data
 strikes = np.arange(80, 121, 5) # Strike range
 days_to_expiry = np.arange(7, 91, 7) # Days to expiration

 # Create volatility matrix with realistic patterns
 vol_matrix = np.zeros((len(days_to_expiry), len(strikes)))

 for i, dte in enumerate(days_to_expiry):
 for j, strike in enumerate(strikes):
 # Base volatility with term structure
 base_vol = 0.25 + (dte / 365) * 0.05

 # Add volatility skew (puts more expensive)
 moneyness = strike / 100 # Assume ATM at 100
 if moneyness < 1:
 skew = (1 - moneyness) * 0.3
 else:
 skew = (moneyness - 1) * 0.1

 # Add some randomness
 noise = np.random.normal(0, 0.02)

 vol_matrix[i, j] = max(0.1, (base_vol + skew + noise)) * 100

 # Create 3D surface plot
 fig_3d = go.Figure(data=[
 go.Surface(
 x=strikes,
 y=days_to_expiry,
 z=vol_matrix,
 colorscale='Viridis',
 name=f'{symbol} Vol Surface',
 hovertemplate='<b>Strike:</b> %{x}<br>' +
 '<b>DTE:</b> %{y}<br>' +
 '<b>IV:</b> %{z:.1f}%<extra></extra>'
 )
 ])

 fig_3d.update_layout(
 title=f'{symbol} Implied Volatility Surface',
 scene=dict(
 xaxis_title='Strike Price',
 yaxis_title='Days to Expiration',
 zaxis_title='Implied Volatility (%)',
 camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
 ),
 height=600
 )

 st.plotly_chart(fig_3d, use_container_width=True)

 # Volatility smile for different expirations
 st.subheader("😊 Volatility Smile Analysis")

 fig_smile = go.Figure()

 colors = ['blue', 'red', 'green', 'orange', 'purple']
 for i, (dte, color) in enumerate(zip([7, 14, 30, 60, 90], colors)):
 idx = list(days_to_expiry).index(dte) if dte in days_to_expiry else 0
 fig_smile.add_trace(go.Scatter(
 x=strikes,
 y=vol_matrix[idx],
 mode='lines+markers',
 name=f'{dte} DTE',
 line=dict(color=color, width=2)
 ))

 fig_smile.update_layout(
 title=f'{symbol} Volatility Smile by Expiration',
 xaxis_title='Strike Price',
 yaxis_title='Implied Volatility (%)',
 height=400
 )

 st.plotly_chart(fig_smile, use_container_width=True)

def demo_options_chain():
 """Demo options chain visualization"""
 st.subheader("📋 Interactive Options Chain")

 # Controls
 col1, col2, col3 = st.columns(3)

 with col1:
 symbol = st.selectbox("Symbol", ['AAPL', 'MSFT', 'GOOGL'], key="chain_symbol")
 with col2:
 expiry = st.selectbox("Expiration", ['2024-01-19', '2024-01-26', '2024-02-02'])
 with col3:
 option_type = st.selectbox("Type", ['Both', 'Calls Only', 'Puts Only'])

 # Generate sample options chain data
 current_price = np.random.uniform(170, 180)
 strikes = np.arange(current_price - 20, current_price + 21, 5)

 options_data = []

 for strike in strikes:
 for opt_type in (['CALL', 'PUT'] if option_type == 'Both'
 else ['CALL'] if option_type == 'Calls Only'
 else ['PUT']):

 # Calculate realistic option prices and Greeks
 moneyness = strike / current_price
 base_iv = 0.25 + abs(moneyness - 1) * 0.1

 if opt_type == 'CALL':
 intrinsic = max(0, current_price - strike)
 delta = 0.5 if moneyness == 1 else (0.8 if moneyness < 1 else 0.2)
 else:
 intrinsic = max(0, strike - current_price)
 delta = -0.5 if moneyness == 1 else (-0.2 if moneyness < 1 else -0.8)

 time_value = base_iv * current_price * 0.1
 option_price = intrinsic + time_value

 options_data.append({
 'Type': opt_type,
 'Strike': strike,
 'Last': round(option_price, 2),
 'Bid': round(option_price - 0.1, 2),
 'Ask': round(option_price + 0.1, 2),
 'Volume': np.random.randint(0, 1000),
 'Open Int': np.random.randint(0, 5000),
 'IV': round(base_iv * 100, 1),
 'Delta': round(delta, 3),
 'Gamma': round(0.01, 3),
 'Theta': round(-0.05, 3),
 'Vega': round(0.15, 3)
 })

 df = pd.DataFrame(options_data)

 # Style the dataframe
 def highlight_itm(row):
 if row['Type'] == 'CALL':
 color = 'background-color: lightgreen' if row['Strike'] < current_price else 'background-color: lightgray'
 else:
 color = 'background-color: lightcoral' if row['Strike'] > current_price else 'background-color: lightgray'
 return [color] * len(row)

 styled_df = df.style.apply(highlight_itm, axis=1)

 # Display current stock price
 st.markdown(f"**{symbol} Current Price: ${current_price:.2f}**")

 # Display options chain
 st.dataframe(styled_df, use_container_width=True, height=400)

 # Greeks heatmap
 st.subheader("🔥 Greeks Heatmap")

 col1, col2 = st.columns(2)

 with col1:
 # Delta heatmap
 delta_data = df.pivot(index='Strike', columns='Type', values='Delta').fillna(0)
 fig_delta = px.imshow(
 delta_data.T,
 title="Delta Heatmap",
 color_continuous_scale='RdYlBu_r',
 aspect='auto'
 )
 st.plotly_chart(fig_delta, use_container_width=True)

 with col2:
 # Volume heatmap
 volume_data = df.pivot(index='Strike', columns='Type', values='Volume').fillna(0)
 fig_volume = px.imshow(
 volume_data.T,
 title="Volume Heatmap",
 color_continuous_scale='Viridis',
 aspect='auto'
 )
 st.plotly_chart(fig_volume, use_container_width=True)

def demo_real_time_alerts():
 """Demo real-time alerts system"""
 st.subheader("⚡ Real-time Monitoring & Alerts")

 # Alert controls
 col1, col2 = st.columns(2)

 with col1:
 auto_refresh = st.toggle("🔄 Auto Refresh Alerts", value=True)
 refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 3)

 with col2:
 alert_level = st.selectbox("Alert Level Filter", ['All', 'High', 'Medium', 'Low'])
 show_resolved = st.checkbox("Show Resolved Alerts")

 # Simulate real-time alerts
 alert_types = [
 ("🔴 Risk Limit Breach", "Portfolio delta exceeded limit: 1,150 (limit: 1,000)", "High"),
 ("🟡 Unusual Volume", "AAPL showing 3.2x normal volume", "Medium"),
 ("🟠 Volatility Spike", "VIX increased to 28.5 (+15% intraday)", "High"),
 ("🔵 Position Update", "Iron Condor position auto-adjusted", "Low"),
 ("🟡 P&L Alert", "Daily P&L reached $5,000 profit target", "Medium"),
 ]

 # Display alerts
 st.markdown("### 🚨 Active Alerts")

 if auto_refresh:
 # Simulate new alerts appearing
 num_alerts = np.random.randint(2, 5)
 displayed_alerts = np.random.choice(len(alert_types), num_alerts, replace=False)

 for i in displayed_alerts:
 icon, message, level = alert_types[i]
 timestamp = datetime.now() - timedelta(minutes=np.random.randint(1, 60))

 if alert_level == 'All' or alert_level == level:
 if level == 'High':
 st.error(f"{icon} **{level}**: {message}")
 elif level == 'Medium':
 st.warning(f"{icon} **{level}**: {message}")
 else:
 st.info(f"{icon} **{level}**: {message}")

 st.caption(f"🕐 {timestamp.strftime('%H:%M:%S')}")

 # Auto-refresh
 if auto_refresh:
 time.sleep(refresh_interval)
 st.rerun()

 # Market sentiment gauges
 st.markdown("### 📊 Market Sentiment Indicators")

 col1, col2, col3 = st.columns(3)

 with col1:
 # VIX gauge
 vix_value = np.random.uniform(15, 35)
 fig_vix = go.Figure(go.Indicator(
 mode="gauge+number",
 value=vix_value,
 title={'text': "VIX Level"},
 gauge={
 'axis': {'range': [None, 50]},
 'bar': {'color': "red" if vix_value > 30 else "orange" if vix_value > 20 else "green"},
 'steps': [
 {'range': [0, 20], 'color': "lightgreen"},
 {'range': [20, 30], 'color': "yellow"},
 {'range': [30, 50], 'color': "lightcoral"}
 ]
 }
 ))
 fig_vix.update_layout(height=250)
 st.plotly_chart(fig_vix, use_container_width=True)

 with col2:
 # Put/Call ratio gauge
 pc_ratio = np.random.uniform(0.7, 1.5)
 fig_pc = go.Figure(go.Indicator(
 mode="gauge+number",
 value=pc_ratio,
 title={'text': "Put/Call Ratio"},
 gauge={
 'axis': {'range': [0, 2]},
 'bar': {'color': "red" if pc_ratio > 1.2 else "green" if pc_ratio < 0.8 else "orange"},
 'steps': [
 {'range': [0, 0.8], 'color': "lightgreen"},
 {'range': [0.8, 1.2], 'color': "yellow"},
 {'range': [1.2, 2], 'color': "lightcoral"}
 ]
 }
 ))
 fig_pc.update_layout(height=250)
 st.plotly_chart(fig_pc, use_container_width=True)

 with col3:
 # Volume surge indicator
 volume_surge = np.random.uniform(0.5, 3.0)
 fig_vol = go.Figure(go.Indicator(
 mode="gauge+number",
 value=volume_surge,
 title={'text': "Volume Surge"},
 gauge={
 'axis': {'range': [0, 4]},
 'bar': {'color': "green" if volume_surge > 1.5 else "orange"},
 'steps': [
 {'range': [0, 1], 'color': "lightgray"},
 {'range': [1, 2], 'color': "yellow"},
 {'range': [2, 4], 'color': "lightgreen"}
 ]
 }
 ))
 fig_vol.update_layout(height=250)
 st.plotly_chart(fig_vol, use_container_width=True)

 # Options flow analysis
 st.markdown("### 🌊 Live Options Flow")

 # Generate sample flow data
 flow_data = []
 symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']

 for _ in range(20):
 flow_data.append({
 'Time': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
 'Symbol': np.random.choice(symbols),
 'Type': np.random.choice(['CALL', 'PUT']),
 'Strike': np.random.uniform(150, 200),
 'Volume': np.random.randint(100, 2000),
 'Premium': np.random.uniform(10000, 100000),
 'Sentiment': np.random.choice(['Bullish', 'Bearish', 'Neutral'])
 })

 flow_df = pd.DataFrame(flow_data)
 flow_df = flow_df.sort_values('Time', ascending=False)

 # Style the flow data
 def color_sentiment(val):
 if val == 'Bullish':
 return 'background-color: lightgreen'
 elif val == 'Bearish':
 return 'background-color: lightcoral'
 else:
 return 'background-color: lightyellow'

 styled_flow = flow_df.style.applymap(color_sentiment, subset=['Sentiment'])
 st.dataframe(styled_flow, use_container_width=True, height=300)

if __name__ == "__main__":
 main()