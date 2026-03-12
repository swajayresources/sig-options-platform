"""
Real-time Strategy Monitoring Dashboard

This module implements a comprehensive web-based monitoring dashboard for options
market making strategies with real-time P&L tracking, risk monitoring, position
management, and strategy performance visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import threading
import time
import asyncio
import websockets
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

try:
 from flask import Flask, render_template, jsonify, request, send_from_directory
 from flask_socketio import SocketIO, emit
 import plotly.graph_objects as go
 import plotly.express as px
 from plotly.utils import PlotlyJSONEncoder
 FLASK_AVAILABLE = True
except ImportError:
 FLASK_AVAILABLE = False
 warnings.warn("Flask/Plotly not available. Dashboard functionality will be limited.")

from market_making_strategies import StrategyManager, Greeks, Position, Trade, MarketData
from performance_monitoring import StrategyPerformanceMonitor, RiskMonitor, PerformanceSnapshot
from hedging_risk_management import HedgingRiskManager, RiskLimits, HedgingConfig
from portfolio_optimization import PortfolioOptimizer


@dataclass
class DashboardConfig:
 """Configuration for monitoring dashboard"""
 host: str = "localhost"
 port: int = 5000
 debug: bool = False
 update_frequency_ms: int = 1000 # 1 second updates

 # Display settings
 max_chart_points: int = 1000
 default_timeframe: str = "1D" # 1H, 4H, 1D, 1W
 refresh_rate_ms: int = 500

 # Alert settings
 enable_audio_alerts: bool = True
 alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
 'pnl_alert_threshold': -5000.0,
 'risk_alert_threshold': 0.8,
 'position_limit_threshold': 0.9
 })

 # Data retention
 max_history_hours: int = 24
 snapshot_frequency_seconds: int = 60


@dataclass
class AlertMessage:
 """Alert message structure"""
 timestamp: datetime
 severity: str # 'info', 'warning', 'error', 'critical'
 category: str # 'pnl', 'risk', 'position', 'strategy', 'system'
 title: str
 message: str
 strategy_id: Optional[str] = None
 acknowledged: bool = False


class DataAggregator:
 """Aggregates data from multiple sources for dashboard display"""

 def __init__(self, config: DashboardConfig):
 self.config = config
 self.data_cache: Dict[str, Any] = {}
 self.last_update: Dict[str, datetime] = {}
 self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

 def register_data_source(self, source_id: str, update_callback: Callable):
 """Register a data source"""
 self.subscribers[source_id].append(update_callback)

 def update_data(self, source_id: str, data: Any):
 """Update data from a source"""
 self.data_cache[source_id] = data
 self.last_update[source_id] = datetime.now()

 # Notify subscribers
 for callback in self.subscribers.get(source_id, []):
 try:
 callback(data)
 except Exception as e:
 warnings.warn(f"Data callback failed: {e}")

 def get_data(self, source_id: str) -> Any:
 """Get data from cache"""
 return self.data_cache.get(source_id)

 def get_consolidated_snapshot(self) -> Dict[str, Any]:
 """Get consolidated data snapshot for dashboard"""
 return {
 'timestamp': datetime.now().isoformat(),
 'data': self.data_cache.copy(),
 'last_updates': {k: v.isoformat() for k, v in self.last_update.items()}
 }


class AlertManager:
 """Manages alerts and notifications"""

 def __init__(self, config: DashboardConfig):
 self.config = config
 self.alerts: deque = deque(maxlen=1000)
 self.alert_callbacks: List[Callable] = []
 self.alert_counts = defaultdict(int)

 def add_alert(self, severity: str, category: str, title: str, message: str,
 strategy_id: Optional[str] = None):
 """Add new alert"""
 alert = AlertMessage(
 timestamp=datetime.now(),
 severity=severity,
 category=category,
 title=title,
 message=message,
 strategy_id=strategy_id
 )

 self.alerts.append(alert)
 self.alert_counts[severity] += 1

 # Notify callbacks
 for callback in self.alert_callbacks:
 try:
 callback(alert)
 except Exception as e:
 warnings.warn(f"Alert callback failed: {e}")

 return alert

 def acknowledge_alert(self, alert_index: int):
 """Acknowledge an alert"""
 if 0 <= alert_index < len(self.alerts):
 self.alerts[alert_index].acknowledged = True

 def get_recent_alerts(self, limit: int = 50) -> List[AlertMessage]:
 """Get recent alerts"""
 return list(self.alerts)[-limit:]

 def get_alert_summary(self) -> Dict[str, int]:
 """Get alert count summary"""
 recent_alerts = [a for a in self.alerts
 if a.timestamp > datetime.now() - timedelta(hours=24)]

 summary = defaultdict(int)
 for alert in recent_alerts:
 summary[alert.severity] += 1

 return dict(summary)

 def register_alert_callback(self, callback: Callable):
 """Register alert callback"""
 self.alert_callbacks.append(callback)


class ChartGenerator:
 """Generates charts for dashboard display"""

 def __init__(self):
 self.color_scheme = {
 'positive': '#00FF00',
 'negative': '#FF0000',
 'neutral': '#0080FF',
 'background': '#1E1E1E',
 'grid': '#404040'
 }

 def create_pnl_chart(self, pnl_data: List[Tuple[datetime, float]],
 timeframe: str = "1D") -> Dict[str, Any]:
 """Create P&L chart"""
 if not pnl_data:
 return self._empty_chart("P&L Chart", "No data available")

 timestamps, values = zip(*pnl_data)

 # Filter data based on timeframe
 filtered_data = self._filter_by_timeframe(list(zip(timestamps, values)), timeframe)
 if not filtered_data:
 return self._empty_chart("P&L Chart", "No data in timeframe")

 timestamps, values = zip(*filtered_data)

 fig = go.Figure()

 # Add P&L line
 fig.add_trace(go.Scatter(
 x=timestamps,
 y=values,
 mode='lines',
 name='P&L',
 line=dict(color=self.color_scheme['positive'] if values[-1] >= 0 else self.color_scheme['negative'])
 ))

 # Add zero line
 fig.add_hline(y=0, line_dash="dash", line_color="gray")

 fig.update_layout(
 title="Cumulative P&L",
 xaxis_title="Time",
 yaxis_title="P&L ($)",
 template="plotly_dark",
 height=400
 )

 return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

 def create_greeks_chart(self, greeks_data: List[Tuple[datetime, Greeks]]) -> Dict[str, Any]:
 """Create Greeks monitoring chart"""
 if not greeks_data:
 return self._empty_chart("Greeks Chart", "No Greeks data available")

 timestamps = [ts for ts, _ in greeks_data]
 delta_values = [g.delta for _, g in greeks_data]
 gamma_values = [g.gamma for _, g in greeks_data]
 vega_values = [g.vega for _, g in greeks_data]
 theta_values = [g.theta for _, g in greeks_data]

 fig = go.Figure()

 fig.add_trace(go.Scatter(x=timestamps, y=delta_values, name='Delta', line=dict(color='blue')))
 fig.add_trace(go.Scatter(x=timestamps, y=gamma_values, name='Gamma', line=dict(color='green')))
 fig.add_trace(go.Scatter(x=timestamps, y=vega_values, name='Vega', line=dict(color='red')))
 fig.add_trace(go.Scatter(x=timestamps, y=theta_values, name='Theta', line=dict(color='orange')))

 fig.update_layout(
 title="Portfolio Greeks",
 xaxis_title="Time",
 yaxis_title="Greeks Value",
 template="plotly_dark",
 height=400
 )

 return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

 def create_position_breakdown_chart(self, positions: Dict[str, Position]) -> Dict[str, Any]:
 """Create position breakdown pie chart"""
 if not positions:
 return self._empty_chart("Positions", "No positions")

 symbols = []
 values = []
 colors = []

 for symbol, position in positions.items():
 if position.quantity != 0:
 symbols.append(symbol)
 values.append(abs(position.market_value))
 colors.append(self.color_scheme['positive'] if position.quantity > 0 else self.color_scheme['negative'])

 if not symbols:
 return self._empty_chart("Positions", "No open positions")

 fig = go.Figure(data=[go.Pie(
 labels=symbols,
 values=values,
 marker_colors=colors,
 hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
 )])

 fig.update_layout(
 title="Position Breakdown",
 template="plotly_dark",
 height=400
 )

 return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

 def create_risk_gauge(self, current_var: float, var_limit: float) -> Dict[str, Any]:
 """Create risk gauge chart"""
 utilization = (current_var / var_limit) * 100 if var_limit > 0 else 0

 fig = go.Figure(go.Indicator(
 mode="gauge+number+delta",
 value=utilization,
 domain={'x': [0, 1], 'y': [0, 1]},
 title={'text': "VaR Utilization (%)"},
 delta={'reference': 80},
 gauge={
 'axis': {'range': [None, 100]},
 'bar': {'color': "darkblue"},
 'steps': [
 {'range': [0, 50], 'color': "lightgray"},
 {'range': [50, 80], 'color': "yellow"},
 {'range': [80, 100], 'color': "red"}
 ],
 'threshold': {
 'line': {'color': "red", 'width': 4},
 'thickness': 0.75,
 'value': 90
 }
 }
 ))

 fig.update_layout(
 template="plotly_dark",
 height=300
 )

 return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

 def _filter_by_timeframe(self, data: List[Tuple[datetime, float]], timeframe: str) -> List[Tuple[datetime, float]]:
 """Filter data by timeframe"""
 if not data:
 return data

 now = datetime.now()
 if timeframe == "1H":
 cutoff = now - timedelta(hours=1)
 elif timeframe == "4H":
 cutoff = now - timedelta(hours=4)
 elif timeframe == "1D":
 cutoff = now - timedelta(days=1)
 elif timeframe == "1W":
 cutoff = now - timedelta(weeks=1)
 else:
 return data

 return [(ts, val) for ts, val in data if ts >= cutoff]

 def _empty_chart(self, title: str, message: str) -> Dict[str, Any]:
 """Create empty chart with message"""
 fig = go.Figure()
 fig.add_annotation(
 text=message,
 xref="paper", yref="paper",
 x=0.5, y=0.5,
 showarrow=False,
 font=dict(size=16)
 )
 fig.update_layout(
 title=title,
 template="plotly_dark",
 height=400
 )
 return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


class MonitoringDashboard:
 """Main monitoring dashboard class"""

 def __init__(self, config: DashboardConfig = None):
 self.config = config or DashboardConfig()

 # Core components
 self.data_aggregator = DataAggregator(self.config)
 self.alert_manager = AlertManager(self.config)
 self.chart_generator = ChartGenerator()

 # Strategy management
 self.strategy_managers: Dict[str, StrategyManager] = {}
 self.performance_monitors: Dict[str, StrategyPerformanceMonitor] = {}
 self.risk_monitors: Dict[str, RiskMonitor] = {}

 # Data storage
 self.pnl_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_chart_points))
 self.greeks_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_chart_points))
 self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

 # Flask app and SocketIO
 if FLASK_AVAILABLE:
 self.app = Flask(__name__)
 self.app.config['SECRET_KEY'] = 'market_making_dashboard'
 self.socketio = SocketIO(self.app, cors_allowed_origins="*")
 self._setup_routes()
 else:
 self.app = None
 self.socketio = None

 # Background tasks
 self.is_running = False
 self.update_thread: Optional[threading.Thread] = None

 def add_strategy_manager(self, manager_id: str, strategy_manager: StrategyManager):
 """Add strategy manager to monitor"""
 self.strategy_managers[manager_id] = strategy_manager
 self.performance_monitors[manager_id] = StrategyPerformanceMonitor(manager_id)
 self.risk_monitors[manager_id] = RiskMonitor()

 # Register data sources
 self.data_aggregator.register_data_source(
 f"{manager_id}_performance",
 lambda data: self._update_performance_data(manager_id, data)
 )

 def start_monitoring(self):
 """Start the monitoring dashboard"""
 self.is_running = True

 # Start background update thread
 self.update_thread = threading.Thread(target=self._background_update_loop)
 self.update_thread.daemon = True
 self.update_thread.start()

 # Start Flask app if available
 if self.app and self.socketio:
 print(f"Starting dashboard on http://{self.config.host}:{self.config.port}")
 self.socketio.run(
 self.app,
 host=self.config.host,
 port=self.config.port,
 debug=self.config.debug
 )

 def stop_monitoring(self):
 """Stop the monitoring dashboard"""
 self.is_running = False
 if self.update_thread:
 self.update_thread.join()

 def _setup_routes(self):
 """Setup Flask routes"""

 @self.app.route('/')
 def index():
 return render_template('dashboard.html')

 @self.app.route('/api/overview')
 def api_overview():
 """Get dashboard overview data"""
 overview_data = self._get_overview_data()
 return jsonify(overview_data)

 @self.app.route('/api/strategy/<strategy_id>')
 def api_strategy_detail(strategy_id):
 """Get detailed strategy data"""
 strategy_data = self._get_strategy_data(strategy_id)
 return jsonify(strategy_data)

 @self.app.route('/api/charts/<chart_type>')
 def api_charts(chart_type):
 """Get chart data"""
 timeframe = request.args.get('timeframe', '1D')
 strategy_id = request.args.get('strategy_id', 'all')

 chart_data = self._get_chart_data(chart_type, timeframe, strategy_id)
 return jsonify(chart_data)

 @self.app.route('/api/alerts')
 def api_alerts():
 """Get recent alerts"""
 alerts = self.alert_manager.get_recent_alerts()
 alerts_data = [asdict(alert) for alert in alerts]

 # Convert datetime to string for JSON serialization
 for alert in alerts_data:
 alert['timestamp'] = alert['timestamp'].isoformat()

 return jsonify(alerts_data)

 @self.app.route('/api/positions')
 def api_positions():
 """Get current positions"""
 positions_data = self._get_positions_data()
 return jsonify(positions_data)

 @self.socketio.on('connect')
 def handle_connect():
 """Handle client connection"""
 print(f"Client connected")
 emit('status', {'status': 'connected'})

 @self.socketio.on('disconnect')
 def handle_disconnect():
 """Handle client disconnection"""
 print(f"Client disconnected")

 @self.socketio.on('subscribe_updates')
 def handle_subscribe(data):
 """Handle subscription to real-time updates"""
 print(f"Client subscribed to updates: {data}")

 # Register alert callback for real-time notifications
 self.alert_manager.register_alert_callback(self._emit_alert)

 def _background_update_loop(self):
 """Background loop for data updates"""
 while self.is_running:
 try:
 # Update all strategy data
 for manager_id, strategy_manager in self.strategy_managers.items():
 self._update_strategy_data(manager_id, strategy_manager)

 # Emit real-time updates if SocketIO is available
 if self.socketio:
 self._emit_real_time_updates()

 time.sleep(self.config.update_frequency_ms / 1000.0)

 except Exception as e:
 warnings.warn(f"Background update failed: {e}")

 def _update_strategy_data(self, manager_id: str, strategy_manager: StrategyManager):
 """Update data for a strategy manager"""
 try:
 # Get current state
 current_time = datetime.now()

 # Simulate getting data from strategy manager
 # In practice, this would come from real strategy managers
 total_pnl = 1000.0 # Example data
 portfolio_greeks = Greeks(100, 5, -10, 200, 1, 150, current_time)

 # Update P&L history
 self.pnl_history[manager_id].append((current_time, total_pnl))

 # Update Greeks history
 self.greeks_history[manager_id].append((current_time, portfolio_greeks))

 # Check for alerts
 self._check_alerts(manager_id, total_pnl, portfolio_greeks)

 except Exception as e:
 warnings.warn(f"Failed to update strategy data for {manager_id}: {e}")

 def _check_alerts(self, strategy_id: str, total_pnl: float, greeks: Greeks):
 """Check for alert conditions"""

 # P&L alert
 if total_pnl < self.config.alert_thresholds['pnl_alert_threshold']:
 self.alert_manager.add_alert(
 'warning',
 'pnl',
 'P&L Alert',
 f'Strategy {strategy_id} P&L below threshold: ${total_pnl:,.2f}',
 strategy_id
 )

 # Greeks alerts
 if abs(greeks.delta) > 1000:
 self.alert_manager.add_alert(
 'warning',
 'risk',
 'Delta Risk Alert',
 f'Strategy {strategy_id} delta exposure: {greeks.delta:.2f}',
 strategy_id
 )

 def _emit_real_time_updates(self):
 """Emit real-time updates to connected clients"""
 if not self.socketio:
 return

 try:
 # Emit overview data
 overview_data = self._get_overview_data()
 self.socketio.emit('overview_update', overview_data)

 # Emit chart updates
 for chart_type in ['pnl', 'greeks', 'positions']:
 chart_data = self._get_chart_data(chart_type, '1D', 'all')
 self.socketio.emit(f'{chart_type}_chart_update', chart_data)

 except Exception as e:
 warnings.warn(f"Failed to emit real-time updates: {e}")

 def _emit_alert(self, alert: AlertMessage):
 """Emit alert to connected clients"""
 if self.socketio:
 alert_data = asdict(alert)
 alert_data['timestamp'] = alert.timestamp.isoformat()
 self.socketio.emit('new_alert', alert_data)

 def _get_overview_data(self) -> Dict[str, Any]:
 """Get dashboard overview data"""
 total_pnl = 0.0
 total_positions = 0
 active_strategies = len(self.strategy_managers)

 # Aggregate data from all strategies
 for manager_id in self.strategy_managers:
 if self.pnl_history[manager_id]:
 latest_pnl = self.pnl_history[manager_id][-1][1]
 total_pnl += latest_pnl

 # Get recent alerts summary
 alert_summary = self.alert_manager.get_alert_summary()

 return {
 'timestamp': datetime.now().isoformat(),
 'total_pnl': total_pnl,
 'total_positions': total_positions,
 'active_strategies': active_strategies,
 'alert_summary': alert_summary,
 'system_status': 'Running' if self.is_running else 'Stopped'
 }

 def _get_strategy_data(self, strategy_id: str) -> Dict[str, Any]:
 """Get detailed data for a specific strategy"""

 if strategy_id not in self.strategy_managers:
 return {'error': 'Strategy not found'}

 # Get latest P&L
 latest_pnl = 0.0
 if self.pnl_history[strategy_id]:
 latest_pnl = self.pnl_history[strategy_id][-1][1]

 # Get latest Greeks
 latest_greeks = None
 if self.greeks_history[strategy_id]:
 latest_greeks = self.greeks_history[strategy_id][-1][1]

 return {
 'strategy_id': strategy_id,
 'current_pnl': latest_pnl,
 'current_greeks': {
 'delta': latest_greeks.delta if latest_greeks else 0,
 'gamma': latest_greeks.gamma if latest_greeks else 0,
 'theta': latest_greeks.theta if latest_greeks else 0,
 'vega': latest_greeks.vega if latest_greeks else 0,
 } if latest_greeks else {},
 'trade_count': len(self.trade_history[strategy_id]),
 'status': 'Active'
 }

 def _get_chart_data(self, chart_type: str, timeframe: str, strategy_id: str) -> Dict[str, Any]:
 """Get chart data for dashboard"""

 try:
 if chart_type == 'pnl':
 if strategy_id == 'all':
 # Aggregate P&L from all strategies
 all_pnl_data = []
 for manager_id in self.strategy_managers:
 all_pnl_data.extend(list(self.pnl_history[manager_id]))

 # Sort by timestamp and aggregate
 all_pnl_data.sort(key=lambda x: x[0])
 return self.chart_generator.create_pnl_chart(all_pnl_data, timeframe)
 else:
 pnl_data = list(self.pnl_history[strategy_id])
 return self.chart_generator.create_pnl_chart(pnl_data, timeframe)

 elif chart_type == 'greeks':
 if strategy_id == 'all':
 # Use first strategy for now (could aggregate)
 first_strategy = next(iter(self.strategy_managers.keys()), None)
 if first_strategy:
 greeks_data = list(self.greeks_history[first_strategy])
 return self.chart_generator.create_greeks_chart(greeks_data)
 else:
 greeks_data = list(self.greeks_history[strategy_id])
 return self.chart_generator.create_greeks_chart(greeks_data)

 elif chart_type == 'positions':
 # Mock positions data
 positions = {
 'AAPL_231215C150': Position('AAPL_231215C150', 100, 5.0, 50000, 1000, 500),
 'AAPL_231215P140': Position('AAPL_231215P140', -50, 3.0, -15000, -500, 200)
 }
 return self.chart_generator.create_position_breakdown_chart(positions)

 elif chart_type == 'risk_gauge':
 current_var = 25000 # Mock data
 var_limit = 50000
 return self.chart_generator.create_risk_gauge(current_var, var_limit)

 else:
 return self.chart_generator._empty_chart("Unknown Chart", f"Chart type '{chart_type}' not recognized")

 except Exception as e:
 return self.chart_generator._empty_chart("Error", f"Failed to generate chart: {str(e)}")

 def _get_positions_data(self) -> Dict[str, Any]:
 """Get current positions data"""
 # Mock positions data
 positions_data = {
 'positions': [
 {
 'symbol': 'AAPL_231215C150',
 'quantity': 100,
 'avg_price': 5.00,
 'market_value': 50000,
 'unrealized_pnl': 1000,
 'delta': 60,
 'gamma': 5,
 'theta': -10,
 'vega': 30
 },
 {
 'symbol': 'AAPL_231215P140',
 'quantity': -50,
 'avg_price': 3.00,
 'market_value': -15000,
 'unrealized_pnl': -500,
 'delta': -20,
 'gamma': 5,
 'theta': -8,
 'vega': 25
 }
 ],
 'total_market_value': 35000,
 'total_unrealized_pnl': 500,
 'net_delta': 40,
 'net_gamma': 10,
 'net_theta': -18,
 'net_vega': 55
 }

 return positions_data

 def _update_performance_data(self, manager_id: str, data: Any):
 """Update performance data for a manager"""
 # This would be called when performance data is updated
 pass


# HTML Template for Dashboard
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Options Market Making Dashboard</title>
 <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
 <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
 <style>
 body {
 background-color: #1e1e1e;
 color: #ffffff;
 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
 margin: 0;
 padding: 20px;
 }.dashboard-header {
 text-align: center;
 margin-bottom: 30px;
 }.dashboard-grid {
 display: grid;
 grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
 gap: 20px;
 }.dashboard-card {
 background-color: #2d2d2d;
 border-radius: 8px;
 padding: 20px;
 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
 }.metric-card {
 background-color: #2d2d2d;
 border-radius: 8px;
 padding: 15px;
 text-align: center;
 margin: 10px;
 }.metric-value {
 font-size: 2em;
 font-weight: bold;
 margin: 10px 0;
 }.metric-label {
 font-size: 0.9em;
 color: #cccccc;
 }.positive { color: #00ff00; }.negative { color: #ff0000; }.neutral { color: #0080ff; }.alerts-container {
 max-height: 300px;
 overflow-y: auto;
 }.alert-item {
 background-color: #404040;
 border-left: 4px solid #ff6b6b;
 padding: 10px;
 margin: 5px 0;
 border-radius: 4px;
 }.alert-warning { border-left-color: #feca57; }.alert-error { border-left-color: #ff6b6b; }.alert-info { border-left-color: #48dbfb; }.positions-table {
 width: 100%;
 border-collapse: collapse;
 }.positions-table th,.positions-table td {
 border: 1px solid #404040;
 padding: 8px;
 text-align: right;
 }.positions-table th {
 background-color: #404040;
 }.timeframe-selector {
 margin: 10px 0;
 }.timeframe-btn {
 background-color: #404040;
 color: white;
 border: none;
 padding: 5px 10px;
 margin: 2px;
 cursor: pointer;
 border-radius: 4px;
 }.timeframe-btn.active {
 background-color: #0080ff;
 }.status-indicator {
 display: inline-block;
 width: 12px;
 height: 12px;
 border-radius: 50%;
 margin-right: 8px;
 }.status-running { background-color: #00ff00; }.status-stopped { background-color: #ff0000; }
 </style>
</head>
<body>
 <div class="dashboard-header">
 <h1>Options Market Making Dashboard</h1>
 <div id="connection-status">
 <span class="status-indicator status-running"></span>
 <span id="status-text">Connected</span>
 </div>
 </div>

 <div class="dashboard-grid">
 <div class="metric-card">
 <div class="metric-label">Total P&L</div>
 <div class="metric-value" id="total-pnl">$0.00</div>
 </div>

 <div class="metric-card">
 <div class="metric-label">Active Strategies</div>
 <div class="metric-value" id="active-strategies">0</div>
 </div>

 <div class="metric-card">
 <div class="metric-label">Open Positions</div>
 <div class="metric-value" id="open-positions">0</div>
 </div>

 <div class="metric-card">
 <div class="metric-label">Net Delta</div>
 <div class="metric-value" id="net-delta">0</div>
 </div>
 </div>

 <div class="dashboard-grid">
 <div class="dashboard-card">
 <h3>P&L Chart</h3>
 <div class="timeframe-selector">
 <button class="timeframe-btn active" onclick="updateChart('pnl', '1H')">1H</button>
 <button class="timeframe-btn" onclick="updateChart('pnl', '4H')">4H</button>
 <button class="timeframe-btn" onclick="updateChart('pnl', '1D')">1D</button>
 <button class="timeframe-btn" onclick="updateChart('pnl', '1W')">1W</button>
 </div>
 <div id="pnl-chart"></div>
 </div>

 <div class="dashboard-card">
 <h3>Portfolio Greeks</h3>
 <div id="greeks-chart"></div>
 </div>

 <div class="dashboard-card">
 <h3>Position Breakdown</h3>
 <div id="positions-chart"></div>
 </div>

 <div class="dashboard-card">
 <h3>Risk Monitor</h3>
 <div id="risk-gauge"></div>
 </div>
 </div>

 <div class="dashboard-grid">
 <div class="dashboard-card">
 <h3>Recent Alerts</h3>
 <div class="alerts-container" id="alerts-container">
 <!-- Alerts will be populated here -->
 </div>
 </div>

 <div class="dashboard-card">
 <h3>Current Positions</h3>
 <div id="positions-table">
 <!-- Positions table will be populated here -->
 </div>
 </div>
 </div>

 <script>
 // WebSocket connection
 const socket = io();

 socket.on('connect', function() {
 console.log('Connected to dashboard');
 document.getElementById('status-text').textContent = 'Connected';
 document.querySelector('.status-indicator').className = 'status-indicator status-running';
 });

 socket.on('disconnect', function() {
 console.log('Disconnected from dashboard');
 document.getElementById('status-text').textContent = 'Disconnected';
 document.querySelector('.status-indicator').className = 'status-indicator status-stopped';
 });

 // Real-time updates
 socket.on('overview_update', function(data) {
 updateOverviewMetrics(data);
 });

 socket.on('pnl_chart_update', function(data) {
 Plotly.react('pnl-chart', data.data, data.layout);
 });

 socket.on('greeks_chart_update', function(data) {
 Plotly.react('greeks-chart', data.data, data.layout);
 });

 socket.on('positions_chart_update', function(data) {
 Plotly.react('positions-chart', data.data, data.layout);
 });

 socket.on('new_alert', function(alert) {
 addAlert(alert);
 });

 // Update functions
 function updateOverviewMetrics(data) {
 document.getElementById('total-pnl').textContent = `$${data.total_pnl.toLocaleString()}`;
 document.getElementById('active-strategies').textContent = data.active_strategies;
 document.getElementById('open-positions').textContent = data.total_positions;

 // Update P&L color
 const pnlElement = document.getElementById('total-pnl');
 pnlElement.className = data.total_pnl >= 0 ? 'metric-value positive' : 'metric-value negative';
 }

 function updateChart(chartType, timeframe) {
 // Update button states
 document.querySelectorAll('.timeframe-btn').forEach(btn => btn.classList.remove('active'));
 event.target.classList.add('active');

 // Fetch new chart data
 fetch(`/api/charts/${chartType}?timeframe=${timeframe}&strategy_id=all`).then(response => response.json()).then(data => {
 Plotly.react(`${chartType}-chart`, data.data, data.layout);
 });
 }

 function addAlert(alert) {
 const alertsContainer = document.getElementById('alerts-container');
 const alertElement = document.createElement('div');
 alertElement.className = `alert-item alert-${alert.severity}`;
 alertElement.innerHTML = `
 <strong>${alert.title}</strong><br>
 <small>${new Date(alert.timestamp).toLocaleString()}</small><br>
 ${alert.message}
 `;
 alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);

 // Limit to 10 alerts
 while (alertsContainer.children.length > 10) {
 alertsContainer.removeChild(alertsContainer.lastChild);
 }
 }

 // Initial data load
 function loadInitialData() {
 // Load overview
 fetch('/api/overview').then(response => response.json()).then(data => updateOverviewMetrics(data));

 // Load charts
 fetch('/api/charts/pnl?timeframe=1D&strategy_id=all').then(response => response.json()).then(data => Plotly.newPlot('pnl-chart', data.data, data.layout));

 fetch('/api/charts/greeks?timeframe=1D&strategy_id=all').then(response => response.json()).then(data => Plotly.newPlot('greeks-chart', data.data, data.layout));

 fetch('/api/charts/positions?strategy_id=all').then(response => response.json()).then(data => Plotly.newPlot('positions-chart', data.data, data.layout));

 fetch('/api/charts/risk_gauge').then(response => response.json()).then(data => Plotly.newPlot('risk-gauge', data.data, data.layout));

 // Load alerts
 fetch('/api/alerts').then(response => response.json()).then(alerts => {
 alerts.forEach(alert => addAlert(alert));
 });

 // Load positions
 fetch('/api/positions').then(response => response.json()).then(data => updatePositionsTable(data));
 }

 function updatePositionsTable(data) {
 const positionsTable = document.getElementById('positions-table');
 let tableHTML = `
 <table class="positions-table">
 <thead>
 <tr>
 <th>Symbol</th>
 <th>Quantity</th>
 <th>Avg Price</th>
 <th>Market Value</th>
 <th>P&L</th>
 <th>Delta</th>
 <th>Gamma</th>
 <th>Theta</th>
 <th>Vega</th>
 </tr>
 </thead>
 <tbody>
 `;

 data.positions.forEach(pos => {
 const pnlClass = pos.unrealized_pnl >= 0 ? 'positive' : 'negative';
 tableHTML += `
 <tr>
 <td>${pos.symbol}</td>
 <td>${pos.quantity}</td>
 <td>$${pos.avg_price.toFixed(2)}</td>
 <td>$${pos.market_value.toLocaleString()}</td>
 <td class="${pnlClass}">$${pos.unrealized_pnl.toLocaleString()}</td>
 <td>${pos.delta}</td>
 <td>${pos.gamma}</td>
 <td>${pos.theta}</td>
 <td>${pos.vega}</td>
 </tr>
 `;
 });

 tableHTML += `
 </tbody>
 </table>
 `;

 positionsTable.innerHTML = tableHTML;

 // Update net Greeks
 document.getElementById('net-delta').textContent = data.net_delta;
 }

 // Load initial data when page loads
 document.addEventListener('DOMContentLoaded', loadInitialData);

 // Refresh data every 5 seconds
 setInterval(loadInitialData, 5000);
 </script>
</body>
</html>
"""

def create_dashboard_templates():
 """Create templates directory and dashboard.html"""
 import os

 templates_dir = "templates"
 if not os.path.exists(templates_dir):
 os.makedirs(templates_dir)

 with open(os.path.join(templates_dir, "dashboard.html"), "w") as f:
 f.write(DASHBOARD_HTML_TEMPLATE)


# Example usage and factory functions
def create_monitoring_dashboard(config: Optional[DashboardConfig] = None) -> MonitoringDashboard:
 """Create a monitoring dashboard with configuration"""
 if not FLASK_AVAILABLE:
 warnings.warn("Flask not available. Dashboard will have limited functionality.")

 return MonitoringDashboard(config)


# Example usage
if __name__ == "__main__":
 # Create dashboard configuration
 config = DashboardConfig(
 host="localhost",
 port=5000,
 debug=True,
 update_frequency_ms=1000
 )

 # Create and start dashboard
 dashboard = create_monitoring_dashboard(config)

 # Create templates
 create_dashboard_templates()

 # Add mock strategy manager
 from market_making_strategies import StrategyManager
 strategy_manager = StrategyManager()
 dashboard.add_strategy_manager("demo_strategy", strategy_manager)

 # Add some mock alerts
 dashboard.alert_manager.add_alert(
 'info', 'system', 'System Started',
 'Market making dashboard started successfully'
 )

 dashboard.alert_manager.add_alert(
 'warning', 'risk', 'Delta Exposure',
 'Portfolio delta exposure approaching limits'
 )

 print("Starting Options Market Making Dashboard...")
 print("Navigate to http://localhost:5000 to view the dashboard")

 # Start monitoring (this will block)
 dashboard.start_monitoring()