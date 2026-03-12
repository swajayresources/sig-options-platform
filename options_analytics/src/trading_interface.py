"""
Professional Trading Interface with Real-time Updates
Advanced web-based trading interface for options analytics platform
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import socketio
from jinja2 import Environment, FileSystemLoader
import weakref

from analytics_framework import OptionsAnalyticsFramework, MarketData, Position
from portfolio_monitor import PortfolioMonitor
from market_analysis import MarketAnalysisEngine
from flow_analysis import FlowAnalysisEngine
from visualization_components import VisualizationEngine
from alert_system import AlertManager
from performance_analytics import PerformanceAnalyzer

@dataclass
class ClientSession:
 session_id: str
 user_id: str
 connected_at: datetime
 last_activity: datetime
 subscriptions: List[str]
 permissions: List[str]

@dataclass
class DashboardConfig:
 layout: str
 widgets: List[Dict[str, Any]]
 refresh_interval: int
 auto_save: bool
 theme: str

class TradingInterface:
 """Professional trading interface with real-time updates"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 # Core components
 self.analytics_framework = OptionsAnalyticsFramework(config)
 self.portfolio_monitor = PortfolioMonitor(config)
 self.market_analysis = MarketAnalysisEngine(config)
 self.flow_analysis = FlowAnalysisEngine(config)
 self.visualization_engine = VisualizationEngine(config)
 self.alert_manager = AlertManager(config)
 self.performance_analyzer = PerformanceAnalyzer(config)

 # Web server components
 self.app = web.Application()
 self.sio = socketio.AsyncServer(
 cors_allowed_origins="*",
 async_mode='aiohttp',
 ping_timeout=60,
 ping_interval=25
 )
 self.sio.attach(self.app)

 # Session management
 self.client_sessions: Dict[str, ClientSession] = {}
 self.active_connections: weakref.WeakSet = weakref.WeakSet()

 # Configuration
 self.host = config.get('host', '0.0.0.0')
 self.port = config.get('port', 8080)
 self.static_dir = config.get('static_dir', 'static')
 self.template_dir = config.get('template_dir', 'templates')

 # Real-time data broadcasting
 self.broadcast_interval = config.get('broadcast_interval', 1000) # milliseconds
 self.data_cache = {}

 # Setup routes and event handlers
 self._setup_routes()
 self._setup_socketio_handlers()

 async def initialize(self):
 """Initialize trading interface"""
 self.logger.info("Initializing Trading Interface")

 # Initialize all components
 await asyncio.gather(
 self.analytics_framework.initialize(),
 self.portfolio_monitor.initialize(),
 self.market_analysis.initialize(),
 self.flow_analysis.initialize(),
 self.visualization_engine.initialize(),
 self.alert_manager.initialize(),
 self.performance_analyzer.initialize()
 )

 # Start background tasks
 asyncio.create_task(self._data_broadcast_loop())
 asyncio.create_task(self._session_cleanup_loop())

 self.logger.info(f"Trading interface ready on {self.host}:{self.port}")

 async def start_server(self):
 """Start the web server"""
 runner = web.AppRunner(self.app)
 await runner.setup()

 site = web.TCPSite(runner, self.host, self.port)
 await site.start()

 self.logger.info(f"Trading interface server started on {self.host}:{self.port}")

 def _setup_routes(self):
 """Setup HTTP routes"""
 # Static files
 self.app.router.add_static('/static/', path=self.static_dir, name='static')

 # Main application routes
 self.app.router.add_get('/', self.home_handler)
 self.app.router.add_get('/dashboard', self.dashboard_handler)
 self.app.router.add_get('/portfolio', self.portfolio_handler)
 self.app.router.add_get('/market-analysis', self.market_analysis_handler)
 self.app.router.add_get('/flow-analysis', self.flow_analysis_handler)
 self.app.router.add_get('/alerts', self.alerts_handler)
 self.app.router.add_get('/performance', self.performance_handler)

 # API routes
 self.app.router.add_get('/api/portfolio/summary', self.api_portfolio_summary)
 self.app.router.add_get('/api/portfolio/positions', self.api_portfolio_positions)
 self.app.router.add_get('/api/portfolio/greeks', self.api_portfolio_greeks)
 self.app.router.add_get('/api/market/overview', self.api_market_overview)
 self.app.router.add_get('/api/market/signals', self.api_market_signals)
 self.app.router.add_get('/api/flow/analytics', self.api_flow_analytics)
 self.app.router.add_get('/api/flow/unusual-activity', self.api_unusual_activity)
 self.app.router.add_get('/api/alerts/active', self.api_active_alerts)
 self.app.router.add_get('/api/performance/metrics', self.api_performance_metrics)
 self.app.router.add_get('/api/charts/{chart_type}', self.api_chart_data)

 # WebSocket endpoint (handled by socketio)
 self.app.router.add_get('/ws/', self.websocket_handler)

 # CORS setup
 cors = aiohttp_cors.setup(self.app, defaults={
 "*": aiohttp_cors.ResourceOptions(
 allow_credentials=True,
 expose_headers="*",
 allow_headers="*",
 allow_methods="*"
 )
 })

 # Add CORS to all routes
 for route in list(self.app.router.routes()):
 cors.add(route)

 def _setup_socketio_handlers(self):
 """Setup SocketIO event handlers"""

 @self.sio.event
 async def connect(sid, environ):
 """Handle client connection"""
 self.logger.info(f"Client connected: {sid}")

 # Create session
 session = ClientSession(
 session_id=sid,
 user_id=f"user_{sid[:8]}",
 connected_at=datetime.now(),
 last_activity=datetime.now(),
 subscriptions=[],
 permissions=['read']
 )
 self.client_sessions[sid] = session

 # Send initial data
 await self._send_initial_data(sid)

 @self.sio.event
 async def disconnect(sid):
 """Handle client disconnection"""
 self.logger.info(f"Client disconnected: {sid}")
 if sid in self.client_sessions:
 del self.client_sessions[sid]

 @self.sio.event
 async def subscribe(sid, data):
 """Handle subscription requests"""
 if sid not in self.client_sessions:
 return

 subscription_type = data.get('type')
 if subscription_type:
 self.client_sessions[sid].subscriptions.append(subscription_type)
 self.logger.info(f"Client {sid} subscribed to {subscription_type}")

 # Send initial data for subscription
 await self._send_subscription_data(sid, subscription_type)

 @self.sio.event
 async def unsubscribe(sid, data):
 """Handle unsubscription requests"""
 if sid not in self.client_sessions:
 return

 subscription_type = data.get('type')
 if subscription_type in self.client_sessions[sid].subscriptions:
 self.client_sessions[sid].subscriptions.remove(subscription_type)
 self.logger.info(f"Client {sid} unsubscribed from {subscription_type}")

 @self.sio.event
 async def get_chart_data(sid, data):
 """Handle chart data requests"""
 chart_type = data.get('chart_type')
 chart_config = data.get('config', {})

 try:
 chart_data = await self._generate_chart_data(chart_type, chart_config)
 await self.sio.emit('chart_data', {
 'chart_type': chart_type,
 'data': chart_data
 }, room=sid)
 except Exception as e:
 self.logger.error(f"Error generating chart data: {e}")
 await self.sio.emit('error', {
 'message': f"Failed to generate chart: {str(e)}"
 }, room=sid)

 @self.sio.event
 async def acknowledge_alert(sid, data):
 """Handle alert acknowledgment"""
 alert_id = data.get('alert_id')
 user_id = self.client_sessions.get(sid, {}).user_id

 if alert_id and user_id:
 success = await self.alert_manager.acknowledge_alert(alert_id, user_id)
 await self.sio.emit('alert_acknowledged', {
 'alert_id': alert_id,
 'success': success
 }, room=sid)

 # HTTP Route Handlers
 async def home_handler(self, request):
 """Home page handler"""
 return web.Response(text=self._render_template('index.html'), content_type='text/html')

 async def dashboard_handler(self, request):
 """Dashboard page handler"""
 return web.Response(text=self._render_template('dashboard.html'), content_type='text/html')

 async def portfolio_handler(self, request):
 """Portfolio page handler"""
 return web.Response(text=self._render_template('portfolio.html'), content_type='text/html')

 async def market_analysis_handler(self, request):
 """Market analysis page handler"""
 return web.Response(text=self._render_template('market_analysis.html'), content_type='text/html')

 async def flow_analysis_handler(self, request):
 """Flow analysis page handler"""
 return web.Response(text=self._render_template('flow_analysis.html'), content_type='text/html')

 async def alerts_handler(self, request):
 """Alerts page handler"""
 return web.Response(text=self._render_template('alerts.html'), content_type='text/html')

 async def performance_handler(self, request):
 """Performance page handler"""
 return web.Response(text=self._render_template('performance.html'), content_type='text/html')

 async def websocket_handler(self, request):
 """WebSocket handler (redirects to SocketIO)"""
 return web.Response(status=404, text="Use SocketIO endpoint")

 # API Handlers
 async def api_portfolio_summary(self, request):
 """API: Portfolio summary"""
 try:
 summary = await self.portfolio_monitor.get_portfolio_summary()
 return web.json_response(summary)
 except Exception as e:
 self.logger.error(f"Error getting portfolio summary: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_portfolio_positions(self, request):
 """API: Portfolio positions"""
 try:
 symbol = request.query.get('symbol')
 positions = await self.portfolio_monitor.get_position_details(symbol)
 return web.json_response(positions)
 except Exception as e:
 self.logger.error(f"Error getting portfolio positions: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_portfolio_greeks(self, request):
 """API: Portfolio Greeks"""
 try:
 greeks_data = await self.portfolio_monitor.get_greeks_heatmap()
 return web.json_response({
 'symbols': greeks_data.symbols,
 'strikes': greeks_data.strikes,
 'delta_matrix': greeks_data.delta_matrix.tolist(),
 'gamma_matrix': greeks_data.gamma_matrix.tolist(),
 'theta_matrix': greeks_data.theta_matrix.tolist(),
 'vega_matrix': greeks_data.vega_matrix.tolist()
 })
 except Exception as e:
 self.logger.error(f"Error getting Greeks data: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_market_overview(self, request):
 """API: Market overview"""
 try:
 overview = await self.market_analysis.get_market_overview()
 return web.json_response(overview)
 except Exception as e:
 self.logger.error(f"Error getting market overview: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_market_signals(self, request):
 """API: Trading signals"""
 try:
 min_strength = float(request.query.get('min_strength', 0.0))
 signal_types = request.query.getall('types') if 'types' in request.query else None

 signals = await self.market_analysis.get_active_signals(signal_types, min_strength)

 # Convert signals to serializable format
 signals_data = []
 for signal in signals:
 signals_data.append({
 'signal_id': signal.signal_id,
 'symbol': signal.symbol,
 'signal_type': signal.signal_type,
 'direction': signal.direction,
 'strength': signal.strength,
 'confidence': signal.confidence,
 'timestamp': signal.timestamp.isoformat(),
 'metadata': signal.metadata
 })

 return web.json_response(signals_data)
 except Exception as e:
 self.logger.error(f"Error getting market signals: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_flow_analytics(self, request):
 """API: Flow analytics"""
 try:
 symbol = request.query.get('symbol')
 timeframe = request.query.get('timeframe', '1h')

 analytics = await self.flow_analysis.get_flow_analytics(symbol, timeframe)

 # Convert to serializable format
 analytics_data = {}
 for sym, data in analytics.items():
 analytics_data[sym] = {
 'symbol': data.symbol,
 'timeframe': data.timeframe,
 'total_volume': data.total_volume,
 'call_volume': data.call_volume,
 'put_volume': data.put_volume,
 'put_call_ratio': data.put_call_ratio,
 'dominant_strikes': data.dominant_strikes,
 'flow_direction': data.flow_direction,
 'sentiment_score': data.sentiment_score
 }

 return web.json_response(analytics_data)
 except Exception as e:
 self.logger.error(f"Error getting flow analytics: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_unusual_activity(self, request):
 """API: Unusual activity"""
 try:
 symbol = request.query.get('symbol')
 min_significance = float(request.query.get('min_significance', 0.0))

 activities = await self.flow_analysis.get_unusual_activities(symbol, min_significance)

 # Convert to serializable format
 activities_data = []
 for activity in activities:
 activities_data.append({
 'activity_id': activity.activity_id,
 'symbol': activity.symbol,
 'activity_type': activity.activity_type,
 'significance_score': activity.significance_score,
 'alert_level': activity.alert_level,
 'timestamp': activity.timestamp.isoformat(),
 'market_context': activity.market_context
 })

 return web.json_response(activities_data)
 except Exception as e:
 self.logger.error(f"Error getting unusual activities: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_active_alerts(self, request):
 """API: Active alerts"""
 try:
 alerts = await self.alert_manager.get_active_alerts()

 # Convert to serializable format
 alerts_data = []
 for alert in alerts:
 alerts_data.append({
 'alert_id': alert.alert_id,
 'alert_type': alert.alert_type.value,
 'priority': alert.priority.value,
 'title': alert.title,
 'message': alert.message,
 'timestamp': alert.timestamp.isoformat(),
 'acknowledged': alert.acknowledged,
 'data': alert.data
 })

 return web.json_response(alerts_data)
 except Exception as e:
 self.logger.error(f"Error getting active alerts: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_performance_metrics(self, request):
 """API: Performance metrics"""
 try:
 period = request.query.get('period', '1m')
 metrics = await self.performance_analyzer.calculate_performance_metrics(period)

 # Convert to serializable format
 metrics_data = {
 'total_return': metrics.total_return,
 'annualized_return': metrics.annualized_return,
 'volatility': metrics.volatility,
 'sharpe_ratio': metrics.sharpe_ratio,
 'max_drawdown': metrics.max_drawdown,
 'win_rate': metrics.win_rate,
 'total_trades': metrics.total_trades
 }

 return web.json_response(metrics_data)
 except Exception as e:
 self.logger.error(f"Error getting performance metrics: {e}")
 return web.json_response({'error': str(e)}, status=500)

 async def api_chart_data(self, request):
 """API: Chart data"""
 try:
 chart_type = request.match_info['chart_type']
 config = dict(request.query)

 chart_data = await self._generate_chart_data(chart_type, config)
 return web.json_response(chart_data)
 except Exception as e:
 self.logger.error(f"Error generating chart data: {e}")
 return web.json_response({'error': str(e)}, status=500)

 # Real-time data methods
 async def _send_initial_data(self, sid: str):
 """Send initial data to newly connected client"""
 try:
 # Portfolio summary
 portfolio_summary = await self.portfolio_monitor.get_portfolio_summary()
 await self.sio.emit('portfolio_summary', portfolio_summary, room=sid)

 # Market overview
 market_overview = await self.market_analysis.get_market_overview()
 await self.sio.emit('market_overview', market_overview, room=sid)

 # Active alerts
 alerts = await self.alert_manager.get_active_alerts()
 alerts_data = [self._serialize_alert(alert) for alert in alerts[:10]] # Latest 10
 await self.sio.emit('active_alerts', alerts_data, room=sid)

 except Exception as e:
 self.logger.error(f"Error sending initial data: {e}")

 async def _send_subscription_data(self, sid: str, subscription_type: str):
 """Send data for specific subscription"""
 try:
 if subscription_type == 'portfolio':
 data = await self.portfolio_monitor.get_portfolio_summary()
 await self.sio.emit('portfolio_update', data, room=sid)

 elif subscription_type == 'market':
 data = await self.market_analysis.get_market_overview()
 await self.sio.emit('market_update', data, room=sid)

 elif subscription_type == 'flow':
 data = await self.flow_analysis.get_flow_analytics()
 serialized_data = {k: self._serialize_flow_analytics(v) for k, v in data.items()}
 await self.sio.emit('flow_update', serialized_data, room=sid)

 elif subscription_type == 'alerts':
 alerts = await self.alert_manager.get_active_alerts()
 alerts_data = [self._serialize_alert(alert) for alert in alerts]
 await self.sio.emit('alerts_update', alerts_data, room=sid)

 except Exception as e:
 self.logger.error(f"Error sending subscription data: {e}")

 async def _generate_chart_data(self, chart_type: str, config: Dict[str, Any]):
 """Generate chart data based on type and configuration"""
 if chart_type == 'portfolio_overview':
 positions = await self.portfolio_monitor.get_position_details()
 summary = await self.portfolio_monitor.get_portfolio_summary()
 return {'positions': positions, 'summary': summary}

 elif chart_type == 'volatility_surface':
 # Mock volatility surface data
 return {
 'vol_surfaces': {
 'AAPL': {
 'strikes': [90, 95, 100, 105, 110],
 'expiries': ['2024-01-15', '2024-02-15', '2024-03-15'],
 'implied_vols': [[0.2, 0.22, 0.25, 0.27, 0.3] for _ in range(3)]
 }
 }
 }

 elif chart_type == 'greeks_heatmap':
 greeks_data = await self.portfolio_monitor.get_greeks_heatmap()
 return {
 'symbols': greeks_data.symbols,
 'strikes': greeks_data.strikes,
 'delta_matrix': greeks_data.delta_matrix.tolist(),
 'gamma_matrix': greeks_data.gamma_matrix.tolist()
 }

 else:
 return {}

 async def _data_broadcast_loop(self):
 """Background task to broadcast real-time data"""
 while True:
 try:
 if self.client_sessions:
 # Broadcast portfolio updates
 await self._broadcast_portfolio_updates()

 # Broadcast market updates
 await self._broadcast_market_updates()

 # Broadcast alerts
 await self._broadcast_alert_updates()

 await asyncio.sleep(self.broadcast_interval / 1000.0)

 except Exception as e:
 self.logger.error(f"Error in data broadcast loop: {e}")
 await asyncio.sleep(5)

 async def _broadcast_portfolio_updates(self):
 """Broadcast portfolio updates to subscribed clients"""
 try:
 portfolio_summary = await self.portfolio_monitor.get_portfolio_summary()

 for sid, session in self.client_sessions.items():
 if 'portfolio' in session.subscriptions:
 await self.sio.emit('portfolio_update', portfolio_summary, room=sid)

 except Exception as e:
 self.logger.error(f"Error broadcasting portfolio updates: {e}")

 async def _broadcast_market_updates(self):
 """Broadcast market updates to subscribed clients"""
 try:
 market_overview = await self.market_analysis.get_market_overview()

 for sid, session in self.client_sessions.items():
 if 'market' in session.subscriptions:
 await self.sio.emit('market_update', market_overview, room=sid)

 except Exception as e:
 self.logger.error(f"Error broadcasting market updates: {e}")

 async def _broadcast_alert_updates(self):
 """Broadcast alert updates to subscribed clients"""
 try:
 alerts = await self.alert_manager.get_active_alerts()
 alerts_data = [self._serialize_alert(alert) for alert in alerts]

 for sid, session in self.client_sessions.items():
 if 'alerts' in session.subscriptions:
 await self.sio.emit('alerts_update', alerts_data, room=sid)

 except Exception as e:
 self.logger.error(f"Error broadcasting alert updates: {e}")

 async def _session_cleanup_loop(self):
 """Background task to cleanup inactive sessions"""
 while True:
 try:
 current_time = datetime.now()
 inactive_sessions = []

 for sid, session in self.client_sessions.items():
 if (current_time - session.last_activity).total_seconds() > 3600: # 1 hour timeout
 inactive_sessions.append(sid)

 for sid in inactive_sessions:
 if sid in self.client_sessions:
 del self.client_sessions[sid]

 await asyncio.sleep(300) # Check every 5 minutes

 except Exception as e:
 self.logger.error(f"Error in session cleanup: {e}")
 await asyncio.sleep(60)

 def _serialize_alert(self, alert) -> Dict[str, Any]:
 """Serialize alert for JSON transmission"""
 return {
 'alert_id': alert.alert_id,
 'alert_type': alert.alert_type.value,
 'priority': alert.priority.value,
 'title': alert.title,
 'message': alert.message,
 'timestamp': alert.timestamp.isoformat(),
 'acknowledged': alert.acknowledged,
 'resolved': alert.resolved,
 'data': alert.data
 }

 def _serialize_flow_analytics(self, analytics) -> Dict[str, Any]:
 """Serialize flow analytics for JSON transmission"""
 return {
 'symbol': analytics.symbol,
 'timeframe': analytics.timeframe,
 'total_volume': analytics.total_volume,
 'call_volume': analytics.call_volume,
 'put_volume': analytics.put_volume,
 'put_call_ratio': analytics.put_call_ratio,
 'flow_direction': analytics.flow_direction,
 'sentiment_score': analytics.sentiment_score,
 'unusual_activity_score': analytics.unusual_activity_score
 }

 def _render_template(self, template_name: str, **kwargs) -> str:
 """Render HTML template"""
 # Simple template rendering - in production use proper template engine
 template_content = f"""
 <!DOCTYPE html>
 <html lang="en">
 <head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Options Analytics Platform</title>
 <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
 <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
 <style>
 body {{ background-color: #1a1a1a; color: #ffffff; }}.navbar {{ background-color: #2d2d2d !important; }}.card {{ background-color: #2d2d2d; border: 1px solid #444; }}.btn-primary {{ background-color: #007bff; border-color: #007bff; }}.table-dark {{ background-color: #2d2d2d; }}.alert-success {{ background-color: #155724; border-color: #28a745; color: #d4edda; }}.alert-danger {{ background-color: #721c24; border-color: #dc3545; color: #f8d7da; }}.chart-container {{ height: 400px; margin: 20px 0; }}
 </style>
 </head>
 <body>
 <nav class="navbar navbar-expand-lg navbar-dark">
 <div class="container-fluid">
 <a class="navbar-brand" href="/">Options Analytics</a>
 <div class="navbar-nav">
 <a class="nav-link" href="/dashboard">Dashboard</a>
 <a class="nav-link" href="/portfolio">Portfolio</a>
 <a class="nav-link" href="/market-analysis">Market</a>
 <a class="nav-link" href="/flow-analysis">Flow</a>
 <a class="nav-link" href="/alerts">Alerts</a>
 <a class="nav-link" href="/performance">Performance</a>
 </div>
 </div>
 </nav>

 <div class="container-fluid mt-4">
 <div id="main-content">
 <h1>Options Trading Analytics Platform</h1>
 <p>Professional-grade options analytics with real-time monitoring and advanced signal generation.</p>

 <div class="row">
 <div class="col-md-6">
 <div class="card">
 <div class="card-header">Portfolio Overview</div>
 <div class="card-body" id="portfolio-overview">
 <div class="chart-container" id="portfolio-chart"></div>
 </div>
 </div>
 </div>
 <div class="col-md-6">
 <div class="card">
 <div class="card-header">Market Signals</div>
 <div class="card-body" id="market-signals">
 <div class="chart-container" id="signals-chart"></div>
 </div>
 </div>
 </div>
 </div>

 <div class="row mt-4">
 <div class="col-md-12">
 <div class="card">
 <div class="card-header">Real-time Alerts</div>
 <div class="card-body" id="alerts-panel">
 <div id="alerts-list"></div>
 </div>
 </div>
 </div>
 </div>
 </div>
 </div>

 <script>
 // Initialize Socket.IO connection
 const socket = io();

 socket.on('connect', function() {{
 console.log('Connected to server');
 // Subscribe to real-time updates
 socket.emit('subscribe', {{type: 'portfolio'}});
 socket.emit('subscribe', {{type: 'market'}});
 socket.emit('subscribe', {{type: 'alerts'}});
 }});

 socket.on('portfolio_update', function(data) {{
 updatePortfolioDisplay(data);
 }});

 socket.on('market_update', function(data) {{
 updateMarketDisplay(data);
 }});

 socket.on('alerts_update', function(data) {{
 updateAlertsDisplay(data);
 }});

 function updatePortfolioDisplay(data) {{
 // Update portfolio charts and metrics
 console.log('Portfolio update:', data);
 }}

 function updateMarketDisplay(data) {{
 // Update market analysis displays
 console.log('Market update:', data);
 }}

 function updateAlertsDisplay(alerts) {{
 const alertsList = document.getElementById('alerts-list');
 alertsList.innerHTML = '';

 alerts.forEach(alert => {{
 const alertDiv = document.createElement('div');
 alertDiv.className = `alert alert-${{alert.priority === 'high' ? 'danger' : 'warning'}}`;
 alertDiv.innerHTML = `
 <strong>${{alert.title}}</strong><br>
 ${{alert.message}}<br>
 <small>${{alert.timestamp}}</small>
 <button class="btn btn-sm btn-outline-light float-end" onclick="acknowledgeAlert('${{alert.alert_id}}')">
 Acknowledge
 </button>
 `;
 alertsList.appendChild(alertDiv);
 }});
 }}

 function acknowledgeAlert(alertId) {{
 socket.emit('acknowledge_alert', {{alert_id: alertId}});
 }}

 // Initialize charts
 document.addEventListener('DOMContentLoaded', function() {{
 // Request initial chart data
 socket.emit('get_chart_data', {{
 chart_type: 'portfolio_overview',
 config: {{}}
 }});
 }});

 socket.on('chart_data', function(data) {{
 if (data.chart_type === 'portfolio_overview') {{
 renderPortfolioChart(data.data);
 }}
 }});

 function renderPortfolioChart(data) {{
 // Render portfolio chart using Plotly
 const chartDiv = document.getElementById('portfolio-chart');
 Plotly.newPlot(chartDiv, [{{
 type: 'bar',
 x: ['Total Value', 'P&L', 'Risk'],
 y: [data.summary?.portfolio_value || 0, data.summary?.total_pnl || 0, data.summary?.var_95 || 0],
 marker: {{color: ['#007bff', '#28a745', '#dc3545']}}
 }}], {{
 title: 'Portfolio Metrics',
 paper_bgcolor: '#2d2d2d',
 plot_bgcolor: '#2d2d2d',
 font: {{color: '#ffffff'}}
 }});
 }}
 </script>
 </body>
 </html>
 """
 return template_content

# Main application entry point
async def create_trading_interface(config: Dict[str, Any]) -> TradingInterface:
 """Create and initialize trading interface"""
 interface = TradingInterface(config)
 await interface.initialize()
 return interface

async def main():
 """Main entry point"""
 config = {
 'host': '0.0.0.0',
 'port': 8080,
 'broadcast_interval': 1000,
 'risk_limits': {
 'max_delta': 1000,
 'max_gamma': 500,
 'max_vega': 2000
 },
 'notifications': {
 'email': {
 'enabled': False
 }
 }
 }

 interface = await create_trading_interface(config)
 await interface.start_server()

 # Keep the server running
 try:
 while True:
 await asyncio.sleep(1)
 except KeyboardInterrupt:
 print("Shutting down...")

if __name__ == "__main__":
 asyncio.run(main())