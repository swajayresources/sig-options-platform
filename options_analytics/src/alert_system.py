"""
Alert Management and Notification System
Comprehensive alerting system for options trading analytics
"""

import asyncio
import smtplib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import websockets

from analytics_framework import MarketData, Position, PortfolioGreeks
from market_analysis import TradingSignal, ArbitrageOpportunity
from flow_analysis import UnusualActivity, SentimentIndicator

class AlertPriority(Enum):
 LOW = "low"
 MEDIUM = "medium"
 HIGH = "high"
 CRITICAL = "critical"

class AlertType(Enum):
 RISK_LIMIT = "risk_limit"
 UNUSUAL_ACTIVITY = "unusual_activity"
 TRADING_SIGNAL = "trading_signal"
 ARBITRAGE = "arbitrage"
 PORTFOLIO = "portfolio"
 MARKET_EVENT = "market_event"
 SYSTEM = "system"
 PERFORMANCE = "performance"

class NotificationChannel(Enum):
 EMAIL = "email"
 SMS = "sms"
 SLACK = "slack"
 DISCORD = "discord"
 WEBHOOK = "webhook"
 IN_APP = "in_app"

@dataclass
class Alert:
 alert_id: str
 alert_type: AlertType
 priority: AlertPriority
 title: str
 message: str
 data: Dict[str, Any]
 timestamp: datetime
 expires_at: Optional[datetime] = None
 acknowledged: bool = False
 acknowledged_by: Optional[str] = None
 acknowledged_at: Optional[datetime] = None
 resolved: bool = False
 resolved_at: Optional[datetime] = None
 notification_channels: List[NotificationChannel] = field(default_factory=list)
 tags: List[str] = field(default_factory=list)

@dataclass
class AlertRule:
 rule_id: str
 name: str
 description: str
 alert_type: AlertType
 priority: AlertPriority
 condition: str
 parameters: Dict[str, Any]
 enabled: bool = True
 notification_channels: List[NotificationChannel] = field(default_factory=list)
 cooldown_minutes: int = 5
 last_triggered: Optional[datetime] = None

@dataclass
class NotificationConfig:
 channel: NotificationChannel
 enabled: bool
 config: Dict[str, Any]
 rate_limit: Optional[int] = None
 retry_attempts: int = 3

class AlertManager:
 """Central alert management system"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)

 # Alert storage
 self.active_alerts: Dict[str, Alert] = {}
 self.alert_history: List[Alert] = []
 self.alert_rules: Dict[str, AlertRule] = {}

 # Alert processors
 self.rule_engine = AlertRuleEngine(config)
 self.notification_manager = NotificationManager(config)
 self.alert_aggregator = AlertAggregator(config)

 # Event handlers
 self.alert_handlers: Dict[AlertType, List[Callable]] = {}

 # Configuration
 self.max_active_alerts = config.get('max_active_alerts', 1000)
 self.max_history_alerts = config.get('max_history_alerts', 10000)
 self.cleanup_interval = config.get('cleanup_interval_minutes', 60)

 # Performance tracking
 self.alert_stats = {
 'total_generated': 0,
 'total_acknowledged': 0,
 'total_resolved': 0,
 'avg_resolution_time': 0.0
 }

 async def initialize(self):
 """Initialize alert manager"""
 self.logger.info("Initializing Alert Manager")

 await asyncio.gather(
 self.rule_engine.initialize(),
 self.notification_manager.initialize(),
 self.alert_aggregator.initialize()
 )

 # Load default alert rules
 await self._load_default_rules()

 # Start background tasks
 asyncio.create_task(self._cleanup_expired_alerts())
 asyncio.create_task(self._process_alert_queue())

 async def create_alert(self, alert_type: AlertType, priority: AlertPriority,
 title: str, message: str, data: Dict[str, Any] = None,
 notification_channels: List[NotificationChannel] = None,
 expires_in_minutes: int = None) -> Alert:
 """Create a new alert"""
 alert_id = f"{alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

 expires_at = None
 if expires_in_minutes:
 expires_at = datetime.now() + timedelta(minutes=expires_in_minutes)

 alert = Alert(
 alert_id=alert_id,
 alert_type=alert_type,
 priority=priority,
 title=title,
 message=message,
 data=data or {},
 timestamp=datetime.now(),
 expires_at=expires_at,
 notification_channels=notification_channels or []
 )

 # Store alert
 self.active_alerts[alert_id] = alert
 self.alert_history.append(alert)
 self.alert_stats['total_generated'] += 1

 # Process alert
 await self._process_alert(alert)

 self.logger.info(f"Created alert: {alert_id} - {title}")
 return alert

 async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
 """Acknowledge an alert"""
 if alert_id not in self.active_alerts:
 return False

 alert = self.active_alerts[alert_id]
 alert.acknowledged = True
 alert.acknowledged_by = acknowledged_by
 alert.acknowledged_at = datetime.now()

 self.alert_stats['total_acknowledged'] += 1

 # Trigger acknowledgment handlers
 await self._trigger_handlers(AlertType.SYSTEM, alert, 'acknowledged')

 self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
 return True

 async def resolve_alert(self, alert_id: str) -> bool:
 """Resolve an alert"""
 if alert_id not in self.active_alerts:
 return False

 alert = self.active_alerts[alert_id]
 alert.resolved = True
 alert.resolved_at = datetime.now()

 # Move to history and remove from active
 del self.active_alerts[alert_id]
 self.alert_stats['total_resolved'] += 1

 # Update average resolution time
 resolution_time = (alert.resolved_at - alert.timestamp).total_seconds() / 60
 self._update_avg_resolution_time(resolution_time)

 # Trigger resolution handlers
 await self._trigger_handlers(AlertType.SYSTEM, alert, 'resolved')

 self.logger.info(f"Alert resolved: {alert_id}")
 return True

 async def get_active_alerts(self, alert_type: AlertType = None,
 priority: AlertPriority = None) -> List[Alert]:
 """Get active alerts with optional filtering"""
 alerts = list(self.active_alerts.values())

 if alert_type:
 alerts = [a for a in alerts if a.alert_type == alert_type]

 if priority:
 alerts = [a for a in alerts if a.priority == priority]

 return sorted(alerts, key=lambda x: (x.priority.value, x.timestamp), reverse=True)

 async def get_alert_history(self, hours: int = 24) -> List[Alert]:
 """Get alert history for specified time period"""
 cutoff_time = datetime.now() - timedelta(hours=hours)
 return [a for a in self.alert_history if a.timestamp >= cutoff_time]

 async def add_alert_rule(self, rule: AlertRule):
 """Add a new alert rule"""
 self.alert_rules[rule.rule_id] = rule
 await self.rule_engine.add_rule(rule)
 self.logger.info(f"Added alert rule: {rule.rule_id}")

 async def remove_alert_rule(self, rule_id: str) -> bool:
 """Remove an alert rule"""
 if rule_id not in self.alert_rules:
 return False

 del self.alert_rules[rule_id]
 await self.rule_engine.remove_rule(rule_id)
 self.logger.info(f"Removed alert rule: {rule_id}")
 return True

 async def process_portfolio_data(self, positions: Dict[str, Position],
 portfolio_greeks: PortfolioGreeks):
 """Process portfolio data for alerts"""
 await self.rule_engine.evaluate_portfolio_rules(positions, portfolio_greeks)

 async def process_market_data(self, market_data: Dict[str, MarketData]):
 """Process market data for alerts"""
 await self.rule_engine.evaluate_market_rules(market_data)

 async def process_trading_signals(self, signals: List[TradingSignal]):
 """Process trading signals for alerts"""
 await self.rule_engine.evaluate_signal_rules(signals)

 async def process_unusual_activity(self, activities: List[UnusualActivity]):
 """Process unusual activities for alerts"""
 await self.rule_engine.evaluate_unusual_activity_rules(activities)

 async def add_alert_handler(self, alert_type: AlertType, handler: Callable):
 """Add alert event handler"""
 if alert_type not in self.alert_handlers:
 self.alert_handlers[alert_type] = []
 self.alert_handlers[alert_type].append(handler)

 async def get_alert_statistics(self) -> Dict[str, Any]:
 """Get alert system statistics"""
 active_by_priority = {}
 for priority in AlertPriority:
 active_by_priority[priority.value] = len([
 a for a in self.active_alerts.values() if a.priority == priority
 ])

 active_by_type = {}
 for alert_type in AlertType:
 active_by_type[alert_type.value] = len([
 a for a in self.active_alerts.values() if a.alert_type == alert_type
 ])

 return {
 'active_alerts_count': len(self.active_alerts),
 'total_rules': len(self.alert_rules),
 'active_by_priority': active_by_priority,
 'active_by_type': active_by_type,
 'alert_stats': self.alert_stats.copy(),
 'notification_channels': await self.notification_manager.get_channel_stats()
 }

 async def _process_alert(self, alert: Alert):
 """Process a newly created alert"""
 # Apply aggregation logic
 aggregated_alert = await self.alert_aggregator.process_alert(alert)

 if aggregated_alert != alert:
 # Alert was aggregated, update storage
 if alert.alert_id in self.active_alerts:
 del self.active_alerts[alert.alert_id]
 self.active_alerts[aggregated_alert.alert_id] = aggregated_alert

 # Send notifications
 await self.notification_manager.send_notifications(aggregated_alert)

 # Trigger handlers
 await self._trigger_handlers(alert.alert_type, aggregated_alert, 'created')

 async def _trigger_handlers(self, alert_type: AlertType, alert: Alert, event: str):
 """Trigger alert event handlers"""
 if alert_type in self.alert_handlers:
 for handler in self.alert_handlers[alert_type]:
 try:
 await handler(alert, event)
 except Exception as e:
 self.logger.error(f"Error in alert handler: {e}")

 async def _load_default_rules(self):
 """Load default alert rules"""
 default_rules = [
 AlertRule(
 rule_id="portfolio_delta_limit",
 name="Portfolio Delta Limit",
 description="Alert when portfolio delta exceeds limits",
 alert_type=AlertType.RISK_LIMIT,
 priority=AlertPriority.HIGH,
 condition="abs(portfolio_greeks.total_delta) > parameters.max_delta",
 parameters={"max_delta": 1000},
 notification_channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP]
 ),
 AlertRule(
 rule_id="portfolio_gamma_limit",
 name="Portfolio Gamma Limit",
 description="Alert when portfolio gamma exceeds limits",
 alert_type=AlertType.RISK_LIMIT,
 priority=AlertPriority.HIGH,
 condition="abs(portfolio_greeks.total_gamma) > parameters.max_gamma",
 parameters={"max_gamma": 500},
 notification_channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP]
 ),
 AlertRule(
 rule_id="high_volatility_signal",
 name="High Volatility Trading Signal",
 description="Alert on high confidence volatility signals",
 alert_type=AlertType.TRADING_SIGNAL,
 priority=AlertPriority.MEDIUM,
 condition="signal.strength > 0.8 and signal.confidence > 0.7",
 parameters={},
 notification_channels=[NotificationChannel.IN_APP]
 ),
 AlertRule(
 rule_id="large_unusual_activity",
 name="Large Unusual Activity",
 description="Alert on significant unusual options activity",
 alert_type=AlertType.UNUSUAL_ACTIVITY,
 priority=AlertPriority.HIGH,
 condition="activity.significance_score > 2.0",
 parameters={},
 notification_channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP]
 ),
 AlertRule(
 rule_id="arbitrage_opportunity",
 name="Arbitrage Opportunity",
 description="Alert on arbitrage opportunities",
 alert_type=AlertType.ARBITRAGE,
 priority=AlertPriority.MEDIUM,
 condition="opportunity.expected_profit > 1000 and opportunity.profit_probability > 0.7",
 parameters={},
 notification_channels=[NotificationChannel.IN_APP]
 )
 ]

 for rule in default_rules:
 await self.add_alert_rule(rule)

 async def _cleanup_expired_alerts(self):
 """Background task to cleanup expired alerts"""
 while True:
 try:
 now = datetime.now()
 expired_alerts = []

 for alert_id, alert in self.active_alerts.items():
 if alert.expires_at and now > alert.expires_at:
 expired_alerts.append(alert_id)

 for alert_id in expired_alerts:
 await self.resolve_alert(alert_id)

 # Cleanup old history
 if len(self.alert_history) > self.max_history_alerts:
 self.alert_history = self.alert_history[-self.max_history_alerts:]

 await asyncio.sleep(self.cleanup_interval * 60)

 except Exception as e:
 self.logger.error(f"Error in alert cleanup: {e}")
 await asyncio.sleep(60)

 async def _process_alert_queue(self):
 """Background task to process alert queue"""
 while True:
 try:
 # Process any queued operations
 await asyncio.sleep(1)
 except Exception as e:
 self.logger.error(f"Error in alert queue processing: {e}")
 await asyncio.sleep(5)

 def _update_avg_resolution_time(self, resolution_time_minutes: float):
 """Update average resolution time"""
 current_avg = self.alert_stats['avg_resolution_time']
 total_resolved = self.alert_stats['total_resolved']

 # Weighted average
 self.alert_stats['avg_resolution_time'] = (
 (current_avg * (total_resolved - 1) + resolution_time_minutes) / total_resolved
 )

class AlertRuleEngine:
 """Alert rule evaluation engine"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)
 self.rules: Dict[str, AlertRule] = {}

 async def initialize(self):
 """Initialize rule engine"""
 self.logger.info("Initializing Alert Rule Engine")

 async def add_rule(self, rule: AlertRule):
 """Add alert rule"""
 self.rules[rule.rule_id] = rule

 async def remove_rule(self, rule_id: str):
 """Remove alert rule"""
 if rule_id in self.rules:
 del self.rules[rule_id]

 async def evaluate_portfolio_rules(self, positions: Dict[str, Position],
 portfolio_greeks: PortfolioGreeks):
 """Evaluate portfolio-related alert rules"""
 for rule in self.rules.values():
 if not rule.enabled or rule.alert_type != AlertType.RISK_LIMIT:
 continue

 if await self._is_rule_in_cooldown(rule):
 continue

 try:
 if await self._evaluate_portfolio_condition(rule, positions, portfolio_greeks):
 await self._trigger_rule(rule, {
 'positions': len(positions),
 'portfolio_greeks': {
 'delta': portfolio_greeks.total_delta,
 'gamma': portfolio_greeks.total_gamma,
 'theta': portfolio_greeks.total_theta,
 'vega': portfolio_greeks.total_vega
 }
 })
 except Exception as e:
 self.logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

 async def evaluate_market_rules(self, market_data: Dict[str, MarketData]):
 """Evaluate market-related alert rules"""
 for rule in self.rules.values():
 if not rule.enabled or rule.alert_type != AlertType.MARKET_EVENT:
 continue

 if await self._is_rule_in_cooldown(rule):
 continue

 try:
 if await self._evaluate_market_condition(rule, market_data):
 await self._trigger_rule(rule, {'market_data_count': len(market_data)})
 except Exception as e:
 self.logger.error(f"Error evaluating market rule {rule.rule_id}: {e}")

 async def evaluate_signal_rules(self, signals: List[TradingSignal]):
 """Evaluate trading signal alert rules"""
 for rule in self.rules.values():
 if not rule.enabled or rule.alert_type != AlertType.TRADING_SIGNAL:
 continue

 if await self._is_rule_in_cooldown(rule):
 continue

 for signal in signals:
 try:
 if await self._evaluate_signal_condition(rule, signal):
 await self._trigger_rule(rule, {
 'signal_id': signal.signal_id,
 'symbol': signal.symbol,
 'strength': signal.strength,
 'confidence': signal.confidence
 })
 break # Only trigger once per rule evaluation
 except Exception as e:
 self.logger.error(f"Error evaluating signal rule {rule.rule_id}: {e}")

 async def evaluate_unusual_activity_rules(self, activities: List[UnusualActivity]):
 """Evaluate unusual activity alert rules"""
 for rule in self.rules.values():
 if not rule.enabled or rule.alert_type != AlertType.UNUSUAL_ACTIVITY:
 continue

 if await self._is_rule_in_cooldown(rule):
 continue

 for activity in activities:
 try:
 if await self._evaluate_unusual_activity_condition(rule, activity):
 await self._trigger_rule(rule, {
 'activity_id': activity.activity_id,
 'symbol': activity.symbol,
 'significance_score': activity.significance_score,
 'activity_type': activity.activity_type
 })
 break
 except Exception as e:
 self.logger.error(f"Error evaluating unusual activity rule {rule.rule_id}: {e}")

 async def _evaluate_portfolio_condition(self, rule: AlertRule, positions: Dict[str, Position],
 portfolio_greeks: PortfolioGreeks) -> bool:
 """Evaluate portfolio condition"""
 # Create evaluation context
 context = {
 'positions': positions,
 'portfolio_greeks': portfolio_greeks,
 'parameters': rule.parameters
 }

 # Simple condition evaluation (in production, use safer evaluation)
 try:
 return eval(rule.condition, {"__builtins__": {}}, context)
 except:
 return False

 async def _evaluate_market_condition(self, rule: AlertRule, market_data: Dict[str, MarketData]) -> bool:
 """Evaluate market condition"""
 context = {
 'market_data': market_data,
 'parameters': rule.parameters
 }

 try:
 return eval(rule.condition, {"__builtins__": {}}, context)
 except:
 return False

 async def _evaluate_signal_condition(self, rule: AlertRule, signal: TradingSignal) -> bool:
 """Evaluate signal condition"""
 context = {
 'signal': signal,
 'parameters': rule.parameters
 }

 try:
 return eval(rule.condition, {"__builtins__": {}}, context)
 except:
 return False

 async def _evaluate_unusual_activity_condition(self, rule: AlertRule, activity: UnusualActivity) -> bool:
 """Evaluate unusual activity condition"""
 context = {
 'activity': activity,
 'parameters': rule.parameters
 }

 try:
 return eval(rule.condition, {"__builtins__": {}}, context)
 except:
 return False

 async def _is_rule_in_cooldown(self, rule: AlertRule) -> bool:
 """Check if rule is in cooldown period"""
 if not rule.last_triggered:
 return False

 cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
 return datetime.now() < cooldown_end

 async def _trigger_rule(self, rule: AlertRule, data: Dict[str, Any]):
 """Trigger alert rule"""
 from alert_system import AlertManager

 rule.last_triggered = datetime.now()

 # Create alert through the main alert manager
 # This would be injected in a real implementation
 alert_manager = AlertManager(self.config)

 await alert_manager.create_alert(
 alert_type=rule.alert_type,
 priority=rule.priority,
 title=rule.name,
 message=f"Rule triggered: {rule.description}",
 data=data,
 notification_channels=rule.notification_channels
 )

class NotificationManager:
 """Manages alert notifications across different channels"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)
 self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
 self.notification_stats = {
 'sent': 0,
 'failed': 0,
 'by_channel': {}
 }

 async def initialize(self):
 """Initialize notification manager"""
 self.logger.info("Initializing Notification Manager")
 await self._load_notification_configs()

 async def send_notifications(self, alert: Alert):
 """Send notifications for alert"""
 if not alert.notification_channels:
 return

 for channel in alert.notification_channels:
 if channel not in self.notification_configs:
 continue

 config = self.notification_configs[channel]
 if not config.enabled:
 continue

 try:
 await self._send_notification(channel, alert, config)
 self.notification_stats['sent'] += 1
 self.notification_stats['by_channel'][channel.value] = (
 self.notification_stats['by_channel'].get(channel.value, 0) + 1
 )
 except Exception as e:
 self.logger.error(f"Failed to send {channel.value} notification: {e}")
 self.notification_stats['failed'] += 1

 async def get_channel_stats(self) -> Dict[str, Any]:
 """Get notification channel statistics"""
 return {
 'total_sent': self.notification_stats['sent'],
 'total_failed': self.notification_stats['failed'],
 'by_channel': self.notification_stats['by_channel'].copy(),
 'enabled_channels': [
 channel.value for channel, config in self.notification_configs.items()
 if config.enabled
 ]
 }

 async def _load_notification_configs(self):
 """Load notification configurations"""
 # Email configuration
 if 'email' in self.config.get('notifications', {}):
 email_config = self.config['notifications']['email']
 self.notification_configs[NotificationChannel.EMAIL] = NotificationConfig(
 channel=NotificationChannel.EMAIL,
 enabled=email_config.get('enabled', False),
 config=email_config
 )

 # Slack configuration
 if 'slack' in self.config.get('notifications', {}):
 slack_config = self.config['notifications']['slack']
 self.notification_configs[NotificationChannel.SLACK] = NotificationConfig(
 channel=NotificationChannel.SLACK,
 enabled=slack_config.get('enabled', False),
 config=slack_config
 )

 # In-app configuration (always enabled)
 self.notification_configs[NotificationChannel.IN_APP] = NotificationConfig(
 channel=NotificationChannel.IN_APP,
 enabled=True,
 config={}
 )

 async def _send_notification(self, channel: NotificationChannel, alert: Alert,
 config: NotificationConfig):
 """Send notification through specific channel"""
 if channel == NotificationChannel.EMAIL:
 await self._send_email_notification(alert, config)
 elif channel == NotificationChannel.SLACK:
 await self._send_slack_notification(alert, config)
 elif channel == NotificationChannel.WEBHOOK:
 await self._send_webhook_notification(alert, config)
 elif channel == NotificationChannel.IN_APP:
 await self._send_in_app_notification(alert, config)

 async def _send_email_notification(self, alert: Alert, config: NotificationConfig):
 """Send email notification"""
 smtp_config = config.config

 msg = MimeMultipart()
 msg['From'] = smtp_config['from_email']
 msg['To'] = smtp_config['to_email']
 msg['Subject'] = f"[{alert.priority.value.upper()}] {alert.title}"

 body = f"""
 Alert: {alert.title}
 Priority: {alert.priority.value}
 Type: {alert.alert_type.value}
 Time: {alert.timestamp}

 Message:
 {alert.message}

 Data:
 {json.dumps(alert.data, indent=2)}
 """

 msg.attach(MimeText(body, 'plain'))

 server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
 if smtp_config.get('use_tls', False):
 server.starttls()
 if smtp_config.get('username') and smtp_config.get('password'):
 server.login(smtp_config['username'], smtp_config['password'])

 server.send_message(msg)
 server.quit()

 async def _send_slack_notification(self, alert: Alert, config: NotificationConfig):
 """Send Slack notification"""
 webhook_url = config.config['webhook_url']

 color = {
 AlertPriority.LOW: '#36a64f',
 AlertPriority.MEDIUM: '#ff9800',
 AlertPriority.HIGH: '#ff5722',
 AlertPriority.CRITICAL: '#f44336'
 }.get(alert.priority, '#36a64f')

 payload = {
 "attachments": [
 {
 "color": color,
 "title": alert.title,
 "text": alert.message,
 "fields": [
 {"title": "Priority", "value": alert.priority.value, "short": True},
 {"title": "Type", "value": alert.alert_type.value, "short": True},
 {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
 ]
 }
 ]
 }

 async with aiohttp.ClientSession() as session:
 async with session.post(webhook_url, json=payload) as response:
 if response.status != 200:
 raise Exception(f"Slack notification failed: {response.status}")

 async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig):
 """Send webhook notification"""
 webhook_url = config.config['url']

 payload = {
 'alert_id': alert.alert_id,
 'alert_type': alert.alert_type.value,
 'priority': alert.priority.value,
 'title': alert.title,
 'message': alert.message,
 'timestamp': alert.timestamp.isoformat(),
 'data': alert.data
 }

 async with aiohttp.ClientSession() as session:
 async with session.post(webhook_url, json=payload) as response:
 if response.status not in [200, 201, 202]:
 raise Exception(f"Webhook notification failed: {response.status}")

 async def _send_in_app_notification(self, alert: Alert, config: NotificationConfig):
 """Send in-app notification (stored for UI retrieval)"""
 # In a real implementation, this would store the notification
 # in a database or message queue for the UI to retrieve
 self.logger.info(f"In-app notification: {alert.title}")

class AlertAggregator:
 """Aggregates similar alerts to reduce noise"""

 def __init__(self, config: Dict[str, Any]):
 self.config = config
 self.logger = logging.getLogger(__name__)
 self.aggregation_window = config.get('aggregation_window_minutes', 5)
 self.recent_alerts: List[Alert] = []

 async def initialize(self):
 """Initialize alert aggregator"""
 self.logger.info("Initializing Alert Aggregator")

 async def process_alert(self, alert: Alert) -> Alert:
 """Process alert and potentially aggregate with similar ones"""
 # Check for similar recent alerts
 similar_alerts = await self._find_similar_alerts(alert)

 if similar_alerts:
 # Aggregate alerts
 return await self._aggregate_alerts(alert, similar_alerts)
 else:
 # Store alert for potential future aggregation
 self.recent_alerts.append(alert)
 await self._cleanup_old_alerts()
 return alert

 async def _find_similar_alerts(self, alert: Alert) -> List[Alert]:
 """Find similar alerts within aggregation window"""
 cutoff_time = datetime.now() - timedelta(minutes=self.aggregation_window)

 similar_alerts = []
 for recent_alert in self.recent_alerts:
 if (recent_alert.timestamp >= cutoff_time and
 recent_alert.alert_type == alert.alert_type and
 recent_alert.priority == alert.priority and
 await self._are_alerts_similar(alert, recent_alert)):
 similar_alerts.append(recent_alert)

 return similar_alerts

 async def _are_alerts_similar(self, alert1: Alert, alert2: Alert) -> bool:
 """Check if two alerts are similar enough to aggregate"""
 # Simple similarity check - in practice this would be more sophisticated
 return (alert1.alert_type == alert2.alert_type and
 alert1.priority == alert2.priority and
 abs((alert1.timestamp - alert2.timestamp).total_seconds()) < 300)

 async def _aggregate_alerts(self, new_alert: Alert, similar_alerts: List[Alert]) -> Alert:
 """Aggregate similar alerts into one"""
 all_alerts = similar_alerts + [new_alert]

 # Remove similar alerts from recent list
 for alert in similar_alerts:
 if alert in self.recent_alerts:
 self.recent_alerts.remove(alert)

 # Create aggregated alert
 aggregated_alert = Alert(
 alert_id=f"agg_{new_alert.alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
 alert_type=new_alert.alert_type,
 priority=new_alert.priority,
 title=f"{new_alert.title} (+{len(similar_alerts)} similar)",
 message=f"Aggregated {len(all_alerts)} similar alerts:\n" +
 "\n".join([f"- {a.message}" for a in all_alerts]),
 data={
 'aggregated_count': len(all_alerts),
 'individual_alerts': [a.alert_id for a in all_alerts],
 'first_alert_data': all_alerts[0].data,
 'latest_alert_data': new_alert.data
 },
 timestamp=new_alert.timestamp,
 notification_channels=new_alert.notification_channels
 )

 self.recent_alerts.append(aggregated_alert)
 return aggregated_alert

 async def _cleanup_old_alerts(self):
 """Remove old alerts from recent list"""
 cutoff_time = datetime.now() - timedelta(minutes=self.aggregation_window * 2)
 self.recent_alerts = [a for a in self.recent_alerts if a.timestamp >= cutoff_time]