"""
Portfolio Optimization and Greeks Management System

This module implements sophisticated portfolio optimization algorithms for options
market making, including Greeks-based optimization, risk-adjusted position sizing,
dynamic Greeks management, and portfolio rebalancing algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from enum import Enum
from collections import defaultdict, deque
import scipy.optimize as sco
from scipy.linalg import inv, LinAlgError
import cvxpy as cp
from concurrent.futures import ThreadPoolExecutor

from market_making_strategies import Greeks, Position, MarketData, OptionContract, OrderSide
from hedging_risk_management import RiskLimits, HedgeOrder


@dataclass
class OptimizationObjective:
 """Portfolio optimization objective configuration"""
 maximize_sharpe: bool = True
 minimize_var: bool = False
 maximize_expected_return: bool = False
 minimize_greeks_variance: bool = True
 target_return: Optional[float] = None
 risk_aversion: float = 1.0
 transaction_cost_penalty: float = 0.1


@dataclass
class PortfolioConstraints:
 """Portfolio optimization constraints"""
 max_weight_per_position: float = 0.1 # 10% max per position
 max_sector_concentration: float = 0.3 # 30% max per sector
 min_liquidity_threshold: float = 0.1 # Minimum liquidity score
 max_correlation_exposure: float = 0.5 # Max exposure to correlated positions

 # Greeks constraints
 delta_neutral_tolerance: float = 100.0
 gamma_target_range: Tuple[float, float] = (-200.0, 200.0)
 vega_target_range: Tuple[float, float] = (-1000.0, 1000.0)
 theta_target_range: Tuple[float, float] = (-200.0, 50.0)

 # Risk constraints
 max_portfolio_var: float = 0.02 # 2% daily VaR
 max_drawdown: float = 0.05 # 5% max drawdown
 min_diversification_ratio: float = 0.7


@dataclass
class PortfolioMetrics:
 """Portfolio performance and risk metrics"""
 total_value: float
 expected_return: float
 volatility: float
 sharpe_ratio: float
 var_95: float
 max_drawdown: float
 diversification_ratio: float

 # Greeks metrics
 portfolio_delta: float
 portfolio_gamma: float
 portfolio_vega: float
 portfolio_theta: float
 portfolio_rho: float

 # Risk decomposition
 systematic_risk: float
 idiosyncratic_risk: float
 concentration_risk: float
 liquidity_risk: float

 timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
 """Portfolio optimization result"""
 target_weights: Dict[str, float]
 rebalancing_trades: List[Tuple[str, OrderSide, int, float]]
 expected_improvement: Dict[str, float]
 constraints_satisfied: bool
 optimization_time: float
 objective_value: float
 portfolio_metrics: PortfolioMetrics


class GreeksTargetManager:
 """Manages target Greeks exposures and rebalancing"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'delta_neutral_weight': 0.8, # 80% weight on delta neutrality
 'gamma_target_weight': 0.6, # 60% weight on gamma target
 'vega_target_weight': 0.7, # 70% weight on vega target
 'theta_optimization_weight': 0.5, # 50% weight on theta optimization
 'rebalance_frequency_minutes': 15, # Rebalance every 15 minutes
 'greeks_decay_alpha': 0.95 # Exponential decay for Greeks smoothing
 }

 # Target Greeks (dynamic)
 self.target_greeks = Greeks(0, 0, 0, 0, 0, 0, datetime.now())
 self.greeks_history: deque = deque(maxlen=100)
 self.last_rebalance_time: Optional[datetime] = None

 def update_target_greeks(self, market_conditions: Dict[str, Any],
 current_greeks: Greeks, risk_appetite: float = 1.0):
 """Update target Greeks based on market conditions"""

 # Delta target (usually neutral)
 delta_target = 0.0

 # Gamma target based on volatility environment
 implied_vol_avg = market_conditions.get('avg_implied_vol', 0.2)
 if implied_vol_avg > 0.3: # High vol environment
 gamma_target = -50.0 * risk_appetite # Short gamma in high vol
 elif implied_vol_avg < 0.15: # Low vol environment
 gamma_target = 100.0 * risk_appetite # Long gamma in low vol
 else:
 gamma_target = 0.0

 # Vega target based on vol term structure
 vol_term_structure = market_conditions.get('vol_term_structure', 'flat')
 if vol_term_structure == 'backwardated':
 vega_target = -500.0 * risk_appetite # Short vega in backwardation
 elif vol_term_structure == 'contango':
 vega_target = 500.0 * risk_appetite # Long vega in contango
 else:
 vega_target = 0.0

 # Theta target (usually want positive theta for income)
 theta_target = 50.0 * risk_appetite

 # Rho target (usually small)
 rho_target = 0.0

 # Smooth the transition to new targets
 alpha = self.config['greeks_decay_alpha']

 self.target_greeks = Greeks(
 delta=alpha * self.target_greeks.delta + (1 - alpha) * delta_target,
 gamma=alpha * self.target_greeks.gamma + (1 - alpha) * gamma_target,
 theta=alpha * self.target_greeks.theta + (1 - alpha) * theta_target,
 vega=alpha * self.target_greeks.vega + (1 - alpha) * vega_target,
 rho=alpha * self.target_greeks.rho + (1 - alpha) * rho_target,
 underlying_price=current_greeks.underlying_price,
 timestamp=datetime.now()
 )

 def calculate_greeks_deviation(self, current_greeks: Greeks) -> Dict[str, float]:
 """Calculate deviation from target Greeks"""

 deviations = {
 'delta_deviation': current_greeks.delta - self.target_greeks.delta,
 'gamma_deviation': current_greeks.gamma - self.target_greeks.gamma,
 'vega_deviation': current_greeks.vega - self.target_greeks.vega,
 'theta_deviation': current_greeks.theta - self.target_greeks.theta,
 'rho_deviation': current_greeks.rho - self.target_greeks.rho
 }

 # Calculate weighted total deviation
 weights = {
 'delta_deviation': self.config['delta_neutral_weight'],
 'gamma_deviation': self.config['gamma_target_weight'],
 'vega_deviation': self.config['vega_target_weight'],
 'theta_deviation': self.config['theta_optimization_weight'],
 'rho_deviation': 0.2
 }

 total_deviation = sum(abs(deviations[key]) * weights[key] for key in deviations)
 deviations['total_weighted_deviation'] = total_deviation

 return deviations

 def should_rebalance_greeks(self, current_greeks: Greeks) -> bool:
 """Determine if Greeks rebalancing is needed"""

 # Time-based rebalancing
 if self.last_rebalance_time:
 time_since_rebalance = (datetime.now() - self.last_rebalance_time).total_seconds() / 60
 if time_since_rebalance < self.config['rebalance_frequency_minutes']:
 return False

 # Threshold-based rebalancing
 deviations = self.calculate_greeks_deviation(current_greeks)

 # Define thresholds
 thresholds = {
 'delta_deviation': 100.0,
 'gamma_deviation': 50.0,
 'vega_deviation': 300.0,
 'theta_deviation': 100.0
 }

 # Check if any deviation exceeds threshold
 for key, threshold in thresholds.items():
 if abs(deviations[key]) > threshold:
 return True

 return False

 def generate_greeks_rebalancing_trades(self, current_positions: Dict[str, Position],
 available_instruments: Dict[str, Dict],
 target_notional: float = 100000.0) -> List[HedgeOrder]:
 """Generate trades to rebalance Greeks to targets"""

 deviations = self.calculate_greeks_deviation(self._calculate_portfolio_greeks(current_positions))
 rebalancing_trades = []

 # Delta rebalancing
 if abs(deviations['delta_deviation']) > 50.0:
 delta_trades = self._generate_delta_trades(deviations['delta_deviation'], available_instruments)
 rebalancing_trades.extend(delta_trades)

 # Gamma rebalancing
 if abs(deviations['gamma_deviation']) > 25.0:
 gamma_trades = self._generate_gamma_trades(deviations['gamma_deviation'], available_instruments)
 rebalancing_trades.extend(gamma_trades)

 # Vega rebalancing
 if abs(deviations['vega_deviation']) > 200.0:
 vega_trades = self._generate_vega_trades(deviations['vega_deviation'], available_instruments)
 rebalancing_trades.extend(vega_trades)

 if rebalancing_trades:
 self.last_rebalance_time = datetime.now()

 return rebalancing_trades

 def _calculate_portfolio_greeks(self, positions: Dict[str, Position]) -> Greeks:
 """Calculate current portfolio Greeks"""
 # Simplified calculation - would need actual Greeks per position
 total_delta = sum(pos.greeks.delta * pos.quantity for pos in positions.values() if pos.greeks)
 total_gamma = sum(pos.greeks.gamma * pos.quantity for pos in positions.values() if pos.greeks)
 total_vega = sum(pos.greeks.vega * pos.quantity for pos in positions.values() if pos.greeks)
 total_theta = sum(pos.greeks.theta * pos.quantity for pos in positions.values() if pos.greeks)
 total_rho = sum(pos.greeks.rho * pos.quantity for pos in positions.values() if pos.greeks)

 return Greeks(total_delta, total_gamma, total_theta, total_vega, total_rho, 0, datetime.now())

 def _generate_delta_trades(self, delta_deviation: float, available_instruments: Dict[str, Dict]) -> List[HedgeOrder]:
 """Generate trades to fix delta deviation"""
 # Find underlying instruments for delta hedging
 trades = []
 # Implementation would find best delta hedge instruments
 return trades

 def _generate_gamma_trades(self, gamma_deviation: float, available_instruments: Dict[str, Dict]) -> List[HedgeOrder]:
 """Generate trades to fix gamma deviation"""
 trades = []
 # Implementation would find best gamma hedge instruments
 return trades

 def _generate_vega_trades(self, vega_deviation: float, available_instruments: Dict[str, Dict]) -> List[HedgeOrder]:
 """Generate trades to fix vega deviation"""
 trades = []
 # Implementation would find best vega hedge instruments
 return trades


class RiskBudgetOptimizer:
 """Risk budgeting and allocation optimizer"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'risk_budget_method': 'equal_risk_contribution', # 'equal_risk_contribution', 'target_volatility'
 'volatility_lookback_days': 30,
 'correlation_lookback_days': 60,
 'risk_budget_rebalance_frequency': 'daily',
 'max_concentration_single_position': 0.15,
 'min_risk_contribution': 0.01 # 1% minimum risk contribution
 }

 self.risk_budgets: Dict[str, float] = {}
 self.risk_contributions: Dict[str, float] = {}
 self.covariance_matrix: Optional[np.ndarray] = None
 self.symbols_list: List[str] = []

 def calculate_risk_budgets(self, positions: Dict[str, Position],
 returns_history: Dict[str, List[float]]) -> Dict[str, float]:
 """Calculate optimal risk budgets for portfolio positions"""

 if not returns_history:
 # Equal weight if no history
 n_positions = len(positions)
 return {symbol: 1.0 / n_positions for symbol in positions.keys()}

 # Build covariance matrix
 self.symbols_list = list(positions.keys())
 returns_matrix = self._build_returns_matrix(returns_history, self.symbols_list)

 if returns_matrix.shape[0] < 10: # Need sufficient history
 n_positions = len(positions)
 return {symbol: 1.0 / n_positions for symbol in positions.keys()}

 self.covariance_matrix = np.cov(returns_matrix.T)

 # Calculate risk budgets based on method
 if self.config['risk_budget_method'] == 'equal_risk_contribution':
 risk_budgets = self._equal_risk_contribution_weights(self.covariance_matrix)
 else:
 # Equal weights as fallback
 n_positions = len(positions)
 risk_budgets = np.ones(n_positions) / n_positions

 # Convert to dictionary
 return {symbol: budget for symbol, budget in zip(self.symbols_list, risk_budgets)}

 def calculate_risk_contributions(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
 """Calculate risk contributions for given weights"""

 portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
 portfolio_volatility = np.sqrt(portfolio_variance)

 if portfolio_volatility == 0:
 return np.zeros_like(weights)

 # Risk contribution = weight * (covariance with portfolio) / portfolio_volatility
 marginal_contributions = np.dot(covariance_matrix, weights) / portfolio_volatility
 risk_contributions = weights * marginal_contributions / portfolio_volatility

 return risk_contributions

 def optimize_risk_parity_weights(self, covariance_matrix: np.ndarray,
 target_risk_budgets: Optional[np.ndarray] = None) -> np.ndarray:
 """Optimize portfolio weights for risk parity"""

 n_assets = covariance_matrix.shape[0]

 if target_risk_budgets is None:
 target_risk_budgets = np.ones(n_assets) / n_assets

 def objective_function(weights):
 risk_contributions = self.calculate_risk_contributions(weights, covariance_matrix)
 # Minimize squared deviations from target risk budgets
 return np.sum((risk_contributions - target_risk_budgets) ** 2)

 # Constraints
 constraints = [
 {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, # Weights sum to 1
 ]

 # Bounds (non-negative weights)
 bounds = [(0, self.config['max_concentration_single_position']) for _ in range(n_assets)]

 # Initial guess (equal weights)
 initial_weights = np.ones(n_assets) / n_assets

 # Optimize
 result = sco.minimize(
 objective_function,
 initial_weights,
 method='SLSQP',
 bounds=bounds,
 constraints=constraints,
 options={'maxiter': 1000}
 )

 if result.success:
 return result.x
 else:
 warnings.warn("Risk parity optimization failed, using equal weights")
 return initial_weights

 def _equal_risk_contribution_weights(self, covariance_matrix: np.ndarray) -> np.ndarray:
 """Calculate equal risk contribution weights"""

 n_assets = covariance_matrix.shape[0]
 target_risk_budgets = np.ones(n_assets) / n_assets

 return self.optimize_risk_parity_weights(covariance_matrix, target_risk_budgets)

 def _build_returns_matrix(self, returns_history: Dict[str, List[float]], symbols: List[str]) -> np.ndarray:
 """Build returns matrix from history"""

 min_length = min(len(returns_history[symbol]) for symbol in symbols)
 returns_matrix = np.zeros((min_length, len(symbols)))

 for i, symbol in enumerate(symbols):
 returns_matrix[:, i] = returns_history[symbol][-min_length:]

 return returns_matrix


class MeanVarianceOptimizer:
 """Mean-variance portfolio optimization"""

 def __init__(self, config: Dict[str, Any] = None):
 self.config = config or {
 'return_estimation_method': 'historical', # 'historical', 'factor_model', 'shrinkage'
 'covariance_estimation_method': 'sample', # 'sample', 'shrinkage', 'robust'
 'risk_aversion': 5.0,
 'transaction_cost_bps': 10, # 10 bps transaction cost
 'turnover_penalty': 0.01,
 'lookback_days': 60
 }

 def optimize_portfolio(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
 current_weights: np.ndarray, constraints: PortfolioConstraints) -> np.ndarray:
 """Optimize portfolio using mean-variance optimization"""

 n_assets = len(expected_returns)

 # Decision variable (portfolio weights)
 w = cp.Variable(n_assets)

 # Portfolio return
 portfolio_return = expected_returns.T @ w

 # Portfolio risk
 portfolio_risk = cp.quad_form(w, covariance_matrix)

 # Transaction costs
 turnover = cp.norm(w - current_weights, 1)
 transaction_costs = self.config['transaction_cost_bps'] / 10000.0 * turnover

 # Objective: maximize return - risk_aversion * risk - transaction_costs
 objective = cp.Maximize(
 portfolio_return -
 self.config['risk_aversion'] * portfolio_risk -
 transaction_costs -
 self.config['turnover_penalty'] * turnover
 )

 # Constraints
 constraint_list = [
 cp.sum(w) == 1, # Fully invested
 w >= 0, # Long-only
 w <= constraints.max_weight_per_position # Position size limits
 ]

 # Additional constraints based on portfolio constraints
 if constraints.max_portfolio_var < np.inf:
 constraint_list.append(portfolio_risk <= constraints.max_portfolio_var ** 2)

 # Solve optimization
 problem = cp.Problem(objective, constraint_list)

 try:
 problem.solve(solver=cp.OSQP, verbose=False)

 if problem.status == cp.OPTIMAL:
 return w.value
 else:
 warnings.warn(f"Optimization failed with status: {problem.status}")
 return current_weights

 except Exception as e:
 warnings.warn(f"Optimization error: {e}")
 return current_weights

 def estimate_expected_returns(self, returns_history: Dict[str, List[float]],
 market_data: Dict[str, MarketData]) -> np.ndarray:
 """Estimate expected returns"""

 symbols = list(returns_history.keys())
 expected_returns = np.zeros(len(symbols))

 for i, symbol in enumerate(symbols):
 returns = np.array(returns_history[symbol])

 if self.config['return_estimation_method'] == 'historical':
 expected_returns[i] = np.mean(returns)
 elif self.config['return_estimation_method'] == 'shrinkage':
 # James-Stein shrinkage towards market return
 market_return = 0.08 / 252 # 8% annual return
 sample_mean = np.mean(returns)
 n = len(returns)
 var_estimate = np.var(returns)

 # Shrinkage intensity
 shrinkage = min(1.0, (var_estimate / n) / (sample_mean - market_return) ** 2)
 expected_returns[i] = shrinkage * market_return + (1 - shrinkage) * sample_mean
 else:
 expected_returns[i] = np.mean(returns)

 return expected_returns

 def estimate_covariance_matrix(self, returns_history: Dict[str, List[float]]) -> np.ndarray:
 """Estimate covariance matrix"""

 symbols = list(returns_history.keys())
 min_length = min(len(returns_history[symbol]) for symbol in symbols)

 returns_matrix = np.zeros((min_length, len(symbols)))
 for i, symbol in enumerate(symbols):
 returns_matrix[:, i] = returns_history[symbol][-min_length:]

 if self.config['covariance_estimation_method'] == 'sample':
 return np.cov(returns_matrix.T)
 elif self.config['covariance_estimation_method'] == 'shrinkage':
 return self._shrinkage_covariance(returns_matrix)
 else:
 return np.cov(returns_matrix.T)

 def _shrinkage_covariance(self, returns_matrix: np.ndarray) -> np.ndarray:
 """Ledoit-Wolf shrinkage covariance estimator"""

 n, p = returns_matrix.shape
 sample_cov = np.cov(returns_matrix.T)

 # Target: identity matrix scaled by average variance
 target = np.eye(p) * np.trace(sample_cov) / p

 # Calculate optimal shrinkage intensity
 # Simplified calculation - full implementation would be more complex
 shrinkage_intensity = min(1.0, 0.2) # 20% shrinkage

 return shrinkage_intensity * target + (1 - shrinkage_intensity) * sample_cov


class PortfolioOptimizer:
 """Main portfolio optimization engine"""

 def __init__(self, objective: OptimizationObjective, constraints: PortfolioConstraints):
 self.objective = objective
 self.constraints = constraints

 # Initialize sub-optimizers
 self.greeks_manager = GreeksTargetManager()
 self.risk_budgeter = RiskBudgetOptimizer()
 self.mean_var_optimizer = MeanVarianceOptimizer()

 # State
 self.optimization_history: deque = deque(maxlen=1000)
 self.current_weights: Dict[str, float] = {}

 def optimize_portfolio(self, positions: Dict[str, Position],
 market_data: Dict[str, MarketData],
 returns_history: Dict[str, List[float]],
 greeks_data: Dict[str, Greeks]) -> OptimizationResult:
 """Run comprehensive portfolio optimization"""

 start_time = time.time()

 try:
 # Calculate current portfolio metrics
 current_metrics = self.calculate_portfolio_metrics(positions, market_data, returns_history, greeks_data)

 # Estimate expected returns and covariance
 symbols = list(positions.keys())
 expected_returns = self.mean_var_optimizer.estimate_expected_returns(returns_history, market_data)
 covariance_matrix = self.mean_var_optimizer.estimate_covariance_matrix(returns_history)

 # Current weights
 total_value = sum(abs(pos.market_value) for pos in positions.values())
 current_weights_array = np.array([
 positions[symbol].market_value / total_value if total_value > 0 else 0
 for symbol in symbols
 ])

 # Multi-objective optimization
 optimal_weights = self._multi_objective_optimization(
 expected_returns, covariance_matrix, current_weights_array,
 positions, greeks_data
 )

 # Calculate target weights
 target_weights = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}

 # Generate rebalancing trades
 rebalancing_trades = self._generate_rebalancing_trades(
 positions, target_weights, market_data
 )

 # Calculate expected improvement
 expected_improvement = self._calculate_expected_improvement(
 current_metrics, optimal_weights, expected_returns, covariance_matrix
 )

 # Check constraints satisfaction
 constraints_satisfied = self._check_constraints_satisfaction(
 optimal_weights, greeks_data, positions
 )

 # Calculate optimized portfolio metrics
 optimized_metrics = self._calculate_optimized_metrics(
 optimal_weights, expected_returns, covariance_matrix, greeks_data
 )

 optimization_time = time.time() - start_time

 result = OptimizationResult(
 target_weights=target_weights,
 rebalancing_trades=rebalancing_trades,
 expected_improvement=expected_improvement,
 constraints_satisfied=constraints_satisfied,
 optimization_time=optimization_time,
 objective_value=self._calculate_objective_value(optimal_weights, expected_returns, covariance_matrix),
 portfolio_metrics=optimized_metrics
 )

 # Update state
 self.current_weights = target_weights
 self.optimization_history.append(result)

 return result

 except Exception as e:
 warnings.warn(f"Portfolio optimization failed: {e}")

 # Return current state as fallback
 return OptimizationResult(
 target_weights=self.current_weights,
 rebalancing_trades=[],
 expected_improvement={},
 constraints_satisfied=False,
 optimization_time=time.time() - start_time,
 objective_value=0.0,
 portfolio_metrics=current_metrics
 )

 def calculate_portfolio_metrics(self, positions: Dict[str, Position],
 market_data: Dict[str, MarketData],
 returns_history: Dict[str, List[float]],
 greeks_data: Dict[str, Greeks]) -> PortfolioMetrics:
 """Calculate comprehensive portfolio metrics"""

 # Portfolio value
 total_value = sum(pos.market_value for pos in positions.values())

 # Greeks aggregation
 portfolio_delta = sum(pos.greeks.delta * pos.quantity for pos in positions.values() if pos.greeks)
 portfolio_gamma = sum(pos.greeks.gamma * pos.quantity for pos in positions.values() if pos.greeks)
 portfolio_vega = sum(pos.greeks.vega * pos.quantity for pos in positions.values() if pos.greeks)
 portfolio_theta = sum(pos.greeks.theta * pos.quantity for pos in positions.values() if pos.greeks)
 portfolio_rho = sum(pos.greeks.rho * pos.quantity for pos in positions.values() if pos.greeks)

 # Risk metrics (simplified)
 if returns_history and len(next(iter(returns_history.values()))) > 20:
 portfolio_returns = self._calculate_portfolio_returns(positions, returns_history)
 expected_return = np.mean(portfolio_returns)
 volatility = np.std(portfolio_returns)
 sharpe_ratio = expected_return / volatility if volatility > 0 else 0
 var_95 = np.percentile(portfolio_returns, 5)
 max_drawdown = self._calculate_max_drawdown(portfolio_returns)
 else:
 expected_return = 0.0
 volatility = 0.0
 sharpe_ratio = 0.0
 var_95 = 0.0
 max_drawdown = 0.0

 # Diversification metrics
 weights = np.array([pos.market_value / total_value for pos in positions.values()]) if total_value > 0 else np.array([])
 diversification_ratio = 1.0 / np.sqrt(np.sum(weights ** 2)) if len(weights) > 0 else 1.0

 return PortfolioMetrics(
 total_value=total_value,
 expected_return=expected_return,
 volatility=volatility,
 sharpe_ratio=sharpe_ratio,
 var_95=var_95,
 max_drawdown=max_drawdown,
 diversification_ratio=diversification_ratio,
 portfolio_delta=portfolio_delta,
 portfolio_gamma=portfolio_gamma,
 portfolio_vega=portfolio_vega,
 portfolio_theta=portfolio_theta,
 portfolio_rho=portfolio_rho,
 systematic_risk=0.0, # Would need factor model
 idiosyncratic_risk=0.0, # Would need factor model
 concentration_risk=1.0 - diversification_ratio,
 liquidity_risk=0.0 # Would need liquidity metrics
 )

 def _multi_objective_optimization(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
 current_weights: np.ndarray, positions: Dict[str, Position],
 greeks_data: Dict[str, Greeks]) -> np.ndarray:
 """Multi-objective optimization combining different objectives"""

 # Risk parity weights
 risk_parity_weights = self.risk_budgeter.optimize_risk_parity_weights(covariance_matrix)

 # Mean-variance weights
 mv_weights = self.mean_var_optimizer.optimize_portfolio(
 expected_returns, covariance_matrix, current_weights, self.constraints
 )

 # Greeks-neutral weights
 greeks_weights = self._optimize_greeks_neutral_weights(
 current_weights, positions, greeks_data, covariance_matrix
 )

 # Combine objectives with weights
 if self.objective.maximize_sharpe:
 sharpe_weight = 0.4
 else:
 sharpe_weight = 0.0

 if self.objective.minimize_var:
 risk_weight = 0.3
 else:
 risk_weight = 0.0

 if self.objective.minimize_greeks_variance:
 greeks_weight = 0.3
 else:
 greeks_weight = 0.0

 # Normalize weights
 total_weight = sharpe_weight + risk_weight + greeks_weight
 if total_weight == 0:
 total_weight = 1.0

 sharpe_weight /= total_weight
 risk_weight /= total_weight
 greeks_weight /= total_weight

 # Combine solutions
 combined_weights = (sharpe_weight * mv_weights +
 risk_weight * risk_parity_weights +
 greeks_weight * greeks_weights)

 # Ensure weights sum to 1 and satisfy constraints
 combined_weights = np.maximum(combined_weights, 0) # Non-negative
 combined_weights = np.minimum(combined_weights, self.constraints.max_weight_per_position) # Position limits
 combined_weights = combined_weights / np.sum(combined_weights) # Normalize

 return combined_weights

 def _optimize_greeks_neutral_weights(self, current_weights: np.ndarray,
 positions: Dict[str, Position],
 greeks_data: Dict[str, Greeks],
 covariance_matrix: np.ndarray) -> np.ndarray:
 """Optimize for Greeks neutrality"""

 n_assets = len(current_weights)
 symbols = list(positions.keys())

 # Build Greeks matrix
 greeks_matrix = np.zeros((4, n_assets)) # delta, gamma, vega, theta
 for i, symbol in enumerate(symbols):
 if symbol in greeks_data:
 greeks = greeks_data[symbol]
 greeks_matrix[0, i] = greeks.delta
 greeks_matrix[1, i] = greeks.gamma
 greeks_matrix[2, i] = greeks.vega
 greeks_matrix[3, i] = greeks.theta

 # Decision variable
 w = cp.Variable(n_assets)

 # Greeks constraints (soft constraints with penalties)
 greeks_targets = np.array([0, 0, 0, 50]) # delta=0, gamma=0, vega=0, theta=50
 greeks_deviations = greeks_matrix @ w - greeks_targets

 # Objective: minimize risk + Greeks deviations
 portfolio_risk = cp.quad_form(w, covariance_matrix)
 greeks_penalty = cp.sum_squares(greeks_deviations)

 objective = cp.Minimize(portfolio_risk + 0.1 * greeks_penalty)

 # Constraints
 constraints = [
 cp.sum(w) == 1, # Fully invested
 w >= 0, # Long-only
 w <= self.constraints.max_weight_per_position # Position limits
 ]

 # Solve
 problem = cp.Problem(objective, constraints)

 try:
 problem.solve(solver=cp.OSQP, verbose=False)
 if problem.status == cp.OPTIMAL:
 return w.value
 except:
 pass

 # Fallback to current weights
 return current_weights

 def _generate_rebalancing_trades(self, positions: Dict[str, Position],
 target_weights: Dict[str, float],
 market_data: Dict[str, MarketData]) -> List[Tuple[str, OrderSide, int, float]]:
 """Generate trades to achieve target weights"""

 trades = []
 total_value = sum(abs(pos.market_value) for pos in positions.values())

 for symbol, target_weight in target_weights.items():
 if symbol not in positions:
 continue

 current_position = positions[symbol]
 current_weight = current_position.market_value / total_value if total_value > 0 else 0

 weight_diff = target_weight - current_weight
 value_diff = weight_diff * total_value

 if abs(value_diff) < 100: # Minimum trade size
 continue

 # Estimate shares to trade
 if symbol in market_data:
 current_price = (market_data[symbol].bid + market_data[symbol].ask) / 2
 if current_price > 0:
 shares_to_trade = int(value_diff / current_price)

 if shares_to_trade != 0:
 side = OrderSide.BID if shares_to_trade > 0 else OrderSide.ASK
 trades.append((symbol, side, abs(shares_to_trade), current_price))

 return trades

 def _calculate_expected_improvement(self, current_metrics: PortfolioMetrics,
 optimal_weights: np.ndarray,
 expected_returns: np.ndarray,
 covariance_matrix: np.ndarray) -> Dict[str, float]:
 """Calculate expected improvement from optimization"""

 # Calculate expected metrics for optimal portfolio
 expected_return = np.dot(optimal_weights, expected_returns)
 expected_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
 expected_volatility = np.sqrt(expected_variance)
 expected_sharpe = expected_return / expected_volatility if expected_volatility > 0 else 0

 return {
 'expected_return_improvement': expected_return - current_metrics.expected_return,
 'volatility_reduction': current_metrics.volatility - expected_volatility,
 'sharpe_improvement': expected_sharpe - current_metrics.sharpe_ratio,
 'var_improvement': current_metrics.var_95 - (-expected_volatility * 1.645) # 95% VaR
 }

 def _check_constraints_satisfaction(self, weights: np.ndarray,
 greeks_data: Dict[str, Greeks],
 positions: Dict[str, Position]) -> bool:
 """Check if optimization satisfies all constraints"""

 # Weight constraints
 if np.max(weights) > self.constraints.max_weight_per_position:
 return False

 if not np.allclose(np.sum(weights), 1.0, rtol=1e-3):
 return False

 # Greeks constraints (if we have Greeks data)
 symbols = list(positions.keys())
 portfolio_delta = sum(weights[i] * greeks_data[symbols[i]].delta
 for i in range(len(weights))
 if symbols[i] in greeks_data)

 if abs(portfolio_delta) > self.constraints.delta_neutral_tolerance:
 return False

 return True

 def _calculate_optimized_metrics(self, weights: np.ndarray,
 expected_returns: np.ndarray,
 covariance_matrix: np.ndarray,
 greeks_data: Dict[str, Greeks]) -> PortfolioMetrics:
 """Calculate metrics for optimized portfolio"""

 expected_return = np.dot(weights, expected_returns)
 expected_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
 volatility = np.sqrt(expected_variance)
 sharpe_ratio = expected_return / volatility if volatility > 0 else 0
 var_95 = -volatility * 1.645 # Approximate 95% VaR

 # Diversification ratio
 diversification_ratio = 1.0 / np.sqrt(np.sum(weights ** 2))

 return PortfolioMetrics(
 total_value=0, # Would need current portfolio value
 expected_return=expected_return,
 volatility=volatility,
 sharpe_ratio=sharpe_ratio,
 var_95=var_95,
 max_drawdown=0, # Would need historical simulation
 diversification_ratio=diversification_ratio,
 portfolio_delta=0, # Would calculate from Greeks
 portfolio_gamma=0,
 portfolio_vega=0,
 portfolio_theta=0,
 portfolio_rho=0,
 systematic_risk=0,
 idiosyncratic_risk=0,
 concentration_risk=1.0 - diversification_ratio,
 liquidity_risk=0
 )

 def _calculate_objective_value(self, weights: np.ndarray,
 expected_returns: np.ndarray,
 covariance_matrix: np.ndarray) -> float:
 """Calculate objective function value"""

 expected_return = np.dot(weights, expected_returns)
 expected_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

 if self.objective.maximize_sharpe:
 return expected_return / np.sqrt(expected_variance) if expected_variance > 0 else 0
 elif self.objective.minimize_var:
 return -expected_variance
 elif self.objective.maximize_expected_return:
 return expected_return
 else:
 return expected_return - self.objective.risk_aversion * expected_variance

 def _calculate_portfolio_returns(self, positions: Dict[str, Position],
 returns_history: Dict[str, List[float]]) -> np.ndarray:
 """Calculate historical portfolio returns"""

 symbols = list(positions.keys())
 total_value = sum(abs(pos.market_value) for pos in positions.values())

 if total_value == 0:
 return np.array([])

 weights = {symbol: positions[symbol].market_value / total_value for symbol in symbols}

 # Assume all return series have same length
 min_length = min(len(returns_history[symbol]) for symbol in symbols if symbol in returns_history)

 portfolio_returns = np.zeros(min_length)

 for i in range(min_length):
 period_return = sum(weights.get(symbol, 0) * returns_history[symbol][i]
 for symbol in symbols if symbol in returns_history)
 portfolio_returns[i] = period_return

 return portfolio_returns

 def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
 """Calculate maximum drawdown"""

 if len(returns) == 0:
 return 0.0

 cumulative_returns = np.cumprod(1 + returns)
 running_max = np.maximum.accumulate(cumulative_returns)
 drawdown = (cumulative_returns - running_max) / running_max

 return abs(np.min(drawdown))


# Factory functions
def create_portfolio_optimizer(config: Optional[Dict[str, Any]] = None) -> PortfolioOptimizer:
 """Create a configured portfolio optimizer"""

 default_objective = OptimizationObjective(
 maximize_sharpe=True,
 minimize_greeks_variance=True,
 risk_aversion=3.0
 )

 default_constraints = PortfolioConstraints(
 max_weight_per_position=0.15,
 delta_neutral_tolerance=100.0,
 max_portfolio_var=0.02
 )

 if config:
 # Update configurations
 for key, value in config.items():
 if hasattr(default_objective, key):
 setattr(default_objective, key, value)
 elif hasattr(default_constraints, key):
 setattr(default_constraints, key, value)

 return PortfolioOptimizer(default_objective, default_constraints)


# Example usage
if __name__ == "__main__":
 # Create optimizer
 optimizer = create_portfolio_optimizer()

 # Example positions and data
 positions = {
 "AAPL_231215C150": Position("AAPL_231215C150", 100, 5.0, 50000, 1000, 500),
 "AAPL_231215P140": Position("AAPL_231215P140", -50, 3.0, -15000, -500, 200)
 }

 market_data = {
 "AAPL_231215C150": MarketData("AAPL_231215C150", datetime.now(), 4.90, 5.10, 5.00, 50, 50, 1000, 5000),
 "AAPL_231215P140": MarketData("AAPL_231215P140", datetime.now(), 2.90, 3.10, 3.00, 30, 30, 800, 3000)
 }

 # Example returns history (simplified)
 returns_history = {
 "AAPL_231215C150": np.random.normal(0.001, 0.02, 60).tolist(),
 "AAPL_231215P140": np.random.normal(0.0005, 0.015, 60).tolist()
 }

 greeks_data = {
 "AAPL_231215C150": Greeks(0.6, 0.05, -0.1, 0.3, 0.02, 150.0, datetime.now()),
 "AAPL_231215P140": Greeks(-0.4, 0.05, -0.08, 0.25, -0.01, 150.0, datetime.now())
 }

 # Run optimization
 result = optimizer.optimize_portfolio(positions, market_data, returns_history, greeks_data)

 print("Optimization Results:")
 print(f"Target weights: {result.target_weights}")
 print(f"Expected improvement: {result.expected_improvement}")
 print(f"Constraints satisfied: {result.constraints_satisfied}")
 print(f"Optimization time: {result.optimization_time:.3f}s")