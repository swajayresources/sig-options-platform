"""
Professional Volatility Surface Visualization Framework

This module provides comprehensive visualization capabilities for volatility surfaces,
including 3D surface plots, interactive visualizations, risk analytics displays,
and professional-grade plotting for trading desks and research.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import seaborn as sns
from datetime import datetime, timedelta
import warnings

try:
 import plotly.graph_objects as go
 import plotly.express as px
 from plotly.subplots import make_subplots
 import plotly.figure_factory as ff
 PLOTLY_AVAILABLE = True
except ImportError:
 PLOTLY_AVAILABLE = False
 warnings.warn("Plotly not available. Interactive visualizations will be limited.")

from surface_models import VolatilitySurfaceModel
from calibration_engine import VolatilityQuote
from arbitrage_detection import ArbitrageViolation


@dataclass
class PlotConfig:
 """Configuration for volatility surface plots"""
 figure_size: Tuple[int, int] = (12, 8)
 dpi: int = 300
 style: str = 'seaborn-v0_8'
 colormap: str = 'viridis'
 font_size: int = 12
 title_size: int = 14
 show_grid: bool = True
 alpha: float = 0.8
 surface_resolution: int = 50


@dataclass
class SurfaceVisualizationData:
 """Container for surface visualization data"""
 strikes: np.ndarray
 expiries: np.ndarray
 volatilities: np.ndarray
 log_moneyness: np.ndarray
 forward_prices: np.ndarray
 quotes: List[VolatilityQuote]
 arbitrage_violations: Optional[List[ArbitrageViolation]] = None


class VolatilitySurfaceVisualizer:
 """Professional volatility surface visualization framework"""

 def __init__(self, config: PlotConfig = None):
 self.config = config or PlotConfig()
 plt.style.use(self.config.style)
 sns.set_palette("husl")

 def plot_surface_3d(self, surface_data: SurfaceVisualizationData,
 model: Optional[VolatilitySurfaceModel] = None,
 title: str = "Implied Volatility Surface",
 save_path: Optional[str] = None) -> plt.Figure:
 """Create 3D volatility surface plot"""

 fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
 ax = fig.add_subplot(111, projection='3d')

 # Create meshgrid for surface
 strikes_grid, expiries_grid = np.meshgrid(surface_data.strikes, surface_data.expiries)

 # Plot surface
 surface = ax.plot_surface(strikes_grid, expiries_grid, surface_data.volatilities,
 cmap=self.config.colormap, alpha=self.config.alpha,
 linewidth=0, antialiased=True)

 # Plot market quotes as scatter points
 if surface_data.quotes:
 quote_strikes = [q.strike for q in surface_data.quotes]
 quote_expiries = [q.expiry for q in surface_data.quotes]
 quote_vols = [q.implied_vol for q in surface_data.quotes]

 ax.scatter(quote_strikes, quote_expiries, quote_vols,
 c='red', s=50, alpha=0.8, label='Market Quotes')

 # Highlight arbitrage violations
 if surface_data.arbitrage_violations:
 self._highlight_arbitrage_3d(ax, surface_data.arbitrage_violations)

 # Formatting
 ax.set_xlabel('Strike', fontsize=self.config.font_size)
 ax.set_ylabel('Time to Expiry', fontsize=self.config.font_size)
 ax.set_zlabel('Implied Volatility', fontsize=self.config.font_size)
 ax.set_title(title, fontsize=self.config.title_size, pad=20)

 # Add colorbar
 fig.colorbar(surface, ax=ax, shrink=0.6, aspect=20, label='Implied Volatility')

 if surface_data.quotes:
 ax.legend()

 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

 return fig

 def plot_volatility_heatmap(self, surface_data: SurfaceVisualizationData,
 title: str = "Volatility Surface Heatmap",
 save_path: Optional[str] = None) -> plt.Figure:
 """Create volatility surface heatmap"""

 fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

 # Create heatmap
 im = ax.imshow(surface_data.volatilities, cmap=self.config.colormap,
 aspect='auto', origin='lower', interpolation='bilinear')

 # Set ticks and labels
 n_strikes, n_expiries = len(surface_data.strikes), len(surface_data.expiries)
 strike_ticks = np.linspace(0, n_strikes-1, min(10, n_strikes))
 expiry_ticks = np.linspace(0, n_expiries-1, min(10, n_expiries))

 ax.set_xticks(strike_ticks)
 ax.set_xticklabels([f"{surface_data.strikes[int(i)]:.0f}" for i in strike_ticks])
 ax.set_yticks(expiry_ticks)
 ax.set_yticklabels([f"{surface_data.expiries[int(i)]:.2f}" for i in expiry_ticks])

 ax.set_xlabel('Strike', fontsize=self.config.font_size)
 ax.set_ylabel('Time to Expiry', fontsize=self.config.font_size)
 ax.set_title(title, fontsize=self.config.title_size)

 # Add colorbar
 cbar = plt.colorbar(im, ax=ax)
 cbar.set_label('Implied Volatility', fontsize=self.config.font_size)

 # Overlay market quotes
 if surface_data.quotes:
 self._overlay_quotes_heatmap(ax, surface_data)

 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

 return fig

 def plot_volatility_smile_evolution(self, quotes_by_expiry: Dict[float, List[VolatilityQuote]],
 title: str = "Volatility Smile Evolution",
 save_path: Optional[str] = None) -> plt.Figure:
 """Plot evolution of volatility smile across expiries"""

 fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

 colors = cm.viridis(np.linspace(0, 1, len(quotes_by_expiry)))

 for i, (expiry, quotes) in enumerate(sorted(quotes_by_expiry.items())):
 if not quotes:
 continue

 strikes = [q.strike for q in quotes]
 vols = [q.implied_vol for q in quotes]
 log_moneyness = [np.log(q.strike / q.forward) for q in quotes]

 # Sort by log moneyness for plotting
 sorted_data = sorted(zip(log_moneyness, vols))
 log_k, sorted_vols = zip(*sorted_data)

 ax.plot(log_k, sorted_vols, 'o-', color=colors[i],
 label=f'T={expiry:.2f}', linewidth=2, markersize=6)

 ax.set_xlabel('Log Moneyness', fontsize=self.config.font_size)
 ax.set_ylabel('Implied Volatility', fontsize=self.config.font_size)
 ax.set_title(title, fontsize=self.config.title_size)
 ax.grid(True, alpha=0.3)
 ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

 return fig

 def plot_term_structure(self, atm_vols_by_expiry: Dict[float, float],
 title: str = "At-the-Money Volatility Term Structure",
 save_path: Optional[str] = None) -> plt.Figure:
 """Plot volatility term structure"""

 fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

 expiries = sorted(atm_vols_by_expiry.keys())
 vols = [atm_vols_by_expiry[t] for t in expiries]

 ax.plot(expiries, vols, 'o-', linewidth=3, markersize=8, color='steelblue')
 ax.fill_between(expiries, vols, alpha=0.3, color='steelblue')

 ax.set_xlabel('Time to Expiry', fontsize=self.config.font_size)
 ax.set_ylabel('At-the-Money Implied Volatility', fontsize=self.config.font_size)
 ax.set_title(title, fontsize=self.config.title_size)
 ax.grid(True, alpha=0.3)

 # Add annotations for key points
 max_vol_idx = np.argmax(vols)
 min_vol_idx = np.argmin(vols)

 ax.annotate(f'Max: {vols[max_vol_idx]:.3f}',
 xy=(expiries[max_vol_idx], vols[max_vol_idx]),
 xytext=(10, 10), textcoords='offset points',
 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

 ax.annotate(f'Min: {vols[min_vol_idx]:.3f}',
 xy=(expiries[min_vol_idx], vols[min_vol_idx]),
 xytext=(10, -20), textcoords='offset points',
 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

 return fig

 def plot_risk_scenarios(self, base_surface: SurfaceVisualizationData,
 shock_surfaces: Dict[str, SurfaceVisualizationData],
 title: str = "Volatility Surface Risk Scenarios",
 save_path: Optional[str] = None) -> plt.Figure:
 """Plot risk scenario analysis"""

 n_scenarios = len(shock_surfaces)
 n_cols = min(3, n_scenarios + 1)
 n_rows = (n_scenarios + 2) // n_cols

 fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4),
 dpi=self.config.dpi)
 if n_rows == 1:
 axes = axes.reshape(1, -1)

 # Plot base surface
 ax = axes[0, 0] if n_cols > 1 else axes[0]
 im_base = ax.imshow(base_surface.volatilities, cmap=self.config.colormap,
 aspect='auto', origin='lower')
 ax.set_title('Base Surface', fontsize=self.config.font_size)
 self._format_heatmap_axes(ax, base_surface)

 # Plot shock scenarios
 for i, (scenario_name, shock_surface) in enumerate(shock_surfaces.items(), 1):
 row = i // n_cols
 col = i % n_cols
 ax = axes[row, col] if n_rows > 1 else axes[col]

 # Calculate difference from base
 vol_diff = shock_surface.volatilities - base_surface.volatilities

 im = ax.imshow(vol_diff, cmap='RdBu_r', aspect='auto', origin='lower')
 ax.set_title(f'{scenario_name}', fontsize=self.config.font_size)
 self._format_heatmap_axes(ax, shock_surface)

 # Add colorbar for difference
 cbar = plt.colorbar(im, ax=ax)
 cbar.set_label('Vol Difference', fontsize=self.config.font_size-2)

 # Hide unused subplots
 for i in range(n_scenarios + 1, n_rows * n_cols):
 row = i // n_cols
 col = i % n_cols
 axes[row, col].set_visible(False)

 fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

 return fig

 def create_interactive_surface(self, surface_data: SurfaceVisualizationData,
 title: str = "Interactive Volatility Surface") -> Optional[go.Figure]:
 """Create interactive 3D surface using Plotly"""

 if not PLOTLY_AVAILABLE:
 warnings.warn("Plotly not available. Cannot create interactive surface.")
 return None

 fig = go.Figure()

 # Add surface
 fig.add_trace(go.Surface(
 x=surface_data.strikes,
 y=surface_data.expiries,
 z=surface_data.volatilities,
 colorscale='Viridis',
 name='Volatility Surface',
 hovertemplate='Strike: %{x}<br>Expiry: %{y}<br>Vol: %{z:.3f}<extra></extra>'
 ))

 # Add market quotes as scatter points
 if surface_data.quotes:
 quote_strikes = [q.strike for q in surface_data.quotes]
 quote_expiries = [q.expiry for q in surface_data.quotes]
 quote_vols = [q.implied_vol for q in surface_data.quotes]

 fig.add_trace(go.Scatter3d(
 x=quote_strikes,
 y=quote_expiries,
 z=quote_vols,
 mode='markers',
 marker=dict(size=5, color='red'),
 name='Market Quotes',
 hovertemplate='Strike: %{x}<br>Expiry: %{y}<br>Vol: %{z:.3f}<extra></extra>'
 ))

 fig.update_layout(
 title=title,
 scene=dict(
 xaxis_title='Strike',
 yaxis_title='Time to Expiry',
 zaxis_title='Implied Volatility',
 camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
 ),
 width=900,
 height=700
 )

 return fig

 def create_dashboard(self, surface_data: SurfaceVisualizationData,
 quotes_by_expiry: Dict[float, List[VolatilityQuote]],
 atm_vols: Dict[float, float]) -> Optional[go.Figure]:
 """Create comprehensive volatility dashboard"""

 if not PLOTLY_AVAILABLE:
 warnings.warn("Plotly not available. Cannot create dashboard.")
 return None

 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('3D Surface', 'Volatility Heatmap',
 'Smile Evolution', 'Term Structure'),
 specs=[[{'type': 'surface'}, {'type': 'heatmap'}],
 [{'colspan': 2}, None]],
 vertical_spacing=0.08,
 horizontal_spacing=0.05
 )

 # 3D Surface
 fig.add_trace(go.Surface(
 x=surface_data.strikes,
 y=surface_data.expiries,
 z=surface_data.volatilities,
 colorscale='Viridis',
 showscale=False
 ), row=1, col=1)

 # Heatmap
 fig.add_trace(go.Heatmap(
 x=surface_data.strikes,
 y=surface_data.expiries,
 z=surface_data.volatilities,
 colorscale='Viridis'
 ), row=1, col=2)

 # Smile Evolution
 colors = px.colors.qualitative.Set1
 for i, (expiry, quotes) in enumerate(sorted(quotes_by_expiry.items())):
 if quotes:
 log_moneyness = [np.log(q.strike / q.forward) for q in quotes]
 vols = [q.implied_vol for q in quotes]
 sorted_data = sorted(zip(log_moneyness, vols))
 log_k, sorted_vols = zip(*sorted_data)

 fig.add_trace(go.Scatter(
 x=log_k, y=sorted_vols,
 mode='lines+markers',
 name=f'T={expiry:.2f}',
 line=dict(color=colors[i % len(colors)])
 ), row=2, col=1)

 # Term Structure
 expiries = sorted(atm_vols.keys())
 vols = [atm_vols[t] for t in expiries]

 fig.add_trace(go.Scatter(
 x=expiries, y=vols,
 mode='lines+markers',
 name='ATM Term Structure',
 line=dict(color='steelblue', width=3),
 marker=dict(size=8)
 ), row=2, col=1)

 fig.update_layout(
 title='Volatility Surface Dashboard',
 height=800,
 showlegend=True
 )

 return fig

 def _highlight_arbitrage_3d(self, ax, violations: List[ArbitrageViolation]):
 """Highlight arbitrage violations on 3D plot"""
 for violation in violations:
 # Simple highlighting - in practice would be more sophisticated
 ax.text(violation.strike, violation.expiry,
 violation.observed_value,
 'ARB', color='red', fontsize=8, weight='bold')

 def _overlay_quotes_heatmap(self, ax, surface_data: SurfaceVisualizationData):
 """Overlay market quotes on heatmap"""
 # Convert quote positions to heatmap coordinates
 for quote in surface_data.quotes:
 # Find closest grid positions
 strike_idx = np.argmin(np.abs(surface_data.strikes - quote.strike))
 expiry_idx = np.argmin(np.abs(surface_data.expiries - quote.expiry))

 # Add marker
 ax.plot(strike_idx, expiry_idx, 'ro', markersize=4, alpha=0.8)

 def _format_heatmap_axes(self, ax, surface_data: SurfaceVisualizationData):
 """Format heatmap axes"""
 n_strikes, n_expiries = len(surface_data.strikes), len(surface_data.expiries)
 strike_ticks = np.linspace(0, n_strikes-1, min(8, n_strikes))
 expiry_ticks = np.linspace(0, n_expiries-1, min(8, n_expiries))

 ax.set_xticks(strike_ticks)
 ax.set_xticklabels([f"{surface_data.strikes[int(i)]:.0f}" for i in strike_ticks],
 fontsize=self.config.font_size-2)
 ax.set_yticks(expiry_ticks)
 ax.set_yticklabels([f"{surface_data.expiries[int(i)]:.2f}" for i in expiry_ticks],
 fontsize=self.config.font_size-2)

 ax.set_xlabel('Strike', fontsize=self.config.font_size-1)
 ax.set_ylabel('Expiry', fontsize=self.config.font_size-1)


class RiskVisualization:
 """Specialized visualization for risk metrics and Greeks"""

 def __init__(self, config: PlotConfig = None):
 self.config = config or PlotConfig()
 self.visualizer = VolatilitySurfaceVisualizer(config)

 def plot_volatility_greeks(self, surface_data: SurfaceVisualizationData,
 greeks_data: Dict[str, np.ndarray],
 title: str = "Volatility Greeks",
 save_path: Optional[str] = None) -> plt.Figure:
 """Plot volatility Greeks (vega, volga, vanna)"""

 n_greeks = len(greeks_data)
 n_cols = min(3, n_greeks)
 n_rows = (n_greeks + n_cols - 1) // n_cols

 fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4),
 dpi=self.config.dpi)

 if n_greeks == 1:
 axes = [axes]
 elif n_rows == 1:
 axes = axes.reshape(1, -1)

 for i, (greek_name, greek_values) in enumerate(greeks_data.items()):
 row = i // n_cols
 col = i % n_cols
 ax = axes[row][col] if n_rows > 1 else axes[col]

 im = ax.imshow(greek_values, cmap='RdBu_r', aspect='auto', origin='lower')
 ax.set_title(f'{greek_name.capitalize()}', fontsize=self.config.font_size)
 self.visualizer._format_heatmap_axes(ax, surface_data)

 cbar = plt.colorbar(im, ax=ax)
 cbar.set_label(greek_name.capitalize(), fontsize=self.config.font_size-2)

 # Hide unused subplots
 for i in range(n_greeks, n_rows * n_cols):
 row = i // n_cols
 col = i % n_cols
 if n_rows > 1:
 axes[row][col].set_visible(False)
 else:
 axes[col].set_visible(False)

 fig.suptitle(title, fontsize=self.config.title_size, y=0.98)
 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

 return fig

 def plot_pnl_attribution(self, pnl_components: Dict[str, float],
 title: str = "P&L Attribution",
 save_path: Optional[str] = None) -> plt.Figure:
 """Plot P&L attribution breakdown"""

 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size,
 dpi=self.config.dpi)

 components = list(pnl_components.keys())
 values = list(pnl_components.values())
 colors = plt.cm.Set3(np.linspace(0, 1, len(components)))

 # Waterfall chart
 cumulative = 0
 bar_positions = []
 bar_heights = []
 bar_colors = []

 for i, (comp, val) in enumerate(pnl_components.items()):
 bar_positions.append(i)
 bar_heights.append(val)
 bar_colors.append('green' if val >= 0 else 'red')

 if i > 0:
 ax1.plot([i-1+0.4, i-0.4], [cumulative, cumulative], 'k--', alpha=0.5)

 cumulative += val

 bars = ax1.bar(bar_positions, bar_heights, color=bar_colors, alpha=0.7)
 ax1.set_xticks(bar_positions)
 ax1.set_xticklabels(components, rotation=45, ha='right')
 ax1.set_ylabel('P&L', fontsize=self.config.font_size)
 ax1.set_title('P&L Waterfall', fontsize=self.config.font_size)
 ax1.grid(True, alpha=0.3)

 # Add value labels on bars
 for bar, val in zip(bars, values):
 height = bar.get_height()
 ax1.text(bar.get_x() + bar.get_width()/2., height,
 f'{val:.0f}', ha='center', va='bottom' if height >= 0 else 'top')

 # Pie chart
 positive_values = [max(0, v) for v in values]
 if sum(positive_values) > 0:
 ax2.pie(positive_values, labels=components, autopct='%1.1f%%',
 colors=colors, startangle=90)
 ax2.set_title('Positive P&L Breakdown', fontsize=self.config.font_size)

 fig.suptitle(title, fontsize=self.config.title_size)
 plt.tight_layout()

 if save_path:
 plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')

 return fig


# Utility functions for data preparation
def prepare_surface_data(quotes: List[VolatilityQuote],
 model: Optional[VolatilitySurfaceModel] = None,
 resolution: int = 50) -> SurfaceVisualizationData:
 """Prepare data for surface visualization"""

 if not quotes:
 raise ValueError("No quotes provided")

 # Extract unique strikes and expiries
 strikes = sorted(set(q.strike for q in quotes))
 expiries = sorted(set(q.expiry for q in quotes))
 forward_prices = {q.expiry: q.forward for q in quotes}

 # Create regular grid
 strike_min, strike_max = min(strikes), max(strikes)
 expiry_min, expiry_max = min(expiries), max(expiries)

 strike_grid = np.linspace(strike_min, strike_max, resolution)
 expiry_grid = np.linspace(expiry_min, expiry_max, resolution)

 # Create volatility surface
 vol_surface = np.zeros((len(expiry_grid), len(strike_grid)))

 if model:
 # Use calibrated model
 for i, expiry in enumerate(expiry_grid):
 forward = np.interp(expiry, sorted(forward_prices.keys()),
 [forward_prices[t] for t in sorted(forward_prices.keys())])
 for j, strike in enumerate(strike_grid):
 try:
 log_moneyness = np.log(strike / forward)
 vol = model.calculate_volatility(log_moneyness, expiry)
 vol_surface[i, j] = vol
 except:
 vol_surface[i, j] = np.nan
 else:
 # Interpolate from market quotes
 from scipy.interpolate import griddata
 points = []
 values = []
 for q in quotes:
 points.append([q.strike, q.expiry])
 values.append(q.implied_vol)

 for i, expiry in enumerate(expiry_grid):
 for j, strike in enumerate(strike_grid):
 vol = griddata(points, values, (strike, expiry), method='cubic')
 vol_surface[i, j] = vol if not np.isnan(vol) else 0

 log_moneyness_grid = np.log(strike_grid[:, np.newaxis] /
 np.array([forward_prices.get(t, strike_grid[0])
 for t in expiry_grid]))

 return SurfaceVisualizationData(
 strikes=strike_grid,
 expiries=expiry_grid,
 volatilities=vol_surface,
 log_moneyness=log_moneyness_grid.T,
 forward_prices=np.array([forward_prices.get(t, strike_grid[0])
 for t in expiry_grid]),
 quotes=quotes
 )


def create_visualization_suite(quotes: List[VolatilityQuote],
 model: Optional[VolatilitySurfaceModel] = None) -> Dict[str, plt.Figure]:
 """Create complete visualization suite"""

 config = PlotConfig(figure_size=(10, 8))
 visualizer = VolatilitySurfaceVisualizer(config)

 surface_data = prepare_surface_data(quotes, model)

 # Group quotes by expiry for smile evolution
 quotes_by_expiry = {}
 for quote in quotes:
 if quote.expiry not in quotes_by_expiry:
 quotes_by_expiry[quote.expiry] = []
 quotes_by_expiry[quote.expiry].append(quote)

 # Extract ATM volatilities
 atm_vols = {}
 for expiry, expiry_quotes in quotes_by_expiry.items():
 # Find ATM quote (closest to forward)
 atm_quote = min(expiry_quotes, key=lambda q: abs(q.strike - q.forward))
 atm_vols[expiry] = atm_quote.implied_vol

 figures = {
 'surface_3d': visualizer.plot_surface_3d(surface_data),
 'heatmap': visualizer.plot_volatility_heatmap(surface_data),
 'smile_evolution': visualizer.plot_volatility_smile_evolution(quotes_by_expiry),
 'term_structure': visualizer.plot_term_structure(atm_vols)
 }

 return figures