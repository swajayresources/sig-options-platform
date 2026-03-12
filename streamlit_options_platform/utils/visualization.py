"""
Visualization Module
Advanced chart generation and visualization utilities
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

class ChartGenerator:
    """Advanced chart generation for options analytics"""

    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }

        self.theme = {
            'background': 'white',
            'grid_color': '#f0f0f0',
            'text_color': '#333333'
        }

    def create_portfolio_overview_chart(self, positions_data: pd.DataFrame) -> go.Figure:
        """Create comprehensive portfolio overview chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Position Distribution', 'P&L by Symbol', 'Greeks Distribution', 'Risk Metrics'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )

        if not positions_data.empty:
            # Position distribution pie chart
            symbols = positions_data['Symbol'].str.split('_').str[0].value_counts()

            fig.add_trace(
                go.Pie(
                    labels=symbols.index,
                    values=symbols.values,
                    name="Position Distribution"
                ),
                row=1, col=1
            )

            # P&L by symbol
            pnl_by_symbol = positions_data.groupby(
                positions_data['Symbol'].str.split('_').str[0]
            )['Unrealized_PnL'].sum()

            colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_by_symbol.values]

            fig.add_trace(
                go.Bar(
                    x=pnl_by_symbol.index,
                    y=pnl_by_symbol.values,
                    marker_color=colors,
                    name="P&L by Symbol"
                ),
                row=1, col=2
            )

        fig.update_layout(
            title_text="Portfolio Overview",
            showlegend=False,
            height=600
        )

        return fig

    def create_volatility_surface_3d(self, strikes: List[float], expiries: List[int],
                                   vol_matrix: np.ndarray, symbol: str) -> go.Figure:
        """Create 3D volatility surface"""
        fig = go.Figure(data=[
            go.Surface(
                x=strikes,
                y=expiries,
                z=vol_matrix,
                colorscale='Viridis',
                name=f'{symbol} Volatility Surface',
                hovertemplate='<b>Strike:</b> %{x}<br>' +
                             '<b>Days to Exp:</b> %{y}<br>' +
                             '<b>Implied Vol:</b> %{z:.1f}%<extra></extra>'
            )
        ])

        fig.update_layout(
            title=f'{symbol} Implied Volatility Surface',
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiration',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            height=600
        )

        return fig

    def create_greeks_heatmap(self, greeks_data: Dict[str, np.ndarray],
                            strikes: List[float], symbols: List[str]) -> go.Figure:
        """Create Greeks heatmap visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]]
        )

        greek_names = ['Delta', 'Gamma', 'Theta', 'Vega']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colorscales = ['RdYlBu_r', 'Viridis', 'Reds', 'Blues']

        for i, (greek, pos, colorscale) in enumerate(zip(greek_names, positions, colorscales)):
            if greek.lower() in greeks_data:
                fig.add_trace(
                    go.Heatmap(
                        z=greeks_data[greek.lower()],
                        x=strikes,
                        y=symbols,
                        colorscale=colorscale,
                        name=greek,
                        hovertemplate=f'<b>{greek}:</b> %{{z:.3f}}<br>' +
                                     '<b>Strike:</b> %{x}<br>' +
                                     '<b>Symbol:</b> %{y}<extra></extra>'
                    ),
                    row=pos[0], col=pos[1]
                )

        fig.update_layout(
            title_text="Greeks Heatmap",
            height=600
        )

        return fig

    def create_pnl_waterfall_chart(self, pnl_attribution: Dict[str, float]) -> go.Figure:
        """Create P&L attribution waterfall chart"""
        categories = list(pnl_attribution.keys())
        values = list(pnl_attribution.values())

        # Prepare waterfall data
        measures = ["relative"] * (len(categories) - 1) + ["total"]

        fig = go.Figure(go.Waterfall(
            name="P&L Attribution",
            orientation="v",
            measure=measures,
            x=categories,
            y=values,
            textposition="outside",
            text=[f"${v:,.0f}" for v in values],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))

        fig.update_layout(
            title="P&L Attribution Waterfall",
            xaxis_title="Attribution Category",
            yaxis_title="P&L ($)",
            height=500
        )

        return fig

    def create_correlation_heatmap(self, correlation_matrix: np.ndarray,
                                 assets: List[str]) -> go.Figure:
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=assets,
            y=assets,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title="Cross-Asset Correlation Matrix",
            xaxis_title="Assets",
            yaxis_title="Assets",
            height=500
        )

        return fig

    def create_options_flow_bubble_chart(self, flow_data: pd.DataFrame) -> go.Figure:
        """Create options flow bubble chart"""
        if flow_data.empty:
            return go.Figure().add_annotation(text="No flow data available", showarrow=False)

        fig = px.scatter(
            flow_data,
            x='time',
            y='strike',
            size='premium',
            color='sentiment',
            hover_data=['symbol', 'volume', 'type'],
            title="Options Flow Analysis",
            color_discrete_map={
                'bullish': 'green',
                'bearish': 'red',
                'neutral': 'gray'
            }
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Strike Price",
            height=500
        )

        return fig

    def create_volatility_term_structure(self, vol_data: Dict[str, List[float]],
                                       days_to_expiry: List[int]) -> go.Figure:
        """Create volatility term structure chart"""
        fig = go.Figure()

        colors = px.colors.qualitative.Set1

        for i, (asset, vols) in enumerate(vol_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=days_to_expiry,
                    y=vols,
                    mode='lines+markers',
                    name=asset,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                )
            )

        fig.update_layout(
            title="Volatility Term Structure Comparison",
            xaxis_title="Days to Expiration",
            yaxis_title="Implied Volatility (%)",
            hovermode='x unified',
            height=500
        )

        return fig

    def create_risk_gauge_chart(self, current_value: float, max_value: float,
                              title: str, thresholds: Dict[str, float] = None) -> go.Figure:
        """Create risk gauge chart"""
        if thresholds is None:
            thresholds = {
                'low': max_value * 0.3,
                'medium': max_value * 0.7,
                'high': max_value
            }

        utilization = (current_value / max_value) * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=utilization,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{title} ({current_value:.0f}/{max_value:.0f})"},
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

        fig.update_layout(height=300)
        return fig

    def create_performance_timeline(self, performance_data: Dict[str, List]) -> go.Figure:
        """Create performance timeline chart"""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=performance_data['dates'],
                y=performance_data['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='blue', width=2),
                fill='tonexty'
            )
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)

        fig.update_layout(
            title="Portfolio Performance Timeline",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
            hovermode='x unified',
            height=400
        )

        return fig

    def create_sector_rotation_chart(self, sector_data: pd.DataFrame) -> go.Figure:
        """Create sector rotation bubble chart"""
        fig = px.scatter(
            sector_data,
            x='momentum',
            y='volatility',
            size='volume',
            color='performance',
            hover_name='sector',
            title="Sector Rotation Analysis",
            color_continuous_scale='RdYlGn',
            labels={
                'momentum': 'Momentum Score',
                'volatility': 'Volatility',
                'performance': 'Performance (%)'
            }
        )

        # Add quadrant lines
        fig.add_hline(y=sector_data['volatility'].median(), line_dash="dash", opacity=0.5)
        fig.add_vline(x=sector_data['momentum'].median(), line_dash="dash", opacity=0.5)

        fig.update_layout(height=500)
        return fig

    def create_sentiment_gauge_dashboard(self, sentiment_data: Dict[str, float]) -> go.Figure:
        """Create sentiment gauge dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=list(sentiment_data.keys())
        )

        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']

        for i, (name, value) in enumerate(sentiment_data.items()):
            if i >= len(positions):
                break

            # Determine color based on value
            if value < 20:
                color = 'red'
            elif value < 40:
                color = 'orange'
            elif value < 60:
                color = 'yellow'
            elif value < 80:
                color = 'lightgreen'
            else:
                color = 'green'

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': name.replace('_', ' ').title()},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgray"},
                            {'range': [20, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=positions[i][0], col=positions[i][1]
            )

        fig.update_layout(
            title_text="Market Sentiment Dashboard",
            height=600
        )

        return fig

    def create_options_chain_heatmap(self, options_data: pd.DataFrame, current_price: float) -> go.Figure:
        """Create options chain heatmap"""
        if options_data.empty:
            return go.Figure().add_annotation(text="No options data available", showarrow=False)

        # Pivot data for heatmap
        pivot_data = options_data.pivot_table(
            index='Strike',
            columns='Type',
            values='Volume',
            fill_value=0
        )

        fig = go.Figure()

        # Add call and put volume heatmaps
        if 'CALL' in pivot_data.columns:
            fig.add_trace(
                go.Heatmap(
                    z=[pivot_data['CALL'].values],
                    x=pivot_data.index,
                    y=['Calls'],
                    colorscale='Greens',
                    name='Call Volume',
                    showscale=False
                )
            )

        if 'PUT' in pivot_data.columns:
            fig.add_trace(
                go.Heatmap(
                    z=[pivot_data['PUT'].values],
                    x=pivot_data.index,
                    y=['Puts'],
                    colorscale='Reds',
                    name='Put Volume',
                    showscale=False
                )
            )

        # Add current price line
        fig.add_vline(x=current_price, line_dash="dash", line_color="blue", line_width=3)

        fig.update_layout(
            title="Options Chain Volume Heatmap",
            xaxis_title="Strike Price",
            yaxis_title="Option Type",
            height=300
        )

        return fig

    def create_strategy_comparison_radar(self, strategies_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create radar chart comparing strategies"""
        fig = go.Figure()

        metrics = ['Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor', 'Volatility']

        for strategy_name, strategy_metrics in strategies_data.items():
            values = [strategy_metrics.get(metric.lower().replace(' ', '_'), 0) for metric in metrics]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=strategy_name
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Strategy Performance Comparison",
            height=500
        )

        return fig