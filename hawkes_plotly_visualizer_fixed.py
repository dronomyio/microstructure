#!/usr/bin/env python3
"""
Ultra-Fast Hawkes Processes - Interactive Plotly Visualizations (FIXED VERSION)
Creates comprehensive interactive dashboards from C++/CUDA output
"""

import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import math

class HawkesPlotlyVisualizer:
    """
    Interactive Plotly visualizations for Hawkes process analysis results (FIXED VERSION)
    """
    
    def __init__(self, theme: str = "plotly_dark"):
        """
        Initialize the visualizer
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white')
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set3
        self.setup_default_layout()
    
    def setup_default_layout(self):
        """Setup default layout configuration"""
        self.default_layout = dict(
            template=self.theme,
            font=dict(family="Arial, sans-serif", size=12),
            title_font=dict(size=16, family="Arial Black, sans-serif"),
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=60, r=60, t=80, b=60)
        )
    
    def create_comprehensive_dashboard(
        self,
        results: Dict,
        events_df: pd.DataFrame,
        ticker: str = "DEFAULT",
        save_html: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive interactive dashboard with all Hawkes analysis visualizations
        
        Args:
            results: Dictionary containing Hawkes analysis results from C++/CUDA
            events_df: Original events DataFrame
            ticker: Stock ticker symbol
            save_html: Optional path to save HTML file
            
        Returns:
            Plotly Figure object
        """
        
        # Create subplot layout (3 rows x 3 columns) - FIXED: Simplified layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Event Timeline & Intensity Evolution",
                "Intensity Distribution", 
                "Parameter Summary",
                "Clustering Coefficient Timeline",
                "Residuals Analysis",
                "Inter-arrival Time Distribution",
                "Intensity Autocorrelation",
                "Performance Metrics",
                "Burstiness Analysis"
            ],
            specs=[
                [{"secondary_y": True}, {"type": "histogram"}, {"type": "table"}],
                [{}, {}, {}],
                [{}, {"type": "bar"}, {}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        # Convert timestamps to relative time in seconds
        times = (events_df['timestamp_ns'].values - events_df['timestamp_ns'].iloc[0]) * 1e-9
        
        # Extract results data
        intensities = np.array(results['intensities'])
        clustering_coeffs = np.array(results['clustering_coefficients'])
        residuals = np.array(results['residuals'])
        
        # 1. Event Timeline & Intensity Evolution (Row 1, Col 1)
        self._add_timeline_and_intensity(fig, times, intensities, results, row=1, col=1)
        
        # 2. Intensity Distribution (Row 1, Col 2)
        self._add_intensity_distribution(fig, intensities, results, row=1, col=2)
        
        # 3. Parameter Summary Table (Row 1, Col 3)
        self._add_parameter_table(fig, results, ticker, row=1, col=3)
        
        # 4. Clustering Coefficient Timeline (Row 2, Col 1)
        self._add_clustering_analysis_fixed(fig, times, clustering_coeffs, results, row=2, col=1)
        
        # 5. Residuals Analysis (Row 2, Col 2)
        self._add_residuals_analysis_fixed(fig, times, residuals, row=2, col=2)
        
        # 6. Inter-arrival Time Distribution (Row 2, Col 3)
        self._add_interarrival_analysis(fig, times, row=2, col=3)
        
        # 7. Intensity Autocorrelation (Row 3, Col 1)
        self._add_autocorrelation_analysis(fig, intensities, row=3, col=1)
        
        # 8. Performance Metrics (Row 3, Col 2)
        self._add_performance_metrics(fig, results, row=3, col=2)
        
        # 9. Burstiness Analysis (Row 3, Col 3)
        self._add_burstiness_analysis_fixed(fig, times, results, row=3, col=3)
        
        # Update layout
        fig.update_layout(
            title=f"Hawkes Process Analysis Dashboard - {ticker} ({len(events_df):,} Events)",
            height=1200,
            **self.default_layout
        )
        
        # Save HTML if requested
        if save_html:
            fig.write_html(save_html)
            print(f"✓ Interactive dashboard saved to {save_html}")
        
        return fig
    
    def _add_timeline_and_intensity(self, fig, times, intensities, results, row, col):
        """Add event timeline and intensity evolution"""
        
        # Event timeline (scatter)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.ones(len(times)) * 0.1,
                mode='markers',
                marker=dict(size=3, color=self.colors[0], opacity=0.6),
                name='Events',
                hovertemplate='Time: %{x:.2f}s<br>Event<extra></extra>'
            ),
            row=row, col=col, secondary_y=False
        )
        
        # Intensity evolution (line)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=intensities,
                mode='lines',
                line=dict(color=self.colors[1], width=2),
                name='Intensity λ(t)',
                hovertemplate='Time: %{x:.2f}s<br>Intensity: %{y:.4f}<extra></extra>'
            ),
            row=row, col=col, secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", row=row, col=col)
        fig.update_yaxes(title_text="Events", secondary_y=False, row=row, col=col)
        fig.update_yaxes(title_text="Intensity λ(t)", secondary_y=True, row=row, col=col)
    
    def _add_intensity_distribution(self, fig, intensities, results, row, col):
        """Add intensity distribution histogram"""
        
        fig.add_trace(
            go.Histogram(
                x=intensities,
                nbinsx=50,
                name='Intensity Distribution',
                marker_color=self.colors[2],
                opacity=0.7,
                hovertemplate='Intensity: %{x:.4f}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Intensity", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def _add_parameter_table(self, fig, results, ticker, row, col):
        """Add parameter summary table"""
        
        # Prepare table data
        parameters = [
            ["Parameter", "Value", "Description"],
            ["μ (Base)", f"{results['mu']:.6f}", "Background intensity"],
            ["α (Jump)", f"{results['alpha']:.6f}", "Self-excitation strength"],
            ["β (Decay)", f"{results['beta']:.6f}", "Decay rate"],
            ["Branching Ratio", f"{results['branching_ratio']:.6f}", "α/β stability measure"],
            ["Log-Likelihood", f"{results['log_likelihood']:.2f}", "Model fit quality"],
            ["AIC", f"{results['aic']:.2f}", "Information criterion"],
            ["BIC", f"{results['bic']:.2f}", "Bayesian criterion"],
            ["Processing Time", f"{results['processing_time_ms']:.2f} ms", "Computation speed"],
            ["Throughput", f"{results['throughput_events_per_sec']:,.0f}/sec", "Events processed"],
            ["Method", results['processing_method'], "GPU/SIMD acceleration"],
            ["Subcritical", str(results['is_subcritical']), "Process stability"]
        ]
        
        # Color code based on values
        cell_colors = [['lightgray'] * 3]  # Header
        for i, row_data in enumerate(parameters[1:]):
            if 'Branching Ratio' in row_data[0]:
                color = 'lightgreen' if float(row_data[1]) < 1.0 else 'lightcoral'
            elif 'Processing Time' in row_data[0]:
                color = 'lightgreen' if float(row_data[1].split()[0]) < 100 else 'lightyellow'
            elif 'Subcritical' in row_data[0]:
                color = 'lightgreen' if row_data[1] == 'True' else 'lightcoral'
            else:
                color = 'white'
            cell_colors.append([color] * 3)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=parameters[0],
                    fill_color='darkblue',
                    font_color='white',
                    font_size=12,
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*parameters[1:])),
                    fill_color=list(zip(*cell_colors[1:])),
                    font_size=10,
                    align=['left', 'center', 'left'],
                    height=25
                )
            ),
            row=row, col=col
        )
    
    def _add_clustering_analysis_fixed(self, fig, times, clustering_coeffs, results, row, col):
        """Add clustering coefficient analysis (FIXED VERSION)"""
        
        # Clustering timeline
        fig.add_trace(
            go.Scatter(
                x=times,
                y=clustering_coeffs,
                mode='lines+markers',
                line=dict(color=self.colors[3], width=1),
                marker=dict(size=3),
                name='Clustering Coefficient',
                hovertemplate='Time: %{x:.2f}s<br>Clustering: %{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # FIXED: Add average line using shapes instead of add_hline
        avg_clustering = results['clustering_coefficient']
        fig.add_shape(
            type="line",
            x0=times[0], x1=times[-1],
            y0=avg_clustering, y1=avg_clustering,
            line=dict(color="orange", dash="dash"),
            row=row, col=col
        )
        
        # Add annotation for average line
        fig.add_annotation(
            x=times[len(times)//2],
            y=avg_clustering,
            text=f"Average: {avg_clustering:.4f}",
            showarrow=False,
            font=dict(color="orange"),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=row, col=col)
        fig.update_yaxes(title_text="Clustering Coefficient", row=row, col=col)
    
    def _add_residuals_analysis_fixed(self, fig, times, residuals, row, col):
        """Add residuals analysis (FIXED VERSION)"""
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=residuals,
                mode='markers',
                marker=dict(size=3, color=self.colors[5], opacity=0.6),
                name='Residuals',
                hovertemplate='Time: %{x:.2f}s<br>Residual: %{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # FIXED: Add zero line using shapes
        fig.add_shape(
            type="line",
            x0=times[0], x1=times[-1],
            y0=0, y1=0,
            line=dict(color="black", dash="dash"),
            row=row, col=col
        )
        
        # Add confidence bands (±2σ)
        residual_std = np.std(residuals)
        fig.add_shape(
            type="line",
            x0=times[0], x1=times[-1],
            y0=2*residual_std, y1=2*residual_std,
            line=dict(color="red", dash="dot"),
            row=row, col=col
        )
        fig.add_shape(
            type="line",
            x0=times[0], x1=times[-1],
            y0=-2*residual_std, y1=-2*residual_std,
            line=dict(color="red", dash="dot"),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=row, col=col)
        fig.update_yaxes(title_text="Residuals", row=row, col=col)
    
    def _add_interarrival_analysis(self, fig, times, row, col):
        """Add inter-arrival time distribution"""
        
        inter_arrivals = np.diff(times)
        
        fig.add_trace(
            go.Histogram(
                x=inter_arrivals,
                nbinsx=30,
                name='Inter-arrival Times',
                marker_color=self.colors[4],
                opacity=0.7,
                hovertemplate='Inter-arrival: %{x:.4f}s<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add exponential fit line (theoretical)
        x_exp = np.linspace(0, np.max(inter_arrivals), 100)
        rate = 1.0 / np.mean(inter_arrivals)
        y_exp = rate * np.exp(-rate * x_exp) * len(inter_arrivals) * (np.max(inter_arrivals) / 30)
        
        fig.add_trace(
            go.Scatter(
                x=x_exp,
                y=y_exp,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Exponential Fit',
                hovertemplate='Time: %{x:.4f}s<br>Expected: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Inter-arrival Time (seconds)", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def _add_autocorrelation_analysis(self, fig, intensities, row, col):
        """Add intensity autocorrelation analysis"""
        
        if len(intensities) > 50:
            max_lag = min(50, len(intensities)//2)
            lags = list(range(max_lag))
            autocorr = []
            
            for lag in lags:
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    corr = np.corrcoef(intensities[:-lag], intensities[lag:])[0,1]
                    autocorr.append(corr if not np.isnan(corr) else 0.0)
            
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=autocorr,
                    mode='lines+markers',
                    line=dict(color=self.colors[6], width=2),
                    marker=dict(size=4),
                    name='Autocorrelation',
                    hovertemplate='Lag: %{x}<br>Correlation: %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # FIXED: Add significance bands using shapes
            significance = 1.96 / np.sqrt(len(intensities))
            fig.add_shape(
                type="line",
                x0=0, x1=max_lag-1,
                y0=significance, y1=significance,
                line=dict(color="red", dash="dash"),
                row=row, col=col
            )
            fig.add_shape(
                type="line",
                x0=0, x1=max_lag-1,
                y0=-significance, y1=-significance,
                line=dict(color="red", dash="dash"),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Lag", row=row, col=col)
        fig.update_yaxes(title_text="Autocorrelation", row=row, col=col)
    
    def _add_performance_metrics(self, fig, results, row, col):
        """Add performance metrics bar chart"""
        
        metrics = ['Processing Time (ms)', 'Throughput (K events/sec)', 'Memory Efficiency']
        values = [
            results['processing_time_ms'],
            results['throughput_events_per_sec'] / 1000,
            85.0  # Placeholder for memory efficiency
        ]
        
        colors = ['blue', 'green', 'orange']
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                name='Performance',
                hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Metrics", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    def _add_burstiness_analysis_fixed(self, fig, times, results, row, col):
        """Add burstiness and memory analysis (FIXED VERSION)"""
        
        # Calculate burstiness over time windows
        window_size = max(50, len(times) // 20)
        burstiness_timeline = []
        time_windows = []
        
        for i in range(0, len(times) - window_size, window_size // 2):
            window_times = times[i:i+window_size]
            if len(window_times) > 10:
                inter_arrivals = np.diff(window_times)
                mean_ia = np.mean(inter_arrivals)
                std_ia = np.std(inter_arrivals)
                burstiness = (std_ia - mean_ia) / (std_ia + mean_ia) if (std_ia + mean_ia) > 0 else 0
                burstiness_timeline.append(burstiness)
                time_windows.append(np.mean(window_times))
        
        if len(burstiness_timeline) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_windows,
                    y=burstiness_timeline,
                    mode='lines+markers',
                    line=dict(color=self.colors[7], width=2),
                    marker=dict(size=4),
                    name='Burstiness',
                    hovertemplate='Time: %{x:.2f}s<br>Burstiness: %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # FIXED: Add overall burstiness line using shapes
            overall_burstiness = results['burstiness']
            if len(time_windows) > 0:
                fig.add_shape(
                    type="line",
                    x0=time_windows[0], x1=time_windows[-1],
                    y0=overall_burstiness, y1=overall_burstiness,
                    line=dict(color="purple", dash="dash"),
                    row=row, col=col
                )
                
                # Add annotation
                fig.add_annotation(
                    x=time_windows[len(time_windows)//2],
                    y=overall_burstiness,
                    text=f"Overall: {overall_burstiness:.4f}",
                    showarrow=False,
                    font=dict(color="purple"),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Time (seconds)", row=row, col=col)
        fig.update_yaxes(title_text="Burstiness Index", row=row, col=col)
    
    def create_simple_intensity_plot(self, results: Dict, events_df: pd.DataFrame, 
                                   ticker: str = "DEFAULT") -> go.Figure:
        """
        Create a simple intensity plot (SAFE VERSION)
        
        Args:
            results: Hawkes analysis results
            events_df: Original events DataFrame
            ticker: Stock ticker symbol
            
        Returns:
            Simple Plotly Figure
        """
        
        times = (events_df['timestamp_ns'].values - events_df['timestamp_ns'].iloc[0]) * 1e-9
        intensities = np.array(results['intensities'])
        
        fig = go.Figure()
        
        # Add intensity line
        fig.add_trace(
            go.Scatter(
                x=times,
                y=intensities,
                mode='lines',
                name='Hawkes Intensity λ(t)',
                line=dict(color='blue', width=2),
                hovertemplate='Time: %{x:.2f}s<br>Intensity: %{y:.4f}<extra></extra>'
            )
        )
        
        # Add events as markers
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.ones(len(times)) * np.max(intensities) * 0.1,
                mode='markers',
                name='Events',
                marker=dict(size=3, color='red', opacity=0.6),
                hovertemplate='Time: %{x:.2f}s<br>Event<extra></extra>'
            )
        )
        
        # Add parameter annotations
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"μ = {results['mu']:.4f}<br>α = {results['alpha']:.4f}<br>β = {results['beta']:.4f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f"Hawkes Process Analysis - {ticker}",
            xaxis_title="Time (seconds)",
            yaxis_title="Intensity",
            **self.default_layout
        )
        
        return fig
    
    def create_parameter_summary_plot(self, results: Dict, ticker: str = "DEFAULT") -> go.Figure:
        """
        Create parameter summary visualization
        
        Args:
            results: Hawkes analysis results
            ticker: Stock ticker symbol
            
        Returns:
            Parameter summary Figure
        """
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Parameter Values",
                "Model Quality Metrics", 
                "Performance Metrics",
                "Stability Analysis"
            ],
            specs=[
                [{"type": "bar"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Parameter Values
        params = ['μ (Base)', 'α (Jump)', 'β (Decay)']
        values = [results['mu'], results['alpha'], results['beta']]
        colors = ['blue', 'red', 'green']
        
        fig.add_trace(
            go.Bar(x=params, y=values, marker_color=colors, name='Parameters'),
            row=1, col=1
        )
        
        # 2. Model Quality (AIC)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results['aic'],
                title={'text': "AIC (lower is better)"},
                gauge={
                    'axis': {'range': [None, results['aic'] * 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, results['aic'] * 0.8], 'color': "lightgreen"},
                        {'range': [results['aic'] * 0.8, results['aic'] * 1.2], 'color': "lightyellow"},
                        {'range': [results['aic'] * 1.2, results['aic'] * 1.5], 'color': "lightcoral"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # 3. Performance Metrics
        perf_metrics = ['Processing Time (ms)', 'Throughput (K/sec)']
        perf_values = [results['processing_time_ms'], results['throughput_events_per_sec'] / 1000]
        
        fig.add_trace(
            go.Bar(x=perf_metrics, y=perf_values, marker_color=['orange', 'purple'], name='Performance'),
            row=2, col=1
        )
        
        # 4. Stability (Branching Ratio)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results['branching_ratio'],
                title={'text': "Branching Ratio (< 1 = stable)"},
                gauge={
                    'axis': {'range': [0, 1.5]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 1.5], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Hawkes Process Parameter Summary - {ticker}",
            height=800,
            **self.default_layout
        )
        
        return fig

# Example usage and testing
def main():
    """Example usage of the HawkesPlotlyVisualizer (FIXED VERSION)"""
    
    # Sample data (replace with actual C++/CUDA output)
    np.random.seed(42)
    n_events = 1000
    
    # Simulate events DataFrame
    start_time = int(1640995200 * 1e9)  # 2022-01-01 in nanoseconds
    timestamps = np.sort(start_time + np.cumsum(np.random.exponential(1e6, n_events)))
    
    events_df = pd.DataFrame({
        'timestamp_ns': timestamps,
        'price': 100 + np.cumsum(np.random.normal(0, 0.01, n_events)),
        'size': np.random.exponential(100, n_events),
        'exchange_id': np.random.randint(0, 4, n_events)
    })
    
    # Simulate Hawkes analysis results (replace with actual C++/CUDA output)
    results = {
        'mu': 0.15,
        'alpha': 0.35,
        'beta': 1.2,
        'branching_ratio': 0.292,
        'log_likelihood': -1250.5,
        'aic': 2507.0,
        'bic': 2522.1,
        'processing_time_ms': 2.5,
        'throughput_events_per_sec': 400000,
        'processing_method': 'GPU + SIMD',
        'is_subcritical': True,
        'mean_intensity': 0.25,
        'max_intensity': 1.8,
        'intensity_variance': 0.12,
        'clustering_coefficient': 0.45,
        'burstiness': 0.32,
        'memory_coefficient': 0.28,
        'criticality_index': 0.708,
        'goodness_of_fit': 0.085,
        'intensities': 0.1 + 0.3 * np.random.exponential(1, n_events),
        'clustering_coefficients': np.random.beta(2, 3, n_events),
        'residuals': np.random.normal(0, 0.5, n_events)
    }
    
    # Create visualizer
    visualizer = HawkesPlotlyVisualizer(theme="plotly_dark")
    
    # Create comprehensive dashboard (FIXED VERSION)
    print("Creating comprehensive Hawkes analysis dashboard...")
    dashboard_fig = visualizer.create_comprehensive_dashboard(
        results, events_df, ticker="AAPL", 
        save_html="hawkes_dashboard_fixed.html"
    )
    
    # Create simple intensity plot (SAFE VERSION)
    print("Creating simple intensity plot...")
    simple_fig = visualizer.create_simple_intensity_plot(results, events_df, ticker="AAPL")
    simple_fig.write_html("hawkes_simple_plot.html")
    
    # Create parameter summary
    print("Creating parameter summary...")
    param_fig = visualizer.create_parameter_summary_plot(results, ticker="AAPL")
    param_fig.write_html("hawkes_parameter_summary.html")
    
    print("✓ All Plotly visualizations created successfully!")
    print("Files generated:")
    print("  - hawkes_dashboard_fixed.html (comprehensive analysis)")
    print("  - hawkes_simple_plot.html (simple intensity plot)")
    print("  - hawkes_parameter_summary.html (parameter summary)")
    
    # Show the main dashboard
    dashboard_fig.show()

if __name__ == "__main__":
    main()

