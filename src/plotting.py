import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
from typing import Tuple, Union, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

def cluster_features_fn(shap_values: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Cluster features based on their SHAP value patterns."""
    feature_patterns = shap_values.T  # Transpose to get features as samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_patterns)
    return cluster_labels

@st.cache_data
def create_shap_plot(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    std_vals: Optional[np.ndarray] = None,
    title: str = "SHAP Values Distribution",
    height: int = 600,
    width: int = 800,
    marker_size: int = 8,
    marker_opacity: float = 0.6,
    colorscale: str = "RdBu",
    custom_colors: Optional[Dict[str, str]] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    legend_position: str = "right",
    show_confidence: bool = False,
    cluster_features: bool = False,
    n_clusters: int = 3,
    layout: str = "vertical",
    axis_config: Optional[Dict[str, Any]] = None,
    background_color: str = "white",
    font_sizes: Optional[Dict[str, int]] = None,
    grid_style: Optional[Dict[str, Any]] = None,
    zoom_level: Optional[float] = None,
    hover_template: Optional[str] = None,
    feature_labels: Optional[Dict[str, str]] = None
) -> go.Figure:
    """Create an interactive SHAP plot using Plotly."""
    # Validate inputs
    if height <= 0 or width <= 0:
        raise ValueError("Plot dimensions must be positive")
    if marker_size <= 0:
        raise ValueError("Marker size must be positive")
    
    # Use custom labels if provided
    feature_names = [feature_labels.get(col, col) for col in X_sample.columns] if feature_labels else X_sample.columns.tolist()
    
    # Handle feature clustering
    if cluster_features:
        cluster_labels = cluster_features_fn(shap_values.values, n_clusters)
        sorted_indices = np.argsort(cluster_labels)
        shap_values = shap.Explanation(
            values=shap_values.values[:, sorted_indices],
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=[feature_names[i] for i in sorted_indices]
        )
        feature_names = [feature_names[i] for i in sorted_indices]
    else:
        feature_importance = np.abs(shap_values.values).mean(0)
        sorted_indices = np.argsort(feature_importance)
        shap_values = shap.Explanation(
            values=shap_values.values[:, sorted_indices],
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=[feature_names[i] for i in sorted_indices]
        )
        feature_names = [feature_names[i] for i in sorted_indices]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plots for each feature
    for idx, feature in enumerate(feature_names):
        shap_vals = shap_values.values[:, idx]
        
        # Prepare error bars for confidence intervals
        error_x = None
        if show_confidence and std_vals is not None:
            error_x = {
                'type': 'data',
                'array': std_vals[:, idx],
                'visible': True,
                'color': 'rgba(0,0,0,0.3)'
            }
        
        # Set colors
        marker_color = shap_vals
        if custom_colors:
            marker_color = [custom_colors['positive'] if v > 0 else custom_colors['negative'] for v in shap_vals]
        
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=[feature] * len(shap_vals),
            mode='markers',
            name=feature,
            error_x=error_x,
            hovertemplate=hover_template or "%{y}: %{x:.4f}<extra></extra>",
            marker={
                'size': marker_size,
                'opacity': marker_opacity,
                'color': marker_color,
                'colorscale': colorscale if not custom_colors else None,
                'showscale': idx == len(feature_names) - 1
            },
            showlegend=False
        ))
        
        # Clear memory after each feature
        import gc
        gc.collect()  # Force garbage collection
    
    # Update layout
    layout_args = {
        'title': title,
        'height': height,
        'width': width,
        'plot_bgcolor': background_color,
        'paper_bgcolor': background_color,
        'showlegend': show_legend,
        'margin': dict(l=100, r=100, t=50, b=50)
    }
    
    if font_sizes:
        layout_args['font'] = font_sizes
    
    fig.update_layout(**layout_args)
    
    # Update axes
    xaxis_args = {
        'showgrid': show_grid,
        'gridwidth': grid_style.get('width', 1) if grid_style else 1,
        'gridcolor': grid_style.get('color', 'lightgray') if grid_style else 'lightgray',
        'zeroline': True,
        'zerolinewidth': 1,
        'zerolinecolor': 'black'
    }
    
    if grid_style and 'style' in grid_style:
        xaxis_args['griddash'] = grid_style['style']
    
    if zoom_level:
        xaxis_args['range'] = [-zoom_level, zoom_level]
    
    if axis_config:
        xaxis_args.update(axis_config)
    
    fig.update_xaxes(**xaxis_args)
    fig.update_yaxes(showgrid=False)
    
    return fig

def create_native_shap_plot(
    shap_values: Union[np.ndarray, shap.Explanation],
    X: pd.DataFrame,
    title: str = "SHAP Values Distribution",
    figsize: Tuple[int, int] = (10, 8),
    max_display: Optional[int] = None,
    plot_size: Tuple[float, float] = (8, 12),
    feature_labels: Optional[Dict[str, str]] = None,
    plot_type: str = "beeswarm"
) -> bytes:
    """Create a native SHAP plot and return it as bytes."""
    plt.figure(figsize=figsize)
    
    # If custom labels are provided, create a copy of X with renamed columns
    if feature_labels:
        X = X.copy()
        X.columns = [feature_labels.get(col, col) for col in X.columns]
        if isinstance(shap_values, shap.Explanation):
            shap_values.feature_names = [feature_labels.get(name, name) for name in shap_values.feature_names]
    
    try:
        if plot_type == "beeswarm":
            shap.plots.beeswarm(
                shap_values,
                show=False,
                max_display=max_display or len(X.columns),
                plot_size=plot_size
            )
        elif plot_type == "bar":
            shap.plots.bar(
                shap_values,
                show=False,
                max_display=max_display or len(X.columns)
            )
        elif plot_type == "violin":
            shap.plots.violin(
                shap_values,
                show=False,
                max_display=max_display or len(X.columns)
            )
        else:  # Default to summary plot
            shap.summary_plot(
                shap_values.values if isinstance(shap_values, shap.Explanation) else shap_values,
                X,
                show=False,
                max_display=max_display or len(X.columns),
                plot_size=plot_size
            )
    except Exception as e:
        st.error(f"Error creating SHAP plot: {str(e)}")
        # Fallback to summary plot
        shap.summary_plot(
            shap_values.values if isinstance(shap_values, shap.Explanation) else shap_values,
            X,
            show=False,
            max_display=max_display or len(X.columns),
            plot_size=plot_size
        )
    
    plt.title(title)
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    return buf.getvalue()

def calculate_feature_importance(
    shap_values: Union[np.ndarray, shap.Explanation],
    feature_names: List[str]
) -> pd.DataFrame:
    """Calculate feature importance from SHAP values."""
    if isinstance(shap_values, shap.Explanation):
        values = shap_values.values
    else:
        values = shap_values
    
    # Calculate mean absolute SHAP value for each feature
    importances = np.abs(values).mean(axis=0)
    
    # Create DataFrame with feature names and importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Calculate relative importance (percentage)
    importance_df['Relative Importance (%)'] = (
        importance_df['Importance'] / importance_df['Importance'].sum() * 100
    )
    
    # Round numeric columns
    importance_df['Importance'] = importance_df['Importance'].round(4)
    importance_df['Relative Importance (%)'] = importance_df['Relative Importance (%)'].round(2)
    
    return importance_df