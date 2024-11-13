import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb
import plotly.graph_objects as go
from typing import Tuple, Union, Dict, Any, Optional, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 4) -> pd.DataFrame:
    """Generate synthetic data with different distributions and relationships."""
    np.random.seed(42)
    
    # Generate features with different distributions
    features = {
        'normal': np.random.normal(0, 1, n_samples),
        'uniform': np.random.uniform(-2, 2, n_samples),
        'exponential': np.random.exponential(1, n_samples),
        'lognormal': np.random.lognormal(0, 0.5, n_samples)
    }
    
    # Create feature DataFrame
    df = pd.DataFrame(features)
    
    # Generate target with mix of linear and non-linear relationships
    target = (
        2.0 * df['normal']  # Linear relationship
        - 1.5 * df['uniform']  # Linear relationship
        + 0.5 * np.square(df['exponential'])  # Non-linear (quadratic)
        + np.sin(df['lognormal'])  # Non-linear (sinusoidal)
        + np.random.normal(0, 0.1, n_samples)  # Random noise
    )
    
    # Add target to DataFrame
    df['target'] = target
    
    # Rename columns to be more descriptive
    df.columns = [f'feature{i+1}' if i < n_features else 'target' 
                 for i in range(len(df.columns))]
    
    return df

def validate_file(uploaded_file) -> pd.DataFrame:
    """Validate and load the uploaded file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Basic validation
        if df.empty:
            raise ValueError("The uploaded file is empty")
        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names found")
        
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def get_model(model_type: str, task_type: str):
    """Get the appropriate model based on model_type and task_type."""
    models = {
        'random_forest': {
            'regression': RandomForestRegressor(n_estimators=100, random_state=42),
            'classification': RandomForestClassifier(n_estimators=100, random_state=42)
        },
        'linear': {
            'regression': LinearRegression(),
            'classification': LogisticRegression(random_state=42)
        },
        'svm': {
            'regression': SVR(kernel='rbf'),
            'classification': SVC(kernel='rbf', probability=True)
        },
        'xgboost': {
            'regression': xgb.XGBRegressor(random_state=42),
            'classification': xgb.XGBClassifier(random_state=42)
        }
    }
    return models[model_type][task_type]

def calculate_shap_values(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    model_type: str = 'random_forest',
    task_type: str = 'regression'
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate SHAP values using the specified model type."""
    # Determine if classification task needs label encoding
    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Get and train the appropriate model
    model = get_model(model_type, task_type)
    model.fit(X, y)
    
    # Calculate SHAP values based on model type
    if model_type in ['random_forest', 'xgboost']:
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X)
    else:  # SVM
        explainer = shap.KernelExplainer(model.predict_proba if task_type == 'classification' else model.predict, X)
    
    # Sample data if needed
    if n_samples < len(X):
        X_sample = X.sample(n=n_samples, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        std_vals = np.std([explainer.shap_values(X_sample) for _ in range(5)], axis=0)
    else:
        shap_values = explainer.shap_values(X)
        std_vals = np.std([explainer.shap_values(X) for _ in range(5)], axis=0)
    
    # Handle different output formats
    if task_type == 'classification' and isinstance(shap_values, list):
        # For binary classification, take the values for class 1
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        std_vals = std_vals[1] if len(std_vals.shape) > 2 else std_vals
    
    return shap_values, std_vals

def cluster_features(shap_values: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Cluster features based on their SHAP value patterns."""
    feature_patterns = shap_values.T  # Transpose to get feature patterns
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(feature_patterns)

def create_shap_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    std_vals: np.ndarray = None,
    title: str = "SHAP Values Distribution",
    height: int = 600,
    width: int = 800,
    marker_size: int = 8,
    marker_opacity: float = 0.6,
    colorscale: str = "RdBu",
    custom_colors: Dict[str, str] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    legend_position: str = "right",
    show_confidence: bool = False,
    cluster_features: bool = False,
    n_clusters: int = 3,
    layout: str = "vertical",
    axis_config: Dict[str, Any] = None,
    background_color: str = "white",
    font_sizes: Dict[str, int] = None,
    grid_style: Dict[str, Any] = None,
    zoom_level: float = None,
    hover_template: str = None
) -> go.Figure:
    """Create an interactive SHAP plot using Plotly with enhanced customization options."""
    feature_names = X.columns
    
    # Calculate feature importance and sort features
    feature_importance = np.abs(shap_values).mean(0)
    feature_order = np.argsort(feature_importance)
    
    # Apply feature clustering if requested
    cluster_labels = None
    if cluster_features:
        cluster_labels = cluster_features(shap_values, n_clusters)
    
    # Create figure
    fig = go.Figure()
    
    # Set default font sizes if not provided
    default_font_sizes = {
        'title': 20,
        'axis_title': 16,
        'axis_text': 12,
        'legend': 12
    }
    font_sizes = font_sizes or default_font_sizes
    
    # Set default grid style if not provided
    default_grid_style = {
        'style': 'solid',
        'width': 1,
        'color': 'LightGray'
    }
    grid_style = grid_style or default_grid_style
    
    # Set default hover template if not provided
    default_hover_template = (
        "<b>Feature:</b> %{y}<br>" +
        "<b>SHAP value:</b> %{x:.4f}<br>" +
        "<extra></extra>"
    )
    hover_template = hover_template or default_hover_template
    
    # Add scatter plots for each feature
    for idx in feature_order:
        # Prepare error bars for confidence intervals
        error_x = None
        if show_confidence and std_vals is not None:
            error_x = dict(
                type='data',
                array=std_vals[:, idx],
                visible=True,
                color='rgba(0,0,0,0.3)'
            )
        
        # Set colors based on custom colors if provided
        marker_color = shap_values[:, idx]
        if custom_colors:
            marker_color = [custom_colors['positive'] if v > 0 else custom_colors['negative'] 
                          for v in shap_values[:, idx]]
        
        fig.add_trace(go.Scatter(
            x=shap_values[:, idx],
            y=[feature_names[idx]] * len(shap_values),
            mode='markers',
            name=feature_names[idx],
            error_x=error_x,
            hovertemplate=hover_template,
            marker=dict(
                size=marker_size,
                opacity=marker_opacity,
                color=marker_color,
                colorscale=colorscale if not custom_colors else None,
                showscale=True if idx == feature_order[-1] and not custom_colors else False
            ),
            showlegend=show_legend
        ))
    
    # Update layout with enhanced customization options
    layout_config = {
        'title': {
            'text': title,
            'font': {'size': font_sizes['title']}
        },
        'xaxis': {
            'title': {
                'text': axis_config.get('xaxis_title', "SHAP value (impact on model output)"),
                'font': {'size': font_sizes['axis_title']}
            },
            'tickfont': {'size': font_sizes['axis_text']},
            'tickformat': axis_config.get('xaxis_tickformat', ''),
            'range': axis_config.get('xaxis_range', None),
            'showgrid': show_grid,
            'gridwidth': grid_style['width'],
            'gridcolor': grid_style['color'],
            'griddash': grid_style['style'],
            'range': None if zoom_level is None else [-zoom_level, zoom_level]
        },
        'yaxis': {
            'title': {
                'text': axis_config.get('yaxis_title', "Feature"),
                'font': {'size': font_sizes['axis_title']}
            },
            'tickfont': {'size': font_sizes['axis_text']},
            'tickformat': axis_config.get('yaxis_tickformat', ''),
            'categoryorder': 'array',
            'categoryarray': feature_names[feature_order],
            'showgrid': show_grid,
            'gridwidth': grid_style['width'],
            'gridcolor': grid_style['color'],
            'griddash': grid_style['style']
        },
        'height': height,
        'width': width,
        'plot_bgcolor': background_color,
        'paper_bgcolor': background_color,
        'hovermode': 'closest',
        'showlegend': show_legend,
        'legend': {
            'font': {'size': font_sizes['legend']},
            'yanchor': "middle",
            'y': 0.5,
            'xanchor': "right" if legend_position == "right" else "center",
            'x': 1.1 if legend_position == "right" else 0.5,
            'orientation': "v" if layout == "vertical" else "h"
        }
    }
    
    fig.update_layout(**layout_config)
    
    return fig

def process_batch_files(
    files: List[Any],
    settings: Dict[str, Any],
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """Process multiple files with shared settings and return results.
    
    Args:
        files: List of uploaded files
        settings: Dictionary containing shared processing settings
        progress_callback: Optional callback function for progress updates
    
    Returns:
        List of dictionaries containing results for each file
    """
    results = []
    n_files = len(files)
    
    for idx, file in enumerate(files):
        try:
            # Update progress if callback provided
            if progress_callback:
                progress_callback(idx / n_files, f"Processing {file.name}")
            
            # Load and validate file
            df = validate_file(file)
            
            # Extract features and target
            X = df.drop(columns=[settings['target_column']])
            y = df[settings['target_column']]
            
            # Calculate SHAP values
            shap_values, std_vals = calculate_shap_values(
                X=X,
                y=y,
                n_samples=settings['n_samples'],
                model_type=settings['model_type'],
                task_type=settings['task_type']
            )
            
            # Create plot with shared settings
            fig = create_shap_plot(
                shap_values=shap_values,
                X=X,
                std_vals=std_vals if settings.get('show_confidence') else None,
                title=f"{settings['title']} - {file.name}",
                height=settings['plot_height'],
                width=settings['plot_width'],
                marker_size=settings['marker_size'],
                marker_opacity=settings['marker_opacity'],
                colorscale=settings['color_scheme'] if settings['color_scheme'] != "Custom" else None,
                custom_colors=settings.get('custom_colors'),
                show_grid=settings['show_grid'],
                show_legend=settings['show_legend'],
                legend_position=settings['legend_position'],
                show_confidence=settings.get('show_confidence', False),
                cluster_features=settings.get('enable_clustering', False),
                n_clusters=settings.get('n_clusters', 3),
                layout=settings['plot_layout'],
                axis_config=settings['axis_config'],
                background_color=settings['background_color'],
                font_sizes=settings['font_sizes'],
                grid_style=settings['grid_style'],
                zoom_level=settings.get('zoom_level'),
                hover_template=settings.get('hover_template')
            )
            
            results.append({
                'filename': file.name,
                'figure': fig,
                'shap_values': shap_values,
                'feature_names': X.columns.tolist(),
                'success': True,
                'error': None
            })
            
        except Exception as e:
            results.append({
                'filename': file.name,
                'figure': None,
                'shap_values': None,
                'feature_names': None,
                'success': False,
                'error': str(e)
            })
        
        # Final progress update
        if progress_callback and idx == n_files - 1:
            progress_callback(1.0, "Processing complete")
    
    return results