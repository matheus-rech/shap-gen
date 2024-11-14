import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io
from typing import Tuple, Union, Dict, Any, Optional, List
import streamlit as st
import pickle
import joblib
from joblib import Parallel, delayed

def cluster_features_fn(shap_values: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Cluster features based on their SHAP value patterns."""
    feature_patterns = shap_values.T  # Transpose to get features as samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_patterns)
    return cluster_labels

@st.cache_resource
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

@st.cache_data
def calculate_shap_values(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    model_type: str = 'random_forest',
    task_type: str = 'regression',
    approximate: bool = False
) -> Tuple[shap.Explanation, Optional[np.ndarray]]:
    """Calculate SHAP values using parallel processing."""
    # Label encoding for classification tasks
    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Get and train the model on the full dataset
    model = get_model(model_type, task_type)
    
    # Sample data for SHAP calculation if necessary
    if n_samples < len(X):
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X.copy()
    
    # Add progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        progress_bar.progress(0.2)
        status_text.text("Training model...")
        
        # Model training
        model.fit(X, y)
        
        progress_bar.progress(0.5)
        status_text.text("Calculating SHAP values...")
        
        # Calculate SHAP values
        if model_type in ['random_forest', 'xgboost']:
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation='interventional',
                approximate=approximate
            )
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X)
        else:  # SVM and others
            background = shap.kmeans(X, 10)
            explainer = shap.KernelExplainer(
                model.predict_proba if task_type == 'classification' else model.predict,
                background
            )
        
        # SHAP calculation with chunking for large datasets
        if len(X) > 10000:  # For large datasets
            chunk_size = min(1000, n_samples)
            chunks = np.array_split(X_sample, len(X_sample) // chunk_size + 1)
            
            # Process chunks
            shap_values_list = []
            for chunk in chunks:
                chunk_values = explainer(chunk)
                shap_values_list.append(chunk_values)
            
            # Combine results
            shap_values = np.concatenate(shap_values_list)
        else:
            shap_values = explainer(X_sample)
        
        progress_bar.progress(1.0)
        status_text.text("Done!")
        
        # Handle classification tasks
        if task_type == 'classification':
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            elif shap_values.values.ndim > 2:
                # For multiclass, select the class with the highest probability
                class_idx = np.argmax(model.predict_proba(X_sample), axis=1)
                shap_values = shap.Explanation(
                    values=np.array([shap_values.values[i, :, class_idx[i]] for i in range(len(X_sample))]),
                    base_values=np.array([shap_values.base_values[i, class_idx[i]] for i in range(len(X_sample))]),
                    data=shap_values.data,
                    feature_names=shap_values.feature_names
                )
        
        # Optional: compute standard deviation (disabled by default)
        std_vals = None
        
        return shap_values, std_vals
    
    finally:
        # Clean up
        progress_bar.empty()
        status_text.empty()

def create_shap_plot(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
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
    hover_template: str = None,
    feature_labels: Dict[str, str] = None,
    simplified: bool = False
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
        # Reorder features based on clusters
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
    
    if simplified:
        # Create simpler version with fewer features
        feature_importance = np.abs(shap_values.values).mean(0)
        top_features_idx = np.argsort(feature_importance)[-10:]  # Show only top 10 features
        
        shap_values = shap.Explanation(
            values=shap_values.values[:, top_features_idx],
            base_values=shap_values.base_values,
            data=shap_values.data[:, top_features_idx],
            feature_names=[feature_names[i] for i in top_features_idx]
        )
    
    # Clear memory after each feature
    import gc
    for idx, feature in enumerate(feature_names):
        # ... plot creation code ...
        gc.collect()  # Force garbage collection
    
    return fig

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

def create_native_shap_plot(
    shap_values: Union[np.ndarray, shap.Explanation],
    X: pd.DataFrame,
    title: str = "SHAP Values Distribution",
    figsize: Tuple[int, int] = (10, 8),
    max_display: int = None,
    plot_size: Tuple[float, float] = (8, 12),
    feature_labels: Dict[str, str] = None,
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

def process_batch_files(
    files: List[Any],
    settings: Dict[str, Any],
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """Process multiple files with shared settings."""
    results = []
    n_files = len(files)
    
    for idx, file in enumerate(files):
        try:
            if progress_callback:
                progress_callback(idx / n_files, f"Processing {file.name}")
            
            df = validate_file(file)
            X = df.drop(columns=[settings['target_column']])
            y = df[settings['target_column']]
            
            shap_values, _ = calculate_shap_values(
                X=X,
                y=y,
                n_samples=settings['n_samples'],
                model_type=settings['model_type'],
                task_type=settings['task_type']
            )
            
            fig = create_shap_plot(
                shap_values=shap_values,
                X=X,
                title=f"{settings['title']} - {file.name}",
                **{k: v for k, v in settings.items() if k not in ['title', 'model_type', 'task_type', 'target_column', 'n_samples']}
            )
            
            results.append({
                'filename': file.name,
                'figure': fig,
                'success': True,
                'error': None
            })
            
        except Exception as e:
            results.append({
                'filename': file.name,
                'figure': None,
                'success': False,
                'error': str(e)
            })
        
        if progress_callback and idx == n_files - 1:
            progress_callback(1.0, "Processing complete")
    
    return results

def load_model(model_file) -> Any:
    """Load a model from a pickle or joblib file."""
    try:
        # Try to determine file type from extension
        file_extension = model_file.name.lower().split('.')[-1]
        
        # Read the file content
        file_content = model_file.read()
        
        if file_extension in ['pkl', 'pickle']:
            # Load pickle file
            model = pickle.loads(file_content)
        elif file_extension in ['joblib']:
            # Load joblib file
            model = joblib.load(io.BytesIO(file_content))
        else:
            raise ValueError("Unsupported file format. Please use .pkl, .pickle, or .joblib files.")
        
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def calculate_shap_values_custom_model(
    model: Any,
    X: pd.DataFrame,
    n_samples: int,
    task_type: str = 'regression'
) -> Tuple[shap.Explanation, Optional[np.ndarray]]:
    """Calculate SHAP values using a custom pre-trained model."""
    # Sample data for SHAP calculation if necessary
    if n_samples < len(X):
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X.copy()
    
    try:
        # Try to determine if it's a tree-based model
        is_tree_model = hasattr(model, 'estimators_') or hasattr(model, 'trees_') or type(model).__name__.lower().find('tree') != -1
        
        if is_tree_model:
            explainer = shap.TreeExplainer(model)
        else:
            # For other model types, use KernelExplainer
            background = shap.kmeans(X, 10)
            predict_fn = (model.predict_proba if task_type == 'classification' and hasattr(model, 'predict_proba') 
                         else model.predict)
            explainer = shap.KernelExplainer(predict_fn, background)
        
        shap_values = explainer(X_sample)
        
        # Handle classification tasks
        if task_type == 'classification':
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            elif hasattr(shap_values, 'values') and shap_values.values.ndim > 2:
                # For multiclass, select the class with the highest probability
                if hasattr(model, 'predict_proba'):
                    class_idx = np.argmax(model.predict_proba(X_sample), axis=1)
                else:
                    class_idx = np.argmax(shap_values.values, axis=2)
                shap_values = shap.Explanation(
                    values=np.array([shap_values.values[i, :, class_idx[i]] for i in range(len(X_sample))]),
                    base_values=np.array([shap_values.base_values[i, class_idx[i]] for i in range(len(X_sample))]),
                    data=shap_values.data,
                    feature_names=shap_values.feature_names
                )
        
        # Optional: compute standard deviation (disabled by default)
        std_vals = None
        
        return shap_values, std_vals
    
    except Exception as e:
        raise ValueError(f"Error calculating SHAP values: {str(e)}")
