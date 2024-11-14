import streamlit as st
import pandas as pd
import numpy as np    
import shap    
import plotly.graph_objects as go    
from utils import *  # This will import all functions from utils.py
import io
import base64    
import kaleido

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'filename' not in st.session_state:
    st.session_state['filename'] = None

# Page configuration
st.set_page_config(
    page_title="SHAP Plot Generator",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stAlert {border-radius: 10px;}
    .stButton>button {border-radius: 5px;}
    .uploaded-file {margin-bottom: 2rem;}
    .tooltip {position: relative; display: inline-block;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 SHAP Plot Generator")
    st.markdown("""
        This tool helps you generate SHAP (SHapley Additive exPlanations) plots 
        to understand your model's predictions.
        
        ### How to use:
        1. Upload your dataset or generate synthetic data
        2. Preview your data
        3. Configure plot settings
        4. Generate SHAP plots
        5. Download visualizations
    """)
    
    # Data loading options
    st.subheader("Data Options")
    
    # Add sample data option
    if st.button("Load Sample Data"):
        try:
            df = pd.read_csv("sample_data.csv")
            st.session_state['data'] = df
            st.session_state['filename'] = "sample_data.csv"
        except FileNotFoundError:
            st.error("Sample data file not found. Please make sure 'sample_data.csv' exists in the project directory.")
        except pd.errors.EmptyDataError:
            st.error("The sample data file is empty.")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
    
    # Synthetic data options
    st.markdown("### Generate Synthetic Data")
    st.markdown("""
        Create a synthetic dataset with:
        - Features from different distributions
        - Mix of linear and non-linear relationships
        - Realistic noise in the target variable
    """)
    
    n_samples = st.slider(
        "Number of samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Choose the number of data points to generate"
    )
    
    n_features = st.slider(
        "Number of features",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Choose the number of features to generate"
    )
    
    if st.button("Generate Synthetic Data"):
        try:
            df = generate_synthetic_data(n_samples=n_samples, n_features=n_features)
            st.session_state['data'] = df
            st.session_state['filename'] = "synthetic_data.csv"
            st.success("Synthetic data generated successfully!")
        except Exception as e:
            st.error(f"Error generating synthetic data: {str(e)}")

# Main content
st.title("SHAP Plot Generator")

# File upload
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)", 
    type=['csv', 'xlsx'],
    help="Upload a CSV or Excel file containing your feature data"
)

if uploaded_file is not None or 'data' in st.session_state:
    try:
        if uploaded_file is not None:
            df = validate_file(uploaded_file)
            st.session_state['data'] = df
            st.session_state['filename'] = uploaded_file.name
        
        # Check if data exists in session state
        if 'data' in st.session_state and st.session_state['data'] is not None:
            df = st.session_state['data']
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Add tabs for different visualizations
            viz_tabs = st.tabs(["SHAP Plot", "Feature Importance"])
            
            with viz_tabs[0]:
                # Settings
                with st.expander("Plot Settings", expanded=True):
                    # Plot Type Selection
                    st.subheader("Plot Type")
                    plot_type = st.radio(
                        "Select plot type",
                        ["Interactive Plotly", "Native SHAP Beeswarm"],
                        help="Choose between interactive Plotly visualization or native SHAP beeswarm plot"
                    )
                    
                    # Model Settings
                    st.subheader("Model Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        use_custom_model = st.checkbox(
                            "Use Custom Model",
                            value=False,
                            help="Upload your own pre-trained model"
                        )
                        
                        if use_custom_model:
                            model_file = st.file_uploader(
                                "Upload Model File",
                                type=['pkl', 'pickle', 'joblib'],
                                help="Upload your pre-trained model (pickle or joblib format)"
                            )
                            
                            if model_file is not None:
                                try:
                                    custom_model = load_model(model_file)
                                    st.success("Model loaded successfully!")
                                except Exception as e:
                                    st.error(f"Error loading model: {str(e)}")
                                    custom_model = None
                            else:
                                custom_model = None
                        
                        if not use_custom_model:
                            model_type = st.selectbox(
                                "Select model type",
                                ['random_forest', 'linear', 'svm', 'xgboost'],
                                help="Choose the type of model to use for SHAP calculation"
                            )
                        
                        task_type = st.selectbox(
                            "Select task type",
                            ['regression', 'classification'],
                            help="Choose between regression or classification task"
                        )
                        
                        target_column = st.selectbox(
                            "Select target column",
                            df.columns,
                            help="Choose the column you want to predict"
                        )
                    
                    with col2:
                        n_samples = st.slider(
                            "Number of samples for SHAP calculation",
                            min_value=50,
                            max_value=1000,
                            value=50,
                            help="More samples = more accurate but slower. Use fewer samples for faster results."
                        )
                        
                        if n_samples > 100:
                            st.warning("Using more than 100 samples may slow down the calculation significantly.")
                        
                        title = st.text_input(
                            "Plot Title",
                            value="SHAP Values Distribution",
                            help="Enter a custom title for the plot",
                            key="plot_title"
                        )
                        
                        show_confidence = st.checkbox(
                            "Show Confidence Intervals",
                            value=False,
                            help="Display uncertainty in SHAP values"
                        )
                    
                    fast_mode = st.checkbox(
                        "Fast Mode",
                        value=False,
                        help="Enable fast mode for quicker but less detailed plots"
                    )
                    
                    if fast_mode:
                        # Use smaller sample size
                        n_samples = min(50, n_samples)
                        # Disable confidence intervals
                        show_confidence = False
                    
                    if plot_type == "Interactive Plotly":
                        # Basic Settings
                        st.subheader("Basic Settings")
                        col1, col2 = st.columns(2)
                        with col2:
                            plot_width = st.slider(
                                "Plot Width",
                                min_value=600,
                                max_value=1600,
                                value=800,
                                step=50,
                                help="Adjust the width of the plot"
                            )
                            
                            plot_height = st.slider(
                                "Plot Height",
                                min_value=400,
                                max_value=1200,
                                value=600,
                                step=50,
                                help="Adjust the height of the plot"
                            )
                            
                            plot_layout = st.selectbox(
                                "Plot Layout",
                                ["vertical", "horizontal"],
                                help="Choose the orientation of the plot"
                            )
                        
                        # Visual Settings
                        st.subheader("Visual Customization")
                        col5, col6 = st.columns(2)
                        with col5:
                            color_scheme = st.selectbox(
                                "Color Scheme",
                                ["RdBu", "Viridis", "Plasma", "Blues", "Reds", "Custom"],
                                help="Choose the color scheme for the plot"
                            )
                            
                            if color_scheme == "Custom":
                                positive_color = st.color_picker(
                                    "Positive Values Color",
                                    "#2E86C1",
                                    help="Choose color for positive SHAP values"
                                )
                                negative_color = st.color_picker(
                                    "Negative Values Color",
                                    "#E74C3C",
                                    help="Choose color for negative SHAP values"
                                )
                            
                            background_color = st.color_picker(
                                "Background Color",
                                "#FFFFFF",
                                help="Choose plot background color"
                            )
                        
                        with col6:
                            marker_size = st.slider(
                                "Marker Size",
                                min_value=4,
                                max_value=20,
                                value=8,
                                help="Adjust the size of data points"
                            )
                            
                            marker_opacity = st.slider(
                                "Marker Opacity",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.6,
                                step=0.1,
                                help="Adjust the transparency of data points"
                            )
                        
                        # Grid Settings
                        st.subheader("Grid Settings")
                        col13, col14 = st.columns(2)
                        with col13:
                            show_grid = st.checkbox("Show Grid", value=True)
                            if show_grid:
                                grid_style = st.selectbox(
                                    "Grid Line Style",
                                    ["solid", "dashed", "dotted", "dashdot"],
                                    help="Choose the style of grid lines"
                                )
                                grid_width = st.slider(
                                    "Grid Line Width",
                                    min_value=1,
                                    max_value=5,
                                    value=1,
                                    help="Adjust the width of grid lines"
                                )
                                grid_color = st.color_picker(
                                    "Grid Color",
                                    "#D3D3D3",
                                    help="Choose the color of grid lines"
                                )
                    else:
                        # Native SHAP Plot Settings
                        st.subheader("Native SHAP Plot Settings")
                        col1, col2 = st.columns(2)
                        with col1:
                            figsize_width = st.slider(
                                "Figure Width",
                                min_value=6,
                                max_value=20,
                                value=10,
                                help="Width of the figure in inches"
                            )
                            figsize_height = st.slider(
                                "Figure Height",
                                min_value=4,
                                max_value=16,
                                value=8,
                                help="Height of the figure in inches"
                            )
                            
                            # Add SHAP plot type selection
                            shap_plot_type = st.selectbox(
                                "SHAP Plot Type",
                                ["beeswarm", "bar", "violin", "summary"],
                                help="Choose the type of SHAP plot to display"
                            )
                        
                        with col2:
                            max_display = st.number_input(
                                "Maximum Features to Display",
                                min_value=1,
                                max_value=len(df.columns)-1,
                                value=min(20, len(df.columns)-1),
                                help="Maximum number of features to show in the plot"
                            )
                            plot_size = st.slider(
                                "Plot Size",
                                min_value=0.1,
                                max_value=2.0,
                                value=0.5,
                                step=0.1,
                                help="Size multiplier for the plot"
                            )
                
                # Feature Label Settings
                with st.expander("Feature Label Settings", expanded=False):
                    st.info("Edit feature labels that will appear in the plot. Leave blank to keep original names.")
                    
                    # Create a dictionary to store original and new labels
                    feature_labels = {}
                    
                    # Create two columns for better layout
                    label_cols = st.columns(2)
                    
                    # Get feature names (excluding target column)
                    feature_names = [col for col in df.columns if col != target_column]
                    
                    # Create input fields for each feature
                    for i, feature in enumerate(feature_names):
                        col_idx = i % 2  # Alternate between columns
                        with label_cols[col_idx]:
                            new_label = st.text_input(
                                f"Label for {feature}",
                                value="",
                                key=f"label_{feature}",
                                help=f"Enter new label for {feature}"
                            )
                            if new_label:
                                feature_labels[feature] = new_label
                            else:
                                feature_labels[feature] = feature
                
                # Add performance options in the Plot Settings
                with st.expander("Performance Options", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        fast_mode = st.checkbox("Fast Mode", value=False)
                        approximate_shap = st.checkbox("Use Approximate SHAP", value=False)
                    with col2:
                        simplified_plot = st.checkbox("Simplified Plot (Top 10 features)", value=False)
                        use_parallel = st.checkbox("Use Parallel Processing", value=True)
                    
                    if fast_mode or approximate_shap:
                        st.info("Using performance optimizations may slightly reduce accuracy but will be much faster.")
                
                # Generate plot button
                if st.button("Generate SHAP Plot", key="generate_plot"):
                    try:
                        with st.spinner("Calculating SHAP values..."):
                            # Prepare data
                            X = df.drop(columns=[target_column])
                            y = df[target_column]
                            
                            # Set default values for all plot parameters
                            plot_defaults = {
                                'plot_height': 600,
                                'plot_width': 800,
                                'marker_size': 8,
                                'marker_opacity': 0.6,
                                'color_scheme': "RdBu",
                                'show_grid': True,
                                'grid_style': "solid",
                                'grid_width': 1,
                                'grid_color': "#D3D3D3",
                                'background_color': "#FFFFFF",
                                'positive_color': "#2E86C1",
                                'negative_color': "#E74C3C",
                                'show_confidence': False,
                                'simplified_plot': False,
                                'approximate_shap': False
                            }
                            
                            # Update defaults with actual values if they exist
                            for param, default in plot_defaults.items():
                                if param not in locals():
                                    locals()[param] = default
                            
                            # Create plot parameters dictionary
                            plot_params = {
                                'height': plot_height,
                                'width': plot_width,
                                'marker_size': marker_size,
                                'marker_opacity': marker_opacity,
                                'colorscale': color_scheme,
                                'show_grid': show_grid,
                                'background_color': background_color,
                                'grid_style': {
                                    'style': grid_style,
                                    'width': grid_width,
                                    'color': grid_color
                                } if show_grid else None,
                                'custom_colors': {
                                    'positive': positive_color,
                                    'negative': negative_color
                                } if color_scheme == "Custom" else None
                            }
                            
                            # Calculate SHAP values
                            shap_values, std_vals = calculate_shap_values(
                                X=X,
                                y=y,
                                n_samples=n_samples if not fast_mode else min(50, n_samples),
                                model_type=model_type,
                                task_type=task_type,
                                approximate=approximate_shap
                            )
                            
                            # Create plot
                            if plot_type == "Native SHAP Beeswarm":
                                plot_bytes = create_native_shap_plot(
                                    shap_values=shap_values,
                                    X=X,
                                    title=title,
                                    figsize=(figsize_width, figsize_height),
                                    max_display=max_display,
                                    plot_size=(plot_size * 8, plot_size * 12),
                                    feature_labels=feature_labels,
                                    plot_type=shap_plot_type  # Add this parameter
                                )
                                
                                # Display plot
                                st.image(plot_bytes, use_container_width=True)
                                
                                # Add download button
                                st.download_button(
                                    label="Download Plot as PNG",
                                    data=plot_bytes,
                                    file_name="shap_plot.png",
                                    mime="image/png"
                                )
                            else:
                                fig = create_shap_plot(
                                    shap_values=shap_values,
                                    X_sample=X,
                                    std_vals=std_vals if show_confidence else None,
                                    title=title,
                                    simplified=simplified_plot,
                                    feature_labels=feature_labels,
                                    **plot_params
                                )
                                
                                # Display plot
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add download buttons for plot
                                col_download1, col_download2 = st.columns(2)
                                with col_download1:
                                    # Download plot as HTML
                                    buffer = io.StringIO()
                                    fig.write_html(buffer)
                                    html_bytes = buffer.getvalue().encode()
                                    st.download_button(
                                        label="Download Plot as HTML",
                                        data=html_bytes,
                                        file_name="shap_plot.html",
                                        mime="text/html"
                                    )
                                
                                with col_download2:
                                    # Download plot as PNG
                                    img_bytes = fig.to_image(format="png", scale=2)
                                    st.download_button(
                                        label="Download Plot as PNG",
                                        data=img_bytes,
                                        file_name="shap_plot.png",
                                        mime="image/png"
                                    )
                    
                    except Exception as e:
                        st.error(f"Error generating plot: {str(e)}")
            
            with viz_tabs[1]:
                # Feature Importance Settings
                st.subheader("Feature Importance Settings")
                importance_display = st.radio(
                    "Display Format",
                    ["Table", "Bar Chart"],
                    help="Choose how to display feature importance"
                )
                
                show_relative = st.checkbox(
                    "Show Relative Importance (%)",
                    value=True,
                    help="Display importance values as percentages"
                )
                
                if importance_display == "Bar Chart":
                    chart_orientation = st.radio(
                        "Chart Orientation",
                        ["Horizontal", "Vertical"],
                        help="Choose the orientation of the bar chart"
                    )
                    
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        ["blues", "greens", "reds", "purples", "oranges"],
                        help="Choose the color scheme for the bar chart",
                        key="importance_color_scheme"
                    )
                
                # Generate Feature Importance button
                if st.button("Calculate Feature Importance", key="generate_importance"):
                    try:
                        with st.spinner("Calculating feature importance..."):
                            # Prepare data
                            X = df.drop(columns=[target_column])
                            y = df[target_column]
                            
                            # Calculate SHAP values
                            shap_values, _ = calculate_shap_values(
                                X=X,
                                y=y,
                                n_samples=n_samples,
                                model_type=model_type,
                                task_type=task_type
                            )
                            
                            # Calculate feature importance
                            importance_df = calculate_feature_importance(shap_values, X.columns)
                            
                            # Display results
                            if importance_display == "Table":
                                st.dataframe(
                                    importance_df.style.background_gradient(
                                        subset=['Importance'],
                                        cmap='Blues'
                                    ),
                                    use_container_width=True
                                )
                                
                                # Add download button for CSV
                                csv = importance_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Feature Importance (CSV)",
                                    data=csv,
                                    file_name="feature_importance.csv",
                                    mime="text/csv"
                                )
                            else:
                                # Create bar chart using plotly
                                importance_col = 'Relative Importance (%)' if show_relative else 'Importance'
                                fig = go.Figure()
                                
                                if chart_orientation == "Horizontal":
                                    fig.add_trace(go.Bar(
                                        y=importance_df['Feature'],
                                        x=importance_df[importance_col],
                                        orientation='h',
                                        marker_color=np.linspace(0, 1, len(importance_df)),
                                        marker_colorscale=color_scheme
                                    ))
                                    
                                    fig.update_layout(
                                        xaxis_title=importance_col,
                                        yaxis_title="Feature",
                                        height=max(400, len(importance_df) * 25)
                                    )
                                else:
                                    fig.add_trace(go.Bar(
                                        x=importance_df['Feature'],
                                        y=importance_df[importance_col],
                                        marker_color=np.linspace(0, 1, len(importance_df)),
                                        marker_colorscale=color_scheme
                                    ))
                                    
                                    fig.update_layout(
                                        xaxis_title="Feature",
                                        yaxis_title=importance_col,
                                        height=500
                                    )
                                
                                fig.update_layout(
                                    title="Feature Importance",
                                    showlegend=False,
                                    plot_bgcolor='white'
                                )
                                
                                # Display plot
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add download buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    # Download plot as HTML
                                    buffer = io.StringIO()
                                    fig.write_html(buffer)
                                    html_bytes = buffer.getvalue().encode()
                                    st.download_button(
                                        label="Download Plot as HTML",
                                        data=html_bytes,
                                        file_name="feature_importance_plot.html",
                                        mime="text/html"
                                    )
                                
                                with col2:
                                    # Download plot as PNG
                                    img_bytes = fig.to_image(format="png", scale=2)
                                    st.download_button(
                                        label="Download Plot as PNG",
                                        data=img_bytes,
                                        file_name="feature_importance_plot.png",
                                        mime="image/png"
                                    )
                    
                    except Exception as e:
                        st.error(f"Error calculating feature importance: {str(e)}")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("""
    ---
    Made with ❤️ using Streamlit | [Documentation](https://github.com/slundberg/shap)
""")
