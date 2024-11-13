import streamlit as st
import pandas as pd
import numpy as np    
import shap    
import plotly.graph_objects as go    
from utils import calculate_shap_values, create_shap_plot, validate_file, generate_synthetic_data, process_batch_files, create_native_shap_plot
import io
import base64    
import kaleido

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
        df = pd.read_csv("sample_data.csv")
        st.session_state['data'] = df
        st.session_state['filename'] = "sample_data.csv"
    
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
        df = generate_synthetic_data(n_samples=n_samples, n_features=n_features)
        st.session_state['data'] = df
        st.session_state['filename'] = "synthetic_data.csv"

    # Add batch processing section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Batch Processing")
    
    # Batch file upload
    batch_files = st.sidebar.file_uploader(
        "Upload multiple files for batch processing",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        help="Upload multiple CSV or Excel files for batch processing"
    )
    
    # Batch processing settings
    if batch_files:
        st.sidebar.markdown("### Batch Processing Settings")
        
        # Shared settings for batch processing
        batch_model_type = st.sidebar.selectbox(
            "Model Type (Batch)",
            ['random_forest', 'linear', 'svm', 'xgboost'],
            help="Choose the model type for all files",
            key="batch_model_type"
        )
        
        batch_task_type = st.sidebar.selectbox(
            "Task Type (Batch)",
            ['regression', 'classification'],
            help="Choose between regression or classification for all files",
            key="batch_task_type"
        )
        
        batch_target_column = st.sidebar.text_input(
            "Target Column Name (Batch)",
            value="target",
            help="Enter the name of the target column (must be same across all files)",
            key="batch_target_column"
        )
        
        batch_n_samples = st.sidebar.slider(
            "Number of Samples (Batch)",
            min_value=50,
            max_value=1000,
            value=100,
            help="Number of samples for SHAP calculation (applies to all files)",
            key="batch_n_samples"
        )
        
        # Batch processing button
        if st.sidebar.button("Process Batch Files"):
            # Collect shared settings
            shared_settings = {
                'model_type': batch_model_type,
                'task_type': batch_task_type,
                'target_column': batch_target_column,
                'n_samples': batch_n_samples,
                'title': "SHAP Values Distribution",
                'plot_height': 600,
                'plot_width': 800,
                'marker_size': 8,
                'marker_opacity': 0.6,
                'color_scheme': "RdBu",
                'show_grid': True,
                'show_legend': True,
                'legend_position': "right",
                'plot_layout': "vertical",
                'axis_config': {
                    'xaxis_title': "SHAP value (impact on model output)",
                    'yaxis_title': "Feature"
                },
                'background_color': "#FFFFFF",
                'font_sizes': {
                    'title': 20,
                    'axis_title': 16,
                    'axis_text': 12,
                    'legend': 12
                },
                'grid_style': {
                    'style': "solid",
                    'width': 1,
                    'color': "#D3D3D3"
                }
            }
            
            # Create progress bar
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            def update_progress(progress, text):
                progress_bar.progress(progress)
                status_text.text(text)
            
            # Process batch files
            with st.spinner("Processing batch files..."):
                results = process_batch_files(
                    batch_files,
                    shared_settings,
                    progress_callback=update_progress
                )
            
            # Display results in tabs
            if results:
                st.markdown("## Batch Processing Results")
                
                # Create tabs for successful and failed files
                successful_files = [r for r in results if r['success']]
                failed_files = [r for r in results if not r['success']]
                
                # Show summary
                st.markdown(f"""
                    ### Processing Summary
                    - Total files: {len(results)}
                    - Successfully processed: {len(successful_files)}
                    - Failed: {len(failed_files)}
                """)
                
                # Display successful results
                if successful_files:
                    st.markdown("### Successful Results")
                    tabs = st.tabs([r['filename'] for r in successful_files])
                    
                    for tab, result in zip(tabs, successful_files):
                        with tab:
                            st.plotly_chart(result['figure'], use_container_width=True)
                
                # Display errors for failed files
                if failed_files:
                    st.markdown("### Failed Files")
                    for result in failed_files:
                        st.error(f"{result['filename']}: {result['error']}")

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
        
        df = st.session_state['data']
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
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
                    value=100,
                    help="More samples = more accurate but slower"
                )
                
                title = st.text_input(
                    "Plot Title",
                    value="SHAP Values Distribution",
                    help="Enter a custom title for the plot",
                    key="plot_title"
                )

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
                
                # Axis Settings
                st.subheader("Axis Customization")
                col3, col4 = st.columns(2)
                with col3:
                    x_axis_label = st.text_input(
                        "X-axis Label",
                        value="SHAP value (impact on model output)",
                        help="Customize x-axis label",
                        key="x_axis_label"
                    )
                    
                    x_tick_format = st.text_input(
                        "X-axis Tick Format",
                        value="",
                        help="e.g., '.2f' for 2 decimal places",
                        key="x_tick_format"
                    )
                    
                    x_min = st.number_input("X-axis Min", value=None)
                    x_max = st.number_input("X-axis Max", value=None)
                
                with col4:
                    y_axis_label = st.text_input(
                        "Y-axis Label",
                        value="Feature",
                        help="Customize y-axis label",
                        key="y_axis_label"
                    )
                    
                    y_tick_format = st.text_input(
                        "Y-axis Tick Format",
                        value="",
                        help="Custom format for y-axis ticks",
                        key="y_tick_format"
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
                
                # Font Settings
                st.subheader("Font Settings")
                col11, col12 = st.columns(2)
                with col11:
                    title_font_size = st.slider(
                        "Title Font Size",
                        min_value=12,
                        max_value=32,
                        value=20,
                        help="Adjust the size of the plot title"
                    )
                    axis_title_font_size = st.slider(
                        "Axis Title Font Size",
                        min_value=10,
                        max_value=24,
                        value=16,
                        help="Adjust the size of axis titles"
                    )
                
                with col12:
                    axis_text_font_size = st.slider(
                        "Axis Text Font Size",
                        min_value=8,
                        max_value=20,
                        value=12,
                        help="Adjust the size of axis labels"
                    )
                    legend_font_size = st.slider(
                        "Legend Font Size",
                        min_value=8,
                        max_value=20,
                        value=12,
                        help="Adjust the size of legend text"
                    )
                
                # Grid Settings
                st.subheader("Grid Settings")
                col13, col14 = st.columns(2)
                with col13:
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
                
                with col14:
                    grid_color = st.color_picker(
                        "Grid Color",
                        "#D3D3D3",
                        help="Choose the color of grid lines"
                    )
                    show_grid = st.checkbox("Show Grid", value=True, key="grid_settings_show_grid")
                
                # Zoom and Hover Settings
                st.subheader("Zoom and Hover Settings")
                col15, col16 = st.columns(2)
                with col15:
                    enable_zoom = st.checkbox("Enable Fixed Zoom Level", value=False)
                    if enable_zoom:
                        zoom_level = st.slider(
                            "Zoom Level",
                            min_value=0.1,
                            max_value=5.0,
                            value=1.0,
                            step=0.1,
                            help="Set the zoom level for the plot"
                        )
                    else:
                        zoom_level = None
                
                with col16:
                    hover_template = st.text_input(
                        "Custom Hover Template",
                        value="%{y}: %{x:.4f}",
                        help="Customize the hover text template"
                    )
                
                # Additional Settings
                st.subheader("Additional Settings")
                col17, col18 = st.columns(2)
                with col17:
                    show_confidence = st.checkbox(
                        "Show Confidence Intervals",
                        value=False,
                        help="Display uncertainty in SHAP values"
                    )
                
                with col18:
                    enable_clustering = st.checkbox(
                        "Enable Feature Clustering",
                        value=False,
                        help="Group similar features together"
                    )
                    if enable_clustering:
                        n_clusters = st.slider(
                            "Number of Clusters",
                            min_value=2,
                            max_value=10,
                            value=3,
                            help="Choose the number of feature clusters"
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
                    
            # Generate plot button
            if st.button("Generate SHAP Plot", key="generate_plot"):
                try:
                    with st.spinner("Calculating SHAP values..."):
                        # Prepare data
                        X = df.drop(columns=[target_column])
                        y = df[target_column]
                        
                        # Calculate SHAP values
                        shap_values, std_vals = calculate_shap_values(
                            X=X,
                            y=y,
                            n_samples=n_samples,
                            model_type=model_type,
                            task_type=task_type
                        )
                        
                        if plot_type == "Interactive Plotly":
                            # Create Plotly plot
                            fig = create_shap_plot(
                                shap_values=shap_values,
                                X=X,
                                std_vals=std_vals if show_confidence else None,
                                title=title,
                                height=plot_height,
                                width=plot_width,
                                marker_size=marker_size,
                                marker_opacity=marker_opacity,
                                colorscale=color_scheme if color_scheme != "Custom" else None,
                                custom_colors={'positive': positive_color, 'negative': negative_color} if color_scheme == "Custom" else None,
                                show_grid=show_grid,
                                show_legend=True,
                                legend_position="right",
                                show_confidence=show_confidence,
                                cluster_features=enable_clustering,
                                n_clusters=n_clusters if enable_clustering else None,
                                layout=plot_layout,
                                axis_config={
                                    'xaxis_title': x_axis_label,
                                    'yaxis_title': y_axis_label,
                                    'xaxis_tickformat': x_tick_format,
                                    'yaxis_tickformat': y_tick_format,
                                    'xaxis_range': [x_min, x_max] if x_min is not None and x_max is not None else None
                                },
                                background_color=background_color,
                                font_sizes={
                                    'title': title_font_size,
                                    'axis_title': axis_title_font_size,
                                    'axis_text': axis_text_font_size,
                                    'legend': legend_font_size
                                },
                                grid_style={
                                    'style': grid_style,
                                    'width': grid_width,
                                    'color': grid_color
                                },
                                zoom_level=zoom_level,
                                hover_template=hover_template
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
                        else:
                            # Create native SHAP plot
                            plot_bytes = create_native_shap_plot(
                                shap_values=shap_values,
                                X=X,
                                title=title,
                                figsize=(figsize_width, figsize_height),
                                max_display=max_display,
                                plot_size=(plot_size * 8, plot_size * 12)
                            )
                            
                            # Display plot
                            st.image(plot_bytes, use_column_width=True)
                            
                            # Add download button
                            st.download_button(
                                label="Download Plot as PNG",
                                data=plot_bytes,
                                file_name="shap_beeswarm_plot.png",
                                mime="image/png"
                            )
                
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("""
    ---
    Made with ❤️ using Streamlit | [Documentation](https://github.com/slundberg/shap)
""")
