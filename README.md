# SHAP Plot Generator

A Streamlit application for generating and customizing SHAP (SHapley Additive exPlanations) plots to understand machine learning model predictions.

## Features

- Interactive SHAP plot generation
- Support for multiple model types:
  - Random Forest
  - Linear Regression/Classification
  - SVM
  - XGBoost
- Batch processing capabilities
- Extensive plot customization options
- Sample data generation
- Export plots in multiple formats (HTML, PNG)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/shap-gen.git
cd shap-gen
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Use the app:
   - Upload your dataset (CSV or Excel)
   - Or use the sample data/synthetic data generator
   - Configure model and plot settings
   - Generate and customize SHAP plots
   - Download visualizations

## Data Format

The input data should be a CSV or Excel file with:
- Features as columns
- Target variable in a separate column
- No missing values
- Numerical data (categorical variables should be encoded)

Example:
```
feature1,feature2,feature3,feature4,target
0.5,0.3,0.7,0.2,1.2
0.8,0.4,0.1,0.9,2.1
...
```

## Customization Options

- Plot dimensions
- Color schemes
- Marker size and opacity
- Grid settings
- Font sizes
- Axis labels and formats
- Background color
- Legend position
- Confidence intervals
- Feature clustering

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SHAP](https://github.com/slundberg/shap) library for the core functionality
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for interactive visualizations

## Deployment

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run main.py
```

### Docker Deployment
```bash
# Build image
docker build -t shap-gen .

# Run container
docker run -p 8501:8501 shap-gen
```

### Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io
3. Deploy from repository
