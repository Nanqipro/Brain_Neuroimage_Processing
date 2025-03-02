# Brain Neuroimage Processing Platform

## Project Overview

This repository contains a comprehensive set of tools for analyzing neural calcium imaging data from mouse brain neurons. The platform consists of four integrated modules that facilitate end-to-end neural data analysis, from preprocessing to advanced machine learning and topological analysis.

## Key Components

### 1. Pre-analysis Module

A preprocessing and exploratory data analysis toolkit that prepares neural imaging data for advanced analysis:

- **Data Integration & Cleaning**: Combines and standardizes data from multiple imaging sessions
- **Exploratory Data Analysis (EDA)**: Initial statistical analysis and visualization of neural activity patterns
- **Feature Extraction**: Identifies key characteristics in calcium wave signals
- **Smoothing Algorithms**: Reduces noise while preserving important signal features
- **Periodicity Analysis**: Detects and quantifies rhythmic patterns in neural activity
- **Correlation Analysis**: Examines relationships between neural signals and behavioral markers

### 2. Cluster Analysis Module

Implements various clustering algorithms to identify functional groups of neurons:

- **Multiple Clustering Algorithms**: K-means, DBSCAN, GMM, Hierarchical, Spectral
- **Dimensionality Reduction**: PCA, t-SNE, UMAP for visualizing high-dimensional neural data
- **Distance Metrics**: Euclidean, Manhattan, EMD, Hausdorff for different similarity measures
- **Indicator Extraction**: Quantitative metrics for evaluating cluster quality
- **Active Neuron Visualization**: Dynamic bar charts showing neuronal activity patterns

### 3. Topology Analysis Module

Analyzes the topological structure of neuronal networks:

- **Topology Structure Generation**: Builds time-series topological structures based on neuronal activity states
- **Topology Matrix Conversion**: Converts topological structures into standardized matrix formats
- **Multi-algorithm Clustering Analysis**: Identifies patterns in topology matrices
- **Spatiotemporal Pattern Analysis**: Combines time and space information for comprehensive analysis
- **Interactive Visualization**: 2D/3D visualization of neural network topology

### 4. LSTM Analysis Module

Leverages deep learning to analyze temporal patterns in neural activity:

- **LSTM-based Time Series Analysis**: Encodes and predicts neuronal activity sequences
- **Clustering of Temporal Patterns**: Groups neurons by their temporal activation profiles
- **Neural Network Topology Analysis**: Examines functional connectivity based on LSTM embeddings
- **Interactive Network Visualization**: Dynamic visualization of neural networks and their changes
- **Correlation with Behavior**: Associates neural temporal patterns with behavioral events

## Directory Structure

```
Brain_Neuroimage_Processing/
├── Pre_analysis/            # Preprocessing and initial analysis
│   ├── src/                 # Source code for preprocessing
│   │   ├── EDA/             # Exploratory data analysis tools
│   │   ├── DataIntegration/ # Data integration scripts
│   │   ├── Feature/         # Feature extraction tools
│   │   ├── smooth/          # Signal smoothing algorithms
│   │   ├── Periodic/        # Periodicity analysis
│   │   ├── oneNeuronal/     # Single neuron analysis
│   │   ├── heatmap/         # Heatmap visualization
│   │   └── Comparative/     # Comparative analysis
│   ├── datasets/            # Raw and processed datasets
│   └── graph/               # Generated visualizations
│
├── Cluster_analysis/        # Clustering tools and algorithms
│   ├── src/                 # Clustering algorithms implementation
│   │   ├── k-means-*.py     # Various K-means implementations
│   │   ├── DBSCAN.py        # DBSCAN clustering
│   │   ├── GMM.py           # Gaussian Mixture Models
│   │   ├── Hierarchical.py  # Hierarchical clustering
│   │   ├── Spectral.py      # Spectral clustering
│   │   ├── *_analysis.py    # Dimensionality reduction tools
│   │   └── Active_bar_chart.py # Activity visualization
│   └── datasets/            # Input data for clustering
│
├── Topology_analysis/       # Network topology analysis
│   ├── src/                 # Topology analysis code
│   │   ├── TopologyToMatrix*.py # Topology matrix generation
│   │   ├── Cluster_topology*.py # Topology clustering
│   │   ├── Pos_topology.py      # Spatial topology analysis
│   │   ├── Time_topology.py     # Temporal topology analysis
│   │   └── Dynamic_Sorting.py   # Dynamic structure analysis
│   ├── datasets/            # Topology datasets
│   ├── result/              # Analysis results
│   ├── graph/               # Topology visualizations
│   └── requirements.txt     # Topology module dependencies
│
└── LSTM/                    # LSTM-based temporal analysis
    ├── src/                 # LSTM analysis source code
    │   ├── lib/             # Support libraries
    │   ├── kmeans_lstm_analysis.py # LSTM with clustering
    │   ├── analysis_results.py     # Results processing
    │   ├── pos_topology_js.py      # Network visualization
    │   └── visualization.py        # Visualization tools
    ├── datasets/            # LSTM input data
    ├── models/              # Trained LSTM models
    ├── results/             # LSTM analysis results
    ├── README_interactive_network.md # Interactive network guide
    └── requirements.txt     # LSTM module dependencies
```

## Installation Requirements

Each module has its own dependencies specified in respective `requirements.txt` files. The core dependencies for the entire platform include:

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
torch>=2.1.0 (for LSTM module)
networkx>=2.6.0 (for Topology module)
plotly>=5.3.0
```

To set up the environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/Brain_Neuroimage_Processing.git
cd Brain_Neuroimage_Processing

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install module-specific dependencies (as needed)
pip install -r LSTM/requirements.txt
pip install -r Topology_analysis/requirements.txt
```

## Usage Workflow

### 1. Data Preprocessing (Pre_analysis)

```bash
# Run exploratory data analysis
python Pre_analysis/src/EDA/init_show.py

# Perform correlation analysis
python Pre_analysis/src/EDA/Correlation_Analysis.py
```

### 2. Cluster Analysis

```bash
# Run K-means clustering
python Cluster_analysis/src/k-means-ed.py

# Run UMAP dimensionality reduction
python Cluster_analysis/src/umap_analysis.py
```

### 3. Topology Analysis

```bash
# Generate topology matrices
python Topology_analysis/src/TopologyToMatrix.py

# Perform clustering on topology matrices
python Topology_analysis/src/Cluster_topology.py

# Visualize spatial topology
python Topology_analysis/src/Pos_topology.py
```

### 4. LSTM Analysis

```bash
# Run LSTM analysis with K-means
python LSTM/src/kmeans_lstm_analysis.py

# Analyze and visualize results
python LSTM/src/analysis_results.py

# Generate interactive network visualization
python LSTM/src/pos_topology_js.py
```

## Analysis Results

Each module saves its results in its respective output directory:

- **Pre_analysis**: Initial data exploration and feature extraction results
- **Cluster_analysis**: Clustering results and dimensionality reduction visualizations
- **Topology_analysis/result**: Topology matrices and clustering logs
- **Topology_analysis/graph**: Network visualizations and animations
- **LSTM/results**: LSTM model performance metrics and prediction visualizations
- **LSTM/models**: Trained neural network models

## Contributing

Contributions to any module are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: ZhaoJin Pan Jiteng
- **Email**: ZhaoJ@example.com

For module-specific questions, please refer to the README files in each module directory.
