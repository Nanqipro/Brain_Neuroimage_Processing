# Neural Network LSTM Analysis Platform

## Project Overview

This is an LSTM (Long Short-Term Memory) platform for neural network data analysis, focusing on neuron activity sequence analysis, clustering, and network topology visualization. The project leverages deep learning techniques to analyze neural activity patterns, aiming to reveal the dynamic changes in neural network functional connectivity and their association with behavior.

## Key Features

- LSTM-based time series analysis of neuronal activity
- K-means and other clustering algorithms for neuronal functional grouping
- Neural network topology structure analysis
- Interactive and static network visualization
- Correlation analysis between neuronal activity and behavior
- Model training, testing, and validation framework

## Project Structure

```
LSTM/
├── src/                    # Source code directory
│   ├── lib/                # Core library components
│   │   ├── vis-9.1.2/      # Visualization library
│   │   ├── tom-select/     # Selection component library
│   │   └── bindings/       # Language bindings
│   ├── analysis_results.py # Results analysis script
│   ├── pos_topology_js.py  # Topology analysis script
│   ├── analysis_config.py  # Analysis configuration
│   ├── kmeans_lstm_analysis.py # LSTM clustering analysis
│   ├── visualization.py    # Visualization tools
│   ├── analysis_utils.py   # Analysis utility functions
│   ├── enhanced_analysis.py # Enhanced analysis methods
│   └── test_env.py         # Environment test script
├── datasets/               # Datasets directory 
├── models/                 # Trained model storage
│   ├── neuron_lstm_model_Day3.pth # Day 3 neuron model
│   ├── neuron_lstm_model_Day6.pth # Day 6 neuron model
│   └── neuron_lstm_model_Day9.pth # Day 9 neuron model
├── results/                # Analysis results output directory
├── README_interactive_network.md # Interactive network usage guide
└── requirements.txt        # Project dependencies
```

## Installation Requirements

Running this project requires the following dependencies (Python 3.8+ recommended):

```
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage Guide

### Environment Testing

First, verify that your environment is correctly configured:

```bash
python src/test_env.py
```

### Data Analysis Workflow

1. **Configure Analysis Parameters**
   
   Edit the `src/analysis_config.py` file to set analysis parameters

2. **Run LSTM Analysis**

   ```bash
   python src/kmeans_lstm_analysis.py
   ```

3. **Results Analysis and Visualization**

   ```bash
   python src/analysis_results.py
   ```

4. **Visualize Network Topology**

   ```bash
   python src/pos_topology_js.py
   ```

### Interactive Network Visualization

For detailed instructions, refer to `README_interactive_network.md`

## Main Module Descriptions

### LSTM Model

The project uses Long Short-Term Memory networks (LSTM) to encode and analyze neuronal activity time series. The model structure follows the project's standard architecture, including:

- Multi-layer LSTM encoder
- Gradient clipping
- Learning rate scheduler
- Standardized input processing

### Clustering Analysis

Uses K-means and other clustering algorithms to analyze neuronal functional organization:

- Automatic parameter selection
- Cluster stability analysis
- Cluster evolution tracking over time

### Topology Analysis

Analyzes functional connections between neurons:

- Correlation-based connection strength calculation
- Community detection algorithms
- Centrality and path length analysis

### Visualization

Provides various visualization tools:

- Static network graphs
- Interactive network visualization
- Activity pattern heatmaps
- Clustering results visualization

## Results Interpretation

Analysis results are saved in the `results` directory, including:

- Training and validation metrics
- Clustering results
- Network analysis results
- Visualization charts

## Troubleshooting

Common issues and solutions:

1. **CUDA Errors**: Ensure proper GPU environment variables and compatible CUDA version installation
2. **Memory Errors**: Reduce batch size or use data sampling
3. **Visualization Issues**: Ensure all relevant visualization libraries are installed

## Contact and Contributions

Questions and suggestions for improvements are welcome!

## Citation

If you use this platform in your research, please cite:

```
@software{neural_network_lstm_platform,
  author = {Your Research Team},
  title = {Neural Network LSTM Analysis Platform},
  year = {2023},
  url = {https://github.com/yourusername/neural-network-lstm}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

