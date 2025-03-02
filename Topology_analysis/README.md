# Neural Topology Analysis Toolkit

## Project Overview

The Neural Topology Analysis Toolkit is a comprehensive library specifically designed for analyzing the topological structure of neuronal activity patterns. By analyzing calcium concentration data, constructing neuronal connection topologies, and applying machine learning techniques to identify activity patterns, this project reveals the spatiotemporal organizational structure of neural networks.

## Key Features

- **Topology Structure Generation**: Builds time-series topological structures based on neuronal activity states (ON/OFF)
- **Topology Matrix Conversion**: Converts topological structures into standardized matrix formats for further analysis
- **Multi-algorithm Clustering Analysis**: Supports various clustering algorithms (KMeans, DBSCAN, Hierarchical, Spectral, Gaussian Mixture Models)
- **Automatic Parameter Optimization**: Includes functionality for automatically determining optimal clustering parameters
- **Visualization Tools**: Provides rich visualization options, including 2D/3D interactive charts
- **Spatiotemporal Pattern Analysis**: Combines time and space information for comprehensive analysis
- **Neuronal Behavior Association**: Associates neuronal activity patterns with behavioral labels

## Directory Structure

```
Topology_analysis/
├── src/                    # Source code directory
│   ├── TopologyToMatrix.py          # Basic topology matrix generation
│   ├── TopologyToMatrix_plus.py     # Enhanced topology matrix generation
│   ├── TopologyToMatrix_light.py    # Lightweight topology matrix generation
│   ├── TopologToMatrix_integrated.py # Integrated topology matrix generation
│   ├── Cluster_topology.py          # Main topology clustering analysis
│   ├── Cluster_topology_light.py    # Lightweight clustering analysis
│   ├── Cluster_topology_NoExp.py    # Clustering analysis without experimental grouping
│   ├── Cluster_topology_integrated.py # Integrated clustering analysis
│   ├── Pos_topology.py              # Spatial position topology analysis
│   ├── Time_topology.py             # Time series topology analysis
│   ├── Dynamic_Sorting.py           # Dynamic sorting analysis
│   ├── get_position.py              # Neuron position extraction
│   └── html_To_gif.py               # Interactive HTML to GIF converter
├── datasets/               # Datasets directory
│   ├── *.tif               # Original neuron imaging data
│   ├── *_Max.png           # Maximum intensity projection images
│   ├── *_Max_position.csv  # Neuron position coordinates
│   ├── *_neuron_states.xlsx # Neuron state data
│   ├── *_topology_matrix.xlsx # Basic topology matrices
│   ├── *_topology_matrix_plus.xlsx # Enhanced topology matrices
│   ├── integrated_topology_matrix.xlsx # Integrated topology matrix
│   ├── *_with_behavior_labels_filled.xlsx # Data with behavior labels
│   └── EMtrace.xlsx        # Experiment tracking data
├── result/                # Analysis results directory
│   └── clustering_*.log    # Clustering analysis log files
└── graph/                 # Visualization charts directory
    ├── *_pos_topology.html # Position topology visualization HTML
    ├── *_pos_topology.gif  # Position topology visualization GIF
    ├── *_Time_Topology.html # Time topology visualization HTML
    ├── *_Time_Topology.gif  # Time topology visualization GIF
    ├── *_Dynamic_Sorting.html # Dynamic sorting visualization
    └── *_Max_coordinates.png # Neuron coordinate visualization
```

## Installation Requirements

The project depends on the following Python libraries:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
networkx
plotly
scipy
typing
dataclasses
datetime
logging
```

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Topology Matrix Generation

Generate topology matrices from neuronal activity data:

```python
from src.TopologyToMatrix import main as create_topology

# Basic topology matrix generation
edge_records = create_topology(
    file_path="datasets/Day3_neuron_states.xlsx",
    topology_matrix_path="datasets/Day3_topology_matrix.xlsx"
)

# Enhanced topology matrix (with more edge information)
from src.TopologyToMatrix_plus import main as create_topology_plus
edge_records_plus = create_topology_plus(
    file_path="datasets/Day3_neuron_states.xlsx",
    topology_matrix_path="datasets/Day3_topology_matrix_plus.xlsx"
)
```

### 2. Clustering Analysis

Perform clustering analysis on topology matrices:

```python
from src.Cluster_topology import main as cluster_analysis

# Using KMeans clustering (algorithm ID=0)
cluster_analysis(
    file_path="datasets/Day3_topology_matrix_plus.xlsx",
    behavior_file_path="datasets/Day3_with_behavior_labels_filled.xlsx",
    algorithm_ids=0
)

# Using multiple clustering algorithms
cluster_analysis(
    file_path="datasets/Day3_topology_matrix_plus.xlsx",
    algorithm_ids=[0, 1, 4]  # KMeans, DBSCAN, GMM
)
```

### 3. Spatiotemporal Topology Analysis

Combine time and space information for topology analysis:

```python
from src.Time_topology import main as time_topology
from src.Pos_topology import main as pos_topology

# Time series topology analysis
time_topology(
    file_path="datasets/Day3_topology_matrix_plus.xlsx",
    output_html_path="graph/Day3_Time_Topology.html"
)

# Position topology analysis
pos_topology(
    file_path="datasets/Day3_topology_matrix_plus.xlsx",
    position_file="datasets/Day3_Max_position.csv",
    output_html_path="graph/Day3_pos_topology.html"
)

# Convert HTML visualization to GIF
from src.html_To_gif import html_to_gif
html_to_gif(
    html_file="graph/Day3_pos_topology.html",
    gif_file="graph/Day3_pos_topology.gif"
)
```

### 4. Dynamic Sorting Analysis

Perform dynamic sorting analysis on neurons:

```python
from src.Dynamic_Sorting import main as dynamic_sorting

dynamic_sorting(
    file_path="datasets/Day3_topology_matrix_plus.xlsx",
    output_html_path="graph/Day3_Dynamic_Sorting.html"
)
```

## Dataset Description

The project uses data from three time points (Day3, Day6, Day9):

- **Raw Data**: TIF format neuron imaging data
- **Neuron States**: Excel files recording ON/OFF states of neurons at each time point
- **Topology Matrices**: Topology connection matrices generated from neuron states
- **Behavior Labels**: Behavioral marker data associated with neuronal activity
- **Position Data**: Spatial coordinates of neurons

## Result Interpretation

- **Clustering Analysis Logs**: Records of clustering parameter selection, performance metrics, and analysis results
- **Visualization HTML**: Interactive charts showing dynamic changes in neuronal connection patterns
- **Coordinate Visualization**: Shows the spatial distribution of neurons

## Contributing

Contributions, issue reports, and feature requests are welcome. To contribute code, please fork this repository and submit a pull request.

## Notes

- Processing large datasets may require significant memory resources
- Generated HTML visualization files may be large; it's recommended to open them in a dedicated browser
- Random seeds are set during analysis to ensure reproducibility of results

## License

[MIT License] 