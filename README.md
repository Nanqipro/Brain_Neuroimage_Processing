# Mouse Brain Neuronal Calcium Imaging Data Analysis

## Introduction

This repository contains code for analyzing calcium imaging data from mouse brain neurons. It includes:

- **Calcium Wave Fluctuation Analysis**: Investigates the dynamics of neuronal calcium signals to explore patterns of neural activity.
- **Neuronal Clustering Analysis**: Uses clustering algorithms to categorize neurons into functionally similar groups.
- **Behavioral Feature Correlation Analysis**: Associates neural activities with mouse behavioral traits to identify connections.
- **Statistical Analysis Methods**: Applies statistical techniques to reveal regular patterns in mouse brain neural activities.

## Directory Structure

- `data/`: Raw and preprocessed datasets.
- `scripts/`: Primary data analysis scripts.
- `notebooks/`: Jupyter Notebook files for interactive analysis and visualization.
- `results/`: Analytical results and generated charts.
- `docs/`: Project documentation and descriptions.
- `LICENSE`: License file.

## Environmental Dependencies

Ensure the following environment and dependencies are installed:

- Python 3.7 or higher
- Required Python libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - SciPy
  - scikit-learn
  - Jupyter Notebook

## Installation Guide

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Create a Virtual Environment (Optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preprocessing

Run the preprocessing script to clean and format the raw data.

```bash
python scripts/preprocess_data.py
```

### 2. Calcium Fluctuation Analysis

Perform calcium signal fluctuation analysis and generate related charts and statistics.

```bash
python scripts/calcium_fluctuation_analysis.py
```

### 3. Neuron Clustering Analysis

Cluster the neuron data to identify functional groups.

```bash
python scripts/neuron_clustering.py
```

### 4. Behavioral Feature Correlation Analysis

Conduct correlation analysis between neural activities and behavioral traits.

```bash
python scripts/behavior_correlation_analysis.py
```

### 5. Statistical Analysis

Use Jupyter Notebook for in-depth statistical analysis.

```bash
jupyter notebook notebooks/statistical_analysis.ipynb
```

## Analysis Results

All analytical results and charts are saved in the `results/` directory, including:

- Calcium signal fluctuation charts
- Neuron clustering results visualization
- Behavioral correlation analysis charts
- Statistical reports

## Contribution Guidelines

Feel free to open issues, suggest improvements, or submit Pull Requests:

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Submit a Pull Request.

## License

This project is open-sourced under the MIT license. For more details, see the [LICENSE](LICENSE) file.

## Contact

- **Author**: ZhaoJin Pan Jiteng
- **Email**: ZhaoJ@example.com

Thank you for your interest in this project!
