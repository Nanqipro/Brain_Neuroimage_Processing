import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the provided Excel data (replace with your file path)
file_path = './data/smoothed_normalized_2979_CSDS_Day3.xlsx'
data = pd.read_excel(file_path)

# Step 2: Define feature extraction function for K-means clustering
def extract_features(data):
    """
    Extracts features such as peak amplitude, signal duration, and event frequency
    from each neuron's calcium signal.

    Args:
    - data (DataFrame): Smoothed ΔF/F data with rows as time points and columns as neurons.

    Returns:
    - features (DataFrame): A DataFrame of extracted features for each neuron.
    """
    feature_list = []  # Using a list to collect the features and later convert to DataFrame

    for column in data.columns[1:]:  # Skip the 'stamp' column
        signal = data[column]

        # Peak amplitude (max value)
        peak_amplitude = signal.max()

        # Duration (total signal duration above a threshold, e.g., 0.1)
        duration = np.sum(signal > 0.1)

        # Event frequency (number of peaks above a certain threshold, e.g., 0.1)
        frequency = np.sum(signal > 0.1) / len(signal)

        # Append to the list as a dictionary
        feature_list.append({
            'Neuron': column,
            'Peak_Amplitude': peak_amplitude,
            'Duration': duration,
            'Frequency': frequency
        })

    # Convert list of dictionaries to a DataFrame
    features = pd.DataFrame(feature_list)
    return features

# Step 3: Extract features from the smoothed data
features = extract_features(data)

# Step 4: Perform K-means clustering
def perform_kmeans_clustering(features, n_clusters=5):
    """
    Perform K-means clustering on the extracted features from calcium signals.

    Args:
    - features (DataFrame): Extracted features for each neuron.
    - n_clusters (int): The number of clusters for K-means.

    Returns:
    - labels (ndarray): Cluster labels for each neuron.
    - kmeans (KMeans): The trained KMeans object.
    """
    # Standardize the features before clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[['Peak_Amplitude', 'Duration', 'Frequency']])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)

    return labels, kmeans

# Step 5: Perform K-means clustering on the features
labels, kmeans_model = perform_kmeans_clustering(features, n_clusters=5)

# Step 6: Add cluster labels to the features DataFrame
features['Cluster'] = labels

# Step 7: Save the clustering results to a new Excel file (replace with your desired file path)
output_file_path = './data/kmeans_clustering_results_2979_CSDS_Day3.xlsx'
features.to_excel(output_file_path, index=False)

print(f"Clustering results saved to: {output_file_path}")
