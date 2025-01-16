import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the provided Excel data (replace with your file path)
file_path = './data/smoothed_normalized_2979_CSDS_Day6.xlsx'
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

# Step 4: Perform K-means++ clustering with optimal number of clusters
def perform_kmeans_plus_clustering(features):
    """
    Perform K-means++ clustering on the extracted features from calcium signals,
    determining the optimal number of clusters.

    Args:
    - features (DataFrame): Extracted features for each neuron.

    Returns:
    - labels (ndarray): Cluster labels for each neuron.
    - kmeans (KMeans): The trained KMeans object.
    """
    # Standardize the features before clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[['Peak_Amplitude', 'Duration', 'Frequency']])

    # Determine the optimal number of clusters using silhouette score
    best_score = -1
    optimal_k = 2

    for k in range(2, 11):  # Trying cluster sizes from 2 to 10
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)

        if score > best_score:
            best_score = score
            optimal_k = k

    # Perform final K-means++ clustering with the optimal number of clusters
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    final_labels = kmeans_final.fit_predict(scaled_features)

    return final_labels, kmeans_final

# Step 5: Perform K-means++ clustering on the features
labels, kmeans_model = perform_kmeans_plus_clustering(features)

# Step 6: Add cluster labels to the features DataFrame
features['Cluster'] = labels

# Step 7: Save the clustering results to a new Excel file (replace with your desired file path)
output_file_path = './data/kmeans++_clustering_results_2979_CSDS_Day6.xlsx'
features.to_excel(output_file_path, index=False)

print(f"Clustering results saved to: {output_file_path}")
