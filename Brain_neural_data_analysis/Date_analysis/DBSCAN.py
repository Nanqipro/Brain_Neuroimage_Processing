import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path = './data/smoothed_normalized_2979_CSDS_Day6.xlsx'
data = pd.read_excel(file_path)

# Calculate the mean of each neuron's signal (used for threshold determination)
thresholds = data.iloc[:, 1:].mean()

# Initialize the feature DataFrame
features = pd.DataFrame(index=data.columns[1:], columns=['Peak_Amplitude', 'Duration', 'Frequency'])

# Iterate over each neuron to calculate the features
for neuron in data.columns[1:]:
    signal = data[neuron]
    peak_amplitude = signal.max()
    events = signal > thresholds[neuron]
    frequency = ((events.shift(1) == False) & (events == True)).sum()
    event_starts = data.index[events & (events.shift(1) == False)]
    event_ends = data.index[events & (events.shift(-1) == False)]
    durations = [end - start for start, end in zip(event_starts, event_ends) if end in event_ends]
    average_duration = np.mean(durations) if durations else 0
    features.loc[neuron] = [peak_amplitude, average_duration, frequency]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Estimate epsilon using a k-distance graph
min_pts = 3
nn = NearestNeighbors(n_neighbors=min_pts - 1)
nn.fit(features_scaled)
distances, indices = nn.kneighbors(features_scaled)
sorted_distances = np.sort(distances[:, -1])
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.title("K-Distance Graph")
plt.xlabel("Points sorted by distance to the k-th nearest neighbor")
plt.ylabel("k-th nearest neighbor distance")
plt.grid(True)
plt.show()

# DBSCAN clustering on scaled features
epsilon = 0.6 # Adjust based on the k-distance graph
dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
clusters = dbscan.fit_predict(features_scaled)

# Add clusters to the features DataFrame and reset index
features['Cluster'] = clusters
features.reset_index(inplace=True)
features.rename(columns={'index': 'Neuron'}, inplace=True)

# Save the clustering results
features.to_excel('./data/DBSCAN_clustering_results_2979_CSDS_Day6.xlsx', index=False)

print(features.head())
