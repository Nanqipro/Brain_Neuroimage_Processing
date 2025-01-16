import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the K-means clustering results (replace with your file path)
file_path = './data/DBSCAN_clustering_results_2979_CSDS_Day6.xlsx'
clustering_results = pd.read_excel(file_path)

# Step 2: Define the function for plotting the 3D scatter plot
def plot_3d_clusters(data):
    """
    Plots a 3D scatter plot of the clustering results.

    Args:
    - data (DataFrame): Clustering results with 'Peak_Amplitude', 'Duration', 'Frequency', and 'Cluster'.

    Returns:
    - None
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    print(data.columns)

    # Extract the three dimensions and the cluster labels
    x = data['Peak_Amplitude']
    y = data['Duration']
    z = data['Frequency']
    clusters = data['Cluster']

    # Plot the points with different colors for each cluster
    scatter = ax.scatter(x, y, z, c=clusters, cmap='viridis', s=50)

    # Labels and title
    ax.set_xlabel('Peak Amplitude')
    ax.set_ylabel('Duration')
    ax.set_zlabel('Frequency')
    ax.set_title('3D DBSCAN Clustering of Neuron Data')

    # Add a color bar to show cluster assignment
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')

    # Show the plot
    plt.show()

# Step 3: Call the function to plot the 3D scatter plot
plot_3d_clusters(clustering_results)
