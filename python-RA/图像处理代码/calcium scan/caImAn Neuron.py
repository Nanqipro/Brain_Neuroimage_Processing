import caiman as cm
from caiman.source_extraction.cnmf import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 1: Set File Paths
video_path = r'C:\Users\PAN\Desktop\RA\homecage_testing.avi'  # Replace with your actual AVI file path
output_folder = r'C:\Users\PAN\Desktop\RA\CaImAn_Output'
os.makedirs(output_folder, exist_ok=True)

# Memory-mapped file path
mmap_file = os.path.join(output_folder, 'memory_mapped_movie.mmap')

# Step 2: Load Video and Create Memory-Mapped File
print("Loading video and creating memory-mapped file...")
movie = cm.load(video_path)

# Save the movie as a memory-mapped file
movie.save(mmap_file)

# Verify that the memory-mapped file was created
if not os.path.exists(mmap_file):
    raise FileNotFoundError(f"Memory-mapped file not created: {mmap_file}")
print(f"Memory-mapped file saved to: {mmap_file}")

# Step 3: Set CNMF Parameters
cnmf_params = cnmf.params.CNMFParams(params_dict={
    'fnames': mmap_file,
    'fr': 30,               # Frame rate (adjust if different)
    'decay_time': 0.4,      # Fluorescence decay time
    'pw_rigid': False,      # Turn off piecewise-rigid motion correction
    'nb': 2,                # Number of background components
    'gnb': 0,               # If 'gnb' is 0, 'nb' is not constrained
    'merge_thr': 0.85,      # Merging threshold for components
    'rf': 40,               # Half-size of patches in pixels
    'stride_cnmf': 20,      # Amount of overlap between patches
    'K': None,              # Number of components per patch (None for automatic estimation)
    'gSig': [4, 4],         # Expected half size of neurons in pixels
    'method_init': 'greedy_roi',  # Initialization method
    'rolling_sum': True,    # Use rolling sum for initialization
    'only_init': False      # Whether to stop after initialization
})

# Step 4: Compute Summary Images (Correlation and PNR)
print("Computing summary images...")
cn_filter, pnr = cm.summary_images.correlation_pnr(movie, gSig=cnmf_params.get('gSig'), swap_dim=False)

# Save and visualize the summary images
inspect_correlation_pnr(cn_filter, pnr)
plt.savefig(os.path.join(output_folder, 'correlation_pnr.png'))
plt.close()

# Step 5: Run CNMF Algorithm
print("Running CNMF algorithm...")
cnm = cnmf.CNMF(n_processes=1, params=cnmf_params)
cnm.fit_file(mmap_file)

# Step 6: Extract Neuron Positions
print("Extracting neuron positions...")
coordinates = cnm.estimates.coordinates
print(f"Total neurons detected: {len(coordinates)}")

# Step 7: Visualize Neuron Positions
plt.figure(figsize=(10, 10))
plt.imshow(cn_filter, cmap='gray', origin='upper')
for coord in coordinates:
    y, x = coord['CoM']
    plt.scatter(x, y, color='red', s=15)
plt.title('Neuron Positions')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(output_folder, 'neuron_positions.png'))
plt.show()

# Step 8: Save Results
print("Saving results...")
cnm.save(os.path.join(output_folder, 'cnmf_results.hdf5'))
np.savetxt(os.path.join(output_folder, 'neuron_positions.csv'), [coord['CoM'] for coord in coordinates], delimiter=',')

print(f"All results saved in {output_folder}.")
