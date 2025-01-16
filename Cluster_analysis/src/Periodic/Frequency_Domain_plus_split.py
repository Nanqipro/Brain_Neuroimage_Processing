
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram, butter, filtfilt, coherence, hilbert
import pywt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

class FrequencyAnalyzer:
    def __init__(self, sampling_rate=10):
        """Initialize the FrequencyAnalyzer with sampling rate"""
        self.fs = sampling_rate
        self.delta_t = 1/sampling_rate

    def load_data(self, file_path, start_time=None, end_time=None):
        """
        Load neuron time series data from Excel file with optional time window selection
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file
        start_time : float, optional
            Start time in seconds for analysis window
        end_time : float, optional
            End time in seconds for analysis window
        """
        df = pd.read_excel(file_path)
        
        # Select time window if specified
        if 'stamp' in df.columns:
            if start_time is not None or end_time is not None:
                start_time = start_time if start_time is not None else df['stamp'].min()
                end_time = end_time if end_time is not None else df['stamp'].max()
                df = df[(df['stamp'] >= start_time) & (df['stamp'] <= end_time)].copy()
        
        # Select only neuron columns (n1 to n62)
        neuron_columns = [f'n{i}' for i in range(1, 63)]
        existing_neuron_cols = [col for col in neuron_columns if col in df.columns]
        
        # Store time information
        self.timestamps = df['stamp'].values if 'stamp' in df.columns else None
        
        return df[existing_neuron_cols]

    def perform_fft_analysis(self, signal_data):
        """Perform FFT analysis on the input signal"""
        n = len(signal_data)
        fft_result = fft(signal_data)
        freqs = fftfreq(n, self.delta_t)
        
        # Get the positive frequencies and their magnitudes
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitudes = np.abs(fft_result)[pos_mask]
        
        return freqs, magnitudes

    def calculate_psd(self, signal_data):
        """Calculate Power Spectral Density using Welch's method"""
        # Adjust nperseg based on signal length
        nperseg = min(1024, len(signal_data))
        if nperseg < 4:  # Ensure minimum segment length
            nperseg = len(signal_data)
        noverlap = nperseg // 2
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
        return freqs, psd

    def compute_spectrogram(self, signal_data):
        """Compute spectrogram of the signal"""
        # Adjust nperseg based on signal length
        nperseg = min(256, len(signal_data) // 4)  # Ensure at least 4 segments
        if nperseg < 4:  # Ensure minimum segment length
            nperseg = len(signal_data)
        noverlap = min(128, nperseg // 2)  # Adjust overlap accordingly
        
        freqs, times, Sxx = spectrogram(signal_data, fs=self.fs, 
                                      nperseg=nperseg, noverlap=noverlap)
        return freqs, times, Sxx

    def perform_wavelet_analysis(self, signal_data, wavelet='db4', level=5):
        """Perform wavelet decomposition"""
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        return coeffs

    def bandpass_filter(self, signal_data, lowcut, highcut, order=4):
        """Apply bandpass filter to the signal"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure frequencies are within valid range (0 < freq < 1)
        low = np.clip(low, 0.001, 0.99)
        high = np.clip(high, 0.001, 0.99)
        
        # Ensure low < high
        if low >= high:
            low, high = high/2, high
        
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal_data)
        return y

    def calculate_phase(self, signal_data):
        """Calculate instantaneous phase using Hilbert transform"""
        # Normalize signal before Hilbert transform
        signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        analytic_signal = hilbert(signal_normalized)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        return amplitude_envelope, instantaneous_phase

    def calculate_coherence(self, signal1, signal2):
        """Calculate coherence between two signals"""
        freqs, coh = coherence(signal1, signal2, fs=self.fs, nperseg=1024)
        return freqs, coh

    def extract_features(self, signal_data):
        """Extract comprehensive frequency domain features"""
        # FFT features
        freqs, magnitudes = self.perform_fft_analysis(signal_data)
        main_freq = freqs[np.argmax(magnitudes)]
        peak_power = np.max(magnitudes)
        total_power = np.sum(magnitudes)

        # Band power features
        theta = self.bandpass_filter(signal_data, 4, 8)
        gamma = self.bandpass_filter(signal_data, 30, min(100, self.fs/2 - 1))
        theta_power = np.sum(theta**2)
        gamma_power = np.sum(gamma**2)

        # Time domain features
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        max_val = np.max(signal_data)
        min_val = np.min(signal_data)

        features = {
            'main_frequency': main_freq,
            'peak_power': peak_power,
            'total_power': total_power,
            'theta_power': theta_power,
            'gamma_power': gamma_power,
            'mean': mean_val,
            'std': std_val,
            'max': max_val,
            'min': min_val
        }
        return features

    def plot_comprehensive_analysis(self, neuron_id, time_series, output_dir):
        """Generate comprehensive frequency analysis plots"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 20))
        
        # 1. Original Signal
        ax1 = plt.subplot(611)
        # Use actual timestamps if available, otherwise use indices
        if self.timestamps is not None:
            time = self.timestamps
            ax1.set_xlabel('Time (s)')
        else:
            time = np.arange(len(time_series))/self.fs
            ax1.set_xlabel('Time (s)')
        
        ax1.plot(time, time_series)
        ax1.set_title(f'Original Signal - {neuron_id}')
        ax1.set_ylabel('Amplitude')
        
        # 2. FFT
        ax2 = plt.subplot(612)
        freqs, magnitudes = self.perform_fft_analysis(time_series)
        ax2.plot(freqs, magnitudes)
        ax2.set_title('Frequency Spectrum (FFT)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_xlim(0, self.fs/2)
        
        # 3. PSD
        ax3 = plt.subplot(613)
        freqs_psd, psd = self.calculate_psd(time_series)
        ax3.semilogy(freqs_psd, psd)
        ax3.set_title('Power Spectral Density')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power/Frequency')
        ax3.set_xlim(0, self.fs/2)
        
        # 4. Spectrogram
        ax4 = plt.subplot(614)
        freqs_spec, times_spec, Sxx = self.compute_spectrogram(time_series)
        # Add small constant to avoid log(0)
        Sxx = Sxx + 1e-10
        im = ax4.pcolormesh(times_spec, freqs_spec, 10 * np.log10(Sxx), 
                           shading='gouraud', cmap='viridis')
        ax4.set_title('Spectrogram')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (s)')
        plt.colorbar(im, ax=ax4, label='Power/Frequency (dB/Hz)')
        
        # 5. Filtered Signals
        ax5 = plt.subplot(615)
        theta = self.bandpass_filter(time_series, 4, 8)
        gamma = self.bandpass_filter(time_series, 30, min(100, self.fs/2 - 1))
        ax5.plot(time, theta, label='Theta (4-8 Hz)')
        ax5.plot(time, gamma, label=f'Gamma (30-{min(100, self.fs/2 - 1)} Hz)')
        ax5.set_title('Filtered Signals')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Amplitude')
        ax5.legend()
        
        # 6. Phase Analysis
        ax6 = plt.subplot(616)
        amp_env, inst_phase = self.calculate_phase(time_series)
        ax6.plot(time, inst_phase)
        ax6.set_title('Instantaneous Phase')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Phase (radians)')
        
        plt.tight_layout()
        
        # Add time window information to title if using specific window
        if self.timestamps is not None:
            window_info = f" (Time: {self.timestamps[0]:.1f}s - {self.timestamps[-1]:.1f}s)"
            fig.suptitle(f"Frequency Analysis for {neuron_id}{window_info}", y=1.02, fontsize=14)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'comprehensive_analysis_{neuron_id}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    # Initialize analyzer
    analyzer = FrequencyAnalyzer(sampling_rate=10)
    
    # Set file paths
    file_path = '../../datasets/processed_Day6.xlsx'
    output_dir = '../../graph/frequency_analysis_day6_extended_split'
    
    # Define time windows for analysis (in seconds)
    # Adjust window size to ensure enough samples for analysis
    window_size = 300  # 300 seconds (5 minutes)
    overlap = 60      # 60 seconds overlap
    
    # Load full data first to get time range
    print("Loading full dataset to determine time range...")
    full_df = pd.read_excel(file_path)
    if 'stamp' in full_df.columns:
        total_time = full_df['stamp'].max() - full_df['stamp'].min()
        start_times = np.arange(full_df['stamp'].min(), full_df['stamp'].max() - window_size, window_size - overlap)
        time_windows = [(start, start + window_size) for start in start_times]
    else:
        print("Warning: No timestamp column found. Using default windows.")
        time_windows = [
            (0, 300),       # 0-5 minutes
            (240, 540),     # 4-9 minutes
            (480, 780)      # 8-13 minutes
        ]
    
    print(f"Analyzing {len(time_windows)} time windows...")
    
    # Process each time window
    for start_time, end_time in time_windows:
        # Create output directory for this time window
        window_dir = os.path.join(output_dir, f'time_{start_time:04d}_{end_time:04d}')
        os.makedirs(window_dir, exist_ok=True)
        
        print(f"\nAnalyzing time window: {start_time}s - {end_time}s")
        
        try:
            # Load data for this time window
            print("Loading data...")
            df = analyzer.load_data(file_path, start_time=start_time, end_time=end_time)
            
            if len(df) < 50:  # Skip windows with too few samples
                print(f"Skipping window {start_time}-{end_time}: insufficient samples")
                continue
            
            # Process each neuron
            feature_data = []
            for neuron_id in df.columns:
                print(f"Analyzing {neuron_id}...")
                time_series = df[neuron_id].values
                
                try:
                    # Perform comprehensive analysis and generate plots
                    analyzer.plot_comprehensive_analysis(neuron_id, time_series, window_dir)
                    
                    # Extract features
                    features = analyzer.extract_features(time_series)
                    features.update({
                        'neuron_id': neuron_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'window_size': window_size,
                        'n_samples': len(time_series)
                    })
                    feature_data.append(features)
                except Exception as e:
                    print(f"Error processing neuron {neuron_id}: {str(e)}")
                    continue
            
            # Create and save features DataFrame for this time window
            if feature_data:
                features_df = pd.DataFrame(feature_data)
                features_df.to_excel(os.path.join(window_dir, 'frequency_features.xlsx'), index=False)
        except Exception as e:
            print(f"Error processing window {start_time}-{end_time}: {str(e)}")
            continue
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
