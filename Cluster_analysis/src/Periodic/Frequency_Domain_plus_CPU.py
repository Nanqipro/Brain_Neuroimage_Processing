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

    def load_data(self, file_path):
        """Load neuron time series data from Excel file"""
        df = pd.read_excel(file_path)
        # Select only neuron columns (n1 to n62)
        neuron_columns = [f'n{i}' for i in range(1, 63)]
        existing_neuron_cols = [col for col in neuron_columns if col in df.columns]
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
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=1024)
        return freqs, psd

    def compute_spectrogram(self, signal_data):
        """Compute spectrogram of the signal"""
        freqs, times, Sxx = spectrogram(signal_data, fs=self.fs, 
                                      nperseg=256, noverlap=128)
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
        time = np.arange(len(time_series))/self.fs
        ax1.plot(time, time_series)
        ax1.set_title(f'Original Signal - {neuron_id}')
        ax1.set_xlabel('Time (s)')
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
        im = ax4.pcolormesh(times_spec, freqs_spec, 10 * np.log10(Sxx), shading='gouraud')
        ax4.set_title('Spectrogram')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (s)')
        plt.colorbar(im, ax=ax4, label='Intensity (dB)')
        
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
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'comprehensive_analysis_{neuron_id}.png'))
        plt.close(fig)

def main():
    # Initialize analyzer
    analyzer = FrequencyAnalyzer(sampling_rate=10)
    
    # Set file paths
    file_path = '../../datasets/processed_Day3.xlsx'
    output_dir = '../../graph/frequency_analysis_day3_extended'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = analyzer.load_data(file_path)
    
    # Process each neuron
    feature_data = []
    for neuron_id in df.columns:
        print(f"Analyzing {neuron_id}...")
        time_series = df[neuron_id].values
        
        # Perform comprehensive analysis and generate plots
        analyzer.plot_comprehensive_analysis(neuron_id, time_series, output_dir)
        
        # Extract features
        features = analyzer.extract_features(time_series)
        features['neuron_id'] = neuron_id
        feature_data.append(features)
    
    # Create and save features DataFrame
    features_df = pd.DataFrame(feature_data)
    features_df.to_excel(os.path.join('../../datasets/frequency_features_day3.xlsx'), index=False)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()