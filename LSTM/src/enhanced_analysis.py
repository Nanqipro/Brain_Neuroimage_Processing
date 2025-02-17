import numpy as np
import pandas as pd
import torch
from analysis_config import AnalysisConfig
from analysis_utils import DataProcessor, StatisticalAnalyzer, ResultSaver
from visualization import VisualizationManager
from kmeans_lstm_analysis import NeuronDataProcessor
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnalyzer:
    def __init__(self):
        self.config = AnalysisConfig()
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.result_saver = ResultSaver(self.config)
        self.visualizer = VisualizationManager(self.config)
        self.neuron_processor = NeuronDataProcessor(self.config)
    
    def prepare_data(self):
        """Prepare and preprocess data"""
        print("Loading and preprocessing data...")
        X_scaled, y = self.neuron_processor.preprocess_data()
        behavior_labels = self.neuron_processor.label_encoder.classes_
        
        # Balance data if needed
        if self.config.analysis_params['min_samples_per_behavior'] > 0:
            X_scaled, y = self.data_processor.balance_data(
                X_scaled, y, 
                self.config.analysis_params['min_samples_per_behavior']
            )
        
        # Merge rare behaviors if needed
        X_scaled, y, behavior_labels = self.data_processor.merge_rare_behaviors(
            X_scaled, y, behavior_labels,
            self.config.analysis_params['min_samples_per_behavior']
        )
        
        return X_scaled, y, behavior_labels
    
    def perform_statistical_analysis(self, X_scaled, y, behavior_labels):
        """Perform statistical analysis"""
        print("\nPerforming statistical analysis...")
        
        # ANOVA analysis
        f_values, p_values = self.statistical_analyzer.perform_anova(X_scaled, y)
        
        # Effect size analysis
        effect_sizes = self.statistical_analyzer.calculate_effect_sizes(
            X_scaled, y, behavior_labels
        )
        
        # Temporal correlation analysis
        temporal_correlations = self.statistical_analyzer.analyze_temporal_correlations(
            X_scaled, y
        )
        
        return f_values, p_values, effect_sizes, temporal_correlations
    
    def analyze_behavior_patterns(self, X_scaled, y, behavior_labels):
        """Analyze behavior patterns"""
        print("\nAnalyzing behavior patterns...")
        
        # Calculate behavior transition matrix
        transitions = np.zeros((len(behavior_labels), len(behavior_labels)))
        for i in range(len(y)-1):
            transitions[y[i], y[i+1]] += 1
        
        # Normalize transitions
        row_sums = transitions.sum(axis=1)
        transitions_norm = transitions / row_sums[:, np.newaxis]
        
        # Calculate mean activity for each behavior
        behavior_means = {}
        for behavior_idx, behavior in enumerate(behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_means[behavior] = np.mean(X_scaled[behavior_mask], axis=0)
        
        behavior_activity_df = pd.DataFrame(behavior_means).T
        behavior_activity_df.columns = [f'Neuron {i+1}' for i in range(behavior_activity_df.shape[1])]
        
        return transitions_norm, behavior_activity_df
    
    def save_and_visualize_results(self, results):
        """Save and visualize analysis results"""
        print("\nSaving and visualizing results...")
        
        # Unpack results
        (f_values, p_values, effect_sizes, temporal_correlations,
         transitions_norm, behavior_activity_df, behavior_labels) = results
        
        # Save statistical results
        self.result_saver.save_statistical_results(f_values, p_values, effect_sizes)
        self.result_saver.save_temporal_correlations(temporal_correlations)
        
        # Create visualizations
        self.visualizer.set_plot_style()
        self.visualizer.plot_behavior_neuron_correlation(behavior_activity_df)
        self.visualizer.plot_behavior_transitions(transitions_norm, behavior_labels)
        self.visualizer.plot_neuron_network(effect_sizes)
        self.visualizer.plot_temporal_correlations(temporal_correlations)
        self.visualizer.plot_statistical_summary(f_values, p_values, effect_sizes)
        
        # Print key findings
        print("\nKey Findings:")
        print("\n1. Significant Neurons (p < 0.05):")
        significant_neurons = np.where(p_values < self.config.analysis_params['p_value_threshold'])[0]
        print(f"Found {len(significant_neurons)} significant neurons: {significant_neurons + 1}")
        
        print("\n2. Behavior-Specific Neurons:")
        for behavior, data in effect_sizes.items():
            significant = data['significant_neurons']
            if len(significant) > 0:
                print(f"\n{behavior}:")
                print(f"Significant neurons: {significant + 1}")
                print(f"Top effect sizes: {data['effect_sizes'][significant][:5]}")
        
        print("\n3. Temporal Correlation Summary:")
        for window, corr_values in temporal_correlations.items():
            mean_corr = np.mean(corr_values)
            print(f"Window size {window}: Mean correlation = {mean_corr:.3f}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # Setup
            self.config.setup_directories()
            self.config.validate_paths()
            
            # Data preparation
            X_scaled, y, behavior_labels = self.prepare_data()
            
            # Statistical analysis
            f_values, p_values, effect_sizes, temporal_correlations = (
                self.perform_statistical_analysis(X_scaled, y, behavior_labels)
            )
            
            # Behavior pattern analysis
            transitions_norm, behavior_activity_df = (
                self.analyze_behavior_patterns(X_scaled, y, behavior_labels)
            )
            
            # Save and visualize results
            results = (f_values, p_values, effect_sizes, temporal_correlations,
                      transitions_norm, behavior_activity_df, behavior_labels)
            self.save_and_visualize_results(results)
            
            print(f"\nAnalysis completed! Results saved to: {self.config.analysis_dir}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    analyzer = EnhancedAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 