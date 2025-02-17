import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from kmeans_lstm_analysis import NeuronLSTM, NeuronDataProcessor
from analysis_config import AnalysisConfig
import torch.nn.functional as F

class ResultAnalyzer:
    def __init__(self, config):
        self.config = config
        self.processor = NeuronDataProcessor(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model_and_data(self):
        # Load and preprocess data
        X_scaled, y = self.processor.preprocess_data()
        self.behavior_labels = self.processor.label_encoder.classes_
        
        # Load trained model
        input_size = X_scaled.shape[1] + 1  # +1 for cluster label
        num_classes = len(np.unique(y))
        
        model = NeuronLSTM(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        model.load_state_dict(torch.load(self.config.model_path))
        model.eval()
        
        return model, X_scaled, y
    
    def analyze_behavior_neuron_correlation(self, X_scaled, y):
        """Analyze correlation between behaviors and neuron activities"""
        # Calculate mean activity for each behavior
        behavior_means = {}
        for behavior_idx in range(len(self.behavior_labels)):
            behavior_mask = (y == behavior_idx)
            behavior_means[self.behavior_labels[behavior_idx]] = np.mean(X_scaled[behavior_mask], axis=0)
        
        # Create correlation heatmap
        behavior_activity_df = pd.DataFrame(behavior_means).T
        behavior_activity_df.columns = [f'Neuron {i+1}' for i in range(behavior_activity_df.shape[1])]
        
        plt.figure(figsize=self.config.figure_sizes['correlation'])
        sns.heatmap(behavior_activity_df, cmap='coolwarm', center=0)
        plt.title('Mean Neuron Activity by Behavior')
        plt.xlabel('Neurons')
        plt.ylabel('Behaviors')
        plt.savefig(self.config.correlation_plot)
        plt.close()
        
        return behavior_activity_df
    
    def analyze_temporal_patterns(self, X_scaled, y):
        """Analyze temporal patterns of neuron activity for each behavior"""
        window_size = self.config.temporal_window_size
        
        for behavior_idx, behavior in enumerate(self.behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_data = X_scaled[behavior_mask]
            
            if len(behavior_data) > window_size:
                # Calculate moving average
                rolling_mean = np.array([np.mean(behavior_data[i:i+window_size], axis=0) 
                                       for i in range(0, len(behavior_data)-window_size, window_size)])
                
                plt.figure(figsize=self.config.figure_sizes['temporal'])
                for neuron in range(min(5, rolling_mean.shape[1])):
                    plt.plot(rolling_mean[:, neuron], label=f'Neuron {neuron+1}')
                
                plt.title(f'Temporal Pattern of Neuron Activity During {behavior}')
                plt.xlabel('Time Windows')
                plt.ylabel('Standardized Activity')
                plt.legend()
                plt.savefig(self.config.get_temporal_pattern_path(behavior))
                plt.close()
    
    def analyze_behavior_transitions(self, y):
        """Analyze transitions between behaviors"""
        transitions = np.zeros((len(self.behavior_labels), len(self.behavior_labels)))
        
        for i in range(len(y)-1):
            transitions[y[i], y[i+1]] += 1
        
        # Normalize transitions
        row_sums = transitions.sum(axis=1)
        transitions_norm = transitions / row_sums[:, np.newaxis]
        
        plt.figure(figsize=self.config.figure_sizes['transitions'])
        sns.heatmap(transitions_norm, 
                   xticklabels=self.behavior_labels,
                   yticklabels=self.behavior_labels,
                   cmap='YlOrRd')
        plt.title('Behavior Transition Probabilities')
        plt.xlabel('To Behavior')
        plt.ylabel('From Behavior')
        plt.savefig(self.config.transition_plot)
        plt.close()
        
        return transitions_norm
    
    def identify_key_neurons(self, X_scaled, y):
        """Identify neurons that are most discriminative for each behavior"""
        behavior_importance = {}
        
        for behavior_idx, behavior in enumerate(self.behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_data = X_scaled[behavior_mask]
            other_data = X_scaled[~behavior_mask]
            
            # Calculate effect size (Cohen's d) for each neuron
            behavior_mean = np.mean(behavior_data, axis=0)
            other_mean = np.mean(other_data, axis=0)
            behavior_std = np.std(behavior_data, axis=0)
            other_std = np.std(other_data, axis=0)
            
            pooled_std = np.sqrt((behavior_std**2 + other_std**2) / 2)
            effect_size = np.abs(behavior_mean - other_mean) / pooled_std
            
            # Get top neurons
            top_neurons = np.argsort(effect_size)[-self.config.top_neurons_count:][::-1]
            behavior_importance[behavior] = {
                'neurons': top_neurons + 1,  # +1 for 1-based indexing
                'effect_sizes': effect_size[top_neurons]
            }
        
        # Plot results
        plt.figure(figsize=self.config.figure_sizes['key_neurons'])
        x_pos = np.arange(len(behavior_importance))
        width = 0.15
        
        for i in range(self.config.top_neurons_count):
            effect_sizes = [behavior_importance[b]['effect_sizes'][i] for b in self.behavior_labels]
            plt.bar(x_pos + i*width, effect_sizes, width, label=f'Top {i+1}')
        
        plt.xlabel('Behaviors')
        plt.ylabel('Effect Size')
        plt.title('Most Important Neurons for Each Behavior')
        plt.xticks(x_pos + width*2, self.behavior_labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.key_neurons_plot)
        plt.close()
        
        return behavior_importance

def main():
    # Initialize configuration
    config = AnalysisConfig()
    
    try:
        # Setup and validate directories
        config.setup_directories()
        config.validate_paths()
        
        # Initialize analyzer
        analyzer = ResultAnalyzer(config)
        
        print("Loading model and data...")
        model, X_scaled, y = analyzer.load_model_and_data()
        
        print("\nAnalyzing behavior-neuron correlations...")
        behavior_activity_df = analyzer.analyze_behavior_neuron_correlation(X_scaled, y)
        print(f"Correlation analysis completed. Check: {config.correlation_plot}")
        
        print("\nAnalyzing temporal patterns...")
        analyzer.analyze_temporal_patterns(X_scaled, y)
        print(f"Temporal analysis completed. Check patterns in: {config.temporal_pattern_dir}")
        
        print("\nAnalyzing behavior transitions...")
        transitions = analyzer.analyze_behavior_transitions(y)
        print(f"Transition analysis completed. Check: {config.transition_plot}")
        
        print("\nIdentifying key neurons for each behavior...")
        behavior_importance = analyzer.identify_key_neurons(X_scaled, y)
        print("\nKey neurons for each behavior:")
        for behavior, data in behavior_importance.items():
            print(f"\n{behavior}:")
            for i, (neuron, effect) in enumerate(zip(data['neurons'], data['effect_sizes'])):
                print(f"  Neuron {neuron}: Effect size = {effect:.3f}")
        
        print(f"\nAnalysis completed! All results have been saved to: {config.analysis_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 