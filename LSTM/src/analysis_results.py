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
    """
    结果分析器类：用于分析神经元活动数据和行为之间的关系
    主要功能包括：
    1. 加载训练好的模型和数据
    2. 分析行为和神经元活动的相关性
    3. 分析神经元活动的时间模式
    4. 分析行为转换模式
    5. 识别关键神经元
    """
    def __init__(self, config):
        """
        初始化结果分析器
        参数：
            config: 配置对象，包含所有必要的参数和路径
        """
        self.config = config
        self.processor = NeuronDataProcessor(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model_and_data(self):
        """
        加载预训练模型和预处理数据
        返回：
            model: 训练好的LSTM模型
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        """
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
        """
        分析行为和神经元活动之间的相关性
        参数：
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        返回：
            behavior_activity_df: 行为-神经元活动相关性数据框
        """
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
        plt.title('Mean Correlation between Behaviors and Neural Activity')
        plt.xlabel('Neurons')
        plt.ylabel('Behaviors')
        plt.savefig(self.config.correlation_plot)
        plt.close()
        
        return behavior_activity_df
    
    def analyze_temporal_patterns(self, X_scaled, y):
        """
        分析每种行为下神经元活动的时间模式
        参数：
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        """
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
                    plt.plot(rolling_mean[:, neuron], label=f'神经元 {neuron+1}')
                
                plt.title(f'Neural Activity Temporal Patterns During {behavior}')
                plt.xlabel('Time Window')
                plt.ylabel('Normalized Activity')
                plt.legend()
                plt.savefig(self.config.get_temporal_pattern_path(behavior))
                plt.close()
    
    def analyze_behavior_transitions(self, y):
        """
        分析行为之间的转换模式
        参数：
            y: 行为标签
        返回：
            transitions_norm: 归一化后的行为转换概率矩阵
        """
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
        plt.xlabel('Target Behavior')
        plt.ylabel('Starting Behavior')
        plt.savefig(self.config.transition_plot)
        plt.close()
        
        return transitions_norm
    
    def identify_key_neurons(self, X_scaled, y):
        """
        识别对每种行为最具判别性的关键神经元
        使用Cohen's d效应量来衡量神经元的重要性
        
        参数：
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        返回：
            behavior_importance: 每种行为的关键神经元及其重要性
        """
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
        
        plt.xlabel('Behavior')
        plt.ylabel('Effect Size')
        plt.title('Key Neurons for Each Behavior')
        plt.xticks(x_pos + width*2, self.behavior_labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.key_neurons_plot)
        plt.close()
        
        return behavior_importance

def main():
    """
    主函数：执行完整的分析流程
    1. 加载配置
    2. 初始化分析器
    3. 加载模型和数据
    4. 执行各种分析
    5. 保存分析结果
    """
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
                print(f"  神经元 {neuron}: 效应量 = {effect:.3f}")
        
        print(f"\nAnalysis completed! All results have been saved to: {config.analysis_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 