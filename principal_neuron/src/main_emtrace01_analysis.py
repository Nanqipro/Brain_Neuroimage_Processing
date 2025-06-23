"""
神经元主要分析器 - EMtrace01 数据分析脚本

该脚本用于分析神经元活动数据，包括效应量计算、关键神经元识别和可视化。
所有的路径配置都统一管理在文件开头的PathConfig类中，方便修改和维护。

使用方法：
1. 修改PathConfig类中的路径变量来指定输入输出文件
2. 在main函数中修改dataset_key来切换不同的数据集
3. 运行脚本即可生成分析结果和可视化图表

作者: Assistant
日期: 2025年
"""

import pandas as pd
import numpy as np
import os
from itertools import combinations # Add this import for combinations

# ===============================================================================
# 路径配置部分 - 所有输入输出路径的统一管理
# ===============================================================================

class PathConfig:
    """
    路径配置类：集中管理所有输入输出路径配置
    
    在这里统一修改所有文件路径，便于管理和维护
    
    使用方法：
    --------
    1. 修改以下路径变量来改变输入输出目录
    2. 确保数据文件存在于指定路径
    3. 程序将自动创建输出目录
    """
    
    def __init__(self):
        # === 输出目录配置 ===
        self.OUTPUT_DIR = "output_plots"  # 主要的图表输出目录
        
        # === 输入数据路径配置 ===
        # 原始神经元数据文件（用于计算效应量）
        self.RAW_DATA_FILE = '../data/2980240924EMtrace.xlsx'
        
        # 预计算的效应量数据文件（CSV格式）
        self.EFFECT_DATA_FILE = '../effect_size_output/effect_sizes_2980240924EMtrace.csv'
        
        # 神经元位置数据文件
        self.POSITION_DATA_FILE = '../data/EMtrace01_Max_position.csv'
        
        # === 可选的替代数据文件路径 ===
        # 在这里添加新的数据集配置，格式如下：
        # 'dataset_name': {
        #     'raw': '原始数据文件路径',
        #     'effect': '效应量数据文件路径', 
        #     'position': '位置数据文件路径'
        # }
        self.ALTERNATIVE_DATA_FILES = {
            'emtrace02': {
                'raw': '../data/EMtrace02_plus.xlsx',
                'effect': 'data/EMtrace02-3标签版.csv',
                'position': 'data/EMtrace02_Max_position.csv'
            },
            'bla6250': {
                'raw': '../data/bla6250EM0626goodtrace.xlsx',
                'effect': 'data/bla6250-3标签版.csv',
                'position': 'data/bla6250_Max_position.csv'
            }
            # 可以在这里添加更多数据集配置...
        }
        
        # === 效应量计算输出路径配置 ===
        self.EFFECT_SIZE_OUTPUT_DIR = "effect_size_output"  # 效应量计算结果输出目录
        
        # === 创建必要的目录 ===
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的输出目录存在"""
        directories = [self.OUTPUT_DIR, self.EFFECT_SIZE_OUTPUT_DIR]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建输出目录: {directory}")
    
    def get_data_paths(self, dataset_key='default'):
        """
        获取指定数据集的所有路径
        
        参数:
            dataset_key: 数据集键名 ('default', 'emtrace02', 'bla6250' 等)
        
        返回:
            dict: 包含raw, effect, position三个路径的字典
        """
        if dataset_key == 'default':
            return {
                'raw': self.RAW_DATA_FILE,
                'effect': self.EFFECT_DATA_FILE,
                'position': self.POSITION_DATA_FILE
            }
        elif dataset_key in self.ALTERNATIVE_DATA_FILES:
            return self.ALTERNATIVE_DATA_FILES[dataset_key]
        else:
            raise ValueError(f"未知的数据集键名: {dataset_key}")
    
    def list_available_datasets(self):
        """列出所有可用的数据集"""
        datasets = ['default'] + list(self.ALTERNATIVE_DATA_FILES.keys())
        print("可用的数据集:")
        for dataset in datasets:
            paths = self.get_data_paths(dataset)
            print(f"  {dataset}:")
            print(f"    原始数据: {paths['raw']}")
            print(f"    效应量数据: {paths['effect']}")
            print(f"    位置数据: {paths['position']}")

# 创建全局路径配置实例
PATH_CONFIG = PathConfig()

# 为了向后兼容，保留原始的OUTPUT_DIR变量
OUTPUT_DIR = PATH_CONFIG.OUTPUT_DIR

# ===============================================================================
# 导入其他模块
# ===============================================================================

# Assuming data_loader, config, and plotting_utils are in the same directory (src)
from data_loader import load_effect_sizes, load_neuron_positions
from config import (
    EFFECT_SIZE_THRESHOLD, BEHAVIOR_COLORS, MIXED_BEHAVIOR_COLORS,
    SHOW_BACKGROUND_NEURONS, BACKGROUND_NEURON_COLOR, 
    BACKGROUND_NEURON_SIZE, BACKGROUND_NEURON_ALPHA,
    STANDARD_KEY_NEURON_ALPHA, USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B # New config imports
)
from plotting_utils import (
    plot_single_behavior_activity_map, 
    plot_shared_neurons_map,
    plot_unique_neurons_map,
    plot_combined_9_grid # Import the new 3x3 grid plotting function
)
from effect_size_calculator import EffectSizeCalculator, load_and_calculate_effect_sizes

import matplotlib.pyplot as plt
import seaborn as sns

def analyze_effect_sizes(df_effect_sizes_long):
    """
    Analyzes the effect size data (already in long format) to help determine a threshold.
    Prints descriptive statistics and plots a histogram and boxplot.
    Saves plots to the OUTPUT_DIR.
    Assumes df_effect_sizes_long has columns: 'Behavior', 'NeuronID', 'EffectSize'.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print("Descriptive statistics for effect sizes:")
    # The describe() on the long format will include NeuronID if not careful.
    # We are interested in the distribution of EffectSize values.
    print(df_effect_sizes_long['EffectSize'].describe())

    # Plot histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_effect_sizes_long, x='EffectSize', hue='Behavior', kde=True, element="step")
    plt.title('Distribution of Effect Sizes by Behavior')
    plt.xlabel('Effect Size')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    hist_path = os.path.join(OUTPUT_DIR, 'effect_size_histogram.png')
    plt.savefig(hist_path)
    print(f"\nHistogram of effect sizes saved to {hist_path}")
    # plt.show()

    # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_effect_sizes_long, x='Behavior', y='EffectSize')
    plt.title('Box Plot of Effect Sizes by Behavior')
    plt.xlabel('Behavior')
    plt.ylabel('Effect Size')
    plt.grid(axis='y', alpha=0.75)
    box_path = os.path.join(OUTPUT_DIR, 'effect_size_boxplot.png')
    plt.savefig(box_path)
    print(f"Boxplot of effect sizes saved to {box_path}")
    # plt.show()
    
    print("\nConsider the overall distribution, the spread within each behavior,")
    print("and any natural breaks or clusters when choosing a threshold.")
    print("You might want to choose a threshold that captures the upper quartile, for example,")
    print("or a value that seems to separate 'strong' effects from weaker ones based on the plots.")

def suggest_threshold_for_neuron_count(df_effects, min_neurons=5, max_neurons=10):
    print(f"\nAnalyzing effect sizes to find a threshold that yields {min_neurons}-{max_neurons} neurons per behavior.")

    potential_t_values = set()
    # Add effect sizes around the Nth neuron mark as candidates
    for behavior in df_effects['Behavior'].unique():
        behavior_df = df_effects[df_effects['Behavior'] == behavior].copy()
        behavior_df.sort_values(by='EffectSize', ascending=False, inplace=True)
        
        if len(behavior_df) >= min_neurons:
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[min_neurons - 1], 4)) # N_min_th neuron
        if len(behavior_df) > min_neurons -1 and min_neurons > 1 :
            # Add value slightly above (N_min-1)th neuron's ES to catch exactly N_min
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[min_neurons - 2], 4) + 0.00001) 

        if len(behavior_df) >= max_neurons:
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[max_neurons - 1], 4)) # N_max_th neuron
        if len(behavior_df) > max_neurons:
            # Add value slightly above (N_max+1)th neuron's ES to ensure <= N_max neurons
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[max_neurons], 4) + 0.00001)
    
    # Add some generic sensible thresholds
    generic_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    for gt in generic_thresholds:
        potential_t_values.add(gt)
    
    candidate_thresholds = sorted([val for val in list(potential_t_values) if val > 0])

    best_t = None
    best_t_score = float('inf')
    best_t_counts = {}

    print(f"\nTesting {len(candidate_thresholds)} candidate thresholds...") # ({', '.join(f'{x:.3f}' for x in candidate_thresholds)}) 

    for t in candidate_thresholds:
        current_score_penalty = 0
        counts_for_t = {}
        all_behaviors_in_desired_range = True
        
        for behavior in df_effects['Behavior'].unique():
            behavior_df = df_effects[df_effects['Behavior'] == behavior]
            count = len(behavior_df[behavior_df['EffectSize'] >= t])
            counts_for_t[behavior] = count
            
            if not (min_neurons <= count <= max_neurons):
                all_behaviors_in_desired_range = False
            
            if count < min_neurons:
                current_score_penalty += (min_neurons - count) * 2 # Heavier penalty for too few
            elif count > max_neurons:
                current_score_penalty += (count - max_neurons)
        
        current_full_score = current_score_penalty
        if all_behaviors_in_desired_range:
            # If all counts are in range, prefer solutions that are more 'balanced'
            # (e.g., sum of squared deviations from the midpoint of the desired range)
            mid_point = (min_neurons + max_neurons) / 2.0
            balance_score = sum((c - mid_point)**2 for c in counts_for_t.values())
            current_full_score = balance_score # Override penalty, use balance score for 'good' thresholds
        
        if current_full_score < best_t_score:
            best_t_score = current_full_score
            best_t = t
            best_t_counts = counts_for_t
        elif current_full_score == best_t_score and (best_t is None or t < best_t):
             # Prefer smaller threshold if scores are identical to be slightly more inclusive
            if all_behaviors_in_desired_range == all(min_neurons <= c <= max_neurons for c in best_t_counts.values()): # only if new one is also 'good'
                best_t = t
                best_t_counts = counts_for_t

    if best_t is not None:
        print(f"\nRecommended threshold: T = {best_t:.4f}") # Using 4 decimal places for threshold
        print("Neuron counts for this threshold:")
        all_final_counts_in_range = True
        for b, c in best_t_counts.items():
            print(f"  {b}: {c} neurons")
            if not (min_neurons <= c <= max_neurons):
                all_final_counts_in_range = False
        if not all_final_counts_in_range:
             print(f"  Note: This threshold aims for the best balance, but some behaviors might be slightly outside the {min_neurons}-{max_neurons} range.")
        return best_t
    else:
        print("\nCould not automatically determine a suitable threshold from the candidates.")
        overall_75th = df_effects['EffectSize'].quantile(0.75)
        print(f"The overall 75th percentile of effect sizes is {overall_75th:.4f}. This could be a starting point for manual selection.")
        return None

def get_key_neurons(df_effects, threshold):
    """Identifies key neurons for each behavior based on the effect size threshold."""
    key_neurons_by_behavior = {}
    for behavior in df_effects['Behavior'].unique():
        behavior_df = df_effects[df_effects['Behavior'] == behavior]
        key_neuron_ids = behavior_df[behavior_df['EffectSize'] >= threshold]['NeuronID'].tolist()
        key_neurons_by_behavior[behavior] = sorted(list(set(key_neuron_ids)))
        print(f"Behavior: {behavior}, Threshold >= {threshold}, Key Neurons ({len(key_neuron_ids)}): {key_neurons_by_behavior[behavior]}")
    return key_neurons_by_behavior

def calculate_effect_sizes_from_data(neuron_data_file: str, output_dir: str = None) -> tuple:
    """
    从原始神经元数据文件计算效应量
    
    参数：
        neuron_data_file: 包含神经元活动数据和行为标签的文件路径
        output_dir: 输出目录
        
    返回：
        tuple: (效应量DataFrame (长格式), 效应量计算器实例, 计算结果字典)
    """
    print(f"\n从原始数据计算效应量: {neuron_data_file}")
    
    # 如果未指定输出目录，使用路径配置的默认目录
    if output_dir is None:
        output_dir = PATH_CONFIG.EFFECT_SIZE_OUTPUT_DIR
    
    try:
        # 使用便捷函数加载数据并计算效应量
        results = load_and_calculate_effect_sizes(
            neuron_data_path=neuron_data_file,
            behavior_col=None,  # 假设行为标签在最后一列
            output_dir=output_dir
        )
        
        # 将效应量结果转换为长格式DataFrame（与现有代码兼容）
        effect_sizes_dict = results['effect_sizes']
        behavior_labels = results['behavior_labels']
        
        # 创建长格式DataFrame
        long_format_data = []
        for behavior, effect_array in effect_sizes_dict.items():
            for neuron_idx, effect_value in enumerate(effect_array):
                long_format_data.append({
                    'Behavior': behavior,
                    'NeuronID': neuron_idx + 1,  # 1-based索引
                    'EffectSize': effect_value
                })
        
        df_effect_sizes_long = pd.DataFrame(long_format_data)
        
        print(f"效应量计算完成:")
        print(f"  行为类别: {behavior_labels}")
        print(f"  效应量数据形状: {df_effect_sizes_long.shape}")
        print(f"  输出文件: {results['output_files']['effect_sizes_csv']}")
        
        return df_effect_sizes_long, results['calculator'], results
        
    except Exception as e:
        print(f"从原始数据计算效应量失败: {str(e)}")
        print("将尝试使用预计算的效应量数据...")
        return None, None, None

def create_effect_sizes_workflow(raw_data_file: str = None, 
                                precomputed_file: str = None,
                                recalculate: bool = False) -> pd.DataFrame:
    """
    创建效应量计算工作流
    
    参数：
        raw_data_file: 原始神经元数据文件路径
        precomputed_file: 预计算的效应量文件路径
        recalculate: 是否强制重新计算效应量
        
    返回：
        pd.DataFrame: 效应量数据（长格式）
    """
    print("\n=== 效应量计算工作流 ===")
    
    # 如果指定了原始数据文件且需要重新计算，或者没有预计算文件
    if (raw_data_file and recalculate) or (raw_data_file and not precomputed_file):
        print("使用原始数据计算效应量...")
        df_long, calculator, results = calculate_effect_sizes_from_data(raw_data_file)
        
        if df_long is not None:
            print("效应量计算成功！")
            return df_long
        else:
            print("效应量计算失败，尝试加载预计算数据...")
    
    # 尝试加载预计算的效应量数据
    if precomputed_file and os.path.exists(precomputed_file):
        print(f"加载预计算的效应量数据: {precomputed_file}")
        try:
            df_long = load_effect_sizes(precomputed_file)
            if df_long is not None:
                print("预计算效应量数据加载成功！")
                return df_long
            else:
                print("预计算效应量数据加载失败")
        except Exception as e:
            print(f"加载预计算效应量数据时出错: {str(e)}")
    
    # 如果所有方法都失败，生成示例数据
    print("所有数据源都不可用，生成示例效应量数据用于演示...")
    return generate_sample_effect_sizes()

def generate_sample_effect_sizes() -> pd.DataFrame:
    """
    生成示例效应量数据用于演示
    """
    print("生成示例效应量数据...")
    
    behaviors = ['Close', 'Middle', 'Open']
    n_neurons = 50
    
    # 生成随机效应量数据
    np.random.seed(42)
    long_format_data = []
    
    for behavior in behaviors:
        # 为每种行为生成效应量，部分神经元有较高效应量
        effect_sizes = np.random.exponential(scale=0.3, size=n_neurons)
        
        # 让某些神经元对特定行为有更高的效应量
        if behavior == 'Close':
            effect_sizes[0:10] += np.random.uniform(0.4, 0.8, 10)
        elif behavior == 'Middle':
            effect_sizes[15:25] += np.random.uniform(0.4, 0.8, 10)
        else:  # Open
            effect_sizes[30:40] += np.random.uniform(0.4, 0.8, 10)
        
        for neuron_id in range(1, n_neurons + 1):
            long_format_data.append({
                'Behavior': behavior,
                'NeuronID': neuron_id,
                'EffectSize': effect_sizes[neuron_id - 1]
            })
    
    df_sample = pd.DataFrame(long_format_data)
    print(f"示例数据生成完成: {df_sample.shape}")
    return df_sample

if __name__ == "__main__":
    # ===============================================================================
    # 主程序入口 - 使用路径配置
    # ===============================================================================
    
    print("=" * 80)
    print("神经元主要分析器 - EMtrace01 数据分析")
    print("=" * 80)
    print(f"输出目录: {PATH_CONFIG.OUTPUT_DIR}")
    print(f"效应量输出目录: {PATH_CONFIG.EFFECT_SIZE_OUTPUT_DIR}")
    
    # 可以通过修改下面的dataset_key来切换不同的数据集
    # 可选值: 'default', 'emtrace02', 'bla6250'
    dataset_key = 'default'  # 默认使用EMtrace01数据集
    
    # 如果需要切换到其他数据集，取消注释下面的行
    # dataset_key = 'emtrace02'  # 使用EMtrace02数据集
    # dataset_key = 'bla6250'   # 使用bla6250数据集
    
    # 如果想查看所有可用的数据集，取消注释下面的行
    # PATH_CONFIG.list_available_datasets()
    
    # 获取当前数据集的路径配置
    try:
        data_paths = PATH_CONFIG.get_data_paths(dataset_key)
        raw_data_identifier = data_paths['raw']
        effect_data_identifier = data_paths['effect']
        position_data_identifier = data_paths['position']
        
        print(f"\n使用数据集: {dataset_key}")
        print(f"原始数据文件: {raw_data_identifier}")
        print(f"效应量数据文件: {effect_data_identifier}")
        print(f"位置数据文件: {position_data_identifier}")
        
    except ValueError as e:
        print(f"错误: {e}")
        print("使用默认路径配置...")
        # 备用路径（如果配置出错时使用）
        raw_data_identifier = '../data/EMtrace01_plus.xlsx'
        effect_data_identifier = 'data/EMtrace01-3标签版.csv'
        position_data_identifier = 'data/EMtrace01_Max_position.csv'

    # === 效应量计算工作流 ===
    print("\n=== 开始分析流程 ===")
    
    # 创建效应量计算工作流
    df_effect_sizes_transformed = create_effect_sizes_workflow(
        raw_data_file=raw_data_identifier if os.path.exists(raw_data_identifier) else None,
        precomputed_file=effect_data_identifier,
        recalculate=False  # 设置为True强制重新计算效应量
    )
    
    print(f"\nLoading neuron positions from: {position_data_identifier}")
    df_neuron_positions = load_neuron_positions(position_data_identifier)

    if df_effect_sizes_transformed is not None and df_neuron_positions is not None:
        print(f"\nUsing effect size threshold: {EFFECT_SIZE_THRESHOLD} (from config.py)")
        
        # Get key neurons based on the threshold
        key_neurons_by_behavior = get_key_neurons(df_effect_sizes_transformed, EFFECT_SIZE_THRESHOLD)
        
        # --- Prepare data for 3x3 Combined Plot ---
        print("\nPreparing data for 3x3 combined plot...")
        plot_configurations_for_3x3 = []

        # Common parameters for many plots
        common_plot_params = {
            'all_neuron_positions_df': df_neuron_positions,
            'show_background_neurons': SHOW_BACKGROUND_NEURONS,
            'background_neuron_color': BACKGROUND_NEURON_COLOR,
            'background_neuron_size': BACKGROUND_NEURON_SIZE,
            'background_neuron_alpha': BACKGROUND_NEURON_ALPHA,
            'show_title': True # Titles in subplots are desired
        }

        # Parameters specific to single and unique plots (they have key_neuron_size and key_neuron_alpha)
        single_unique_plot_params = {
            **common_plot_params,
            'key_neuron_size': 150,
            'key_neuron_alpha': STANDARD_KEY_NEURON_ALPHA
        }

        # Parameters specific to shared plots (they don't have key_neuron_size and key_neuron_alpha)
        shared_plot_params = {
            'all_neuron_positions_df': df_neuron_positions,
            'show_background_neurons': SHOW_BACKGROUND_NEURONS,
            'background_neuron_color': BACKGROUND_NEURON_COLOR,
            'background_neuron_size': BACKGROUND_NEURON_SIZE,
            'background_neuron_alpha': BACKGROUND_NEURON_ALPHA,
            'show_title': True,
            'standard_key_neuron_alpha': STANDARD_KEY_NEURON_ALPHA,
            'use_standard_alpha_for_unshared_in_scheme_b': USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B,
            'alpha_non_shared': 0.3,
            'shared_marker_size_factor': 1.5
        }

        # Ensure a consistent order for behaviors (e.g., Close, Middle, Open)
        ordered_behavior_names = [b for b in BEHAVIOR_COLORS.keys() if b in key_neurons_by_behavior]
        if len(ordered_behavior_names) < 3 and len(key_neurons_by_behavior.keys()) ==3:
             # Fallback if BEHAVIOR_COLORS doesn't cover all, though it should
             ordered_behavior_names = list(key_neurons_by_behavior.keys())[:3]
        elif len(ordered_behavior_names) != len(key_neurons_by_behavior.keys()):
            print("Warning: Behavior order for 3x3 grid might be inconsistent or incomplete based on BEHAVIOR_COLORS keys.")
            # If partial, fill up to 3 with remaining from key_neurons_by_behavior
            missing_behaviors = [b for b in key_neurons_by_behavior.keys() if b not in ordered_behavior_names]
            ordered_behavior_names.extend(missing_behaviors)
        
        temp_key_dfs = {} # To store key DFs for behaviors
        for behavior_name in ordered_behavior_names:
            neuron_ids = key_neurons_by_behavior.get(behavior_name, [])
            df = df_neuron_positions[df_neuron_positions['NeuronID'].isin(neuron_ids)] if neuron_ids else pd.DataFrame(columns=['NeuronID', 'x', 'y'])
            temp_key_dfs[behavior_name] = df

        # Row 1: Single behavior plots
        for behavior_name in ordered_behavior_names:
            params_single = {
                **single_unique_plot_params,
                'key_neurons_df': temp_key_dfs[behavior_name],
                'behavior_name': behavior_name,
                'behavior_color': BEHAVIOR_COLORS.get(behavior_name, 'gray'),
                'title': f'{behavior_name} Key' # Simpler title for subplot
            }
            plot_configurations_for_3x3.append({'plot_type': 'single', 'params': params_single})

        # Row 2: Shared neuron plots (e.g., Close-Middle, Close-Open, Middle-Open)
        # Ensure consistent pairing order for title and mixed_color_key
        behavior_pairs = list(combinations(ordered_behavior_names, 2))
        for b1, b2 in behavior_pairs: # This generates 3 pairs if ordered_behavior_names has 3 items
            ids1 = set(key_neurons_by_behavior.get(b1, []))
            ids2 = set(key_neurons_by_behavior.get(b2, []))
            shared_ids_list = sorted(list(ids1.intersection(ids2)))
            
            df_b1_all_key = temp_key_dfs[b1]
            df_b2_all_key = temp_key_dfs[b2]
            df_shared_key = df_neuron_positions[df_neuron_positions['NeuronID'].isin(shared_ids_list)]
            
            mixed_color_key = tuple(sorted((b1, b2)))
            params_shared = {
                **shared_plot_params,
                'behavior1_name': b1,
                'behavior2_name': b2,
                'behavior1_all_key_neurons_df': df_b1_all_key,
                'behavior2_all_key_neurons_df': df_b2_all_key,
                'shared_key_neurons_df': df_shared_key,
                'color1': BEHAVIOR_COLORS.get(b1, 'pink'),
                'color2': BEHAVIOR_COLORS.get(b2, 'lightblue'),
                'mixed_color': MIXED_BEHAVIOR_COLORS.get(mixed_color_key, 'purple'),
                'title': f'{b1}-{b2} Shared',
                'scheme': 'B' # Assuming Scheme B is standard for these subplots
            }
            plot_configurations_for_3x3.append({'plot_type': 'shared', 'params': params_shared})
        
        # Fill remaining shared plots if less than 3 behaviors (won't happen with 3 behaviors)
        while len(plot_configurations_for_3x3) < 6 and len(ordered_behavior_names) <2: # Max 3 single + 3 shared
             # Add placeholder for shared if not enough behaviors to make 3 pairs
            plot_configurations_for_3x3.append({
                'plot_type': 'placeholder', # Need to handle this in plot_combined_9_grid or ensure 9 configs
                'params': {'title': 'N/A'}
            })

        # Row 3: Unique neuron plots
        all_behavior_sets_for_unique = {name: set(key_neurons_by_behavior.get(name,[])) for name in ordered_behavior_names}
        for b_name in ordered_behavior_names:
            other_behaviors_neurons = set()
            for other_b_name in ordered_behavior_names:
                if b_name == other_b_name: continue
                other_behaviors_neurons.update(all_behavior_sets_for_unique.get(other_b_name, set()))
            
            unique_ids = list(all_behavior_sets_for_unique.get(b_name, set()) - other_behaviors_neurons)
            df_unique_key = df_neuron_positions[df_neuron_positions['NeuronID'].isin(unique_ids)] if unique_ids else pd.DataFrame(columns=['NeuronID', 'x', 'y'])

            params_unique = {
                **single_unique_plot_params,
                'unique_neurons_df': df_unique_key,
                'behavior_name': b_name,
                'behavior_color': BEHAVIOR_COLORS.get(b_name, 'gray'),
                'title': f'{b_name} Unique'
            }
            plot_configurations_for_3x3.append({'plot_type': 'unique', 'params': params_unique})

        # Ensure we have exactly 9 configurations for the 3x3 grid
        # If there were fewer than 3 behaviors, some slots might be empty or need placeholders.
        # The logic above tries to fill based on ordered_behavior_names. If still not 9, add placeholders.
        # This placeholder handling should ideally be more robust or data generation should guarantee data for 9 plots.
        while len(plot_configurations_for_3x3) < 9:
            print(f"Warning: Not enough plot configurations for 3x3 grid (currently {len(plot_configurations_for_3x3)}). Adding placeholder(s).")
            plot_configurations_for_3x3.append({
                'plot_type': 'placeholder', 
                'params': {'title': 'Empty Slot'} # Placeholder title
            })
        
        if len(plot_configurations_for_3x3) > 9:
             print("Warning: More than 9 plot configurations generated. Truncating to 9 for 3x3 grid.")
             plot_configurations_for_3x3 = plot_configurations_for_3x3[:9]

        # --- Generate 3x3 Combined Plot ---
        if len(plot_configurations_for_3x3) == 9:
            print("\nGenerating 3x3 combined plot...")
            combined_plot_filename = "plot_all_behaviors_3x3_grid.png"
            combined_output_path = os.path.join(OUTPUT_DIR, combined_plot_filename)
            
            plot_combined_9_grid(
                plot_configurations=plot_configurations_for_3x3,
                output_path=combined_output_path,
                main_title_text=f"Comprehensive View: Neuron Activity Patterns (Effect Size >= {EFFECT_SIZE_THRESHOLD})"
            )
        else:
            print("Error: Could not prepare exactly 9 plot configurations for the 3x3 grid. Skipping combined plot.")

        print("\nAll plots generated.")

    else:
        if df_effect_sizes_transformed is None:
            print("Could not load effect sizes. Please check 'data_loader.py' and the CSV data.")
        if df_neuron_positions is None:
            print("Could not load neuron positions. Please check 'data_loader.py' and the CSV data.")
        # print("Error: Could not load data.") # Covered by more specific messages above

    # ... (suggest_threshold_for_neuron_count function definition if kept for reference) ... 
    # ... (suggest_threshold_for_neuron_count function definition if kept for reference) ... 