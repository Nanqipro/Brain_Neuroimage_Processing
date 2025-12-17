"""
神经元主要分析器 - 高架数据分析脚本

该脚本用于分析神经元活动数据，包括效应量计算、关键神经元识别和可视化。
所有的路径配置都统一管理在文件开头的PathConfig类中，方便修改和维护。

使用方法：
1. 修改PathConfig类中的路径变量来指定输入输出文件
2. 在main函数中修改dataset_key来切换不同的数据集
3. 运行脚本即可生成分析结果和可视化图表

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
        # 获取当前脚本的目录，然后构建绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)  # 上一级目录（principal_neuron）
        
        self.BASE_OUTPUT_DIR = os.path.join(project_dir, "output_plots")  # 基础输出目录
        self.BASE_EFFECT_SIZE_OUTPUT_DIR = os.path.join(project_dir, "effect_size_output")  # 基础效应量输出目录
        self.DATA_DIR = os.path.join(project_dir, "data")  # 数据目录
        
        # === 数据集配置：完整的数据集清单 ===
        # 每个数据集包含三个文件：原始数据、效应量数据、位置数据
        self.DATASETS = {
            # EMtrace系列数据集
            'emtrace01': {
                'name': 'EMtrace01数据集',
                'raw': os.path.join(self.DATA_DIR, 'EMtrace01.xlsx'),
                'effect': os.path.join(self.DATA_DIR, 'EMtrace01-3标签版.csv'),
                'position': os.path.join(self.DATA_DIR, 'EMtrace01_Max_position.csv'),
                'description': 'EMtrace01神经元活动数据（3标签版）'
            },
            'emtrace01_plus': {
                'name': 'EMtrace01增强数据集',
                'raw': os.path.join(self.DATA_DIR, 'EMtrace01_plus.xlsx'),
                'effect': os.path.join(self.DATA_DIR, 'EMtrace01-3标签版.csv'),  # 复用同一个效应量文件
                'position': os.path.join(self.DATA_DIR, 'EMtrace01_Max_position.csv'),
                'description': 'EMtrace01增强版神经元活动数据'
            },
            'emtrace02': {
                'name': 'EMtrace02数据集',
                'raw': os.path.join(self.DATA_DIR, 'EMtrace02.xlsx'),
                'effect': os.path.join(self.DATA_DIR, 'EMtrace02-3标签版.csv'),
                'position': os.path.join(self.DATA_DIR, 'EMtrace02_Max_position.csv'),
                'description': 'EMtrace02神经元活动数据（3标签版）'
            },
            'emtrace02_plus': {
                'name': 'EMtrace02增强数据集',
                'raw': os.path.join(self.DATA_DIR, 'EMtrace02_plus.xlsx'),
                'effect': os.path.join(self.DATA_DIR, 'EMtrace02-3标签版.csv'),
                'position': os.path.join(self.DATA_DIR, 'EMtrace02_Max_position.csv'),
                'description': 'EMtrace02增强版神经元活动数据'
            },
            
            # 其他数据集
            '2980': {
                'name': '2980 datasets',
                'raw': os.path.join(self.DATA_DIR, '2980240924EMtrace.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_2980240924EMtrace.csv'),
                'position': os.path.join(self.DATA_DIR, '2980_Max_position.csv'),
                'description': '2980神经元活动数据'
            },
            '2980_plus': {
                'name': '2980 datasets',
                'raw': os.path.join(self.DATA_DIR, '2980240924EMtrace_plus.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_2980240924EMtrace_plus.csv'),
                'position': os.path.join(self.DATA_DIR, '2980_Max_position.csv'),
                'description': '2980增强版神经元活动数据'
            },
            'bla6250': {
                'name': 'BLA6250 datasets',
                'raw': os.path.join(self.DATA_DIR, 'bla6250EM0626goodtrace.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_bla6250EM0626goodtrace.csv'),
                'position': os.path.join(self.DATA_DIR, '6250_Max_position.csv'),
                'description': 'BLA6250神经元活动数据'
            },
            'bla6250_plus': {
                'name': 'BLA6250 datasets',
                'raw': os.path.join(self.DATA_DIR, 'bla6250EM0626goodtrace_plus.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_bla6250EM0626goodtrace_plus.csv'),
                'position': os.path.join(self.DATA_DIR, '6250_Max_position.csv'),
                'description': 'BLA6250增强版神经元活动数据'
            },
            
            # # Day系列数据集
            # 'day3': {
            #     'name': 'Day3数据集',
            #     'raw': None,  # 只有效应量数据
            #     'effect': os.path.join(self.DATA_DIR, 'day3.csv'),
            #     'position': os.path.join(self.DATA_DIR, 'Day3_Max_position.csv'),
            #     'description': 'Day3神经元活动数据'
            # },
            # 'day6': {
            #     'name': 'Day6数据集',
            #     'raw': None,
            #     'effect': os.path.join(self.DATA_DIR, 'day6.csv'),
            #     'position': os.path.join(self.DATA_DIR, 'Day6_Max_position.csv'),
            #     'description': 'Day6神经元活动数据'
            # },
            # 'day9': {
            #     'name': 'Day9数据集',
            #     'raw': None,
            #     'effect': os.path.join(self.DATA_DIR, 'day9.csv'),
            #     'position': os.path.join(self.DATA_DIR, 'Day9_Max_position.csv'),
            #     'description': 'Day9神经元活动数据'
            # },
            # 矿场系列
            '29800930openfield': {
                'name': '29800930openfield',
                'raw':  os.path.join(self.DATA_DIR, 'no.29800930openfield_CellVideo0_corrected_0_cell_trace.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_no.29800930openfield_CellVideo0_corrected_0_cell_trace.csv'),
                'position': os.path.join(self.DATA_DIR, 'no.29800930openfield神经元编号位置图.csv'),
                'description': '29800930openfield神经元活动数据'
            },
            '29800924openfield': {
                'name': '29800924openfield',
                'raw':  os.path.join(self.DATA_DIR, 'no.2980240924openfield_CellVideo0_corrected_0_cell_trace.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_no.2980240924openfield_CellVideo0_corrected_0_cell_trace.csv'),
                'position': os.path.join(self.DATA_DIR, 'no.2980240924openfield神经元编号位置图.csv'),
                'description': '29800924openfield神经元活动数据'
            },
            '5355EM': {
                'name': '5355EM',
                'raw':  os.path.join(self.DATA_DIR, 'NO5355EM20251106_cell_trace.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_NO5355EM20251106_cell_trace.csv'),
                'position': os.path.join(self.DATA_DIR, 'NO5355EM20251106_cell_trace.csv'),
                'description': '5355EM神经元活动数据'
            },
            '5355EM-plus': {
                'name': '5355EM',
                'raw':  os.path.join(self.DATA_DIR, 'NO5355EM20251106_cell_trace-三区域.xlsx'),
                'effect': os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, 'effect_sizes_NO5355EM20251106_cell_trace-三区域.csv'),
                'position': os.path.join(self.DATA_DIR, 'NO5355EM20251106_cell_trace.csv'),
                'description': '5355EM神经元活动数据'
            }
        }
        
        # === 默认数据集设置 ===
        self.DEFAULT_DATASET = '5355EM'  
        
        
        # === 创建必要的目录 ===
        self._ensure_base_directories()
    
    def _ensure_base_directories(self):
        """确保基础输出目录存在"""
        base_directories = [self.BASE_OUTPUT_DIR, self.BASE_EFFECT_SIZE_OUTPUT_DIR]
        for directory in base_directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建基础输出目录: {directory}")
    
    def get_dataset_output_dir(self, dataset_key):
        """
        获取指定数据集的专用输出目录
        
        参数:
            dataset_key: 数据集键名
            
        返回:
            str: 数据集专用输出目录路径
        """
        if dataset_key not in self.DATASETS:
            raise ValueError(f"未知的数据集键名: {dataset_key}")
        
        dataset_output_dir = os.path.join(self.BASE_OUTPUT_DIR, dataset_key)
        
        # 确保目录存在
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)
            print(f"创建数据集专用输出目录: {dataset_output_dir}")
            
        return dataset_output_dir
    
    def get_dataset_effect_size_output_dir(self, dataset_key):
        """
        获取指定数据集的专用效应量输出目录
        
        参数:
            dataset_key: 数据集键名
            
        返回:
            str: 数据集专用效应量输出目录路径
        """
        if dataset_key not in self.DATASETS:
            raise ValueError(f"未知的数据集键名: {dataset_key}")
        
        effect_size_output_dir = os.path.join(self.BASE_EFFECT_SIZE_OUTPUT_DIR, dataset_key)
        
        # 确保目录存在
        if not os.path.exists(effect_size_output_dir):
            os.makedirs(effect_size_output_dir)
            print(f"创建数据集专用效应量输出目录: {effect_size_output_dir}")
            
        return effect_size_output_dir
    
    def get_data_paths(self, dataset_key=None):
        """
        获取指定数据集的所有路径
        
        参数:
            dataset_key: 数据集键名，如果为None则使用默认数据集
        
        返回:
            dict: 包含raw, effect, position三个路径的字典
        """
        if dataset_key is None:
            dataset_key = self.DEFAULT_DATASET
            
        if dataset_key not in self.DATASETS:
            raise ValueError(f"未知的数据集键名: {dataset_key}。可用数据集: {list(self.DATASETS.keys())}")
        
        dataset_info = self.DATASETS[dataset_key]
        return {
            'raw': dataset_info['raw'],
            'effect': dataset_info['effect'],
            'position': dataset_info['position'],
            'name': dataset_info['name'],
            'description': dataset_info['description'],
            'output_dir': self.get_dataset_output_dir(dataset_key),  # 添加专用输出目录
            'effect_size_output_dir': self.get_dataset_effect_size_output_dir(dataset_key)  # 添加专用效应量输出目录
        }
    
    def list_available_datasets(self):
        """列出所有可用的数据集"""
        print("=" * 60)
        print("可用的数据集:")
        print("=" * 60)
        for key, dataset in self.DATASETS.items():
            print(f"\n📁 数据集键名: '{key}'")
            print(f"   名称: {dataset['name']}")
            print(f"   描述: {dataset['description']}")
            print(f"   原始数据: {dataset['raw'] or '无'}")
            print(f"   效应量数据: {dataset['effect'] or '需要计算'}")
            print(f"   位置数据: {dataset['position'] or '无'}")
            print(f"   输出目录: {self.BASE_OUTPUT_DIR}/{key}/")
        print("=" * 60)
    
    def check_dataset_availability(self, dataset_key=None):
        """
        检查指定数据集的文件是否存在
        
        参数:
            dataset_key: 数据集键名
            
        返回:
            dict: 包含各文件存在状态的字典
        """
        if dataset_key is None:
            dataset_key = self.DEFAULT_DATASET
            
        paths = self.get_data_paths(dataset_key)
        availability = {
            'dataset_key': dataset_key,
            'dataset_name': paths['name'],
            'raw_exists': paths['raw'] and os.path.exists(paths['raw']),
            'effect_exists': paths['effect'] and os.path.exists(paths['effect']),
            'position_exists': paths['position'] and os.path.exists(paths['position']),
            'raw_path': paths['raw'],
            'effect_path': paths['effect'],
            'position_path': paths['position'],
            'output_dir': paths['output_dir'],
            'effect_size_output_dir': paths['effect_size_output_dir']
        }
        
        availability['is_usable'] = (
            availability['position_exists'] and 
            (availability['effect_exists'] or availability['raw_exists'])
        )
        
        return availability
    
    def print_dataset_status(self, dataset_key=None):
        """打印数据集的详细状态信息"""
        status = self.check_dataset_availability(dataset_key)
        
        print(f"\n📊 数据集状态检查: {status['dataset_name']} ('{status['dataset_key']}')")
        print("-" * 50)
        
        # 检查各文件状态
        files_to_check = [
            ('原始数据文件', status['raw_path'], status['raw_exists']),
            ('效应量数据文件', status['effect_path'], status['effect_exists']),
            ('位置数据文件', status['position_path'], status['position_exists'])
        ]
        
        for file_type, file_path, exists in files_to_check:
            if file_path:
                status_icon = "✅" if exists else "❌"
                print(f"{status_icon} {file_type}: {file_path}")
            else:
                print(f"⚪ {file_type}: 无")
        
        # 输出目录信息
        print(f"\n📂 输出目录:")
        print(f"   图表输出: {status['output_dir']}")
        print(f"   效应量输出: {status['effect_size_output_dir']}")
        
        # 总体可用性
        if status['is_usable']:
            print(f"\n✅ 数据集可用！")
        else:
            print(f"\n❌ 数据集不可用 - 缺少必要文件")
            
        return status
    
    def get_recommended_dataset(self):
        """获取推荐的可用数据集"""
        # 按优先级检查数据集
        priority_order = ['emtrace01', 'emtrace02', 'emtrace01_plus', 'emtrace02_plus', '2980', 'bla6250']
        
        for dataset_key in priority_order:
            if dataset_key in self.DATASETS:
                status = self.check_dataset_availability(dataset_key)
                if status['is_usable']:
                    return dataset_key
        
        # 如果优先级列表中没有可用的，检查所有数据集
        for dataset_key in self.DATASETS.keys():
            status = self.check_dataset_availability(dataset_key)
            if status['is_usable']:
                return dataset_key
        
        return None

# 创建全局路径配置实例
PATH_CONFIG = PathConfig()

# 为了向后兼容，保留原始的OUTPUT_DIR变量
OUTPUT_DIR = PATH_CONFIG.BASE_OUTPUT_DIR

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
    plot_combined_9_grid,
    plot_3d_combined_neuron_distribution,
    plot_3d_activation_categories_combined
)
from effect_size_calculator import EffectSizeCalculator, load_and_calculate_effect_sizes

import matplotlib.pyplot as plt
import seaborn as sns

df_effect_sizes_transformed = None
df_neuron_positions = None

def _load_effect_sizes_flexible(csv_path):
    df = pd.read_csv(csv_path)
    if 'Behavior' not in df.columns:
        raise ValueError('效应量CSV缺少Behavior列')
    neuron_cols = [c for c in df.columns if c.startswith('Neuron_')]
    if not neuron_cols:
        neuron_cols = [c for c in df.columns if c.startswith('n') and c[1:].isdigit()]
    if not neuron_cols:
        raise ValueError('未找到神经元列')
    df_use = df[['Behavior'] + neuron_cols]
    df_melted = df_use.melt(id_vars=['Behavior'], var_name='Neuron_str', value_name='EffectSize')
    def _to_id(s):
        if s.startswith('Neuron_'):
            return int(s.replace('Neuron_', ''))
        return int(s.replace('n', ''))
    df_melted['NeuronID'] = df_melted['Neuron_str'].apply(_to_id)
    result_df = df_melted[['Behavior', 'NeuronID', 'EffectSize']]
    print(f"转换后的长格式数据形状: {result_df.shape}")
    return result_df

def _load_neuron_positions_flexible(csv_path):
    df = pd.read_csv(csv_path)
    if {'NeuronID', 'x', 'y'}.issubset(df.columns):
        return df[['NeuronID', 'x', 'y']]
    if {'number', 'relative_x', 'relative_y'}.issubset(df.columns):
        df = df.rename(columns={'number': 'NeuronID', 'relative_x': 'x', 'relative_y': 'y'})
        df['NeuronID'] = df['NeuronID'].astype(int)
        return df[['NeuronID', 'x', 'y']]
    raise ValueError('位置CSV列不匹配')

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
    """
    根据效应量阈值识别每种行为的关键神经元
    
    参数:
        df_effects: 效应量数据DataFrame，包含Behavior、NeuronID、EffectSize列
        threshold: 效应量阈值
    
    返回:
        dict: 每种行为对应的关键神经元ID列表
    """
    key_neurons_by_behavior = {}
    
    # 过滤掉无效的行为名称（如nan值）
    valid_behaviors = df_effects['Behavior'].dropna().unique()
    
    for behavior in valid_behaviors:
        # 跳过nan值或空值
        if pd.isna(behavior) or behavior == '' or str(behavior).lower() == 'nan':
            continue
            
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
        output_dir = PATH_CONFIG.BASE_EFFECT_SIZE_OUTPUT_DIR
    
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
            df_long = _load_effect_sizes_flexible(precomputed_file)
            print("预计算效应量数据加载成功！")
            return df_long
        except Exception as e:
            print(f"加载预计算效应量数据时出错: {str(e)}")
    
    # 如果所有方法都失败，生成示例数据
    print("所有数据源都不可用，生成示例效应量数据用于演示...")
    return None
    # return generate_sample_effect_sizes()

# def generate_sample_effect_sizes() -> pd.DataFrame:
#     """
#     生成示例效应量数据用于演示
#     """
#     print("生成示例效应量数据...")
    
#     behaviors = ['Close', 'Middle', 'Open']
#     n_neurons = 50
    
#     # 生成随机效应量数据
#     np.random.seed(42)
#     long_format_data = []
    
#     for behavior in behaviors:
#         # 为每种行为生成效应量，部分神经元有较高效应量
#         effect_sizes = np.random.exponential(scale=0.3, size=n_neurons)
        
#         # 让某些神经元对特定行为有更高的效应量
#         if behavior == 'Close':
#             effect_sizes[0:10] += np.random.uniform(0.4, 0.8, 10)
#         elif behavior == 'Middle':
#             effect_sizes[15:25] += np.random.uniform(0.4, 0.8, 10)
#         else:  # Open
#             effect_sizes[30:40] += np.random.uniform(0.4, 0.8, 10)
        
#         for neuron_id in range(1, n_neurons + 1):
#             long_format_data.append({
#                 'Behavior': behavior,
#                 'NeuronID': neuron_id,
#                 'EffectSize': effect_sizes[neuron_id - 1]
#             })
    
#     df_sample = pd.DataFrame(long_format_data)
#     print(f"示例数据生成完成: {df_sample.shape}")
#     return df_sample

if __name__ == "__main__":
    # ===============================================================================
    # 主程序入口 - 使用路径配置
    # ===============================================================================
    
    print("=" * 80)
    print("神经元主要分析器 - 多数据集支持版本")
    print("=" * 80)
    print(f"输出目录: {PATH_CONFIG.BASE_OUTPUT_DIR}")
    print(f"效应量输出目录: {PATH_CONFIG.BASE_EFFECT_SIZE_OUTPUT_DIR}")
    
    # ===============================================================================
    # 数据集选择配置
    # ===============================================================================
    
    # 🔧 在这里修改 dataset_key 来切换不同的数据集
    # 可选值: 'emtrace01', 'emtrace02', 'emtrace01_plus', 'emtrace02_plus', 
    #         '2980', 'bla6250', 'day3', 'day6', 'day9'
    # 设置为 None 会自动选择可用的数据集
    
    # dataset_key = None # 🔧 修改这里来指定数据集，None表示自动选择
    # dataset_key = 'emtrace01'    # 使用EMtrace01数据集
    # dataset_key = 'emtrace02'    # 使用EMtrace02数据集  
    # dataset_key = '2980'         # 使用2980数据集
    # dataset_key = '2980_plus'      # 使用2980增强版数据集
    # dataset_key = 'bla6250'      # 使用BLA6250数据集
    # dataset_key = 'bla6250_plus' # 使用BLA6250增强版数据集
    # dataset_key = 'day3'         # 使用Day3数据集
    
    # dataset_key = '29800930openfield'    # 使用29800930openfield数据集
    # dataset_key = '29800924openfield'    # 使用29800924openfield数据集
    dataset_key = '5355EM-plus'    # 使用5355EM数据集
    
    # ===============================================================================
    # 智能数据集选择和验证
    # ===============================================================================
    
    # 显示所有可用数据集
    print("\n" + "=" * 60)
    print("🔍 检查可用数据集...")
    PATH_CONFIG.list_available_datasets()
    
    # 如果没有指定数据集，自动选择可用的数据集
    if dataset_key is None:
        print("\n🤖 未指定数据集，正在自动选择最佳可用数据集...")
        dataset_key = PATH_CONFIG.get_recommended_dataset()
        if dataset_key is None:
            print("❌ 错误：没有找到任何可用的数据集！")
            print("请检查data目录中的文件是否存在。")
            exit(1)
        else:
            print(f"✅ 自动选择数据集: {dataset_key}")
    
    # 验证选择的数据集
    print(f"\n🔍 验证数据集: {dataset_key}")
    status = PATH_CONFIG.print_dataset_status(dataset_key)
    
    if not status['is_usable']:
        print(f"\n❌ 错误：数据集 '{dataset_key}' 不可用！")
        print("请选择其他数据集或检查文件路径。")
        
        # 尝试推荐替代数据集
        alternative = PATH_CONFIG.get_recommended_dataset()
        if alternative and alternative != dataset_key:
            print(f"\n💡 建议使用数据集: {alternative}")
            PATH_CONFIG.print_dataset_status(alternative)
        exit(1)
    
    # 获取当前数据集的路径配置
    try:
        data_paths = PATH_CONFIG.get_data_paths(dataset_key)
        raw_data_identifier = data_paths['raw']
        effect_data_identifier = data_paths['effect']
        position_data_identifier = data_paths['position']
        
        print(f"\n✅ 使用数据集: {data_paths['name']} ('{dataset_key}')")
        print(f"📄 描述: {data_paths['description']}")
        print(f"📁 原始数据文件: {raw_data_identifier or '无'}")
        print(f"📁 效应量数据文件: {effect_data_identifier or '需要计算'}")
        print(f"📁 位置数据文件: {position_data_identifier}")
        
    except ValueError as e:
        print(f"❌ 错误: {e}")
        exit(1)

    # === 效应量计算工作流 ===
    print("\n" + "=" * 60)
    print("🚀 开始分析流程")
    print("=" * 60)
    
    # 创建效应量计算工作流
    df_effect_sizes_transformed = create_effect_sizes_workflow(
        raw_data_file=raw_data_identifier if raw_data_identifier and os.path.exists(raw_data_identifier) else None,
        precomputed_file=effect_data_identifier if effect_data_identifier and os.path.exists(effect_data_identifier) else None,
        recalculate=False  # 设置为True强制重新计算效应量
    )
    
    print(f"\n📍 Loading neuron positions from: {position_data_identifier}")
    df_neuron_positions = _load_neuron_positions_flexible(position_data_identifier)

    if df_effect_sizes_transformed is not None and df_neuron_positions is not None:
        print(f"\n🎯 Using effect size threshold: {EFFECT_SIZE_THRESHOLD} (from config.py)")
        
        # Get key neurons based on the threshold
        key_neurons_by_behavior = get_key_neurons(df_effect_sizes_transformed, EFFECT_SIZE_THRESHOLD)
        
        # 获取所有有效的行为名称（排除nan等无效值）
        all_behaviors = list(key_neurons_by_behavior.keys())
        print(f"\n📊 发现 {len(all_behaviors)} 个有效行为标签: {all_behaviors}")
        
        # ===============================================================================
        # 生成单独的图表（每个行为、每对行为共享、每个行为独有）
        # ===============================================================================
        
        print(f"\n🎨 开始生成单独的图表...")
        
        # --- 1. 为每个行为生成关键神经元图 ---
        print(f"\n📈 生成每个行为的关键神经元图...")
        
        for behavior_name in all_behaviors:
            print(f"  🔸 生成 {behavior_name} 行为的关键神经元图...")
            
            # 获取该行为的关键神经元
            neuron_ids = key_neurons_by_behavior.get(behavior_name, [])
            if not neuron_ids:
                print(f"    ⚠️  {behavior_name} 没有关键神经元，跳过...")
                continue
                
            key_neurons_df = df_neuron_positions[df_neuron_positions['NeuronID'].isin(neuron_ids)]
            
            # 生成图片文件名（处理特殊字符）
            safe_behavior_name = behavior_name.replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')
            output_filename = f"behavior_{safe_behavior_name}_key_neurons.png"
            output_path = os.path.join(data_paths['output_dir'], output_filename)
            
            # 使用现有的绘图函数
            try:
                plot_single_behavior_activity_map(
                    key_neurons_df=key_neurons_df,
                    behavior_name=behavior_name,
                    behavior_color=BEHAVIOR_COLORS.get(behavior_name, 'gray'),
                    title=f'{behavior_name} Key Neurons',
                    output_path=output_path,
                    all_neuron_positions_df=df_neuron_positions,
                    show_background_neurons=SHOW_BACKGROUND_NEURONS,
                    background_neuron_color=BACKGROUND_NEURON_COLOR,
                    background_neuron_size=BACKGROUND_NEURON_SIZE,
                    background_neuron_alpha=BACKGROUND_NEURON_ALPHA,
                    key_neuron_size=300,
                    key_neuron_alpha=STANDARD_KEY_NEURON_ALPHA,
                    show_title=True
                )
                print(f"    ✅ 保存到: {output_filename}")
            except Exception as e:
                print(f"    ❌ 生成 {behavior_name} 图表失败: {str(e)}")
        
        # --- 2. 为每对行为生成共享神经元图 ---
        print(f"\n🔗 生成每对行为的共享神经元图...")
        
        behavior_pairs = list(combinations(all_behaviors, 2))
        print(f"  📊 总共需要生成 {len(behavior_pairs)} 个共享神经元图")
        
        for b1, b2 in behavior_pairs:
            print(f"  🔸 生成 {b1} 与 {b2} 的共享神经元图...")
            
            # 获取两个行为的关键神经元集合
            ids1 = set(key_neurons_by_behavior.get(b1, []))
            ids2 = set(key_neurons_by_behavior.get(b2, []))
            shared_ids = list(ids1.intersection(ids2))
            
            if not shared_ids:
                print(f"    ⚠️  {b1} 与 {b2} 没有共享关键神经元，跳过...")
                continue
            
            # 获取数据框
            df_b1_all_key = df_neuron_positions[df_neuron_positions['NeuronID'].isin(list(ids1))]
            df_b2_all_key = df_neuron_positions[df_neuron_positions['NeuronID'].isin(list(ids2))]
            df_shared_key = df_neuron_positions[df_neuron_positions['NeuronID'].isin(shared_ids)]
            
            # 生成安全的文件名
            safe_b1 = b1.replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')
            safe_b2 = b2.replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')
            output_filename = f"shared_{safe_b1}_and_{safe_b2}.png"
            output_path = os.path.join(data_paths['output_dir'], output_filename)
            
            # 使用现有的绘图函数
            try:
                # 获取混合颜色
                mixed_color_key = tuple(sorted((b1, b2)))
                mixed_color = MIXED_BEHAVIOR_COLORS.get(mixed_color_key, 'purple')
                
                plot_shared_neurons_map(
                    behavior1_name=b1,
                    behavior2_name=b2,
                    behavior1_all_key_neurons_df=df_b1_all_key,
                    behavior2_all_key_neurons_df=df_b2_all_key,
                    shared_key_neurons_df=df_shared_key,
                    color1=BEHAVIOR_COLORS.get(b1, 'pink'),
                    color2=BEHAVIOR_COLORS.get(b2, 'lightblue'),
                    mixed_color=mixed_color,
                    title=f'{b1}-{b2} Shared Neurons',
                    output_path=output_path,
                    all_neuron_positions_df=df_neuron_positions,
                    scheme='B',  # 使用方案B
                    show_background_neurons=SHOW_BACKGROUND_NEURONS,
                    background_neuron_color=BACKGROUND_NEURON_COLOR,
                    background_neuron_size=BACKGROUND_NEURON_SIZE,
                    background_neuron_alpha=BACKGROUND_NEURON_ALPHA,
                    show_title=True,
                    standard_key_neuron_alpha=STANDARD_KEY_NEURON_ALPHA,
                    use_standard_alpha_for_unshared_in_scheme_b=USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B,
                    alpha_non_shared=0.3,
                    shared_marker_size_factor=1.5
                )
                print(f"    ✅ 保存到: {output_filename} (共享神经元数: {len(shared_ids)})")
            except Exception as e:
                print(f"    ❌ 生成 {b1}-{b2} 共享图表失败: {str(e)}")
        
        # --- 3. 为每个行为生成独有神经元图 ---
        print(f"\n🎯 生成每个行为的独有神经元图...")
        
        # 计算每个行为的独有神经元
        all_behavior_sets = {name: set(key_neurons_by_behavior.get(name, [])) for name in all_behaviors}
        
        for behavior_name in all_behaviors:
            print(f"  🔸 生成 {behavior_name} 行为的独有神经元图...")
            
            # 获取该行为的神经元集合
            current_behavior_neurons = all_behavior_sets.get(behavior_name, set())
            
            # 获取其他所有行为的神经元集合
            other_behaviors_neurons = set()
            for other_name in all_behaviors:
                if other_name != behavior_name:
                    other_behaviors_neurons.update(all_behavior_sets.get(other_name, set()))
            
            # 计算独有神经元
            unique_ids = list(current_behavior_neurons - other_behaviors_neurons)
            
            if not unique_ids:
                print(f"    ⚠️  {behavior_name} 没有独有关键神经元，跳过...")
                continue
            
            unique_neurons_df = df_neuron_positions[df_neuron_positions['NeuronID'].isin(unique_ids)]
            
            # 生成安全的文件名
            safe_behavior_name = behavior_name.replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')
            output_filename = f"unique_{safe_behavior_name}_neurons.png"
            output_path = os.path.join(data_paths['output_dir'], output_filename)
            
            # 使用现有的绘图函数
            try:
                plot_unique_neurons_map(
                    unique_neurons_df=unique_neurons_df,
                    behavior_name=behavior_name,
                    behavior_color=BEHAVIOR_COLORS.get(behavior_name, 'gray'),
                    title=f'{behavior_name} Unique Neurons',
                    output_path=output_path,
                    all_neuron_positions_df=df_neuron_positions,
                    show_background_neurons=SHOW_BACKGROUND_NEURONS,
                    background_neuron_color=BACKGROUND_NEURON_COLOR,
                    background_neuron_size=BACKGROUND_NEURON_SIZE,
                    background_neuron_alpha=BACKGROUND_NEURON_ALPHA,
                    key_neuron_size=300,
                    key_neuron_alpha=STANDARD_KEY_NEURON_ALPHA,
                    show_title=True
                )
                print(f"    ✅ 保存到: {output_filename} (独有神经元数: {len(unique_ids)})")
            except Exception as e:
                print(f"    ❌ 生成 {behavior_name} 独有图表失败: {str(e)}")
        
        # ===============================================================================
        # 生成统计汇总
        # ===============================================================================
        
        print(f"\n📊 生成统计汇总...")
        total_individual_plots = len([b for b in all_behaviors if key_neurons_by_behavior.get(b, [])])
        total_shared_plots = len([pair for pair in behavior_pairs if set(key_neurons_by_behavior.get(pair[0], [])).intersection(set(key_neurons_by_behavior.get(pair[1], [])))])
        total_unique_plots = len([b for b in all_behaviors if list(set(key_neurons_by_behavior.get(b, [])) - set().union(*[set(key_neurons_by_behavior.get(other, [])) for other in all_behaviors if other != b]))])
        
        print(f"  📈 个体行为图表: {total_individual_plots} 张")
        print(f"  🔗 共享神经元图表: {total_shared_plots} 张") 
        print(f"  🎯 独有神经元图表: {total_unique_plots} 张")
        print(f"  📦 总计图表数量: {total_individual_plots + total_shared_plots + total_unique_plots} 张")

    print("\n✅ All plots generated successfully!")
    print(f"📁 Output directory: {data_paths['output_dir']}")

else:
    if df_effect_sizes_transformed is None:
        print("❌ Could not load effect sizes. Please check the effect size data file.")
    if df_neuron_positions is None:
        print("❌ Could not load neuron positions. Please check the position data file.")

print("\n" + "=" * 80)
print("🎉 Analysis completed!")
print("=" * 80)

# # === 3D 合并分布图（基于给定数据集中心坐标）===
# try:
#     centers = {
#         'emtrace01_plus': (3.0, -1.31, -4.8),
#         '2980_plus': (3.00, -1.37, -4.80),
#         'bla6250_plus': (3.03, -1.60, -4.85),
#     }
#     out3d = os.path.join(PATH_CONFIG.BASE_OUTPUT_DIR, 'combined_3d_distribution_plus.png')
#     plot_3d_combined_neuron_distribution(centers, PATH_CONFIG, out3d,
#                                          scale_xy_mm=1.6,
#                                          dataset_colors={'emtrace01_plus': '#1f77b4', '2980_plus': '#ff7f0e', 'bla6250_plus': '#7f3fbf'},
#                                          marker_size=70, alpha=0.95,
#                                          bg_color="#eaf2ff", z_scale=0.15, triad_len_ratio=0.18)

#     # 分类版三维图（Open/Middle/Close 激活与 No/all-Activate）
#     cat_out = os.path.join(PATH_CONFIG.BASE_OUTPUT_DIR, 'combined_3d_distribution_plus_categories.png')
#     plot_3d_activation_categories_combined(
#         dataset_keys=['emtrace01_plus', '2980_plus', 'bla6250_plus'],
#         dataset_centers=centers,
#         path_config=PATH_CONFIG,
#         output_path=cat_out,
#         threshold=EFFECT_SIZE_THRESHOLD,
#         colors_map={
#             'Open-Activate': '#1f77b4',
#             'Middle-Activate': '#ff7f0e',
#             'Close-Activate': '#2ca02c',
#             'No-Activate': '#c0c0c0',
#             'all-Activate': '#7f3fbf',
#         },
#         scale_xy_mm=1.6, marker_size=70, alpha=0.95,
#         bg_color="#eaf2ff", z_scale=0.15, triad_len_ratio=0.18
#     )
# except Exception as e:
#     print(f"⚠️ 3D合并分布图生成失败: {e}")
