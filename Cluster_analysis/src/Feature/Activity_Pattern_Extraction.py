import numpy as np
import pandas as pd

def calculate_metrics_for_neurons(data, neuron_columns, window_size=100, step_size=10):
    """
    计算每个神经元的关键指标
    
    Parameters:
    -----------
    data : pandas.DataFrame
        输入数据，包含时间戳和神经元数据
    neuron_columns : list
        神经元列名列表
    window_size : int
        滑动窗口大小，默认100
    step_size : int
        滑动步长，默认10
    
    Returns:
    --------
    pandas.DataFrame
        包含所有神经元指标的DataFrame
    """
    results = {neuron: [] for neuron in neuron_columns}  # 按神经元分组存储结果
    
    for start in range(0, len(data) - window_size + 1, step_size):
        window_end = start + window_size
        window_data = data.iloc[start:window_end]
        
        start_time = start  # 使用索引作为时间戳
        
        for neuron in neuron_columns:
            neuron_signal = window_data[neuron]
            mean_value = neuron_signal.mean()
            peak_value = neuron_signal.max()
            
            # 1. Amplitude: 窗口内信号的最大值减去平均值
            amplitude = peak_value - mean_value
            
            # 2. Decay Time: 从峰值下降到一半的时间
            peak_idx = neuron_signal.values.argmax()
            half_peak = peak_value / 2
            decay_time = np.argmax(neuron_signal.values[peak_idx:] <= half_peak) if np.any(neuron_signal.values[peak_idx:] <= half_peak) else len(neuron_signal) - peak_idx
            
            # 3. Rise Time: 从平均值上升到峰值的时间
            rise_time = np.argmax(neuron_signal.values >= mean_value)
            
            # 4. Latency: 上升时间与下降时间之和
            latency = decay_time + rise_time
            
            # 5. Frequency: 超过平均值的元素比例
            frequency = len(np.where(neuron_signal > mean_value)[0]) / len(neuron_signal)
            
            # 将结果存储到对应神经元的列表中
            results[neuron].append({
                'Start Time': start_time,
                'Amplitude': amplitude,
                'Peak': peak_value,
                'Decay Time': decay_time,
                'Rise Time': rise_time,
                'Latency': latency,
                'Frequency': frequency,
            })
    
    # 将每个神经元的数据转换为单独的DataFrame，并合并
    all_neurons_data = []
    for neuron, neuron_data in results.items():
        neuron_df = pd.DataFrame(neuron_data)
        neuron_df.insert(0, 'Neuron', neuron)  # 插入神经元名称列
        all_neurons_data.append(neuron_df)
    
    # 合并所有神经元的数据
    return pd.concat(all_neurons_data, ignore_index=True)

def main():
    # 读取数据
    data_path = "../../datasets/processed_Day6.xlsx"
    df = pd.read_excel(data_path)
    
    # 打印列名以检查实际的命名格式
    print("Available columns:", df.columns.tolist())
    
    # 获取所有包含'n'的列（不区分大小写）
    neuron_cols = [col for col in df.columns if 'n' in col.lower()]
    if not neuron_cols:
        raise ValueError("No neuron columns found in the Excel file!")
    print(f"Found {len(neuron_cols)} neuron columns:", neuron_cols)
    
    # 计算滑动窗口内的关键指标
    print("Calculating metrics for all neurons...")
    metrics_df = calculate_metrics_for_neurons(df, neuron_cols)
    print(f"Generated metrics shape: {metrics_df.shape}")
    
    # 保存结果
    output_path = "../../datasets/Day6_Neuron_Calcium_Metrics.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='Windows100_step10', index=False)
    print(f"Results saved to: {output_path}")
    
    # 打印前几行结果预览
    print("\nFirst few rows of results:")
    print(metrics_df.head())

if __name__ == "__main__":
    main() 