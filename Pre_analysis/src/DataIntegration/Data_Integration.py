import pandas as pd
import os
import numpy as np

# 路径配置
class PathConfig:
    """路径配置类"""
    def __init__(self, base_path="../../datasets"):
        self.base_path = base_path
        self.day3_path = os.path.join(base_path, "Day3_with_behavior_labels_filled.xlsx")
        self.day6_path = os.path.join(base_path, "Day6_with_behavior_labels_filled.xlsx")
        self.day9_path = os.path.join(base_path, "Day9_with_behavior_labels_filled.xlsx")
        self.neuron_map_path = os.path.join(base_path, "神经元对应表.xlsx")
        self.output_path = os.path.join(base_path, "integrated_data.xlsx")

def get_neuron_columns(df):
    """获取数据框中的神经元列"""
    neuron_columns = [col for col in df.columns if 'n' in str(col).lower() and str(col).lower().replace('n', '').isdigit()]
    if not neuron_columns:
        print("警告: 未找到神经元列")
        return []
    print(f"找到神经元列数量: {len(neuron_columns)}")
    return sorted(list(set(neuron_columns)))  # 确保列名唯一且有序

def clean_neuron_name(name):
    """清理神经元名称，确保格式为'n数字'"""
    if pd.isna(name):
        return name
    # 如果已经是'n数字'格式，直接返回
    if isinstance(name, str) and name.startswith('n') and name[1:].isdigit():
        return name
    # 如果是数字（整数或字符串），加上'n'前缀
    try:
        return f'n{int(str(name).replace("n", ""))}'
    except ValueError:
        return name

def create_day_mapping(neuron_map_df, target_day):
    """创建指定天数的神经元映射"""
    column_map = {}
    for _, row in neuron_map_df.iterrows():
        if pd.notna(row['Day3']) and pd.notna(row[f'Day{target_day}']):
            column_map[row[f'Day{target_day}']] = row['Day3']
    return column_map

def create_renamed_dataframe(df, neurons, column_map):
    """创建重命名后的数据框"""
    data = {'behavior': df['behavior'].values}
    for col in neurons:
        if col in column_map:
            data[column_map[col]] = df[col].values
    return pd.DataFrame(data)

def print_statistics(result_df, day3_df, day6_df, day9_df, day6_column_map, day9_column_map):
    """打印数据统计信息"""
    print("\n数据统计信息:")
    print(f"总行数: {len(result_df)}")
    print(f"总列数: {len(result_df.columns)}")
    print("缺失值统计:")
    print(result_df.isnull().sum().sum())

    print("\n各天数据行数:")
    original_lengths = [len(day3_df), len(day6_df), len(day9_df)]
    print(f"Day3: {original_lengths[0]}行")
    print(f"Day6: {original_lengths[1]}行")
    print(f"Day9: {original_lengths[2]}行")
    print(f"总计: {sum(original_lengths)}行")

    print("\n列映射统计:")
    print(f"Day6映射数量: {len(day6_column_map)}")
    print(f"Day9映射数量: {len(day9_column_map)}")
    print(f"最终列数: {len(result_df.columns)}")

def main():
    # 初始化路径配置
    paths = PathConfig()

    # 读取数据文件
    day3_df = pd.read_excel(paths.day3_path)
    day6_df = pd.read_excel(paths.day6_path)
    day9_df = pd.read_excel(paths.day9_path)
    neuron_map_df = pd.read_excel(paths.neuron_map_path)

    # 清理神经元对应表中的名称
    for day in [3, 6, 9]:
        neuron_map_df[f'Day{day}'] = neuron_map_df[f'Day{day}'].apply(clean_neuron_name)

    # 获取每天的神经元列
    day3_neurons = get_neuron_columns(day3_df)
    day6_neurons = get_neuron_columns(day6_df)
    day9_neurons = get_neuron_columns(day9_df)

    print(f"Day3神经元数量: {len(day3_neurons)}")
    print(f"Day6神经元数量: {len(day6_neurons)}")
    print(f"Day9神经元数量: {len(day9_neurons)}")

    # 创建Day6和Day9的列映射
    day6_column_map = create_day_mapping(neuron_map_df, 6)
    day9_column_map = create_day_mapping(neuron_map_df, 9)

    # 创建重命名后的DataFrame
    day6_renamed = create_renamed_dataframe(day6_df, day6_neurons, day6_column_map)
    day9_renamed = create_renamed_dataframe(day9_df, day9_neurons, day9_column_map)

    # 垂直拼接数据
    result_df = pd.concat([day3_df, day6_renamed, day9_renamed], axis=0, ignore_index=True)

    # 添加连续的stamp列
    result_df['stamp'] = range(1, len(result_df) + 1)

    # 保存结果
    result_df.to_excel(paths.output_path, index=False)
    print(f"\n数据整合完成，已保存至: {paths.output_path}")

    # 打印统计信息
    print_statistics(result_df, day3_df, day6_df, day9_df, day6_column_map, day9_column_map)

if __name__ == "__main__":
    main()
