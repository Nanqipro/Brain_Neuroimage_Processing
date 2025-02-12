import pandas as pd
import os
import numpy as np

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

# 设置数据文件路径
base_path = "../../datasets"
day3_path = os.path.join(base_path, "Day3_with_behavior_labels_filled.xlsx")
day6_path = os.path.join(base_path, "Day6_with_behavior_labels_filled.xlsx")
day9_path = os.path.join(base_path, "Day9_with_behavior_labels_filled.xlsx")
neuron_map_path = os.path.join(base_path, "神经元对应表.xlsx")

# 读取数据文件
day3_df = pd.read_excel(day3_path)
day6_df = pd.read_excel(day6_path)
day9_df = pd.read_excel(day9_path)
neuron_map_df = pd.read_excel(neuron_map_path)

# 清理神经元对应表中的名称
neuron_map_df['Day3'] = neuron_map_df['Day3'].apply(clean_neuron_name)
neuron_map_df['Day6'] = neuron_map_df['Day6'].apply(clean_neuron_name)
neuron_map_df['Day9'] = neuron_map_df['Day9'].apply(clean_neuron_name)

# 获取每天的神经元列
day3_neurons = get_neuron_columns(day3_df)
day6_neurons = get_neuron_columns(day6_df)
day9_neurons = get_neuron_columns(day9_df)

print(f"Day3神经元数量: {len(day3_neurons)}")
print(f"Day6神经元数量: {len(day6_neurons)}")
print(f"Day9神经元数量: {len(day9_neurons)}")

# 创建Day6的列映射
day6_column_map = {}
for _, row in neuron_map_df.iterrows():
    if pd.notna(row['Day3']) and pd.notna(row['Day6']):
        day6_column_map[row['Day6']] = row['Day3']

# 创建Day9的列映射
day9_column_map = {}
for _, row in neuron_map_df.iterrows():
    if pd.notna(row['Day3']) and pd.notna(row['Day9']):
        day9_column_map[row['Day9']] = row['Day3']

# 重命名Day6的列并创建新的DataFrame
day6_data = {}
day6_data['behavior'] = day6_df['behavior'].values
for col in day6_neurons:
    if col in day6_column_map:
        day6_data[day6_column_map[col]] = day6_df[col].values
day6_renamed = pd.DataFrame(day6_data)

# 重命名Day9的列并创建新的DataFrame
day9_data = {}
day9_data['behavior'] = day9_df['behavior'].values
for col in day9_neurons:
    if col in day9_column_map:
        day9_data[day9_column_map[col]] = day9_df[col].values
day9_renamed = pd.DataFrame(day9_data)

# 垂直拼接数据
result_df = pd.concat([day3_df, day6_renamed, day9_renamed], axis=0, ignore_index=True)

# 保存结果
output_path = os.path.join(base_path, "integrated_data.xlsx")
result_df.to_excel(output_path, index=False)
print(f"\n数据整合完成，已保存至: {output_path}")

# 输出数据统计信息
print("\n数据统计信息:")
print(f"总行数: {len(result_df)}")
print(f"总列数: {len(result_df.columns)}")
print("缺失值统计:")
print(result_df.isnull().sum().sum())

# 验证每天的数据行数
print("\n各天数据行数:")
original_lengths = [len(day3_df), len(day6_df), len(day9_df)]
print(f"Day3: {original_lengths[0]}行")
print(f"Day6: {original_lengths[1]}行")
print(f"Day9: {original_lengths[2]}行")
print(f"总计: {sum(original_lengths)}行")

# 输出列映射统计
print("\n列映射统计:")
print(f"Day6映射数量: {len(day6_column_map)}")
print(f"Day9映射数量: {len(day9_column_map)}")
print(f"最终列数: {len(result_df.columns)}")
