import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 加载神经元对应表
neuron_map = pd.read_excel("./data/神经元对应表.xlsx", sheet_name="Sheet1")

# 加载 Day3、Day6 和 Day9 的钙离子浓度数据
day3_data = pd.read_excel("./data/Day3.xlsx")
day6_data = pd.read_excel("./data/Day6.xlsx")
day9_data = pd.read_excel("./data/Day9.xlsx")

# 根据 Day3 的顺序对齐神经元，将 'null' 的神经元放在末尾
aligned_data = {}
for idx, row in neuron_map.iterrows():
    day3_neuron, day6_neuron, day9_neuron = row['Day3'], row['Day6'], row['Day9']

    # 提取并对齐神经元数据，如果缺失则填充 NaN
    day3_values = day3_data[day3_neuron] if day3_neuron in day3_data else pd.Series([np.nan] * len(day3_data))
    day6_values = day6_data[day6_neuron] if day6_neuron in day6_data else pd.Series([np.nan] * len(day6_data))
    day9_values = day9_data[day9_neuron] if day9_neuron in day9_data else pd.Series([np.nan] * len(day9_data))

    # 将每个神经元的数据按列存入字典，键为 Day3 的神经元编号
    aligned_data[row['Day3']] = pd.DataFrame({
        'Day3': day3_values.values,
        'Day6': day6_values.values,
        'Day9': day9_values.values
    })

# 按照 Day3 的神经元编号顺序，将数据组合为一个 DataFrame
final_aligned_df = pd.concat(aligned_data, axis=1)

# 生成热图
plt.figure(figsize=(12, 8))
sns.heatmap(final_aligned_df.T, cmap="viridis", cbar=True)
plt.title("Calcium concentration heat map (aligned by neuron ID)")
plt.xlabel("stamp")
plt.ylabel("Neurons (aligned by day3)")
plt.show()

# 绘制每个神经元的 trace 图
plt.figure(figsize=(14, 10))

for idx, neuron_id in enumerate(neuron_map['Day3']):
    plt.subplot(len(neuron_map), 1, idx + 1)

    # 获取每个神经元在 Day3、Day6、Day9 的时间序列数据
    day3_neuron = neuron_map.loc[idx, 'Day3']
    day6_neuron = neuron_map.loc[idx, 'Day6']
    day9_neuron = neuron_map.loc[idx, 'Day9']

    # 绘制每一天的 trace 图，检查神经元数据是否存在
    if day3_neuron in day3_data:
        plt.plot(day3_data['stamp'], day3_data[day3_neuron], label="Day3")
    if day6_neuron in day6_data:
        plt.plot(day6_data['stamp'], day6_data[day6_neuron], label="Day6")
    if day9_neuron in day9_data:
        plt.plot(day9_data['stamp'], day9_data[day9_neuron], label="Day9")

    plt.title(f"神经元 {neuron_id} 在三天中的 trace 图")
    plt.xlabel("stamp")
    plt.ylabel("Ca2+")
    plt.legend()
    plt.tight_layout()

plt.show()
