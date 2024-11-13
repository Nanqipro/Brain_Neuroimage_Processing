import numpy as np
import pandas as pd

# 假设钙离子浓度数据在 calcium_data.csv 文件中，500 行（时间点）× 60 列（神经元）
df = pd.read_excel('calcium_data9.xlsx')
calcium_data = df.to_numpy()

# 计算每个神经元的均值
mean_concentration_per_neuron = np.mean(calcium_data, axis=0)

# 输出结果
for i, mean_concentration in enumerate(mean_concentration_per_neuron):
    print(f"神经元 {i+1} 的钙离子浓度均值: {mean_concentration}")

# 查找均值大于零的神经元
positive_neurons = []
for i, mean_concentration in enumerate(mean_concentration_per_neuron):
    if mean_concentration > 0:
        positive_neurons.append((i, mean_concentration))  # 存储神经元索引和均值

# 输出结果
print(f"共有 {len(positive_neurons)} 个神经元的均值大于零。")
for neuron_index, mean_value in positive_neurons:
    print(f"神经元 {neuron_index + 1} 的钙离子浓度均值为: {mean_value}")

