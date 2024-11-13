import pandas as pd
import numpy as np

# # 读取神经元数据
# data = pd.read_excel('aligned_behavior_calcium_data Day6.xlsx')
#
# # 设置窗口大小和滑动步长
# window_size = 50  # 窗口大小，例如50个时间点
# step_size = 10    # 滑动步长
#
# # 存储滑动窗口结果
# results = {}
#
# # 遍历每个钙离子浓度列（从第三列开始）
# for neuron in data.columns[2:]:  # 跳过时间和行为列
#     neuron_data = data[neuron].values
#     means, std_devs, peaks = [], [], []
#
#     # 滑动窗口遍历神经元数据
#     for start in range(0, len(neuron_data) - window_size + 1, step_size):
#         window = neuron_data[start:start + window_size]
#         means.append(window.mean())
#         std_devs.append(window.std())
#         peaks.append(window.max())
#
#     # 存储结果
#     results[neuron] = {
#         'Mean': means,
#         'StdDev': std_devs,
#         'Peak': peaks
#     }
#
# # 将滑动窗口结果转换为 DataFrame 方便查看
# window_results_df = pd.DataFrame({
#     (neuron, metric): values
#     for neuron, metrics in results.items()
#     for metric, values in metrics.items()
# })
#
# # 显示结果
# print(window_results_df)


data = pd.read_excel('aligned_behavior_calcium_data Day6.xlsx')

# 设置不同的窗口大小和步长
window_sizes = [30, 50, 100]
step_sizes = [5, 10, 20]

# 存储扫描结果
results = {}

for window_size in window_sizes:
    for step_size in step_sizes:
        temp_results = {}

        for neuron in data.columns[2:]:  # 跳过时间和行为列
            neuron_data = data[neuron].values
            means, std_devs, peaks = [], [], []

            # 滑动窗口计算
            for start in range(0, len(neuron_data) - window_size + 1, step_size):
                window = neuron_data[start:start + window_size]
                means.append(window.mean())
                std_devs.append(window.std())
                peaks.append(window.max())

            temp_results[neuron] = {
                'Mean': means,
                'StdDev': std_devs,
                'Peak': peaks
            }

        # 保存不同参数组合的结果
        results[(window_size, step_size)] = temp_results
        print(f"窗口大小 {window_size}, 步长 {step_size} 的结果已计算完成")

        # 创建一个 Excel writer
        with pd.ExcelWriter('sliding_window_results.xlsx') as writer:
            for (window_size, step_size), temp_results in results.items():
                # 将结果转换为 DataFrame
                result_df = pd.DataFrame({
                    (neuron, metric): values
                    for neuron, metrics in temp_results.items()
                    for metric, values in metrics.items()
                })

                # 将 DataFrame 写入 Excel 的单独工作表
                sheet_name = f'Win{window_size}_Step{step_size}'
                result_df.to_excel(writer, sheet_name=sheet_name)

        print("滑动窗口结果已保存到 sliding_window_results.xlsx 文件中")

# 根据具体研究需求对不同参数的效果进行对比分析

