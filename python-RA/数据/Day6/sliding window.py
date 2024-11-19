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




import pandas as pd
import numpy as np

# 读取数据
data = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\aligned_calcium_data Day3.xlsx')

# 设置不同的窗口大小和步长
window_sizes = [30, 50, 100]
step_sizes = [5, 10]

# 存储扫描结果
results = {}

# 预分配结果
for window_size in window_sizes:
    for step_size in step_sizes:
        temp_results = {}

        # 针对每个神经元进行滑动窗口计算
        for neuron in data.columns[2:]:  # 跳过时间和行为列
            neuron_data = data[neuron].values
            data_length = len(neuron_data)

            # 使用 NumPy 数组快速计算滑动窗口
            indices = np.arange(0, data_length - window_size + 1, step_size)
            windows = np.lib.stride_tricks.sliding_window_view(neuron_data, window_size)[::step_size]

            means = np.mean(windows, axis=1)
            std_devs = np.std(windows, axis=1)
            peaks = np.max(windows, axis=1)

            temp_results[neuron] = {
                'Mean': means,
                'StdDev': std_devs,
                'Peak': peaks
            }

        results[(window_size, step_size)] = temp_results
        print(f"窗口大小 {window_size}, 步长 {step_size} 的结果已计算完成")

# 保存结果到 Excel
with pd.ExcelWriter('sliding_window_results_optimized.xlsx') as writer:
    for (window_size, step_size), temp_results in results.items():
        # 转换为 DataFrame
        result_df = pd.concat(
            {neuron: pd.DataFrame(metrics) for neuron, metrics in temp_results.items()}, axis=1
        )

        # 写入单独工作表
        sheet_name = f'Win{window_size}_Step{step_size}'
        result_df.to_excel(writer, sheet_name=sheet_name)

print("所有滑动窗口结果已保存到 sliding_window_results_optimized.xlsx 文件中")

