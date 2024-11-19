import pandas as pd

# 读取对齐好的钙离子浓度数据（假设文件包含时间、行为列，后续为各神经元数据列）
data = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day3\aligned_calcium_data Day3.xlsx')

# 设置不同的窗口大小和步长
window_sizes = [30, 50, 100]
step_sizes = [5, 10]

# 使用 ExcelWriter 打开一个 Excel 文件，以便将数据写入不同的工作表
with pd.ExcelWriter('calcium_window_data_separate_sheets_Day3.xlsx') as writer:
    # 逐步应用不同的窗口大小和步长
    for window_size in window_sizes:
        for step_size in step_sizes:
            results = []  # 存储当前窗口大小和步长的所有结果

            for neuron in data.columns[2:]:  # 跳过时间和行为列
                neuron_data = data[neuron].values

                # 存储当前神经元、窗口和步长的原始数据
                for start in range(0, len(neuron_data) - window_size + 1, step_size):
                    window = neuron_data[start:start + window_size]

                    # 创建一个包含滑动窗口数据的字典
                    result = {
                        'Neuron': neuron,
                        'Start Time': data.iloc[start, 0]  # 使用时间列的起始时间
                    }

                    # 添加每个时间点的值到字典中
                    for i, value in enumerate(window):
                        result[f'Time Point {i + 1}'] = value

                    results.append(result)

            # 将结果转换为 DataFrame
            results_df = pd.DataFrame(results)

            # 生成工作表名称
            sheet_name = f'Window_{window_size}_Step_{step_size}'

            # 将 DataFrame 写入指定的工作表中
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"窗口大小 {window_size}, 步长 {step_size} 的数据已处理并写入工作表 {sheet_name}")

print("所有滑动窗口数据已分别保存到 calcium_window_data_separate_sheets_Day3.xlsx 文件中的不同工作表里")
