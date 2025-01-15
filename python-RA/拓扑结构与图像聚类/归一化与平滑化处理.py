# 导入必要的库
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体为“SimHei”或“Microsoft YaHei”
plt.rcParams['font.family'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
# 请确保文件路径正确，并且Excel文件存在
file_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\Day6 calcium_data.xlsx'
data = pd.read_excel(file_path, sheet_name=0)  # 根据实际情况调整 sheet_name

# 查看数据
print("原始数据：")
print(data.head())

# 设置 'Time' 列为索引（如果需要）
if 'Time' in data.columns:
    data.set_index('Time', inplace=True)

# 数据归一化
# 选择归一化方法：Min-Max归一化或标准化

# 方法1：Min-Max归一化
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data),
                               columns=data.columns,
                               index=data.index)
print("\n归一化后的数据（Min-Max）：")
print(normalized_data.head())

# 方法2：标准化（Z-score）
# 如果你更倾向于标准化，可以取消以下代码的注释并注释掉Min-Max部分
# scaler = StandardScaler()
# normalized_data = pd.DataFrame(scaler.fit_transform(data),
#                                columns=data.columns,
#                                index=data.index)
# print("\n归一化后的数据（标准化）：")
# print(normalized_data.head())

# 数据平滑化
# 使用移动平均法，窗口大小根据需要调整
window_size = 2  # 例如，窗口大小为2
smoothed_data = normalized_data.rolling(window=window_size).mean()
print(f"\n平滑化后的数据（窗口大小={window_size}）：")
print(smoothed_data.head())

# 处理平滑化后产生的NaN值
# 你可以选择删除这些行或填充
# 示例：删除含有NaN的行
smoothed_data.dropna(inplace=True)

# 或者用前一个值填充
# smoothed_data.fillna(method='ffill', inplace=True)

# 将平滑化后的数据命名为一个新的DataFrame
# 这里我们假设你只需要平滑化后的数据
final_data = smoothed_data.copy()

# 如果你希望在同一个表格中同时包含归一化和/或平滑化后的数据，可以进行以下操作：
# 例如，将平滑化后的数据添加为新的列，后缀为 '_smoothed'

# 方法1：仅保存平滑化后的数据
# final_data = smoothed_data.copy()

# 方法2：将平滑化后的数据作为新的列添加到归一化数据中
# final_data = normalized_data.copy()
# for column in normalized_data.columns:
#     final_data[f'{column}_smoothed'] = smoothed_data[column]

# 可视化原始数据与平滑数据（以n1为例）
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['n1'], label='原始 n1')
plt.plot(smoothed_data.index, smoothed_data['n1'], label='平滑化 n1', linestyle='--')
plt.xlabel('时间')  # 更改为中文
plt.ylabel('n1')
plt.title('n1 原始数据与平滑化数据对比')
plt.legend()
plt.show()

# 保存处理后的数据到新的Excel文件
output_file = 'processed_data_day6.xlsx'
with pd.ExcelWriter(output_file) as writer:
    # 仅保存平滑化后的数据
    final_data.to_excel(writer, sheet_name='归一化与平滑化数据')  # 工作表名称为中文

    # 如果你选择了方法2，可以保存更多数据
    # final_data.to_excel(writer, sheet_name='归一化与平滑化数据')

print(f"\n处理后的数据已保存到 '{output_file}'")
