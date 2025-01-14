import pandas as pd
import numpy as np

# 加载包含行为标记的文件（CHB工作表），只选择 'stamp' 和 'FrameLost' 列
behavior_labels = pd.read_excel('./data/day9cell行为学标注.xlsx', sheet_name='CHB', usecols=['stamp', 'FrameLost'])

# 加载 Day3 数据
day3_data = pd.read_excel('./data/Day9.xlsx')

# 使用 forward fill 填充 FrameLost 列中的空值，使每个 stamp 行都有对应的行为
# 避免 inplace 警告，使用直接赋值
behavior_labels['FrameLost'] = behavior_labels['FrameLost'].ffill()

# 基于 'stamp' 列合并，将填充后的 'FrameLost' 数据添加到 Day3 数据中
merged_data = day3_data.merge(behavior_labels, on='stamp', how='left')

# 如果 'FrameLost' 列为空，将对应行标记为 'null'
merged_data['FrameLost'] = merged_data['FrameLost'].replace('', np.nan)  # 如果空字符串要处理为NaN
merged_data['FrameLost'] = merged_data['FrameLost'].where(merged_data['FrameLost'].notna(), 'null')

# 保存合并后的数据到新的 Excel 文件
output_path = './data/Day9_with_behavior_labels_filled.xlsx'
merged_data.to_excel(output_path, index=False)

print(f"文件已保存为: {output_path}")
