import pandas as pd

# 假设 behavior_data.xlsx 和 calcium_data.xlsx 分别包含行为数据和钙离子浓度数据，每个时间点可能有缺失
behavior_data = pd.read_excel('behavior_data.xlsx')
calcium_data = pd.read_excel('calcium_data.xlsx')

# 确保时间点列为索引，便于自动对齐
behavior_data.set_index('Time', inplace=True)
calcium_data.set_index('Time', inplace=True)

# 合并数据并自动对齐时间点，使用 outer join 保留所有时间点，缺失值填充 NaN
aligned_data = pd.merge(behavior_data, calcium_data, left_index=True, right_index=True, how='outer')

# 检查合并后的数据，确认缺失值填充正确
print(aligned_data.head())

# 可以在对齐后填充特定值，例如 0 或前后插值，以处理 NaN
aligned_data.fillna(method='ffill', inplace=True)  # 前向填充，适合连续性数据
aligned_data.fillna(method='bfill', inplace=True)  # 后向填充，如果前向填充未填满
# 如需保留 NaN 作为缺失标记，可以省略上面两行

# 保存对齐后的数据
aligned_data.to_excel('aligned_behavior_calcium_data.xlsx')
