# 多种行为条件下的神经元波动集成图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# 设置图形后端
matplotlib.use('TkAgg')  # 可以根据系统修改为 'Agg', 'Qt5Agg' 等

# 设置放大因子以增加波动幅度
amplitude_scale = 5  # 增大波动幅度的缩放系数，可以调整此值来增加或减少波动幅度

# 定义数据文件列表和对应的标题
data_files = [
    'processed_Day3.xlsx',
    'processed_Day6.xlsx',
    'processed_Day9.xlsx',
]

# 定义图表标题（和文件名对应）
titles = [
    'Day3_with_behavior_labels_filled',
    'Day6_with_behavior_labels_filled',
    'Day9_with_behavior_labels_filled',
    
]

# 加载神经元对应表
correspondence_table = pd.read_excel('../../datasets/神经元对应表2979.xlsx')

# 定义文件名到神经元对应表2979列名的映射关系
# 这个字典将每个文件名映射到对应表2979中的对应列名
file_to_column_mapping = {
    # 格式: '文件名.xlsx': '对应表中的列名'
    # 这表示该文件在对应表2979中的哪一列
    
    # 请填入实际的宝工关系
    'processed_Day3.xlsx': 'Day3_with_behavior_labels_filled',  
    'processed_Day6.xlsx': 'Day6_with_behavior_labels_filled',
    'processed_Day9.xlsx': 'Day9_with_behavior_labels_filled',
}

# 加载参考排序表（基准神经元排序）
try:
    # 尝试载入基准文件
    reference_file = '../../datasets/processed_Day3.xlsx'
    reference_data = pd.read_excel(reference_file)
    print(f'成功加载基准排序文件: {reference_file}')
except Exception as e:
    print(f'加载基准排序文件失败: {e}')
    # 如果失败，则退出程序
    raise Exception('无法加载基准排序文件，请确保文件路径正确')

# 提取基准文件中的神经元列名（排除'stamp'列）
reference_neurons = [col for col in reference_data.columns if col != 'stamp']
print(f'基准排序文件中的神经元数量: {len(reference_neurons)}')

# 实际神经元数据和对应的数据框字典
data_dict = {}
time_stamps_dict = {}

# 尝试加载所有数据文件
for file_name in data_files:
    try:
        file_path = f'../../datasets/{file_name}'
        data = pd.read_excel(file_path)
        print(f'成功加载数据文件: {file_path}')
        
        # 存储数据和时间戳
        data_dict[file_name] = data
        time_stamps_dict[file_name] = data['stamp'] if 'stamp' in data.columns else pd.Series(range(len(data)))
    except Exception as e:
        print(f'加载数据文件失败 {file_name}: {e}')
        # 如果某个文件加载失败，使用空数据框代替
        data_dict[file_name] = pd.DataFrame()
        time_stamps_dict[file_name] = pd.Series([])

# 检查是否至少有一个文件成功加载
if all(df.empty for df in data_dict.values()):
    raise Exception('所有数据文件均加载失败，请检查文件路径和格式')

# 计算子图布局（行数和列数）
n_files = len(data_files)
ncols = min(5, n_files)  # 最多5列
nrows = (n_files + ncols - 1) // ncols  # 向上取整计算行数

# 创建大尺寸的图形
plt.figure(figsize=(ncols * 10, nrows * 8))

# 处理每个数据文件，创建子图
for file_idx, (file_name, title) in enumerate(zip(data_files, titles)):
    data = data_dict[file_name]
    if data.empty:
        continue  # 跳过空数据
        
    time_stamps = time_stamps_dict[file_name]
    
    # 创建子图
    plt.subplot(nrows, ncols, file_idx + 1)
    
    # 处理该文件中的神经元，按照参考神经元顺序
    neurons_processed = 0
    offset = 10  # 设置垂直偏移量
    
    # 获取当前文件在对应表2979中的列名
    corr_column_name = file_to_column_mapping.get(file_name, None)
    
    # 如果没有映射关系，跳过该文件
    if corr_column_name is None:
        print(f'警告: 文件 {file_name} 没有在映射表中定义对应的神经元列名')
        continue
    
    # 尝试按照基准排序绘制神经元trace
    for i, ref_neuron in enumerate(reference_neurons):
        # 检查当前神经元是否存在于数据中
        if ref_neuron in data.columns:
            # Z-score 标准化并放大
            trace = (data[ref_neuron] - data[ref_neuron].mean()) / data[ref_neuron].std()
            trace = trace * amplitude_scale
            
            # 绘制带偏移的trace
            plt.plot(time_stamps, trace + i * offset, linewidth=0.8)
            neurons_processed += 1
            
    # Display the corresponding neuron column name
    plt.text(0.02, 0.92, f'Corresponding column in 2979: {corr_column_name}', 
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 设置坐标轴和标题
    plt.title(title)
    plt.xlabel('Time Stamp')
    
    # 第一列的图添加y轴标签
    if file_idx % ncols == 0:
        plt.ylabel('Neuron Trace')
    
    # 不显示y轴刻度（神经元太多，显示会很乱）
    plt.yticks([])
    
    # 显示处理的神经元数量
    plt.text(0.02, 0.98, f'Neurons: {neurons_processed}', 
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 调整子图间距
plt.tight_layout()

# 保存图像到文件
output_dir = '../../graph/traces-Integration/'
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
# 保存图像，设置高分辨率
output_file = os.path.join(output_dir, 'processed_multi_behavior_trace_integration.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'图像已保存至: {output_file}')

# 显示图像
# plt.show()