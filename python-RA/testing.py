# import os
# import caiman as cm
# from caiman.source_extraction.cnmf import cnmf as cnmf
# from caiman.utils.visualization import inspect_correlation_pnr
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Step 1: 文件路径设置
# video_path = r'C:\Users\PAN\Desktop\RA\homecage_testing.avi'
# output_folder = r'C:\Users\PAN\Desktop\RA\CaImAn_Output'
# os.makedirs(output_folder, exist_ok=True)
#
# # Step 2: 加载视频并创建 memory-mapped 文件
# print("正在加载视频...")
# movie = cm.load(video_path)
# print("正在保存 memory-mapped 文件...")
# movie.save(os.path.join(output_folder, 'memory_mapped_movie.mmap'))
#
# # 自动检测实际生成的 memory-mapped 文件
# print("检测实际生成的 memory-mapped 文件...")
# mmap_file = None
# for file in os.listdir(output_folder):
#     if file.startswith('memory_mapped_movie') and file.endswith('.mmap'):
#         mmap_file = os.path.join(output_folder, file)
#         break
#
# if mmap_file is None:
#     raise FileNotFoundError("未能找到生成的 memory-mapped 文件。请检查输出目录。")
# print(f"实际生成的 memory-mapped 文件: {mmap_file}")
#
# # 测试加载 memory-mapped 文件
# try:
#     loaded_movie = cm.load(mmap_file)
#     print("内存映射文件加载成功！")
# except Exception as e:
#     print(f"加载 memory-mapped 文件时出错: {e}")
#     raise
#
# # Step 3: 设置 CNMF 参数
# print("设置 CNMF 参数...")
# cnmf_params = cnmf.CNMFParams(params_dict={
#     'fnames': mmap_file,
#     'fr': 30,  # 帧率
#     'decay_time': 0.4,  # 荧光信号衰减时间
#     'pw_rigid': True,  # 是否启用片段配准
#     'max_shifts': (5, 5),  # 最大位移
#     'border_nan': 'copy',  # 边界处理
#     'p': 1,  # AR 模型阶数
#     'nb': 2,  # 后继背景模型的分量数
#     'rf': 50,  # 分区的空间范围
#     'stride': 20,  # 步幅大小
#     'K': None,  # 组件数目
#     'gSig': [4, 4],  # 高斯核的标准差
#     'gSiz': [8, 8],  # 空间滤波器的大小
#     'min_SNR': 2.5,  # 信噪比阈值
#     'rval_thr': 0.85,  # 空间相关性阈值
#     'use_cnn': True,  # 是否使用 CNN
# })
#
# # Step 4: 计算 Correlation 和 PNR 图像
# print("计算 Correlation 和 PNR 图像...")
# cn_filter, pnr = cm.summary_images.correlation_pnr(loaded_movie, gSig=4)  # 将 gSig 替换为单个整数
#
# # 保存 Correlation 和 PNR 图像
# correlation_pnr_path = os.path.join(output_folder, 'correlation_pnr.png')
# inspect_correlation_pnr(cn_filter, pnr)
# plt.savefig(correlation_pnr_path)
# plt.close()
# print(f"Correlation 和 PNR 图像已保存到: {correlation_pnr_path}")
#
# # Step 5: 运行 CNMF
# print("运行 CNMF 分析...")
# cnm = cnmf.CNMF(n_processes=1, dview=None, params=cnmf_params)
# cnm.fit_file()
#
# # Step 6: 提取神经元位置
# print("提取神经元位置...")
# coordinates = cnm.estimates.coordinates
# print(f"检测到的神经元总数: {len(coordinates)}")
#
# # 绘制神经元位置
# neuron_positions_path = os.path.join(output_folder, 'neuron_positions.png')
# plt.figure(figsize=(10, 10))
# plt.imshow(cn_filter, cmap='gray', origin='upper')
# for coord in coordinates:
#     plt.scatter(coord[0], coord[1], color='red', s=15, label='Neuron Position')
# plt.title('Neuron Positions')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig(neuron_positions_path)
# plt.show()
# print(f"神经元位置图已保存到: {neuron_positions_path}")
#
# # Step 7: 保存结果
# results_file = os.path.join(output_folder, 'cnmf_results.hdf5')
# coordinates_file = os.path.join(output_folder, 'neuron_positions.csv')
# cnm.save(results_file)
# np.savetxt(coordinates_file, coordinates, delimiter=',')
# print(f"CNMF 分析结果已保存到: {results_file}")
# print(f"神经元位置已保存到: {coordinates_file}")
#


import os
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.visualization import inspect_correlation_pnr
import matplotlib.pyplot as plt
import numpy as np

# Step 1: 文件路径设置
video_path = r'C:\Users\PAN\Desktop\RA\homecage_testing.avi'
output_folder = r'C:\Users\PAN\Desktop\RA\CaImAn_Output'
os.makedirs(output_folder, exist_ok=True)

# Step 2: 加载视频并创建 memory-mapped 文件
print("正在加载视频...")
movie = cm.load(video_path)

# 确保 movie 数据为 3D 格式
if len(movie.shape) < 3:
    raise ValueError("加载的视频数据不是 3D 格式，请检查输入文件！")
print(f"Movie shape: {movie.shape}")

print("正在保存 memory-mapped 文件为 C-order...")
mmap_file = os.path.join(output_folder, 'memory_mapped_movie.mmap')
movie.save(mmap_file, order='C')

# 自动检测 memory-mapped 文件
print("检测实际生成的 memory-mapped 文件...")
detected_mmap_file = None
for file in os.listdir(output_folder):
    if file.startswith('memory_mapped_movie') and file.endswith('.mmap'):
        detected_mmap_file = os.path.join(output_folder, file)
        break

if detected_mmap_file is None:
    raise FileNotFoundError("未能找到生成的 memory-mapped 文件。请检查输出目录。")
print(f"实际生成的 memory-mapped 文件: {detected_mmap_file}")

# 测试加载 memory-mapped 文件
try:
    loaded_movie = cm.load(detected_mmap_file)
    print(f"加载 memory-mapped 文件成功！Movie shape: {loaded_movie.shape}")
except Exception as e:
    print(f"加载 memory-mapped 文件时出错: {e}")
    raise


# Step 3: 设置 CNMF 参数
print("设置 CNMF 参数...")
cnmf_params = cnmf.CNMFParams(params_dict={
    'fnames': detected_mmap_file,
    'fr': 30,
    'decay_time': 0.4,
    'pw_rigid': True,
    'max_shifts': (5, 5),
    'border_nan': 'copy',
    'p': 1,
    'nb': 2,
    'rf': 50,
    'stride': 20,
    'K': None,
    'gSig': [4, 4],
    'gSiz': [8, 8],
    'min_SNR': 2.5,
    'rval_thr': 0.85,
    'use_cnn': True,
})

# Step 4: 计算 Correlation 和 PNR 图像
print("计算 Correlation 和 PNR 图像...")
cn_filter, pnr = cm.summary_images.correlation_pnr(loaded_movie, gSig=4)

# 保存 Correlation 和 PNR 图像
correlation_pnr_path = os.path.join(output_folder, 'correlation_pnr.png')
inspect_correlation_pnr(cn_filter, pnr)
plt.savefig(correlation_pnr_path)
plt.close()
print(f"Correlation 和 PNR 图像已保存到: {correlation_pnr_path}")

# Step 5: 运行 CNMF
print("运行 CNMF 分析...")
try:
    cnm = cnmf.CNMF(n_processes=1, dview=None, params=cnmf_params)
    cnm.fit_file()
    print("CNMF 分析完成！")
except Exception as e:
    print(f"CNMF 分析时出错: {e}")
    raise

# Step 6: 提取神经元位置
print("提取神经元位置...")
coordinates = cnm.estimates.coordinates
print(f"检测到的神经元总数: {len(coordinates)}")

# 绘制神经元位置
neuron_positions_path = os.path.join(output_folder, 'neuron_positions.png')
plt.figure(figsize=(10, 10))
plt.imshow(cn_filter, cmap='gray', origin='upper')
for coord in coordinates:
    plt.scatter(coord[0], coord[1], color='red', s=15, label='Neuron Position')
plt.title('Neuron Positions')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(neuron_positions_path)
plt.show()
print(f"神经元位置图已保存到: {neuron_positions_path}")

# Step 7: 保存结果
results_file = os.path.join(output_folder, 'cnmf_results.hdf5')
coordinates_file = os.path.join(output_folder, 'neuron_positions.csv')
cnm.save(results_file)
np.savetxt(coordinates_file, coordinates, delimiter=',')
print(f"CNMF 分析结果已保存到: {results_file}")
print(f"神经元位置已保存到: {coordinates_file}")

