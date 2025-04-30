"""
子模块时间预测器批量训练脚本

该脚本用于在三个空间区域上分别批量训练子模块时间预测模型，
支持并行GPU训练，可以同时训练多个不同神经元数量的模型。
"""

import subprocess

# GPU设置
cuda_id = 0
job_list = list()
cuda_id_list = list()
gpu_num = 4

# 对每个随机种子进行循环
for seed in range(5):
    # 对每个空间类别进行循环（共三个空间类别）
    for spatial_class in [1, 2, 3]:
        # 定义不同的神经元数量列表，包括特定数量的神经元
        average_list = [1, 10, 30, 50, 100, 300, 351-1, 500, 600, 606-1, 700-1, 700, 750, 800, 850, 860-1, 900, 1000, 1167-1, 1210-1, 1719-1]
        
        # 对每个神经元数量进行循环
        for num_neuron in average_list:
            # 构建训练命令，指定空间类别、随机种子和神经元数量
            cmd = f'CUDA_VISIBLE_DEVICES={cuda_id%gpu_num} python training_base_1for3.py {spatial_class} {seed} {num_neuron}'
            print(cmd)
            cuda_id += 1
            # 启动训练进程
            job = subprocess.Popen(cmd, shell=True)
            job_list.append(job)
            cuda_id_list.append(cuda_id)
            # 当正在运行的任务数达到GPU数量时等待任务完成
            while len(job_list) >= gpu_num:
                for i, job in enumerate(job_list):
                    try:
                        # 等待任务完成，超时时间为GPU数量
                        job.wait(gpu_num)
                        cuda_id = cuda_id_list[i]
                        # 从列表中移除已完成的任务
                        cuda_id_list = cuda_id_list[:i] + cuda_id_list[i+1:]
                        job_list = job_list[:i] + job_list[i+1:]
                        break
                    except subprocess.TimeoutExpired as e:
                        # 任务尚未完成，继续等待
                        continue
