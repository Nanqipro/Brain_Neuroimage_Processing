"""
神经元活动归因分析训练启动模块

该模块用于批量启动多个神经元活动分析模型的训练任务。
通过subprocess并行处理多个GPU上的训练任务，支持对多个数据集和随机种子的组合进行训练。

作者: SCN研究小组
日期: 2023
"""

import subprocess

# 定义要处理的数据集文件列表
filename_list = [
        '../SCNData/Dataset1_SCNProject.mat',
        '../SCNData/Dataset2_SCNProject.mat',
        '../SCNData/Dataset3_SCNProject.mat',
        '../SCNData/Dataset4_SCNProject.mat',
        '../SCNData/Dataset5_SCNProject.mat', 
        '../SCNData/Dataset6_SCNProject.mat', 
        ]
        
cuda_id = 0                # 当前CUDA设备ID
job_list = list()          # 正在运行的任务列表
cuda_id_list = list()      # 使用中的CUDA设备ID列表
gpu_num = 4                # 可用GPU数量

# 遍历所有数据集和随机种子组合
for filename in filename_list:
    for seed in range(5):  # 对每个数据集使用5个不同的随机种子
        # 根据不同数据集设置相应的神经元数量
        if 'Dataset1' in filename:
            num_neurons = 6049
        elif 'Dataset2' in filename:
            num_neurons = 7782
        elif 'Dataset3' in filename:
            num_neurons = 7828
        elif 'Dataset4' in filename:
            num_neurons = 6445
        elif 'Dataset5' in filename:
            num_neurons = 8229
        elif 'Dataset6' in filename:
            num_neurons = 8968

        # 构建训练命令，指定GPU、数据集、随机种子和神经元数量
        cmd = f'CUDA_VISIBLE_DEVICES={cuda_id%gpu_num} python training_base.py {filename} {seed} {num_neurons}'
        print(cmd)
        
        # 更新CUDA设备ID，实现循环使用多个GPU
        cuda_id += 1
        
        # 启动子进程运行训练任务
        job = subprocess.Popen(cmd, shell=True)
        job_list.append(job)
        cuda_id_list.append(cuda_id)
        
        # 如果当前运行的任务数量达到GPU数量，等待有任务完成后再继续
        while len(job_list) >= gpu_num:
            for i, job in enumerate(job_list):
                try:
                    # 尝试等待任务完成，超时时间设为gpu_num秒
                    job.wait(gpu_num)
                    
                    # 任务完成，释放对应的资源
                    cuda_id = cuda_id_list[i]
                    cuda_id_list = cuda_id_list[:i] + cuda_id_list[i+1:]
                    job_list = job_list[:i] + job_list[i+1:]
                    break
                except subprocess.TimeoutExpired as e:
                    # 任务尚未完成，继续等待
                    continue
