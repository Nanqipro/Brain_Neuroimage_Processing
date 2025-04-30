"""
神经元活动归因分析测试模块

该模块实现了对已训练CNN模型的归因分析，通过整合梯度(Integrated Gradients)方法
计算不同时间点下各个神经元对预测结果的贡献度，并保存分析结果用于后续研究。

作者: SCN研究小组
日期: 2023
"""

import torch
import numpy as np
from src import CNN
from src.dataset import NeuronData
from torch.utils.data import DataLoader
import os
from captum.attr import IntegratedGradients  # 导入整合梯度分析工具

import pickle
import scipy.io as scio 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd

# 定义要分析的数据集目录
dataset_dir = [
    '../SCNData/Dataset1_SCNProject.mat',
    '../SCNData/Dataset2_SCNProject.mat',
    '../SCNData/Dataset3_SCNProject.mat',
    '../SCNData/Dataset4_SCNProject.mat',
    '../SCNData/Dataset5_SCNProject.mat',
    '../SCNData/Dataset6_SCNProject.mat',
    ]


# 遍历所有数据集进行归因分析
for filename in dataset_dir:
    print(f'In processing: {filename}')
    # 加载空间数据信息
    spatial_data = np.array(scio.loadmat(filename, variable_names=['POI'])['POI'])
    spatial_data = np.array([spatial_data[indx][0][0] for indx in range(spatial_data.shape[0])])
    
    # 初始化结果收集列表    
    seed_level_all_random_add_normalNeuron = []
    seed_level_allrandom_add_keyNeuron = []

    # 对每个随机种子训练的模型进行分析
    for seed in range(5):  # 种子只用于加载权重
        np.random.seed(0)
        print(f'In processing seed {seed}') 

        # 根据数据集名称确定对应的神经元数量和数据集名称缩写
        if 'Dataset1' in filename:
            num_neurons = 6049
            sub_name = 'Dataset1'
        elif 'Dataset2' in filename:
            num_neurons = 7782
            sub_name = 'Dataset2'
        elif 'Dataset3' in filename:
            num_neurons = 7828
            sub_name = 'Dataset3'
        elif 'Dataset4' in filename:
            num_neurons = 6445
            sub_name = 'Dataset4'
        elif 'Dataset5' in filename:
            num_neurons = 8229
            sub_name = 'Dataset5'
        elif 'Dataset6' in filename:
            num_neurons = 8968
            sub_name = 'Dataset6'
            
        # 生成权重文件名
        weight_filename = f'CNN_{sub_name}_seed{seed}'
        
        # 加载训练好的模型权重
        with open(os.path.join('training_log', f"{os.path.basename(filename).split('_SCNProject')[0]}_log_num_neuron{num_neurons}_seed{seed}.pkl"), 'rb') as f:
            res = pickle.load(f)
            
        # 初始化CNN模型并加载权重
        model = CNN(num_seq=num_neurons, base_channel=16, num_class=24).to(device)
        model.load_state_dict(res['model'], strict=True)
        
        # 准备验证数据集和数据加载器
        val_dataset = NeuronData(filename, class_num=24)
        test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=None,
                        num_workers=1, pin_memory=True, drop_last=False)

        # 获取数据迭代器
        test_iter = iter(test_loader) 
        
        # 存储所有时间点的关键神经元位置
        all_key_neuron_pos = {}
        
        # 对每个时间类别进行归因分析
        for class_index in range(24):
            # 获取当前时间点的数据和标签
            x, label = next(test_iter)
            x = x.to(device)
            target_class = int(label.item())
            
            # 准备输入张量和基线张量（全零）
            input_tensor = x.to(device, non_blocking=True).view(1, num_neurons)
            baseline_tensor = torch.zeros_like(input_tensor).to(device)
            
            # 定义整合梯度方法
            ig = IntegratedGradients(model)
            
            # 计算归因图谱
            attribution_map, _ = ig.attribute(input_tensor, baseline_tensor, target=target_class, return_convergence_delta=True)
            
            # 归一化归因值到[0,1]范围
            attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-9)
            
            # 将归因图谱转换为numpy数组
            attribution_map = attribution_map.view(1, num_neurons).detach().cpu().numpy()[0, :]
            
            # 保存当前时间点的神经元贡献度
            all_key_neuron_pos['Time'+str(class_index+1)] = attribution_map 
        
        # 创建保存目录
        os.makedirs(f'./neuron_weight/', exist_ok=True)
        os.makedirs(f'./neuron_weight/{sub_name}', exist_ok=True)
        
        # 保存所有时间点的神经元贡献度到MAT文件
        scio.savemat(f'./neuron_weight/{sub_name}/{sub_name}_NeuronWeight_seed{seed}.mat', all_key_neuron_pos)

        # 对每个时间点分析神经元贡献度分布
        for class_id in range(24):
            # 获取当前时间点的神经元权重
            weight_for_class = all_key_neuron_pos['Time'+str(class_id+1)]

            # 创建权重分布统计字典，以0.01为间隔
            dict_weight_per_neuron = {}
            for item in np.arange(0, 1.01, 0.01):
                dict_weight_per_neuron[round(item, 2)] = 0
                
            # 统计各权重值的神经元数量
            for weight_perNeuron in weight_for_class:  
                weight_perNeuron = round(weight_perNeuron, 2)
                dict_weight_per_neuron[weight_perNeuron] = dict_weight_per_neuron[weight_perNeuron]+1
                
            # 输出权重分布和总神经元数
            print(dict_weight_per_neuron)
            print(np.sum(list(dict_weight_per_neuron.values())))
            
            # 创建权重分布保存目录
            os.makedirs(f'./neuron_weight_CNN_perclass/{sub_name}', exist_ok=True)  
            
            # 保存权重分布到CSV文件
            with open(f'./neuron_weight_CNN_perclass/{sub_name}/SCN{sub_name}_time{class_id}_Weight_Dist.csv', 'w') as f:
                [f.write('{0},{1}\n'.format(key, value)) for key, value in dict_weight_per_neuron.items()]
                
            # 将权重分布转换为DataFrame并保存为Excel
            data = {'weight': dict_weight_per_neuron.keys(), 'neuron_num': dict_weight_per_neuron.values()}
            df = pd.DataFrame(data)
            df.to_excel(f'./neuron_weight_CNN_perclass/{sub_name}/SCN{sub_name}_time{class_id}_Weight_Dist.xlsx', index=False)
                