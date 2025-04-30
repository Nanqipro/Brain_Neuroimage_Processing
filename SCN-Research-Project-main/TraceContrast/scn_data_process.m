%%%%%% 3d scn data process pipeline
% 超交叉核(SCN)三维数据处理流程
% 本脚本用于处理SCN钙成像数据，将其转换为不同采样模式的标准格式
% 输出三种不同的数据格式：标准格式、时间采样格式和神经元采样格式

% 清除工作区所有变量，关闭所有窗口，清空命令窗口
clear all;
close all;
clc;

% 定义颜色映射表（用于可能的可视化）
color = [1,86,153;
250,192,15;
243,118,74;
95,198,201;
79,89,100]/255;

%% load data
% 加载数据部分
scn_data_path = '..\SCNData\Dataset1_SCNProject.mat'; % 原始SCN数据文件路径
dataset_order = '01'; % 数据集序号，用于输出文件命名

% 加载.mat数据文件，其中包含dff_set（钙信号）和POI（神经元位置坐标）
load(scn_data_path);

% 计算神经元总数
all_num = size(dff_set, 1);

% 设置时间窗口参数
num_time = 200;     % 每个小时的时间点数量
half_num_time = num_time / 2;  % 时间采样模式的时间点数量

% 处理POI(Points Of Interest)数据，提取神经元三维坐标
poi = zeros(all_num, 3);
for i = 1:all_num
    tmp = cell2mat(POI(i,1));
    poi(i,:) = tmp(1:3);
end
POI = poi;  % 更新POI为标准化的三维坐标矩阵

%% standard
% 标准处理模式：保留所有神经元和所有时间点
trace = zeros(num_time*24, all_num);  % 初始化输出矩阵，行：时间点，列：神经元

% 遍历24小时
for t = 1:24
    count = 0;  % 计数器（未使用）
    % 遍历所有神经元
    for i = 1:all_num
        dff = cell2mat(dff_set(i,t));  % 提取第i个神经元在第t小时的dff信号
        if ~isempty(dff)  % 如果信号不为空
            % 将该小时的信号放入对应位置，连接成连续时间序列
            trace((1:num_time)+num_time*(t-1), i) = dff;
        end
    end
end

% 保存标准格式数据，包含神经元位置POI和钙信号trace
save([dataset_order, '_standard.mat'], 'POI', 'trace');

%% time-sample
% 时间采样模式：对时间维度进行降采样（减少一半时间点）
trace = zeros(half_num_time*24, all_num);  % 初始化降采样后的矩阵

% 遍历24小时
for t = 1:24 
    count = 0;  % 计数器（未使用）
    % 遍历所有神经元
    for i = 1:all_num
        dff = cell2mat(dff_set(i,t));  % 提取钙信号
        if ~isempty(dff)
            % 将信号以2倍下采样放入矩阵(1:2:num_time表示取奇数索引位置的数据)
            trace((1:half_num_time)+half_num_time*(t-1), i) = dff(1:2:num_time);
        end
    end
end

% 保存时间采样格式数据
save([dataset_order, '_time-sample.mat'], 'POI', 'trace');

%% pc-sample
% 神经元采样模式：减少神经元数量但保留所有时间点
trace = zeros(num_time*24, all_num);  % 临时初始化矩阵（未使用）

% 采样神经元数量为总数的50%
SAMPLING_SET = ceil(0.5*size(POI,1));
% 将POI转换为结构体格式，准备进行最远点采样
srf = struct('X',POI(:,1),'Y',POI(:,2),'Z',POI(:,3));
% 使用最远点采样(Farthest Point Sampling)算法选择神经元
% 这种采样保证了空间上的均匀分布
ifps = fps_euc(srf,SAMPLING_SET);
% 提取采样后的神经元位置
tmp_pos = POI(ifps,:);
POI = tmp_pos;  % 更新POI为采样后的神经元位置

% 初始化采样后的钙信号矩阵
ds_trace = zeros(num_time*24, SAMPLING_SET);

% 处理采样后的神经元数据
for t = 1:24
    count = 0;  % 计数器（未使用）
    for i = 1:size(ifps, 2)
        ds_dff = cell2mat(dff_set(ifps(i),t));  % 提取采样的神经元钙信号
        if ~isempty(ds_dff)
            % 将信号放入对应位置
            ds_trace((1:num_time)+num_time*(t-1), i) = ds_dff;
        end
    end
end

% 更新trace为采样后的矩阵
trace = ds_trace;
% 保存神经元采样格式数据
save([dataset_order, '_pc-sample.mat'], 'POI', 'trace');

