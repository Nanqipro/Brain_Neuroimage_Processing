%% 3D SCN数据处理流程
% 该脚本用于处理脑神经元钙成像数据，将时间序列转换为相空间流形，并构建图数据集
% 主要步骤包括：
% 1. 加载原始钙信号数据
% 2. 将钙成像时间序列转换为相空间流形(phase-space manifolds)
% 3. 生成图数据集的节点、边和图属性文件
%
% 作者: SCN研究小组
% 日期: 2023

clearvars; clc; close all; warning off; dbstop if error;
addpath(genpath('./src'))  % 添加src目录到MATLAB路径

%% 加载数据
filePath = './SCNData/Dataset1_SCNProject.mat';  % 输入文件路径，请根据实际情况修改
frameRate = 0.67;  % 帧率，单位Hz
outPath = './data';  % 输出目录路径
mkdir(outPath)  % 创建输出目录

load(filePath);  % 加载MAT格式数据

%% 将钙离子时间序列转换为相空间流形
% 相空间重构是分析动态系统的一种方法，可以揭示时间序列中的非线性动力学特性
[cellNum, timeline] = size(F_set);  % 获取数据尺寸：细胞数量和时间点
trace_zs_set = cell(cellNum, timeline);  % 存储标准化的时间序列
xyz = cell(cellNum, timeline);  % 存储相空间坐标

for tt = 1:timeline
    for ii = 1:10
        dat = F_set{ii, tt};  % 获取第ii个细胞在第tt个时间点的钙信号
        trace_zs = zscore(dat);  % Z-score标准化
        x = 1/frameRate * linspace(1, length(trace_zs), length(trace_zs));  % 时间向量
        trace_zs_set{ii, tt} = trace_zs;  % 保存标准化数据
        
        % 计算互信息以确定最佳时间延迟tau
        mi = mutual(trace_zs);  % 计算互信息函数
        [~, mini] = findpeaks(-mi);  % 寻找互信息的第一个局部最小值
        if isempty(mini) == 1
            mini = 8;  % 如果没有找到局部最小值，使用经验值8
        end
        
        % 设置相空间参数
        dim = 3;  % 嵌入维度(3D相空间)
        tau = mini(1);  % 使用互信息第一个局部最小值作为时间延迟
        
        % 进行相空间重构
        y = phasespace(trace_zs, dim, tau);  % 生成相空间坐标
        xyz{ii, tt} = y;  % 保存相空间坐标
    end
end

% 裁剪相空间轨迹到统一长度，用于后续处理
xyz_len = 170;  % 经验确定的统一长度
xyz_trim = cellset2trim(xyz, xyz_len);  % 裁剪函数

%% 构建图数据集
% 将相空间轨迹转换为图数据集格式，包括节点、边和图属性
forPred = reshape(xyz_trim, [], 1);  % 重塑为列向量
pred_num = length(forPred);  % 样本总数

%%% 生成nodes.csv文件
% 构建节点ID和特征
graph_id = 1:pred_num;  % 图ID从1开始
graph_id = repmat(graph_id, xyz_len, 1);  % 重复xyz_len次
graph_id = reshape(graph_id, pred_num*xyz_len, 1);  % 重塑为列向量

node_id = 1:xyz_len;  % 节点ID从1开始
node_id = node_id';
node_id = repmat(node_id, pred_num, 1);

% 准备节点特征
feat1 = cell2mat(forPred);  % 将细胞数组转换为矩阵
feat = cell(length(feat1), 1);  % 初始化特征单元格数组

% 进度条
f = waitbar(0, '请稍候...');
for ii = 1:length(feat)
    waitbar(ii/length(feat), f, [num2str(ii), filesep, num2str(length(feat))]);
    feat{ii} = formatConvert(feat1(ii, :));  % 格式转换函数
end
close(f)  % 关闭进度条

% 保存节点数据到CSV文件
outfile = 'nodes.csv';
T = table(graph_id, node_id, feat);
writetable(T, fullfile(outPath, outfile));

%%% 生成edges.csv文件
% 构建边的源节点和目标节点
graph_id = 1:pred_num;  % 图ID从1开始
graph_id = repmat(graph_id, xyz_len-1, 1);  % 重复(xyz_len-1)次
graph_id = reshape(graph_id, pred_num*(xyz_len-1), 1);  % 重塑为列向量

src_id = (1:(xyz_len-1))';  % 源节点ID，从1到xyz_len-1
src_id = repmat(src_id, pred_num, 1);  % 为每个图重复

dst_id = src_id + 1;  % 目标节点ID等于源节点ID加1，形成连续连接
feat = ones(length(dst_id), 1);  % 边特征，这里设为1表示连接存在

% 保存边数据到CSV文件
T = table(graph_id, src_id, dst_id, feat);
outfile = 'edges.csv';
writetable(T, fullfile(outPath, outfile));

%%% 生成graphs.csv文件
% 构建图的属性和标签
graph_id = 1:pred_num;  % 图ID从1开始
graph_id = graph_id';

% 图特征和标签
clear feat label
feat{1} = '1,0,0,0,0,0';  % 图特征
feat = repmat(feat, length(graph_id), 1);  % 复制到所有图
label = zeros(length(feat), 1);  % 初始标签为0

% 保存图数据到CSV文件
T = table(graph_id, feat, label);
outfile = 'graphs.csv';
writetable(T, fullfile(outPath, outfile));

disp('全部完成!')  % 处理完成提示