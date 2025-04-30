function [ Y ] = phasespace(signal, dim, tau)
%% 相空间重构函数
%
% 该函数将一维时间序列重构为高维相空间轨迹，用于非线性动力学分析。
% 利用时间延迟嵌入方法，将一维信号映射到多维空间，以揭示系统的动力学特性。
%
%Author: Hui Yang
%Affiliation: 
       %The Pennsylvania State University
       %310 Leohard Building, University Park, PA
       %Email: yanghui@gmail.com
%
% 输入参数:
%   signal: 输入时间序列，一维数组
%   dim: 嵌入维度，表示重构相空间的维数
%   tau: 时间延迟，用于确定相空间点的构建方式
%
% 输出参数:
%   Y: 重构的相空间轨迹矩阵，大小为 T×dim，每行代表相空间中的一个点
%

% If you find this demo useful, please cite the following paper:
% [1]	H. Yang,Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) 
% Signals, IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011
% DOI: 10.1109/TBME.2010.2063704
% [2]	Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and 
% nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012 
% DOI: 10.1016/j.chaos.2012.03.013

% 获取信号长度
N = length(signal);

% 计算相空间中的总点数
% 考虑时间延迟和嵌入维度，相空间中的点数会减少
T = N - (dim-1) * tau;

% 初始化相空间矩阵
Y = zeros(T, dim);

% 构建相空间轨迹
% 对每个时间点，基于时间延迟tau和嵌入维度dim构建相空间中的对应点
for i = 1:T
   % 使用降序排列的延迟索引构建相空间点
   % 每个点由当前时刻及其过去的值组成
   Y(i, :) = signal(i + (dim-1)*tau - sort((0:dim-1), 'descend')*tau)';
end

% 获取相空间维度
sizeY = size(Y, 2);

% 如果没有输出参数，则绘制相空间轨迹图
if nargout == 0
    if sizeY == 2
        % 2D相空间可视化
        plot(Y(:, 1), Y(:, 2));
        xlabel('y1', 'FontSize', 10, 'FontWeight', 'bold');
        ylabel('y2', 'FontSize', 10, 'FontWeight', 'bold');
        get(gcf, 'CurrentAxes');
        set(gca, 'FontSize', 10, 'FontWeight', 'bold');
        grid on;
    else
        % 3D相空间可视化
        plot3(Y(:, 1), Y(:, 2), Y(:, 3));
        xlabel('y1', 'FontSize', 10, 'FontWeight', 'bold');
        ylabel('y2', 'FontSize', 10, 'FontWeight', 'bold');
        zlabel('y3', 'FontSize', 10, 'FontWeight', 'bold');
        get(gcf, 'CurrentAxes');
        set(gca, 'FontSize', 10, 'FontWeight', 'bold');
        grid on;
    end
end