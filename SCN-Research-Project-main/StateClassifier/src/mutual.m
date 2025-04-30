function mi = mutual(signal, partitions, tau)
%% 时间延迟互信息计算函数
%
% 该函数计算时间序列的时间延迟互信息，用于确定相空间重构的最佳时间延迟。
% 互信息是衡量两个变量之间相互依赖性的度量，第一个局部最小值通常被选为最佳时间延迟。
%
%Author: Hui Yang
%Affiliation: 
       %The Pennsylvania State University
       %310 Leohard Building, University Park, PA
       %Email: yanghui@gmail.com
%
% 输入参数:
%   signal - 输入时间序列，一维数组
%   partitions - 分区框数，用于概率估计的离散化
%   tau - 最大时间延迟
%
% 输出参数:
%   mi - 从延迟0到tau的互信息值数组
%

% If you find this demo useful, please cite the following paper:
% [1]	H. Yang,Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) 
% Signals, IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011
% DOI: 10.1109/TBME.2010.2063704
% [2]	Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and 
% nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012 
% DOI: 10.1016/j.chaos.2012.03.013

% 计算信号的基本统计量
av = mean(signal);
variance = var(signal);
minimum = min(signal);
maximum = max(signal);
interval = maximum-minimum;
len = length(signal);

% 参数默认值设置
if nargin<2 | isempty(partitions)
  partitions = 16;  % 默认分区数为16
end
if nargin<3 | isempty(tau)
  tau = 20;  % 默认最大延迟为20
end

% 将信号标准化到[0,1]区间
for i = 1:1:len
    signal(i) =(signal(i)- minimum)/interval;
end

% 将标准化信号离散化为整数值(1到partitions)
for i = 1:1:len
    if signal(i) > 0 
        array(i) = ceil(signal(i)*partitions);
    else
        array(i) = 1;
    end
end

% 计算延迟为0时的香农熵(作为基准)
shannon = make_cond_entropy(0, array, len, partitions);
    
% 确保延迟不超过信号长度
if (tau >= len)
    tau = len-1;
end

% 计算延迟从0到tau的互信息
for i = 0:1:tau
    mi(i+1) = make_cond_entropy(i, array, len, partitions);
end

% 如果没有输出参数，则绘制互信息图
if nargout == 0
    figure('Position', [100 400 460 360]);
    plot(0:1:tau, mi, 'o-', 'MarkerSize', 5);
    title('互信息测试（寻找第一个局部最小值）', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('延迟（采样时间）', 'FontSize', 10, 'FontWeight', 'bold');
    ylabel('互信息值', 'FontSize', 10, 'FontWeight', 'bold');
    get(gcf, 'CurrentAxes');
    set(gca, 'FontSize', 10, 'FontWeight', 'bold');
    grid on;
end


%% 计算条件熵的内部函数
function mi = make_cond_entropy(t, array, len, partitions)
% 该内部函数计算给定时间延迟t的条件熵（互信息）
%
% 输入参数:
%   t - 时间延迟
%   array - 离散化后的信号数组
%   len - 信号长度
%   partitions - 分区数量
%
% 输出参数:
%   mi - 计算得到的互信息值

% 初始化变量
hi = 0;
hii = 0;
count = 0;
hpi = 0;
hpj = 0;
pij = 0;
cond_ent = 0.0;

% 初始化联合概率和边缘概率的计数数组
h2 = zeros(partitions, partitions);  % 联合概率的计数
h1 = zeros(1, partitions);           % 第一个变量的边缘概率计数
h11 = zeros(1, partitions);          % 第二个变量的边缘概率计数

% 初始化边缘计数数组
for i = 1:1:partitions
    h1(i) = 0;
    h11(i) = 0;
end

% 统计不同延迟时的联合出现频率和边缘频率
for i = 1:1:len
    if i > t  % 确保有足够的延迟
        hii = array(i);          % 当前时间点的值
        hi = array(i-t);         % 延迟t后的值
        h1(hi) = h1(hi) + 1;     % 累加边缘频率
        h11(hii) = h11(hii) + 1;
        h2(hi, hii) = h2(hi, hii) + 1;  % 累加联合频率
        count = count + 1;
    end
end

% 计算归一化因子
norm = 1.0 / double(count);
cond_ent = 0.0;

% 计算互信息（基于条件熵）
for i = 1:1:partitions
    hpi = double(h1(i)) * norm;  % 第一个变量的边缘概率
    if hpi > 0.0
        for j = 1:1:partitions
            hpj = double(h11(j)) * norm;  % 第二个变量的边缘概率
            if hpj > 0.0
                pij = double(h2(i, j)) * norm;  % 联合概率
                if (pij > 0.0)
                    % 基于联合概率和边缘概率计算互信息
                    cond_ent = cond_ent + pij * log(pij / hpj / hpi);
                end
            end
        end
    end
end

mi = cond_ent;  % 返回计算得到的互信息值

