function data_trim = cellset2trim(dataset, trim_len)
%% 细胞数组裁剪函数
%
% 该函数将细胞数组中的每个非空元素裁剪到指定长度，
% 主要用于统一相空间轨迹的长度，便于后续处理和分析。
%
% 输入参数:
%   dataset - 输入的细胞数组，包含相空间轨迹数据
%   trim_len - 裁剪后的目标长度
%
% 输出参数:
%   data_trim - 裁剪后的细胞数组，每个非空元素都被裁剪到相同的长度
%

% 获取数据集的维度（细胞数量和时间线）
[cellNum, timeline] = size(dataset);

% 初始化结果细胞数组，与输入数组大小相同
data_trim = cell(size(dataset));

% 遍历所有细胞和时间点
for ii = 1:cellNum
    for jj = 1:timeline
        % 获取当前细胞和时间点的数据
        temp = dataset{ii, jj};
        
        % 如果数据非空，则裁剪到指定长度
        if isempty(temp) == 0
            data_trim{ii, jj} = temp(1:trim_len, :);
        end
    end
end

end