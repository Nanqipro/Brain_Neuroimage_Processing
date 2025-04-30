import numpy as np
import scipy.io as scio
import torch

# 设置日期标识符
date = '20210916'

# 加载数据函数
def load_SCN(scn_data_path, task):
    """
    加载SCN（超交叉核）钙信号数据并根据指定任务进行预处理
    
    参数:
    ----------
    scn_data_path : str
        SCN数据的MATLAB文件路径，通常由scn_data_process.m生成
    task : str
        数据预处理任务类型，支持以下选项:
        - 'standard'或'pc-sample': 标准格式或神经元采样格式(24小时，每小时200个时间点)
        - 'time-sample': 时间采样格式(24小时，每小时100个时间点)
        - '1_3-sample': 第1段8小时数据(0-8小时)
        - '2_3-sample': 第2段8小时数据(8-16小时)
        - '3_3-sample': 第3段8小时数据(16-24小时)
    
    返回:
    -------
    train : numpy.ndarray
        根据任务重塑后的训练数据，形状为(神经元数, 时间段数, 每段时间点数)
    poi : torch.FloatTensor
        神经元的三维空间坐标
    """
    # 加载MATLAB数据文件
    scn_data = scio.loadmat(scn_data_path)
    # 提取钙信号轨迹并转置，使形状变为(神经元数, 时间点数)
    trace = scn_data['trace'].T # trace
    # 提取神经元位置坐标并转换为PyTorch张量
    poi = torch.FloatTensor(scn_data['POI'])

    # 根据不同任务类型处理数据
    if task == 'standard' or task == 'pc-sample':
        # 标准格式：24小时 × 200个点/小时 = 4800个时间点
        trace = trace[:,0:4800]
        # 重塑为三维数组(神经元数, 24小时, 200点/小时)
        train = np.reshape(trace, (trace.shape[0], 24, 200))
    elif task == 'time-sample':
        # 时间采样格式：24小时 × 100个点/小时 = 2400个时间点
        trace = trace[:,0:2400]
        # 重塑为三维数组(神经元数, 24小时, 100点/小时)
        train = np.reshape(trace, (trace.shape[0], 24, 100))
    elif task == '1_3-sample':
        # 第一段8小时数据(0-8小时)
        trace = trace[:,0:1600]
        # 重塑为三维数组(神经元数, 8小时, 200点/小时)
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '2_3-sample':
        # 第二段8小时数据(8-16小时)
        trace = trace[:,1600:3200]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '3_3-sample':
        # 第三段8小时数据(16-24小时)
        trace = trace[:,3200:4800]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
        
    return train, poi