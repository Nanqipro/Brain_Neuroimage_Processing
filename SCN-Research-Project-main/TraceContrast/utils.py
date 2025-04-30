import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime

def pkl_save(name, var):
    """
    将变量保存为pickle格式文件
    
    参数:
    -------
    name : str
        输出文件路径
    var : object
        要保存的变量
    """
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    """
    加载pickle格式的文件
    
    参数:
    -------
    name : str
        pickle文件路径
        
    返回:
    -------
    object
        加载的对象
    """
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    """
    用NaN值填充张量
    
    参数:
        arr: 需要填充的张量
        left: 左侧填充的长度
        right: 右侧填充的长度
        dim: 需要填充的维度
        
    返回:
        填充后的张量
    """
    if left == 0 and right == 0:
        return arr
    
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    
    if dim >= len(arr.shape):
        dim = len(arr.shape) - 1
    
    pad_size = list(arr.shape)
    pad_size[dim] = left
    pad_left = torch.full(pad_size, np.nan)
    pad_size[dim] = right
    pad_right = torch.full(pad_size, np.nan)
    
    res = torch.cat([pad_left, arr, pad_right], dim=dim)
    return res
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    """
    将NumPy数组在指定轴上填充NaN值至目标长度
    
    参数:
    -------
    array : np.ndarray
        要填充的数组
    target_length : int
        目标长度
    axis : int
        填充的轴
    both_side : bool
        是否两侧都填充，若为False则只在右侧填充
        
    返回:
    -------
    np.ndarray
        填充后的数组
    """
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    """
    将数组分割成多个部分，并在每个部分之间插入一行NaN值
    
    参数:
        x: 需要分割的数组
        sections: 分割的段数
        axis: 分割的轴
        
    返回:
        分割后的数组列表
    """
    if axis >= len(x.shape):
        axis = len(x.shape) - 1
        
    arrs = np.array_split(x, sections, axis=axis)
    if sections > 1:
        for i in range(len(arrs) - 1):
            if hasattr(arrs[i], 'columns'):
                arrs[i].loc[arrs[i].shape[0]] = np.nan
            else:
                shape = list(arrs[i].shape)
                shape[axis] = 1
                arrs[i] = np.concatenate([arrs[i], np.full(shape, np.nan)], axis=axis)
    return arrs

def take_per_row(x, starts, ends):
    """
    按行提取数据的指定子序列
    
    参数:
        x: 输入数据，形状为(样本数, 时间点数, 特征数)
        starts: 每行序列的起始索引
        ends: 每行序列的结束索引
        
    返回:
        提取的子序列，形状为(样本数, max(结束索引-起始索引), 特征数)
    """
    b = x.size(0)
    r = torch.arange(b)
    
    if len(x.shape) == 2:
        return x[r[:, None], torch.arange(starts[0], ends[0])[None, :]]
    else:
        return x[r[:, None], torch.arange(starts[0], ends[0])[None, :], :]

def centerize_vary_length_series(x):
    """
    对长度不同的时间序列进行居中处理
    
    通过将所有非NaN值移动到中间位置，使原始序列居中对齐。
    首先寻找序列中第一个和最后一个非NaN的位置，然后计算序列应该移动的距离，
    最后重新构建一个新序列，将非NaN值居中放置。
    
    参数:
        x: 需要居中的序列，形状为(样本数, 时间点数, 特征数)
        
    返回:
        居中后的序列
    """
    prefix_zeros = np.sum(np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.sum(np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    t_len = x.shape[1]
    max_offset = np.max(suffix_zeros - prefix_zeros)
    min_offset = np.min(suffix_zeros - prefix_zeros)
    center_offset = (max_offset + min_offset) // 2
    
    new_x = np.full_like(x, np.nan)
    for i, _ in enumerate(x):
        bulge = suffix_zeros[i] - prefix_zeros[i] - center_offset
        new_x[i, max(bulge, 0):min(t_len + bulge, t_len)] = x[i, max(-bulge, 0):min(t_len - bulge, t_len)]
    return new_x

def data_dropout(arr, p):
    """
    随机将数组中的部分元素替换为NaN值
    
    参数:
    -------
    arr : np.ndarray
        输入数组
    p : float
        元素被替换为NaN的概率，取值范围[0,1]
        
    返回:
    -------
    np.ndarray
        应用dropout后的数组
    """
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    """
    生成带有当前日期时间的字符串名称
    
    参数:
    -------
    prefix : str
        名称前缀
        
    返回:
    -------
    str
        格式为"prefix_YYYYMMDD_HHMMSS"的字符串
    """
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    """
    初始化深度学习程序的运行环境
    
    参数:
    -------
    device_name : str or int or list
        设备名称或ID，如'cuda:0'、0或['cuda:0','cuda:1']
    seed : int, optional
        随机数种子，若提供则设置确定性
    use_cudnn : bool
        是否使用cudnn
    deterministic : bool
        是否使用确定性算法
    benchmark : bool
        是否使用cudnn的benchmark模式
    use_tf32 : bool
        是否使用TensorFloat-32精度
    max_threads : int, optional
        最大线程数
        
    返回:
    -------
    torch.device or list of torch.device
        可用设备对象
    """
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # 设置内部操作线程数
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # 设置跨操作并行线程数
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    # 设置随机种子以确保结果可复现
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device('cpu') ### 注意：这里将设备强制设为CPU
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    # 设置TensorFloat-32精度
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

def setup_seed(seed):
    """
    设置随机种子，确保实验结果可复现
    
    参数:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

