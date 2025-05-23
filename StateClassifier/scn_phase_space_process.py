#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D SCN数据处理流程 - Python版本

该脚本用于处理脑神经元钙成像数据，将时间序列转换为相空间流形，并构建图数据集
主要步骤包括：
1. 加载原始钙信号数据（支持Excel和MAT格式）
2. 将钙成像时间序列转换为相空间流形(phase-space manifolds)
3. 生成图数据集的节点、边和图属性文件

作者: SCN研究小组 (原MATLAB版本)
转换: Python版本，增强Excel支持
日期: 2025年5月23日
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import zscore
from scipy.signal import find_peaks
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
import logging

# 导入项目配置和本地模块
from config import config
sys.path.append('./src')
from src.mutual import mutual, find_optimal_delay
from src.phasespace import phasespace
from src.cellset2trim import cellset2trim
from src.format_convert import format_convert

# 导入新的Excel数据处理器
from excel_data_processor import load_excel_calcium_data, ExcelCalciumDataProcessor

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# 抑制警告
warnings.filterwarnings('ignore')


def load_scn_data(file_path: str) -> dict:
    """
    加载SCN数据文件（支持MAT格式，保持向后兼容性）
    
    Parameters
    ----------
    file_path : str
        MAT文件路径
        
    Returns
    -------
    dict
        加载的数据字典
    """
    try:
        data = scipy.io.loadmat(file_path)
        logger.info(f"✓ 成功加载MAT数据文件: {file_path}")
        return data
    except Exception as e:
        logger.error(f"✗ 加载MAT数据文件失败: {e}")
        sys.exit(1)


def detect_input_format(file_path: str) -> str:
    """
    检测输入文件格式
    
    Parameters
    ----------
    file_path : str
        文件路径
        
    Returns
    -------
    str
        文件格式: 'excel' 或 'mat'
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.xlsx', '.xls']:
        return 'excel'
    elif file_ext == '.mat':
        return 'mat'
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}。支持的格式: .xlsx, .xls, .mat")


def load_calcium_data(file_path: str) -> tuple:
    """
    智能加载钙离子数据（支持Excel和MAT格式）
    
    Parameters
    ----------
    file_path : str
        数据文件路径
        
    Returns
    -------
    tuple
        (F_set, data_info) - 钙信号数据集和信息字典
    """
    logger.info(f"正在检测和加载数据文件: {file_path}")
    
    # 检测文件格式
    format_type = detect_input_format(file_path)
    logger.info(f"检测到文件格式: {format_type}")
    
    if format_type == 'excel':
        # 使用新的Excel处理器
        logger.info("使用Excel数据处理器加载钙离子数据...")
        F_set, data_info = load_excel_calcium_data(file_path)
        
        # 添加格式信息
        data_info['数据格式'] = 'Excel'
        data_info['处理器'] = 'ExcelCalciumDataProcessor'
        
        return F_set, data_info
        
    elif format_type == 'mat':
        # 使用原有的MAT处理逻辑（保持向后兼容）
        logger.info("使用MAT数据处理器加载数据...")
        mat_data = load_scn_data(file_path)
        
        # 这里需要根据具体的MAT文件结构来解析
        # 假设MAT文件包含名为'F_set'的变量
        if 'F_set' in mat_data:
            F_set = mat_data['F_set']
        else:
            # 如果没有F_set，尝试其他可能的变量名
            possible_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            if possible_keys:
                F_set = mat_data[possible_keys[0]]
                logger.warning(f"未找到'F_set'变量，使用'{possible_keys[0]}'")
            else:
                raise ValueError("MAT文件中未找到有效的数据变量")
        
        data_info = {
            '数据格式': 'MAT',
            '文件路径': file_path,
            '变量数量': len([k for k in mat_data.keys() if not k.startswith('__')]),
            '主要变量': [k for k in mat_data.keys() if not k.startswith('__')][:5]
        }
        
        return F_set, data_info
    
    else:
        raise ValueError(f"不支持的文件格式: {format_type}")


def find_first_local_minimum(mi_values: np.ndarray, default_value: int = 8) -> int:
    """
    寻找互信息的第一个局部最小值
    
    Parameters
    ----------
    mi_values : np.ndarray
        互信息值数组
    default_value : int, default=8
        如果找不到局部最小值时的默认值
        
    Returns
    -------
    int
        第一个局部最小值的索引
    """
    # 使用scipy的find_peaks来寻找负值的峰值（即原函数的谷值）
    peaks, _ = find_peaks(-mi_values)
    
    if len(peaks) > 0:
        return peaks[0]  # 返回第一个局部最小值
    else:
        return default_value  # 使用默认值


def process_calcium_signals_to_phasespace(F_set: list, frame_rate: float) -> tuple:
    """
    将钙离子时间序列转换为相空间流形
    
    Parameters
    ----------
    F_set : list
        细胞钙信号数据集，格式为 [cell_num, timeline]
    frame_rate : float
        帧率，单位Hz
        
    Returns
    -------
    tuple
        (trace_zs_set, xyz) - 标准化数据和相空间坐标
    """
    cell_num, timeline = len(F_set), len(F_set[0])
    logger.info(f"数据维度: {cell_num} 个细胞, {timeline} 个时间线")
    
    # 初始化存储结构
    trace_zs_set = [[None for _ in range(timeline)] for _ in range(cell_num)]
    xyz = [[None for _ in range(timeline)] for _ in range(cell_num)]
    
    logger.info("开始处理钙信号数据...")
    
    # 使用配置中的参数
    max_cells = min(config.MAX_CELLS_PROCESS, cell_num)
    total_operations = max_cells * timeline
    
    pbar = tqdm(total=total_operations, desc="处理相空间重构", disable=not config.PROGRESS_BAR)
    
    for tt in range(timeline):
        for ii in range(max_cells):  # 使用配置参数控制处理的细胞数量
            # 获取第ii个细胞在第tt个时间点的钙信号
            dat = F_set[ii][tt]
            
            if dat is not None and len(dat) > 0:
                # 验证信号长度
                if len(dat) < config.MIN_SIGNAL_LENGTH:
                    logger.warning(f"细胞{ii+1},时间线{tt+1} 信号长度过短: {len(dat)}")
                    if not config.CONTINUE_ON_ERROR:
                        break
                    xyz[ii][tt] = None
                    pbar.update(1)
                    continue
                
                # Z-score标准化
                trace_zs = zscore(dat, axis=config.ZSCORE_AXIS)
                
                # 保存标准化数据
                trace_zs_set[ii][tt] = trace_zs
                
                # 计算互信息以确定最佳时间延迟tau
                mi = mutual(trace_zs, partitions=config.NUM_BINS, tau=config.MAX_DELAY)
                
                # 寻找互信息的第一个局部最小值
                mini = find_first_local_minimum(mi, default_value=8)
                
                # 设置相空间参数
                dim = config.EMBEDDING_DIM  # 嵌入维度
                tau = mini  # 使用互信息第一个局部最小值作为时间延迟
                
                # 进行相空间重构
                try:
                    y = phasespace(trace_zs, dim, tau)
                    xyz[ii][tt] = y
                except ValueError as e:
                    logger.warning(f"细胞{ii+1},时间线{tt+1} 相空间重构失败: {e}")
                    if not config.CONTINUE_ON_ERROR:
                        break
                    xyz[ii][tt] = None
            
            pbar.update(1)
    
    pbar.close()
    logger.info("✓ 相空间重构完成")
    
    return trace_zs_set, xyz


def trim_phasespace_data(xyz: list, xyz_len: int = None) -> list:
    """
    裁剪相空间轨迹到统一长度
    
    Parameters
    ----------
    xyz : list
        相空间坐标数据
    xyz_len : int, optional
        统一的轨迹长度，如果为None则使用配置值
        
    Returns
    -------
    list
        裁剪后的相空间数据
    """
    if xyz_len is None:
        xyz_len = config.TRAJECTORY_LENGTH
    
    logger.info(f"开始裁剪数据到统一长度: {xyz_len}")
    
    # 使用已转换的cellset2trim函数
    xyz_trim = cellset2trim(xyz, xyz_len)
    
    logger.info("✓ 数据裁剪完成")
    return xyz_trim


def generate_nodes_csv(xyz_trim: list, output_path: str) -> None:
    """
    生成nodes.csv文件
    
    Parameters
    ----------
    xyz_trim : list
        裁剪后的相空间数据
    output_path : str
        输出目录路径
    """
    logger.info("生成 nodes.csv 文件...")
    
    # 重塑数据为列向量
    forPred = [item for sublist in xyz_trim for item in sublist if item is not None]
    pred_num = len(forPred)
    
    if pred_num == 0:
        raise ValueError("没有有效的相空间数据用于生成节点文件")
    
    # 获取轨迹长度（假设所有非空轨迹长度相同）
    xyz_len = len(forPred[0]) if forPred[0] is not None else config.TRAJECTORY_LENGTH
    
    # 构建节点ID和特征
    graph_id = np.repeat(np.arange(1, pred_num + 1), xyz_len)
    node_id = np.tile(np.arange(1, xyz_len + 1), pred_num)
    
    # 准备节点特征
    feat1 = np.vstack(forPred)  # 将所有轨迹垂直堆叠
    
    # 进度条
    feat = []
    logger.info("转换节点特征格式...")
    for i in tqdm(range(len(feat1)), desc="格式转换", disable=not config.PROGRESS_BAR):
        feat.append(format_convert(feat1[i, :]))
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'graph_id': graph_id,
        'node_id': node_id,
        'feat': feat
    })
    
    output_file = os.path.join(output_path, config.NODES_CSV)
    df.to_csv(output_file, index=False)
    logger.info(f"✓ {config.NODES_CSV} 已保存到: {output_file}")


def generate_edges_csv(pred_num: int, xyz_len: int, output_path: str) -> None:
    """
    生成edges.csv文件
    
    Parameters
    ----------
    pred_num : int
        图的数量
    xyz_len : int
        每个图中的节点数量
    output_path : str
        输出目录路径
    """
    logger.info("生成 edges.csv 文件...")
    
    # 构建边的源节点和目标节点
    graph_id = np.repeat(np.arange(1, pred_num + 1), xyz_len - 1)
    src_id = np.tile(np.arange(1, xyz_len), pred_num)  # 源节点ID，从1到xyz_len-1
    dst_id = src_id + 1  # 目标节点ID等于源节点ID加1，形成连续连接
    feat = np.ones(len(dst_id))  # 边特征，设为1表示连接存在
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'graph_id': graph_id,
        'src_id': src_id,
        'dst_id': dst_id,
        'feat': feat.astype(int)
    })
    
    output_file = os.path.join(output_path, config.EDGES_CSV)
    df.to_csv(output_file, index=False)
    logger.info(f"✓ {config.EDGES_CSV} 已保存到: {output_file}")


def generate_graphs_csv(pred_num: int, output_path: str) -> None:
    """
    生成graphs.csv文件
    
    Parameters
    ----------
    pred_num : int
        图的数量
    output_path : str
        输出目录路径
    """
    logger.info("生成 graphs.csv 文件...")
    
    # 构建图的属性和标签
    graph_id = np.arange(1, pred_num + 1)
    feat = ['1,0,0,0,0,0'] * pred_num  # 图特征
    label = np.zeros(pred_num, dtype=int)  # 初始标签为0
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'graph_id': graph_id,
        'feat': feat,
        'label': label
    })
    
    output_file = os.path.join(output_path, config.GRAPHS_CSV)
    df.to_csv(output_file, index=False)
    logger.info(f"✓ {config.GRAPHS_CSV} 已保存到: {output_file}")


def generate_mock_data():
    """
    生成模拟数据用于演示
    
    Returns
    -------
    list
        模拟的F_set数据
    """
    logger.info("生成模拟数据进行演示...")
    
    # 生成模拟数据
    np.random.seed(config.RANDOM_SEED)
    cell_num, timeline = config.MAX_CELLS_PROCESS, 3
    F_set = []
    for i in range(cell_num):
        cell_data = []
        for j in range(timeline):
            # 生成模拟的钙信号数据
            signal_length = np.random.randint(200, 400)
            calcium_signal = np.sin(np.linspace(0, 4*np.pi, signal_length)) + \
                           0.1 * np.random.randn(signal_length)
            cell_data.append(calcium_signal)
        F_set.append(cell_data)
    
    logger.info(f"✓ 生成了 {cell_num} 个细胞, {timeline} 个时间线的模拟数据")
    return F_set


def main():
    """
    主处理函数 - 支持Excel和MAT格式的钙离子数据
    """
    logger.info("=" * 60)
    logger.info("3D SCN数据处理流程 - Python版本")
    logger.info("增强版本：支持Excel和MAT格式")
    logger.info("=" * 60)
    
    # 创建输出目录
    config.create_directories()
    logger.info(f"输出目录: {config.DATA_DIR}")
    
    # 步骤1: 智能加载数据
    logger.info("\n步骤1: 智能数据加载")
    logger.info("-" * 30)
    
    if not os.path.exists(config.INPUT_DATA_PATH):
        logger.warning(f"数据文件不存在: {config.INPUT_DATA_PATH}")
        logger.info("生成模拟数据进行演示...")
        F_set = generate_mock_data()
        data_info = {
            '数据格式': '模拟数据',
            '处理器': 'generate_mock_data',
            '细胞数量': len(F_set),
            '时间线数量': len(F_set[0]) if F_set else 0
        }
    else:
        try:
            # 使用智能数据加载器
            F_set, data_info = load_calcium_data(config.INPUT_DATA_PATH)
            
            # 打印数据信息
            logger.info("数据加载信息:")
            for key, value in data_info.items():
                logger.info(f"  {key}: {value}")
                
        except Exception as e:
            logger.error(f"✗ 数据加载失败: {e}")
            logger.info("使用模拟数据作为备选...")
            F_set = generate_mock_data()
            data_info = {
                '数据格式': '模拟数据(备选)',
                '原因': str(e)
            }
    
    # 验证数据有效性
    if not F_set or len(F_set) == 0:
        logger.error("✗ 没有有效的输入数据")
        return
    
    # 步骤2: 将钙离子时间序列转换为相空间流形
    logger.info("\n步骤2: 相空间重构")
    logger.info("-" * 30)
    
    try:
        trace_zs_set, xyz = process_calcium_signals_to_phasespace(F_set, config.FRAME_RATE)
    except Exception as e:
        logger.error(f"✗ 相空间重构失败: {e}")
        return
    
    # 步骤3: 裁剪相空间轨迹到统一长度
    logger.info("\n步骤3: 数据裁剪")
    logger.info("-" * 30)
    
    try:
        xyz_trim = trim_phasespace_data(xyz)
    except Exception as e:
        logger.error(f"✗ 数据裁剪失败: {e}")
        return
    
    # 步骤4: 构建图数据集
    logger.info("\n步骤4: 构建图数据集")
    logger.info("-" * 30)
    
    # 计算有效样本数量
    forPred = [item for sublist in xyz_trim for item in sublist if item is not None]
    pred_num = len(forPred)
    
    logger.info(f"有效样本数量: {pred_num}")
    
    if pred_num == 0:
        logger.error("✗ 没有有效的相空间数据，无法生成图数据集")
        logger.info("建议检查:")
        logger.info("  1. 输入数据质量（是否包含有效的钙信号）")
        logger.info("  2. 配置参数（MIN_SIGNAL_LENGTH, MAX_CELLS_PROCESS等）")
        logger.info("  3. 数据预处理结果")
        return
    
    # 生成CSV文件
    try:
        generate_nodes_csv(xyz_trim, str(config.DATA_DIR))
        generate_edges_csv(pred_num, config.TRAJECTORY_LENGTH, str(config.DATA_DIR))
        generate_graphs_csv(pred_num, str(config.DATA_DIR))
        
        logger.info("\n" + "=" * 60)
        logger.info("数据处理流程全部完成!")
        logger.info("=" * 60)
        logger.info(f"数据源信息:")
        logger.info(f"  - 文件: {config.INPUT_DATA_PATH}")
        logger.info(f"  - 格式: {data_info.get('数据格式', '未知')}")
        logger.info(f"  - 处理器: {data_info.get('处理器', '未知')}")
        logger.info(f"\n生成的文件:")
        logger.info(f"  - {config.DATA_DIR}/{config.NODES_CSV}")
        logger.info(f"  - {config.DATA_DIR}/{config.EDGES_CSV}")
        logger.info(f"  - {config.DATA_DIR}/{config.GRAPHS_CSV}")
        logger.info(f"\n处理结果:")
        logger.info(f"  - 总样本数量: {pred_num}")
        logger.info(f"  - 轨迹长度: {config.TRAJECTORY_LENGTH}")
        logger.info(f"  - 嵌入维度: {config.EMBEDDING_DIM}")
        
        # 添加下一步建议
        logger.info(f"\n下一步操作:")
        logger.info(f"  1. 检查生成的CSV文件质量")
        logger.info(f"  2. 运行训练: python main.py 或 python run.py --train")
        logger.info(f"  3. 查看训练结果: {config.RESULT_DIR}")
        
    except Exception as e:
        logger.error(f"✗ 生成CSV文件时出错: {e}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        return


def validate_environment():
    """
    验证运行环境和依赖
    """
    required_modules = ['numpy', 'pandas', 'scipy', 'tqdm']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"✗ 缺少依赖模块: {missing_modules}")
        logger.info("请安装缺少的模块:")
        logger.info(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True


if __name__ == "__main__":
    # 验证配置
    try:
        config.validate_config()
        logger.info("✓ 配置验证通过")
    except Exception as e:
        logger.error(f"✗ 配置验证失败: {e}")
        sys.exit(1)
    
    # 验证环境
    if not validate_environment():
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n用户中断程序")
    except Exception as e:
        logger.error(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 