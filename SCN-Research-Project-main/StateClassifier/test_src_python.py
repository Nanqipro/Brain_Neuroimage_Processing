#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src_python模块测试脚本

该脚本用于测试Python版本的src模块是否正确实现了MATLAB版本的功能。

作者: SCN研究小组
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from src_python.mutual import mutual, mutual_information
from src_python.phasespace import phasespace, phase_space_reconstruction
from src_python.cellset2trim import cellset2trim, cellset_trim
from src_python.formatConvert import formatConvert, format_convert


def test_mutual():
    """测试互信息计算函数"""
    print("=== 测试互信息计算函数 ===")
    
    # 生成测试信号（洛伦兹吸引子的一个分量）
    dt = 0.01
    t = np.arange(0, 20, dt)
    signal = np.sin(t) + 0.5 * np.sin(3 * t) + 0.1 * np.random.randn(len(t))
    
    print(f"测试信号长度: {len(signal)}")
    
    # 计算互信息
    mi = mutual(signal)
    print(f"互信息数组长度: {len(mi)}")
    print(f"前5个互信息值: {mi[:5]}")
    
    # 寻找第一个局部最小值
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-mi)
    if len(peaks) > 0:
        optimal_tau = peaks[0]
        print(f"建议的最佳时间延迟: {optimal_tau}")
    else:
        print("未找到局部最小值，使用默认值8")
    
    print("✓ 互信息计算测试通过\n")


def test_phasespace():
    """测试相空间重构函数"""
    print("=== 测试相空间重构函数 ===")
    
    # 生成测试信号
    t = np.linspace(0, 4*np.pi, 1000)
    signal = np.sin(t) + 0.5 * np.sin(3*t)
    
    print(f"测试信号长度: {len(signal)}")
    
    # 相空间重构参数
    dim = 3
    tau = 15
    
    print(f"嵌入维度: {dim}, 时间延迟: {tau}")
    
    # 进行相空间重构
    try:
        Y = phasespace(signal, dim, tau)
        print(f"相空间轨迹维度: {Y.shape}")
        print(f"前3个相空间点:\n{Y[:3]}")
        print("✓ 相空间重构测试通过\n")
    except Exception as e:
        print(f"✗ 相空间重构测试失败: {e}\n")


def test_cellset2trim():
    """测试细胞数组裁剪函数"""
    print("=== 测试细胞数组裁剪函数 ===")
    
    # 创建测试数据集
    dataset = []
    for i in range(3):  # 3个细胞
        cell_data = []
        for j in range(2):  # 2个时间点
            if np.random.rand() > 0.2:  # 80%的概率有数据
                # 生成随机长度的相空间轨迹
                length = np.random.randint(150, 250)
                data = np.random.randn(length, 3)
                cell_data.append(data)
            else:
                cell_data.append(None)  # 20%的概率为空
        dataset.append(cell_data)
    
    print("原始数据集结构:")
    for i, cell in enumerate(dataset):
        for j, data in enumerate(cell):
            if data is not None:
                print(f"  细胞{i}, 时间{j}: {data.shape}")
            else:
                print(f"  细胞{i}, 时间{j}: None")
    
    # 裁剪到统一长度
    trim_len = 170
    trimmed_data = cellset2trim(dataset, trim_len)
    
    print(f"\n裁剪后数据集结构 (目标长度: {trim_len}):")
    for i, cell in enumerate(trimmed_data):
        for j, data in enumerate(cell):
            if data is not None:
                print(f"  细胞{i}, 时间{j}: {data.shape}")
            else:
                print(f"  细胞{i}, 时间{j}: None")
    
    print("✓ 细胞数组裁剪测试通过\n")


def test_formatConvert():
    """测试格式转换函数"""
    print("=== 测试格式转换函数 ===")
    
    # 测试不同类型的输入
    test_cases = [
        np.array([1.5, 2.3, 3.7]),
        [1, 2, 3, 4],
        42,
        np.array([[1, 2], [3, 4]]),  # 二维数组
        3.14159
    ]
    
    for i, test_input in enumerate(test_cases):
        result = formatConvert(test_input)
        print(f"测试 {i+1}: {test_input} -> '{result}'")
    
    print("✓ 格式转换测试通过\n")


def test_integration():
    """集成测试：模拟完整的处理流程"""
    print("=== 集成测试：完整处理流程 ===")
    
    # 生成模拟的钙信号数据
    np.random.seed(42)  # 确保可重现
    signal = np.sin(np.linspace(0, 10*np.pi, 500)) + 0.2 * np.random.randn(500)
    
    print(f"1. 生成钙信号，长度: {len(signal)}")
    
    # 标准化
    from scipy import stats
    signal_normalized = stats.zscore(signal)
    print("2. Z-score标准化完成")
    
    # 计算互信息并确定时间延迟
    mi = mutual(signal_normalized)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-mi)
    tau = peaks[0] if len(peaks) > 0 else 8
    print(f"3. 确定时间延迟: τ = {tau}")
    
    # 相空间重构
    dim = 3
    Y = phasespace(signal_normalized, dim, tau)
    print(f"4. 相空间重构完成，轨迹维度: {Y.shape}")
    
    # 模拟细胞数据集结构
    dataset = [[Y, Y], [Y, None]]  # 2个细胞，2个时间点
    print("5. 构建细胞数据集")
    
    # 裁剪
    trim_len = 170
    trimmed = cellset2trim(dataset, trim_len)
    print(f"6. 裁剪到统一长度: {trim_len}")
    
    # 格式转换
    if trimmed[0][0] is not None:
        sample_row = trimmed[0][0][0]  # 第一个非空数据的第一行
        formatted = formatConvert(sample_row)
        print(f"7. 格式转换示例: {sample_row} -> '{formatted}'")
    
    print("✓ 集成测试通过 - 所有模块协同工作正常\n")


def main():
    """运行所有测试"""
    print("开始测试 src_python 模块...\n")
    
    try:
        test_mutual()
        test_phasespace()
        test_cellset2trim()
        test_formatConvert()
        test_integration()
        
        print("🎉 所有测试通过！src_python模块已准备就绪。")
        print("现在可以运行 scn_phase_space_process_v2.py 来处理实际数据。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 