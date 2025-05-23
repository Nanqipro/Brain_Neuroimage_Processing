#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的模块功能测试脚本

测试所有从MATLAB转换而来的Python模块的功能，
确保转换后的代码工作正常。

Author: Converted from MATLAB
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import os

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from format_convert import format_convert
    from mutual import mutual, find_optimal_delay
    from phasespace import phasespace, estimate_embedding_params
    from cellset2trim import cellset2trim, cellset2trim_dict, get_dataset_stats, validate_trim_length
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)


def test_format_convert():
    """测试格式转换模块"""
    print("\n" + "="*50)
    print("测试 format_convert 模块")
    print("="*50)
    
    try:
        # 测试1: 基本功能
        data1 = [1.23, 4.56, 7.89]
        result1 = format_convert(data1)
        expected1 = "1.23,4.56,7.89"
        assert result1 == expected1, f"期望: {expected1}, 实际: {result1}"
        print(f"✓ 基本转换测试通过: {result1}")
        
        # 测试2: numpy数组
        data2 = np.array([1.0, 2.0, 3.0, 4.0])
        result2 = format_convert(data2)
        print(f"✓ NumPy数组转换: {result2}")
        
        # 测试3: 二维数组
        data3 = np.array([[1, 2], [3, 4]])
        result3 = format_convert(data3)
        print(f"✓ 二维数组转换: {result3}")
        
        return True
        
    except Exception as e:
        print(f"✗ format_convert 测试失败: {e}")
        return False


def test_mutual():
    """测试互信息计算模块"""
    print("\n" + "="*50)
    print("测试 mutual 模块")
    print("="*50)
    
    try:
        # 生成测试信号
        np.random.seed(42)
        t = np.linspace(0, 10*np.pi, 500)
        signal = np.sin(t) + 0.1*np.random.randn(len(t))
        
        # 测试1: 基本互信息计算
        mi_values = mutual(signal, partitions=16, tau=20)
        assert len(mi_values) == 21, f"期望长度21，实际长度{len(mi_values)}"
        print(f"✓ 互信息计算成功，长度: {len(mi_values)}")
        print(f"  前5个值: {mi_values[:5]}")
        
        # 测试2: 寻找最佳延迟
        optimal_delay = find_optimal_delay(signal, max_tau=20)
        print(f"✓ 最佳延迟估计: {optimal_delay}")
        
        # 测试3: 绘图功能（如果在交互环境中）
        if '--plot' in sys.argv:
            print("✓ 生成互信息图...")
            mutual(signal, partitions=16, tau=20, plot_result=True)
        
        return True
        
    except Exception as e:
        print(f"✗ mutual 测试失败: {e}")
        return False


def test_phasespace():
    """测试相空间重构模块"""
    print("\n" + "="*50)
    print("测试 phasespace 模块")
    print("="*50)
    
    try:
        # 生成测试信号
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, 1000)
        signal = np.sin(t) + 0.5*np.sin(3*t)
        
        # 测试1: 基本相空间重构
        Y = phasespace(signal, dim=3, tau=10)
        expected_shape = (1000 - (3-1)*10, 3)
        assert Y.shape == expected_shape, f"期望形状{expected_shape}，实际形状{Y.shape}"
        print(f"✓ 相空间重构成功，形状: {Y.shape}")
        print(f"  前3个点:\n{Y[:3]}")
        
        # 测试2: 参数估计
        optimal_dim, optimal_tau = estimate_embedding_params(signal)
        print(f"✓ 参数估计 - 维度: {optimal_dim}, 延迟: {optimal_tau}")
        
        # 测试3: 使用估计参数重构
        Y2 = phasespace(signal, dim=optimal_dim, tau=optimal_tau)
        print(f"✓ 使用估计参数重构，形状: {Y2.shape}")
        
        # 测试4: 绘图功能
        if '--plot' in sys.argv:
            print("✓ 生成相空间图...")
            phasespace(signal[:500], dim=3, tau=10, plot_result=True)
        
        return True
        
    except Exception as e:
        print(f"✗ phasespace 测试失败: {e}")
        return False


def test_cellset2trim():
    """测试细胞数组裁剪模块"""
    print("\n" + "="*50)
    print("测试 cellset2trim 模块")
    print("="*50)
    
    try:
        # 创建测试数据集
        np.random.seed(42)
        data1 = np.random.randn(100, 3)
        data2 = np.random.randn(150, 3)
        data3 = np.random.randn(80, 3)
        data4 = np.random.randn(120, 3)
        
        # 测试1: 列表格式裁剪
        dataset = [
            [data1, None, data3],
            [data2, data4, None],
            [None, data1, data2]
        ]
        
        # 获取统计信息
        stats = get_dataset_stats(dataset)
        print(f"✓ 数据集统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 验证裁剪长度
        trim_length = 75
        is_valid = validate_trim_length(dataset, trim_length)
        assert is_valid, "裁剪长度验证失败"
        print(f"✓ 裁剪长度验证通过: {trim_length}")
        
        # 执行裁剪
        trimmed_dataset = cellset2trim(dataset, trim_length)
        print(f"✓ 列表格式裁剪完成")
        
        # 验证结果
        for ii in range(len(trimmed_dataset)):
            for jj in range(len(trimmed_dataset[0])):
                data = trimmed_dataset[ii][jj]
                if data is not None:
                    assert data.shape[0] == trim_length, f"位置[{ii},{jj}]长度不正确"
                    print(f"  位置[{ii},{jj}]: 形状 {data.shape}")
        
        # 测试2: 字典格式裁剪
        dict_dataset = {
            'cell1_t1': np.random.randn(100, 3),
            'cell1_t2': None,
            'cell2_t1': np.random.randn(80, 3),
            'cell2_t2': np.random.randn(120, 3)
        }
        
        trimmed_dict = cellset2trim_dict(dict_dataset, 60)
        print(f"✓ 字典格式裁剪完成")
        
        for key, data in trimmed_dict.items():
            if data is not None:
                assert data.shape[0] == 60, f"键'{key}'长度不正确"
                print(f"  {key}: 形状 {data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ cellset2trim 测试失败: {e}")
        return False


def test_integration():
    """集成测试：模拟完整的数据处理流程"""
    print("\n" + "="*50)
    print("集成测试：完整数据处理流程")
    print("="*50)
    
    try:
        # 步骤1: 生成多个时间序列
        print("步骤1: 生成测试数据...")
        np.random.seed(42)
        
        # 生成3个不同的信号
        t1 = np.linspace(0, 10*np.pi, 800)
        signal1 = np.sin(t1) + 0.1*np.random.randn(len(t1))
        
        t2 = np.linspace(0, 8*np.pi, 600)
        signal2 = np.cos(2*t2) + 0.1*np.random.randn(len(t2))
        
        t3 = np.linspace(0, 12*np.pi, 1000)
        signal3 = np.sin(t3) + 0.5*np.cos(3*t3) + 0.1*np.random.randn(len(t3))
        
        # 步骤2: 为每个信号找到最佳延迟
        print("步骤2: 估计最佳参数...")
        delays = []
        for i, signal in enumerate([signal1, signal2, signal3]):
            delay = find_optimal_delay(signal, max_tau=20)
            delays.append(delay)
            print(f"  信号{i+1}最佳延迟: {delay}")
        
        # 步骤3: 进行相空间重构
        print("步骤3: 相空间重构...")
        trajectories = []
        for i, (signal, delay) in enumerate(zip([signal1, signal2, signal3], delays)):
            Y = phasespace(signal, dim=3, tau=delay)
            trajectories.append(Y)
            print(f"  信号{i+1}轨迹形状: {Y.shape}")
        
        # 步骤4: 创建细胞数组数据集
        print("步骤4: 创建数据集...")
        dataset = [
            [trajectories[0], None, trajectories[1]],
            [trajectories[2], trajectories[0], None],
            [None, trajectories[1], trajectories[2]]
        ]
        
        # 步骤5: 分析和裁剪数据
        print("步骤5: 数据分析和裁剪...")
        stats = get_dataset_stats(dataset)
        print(f"  数据集统计: 非空{stats['non_empty_count']}个，最小长度{stats['min_length']}")
        
        trim_length = int(stats['min_length'] * 0.8)  # 取最小长度的80%
        
        if validate_trim_length(dataset, trim_length):
            trimmed_dataset = cellset2trim(dataset, trim_length)
            print(f"  裁剪完成，统一长度: {trim_length}")
        
        # 步骤6: 数据格式转换
        print("步骤6: 数据格式转换...")
        total_points = 0
        for i, cell_data in enumerate(trimmed_dataset):
            for j, trajectory in enumerate(cell_data):
                if trajectory is not None:
                    # 只转换前几个点作为示例
                    sample_points = trajectory[:3]
                    csv_strings = [format_convert(point) for point in sample_points]
                    total_points += len(trajectory)
                    print(f"  细胞{i+1}_时间线{j+1}: {len(trajectory)}个点")
        
        print(f"✓ 集成测试成功！总共处理了{total_points}个数据点")
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始测试所有模块...")
    print("使用 --plot 参数可以显示图形")
    
    # 抑制一些不重要的警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 运行所有测试
    tests = [
        ("格式转换", test_format_convert),
        ("互信息计算", test_mutual),
        ("相空间重构", test_phasespace),
        ("数据裁剪", test_cellset2trim),
        ("集成测试", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n开始测试 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✓ {test_name} 测试通过")
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<15}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！MATLAB代码已成功转换为Python。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查代码。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 