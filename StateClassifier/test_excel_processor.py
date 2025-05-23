#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel数据处理器测试脚本

用于测试和验证新的Excel钙离子数据处理功能，确保能正确读取和处理
EMtrace01.xlsx格式的数据文件。

作者: Clade 4
日期: 2025年5月23日
"""

import sys
import os
from pathlib import Path
import logging

# 添加项目路径
sys.path.append('.')

from config import config
from excel_data_processor import ExcelCalciumDataProcessor, load_excel_calcium_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_excel_data_loading():
    """
    测试Excel数据加载功能
    """
    logger.info("=" * 60)
    logger.info("测试1: Excel数据加载功能")
    logger.info("=" * 60)
    
    file_path = config.INPUT_DATA_PATH
    
    if not Path(file_path).exists():
        logger.error(f"测试文件不存在: {file_path}")
        return False
    
    try:
        # 创建处理器实例
        processor = ExcelCalciumDataProcessor(file_path)
        
        # 加载数据
        raw_data = processor.load_excel_data()
        
        # 检查数据基本信息
        logger.info(f"✓ 数据形状: {raw_data.shape}")
        logger.info(f"✓ 列名: {list(raw_data.columns)}")
        logger.info(f"✓ 时间戳范围: {raw_data['stamp'].min()} - {raw_data['stamp'].max()}")
        
        # 检查神经元列
        neuron_columns = [col for col in raw_data.columns if col != 'stamp']
        logger.info(f"✓ 神经元数量: {len(neuron_columns)}")
        logger.info(f"✓ 前5个神经元: {neuron_columns[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据加载测试失败: {e}")
        return False


def test_data_preprocessing():
    """
    测试数据预处理功能
    """
    logger.info("=" * 60)
    logger.info("测试2: 数据预处理功能")
    logger.info("=" * 60)
    
    file_path = config.INPUT_DATA_PATH
    
    try:
        processor = ExcelCalciumDataProcessor(file_path)
        processor.load_excel_data()
        
        # 预处理数据
        processed_data = processor.preprocess_data(
            remove_nan=True,
            z_score_normalize=True,
            outlier_threshold=5.0
        )
        
        logger.info(f"✓ 预处理后数据形状: {processed_data.shape}")
        logger.info(f"✓ 数据范围: {processed_data.min():.3f} - {processed_data.max():.3f}")
        logger.info(f"✓ 数据均值: {processed_data.mean():.3f}")
        logger.info(f"✓ 数据标准差: {processed_data.std():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据预处理测试失败: {e}")
        return False


def test_scn_format_conversion():
    """
    测试SCN格式转换功能
    """
    logger.info("=" * 60)
    logger.info("测试3: SCN格式转换功能")
    logger.info("=" * 60)
    
    file_path = config.INPUT_DATA_PATH
    
    try:
        processor = ExcelCalciumDataProcessor(file_path)
        processor.load_excel_data()
        processed_data = processor.preprocess_data()
        
        # 转换为SCN格式
        scn_data = processor.convert_to_scn_format(processed_data)
        
        logger.info(f"✓ SCN格式数据结构:")
        logger.info(f"  - 神经元数量: {len(scn_data)}")
        
        if scn_data:
            segment_counts = [len(segments) for segments in scn_data]
            logger.info(f"  - 时间段数量: {segment_counts}")
            logger.info(f"  - 平均段数: {sum(segment_counts)/len(segment_counts):.1f}")
            
            # 检查第一个神经元的第一个时间段
            if scn_data[0]:
                first_segment = scn_data[0][0]
                logger.info(f"  - 第一个时间段长度: {len(first_segment)}")
                logger.info(f"  - 第一个时间段范围: {first_segment.min():.3f} - {first_segment.max():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ SCN格式转换测试失败: {e}")
        return False


def test_convenience_function():
    """
    测试便捷函数
    """
    logger.info("=" * 60)
    logger.info("测试4: 便捷函数测试")
    logger.info("=" * 60)
    
    file_path = config.INPUT_DATA_PATH
    
    try:
        # 使用便捷函数
        scn_data, data_info = load_excel_calcium_data(file_path)
        
        logger.info("✓ 数据信息:")
        for key, value in data_info.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"✓ SCN数据结构: {len(scn_data)} 个神经元")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 便捷函数测试失败: {e}")
        return False


def test_data_info():
    """
    测试数据信息获取功能
    """
    logger.info("=" * 60)
    logger.info("测试5: 数据信息获取功能")
    logger.info("=" * 60)
    
    file_path = config.INPUT_DATA_PATH
    
    try:
        processor = ExcelCalciumDataProcessor(file_path)
        processor.load_excel_data()
        
        # 获取数据信息
        info = processor.get_data_info()
        
        logger.info("✓ 数据信息详细报告:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据信息获取测试失败: {e}")
        return False


def run_all_tests():
    """
    运行所有测试
    """
    logger.info("开始运行Excel数据处理器完整测试套件...")
    
    tests = [
        ("Excel数据加载", test_excel_data_loading),
        ("数据预处理", test_data_preprocessing), 
        ("SCN格式转换", test_scn_format_conversion),
        ("便捷函数", test_convenience_function),
        ("数据信息获取", test_data_info)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n正在运行测试: {test_name}")
        result = test_func()
        results.append((test_name, result))
        
        if result:
            logger.info(f"✓ {test_name} - 通过")
        else:
            logger.error(f"✗ {test_name} - 失败")
    
    # 测试结果汇总
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！Excel数据处理器工作正常。")
        return True
    else:
        logger.error("❌ 部分测试失败，请检查错误信息。")
        return False


if __name__ == "__main__":
    # 检查配置
    logger.info("Excel数据处理器测试脚本")
    logger.info(f"配置的输入文件: {config.INPUT_DATA_PATH}")
    
    # 检查文件是否存在
    if not Path(config.INPUT_DATA_PATH).exists():
        logger.error(f"测试文件不存在: {config.INPUT_DATA_PATH}")
        logger.info("请确保在datasets目录下有EMtrace01.xlsx文件")
        sys.exit(1)
    
    # 运行测试
    success = run_all_tests()
    
    if success:
        logger.info("\n✅ 可以继续运行完整的数据处理流程:")
        logger.info("python run.py --process")
        sys.exit(0)
    else:
        logger.error("\n❌ 测试失败，请检查并修复问题后重试")
        sys.exit(1) 