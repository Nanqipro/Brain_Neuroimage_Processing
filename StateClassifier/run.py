#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脑网络状态分类器运行脚本

提供便捷的命令行接口来运行数据处理和模型训练的完整流程。

作者: SCN研究小组
日期: 2023
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# 导入项目模块
from config import config


def setup_logging(log_level: str = "INFO"):
    """
    设置日志配置
    
    Parameters
    ----------
    log_level : str
        日志级别
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('scn_classifier.log', encoding='utf-8')
        ]
    )


def run_data_processing():
    """
    运行数据处理流程
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始数据处理流程")
    logger.info("=" * 60)
    
    try:
        # 导入并运行数据处理脚本
        from scn_phase_space_process import main as process_main
        process_main()
        logger.info("✓ 数据处理完成")
        return True
    except Exception as e:
        logger.error(f"✗ 数据处理失败: {e}")
        return False


def run_model_training():
    """
    运行模型训练流程
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始模型训练流程")
    logger.info("=" * 60)
    
    # 检查数据文件是否存在
    required_files = [
        config.DATA_DIR / config.NODES_CSV,
        config.DATA_DIR / config.EDGES_CSV,
        config.DATA_DIR / config.GRAPHS_CSV
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error("缺少必要的数据文件:")
        for f in missing_files:
            logger.error(f"  - {f}")
        logger.info("请先运行数据处理流程: python run.py --process")
        return False
    
    try:
        # 导入并运行训练脚本
        from main import main as train_main
        model, test_acc, test_avg_class_acc = train_main()
        logger.info("✓ 模型训练完成")
        logger.info(f"最终测试准确率: {test_acc:.4f}")
        logger.info(f"平均类别准确率: {test_avg_class_acc:.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ 模型训练失败: {e}")
        return False


def run_complete_pipeline():
    """
    运行完整的处理和训练流程
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始完整的处理和训练流程")
    logger.info("=" * 60)
    
    # 步骤1: 数据处理
    if not run_data_processing():
        logger.error("数据处理失败，中止流程")
        return False
    
    # 步骤2: 模型训练
    if not run_model_training():
        logger.error("模型训练失败")
        return False
    
    logger.info("=" * 60)
    logger.info("完整流程执行成功!")
    logger.info("=" * 60)
    return True


def test_modules():
    """
    测试核心模块
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("测试核心模块")
    logger.info("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 
            str(Path("src") / "test_all_modules.py")
        ], capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            logger.info("✓ 核心模块测试通过")
            print(result.stdout)
            return True
        else:
            logger.error("✗ 核心模块测试失败")
            print(result.stderr)
            return False
    except Exception as e:
        logger.error(f"✗ 运行测试时出错: {e}")
        return False


def check_environment():
    """
    检查运行环境
    """
    logger = logging.getLogger(__name__)
    logger.info("检查运行环境...")
    
    # 检查Python版本
    import sys
    if sys.version_info < (3, 7):
        logger.error("需要Python 3.7或更高版本")
        return False
    
    # 检查必要的模块
    required_modules = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 
        'sklearn', 'tqdm', 'torch'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"缺少必要的模块: {missing_modules}")
        logger.info("请安装缺少的模块:")
        logger.info(f"pip install {' '.join(missing_modules)}")
        return False
    
    logger.info("✓ 运行环境检查通过")
    return True


def show_config():
    """
    显示当前配置
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("当前配置参数")
    logger.info("=" * 60)
    
    config_info = [
        ("数据处理配置", [
            ("帧率", f"{config.FRAME_RATE} Hz"),
            ("嵌入维度", config.EMBEDDING_DIM),
            ("轨迹长度", config.TRAJECTORY_LENGTH),
            ("最大处理细胞数", config.MAX_CELLS_PROCESS),
        ]),
        ("模型配置", [
            ("隐藏层维度", config.HIDDEN_DIM),
            ("GCN层数", config.NUM_GCN_LAYERS),
            ("分类类别数", config.NUM_CLASSES),
            ("Dropout率", config.DROPOUT_RATE),
        ]),
        ("训练配置", [
            ("训练轮数", config.NUM_EPOCHS),
            ("学习率", config.LEARNING_RATE),
            ("批次大小", config.BATCH_SIZE),
            ("计算设备", config.DEVICE),
        ]),
        ("文件路径", [
            ("输入数据", config.INPUT_DATA_PATH),
            ("数据目录", config.DATA_DIR),
            ("结果目录", config.RESULT_DIR),
        ])
    ]
    
    for section_name, items in config_info:
        logger.info(f"\n{section_name}:")
        for name, value in items:
            logger.info(f"  {name}: {value}")


def main():
    """
    主函数，解析命令行参数并执行相应操作
    """
    parser = argparse.ArgumentParser(
        description="脑网络状态分类器运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py --process              # 只运行数据处理
  python run.py --train                # 只运行模型训练
  python run.py --all                  # 运行完整流程
  python run.py --test                 # 测试核心模块
  python run.py --config               # 显示配置信息
  python run.py --check                # 检查运行环境
        """
    )
    
    # 添加命令行参数
    parser.add_argument(
        "--process", 
        action="store_true",
        help="运行数据处理流程"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true",
        help="运行模型训练流程"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="运行完整的处理和训练流程"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="测试核心模块"
    )
    
    parser.add_argument(
        "--config", 
        action="store_true",
        help="显示当前配置"
    )
    
    parser.add_argument(
        "--check", 
        action="store_true",
        help="检查运行环境"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="设置日志级别"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 创建必要的目录
    config.create_directories()
    
    # 执行相应操作
    success = True
    
    if args.check:
        success = check_environment()
    elif args.config:
        show_config()
    elif args.test:
        success = test_modules()
    elif args.process:
        success = run_data_processing()
    elif args.train:
        success = run_model_training()
    elif args.all:
        success = run_complete_pipeline()
    else:
        parser.print_help()
        return
    
    # 返回状态码
    if not success:
        logger.error("操作失败")
        sys.exit(1)
    else:
        logger.info("操作成功完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 