#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的主训练脚本
================

整合所有改进的组件，包括智能标签生成、高级模型架构、先进训练技术

Author: AI Assistant
Date: 2024
"""

import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入项目模块
import config
from utils import get_dataset, set_random_seeds
from improved_models import create_improved_model
from improved_training import AdvancedTrainer
from improved_labeling import generate_improved_labels

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImprovedPipeline:
    """
    改进的训练流水线
    """
    
    def __init__(self):
        """初始化改进的流水线"""
        self.device = None
        self.model = None
        self.trainer = None
        self.results = {}
        
        # 验证配置
        config.validate_config()
        logger.info("配置验证完成")
        
    def setup_environment(self):
        """设置环境"""
        # 设置随机种子
        set_random_seeds()
        logger.info("✓ 随机种子设置完成")
        
        # 设置计算设备
        if config.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"✓ 使用GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            logger.info("✓ 使用CPU")
    
    def load_data(self):
        """加载数据"""
        logger.info("加载数据集...")
        try:
            train_dataloader, valid_dataloader, test_dataloader = get_dataset(
                data_path=str(config.DATA_DIR),
                train_propotion=config.TRAIN_RATIO,
                valid_propotion=config.VALID_RATIO,
                BATCH_SIZE=config.BATCH_SIZE
            )
            
            logger.info(f"✓ 数据集加载成功")
            logger.info(f"  - 训练集批次数: {len(train_dataloader)}")
            logger.info(f"  - 验证集批次数: {len(valid_dataloader)}")
            logger.info(f"  - 测试集批次数: {len(test_dataloader)}")
            
            return train_dataloader, valid_dataloader, test_dataloader
            
        except Exception as e:
            logger.error(f"✗ 数据集加载失败: {e}")
            raise
    
    def create_model(self, model_type='advanced', architecture='hybrid'):
        """
        创建改进的模型
        
        Parameters
        ----------
        model_type : str
            模型类型: 'advanced', 'ensemble', 'adaptive'
        architecture : str
            架构类型: 'gat', 'gcn', 'transformer', 'hybrid'
        """
        logger.info(f"创建模型: {model_type} ({architecture})")
        
        self.model = create_improved_model(
            model_type=model_type,
            input_dim=3,  # 相空间维度
            hidden_dims=[128, 256, 128],  # 增大网络容量
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT_RATE,
            architecture=architecture
        ).to(self.device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"✓ 模型创建成功")
        logger.info(f"模型类型: {model_type}")
        logger.info(f"架构: {architecture}")
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        return self.model
    
    def create_trainer(self, loss_type='focal', optimizer_type='adamw', scheduler_type='reduce_lr'):
        """
        创建高级训练器
        
        Parameters
        ----------
        loss_type : str
            损失函数类型
        optimizer_type : str
            优化器类型
        scheduler_type : str
            学习率调度器类型
        """
        logger.info(f"创建训练器: {loss_type} + {optimizer_type} + {scheduler_type}")
        
        self.trainer = AdvancedTrainer(
            model=self.model,
            device=self.device,
            num_classes=config.NUM_CLASSES,
            loss_type=loss_type,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            use_early_stopping=True
        )
        
        logger.info("✓ 训练器创建成功")
        return self.trainer
    
    def train_model(self, train_loader, val_loader, epochs=100):
        """
        训练模型
        
        Parameters
        ----------
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader
            验证数据加载器
        epochs : int
            训练轮数
            
        Returns
        -------
        dict
            训练历史
        """
        logger.info(f"开始训练模型，总共 {epochs} 个epoch...")
        
        # 训练模型
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=True
        )
        
        # 保存最佳模型
        config.create_directories()
        torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)
        logger.info(f"✓ 最佳模型已保存: {config.BEST_MODEL_PATH}")
        
        return history
    
    def evaluate_model(self, test_loader):
        """
        评估模型
        
        Parameters
        ----------
        test_loader : DataLoader
            测试数据加载器
            
        Returns
        -------
        dict
            评估结果
        """
        logger.info("在测试集上评估模型...")
        
        # 评估模型
        test_loss, test_acc, test_balanced_acc, predictions, labels = self.trainer.evaluate(test_loader)
        
        # 生成详细报告
        classification_report = self.trainer.generate_classification_report(predictions, labels)
        
        # 保存结果
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_balanced_accuracy': test_balanced_acc,
            'predictions': predictions,
            'labels': labels,
            'classification_report': classification_report
        }
        
        logger.info("=" * 60)
        logger.info("最终测试结果:")
        logger.info(f"  测试损失: {test_loss:.4f}")
        logger.info(f"  测试准确率: {test_acc:.4f}")
        logger.info(f"  测试平衡准确率: {test_balanced_acc:.4f}")
        logger.info("=" * 60)
        
        return results
    
    def run_experiment(self, experiment_name="default"):
        """
        运行完整实验
        
        Parameters
        ----------
        experiment_name : str
            实验名称
            
        Returns
        -------
        dict
            实验结果
        """
        logger.info("=" * 80)
        logger.info(f"开始实验: {experiment_name}")
        logger.info("=" * 80)
        
        try:
            # 1. 环境设置
            self.setup_environment()
            
            # 2. 数据加载
            train_loader, val_loader, test_loader = self.load_data()
            
            # 3. 模型创建
            model = self.create_model(model_type='advanced', architecture='hybrid')
            
            # 4. 训练器创建
            trainer = self.create_trainer(
                loss_type='focal',
                optimizer_type='adamw', 
                scheduler_type='reduce_lr'
            )
            
            # 5. 模型训练
            history = self.train_model(train_loader, val_loader, epochs=100)
            
            # 6. 模型评估
            results = self.evaluate_model(test_loader)
            
            # 7. 结果可视化
            self.visualize_results(history, results, experiment_name)
            
            # 8. 保存实验结果
            self.save_experiment_results(experiment_name, history, results)
            
            logger.info(f"实验 '{experiment_name}' 完成！")
            
            return {
                'experiment_name': experiment_name,
                'history': history,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"实验 '{experiment_name}' 失败: {e}")
            raise
    
    def run_ablation_study(self):
        """
        运行消融研究，比较不同配置的性能
        
        Returns
        -------
        dict
            消融研究结果
        """
        logger.info("=" * 80)
        logger.info("开始消融研究")
        logger.info("=" * 80)
        
        # 定义不同的配置
        configurations = [
            {
                'name': 'Baseline_GCN_CE',
                'model_type': 'advanced',
                'architecture': 'gcn',
                'loss_type': 'cross_entropy',
                'optimizer_type': 'adam',
                'scheduler_type': 'reduce_lr'
            },
            {
                'name': 'Improved_GAT_Focal',
                'model_type': 'advanced',
                'architecture': 'gat',
                'loss_type': 'focal',
                'optimizer_type': 'adamw',
                'scheduler_type': 'reduce_lr'
            },
            {
                'name': 'Advanced_Hybrid_Focal',
                'model_type': 'advanced',
                'architecture': 'hybrid',
                'loss_type': 'focal',
                'optimizer_type': 'adamw',
                'scheduler_type': 'cosine'
            },
            {
                'name': 'Ensemble_All',
                'model_type': 'ensemble',
                'architecture': 'hybrid',  # 参数会被忽略
                'loss_type': 'focal',
                'optimizer_type': 'adamw',
                'scheduler_type': 'reduce_lr'
            }
        ]
        
        ablation_results = {}
        
        for config_dict in configurations:
            logger.info(f"\n运行配置: {config_dict['name']}")
            logger.info("-" * 50)
            
            try:
                # 设置环境
                self.setup_environment()
                
                # 加载数据
                train_loader, val_loader, test_loader = self.load_data()
                
                # 创建模型
                model = self.create_model(
                    model_type=config_dict['model_type'],
                    architecture=config_dict['architecture']
                )
                
                # 创建训练器
                trainer = self.create_trainer(
                    loss_type=config_dict['loss_type'],
                    optimizer_type=config_dict['optimizer_type'],
                    scheduler_type=config_dict['scheduler_type']
                )
                
                # 训练模型（使用较少的epoch以节省时间）
                history = self.train_model(train_loader, val_loader, epochs=50)
                
                # 评估模型
                results = self.evaluate_model(test_loader)
                
                # 保存结果
                ablation_results[config_dict['name']] = {
                    'config': config_dict,
                    'best_val_acc': max(history['val_acc']),
                    'test_accuracy': results['test_accuracy'],
                    'test_balanced_accuracy': results['test_balanced_accuracy']
                }
                
                logger.info(f"配置 '{config_dict['name']}' 完成")
                logger.info(f"  最佳验证准确率: {ablation_results[config_dict['name']]['best_val_acc']:.4f}")
                logger.info(f"  测试准确率: {results['test_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"配置 '{config_dict['name']}' 失败: {e}")
                ablation_results[config_dict['name']] = {'error': str(e)}
        
        # 生成消融研究报告
        self.generate_ablation_report(ablation_results)
        
        return ablation_results
    
    def visualize_results(self, history, results, experiment_name):
        """
        可视化结果
        
        Parameters
        ----------
        history : dict
            训练历史
        results : dict
            评估结果
        experiment_name : str
            实验名称
        """
        logger.info("生成可视化结果...")
        
        try:
            # 创建输出目录
            output_dir = Path("results") / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 绘制训练历史
            self.trainer.plot_training_history(
                save_path=output_dir / "training_history.png"
            )
            
            # 绘制混淆矩阵
            self.trainer.plot_confusion_matrix(
                predictions=results['predictions'],
                labels=results['labels'],
                save_path=output_dir / "confusion_matrix.png"
            )
            
            logger.info(f"✓ 可视化结果已保存到: {output_dir}")
            
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")
    
    def save_experiment_results(self, experiment_name, history, results):
        """
        保存实验结果
        
        Parameters
        ----------
        experiment_name : str
            实验名称
        history : dict
            训练历史
        results : dict
            评估结果
        """
        logger.info("保存实验结果...")
        
        try:
            # 创建输出目录
            output_dir = Path("results") / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存训练历史
            pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
            
            # 保存测试结果
            results_summary = {
                'test_loss': results['test_loss'],
                'test_accuracy': results['test_accuracy'],
                'test_balanced_accuracy': results['test_balanced_accuracy']
            }
            pd.DataFrame([results_summary]).to_csv(output_dir / "test_results.csv", index=False)
            
            # 保存分类报告
            with open(output_dir / "classification_report.txt", 'w') as f:
                f.write(results['classification_report'])
            
            logger.info(f"✓ 实验结果已保存到: {output_dir}")
            
        except Exception as e:
            logger.warning(f"结果保存失败: {e}")
    
    def generate_ablation_report(self, ablation_results):
        """
        生成消融研究报告
        
        Parameters
        ----------
        ablation_results : dict
            消融研究结果
        """
        logger.info("生成消融研究报告...")
        
        # 创建结果表格
        report_data = []
        for name, result in ablation_results.items():
            if 'error' not in result:
                report_data.append({
                    'Configuration': name,
                    'Best_Val_Acc': result['best_val_acc'],
                    'Test_Accuracy': result['test_accuracy'],
                    'Test_Balanced_Acc': result['test_balanced_accuracy']
                })
        
        if report_data:
            df = pd.DataFrame(report_data)
            df = df.sort_values('Test_Accuracy', ascending=False)
            
            logger.info("\n" + "=" * 80)
            logger.info("消融研究结果汇总")
            logger.info("=" * 80)
            logger.info(df.to_string(index=False))
            logger.info("=" * 80)
            
            # 保存报告
            output_dir = Path("results") / "ablation_study"
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_dir / "ablation_results.csv", index=False)
            
            logger.info(f"✓ 消融研究报告已保存到: {output_dir}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("脑网络状态分类器 - 改进版本")
    logger.info("=" * 80)
    
    # 创建改进的流水线
    pipeline = ImprovedPipeline()
    
    try:
        # 选择运行模式
        import argparse
        parser = argparse.ArgumentParser(description='改进的脑网络状态分类器')
        parser.add_argument('--mode', type=str, default='experiment',
                          choices=['experiment', 'ablation'],
                          help='运行模式')
        parser.add_argument('--name', type=str, default='improved_experiment',
                          help='实验名称')
        
        args = parser.parse_args()
        
        if args.mode == 'experiment':
            # 运行单个实验
            results = pipeline.run_experiment(args.name)
            logger.info("单个实验完成！")
            
        elif args.mode == 'ablation':
            # 运行消融研究
            results = pipeline.run_ablation_study()
            logger.info("消融研究完成！")
        
        logger.info("程序执行成功！")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 