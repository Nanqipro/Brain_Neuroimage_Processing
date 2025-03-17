"""
模型训练模块

该模块提供训练和评估GNN模型的功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import json
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class ModelTrainer:
    """
    模型训练器
    
    用于训练和评估GNN模型，记录训练过程，保存最佳模型
    """
    
    def __init__(self, model, config, device=None):
        """
        初始化训练器
        
        参数:
            model: 待训练的模型
            config: 配置对象，包含训练参数
            device: 训练设备，如果为None则自动选择
        """
        self.model = model
        self.config = config
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 将模型移动到设备
        self.model = self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=5, 
            verbose=True
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
        
        # 最佳模型参数
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
    def train_epoch(self, data):
        """
        训练一个epoch
        
        参数:
            data: 训练数据
            
        返回:
            训练损失
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 将数据移动到设备
        data = data.to(self.device)
        
        # 前向传播
        embeddings = self.model(data)
        
        # 计算损失（使用对比损失）
        loss = self.contrastive_loss(embeddings)
        
        # 反向传播和优化
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data):
        """
        评估模型
        
        参数:
            data: 评估数据
            
        返回:
            评估损失
        """
        self.model.eval()
        
        with torch.no_grad():
            # 将数据移动到设备
            data = data.to(self.device)
            
            # 前向传播
            embeddings = self.model(data)
            
            # 计算损失
            loss = self.contrastive_loss(embeddings)
            
        return loss.item()
    
    def contrastive_loss(self, embeddings, temperature=0.5):
        """
        计算对比损失
        
        参数:
            embeddings: 节点嵌入
            temperature: 温度参数
            
        返回:
            对比损失
        """
        # 归一化嵌入
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # 屏蔽对角线
        mask = torch.eye(similarity_matrix.shape[0], device=self.device)
        similarity_matrix = similarity_matrix * (1 - mask) - 10.0 * mask
        
        # 计算损失
        logits = torch.log_softmax(similarity_matrix, dim=1)
        loss = -torch.mean(torch.diag(logits))
        
        return loss
    
    def train(self, train_data, val_data=None, epochs=None, patience=None):
        """
        训练模型
        
        参数:
            train_data: 训练数据
            val_data: 验证数据，如果为None则使用训练数据
            epochs: 训练轮数，如果为None则使用配置中的默认值
            patience: 早停耐心值，如果为None则使用配置中的默认值
            
        返回:
            训练历史记录
        """
        if epochs is None:
            epochs = self.config.epochs
        if patience is None:
            patience = self.config.patience
        if val_data is None:
            val_data = train_data
            
        print(f"开始训练，共 {epochs} 轮...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_data)
            
            # 评估
            val_loss = self.evaluate(val_data)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录训练历史
            epoch_time = time.time() - start_time
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_times'].append(epoch_time)
            
            print(f"Epoch {epoch}/{epochs}: 训练损失={train_loss:.4f}，验证损失={val_loss:.4f}，耗时={epoch_time:.2f}秒")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                print(f"新的最佳模型！验证损失={val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"早停：{patience} 轮内验证损失未改善")
                    break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("已恢复最佳模型")
            
        return self.history
    
    def save_model(self, model_path: Optional[Path] = None):
        """
        保存模型
        
        参数:
            model_path: 模型保存路径，如果为None则使用配置中的默认路径
        """
        if model_path is None:
            model_type = self.config.model_type
            model_path = self.config.get_model_path(f"neuron_{model_type}")
            
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, model_path)
        
        print(f"模型已保存至 {model_path}")
    
    def load_model(self, model_path: Optional[Path] = None):
        """
        加载模型
        
        参数:
            model_path: 模型加载路径，如果为None则使用配置中的默认路径
        """
        if model_path is None:
            model_type = self.config.model_type
            model_path = self.config.get_model_path(f"neuron_{model_type}")
            
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"模型已从 {model_path} 加载")
        
    def plot_training_history(self, output_path: Optional[Path] = None):
        """
        绘制训练历史曲线
        
        参数:
            output_path: 图像保存路径，如果为None则使用配置中的默认路径
        """
        if output_path is None:
            model_type = self.config.model_type
            output_path = self.config.get_results_path(f"training_history_{model_type}.png")
            
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.title('训练与验证损失')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 绘制每轮耗时
        plt.subplot(1, 2, 2)
        plt.plot(self.history['epoch_times'])
        plt.xlabel('Epoch')
        plt.ylabel('耗时（秒）')
        plt.title('每轮训练耗时')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"训练历史曲线已保存至 {output_path}")
    
    def extract_neuron_embeddings(self, data, node_names):
        """
        提取神经元嵌入
        
        参数:
            data: PyTorch Geometric数据对象
            node_names: 节点名称列表
            
        返回:
            神经元嵌入和相似度矩阵
        """
        self.model.eval()
        
        with torch.no_grad():
            # 将数据移动到设备
            data = data.to(self.device)
            
            # 获取嵌入
            embeddings = self.model.get_embeddings(data)
            
            # 将嵌入移回CPU
            embeddings = embeddings.cpu().numpy()
            
            # 计算相似度矩阵
            similarities = cosine_similarity(embeddings)
            
        # 创建嵌入字典
        embedding_dict = {name: emb for name, emb in zip(node_names, embeddings)}
        
        return embedding_dict, similarities
    
    def visualize_embeddings(self, embeddings, node_names, output_path: Optional[Path] = None):
        """
        可视化嵌入
        
        参数:
            embeddings: 节点嵌入字典或数组
            node_names: 节点名称列表
            output_path: 图像保存路径，如果为None则使用配置中的默认路径
        """
        if output_path is None:
            model_type = self.config.model_type
            output_path = self.config.get_results_path(f"embeddings_{model_type}.png")
            
        # 如果是字典，提取嵌入数组
        if isinstance(embeddings, dict):
            embeddings_array = np.array([embeddings[name] for name in node_names])
        else:
            embeddings_array = embeddings
            
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
        
        # 添加节点标签
        for i, name in enumerate(node_names):
            plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        
        plt.title(f'神经元嵌入可视化 ({self.config.model_type.upper()})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"嵌入可视化已保存至 {output_path}")
    
    def create_gnn_topology(self, embeddings, node_names, threshold=0.7, output_path: Optional[Path] = None):
        """
        创建基于GNN的拓扑结构
        
        参数:
            embeddings: 节点嵌入字典或数组
            node_names: 节点名称列表
            threshold: 添加边的相似度阈值
            output_path: 拓扑结构保存路径，如果为None则使用配置中的默认路径
            
        返回:
            NetworkX图对象
        """
        if output_path is None:
            model_type = self.config.model_type
            output_path = self.config.get_results_path(f"topology_{model_type}.json")
            
        # 如果是字典，提取嵌入数组
        if isinstance(embeddings, dict):
            embeddings_array = np.array([embeddings[name] for name in node_names])
        else:
            embeddings_array = embeddings
            
        # 计算相似度矩阵
        similarities = cosine_similarity(embeddings_array)
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for i, name in enumerate(node_names):
            # 将numpy数组转换为Python列表，同时将float32转换为Python原生float
            embedding = [float(x) for x in embeddings_array[i].tolist()]
            G.add_node(name, embedding=embedding)
            
        # 添加边
        for i in range(len(node_names)):
            for j in range(i+1, len(node_names)):
                if similarities[i, j] > threshold:
                    # 将float32转换为Python原生float
                    G.add_edge(node_names[i], node_names[j], weight=float(similarities[i, j]))
        
        # 保存拓扑结构
        topology_data = {
            'nodes': [],
            'edges': []
        }
        
        for node, data in G.nodes(data=True):
            topology_data['nodes'].append({
                'id': node,
                'embedding': data['embedding']  # 已经转换为Python列表和float类型
            })
            
        for u, v, data in G.edges(data=True):
            topology_data['edges'].append({
                'source': u,
                'target': v,
                'weight': float(data['weight'])  # 确保转换为Python原生float
            })
        
        # 确保所有类型都是JSON兼容的   
        topology_data = self._convert_to_native_types(topology_data)
            
        try:
            with open(output_path, 'w') as f:
                json.dump(topology_data, f, indent=2)
                
            print(f"拓扑结构已保存至 {output_path}")
        except Exception as e:
            print(f"保存拓扑结构时出错: {str(e)}")
            print("尝试使用备选方法保存...")
            try:
                # 转换为字符串后尝试重新解析，确保所有类型兼容
                topology_data_str = json.dumps(topology_data)
                topology_data_clean = json.loads(topology_data_str)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(topology_data_clean, f, indent=2)
                    
                print(f"使用备选方法成功保存拓扑结构至 {output_path}")
            except Exception as e2:
                print(f"备选保存方法也失败: {str(e2)}")
                print("跳过拓扑结构保存")
        
        return G
        
    def _convert_to_native_types(self, obj):
        """
        递归地将NumPy类型转换为Python原生类型
        
        参数:
            obj: 任意对象，可能包含NumPy类型
            
        返回:
            转换后的对象，所有NumPy类型替换为Python原生类型
        """
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_native_types(obj.tolist())
        elif isinstance(obj, dict):
            return {self._convert_to_native_types(key): self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_native_types(item) for item in obj)
        else:
            return obj 