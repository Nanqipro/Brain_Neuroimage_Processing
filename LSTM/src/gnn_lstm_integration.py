import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 添加GNN模块路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'GNN', 'src'))

# 导入GNN模型
try:
    from gnn_models import (BrainNodeFeatureExtractor, BrainGNN, NeuronGNNLSTM, 
                          convert_networkx_to_pyg, create_dynamic_graph_data)
    print("成功导入GNN模型")
except ImportError as e:
    print(f"导入GNN模型失败: {e}")
    print("请确保GNN/src/gnn_models.py文件存在")
    sys.exit(1)

# 导入LSTM分析模块
from kmeans_lstm_analysis import NeuronDataProcessor, EnhancedNeuronLSTM
from analysis_results import ResultAnalyzer


class GNNLSTMIntegrator:
    """
    GNN和LSTM集成器
    
    将图神经网络与LSTM模型集成，用于神经元连接分析
    """
    def __init__(self, config):
        """
        初始化GNN-LSTM集成器
        
        参数:
            config: 配置对象
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # 创建输出目录
        os.makedirs(config.gnn_output_dir, exist_ok=True)
        
        # 初始化数据处理器
        self.processor = NeuronDataProcessor(config)
        
        # 初始化结果分析器
        self.analyzer = ResultAnalyzer(config)
    
    def create_neuron_graph(self, X_scaled, threshold=0.3):
        """
        基于神经元活动数据创建功能连接图
        
        参数:
            X_scaled: 标准化后的神经元活动数据
            threshold: 相关性阈值
            
        返回:
            G: NetworkX图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
        """
        print("\n构建神经元功能连接网络...")
        
        # 获取实际的神经元数量
        n_neurons = X_scaled.shape[1]
        
        # 获取可用的神经元编号列表
        if hasattr(self.processor, 'available_neuron_cols'):
            available_neurons = self.processor.available_neuron_cols
        else:
            # 如果处理器没有保存可用神经元列表，使用默认方法
            print("警告: 找不到可用神经元列表，使用连续编号")
            available_neurons = [f'n{i+1}' for i in range(n_neurons)]
        
        # 创建相关性矩阵
        correlation_matrix = np.corrcoef(X_scaled.T)
        
        # 确保对角线为0（移除自相关）
        np.fill_diagonal(correlation_matrix, 0)
        
        # 创建图对象
        G = nx.Graph()
        
        # 添加节点(使用实际的神经元编号)
        for neuron in available_neurons:
            G.add_node(neuron)
        
        # 添加边(基于相关性阈值)
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                if abs(correlation_matrix[i, j]) >= threshold:
                    G.add_edge(available_neurons[i], 
                              available_neurons[j], 
                              weight=abs(correlation_matrix[i, j]))
        
        print(f"网络构建完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
        return G, correlation_matrix, available_neurons
    
    def create_windowed_correlation_matrices(self, X_scaled, window_size=20, step_size=10):
        """
        创建滑动窗口相关矩阵序列
        
        参数:
            X_scaled: 标准化后的神经元活动数据
            window_size: 窗口大小
            step_size: 滑动步长
            
        返回:
            correlation_matrices: 相关矩阵列表
            time_indices: 对应的时间索引
        """
        print("\n创建滑动窗口相关矩阵...")
        
        n_samples, n_neurons = X_scaled.shape
        n_windows = (n_samples - window_size) // step_size + 1
        
        correlation_matrices = []
        time_indices = []
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            # 提取当前窗口数据
            window_data = X_scaled[start_idx:end_idx, :]
            
            # 计算相关矩阵
            corr_matrix = np.corrcoef(window_data.T)
            
            # 确保对角线为0（移除自相关）
            np.fill_diagonal(corr_matrix, 0)
            
            correlation_matrices.append(corr_matrix)
            time_indices.append(start_idx + window_size // 2)  # 窗口中点作为时间索引
        
        print(f"生成了 {len(correlation_matrices)} 个相关矩阵")
        return correlation_matrices, time_indices
    
    def prepare_graph_data(self, correlation_matrices, feature_type='connection_profile', threshold=0.3):
        """
        准备图数据用于GNN分析
        
        参数:
            correlation_matrices: 相关矩阵列表
            feature_type: 节点特征类型
            threshold: 相关性阈值
            
        返回:
            graph_data_list: PyG数据对象列表
        """
        print("\n准备图数据...")
        
        time_steps = len(correlation_matrices)
        graph_data_list = create_dynamic_graph_data(
            correlation_matrices, 
            time_steps, 
            threshold, 
            feature_type
        )
        
        print(f"创建了 {len(graph_data_list)} 个PyG图数据对象")
        return graph_data_list
    
    def train_gnn_model(self, graph_data_list, labels, feature_dim, 
                       gnn_hidden_dim=64, output_dim=32, message_type='node_edge_concat'):
        """
        训练独立的GNN模型
        
        参数:
            graph_data_list: PyG数据对象列表
            labels: 标签
            feature_dim: 节点特征维度
            gnn_hidden_dim: GNN隐藏层维度
            output_dim: 输出维度
            message_type: 消息传递类型
            
        返回:
            model: 训练好的GNN模型
            node_embeddings: 节点嵌入
        """
        print("\n训练GNN模型...")
        
        # 转换标签为tensor
        y = torch.tensor(labels, dtype=torch.long)
        
        # 创建GNN模型
        model = BrainGNN(
            in_channels=feature_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=output_dim,
            message_type=message_type,
            num_layers=2,
            pooling='mean'
        ).to(self.device)
        
        # 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        # 划分训练集和验证集
        train_indices, val_indices = train_test_split(
            np.arange(len(graph_data_list)), 
            test_size=0.2, 
            random_state=self.config.random_seed
        )
        
        # 训练模型
        model.train()
        for epoch in range(100):  # 训练100轮
            # 随机打乱训练索引
            np.random.shuffle(train_indices)
            
            # 训练一个epoch
            total_loss = 0
            for idx in train_indices:
                # 获取数据
                data = graph_data_list[idx]
                target = y[idx]
                
                # 前向传播
                optimizer.zero_grad()
                out = model(data)
                loss = torch.nn.functional.cross_entropy(out, target.unsqueeze(0))
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            model.eval()
            correct = 0
            for idx in val_indices:
                data = graph_data_list[idx]
                pred = model(data).argmax(dim=1)
                correct += (pred == y[idx]).sum().item()
            
            val_acc = correct / len(val_indices)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}, Loss: {total_loss/len(train_indices):.4f}, Val Acc: {val_acc:.4f}')
        
        # 提取节点嵌入
        model.eval()
        node_embeddings = []
        with torch.no_grad():
            for data in graph_data_list:
                # 获取GNN每层的节点嵌入
                embeddings = []
                x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
                
                # 通过GNN的每一层提取嵌入
                for i, conv in enumerate(model.convs):
                    x = conv(x, edge_index, edge_attr)
                    x = model.batch_norms[i](x)
                    x = torch.nn.functional.relu(x)
                    embeddings.append(x.cpu().numpy())
                
                node_embeddings.append(embeddings[-1])  # 使用最后一层的嵌入
        
        return model, node_embeddings
    
    def train_combined_gnn_lstm_model(self, X_scaled, y, sequence_length=10,
                                     window_size=20, step_size=10, threshold=0.3,
                                     feature_type='connection_profile',
                                     gnn_hidden_dim=64, lstm_hidden_dim=128):
        """
        训练结合GNN和LSTM的混合模型
        
        参数:
            X_scaled: 标准化后的神经元活动数据
            y: 标签
            sequence_length: 序列长度
            window_size: 滑动窗口大小
            step_size: 滑动步长
            threshold: 相关性阈值
            feature_type: 节点特征类型
            gnn_hidden_dim: GNN隐藏层维度
            lstm_hidden_dim: LSTM隐藏层维度
            
        返回:
            model: 训练好的混合模型
            history: 训练历史
        """
        print("\n训练GNN-LSTM混合模型...")
        
        # 创建滑动窗口相关矩阵
        correlation_matrices, time_indices = self.create_windowed_correlation_matrices(
            X_scaled, window_size, step_size
        )
        
        # 对齐标签
        aligned_labels = []
        for t in time_indices:
            if t < len(y):
                aligned_labels.append(y[t])
            else:
                # 对于超出范围的时间，使用最后一个可用标签
                aligned_labels.append(y[-1])
        
        # 准备序列数据
        sequences = []
        sequence_labels = []
        
        for i in range(len(correlation_matrices) - sequence_length + 1):
            seq_matrices = correlation_matrices[i:i+sequence_length]
            seq_label = aligned_labels[i+sequence_length-1]
            
            # 创建图序列
            graph_sequence = create_dynamic_graph_data(
                seq_matrices, sequence_length, threshold, feature_type
            )
            
            sequences.append(graph_sequence)
            sequence_labels.append(seq_label)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, sequence_labels,
            test_size=0.2,
            random_state=self.config.random_seed
        )
        
        # 获取输入维度和类别数
        input_dim = correlation_matrices[0].shape[0]  # 节点数
        num_classes = len(np.unique(sequence_labels))
        
        # 创建混合模型
        model = NeuronGNNLSTM(
            input_size=input_dim,
            gnn_hidden_size=gnn_hidden_dim,
            lstm_hidden_size=lstm_hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            message_type='node_edge_concat',
            pooling='mean'
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 训练循环
        num_epochs = 50
        batch_size = 8
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            # 随机打乱训练数据
            indices = np.random.permutation(len(X_train))
            
            # 按批次训练
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < 2:  # 跳过太小的批次
                    continue
                    
                batch_sequences = [X_train[idx] for idx in batch_indices]
                batch_labels = torch.tensor([y_train[idx] for idx in batch_indices], 
                                          dtype=torch.long).to(self.device)
                
                # 创建时间序列数据（实际上只是一个占位符，因为我们使用图序列）
                time_series = torch.zeros(len(batch_indices), sequence_length, input_dim).to(self.device)
                
                # 前向传播
                outputs, _ = model(time_series, batch_sequences)
                
                # 计算损失
                loss = criterion(outputs, batch_labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累计损失和准确率
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_labels).sum().item()
            
            # 验证
            model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_sequences = X_val[i:i+min(batch_size, len(X_val)-i)]
                    if len(batch_sequences) < 2:  # 跳过太小的批次
                        continue
                        
                    batch_labels = torch.tensor([y_val[idx] for idx in range(i, i+min(batch_size, len(X_val)-i))], 
                                              dtype=torch.long).to(self.device)
                    
                    # 创建时间序列数据
                    time_series = torch.zeros(len(batch_sequences), sequence_length, input_dim).to(self.device)
                    
                    # 前向传播
                    outputs, _ = model(time_series, batch_sequences)
                    
                    # 计算损失
                    loss = criterion(outputs, batch_labels)
                    
                    # 累计损失和准确率
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # 计算本轮的平均损失和准确率
            train_loss = train_loss / (len(X_train) // batch_size)
            val_loss = val_loss / (len(X_val) // batch_size)
            train_acc = train_correct / len(X_train)
            val_acc = val_correct / len(X_val)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(self.config.gnn_output_dir, 'gnn_lstm_model.pth'))
        
        return model, history
    
    def visualize_embeddings(self, node_embeddings, labels=None, method='tsne'):
        """
        可视化节点嵌入
        
        参数:
            node_embeddings: 节点嵌入列表
            labels: 标签
            method: 降维方法，'tsne'或'pca'
        """
        print("\n可视化节点嵌入...")
        
        # 将所有嵌入合并为一个数组
        all_embeddings = np.vstack(node_embeddings)
        
        # 使用t-SNE或PCA进行降维
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=self.config.random_seed)
        else:  # pca
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=self.config.random_seed)
        
        # 降维
        embeddings_2d = reducer.fit_transform(all_embeddings)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # 确保标签长度与嵌入数量匹配
            if len(labels) >= len(node_embeddings):
                plot_labels = labels[:len(node_embeddings)]
            else:
                # 如果标签不够长，使用最后一个标签填充
                plot_labels = np.concatenate([labels, np.array([labels[-1]] * (len(node_embeddings) - len(labels)))])
            
            # 为每个类别分配不同颜色
            unique_labels = np.unique(plot_labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                idx = plot_labels == label
                plt.scatter(
                    embeddings_2d[idx, 0],
                    embeddings_2d[idx, 1],
                    c=[colors[i]],
                    label=f'类别 {label}',
                    alpha=0.7
                )
            
            plt.legend()
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        plt.title(f'节点嵌入的{method.upper()}可视化')
        plt.xlabel('维度1')
        plt.ylabel('维度2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.gnn_output_dir, f'node_embeddings_{method}.png'), dpi=300)
        plt.close()
    
    def analyze_community_structure(self, G, correlation_matrix, available_neurons, plot=True):
        """
        分析神经元网络的社区结构
        
        参数:
            G: NetworkX图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
            plot: 是否绘制可视化图
            
        返回:
            communities: 社区列表
            metrics: 社区指标
        """
        print("\n分析神经元网络社区结构...")
        
        # 检测社区
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
            
            # 将communities转换为列表
            communities = [nodes for community_id, nodes in communities.items()]
        except:
            # 如果community库不可用，使用NetworkX内置方法
            try:
                communities = list(nx.algorithms.community.greedy_modularity_communities(G))
            except:
                print("警告: 社区检测失败")
                return [], {}
        
        # 计算社区指标
        metrics = {
            'modularity': nx.algorithms.community.modularity(G, communities),
            'num_communities': len(communities),
            'community_sizes': [len(c) for c in communities],
            'avg_community_size': sum(len(c) for c in communities) / len(communities)
        }
        
        print(f"检测到 {metrics['num_communities']} 个社区")
        print(f"模块度: {metrics['modularity']:.4f}")
        print(f"平均社区大小: {metrics['avg_community_size']:.2f}")
        
        # 可视化
        if plot:
            plt.figure(figsize=(12, 10))
            
            # 为每个社区分配颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
            color_map = {}
            
            for i, community in enumerate(communities):
                for node in community:
                    color_map[node] = colors[i]
            
            # 节点颜色列表
            node_colors = [color_map.get(node, 'gray') for node in G.nodes()]
            
            # 获取节点度作为节点大小
            node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
            
            # 获取边权重作为边宽度
            edge_widths = [2 * G[u][v]['weight'] for u, v in G.edges()]
            
            # 使用spring_layout布局
            pos = nx.spring_layout(G, seed=self.config.random_seed)
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
            
            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title('神经元网络社区结构')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.gnn_output_dir, 'neuron_communities.png'), dpi=300)
            plt.close()
        
        return communities, metrics
    
    def run_full_analysis(self, X_scaled, y=None):
        """
        运行完整的GNN-LSTM分析流程
        
        参数:
            X_scaled: 标准化后的神经元活动数据
            y: 标签
            
        返回:
            results: 分析结果
        """
        print("\n运行完整的GNN-LSTM集成分析...")
        
        # 创建神经元功能连接图
        G, correlation_matrix, available_neurons = self.create_neuron_graph(X_scaled)
        
        # 分析社区结构
        communities, community_metrics = self.analyze_community_structure(G, correlation_matrix, available_neurons)
        
        # 创建滑动窗口相关矩阵
        correlation_matrices, time_indices = self.create_windowed_correlation_matrices(X_scaled)
        
        # 准备GNN数据
        graph_data_list = self.prepare_graph_data(correlation_matrices)
        
        # 对齐标签
        if y is not None:
            aligned_labels = []
            for t in time_indices:
                if t < len(y):
                    aligned_labels.append(y[t])
                else:
                    # 对于超出范围的时间点，使用最后一个可用标签
                    aligned_labels.append(y[-1])
        else:
            # 如果没有提供标签，使用0作为默认值
            aligned_labels = np.zeros(len(time_indices), dtype=int)
        
        # 训练独立的GNN模型
        feature_dim = correlation_matrices[0].shape[0]  # 节点数等于相关矩阵的维度
        gnn_model, node_embeddings = self.train_gnn_model(
            graph_data_list, aligned_labels, feature_dim
        )
        
        # 可视化节点嵌入
        self.visualize_embeddings(node_embeddings, aligned_labels)
        
        # 如果有标签，训练GNN-LSTM混合模型
        if y is not None and len(np.unique(y)) > 1:
            hybrid_model, history = self.train_combined_gnn_lstm_model(
                X_scaled, y
            )
            
            # 可视化训练历史
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='训练')
            plt.plot(history['val_loss'], label='验证')
            plt.title('模型损失')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='训练')
            plt.plot(history['val_acc'], label='验证')
            plt.title('模型准确率')
            plt.xlabel('轮次')
            plt.ylabel('准确率')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.gnn_output_dir, 'training_history.png'), dpi=300)
            plt.close()
        else:
            hybrid_model = None
            history = None
        
        # 返回结果
        results = {
            'G': G,
            'correlation_matrix': correlation_matrix,
            'available_neurons': available_neurons,
            'communities': communities,
            'community_metrics': community_metrics,
            'correlation_matrices': correlation_matrices,
            'time_indices': time_indices,
            'graph_data_list': graph_data_list,
            'gnn_model': gnn_model,
            'node_embeddings': node_embeddings,
            'hybrid_model': hybrid_model,
            'training_history': history
        }
        
        print("GNN-LSTM集成分析完成!")
        return results


def main():
    """主函数"""
    from analysis_config import AnalysisConfig
    
    # 创建配置
    config = AnalysisConfig()
    config.gnn_output_dir = os.path.join(config.output_dir, 'gnn_analysis')
    config.random_seed = 42
    
    # 创建集成器
    integrator = GNNLSTMIntegrator(config)
    
    # 加载数据
    processor = NeuronDataProcessor(config)
    X_scaled, y, behavior_labels = processor.preprocess_data()
    
    # 运行分析
    results = integrator.run_full_analysis(X_scaled, y)
    
    print("GNN-LSTM集成分析完成!")


if __name__ == "__main__":
    main() 