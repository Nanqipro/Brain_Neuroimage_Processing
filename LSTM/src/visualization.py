import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import os
from sklearn.decomposition import PCA
import json

class VisualizationManager:
    """可视化管理器类,用于生成各种数据分析的可视化图表"""
    def __init__(self, config):
        self.config = config
        # 设置 seaborn 样式
        sns.set_style("whitegrid")
        # 设置 matplotlib 的基本样式
        plt.style.use('default')
        # 设置全局字体大小
        plt.rcParams.update({
            'font.size': 20,  # 基础字体大小
            'axes.titlesize': 18,  # 标题字体大小
            'axes.labelsize': 18,  # 轴标签字体大小
            'xtick.labelsize': 18,  # x轴刻度标签字体大小
            'ytick.labelsize': 18,  # y轴刻度标签字体大小
            'legend.fontsize': 18,  # 图例字体大小
            'figure.titlesize': 18  # 图形标题字体大小
        })
        
    def set_plot_style(self):
        """设置全局绘图样式"""
        plt.rcParams['lines.linewidth'] = self.config.visualization_params['line_width']
        
    def plot_behavior_neuron_correlation(self, behavior_activity_df):
        """绘制行为和神经元之间的相关性热图"""
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['correlation'])
        sns.heatmap(behavior_activity_df, 
                   cmap=self.config.visualization_params['colormaps']['correlation'],
                   center=0,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Mean Neuron Activity by Behavior')
        plt.xlabel('Neurons')
        plt.ylabel('Behaviors')
        plt.tight_layout()
        plt.savefig(self.config.correlation_plot, 
                   dpi=self.config.visualization_params['dpi'])
        plt.close()
    
    def plot_temporal_patterns(self, X_scaled, y, behavior_labels):
        """绘制每种行为的时间模式图
        
        参数:
            X_scaled: 标准化后的神经元数据
            y: 行为标签
            behavior_labels: 行为标签列表
        """
        window_size = self.config.analysis_params['temporal_window_size']
        
        for behavior_idx, behavior in enumerate(behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_data = X_scaled[behavior_mask]
            
            if len(behavior_data) > window_size:
                # 计算移动平均
                rolling_mean = np.array([
                    np.mean(behavior_data[i:i+window_size], axis=0)
                    for i in range(0, len(behavior_data)-window_size, window_size)
                ])
                
                plt.figure(figsize=self.config.visualization_params['figure_sizes']['temporal'])
                for neuron in range(min(5, rolling_mean.shape[1])):
                    plt.plot(rolling_mean[:, neuron], 
                            label=f'Neuron {neuron+1}',
                            linewidth=self.config.visualization_params['line_width'])
                
                plt.title(f'Temporal Pattern of Neuron Activity During {behavior}')
                plt.xlabel('Time Windows')
                plt.ylabel('Standardized Activity')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.config.get_temporal_pattern_path(behavior),
                          dpi=self.config.visualization_params['dpi'])
                plt.close()
    
    def plot_behavior_transitions(self, transitions, behavior_labels):
        """绘制行为转换矩阵热图
        
        参数:
            transitions: 行为转换概率矩阵
            behavior_labels: 行为标签列表
        """
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['transitions'])
        sns.heatmap(transitions,
                   xticklabels=behavior_labels,
                   yticklabels=behavior_labels,
                   cmap=self.config.visualization_params['colormaps']['transitions'],
                   annot=False,
                   cbar_kws={'label': 'Transition Probability'})
        plt.title('Behavior Transition Probabilities')
        plt.xlabel('To Behavior')
        plt.ylabel('From Behavior')
        plt.tight_layout()
        plt.savefig(self.config.transition_plot,
                   dpi=self.config.visualization_params['dpi'])
        plt.close()
    def plot_neuron_network(self, behavior_importance):
        """绘制行为-神经元网络关系图
        
        参数:
            behavior_importance: 包含行为重要性数据的字典
        """
        G = nx.Graph()
        
        # 添加节点和边
        neuron_counter = 1
        neuron_id_map = {}  # 用于跟踪神经元编号
        
        for behavior, data in behavior_importance.items():
            G.add_node(behavior, node_type='behavior')
            for neuron, effect in zip(data['significant_neurons'], data['effect_sizes'][data['significant_neurons']]):
                neuron_name = f'N{neuron+1}'
                if neuron_name not in neuron_id_map:
                    neuron_id_map[neuron_name] = neuron_counter
                    neuron_counter += 1
                G.add_node(neuron_name, node_type='neuron', id=neuron_id_map[neuron_name])
                if effect > self.config.analysis_params['neuron_significance_threshold']:
                    G.add_edge(behavior, neuron_name, weight=effect)
        
        # 创建布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制图形
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['network'])
        
        # 绘制节点
        behavior_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'behavior']
        neuron_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'neuron']
        
        nx.draw_networkx_nodes(G, pos, nodelist=behavior_nodes, 
                             node_color='lightblue', node_size=2000, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=neuron_nodes,
                             node_color='lightgreen', node_size=1500, alpha=0.7)
        
        # 根据权重绘制边
        edges = G.edges(data=True)
        weights = [d['weight'] for (u, v, d) in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
        
        # 添加标签 - 行为节点使用原始标签，神经元节点使用数字ID
        behavior_labels = {node: node for node in behavior_nodes}
        neuron_labels = {node: str(G.nodes[node]['id']) for node in neuron_nodes}
        
        # 绘制行为标签
        nx.draw_networkx_labels(G, pos, labels=behavior_labels, font_size=20)
        # 绘制神经元ID标签
        nx.draw_networkx_labels(G, pos, labels=neuron_labels, font_size=16, font_color='black', font_weight='bold')
        
        plt.title('Behavior-Neuron Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.config.network_plot,
                   dpi=self.config.visualization_params['dpi'])
        plt.close()
    
    def plot_temporal_correlations(self, correlations):
        """绘制不同时间窗口大小的时间相关性图
        
        参数:
            correlations: 包含不同窗口大小相关性值的字典
        """
        for window, corr_values in correlations.items():
            plt.figure(figsize=self.config.visualization_params['figure_sizes']['temporal'])
            plt.plot(corr_values, 
                    linewidth=self.config.visualization_params['line_width'])
            plt.title(f'Temporal Correlations (Window Size: {window})')
            plt.xlabel('Time')
            plt.ylabel('Mean Correlation')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.config.get_temporal_correlation_path(window),
                       dpi=self.config.visualization_params['dpi'])
            plt.close()
    
    def plot_statistical_summary(self, f_values, p_values, effect_sizes):
        """绘制统计分析总结图
        
        参数:
            f_values: F检验值列表
            p_values: P值列表
            effect_sizes: 效应量字典
        """
        # 创建总结图
        plt.figure(figsize=(15, 10))
        
        # 图1: F值分布
        plt.subplot(2, 2, 1)
        plt.hist(f_values, bins=30)
        plt.title('Distribution of F-values')
        plt.xlabel('F-value')
        plt.ylabel('Count')
        
        # 图2: P值分布
        plt.subplot(2, 2, 2)
        plt.hist(p_values, bins=30)
        plt.title('Distribution of P-values')
        plt.xlabel('P-value')
        plt.ylabel('Count')
        
        # 图3: 各行为的效应量
        plt.subplot(2, 2, 3)
        behaviors = list(effect_sizes.keys())
        mean_effects = [np.mean(effect_sizes[b]['effect_sizes']) for b in behaviors]
        plt.bar(behaviors, mean_effects)
        plt.xticks(rotation=45)
        plt.title('Mean Effect Size by Behavior')
        plt.xlabel('Behavior')
        plt.ylabel('Mean Effect Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'statistical_summary.png'),
                   dpi=self.config.visualization_params['dpi'])
        plt.close()
    
    def plot_significant_neurons_effect_sizes(self, sorted_neurons, sorted_effects):
        """绘制显著性神经元的效应量柱状图
        
        参数:
            sorted_neurons: 按效应量排序的神经元编号列表
            sorted_effects: 对应的效应量列表
        """
        # 创建图形和轴对象
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 创建柱状图
        bars = ax.bar(range(len(sorted_neurons)), sorted_effects)
        
        # 设置柱状图颜色，根据效应量大小渐变
        norm = plt.Normalize(min(sorted_effects), max(sorted_effects))
        colors = plt.cm.viridis(norm(sorted_effects))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 添加数值标签
        for i, v in enumerate(sorted_effects):
            ax.text(i, v, f'N{sorted_neurons[i]}', 
                   ha='center', va='bottom', fontsize=8)
        
        # 设置标题和标签
        ax.set_title('Distribution of Mean Effect Sizes for Significant Neurons')
        ax.set_xlabel('Neuron Ranking')
        ax.set_ylabel('Mean Effect Size')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])  # 隐藏x轴刻度，因为我们已经在柱子上标注了神经元编号
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        plt.colorbar(sm, ax=ax, label='Effect Size Magnitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'significant_neurons_effect_sizes.png'),
                   dpi=self.config.visualization_params['dpi'])
        plt.close()

    def plot_attention_weights(self, attention_weights, temporal_weights, save_path):
        """
        可视化注意力权重
        参数:
            attention_weights: 多头注意力权重 [batch_size, num_heads, seq_len, seq_len]
            temporal_weights: 时间注意力权重 [batch_size, seq_len]
            save_path: 保存路径
        """
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['attention'])
        
        # 1. 多头注意力权重可视化
        plt.subplot(2, 1, 1)
        avg_attention = attention_weights.mean(axis=(0, 1))  # 平均所有批次和头
        sns.heatmap(avg_attention, 
                   cmap=self.config.visualization_params['attention_plot_style']['cmap'],
                   xticklabels=5, 
                   yticklabels=5)
        plt.title('Multi-head Attention Weights')
        plt.xlabel('Time Step')
        plt.ylabel('Time Step')
        
        # 2. 时间注意力权重可视化
        plt.subplot(2, 1, 2)
        avg_temporal = temporal_weights.mean(axis=0)
        plt.plot(avg_temporal, linewidth=2)
        plt.title('Temporal Attention Weights')
        plt.xlabel('Time Step')
        plt.ylabel('Weight')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.visualization_params['dpi'])
        plt.close()

    def plot_autoencoder_reconstruction(self, original, reconstructed, sample_indices, save_path):
        """
        可视化自编码器的重构结果
        参数:
            original: 原始数据
            reconstructed: 重构数据
            sample_indices: 要可视化的样本索引
            save_path: 保存路径
        """
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['autoencoder'])
        
        n_samples = len(sample_indices)
        for i, idx in enumerate(sample_indices):
            plt.subplot(n_samples, 1, i+1)
            plt.plot(original[idx], label='Original', alpha=0.7)
            plt.plot(reconstructed[idx], label='Reconstructed', alpha=0.7)
            plt.title(f'Sample {idx}')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.visualization_params['dpi'])
        plt.close()

    def plot_neuron_embeddings(self, encoded_features, behavior_labels, save_path):
        """
        可视化神经元在潜在空间中的分布
        参数:
            encoded_features: 编码后的特征 [n_samples, latent_dim]
            behavior_labels: 行为标签
            save_path: 保存路径
        """
        # 使用PCA降维到2D进行可视化
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(encoded_features)
        
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['network'])
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=behavior_labels, 
                            cmap='tab10',
                            alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Neuron Embeddings in Latent Space')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.visualization_params['dpi'])
        plt.close()

    def plot_training_metrics(self, metrics):
        """
        绘制增强版训练指标图表
        参数:
            metrics: 包含训练指标的字典
        """
        plt.figure(figsize=(20, 5))
        
        # 1. 分类损失和准确率
        plt.subplot(1, 3, 1)
        plt.plot(metrics['train_losses'], label='Training Loss', linewidth=2)
        plt.plot(metrics['val_losses'], label='Validation Loss', linewidth=2)
        plt.title('Classification Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 2. 准确率
        plt.subplot(1, 3, 2)
        plt.plot(metrics['train_accuracies'], label='Training Accuracy', linewidth=2)
        plt.plot(metrics['val_accuracies'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # 3. 重构损失
        plt.subplot(1, 3, 3)
        plt.plot(metrics['reconstruction_losses'], label='Reconstruction Loss', linewidth=2)
        plt.title('Autoencoder Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.config.accuracy_plot, dpi=self.config.visualization_params['dpi'])
        plt.close()

    def plot_attention_analysis(self, attention_weights, behavior_labels, save_path):
        """
        分析不同行为下的注意力模式
        参数:
            attention_weights: 注意力权重
            behavior_labels: 行为标签
            save_path: 保存路径
        """
        unique_behaviors = np.unique(behavior_labels)
        n_behaviors = len(unique_behaviors)
        
        plt.figure(figsize=(20, 4*n_behaviors))
        
        for i, behavior in enumerate(unique_behaviors):
            mask = behavior_labels == behavior
            behavior_attention = attention_weights[mask].mean(axis=0)
            
            plt.subplot(n_behaviors, 1, i+1)
            sns.heatmap(behavior_attention, 
                       cmap=self.config.visualization_params['attention_plot_style']['cmap'],
                       xticklabels=5, 
                       yticklabels=5)
            plt.title(f'Attention Pattern for {behavior}')
            plt.xlabel('Time Step')
            plt.ylabel('Time Step')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.visualization_params['dpi'])
        plt.close()
        
    def plot_interactive_neuron_network(self, G, metrics=None, modules=None, output_path=None):
        """
        创建交互式神经元网络可视化
        
        参数:
            G: NetworkX图对象
            metrics: 拓扑指标字典
            modules: 功能模块字典
            output_path: 输出HTML文件路径（可选）
        
        返回:
            html_path: 保存的HTML文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.config.analysis_dir, 'interactive', 'interactive_neuron_network.html')
        
        try:
            from pyvis.network import Network
        except ImportError:
            print("警告: 未找到pyvis库，请安装: pip install pyvis")
            return None
            
        print("\n创建交互式神经元网络可视化...")
        
        # 创建输出目录
        interactive_dir = os.path.join(self.config.analysis_dir, 'interactive')
        os.makedirs(interactive_dir, exist_ok=True)
        
        # 创建pyvis网络对象
        net = Network(height="800px", width="100%", directed=False, notebook=False)
        
        # 使用spring布局而不是force_atlas_2based，更接近于静态图的布局
        # 调整参数使节点分布更均匀
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.05, damping=0.09)
        
        # 配置交互选项 - 直接设置各个选项而不是使用复杂的JSON结构
        net.width = "100%"
        net.height = "800px"
        net.bgcolor = "#ffffff"
        net.font_color = "black"
        
        # 使用字符串形式设置选项
        physics_options = """
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 200
            },
            "barnesHut": {
              "gravitationalConstant": -80000,
              "centralGravity": 0.3,
              "springLength": 250,
              "springConstant": 0.05,
              "damping": 0.09,
              "avoidOverlap": 0.5
            }
          },
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "size": 12,
            "font": {
              "size": 10,
              "face": "Arial",
              "bold": true
            },
            "shadow": false
          },
          "edges": {
            "color": {
              "inherit": true,
              "opacity": 0.15
            },
            "smooth": false,
            "width": 0.8,
            "shadow": false,
            "selectionWidth": 2,
            "hoverWidth": 1.5
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "multiselect": true,
            "tooltipDelay": 100
          },
          "layout": {
            "improvedLayout": true,
            "randomSeed": 42
          }
        }
        """
        
        # 设置选项
        net.set_options(physics_options)
            
        # 根据模块给节点着色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(modules)))
        color_map = {}
        
        # 将颜色转换为十六进制格式
        for i, (module_name, module) in enumerate(modules.items()):
            r, g, b, a = colors[i]
            hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            for neuron in module['neurons']:
                color_map[neuron] = hex_color
        
        # 为节点分配ID
        node_id_map = {node: i+1 for i, node in enumerate(G.nodes())}
        
        # 添加节点 - 减小节点大小
        for node in G.nodes():
            node_id = str(node_id_map[node])
            node_color = color_map.get(node, "#CCCCCC")  # 如果没有颜色映射则使用灰色
            net.add_node(node, label=node_id, color=node_color, title=f"神经元 {node_id}", size=12)
        
        # 添加边 - 减小边的宽度，降低透明度
        for u, v, data in G.edges(data=True):
            # 设置边的属性
            weight = data.get('weight', 1.0)
            width = weight * 0.8  # 减小边的宽度
            net.add_edge(u, v, value=weight, width=width, title=f"相关性: {weight:.3f}", color={'opacity': 0.15})
        
        # 使用自定义模板添加标题和CSS
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>神经元功能网络 - 交互式可视化</title>
            <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
            <style type="text/css">
                #mynetwork {
                    width: 100%;
                    height: 800px;
                    background-color: #ffffff;
                    border: 1px solid lightgray;
                    position: relative;
                    overflow: hidden;
                }
                
                .title-bar {
                    text-align: center;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-bottom: 1px solid #e9ecef;
                    margin-bottom: 20px;
                }
                
                .title-bar h1 {
                    margin: 0;
                    font-size: 24px;
                    font-family: Arial, sans-serif;
                    color: #333;
                }
                
                .controls {
                    margin: 10px 0;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                }
                
                .info {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    padding: 10px;
                    background-color: rgba(255, 255, 255, 0.8);
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    z-index: 10;
                }
                
                .vis-tooltip {
                    position: absolute;
                    padding: 5px;
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    background-color: rgba(255, 255, 255, 0.95);
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
            </style>
        </head>
        <body>
            <div class="title-bar">
                <h1>神经元功能网络 - 交互式可视化</h1>
            </div>
            <div class="controls">
                <p><b>使用说明:</b> 将鼠标悬停在神经元上可高亮显示其连接。点击神经元可固定选择。使用鼠标滚轮缩放，拖动可移动视图。</p>
            </div>
            <div class="info">
                <p><b>节点颜色:</b> 代表神经元的功能模块分组</p>
                <p><b>连接线:</b> 表示神经元之间的相关性强度</p>
            </div>
            <div id="mynetwork"></div>
            
            <script type="text/javascript">
        """
        
        # 保存为HTML文件，使用自定义模板
        with open(output_path, 'w', encoding='utf-8') as f:
            html = html_template
            html += net.generate_html()
            html += '\n</script>\n</body>\n</html>'
            f.write(html)
        
        print(f"交互式神经元网络已保存到: {output_path}")
        return output_path 