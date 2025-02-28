"""
交互式神经元网络可视化脚本

此脚本用于生成交互式的神经元网络可视化图，
使得用户可以通过鼠标悬停在神经元上来高亮显示其连接。
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np

from analysis_config import AnalysisConfig
from visualization import VisualizationManager

def load_network_analysis_results(file_path):
    """
    加载网络分析结果JSON文件
    
    参数:
        file_path: 结果文件路径
        
    返回:
        analysis_results: 分析结果字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载网络分析结果失败: {str(e)}")
        return None

def reconstruct_network_from_results(results):
    """
    从分析结果重构NetworkX图
    
    参数:
        results: 网络分析结果字典
        
    返回:
        G: 重构的NetworkX图对象
        modules: 功能模块信息
    """
    # 创建新的图对象
    G = nx.Graph()
    
    # 获取功能模块
    modules = results.get('functional_modules', {})
    
    # 从各模块添加节点
    for module_name, module_data in modules.items():
        for neuron in module_data.get('neurons', []):
            G.add_node(neuron)
    
    # 确保所有节点都被添加（即使不在任何模块中）
    if 'graph' in results.get('topology_metrics', {}):
        for node in results['topology_metrics']['graph'].get('nodes', []):
            if not G.has_node(node):
                G.add_node(node)
    
    # 添加边 - 使用正确的路径获取边信息
    if 'graph' in results.get('topology_metrics', {}) and 'edges' in results['topology_metrics']['graph']:
        print(f"使用实际连接数据添加边...")
        # 从实际连接数据添加边
        for edge in results['topology_metrics']['graph']['edges']:
            if len(edge) == 3:  # 确保边数据格式正确 [source, target, weight]
                G.add_edge(edge[0], edge[1], weight=edge[2])
    else:
        print("警告: 未找到实际连接数据，使用度中心性估计连接")
        # 使用度中心性信息添加边（基于边的权重估计）
        degree_centrality = results.get('topology_metrics', {}).get('degree_centrality', {})
        nodes = list(G.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # 假设度中心性较高的节点之间更可能相连
                if node1 in degree_centrality and node2 in degree_centrality:
                    weight = (degree_centrality[node1] + degree_centrality[node2]) / 2
                    if weight > 0.1:  # 设置阈值
                        G.add_edge(node1, node2, weight=weight)
    
    return G, modules

def create_interactive_visualization(G, modules, output_path):
    """
    创建交互式的神经元网络可视化
    
    参数:
        G: NetworkX图对象
        modules: 功能模块信息
        output_path: 输出HTML文件路径
    """
    # 创建pyvis网络对象
    net = Network(height="800px", width="100%", directed=False, notebook=False)
    
    # 使用spring布局而不是force_atlas_2based，更接近于静态图的布局
    # 调整参数使节点分布更均匀，减少重叠
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=200, spring_strength=0.08, damping=0.09)
    
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
          "iterations": 200,
          "fit": true
        },
        "barnesHut": {
          "gravitationalConstant": -80000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.09,
          "avoidOverlap": 0.5
        }
      },
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "size": 15,
        "color": {
          "border": "#000000",
          "highlight": {
            "border": "#000000",
            "background": "#ffffff"
          }
        },
        "font": {
          "size": 12,
          "face": "Arial",
          "bold": true,
          "color": "#000000"
        },
        "shadow": false
      },
      "edges": {
        "color": {
          "inherit": true,
          "opacity": 0.6
        },
        "smooth": false,
        "width": 0.2,
        "shadow": false,
        "selectionWidth": 1,
        "hoverWidth": 0.5
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
    for i, (module_name, module_data) in enumerate(modules.items()):
        r, g, b, a = colors[i]
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        for neuron in module_data.get('neurons', []):
            color_map[neuron] = hex_color
    
    # 为节点分配ID
    node_id_map = {node: i+1 for i, node in enumerate(G.nodes())}
    
    # 添加节点 - 使用合适的节点大小
    for node in G.nodes():
        node_id = str(node_id_map[node]).replace('n', '')  # 移除节点名称中的'n'前缀
        node_color = color_map.get(node, "#CCCCCC")  # 如果没有颜色映射则使用灰色
        net.add_node(node, label=node_id, color=node_color, title=f"神经元 {node_id}", size=15)
    
    # 添加边 - 使用固定的极细线条
    for u, v, data in G.edges(data=True):
        # 获取边的权重
        weight = data.get('weight', 1.0)
        # 使用固定的极细线条宽度
        width = 0.2
        # 仅根据权重调整透明度，保持线条粗细一致
        opacity = min(0.9, max(0.5, weight * 0.6))
        
        net.add_edge(u, v, value=weight, width=width, title=f"相关性: {weight:.3f}", 
                    color={'opacity': opacity, 'inherit': True})
    
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
            <p><b>节点编号:</b> 对应原始数据中的神经元编号</p>
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

def main():
    """主函数"""
    # 初始化配置
    config = AnalysisConfig()
    config.setup_directories()
    
    # 结果文件路径
    results_path = os.path.join(config.analysis_dir, 'network_analysis_results.json')
    
    # 检查结果文件是否存在
    if not os.path.exists(results_path):
        print(f"错误: 网络分析结果文件不存在: {results_path}")
        print("请先运行analysis_results.py以生成网络分析结果")
        return
    
    print(f"加载网络分析结果: {results_path}")
    results = load_network_analysis_results(results_path)
    
    if not results:
        print("错误: 无法加载网络分析结果")
        return
    
    print("重构神经元网络...")
    G, modules = reconstruct_network_from_results(results)
    
    print(f"重构的网络包含 {len(G.nodes())} 个节点和 {len(G.edges())} 条边")
    
    # 创建输出目录
    interactive_dir = os.path.join(config.analysis_dir, 'interactive')
    os.makedirs(interactive_dir, exist_ok=True)
    
    # 输出文件路径
    output_path = os.path.join(interactive_dir, 'interactive_neuron_network.html')
    
    print("创建交互式可视化...")
    create_interactive_visualization(G, modules, output_path)
    
    print("\n完成! 交互式神经元网络可视化已创建。")
    print(f"请在浏览器中打开以下文件查看交互式图表: {output_path}")
    print("使用说明:")
    print("- 将鼠标悬停在神经元上可高亮显示其连接")
    print("- 点击神经元可固定选择，再次点击可取消选择")
    print("- 使用鼠标滚轮进行缩放")
    print("- 拖动可移动整个网络")

if __name__ == "__main__":
    main() 