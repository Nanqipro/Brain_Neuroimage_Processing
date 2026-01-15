import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter

def analyze_neural_behavior_markov(
    trace_file,
    cluster_file,
    trace_sheet=None,
    n_network_states=6,
    auto_select_network_states=False,
    network_states_range=(2, 10),
    behavior_lag_frames=0,
    edge_threshold=0.1,
    results_dir='../results',
    random_state=42,
):
    """
    基于钙信号Trace和钙爆聚类结果构建神经-行为马尔可夫模型
    """
    print("--- 1. 数据加载与预处理 ---")
    # 加载数据
    trace_file_str = str(trace_file)
    trace_file_lower = trace_file_str.lower()
    if trace_file_lower.endswith(('.xlsx', '.xls')):
        if trace_sheet is None:
            df_trace = pd.read_excel(trace_file)
        else:
            df_trace = pd.read_excel(trace_file, sheet_name=trace_sheet)
    else:
        df_trace = pd.read_csv(trace_file)
    df_clusters = pd.read_csv(cluster_file)
    
    if 'source_file' in df_clusters.columns:
        trace_base = os.path.basename(str(trace_file)).strip()
        trace_base_lower = trace_base.lower()

        source_files = (
            df_clusters['source_file']
            .dropna()
            .astype(str)
            .map(lambda x: os.path.basename(x).strip())
            .unique()
            .tolist()
        )

        matched_bases = [
            sf for sf in source_files
            if sf and (sf.lower() in trace_base_lower or trace_base_lower in sf.lower())
        ]

        if matched_bases:
            best_base = max(matched_bases, key=len)
            best_base_lower = best_base.lower()
            before_rows = len(df_clusters)
            df_clusters = df_clusters[
                df_clusters['source_file']
                .astype(str)
                .map(lambda x: os.path.basename(x).strip().lower())
                .eq(best_base_lower)
            ].copy()
            after_rows = len(df_clusters)
            print(f"source_file 匹配到: {best_base}，过滤行数: {before_rows} -> {after_rows}")
        else:
            print(f"未在 cluster 表的 source_file 中找到与 {trace_base} 匹配的记录，继续使用全表数据。")
    else:
        print("cluster 表缺少 source_file 列，继续使用全表数据。")

    # 获取总帧数
    n_frames = len(df_trace)
    print(f"Trace总帧数: {n_frames}")
    
    # 获取所有唯一的Cluster标签 (假设 label 是 'Cluster 1', 'Cluster 2' 等)
    unique_clusters = sorted(df_clusters['cluster_label'].unique())
    cluster_map = {label: i+1 for i, label in enumerate(unique_clusters)} # 0 保留给静息态
    print(f"检测到的钙爆状态类型: {unique_clusters}")

    # --- 2. 构建逐帧的神经元状态矩阵 ---
    # 创建一个矩阵 [Frames x Neurons]，记录每一帧每个神经元处于哪个Cluster状态
    # 首先识别Trace文件中的神经元列 (以'n'开头的列)
    neuron_cols = [c for c in df_trace.columns if c.startswith('n') and c[1:].isdigit()]
    n_neurons = len(neuron_cols)
    print(f"神经元数量: {n_neurons}")
    
    # 初始化状态矩阵 (默认为0，即静息/无钙爆)
    # neural_state_matrix[t, i] 表示 t时刻 第i个神经元的 Cluster ID
    neural_state_matrix = np.zeros((n_frames, n_neurons), dtype=int)
    
    # 填充矩阵
    # 我们使用 cluster_file 中的 start_idx 和 end_idx 来填充
    # 注意：需要确保 trace 文件的索引和 cluster 文件的 idx 是对齐的
    
    neuron_col_to_idx = {name: i for i, name in enumerate(neuron_cols)}
    
    for _, row in df_clusters.iterrows():
        n_name = row['neuron']
        if n_name in neuron_col_to_idx:
            n_idx = neuron_col_to_idx[n_name]
            start = int(row['start_idx'])
            end = int(row['end_idx'])
            c_label = row['cluster_label']
            c_code = cluster_map.get(c_label, 0)
            
            # 边界检查
            start = max(0, start)
            end = min(n_frames, end)
            
            if start < end:
                neural_state_matrix[start:end, n_idx] = c_code

    print("神经元状态矩阵构建完成。")

    active_codes = list(cluster_map.values())
    X = np.column_stack([(neural_state_matrix == code).sum(axis=1) for code in active_codes])
    
    # 使用 K-Means 将连续的时间帧聚类为 K 个“网络模态” (Network States)
    # 这里的 K 可以根据经验设定，比如设定为 5-8 个典型的网络放电模式
    if auto_select_network_states:
        k_min, k_max = network_states_range
        k_min = max(2, int(k_min))
        k_max = max(k_min, int(k_max))
        best_k = None
        best_score = -1.0
        X_scaled_tmp = StandardScaler().fit_transform(X)
        for k in range(k_min, k_max + 1):
            kmeans_tmp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels_tmp = kmeans_tmp.fit_predict(X_scaled_tmp)
            if len(set(labels_tmp)) < 2:
                continue
            score = silhouette_score(X_scaled_tmp, labels_tmp)
            if score > best_score:
                best_score = score
                best_k = k
        if best_k is not None:
            n_network_states = best_k
            print(f"自动选择网络模态数: {n_network_states} (silhouette={best_score:.3f})")
        else:
            print(f"自动选择网络模态数失败，使用默认值: {n_network_states}")

    print(f"正在将所有时间帧聚类为 {n_network_states} 种网络模态...")
    
    # 标准化数据有助于聚类
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_network_states, random_state=random_state, n_init=10)
    network_state_labels = kmeans.fit_predict(X_scaled)
    
    # 将网络状态添加回 Trace 数据
    df_trace['Network_State'] = network_state_labels
    
    # --- 4. 关联行为数据 (Emission Analysis) ---
    # 分析每个网络模态下，主要出现什么行为
    
    behavior_lag_frames = int(behavior_lag_frames)
    if behavior_lag_frames < 0:
        raise ValueError("behavior_lag_frames must be >= 0")

    if behavior_lag_frames == 0:
        df_clean = df_trace.dropna(subset=['behavior'])
        behavior_series = df_clean['behavior']
        network_state_series = df_clean['Network_State']
    else:
        behavior_future = df_trace['behavior'].shift(-behavior_lag_frames)
        df_aligned = pd.DataFrame({
            'Network_State': df_trace['Network_State'],
            'Behavior_Future': behavior_future,
        }).dropna(subset=['Behavior_Future'])
        df_clean = df_aligned
        behavior_series = df_clean['Behavior_Future']
        network_state_series = df_clean['Network_State']
    
    # 计算联合概率分布 P(Behavior | Network_State)
    confusion = pd.crosstab(network_state_series, behavior_series)
    # 归一化：每个网络状态下，各种行为的概率
    emission_prob = confusion.div(confusion.sum(axis=1), axis=0)
    
    # 找出每个网络状态的主导行为 (Dominant Behavior)
    state_dominant_behavior = emission_prob.idxmax(axis=1)
    
    print("\n--- 网络模态与行为的对应关系 ---")
    print(emission_prob)

    # --- 5. 构建马尔可夫转移矩阵 (Transition Matrix) ---
    # 计算从 Network_State i -> Network_State j 的概率
    
    transitions = np.zeros((n_network_states, n_network_states))
    
    # 遍历时间序列
    for t in range(len(network_state_labels) - 1):
        current_s = network_state_labels[t]
        next_s = network_state_labels[t+1]
        transitions[current_s, next_s] += 1
        
    # 归一化行 (使得每行和为1)
    row_sums = transitions.sum(axis=1)
    transition_matrix = np.divide(transitions, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis]!=0)
    
    # --- 6. 可视化 ---
    plt.figure(figsize=(15, 10))
    
    # 6.1 绘制状态转移图
    G = nx.DiGraph()
    
    # 添加节点，节点名称包含主导行为
    node_labels = {}
    for i in range(n_network_states):
        dom_beh = state_dominant_behavior.get(i, "Unknown")
        label = f"State {i}\n({dom_beh})"
        G.add_node(i, label=label, behavior=dom_beh)
        node_labels[i] = label
    
    # 添加边 (只显示概率 > 阈值的边，避免图太乱)
    for i in range(n_network_states):
        for j in range(n_network_states):
            weight = transition_matrix[i, j]
            if weight > edge_threshold:
                # 自身环或者不同节点
                G.add_edge(i, j, weight=weight)
    
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    # 绘制节点
    # 根据行为给节点上色
    unique_behaviors = behavior_series.unique()
    color_map_beh = {beh: plt.cm.tab10(i) for i, beh in enumerate(unique_behaviors)}
    node_colors = [color_map_beh.get(G.nodes[n]['behavior'], 'gray') for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")
    
    # 绘制边
    edges = G.edges(data=True)
    weights = [d['weight'] * 5 for u, v, d in edges] # 线宽
    nx.draw_networkx_edges(G, pos, width=weights, arrowstyle='-|>', arrowsize=20, edge_color='gray', connectionstyle='arc3,rad=0.1')
    
    # 绘制边的标签 (概率)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges if d['weight'] > 0.2} # 只标注大概率
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Neural population state transition graph (Markov model)\nColor indicates dominant behavior", fontsize=15)
    plt.axis('off')
    
    # 创建图例
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=b, 
                          markerfacecolor=c, markersize=15) for b, c in color_map_beh.items()]
    plt.legend(handles=patches, title="Dominant Behavior", loc='best')
    
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    trace_base_for_output = os.path.splitext(os.path.basename(str(trace_file)))[0]
    if trace_sheet:
        trace_base_for_output = f"{trace_base_for_output}_{trace_sheet}"
    safe_prefix = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in trace_base_for_output)
    lag_suffix = "" if behavior_lag_frames == 0 else f"_lag{behavior_lag_frames}"
    plt.savefig(os.path.join(results_dir, f'{safe_prefix}_markov_model_plot{lag_suffix}.png'))
    
    # 6.2 绘制发射概率热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(emission_prob, annot=True, cmap="YlGnBu", fmt=".2f")
    if behavior_lag_frames == 0:
        heatmap_title = "Emission probability: network state vs behavior"
    else:
        heatmap_title = f"Predictive emission: P(behavior[t+{behavior_lag_frames}] | state[t])"
    plt.title(heatmap_title)
    plt.ylabel("Network State (Latent)")
    plt.xlabel("Behavior")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{safe_prefix}_emission_probability{lag_suffix}.png'))
    
    print("\n分析完成！图片已保存。")
    return transition_matrix, emission_prob

# 运行分析 (使用上传的文件路径)
# 请确保文件名与实际环境一致
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, '..', 'datasets')
    results_dir = os.path.join(script_dir, '..', 'results')

    trace_path = os.path.join(datasets_dir, 'EMtrace01.xlsx')
    trace_sheet = 'Sheet1'
    cluster_path = os.path.join(datasets_dir, 'amplitude_duration_clusters.csv')

    try:
        T_mat, E_mat = analyze_neural_behavior_markov(
            trace_path,
            cluster_path,
            trace_sheet=trace_sheet,
            results_dir=results_dir,
        )
    except FileNotFoundError:
        print("请确认文件路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")
