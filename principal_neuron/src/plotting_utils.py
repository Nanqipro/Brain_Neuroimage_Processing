import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict # 用于 plot_shared_neurons_map 中的图例处理

def plot_single_behavior_activity_map(key_neurons_df, behavior_name, behavior_color, title, output_path, show_title=True):
    """绘制单个行为的关键神经元空间分布图。

    参数:
        key_neurons_df (pd.DataFrame): 包含关键神经元数据的DataFrame，
                                       必须包含 'NeuronID', 'x', 'y' 列。
        behavior_name (str): 行为名称 (例如, 'Close')。
        behavior_color (str): 用于绘制这些神经元的颜色。
        title (str): 图表标题。
        output_path (str): 生成的图表图片的完整保存路径。
        show_title (bool): 是否显示图表标题，默认为 True。
    """
    if key_neurons_df.empty:
        print(f"行为 '{behavior_name}' 没有关键神经元可供绘制。跳过绘图: {output_path}")
        fig, ax = plt.subplots(figsize=(8, 6))
        # 注意: EFFECT_SIZE_THRESHOLD 未在此函数作用域内定义，文本提示可能需要从调用处传递或硬编码一个通用提示
        ax.text(0.5, 0.5, f'No key neurons for {behavior_name}\n(Check threshold setting)', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if show_title:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(output_path)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(key_neurons_df['x'], key_neurons_df['y'], 
               c=behavior_color, 
               s=150, 
               alpha=0.7, 
               edgecolors='black'
              )

    for i, txt in enumerate(key_neurons_df['NeuronID']):
        ax.annotate(str(txt), (key_neurons_df['x'].iloc[i], key_neurons_df['y'].iloc[i]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    if show_title:
        ax.set_title(title)
        
    ax.grid(True, linestyle='--', alpha=0.5) # 淡化网格线
    ax.set_facecolor('white') # 白色背景
    fig.patch.set_facecolor('white')

    ax.set_xlim(0, 1) # 假设相对坐标在0-1之间
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box') # 保持横纵轴等比例
    
    ax.set_xticks([]) # 移除X轴刻度
    ax.set_yticks([]) # 移除Y轴刻度

    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到 {output_path}")
    plt.close(fig) # 关闭图形以释放内存

def plot_shared_neurons_map(behavior1_name, behavior2_name, 
                            behavior1_all_key_neurons_df, behavior2_all_key_neurons_df, 
                            shared_key_neurons_df, 
                            color1, color2, mixed_color, 
                            title, output_path, 
                            scheme='B', show_title=True, alpha_non_shared=0.3, shared_marker_size_factor=1.5):
    """绘制两种行为间共享的关键神经元，或两种行为的所有关键神经元并高亮显示共享部分。

    参数:
        behavior1_name (str): 第一个行为的名称。
        behavior2_name (str): 第二个行为的名称。
        behavior1_all_key_neurons_df (pd.DataFrame): 包含行为1所有关键神经元数据 ('NeuronID', 'x', 'y') 的DataFrame。
        behavior2_all_key_neurons_df (pd.DataFrame): 包含行为2所有关键神经元数据 ('NeuronID', 'x', 'y') 的DataFrame。
        shared_key_neurons_df (pd.DataFrame): 仅包含共享关键神经元数据 ('NeuronID', 'x', 'y') 的DataFrame。
        color1 (str): 行为1神经元的颜色。
        color2 (str): 行为2神经元的颜色。
        mixed_color (str): 共享神经元的颜色。
        title (str): 图表标题。
        output_path (str): 生成图表的完整保存路径。
        scheme (str): 绘制方案，'A' 表示仅绘制共享神经元，'B' 表示绘制两者所有神经元并高亮共享部分。默认为 'B'。
        show_title (bool): 是否显示图表标题。默认为 True。
        alpha_non_shared (float): Scheme B 中非共享神经元的透明度。默认为 0.3。
        shared_marker_size_factor (float): Scheme B 中共享神经元标记点大小的放大系数。默认为 1.5。
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    base_marker_size = 150

    if scheme == 'A':
        if shared_key_neurons_df.empty:
            print(f"行为 {behavior1_name} 和 {behavior2_name} 之间无共享神经元可用于方案A绘制。将绘制空图表。")
            if show_title: ax.set_title(title)
            ax.text(0.5, 0.5, f'No shared neurons between\n{behavior1_name} & {behavior2_name}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            ax.scatter(shared_key_neurons_df['x'], shared_key_neurons_df['y'], c=mixed_color, 
                       s=base_marker_size, edgecolors='black', label=f'Shared ({len(shared_key_neurons_df)})', alpha=0.9)
            for i, txt in enumerate(shared_key_neurons_df['NeuronID']):
                ax.annotate(str(txt), (shared_key_neurons_df['x'].iloc[i], shared_key_neurons_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            if show_title: ax.set_title(title)
            ax.legend(loc='upper right', fontsize='small') # 为方案A添加图例
    
    elif scheme == 'B':
        # 绘制行为1的所有神经元 (非共享部分将半透明)
        ax.scatter(behavior1_all_key_neurons_df['x'], behavior1_all_key_neurons_df['y'], 
                   c=color1, s=base_marker_size, alpha=alpha_non_shared, edgecolors=color1, 
                   label=f'{behavior1_name} ({len(behavior1_all_key_neurons_df)})')
        for i, txt in enumerate(behavior1_all_key_neurons_df['NeuronID']):
            if txt not in shared_key_neurons_df['NeuronID'].values:
                 ax.annotate(str(txt), (behavior1_all_key_neurons_df['x'].iloc[i], behavior1_all_key_neurons_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='dimgray') # 非共享ID用稍暗的灰色

        # 绘制行为2的所有神经元 (非共享部分将半透明)
        ax.scatter(behavior2_all_key_neurons_df['x'], behavior2_all_key_neurons_df['y'], 
                   c=color2, s=base_marker_size, alpha=alpha_non_shared, edgecolors=color2,
                   label=f'{behavior2_name} ({len(behavior2_all_key_neurons_df)})')
        for i, txt in enumerate(behavior2_all_key_neurons_df['NeuronID']):
            if txt not in shared_key_neurons_df['NeuronID'].values:
                ax.annotate(str(txt), (behavior2_all_key_neurons_df['x'].iloc[i], behavior2_all_key_neurons_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='dimgray') # 非共享ID用稍暗的灰色

        if not shared_key_neurons_df.empty:
            # 在顶层绘制共享神经元，高亮显示
            ax.scatter(shared_key_neurons_df['x'], shared_key_neurons_df['y'], 
                       c=mixed_color, 
                       s=base_marker_size * shared_marker_size_factor, 
                       edgecolors='black', 
                       linewidth=1.5, 
                       label=f'Shared ({len(shared_key_neurons_df)})', 
                       alpha=1.0, zorder=3) # 确保共享神经元在最顶层
            for i, txt in enumerate(shared_key_neurons_df['NeuronID']):
                ax.annotate(str(txt), (shared_key_neurons_df['x'].iloc[i], shared_key_neurons_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, weight='bold')
        else:
            print(f"行为 {behavior1_name} 和 {behavior2_name} 之间无共享神经元可用于方案B高亮。")

        if show_title: ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles)) # 去除重复图例项
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    else:
        raise ValueError(f"未知绘图方案: {scheme}。请选择 'A' 或 'B'。")

    # 通用样式
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到 {output_path}")
    plt.close(fig)

def plot_unique_neurons_map(unique_neurons_df, behavior_name, behavior_color, title, output_path, show_title=True):
    """绘制单个行为特有的关键神经元空间分布图。

    参数:
        unique_neurons_df (pd.DataFrame): 包含特定行为的特有关键神经元数据的DataFrame，
                                          必须包含 'NeuronID', 'x', 'y' 列。
        behavior_name (str): 行为名称 (例如, 'Close')。
        behavior_color (str): 用于绘制这些神经元的颜色。
        title (str): 图表标题。
        output_path (str): 生成图表的完整保存路径。
        show_title (bool): 是否显示图表标题。默认为 True。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if unique_neurons_df.empty:
        print(f"行为 '{behavior_name}' 无特有神经元可绘制。跳过绘图: {output_path}")
        if show_title: ax.set_title(title)
        ax.text(0.5, 0.5, f'No unique neurons for\n{behavior_name}', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    else:
        ax.scatter(unique_neurons_df['x'], unique_neurons_df['y'], 
                   c=behavior_color, 
                   s=150, 
                   alpha=0.7, 
                   edgecolors='black',
                   label=f'{behavior_name} Unique ({len(unique_neurons_df)})' # 图例中明确是特有
                  )
        for i, txt in enumerate(unique_neurons_df['NeuronID']):
            ax.annotate(str(txt), (unique_neurons_df['x'].iloc[i], unique_neurons_df['y'].iloc[i]),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        if show_title: ax.set_title(title)
        ax.legend(loc='upper right', fontsize='small')

    # 通用样式
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到 {output_path}")
    plt.close(fig)

if __name__ == '__main__':
    # 此处的示例用法主要用于直接测试本模块的功能。
    print("测试 plot_single_behavior_activity_map 函数...")
    dummy_neuron_data = {
        'NeuronID': [3, 9, 19, 25, 31],
        'x': [0.1, 0.2, 0.3, 0.4, 0.5],
        'y': [0.5, 0.4, 0.3, 0.2, 0.1],
    }
    dummy_df = pd.DataFrame(dummy_neuron_data)
    test_output_dir = "test_plots" # 测试图保存目录
    import os
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    # 测试 plot_single_behavior_activity_map (带标题)
    plot_single_behavior_activity_map(
        key_neurons_df=dummy_df,
        behavior_name="测试行为_带标题",
        behavior_color="purple",
        title="关键神经元分布图 ('测试行为' - 标题可见)",
        output_path=os.path.join(test_output_dir, "single_behavior_test_with_title.png"),
        show_title=True
    )

    # 测试 plot_single_behavior_activity_map (不带标题)
    plot_single_behavior_activity_map(
        key_neurons_df=dummy_df,
        behavior_name="测试行为_无标题",
        behavior_color="teal",
        title="关键神经元分布图 ('测试行为' - 标题隐藏)",
        output_path=os.path.join(test_output_dir, "single_behavior_test_no_title.png"),
        show_title=False
    )

    # 测试 plot_single_behavior_activity_map (空数据)
    empty_df = pd.DataFrame(columns=['NeuronID', 'x', 'y'])
    plot_single_behavior_activity_map(
        key_neurons_df=empty_df,
        behavior_name="测试空数据",
        behavior_color="orange",
        title="关键神经元分布图 ('测试空数据')",
        output_path=os.path.join(test_output_dir, "single_behavior_empty_plot.png"),
        show_title=True
    )
    print(f"单个行为图的测试图表已保存到 {test_output_dir}/ 目录。")

    # 为 plot_shared_neurons_map 和 plot_unique_neurons_map 添加测试用例会更有益
    # 但为简洁起见，此处暂略。可以在实际开发中按需添加。
    print("\nplotting_utils.py 模块测试结束。")