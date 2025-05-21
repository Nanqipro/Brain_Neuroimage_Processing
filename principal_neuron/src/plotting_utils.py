import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict # 用于 plot_shared_neurons_map 中的图例处理

def plot_single_behavior_activity_map(key_neurons_df, behavior_name, behavior_color, title, output_path, 
                                      all_neuron_positions_df=None, # New
                                      show_background_neurons=False, # New
                                      background_neuron_color='lightgray', # New
                                      background_neuron_size=20, # New
                                      background_neuron_alpha=0.5, # New
                                      show_title=True):
    """绘制单个行为的关键神经元空间分布图。

    参数:
        key_neurons_df (pd.DataFrame): 包含关键神经元数据的DataFrame，
                                       必须包含 'NeuronID', 'x', 'y' 列。
        behavior_name (str): 行为名称 (例如, 'Close')。
        behavior_color (str): 用于绘制这些神经元的颜色。
        title (str): 图表标题。
        output_path (str): 生成的图表图片的完整保存路径。
        all_neuron_positions_df (pd.DataFrame, optional): 包含所有神经元位置的DataFrame ('x', 'y' 列)。
                                                            默认为 None。
        show_background_neurons (bool, optional): 是否显示背景中所有非关键神经元。默认为 False。
        background_neuron_color (str, optional): 背景神经元的颜色。默认为 'lightgray'。
        background_neuron_size (int, optional): 背景神经元的标记大小。默认为 20。
        background_neuron_alpha (float, optional): 背景神经元的透明度。默认为 0.5。
        show_title (bool): 是否显示图表标题，默认为 True。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot background neurons if requested
    background_plotted = False
    if show_background_neurons and all_neuron_positions_df is not None and not all_neuron_positions_df.empty:
        # Avoid plotting key neurons as background if they are also in all_neuron_positions_df
        # This assumes key_neurons_df NeuronIDs are a subset of all_neuron_positions_df NeuronIDs
        # For simplicity here, we plot all from all_neuron_positions_df as background first.
        # A more precise way would be to exclude key_neurons from all_neuron_positions_df before plotting background.
        # However, since key neurons are plotted on top with different style, this should be visually acceptable.
        ax.scatter(all_neuron_positions_df['x'], all_neuron_positions_df['y'],
                   c=background_neuron_color,
                   s=background_neuron_size,
                   alpha=background_neuron_alpha,
                   label='All Neurons (Background)', # Updated label
                   zorder=1) # Ensure they are behind key neurons
        background_plotted = True

    if key_neurons_df.empty:
        print(f"行为 '{behavior_name}' 没有关键神经元可供绘制。")
        # If background neurons are shown, the plot might not be "empty"
        # We might still want to show a message about no *key* neurons.
        if not background_plotted:
            # Original empty plot handling if no background neurons are shown either
            ax.text(0.5, 0.5, f'No key neurons for {behavior_name}\\n(Check threshold setting)', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            if show_title:
                ax.set_title(title) # Still set title for clarity
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(output_path)
            plt.close(fig)
            return
        # If background is plotted, we continue to draw the rest of the plot (axes, title, grid)
        # The message about no key neurons is already printed.

    # Plot key neurons (ensure they are on top)
    if not key_neurons_df.empty:
        ax.scatter(key_neurons_df['x'], key_neurons_df['y'], 
                   c=behavior_color, 
                   s=150, # Original size for key neurons
                   alpha=0.7, 
                   edgecolors='black',
                   label=f'Key Neurons ({behavior_name})', # Add label for legend
                   zorder=2) # Ensure key neurons are on top of background

        for i, txt in enumerate(key_neurons_df['NeuronID']):
            ax.annotate(str(txt), (key_neurons_df['x'].iloc[i], key_neurons_df['y'].iloc[i]),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, zorder=3)
    
    if show_title:
        ax.set_title(title)
        
    ax.grid(True, linestyle='--', alpha=0.5) 
    ax.set_facecolor('white') 
    fig.patch.set_facecolor('white')

    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box') 
    
    ax.set_xticks([]) 
    ax.set_yticks([]) 

    # Add legend if background or key neurons are plotted with labels
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Only show legend if there are items to show
        # Use OrderedDict to remove duplicate labels if any, preserving order
        from collections import OrderedDict 
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到 {output_path}")
    plt.close(fig) # 关闭图形以释放内存

def plot_shared_neurons_map(behavior1_name, behavior2_name, 
                            behavior1_all_key_neurons_df, behavior2_all_key_neurons_df, 
                            shared_key_neurons_df, 
                            color1, color2, mixed_color, 
                            title, output_path, 
                            all_neuron_positions_df=None, # New
                            show_background_neurons=False, # New
                            background_neuron_color='lightgray', # New
                            background_neuron_size=20, # New
                            background_neuron_alpha=0.5, # New
                            standard_key_neuron_alpha=0.7, # New default from config
                            use_standard_alpha_for_unshared_in_scheme_b=True, # New default from config
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
        all_neuron_positions_df (pd.DataFrame, optional): 包含所有神经元位置的DataFrame ('x', 'y' 列)。
                                                            默认为 None。
        show_background_neurons (bool, optional): 是否显示背景中所有神经元。默认为 False。
        background_neuron_color (str, optional): 背景神经元的颜色。默认为 'lightgray'。
        background_neuron_size (int, optional): 背景神经元的标记大小。默认为 20。
        background_neuron_alpha (float, optional): 背景神经元的透明度。默认为 0.5。
        standard_key_neuron_alpha (float, optional): 非共享关键神经元在特定条件下使用的标准透明度。默认为 0.7。
        use_standard_alpha_for_unshared_in_scheme_b (bool, optional): 在Scheme B中，非共享关键神经元是否使用 standard_key_neuron_alpha。默认为 True。
        scheme (str): 绘制方案，'A' 表示仅绘制共享神经元，'B' 表示绘制两者所有神经元并高亮共享部分。默认为 'B'。
        show_title (bool): 是否显示图表标题。默认为 True。
        alpha_non_shared (float): Scheme B 中非共享神经元的透明度。默认为 0.3。
        shared_marker_size_factor (float): Scheme B 中共享神经元标记点大小的放大系数。默认为 1.5。
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    base_marker_size = 150

    # Plot background neurons if requested
    if show_background_neurons and all_neuron_positions_df is not None and not all_neuron_positions_df.empty:
        ax.scatter(all_neuron_positions_df['x'], all_neuron_positions_df['y'],
                   c=background_neuron_color,
                   s=background_neuron_size,
                   alpha=background_neuron_alpha,
                   label='All Neurons (Background)',
                   zorder=1) # Ensure they are at the very bottom

    if scheme == 'A':
        if shared_key_neurons_df.empty:
            print(f"行为 {behavior1_name} 和 {behavior2_name} 之间无共享神经元可用于方案A绘制。")
            if not (show_background_neurons and all_neuron_positions_df is not None and not all_neuron_positions_df.empty):
                if show_title: ax.set_title(title)
                ax.text(0.5, 0.5, f'No shared neurons between\\n{behavior1_name} & {behavior2_name}', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            # Continue to draw plot if background is present
        else:
            ax.scatter(shared_key_neurons_df['x'], shared_key_neurons_df['y'], c=mixed_color, 
                       s=base_marker_size, edgecolors='black', label=f'Shared ({len(shared_key_neurons_df)})', alpha=0.9, zorder=4) # zorder above non-shared
            for i, txt in enumerate(shared_key_neurons_df['NeuronID']):
                ax.annotate(str(txt), (shared_key_neurons_df['x'].iloc[i], shared_key_neurons_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, zorder=5)
            # Legend for Scheme A is handled by the main legend call later if show_title is on
    
    elif scheme == 'B':
        # Determine alpha for non-shared key neurons in Scheme B
        current_alpha_for_unshared = alpha_non_shared
        if use_standard_alpha_for_unshared_in_scheme_b:
            current_alpha_for_unshared = standard_key_neuron_alpha

        # Non-shared for Behavior 1 (plotted with specific alpha)
        non_shared_b1_df = behavior1_all_key_neurons_df[~behavior1_all_key_neurons_df['NeuronID'].isin(shared_key_neurons_df['NeuronID'])]
        if not non_shared_b1_df.empty:
            ax.scatter(non_shared_b1_df['x'], non_shared_b1_df['y'], 
                       c=color1, s=base_marker_size, alpha=current_alpha_for_unshared, edgecolors=color1, 
                       label=f'{behavior1_name} (Unique Key: {len(non_shared_b1_df)})', zorder=2)
            for i, txt in enumerate(non_shared_b1_df['NeuronID']):
                 ax.annotate(str(txt), (non_shared_b1_df['x'].iloc[i], non_shared_b1_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='dimgray', zorder=3)

        # Non-shared for Behavior 2
        non_shared_b2_df = behavior2_all_key_neurons_df[~behavior2_all_key_neurons_df['NeuronID'].isin(shared_key_neurons_df['NeuronID'])]
        if not non_shared_b2_df.empty:
            ax.scatter(non_shared_b2_df['x'], non_shared_b2_df['y'], 
                       c=color2, s=base_marker_size, alpha=current_alpha_for_unshared, edgecolors=color2,
                       label=f'{behavior2_name} (Unique Key: {len(non_shared_b2_df)})', zorder=2)
            for i, txt in enumerate(non_shared_b2_df['NeuronID']):
                ax.annotate(str(txt), (non_shared_b2_df['x'].iloc[i], non_shared_b2_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='dimgray', zorder=3)

        if not shared_key_neurons_df.empty:
            ax.scatter(shared_key_neurons_df['x'], shared_key_neurons_df['y'], 
                       c=mixed_color, 
                       s=base_marker_size * shared_marker_size_factor, 
                       edgecolors='black', 
                       linewidth=1.5, 
                       label=f'Shared ({len(shared_key_neurons_df)})', 
                       alpha=1.0, zorder=4) 
            for i, txt in enumerate(shared_key_neurons_df['NeuronID']):
                ax.annotate(str(txt), (shared_key_neurons_df['x'].iloc[i], shared_key_neurons_df['y'].iloc[i]),
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, weight='bold', zorder=5)
        else:
             print(f"行为 {behavior1_name} 和 {behavior2_name} 之间无共享神经元可用于方案B高亮。")
    
    else:
        raise ValueError(f"未知绘图方案: {scheme}。请选择 'A' 或 'B'。")

    if show_title: ax.set_title(title)
    
    # Common styling and legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到 {output_path}")
    plt.close(fig)

def plot_unique_neurons_map(unique_neurons_df, behavior_name, behavior_color, title, output_path, 
                              all_neuron_positions_df=None, # New
                              show_background_neurons=False, # New
                              background_neuron_color='lightgray', # New
                              background_neuron_size=20, # New
                              background_neuron_alpha=0.5, # New
                              show_title=True):
    """绘制单个行为特有的关键神经元空间分布图。

    参数:
        unique_neurons_df (pd.DataFrame): 包含特定行为的特有关键神经元数据的DataFrame，
                                          必须包含 'NeuronID', 'x', 'y' 列。
        behavior_name (str): 行为名称 (例如, 'Close')。
        behavior_color (str): 用于绘制这些神经元的颜色。
        title (str): 图表标题。
        output_path (str): 生成图表的完整保存路径。
        all_neuron_positions_df (pd.DataFrame, optional): 包含所有神经元位置的DataFrame ('x', 'y' 列)。
                                                            默认为 None。
        show_background_neurons (bool, optional): 是否显示背景中所有神经元。默认为 False。
        background_neuron_color (str, optional): 背景神经元的颜色。默认为 'lightgray'。
        background_neuron_size (int, optional): 背景神经元的标记大小。默认为 20。
        background_neuron_alpha (float, optional): 背景神经元的透明度。默认为 0.5。
        show_title (bool): 是否显示图表标题。默认为 True。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot background neurons if requested
    background_plotted = False
    if show_background_neurons and all_neuron_positions_df is not None and not all_neuron_positions_df.empty:
        ax.scatter(all_neuron_positions_df['x'], all_neuron_positions_df['y'],
                   c=background_neuron_color,
                   s=background_neuron_size,
                   alpha=background_neuron_alpha,
                   label='All Neurons (Background)',
                   zorder=1)
        background_plotted = True

    if unique_neurons_df.empty:
        print(f"行为 '{behavior_name}' 无特有神经元可绘制。")
        if not background_plotted:
            if show_title: ax.set_title(title) # Set title even for empty plot text
            ax.text(0.5, 0.5, f'No unique neurons for\\n{behavior_name}', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(output_path)
            plt.close(fig)
            return
        # If background is plotted, continue to draw the rest
    else:
        ax.scatter(unique_neurons_df['x'], unique_neurons_df['y'], 
                   c=behavior_color, 
                   s=150, 
                   alpha=0.7, 
                   edgecolors='black',
                   label=f'{behavior_name} Unique Key ({len(unique_neurons_df)})',
                   zorder=2)
        for i, txt in enumerate(unique_neurons_df['NeuronID']):
            ax.annotate(str(txt), (unique_neurons_df['x'].iloc[i], unique_neurons_df['y'].iloc[i]),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, zorder=3)
    
    if show_title: ax.set_title(title)

    # Common styling and legend
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Use OrderedDict to remove duplicate labels if any, preserving order
        from collections import OrderedDict 
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
    
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