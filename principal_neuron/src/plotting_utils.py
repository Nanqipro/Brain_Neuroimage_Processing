import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict # 用于 plot_shared_neurons_map 中的图例处理

# 导入并应用matplotlib样式配置
try:
    from matplotlib_config import setup_matplotlib_style
    setup_matplotlib_style()
except ImportError:
    print("警告: 无法导入matplotlib_config，使用默认字体设置")

# Internal helper function to draw neurons on a given axis
def _draw_activity_on_ax(ax, neurons_df, color, size, alpha, edgecolors, label, 
                         annotate_ids=True, neuron_id_col='NeuronID', x_col='x', y_col='y', 
                         z_order=1, annotation_fontsize=18, annotation_offset=(0,12), annotation_weight='bold'):
    """
    Helper function to draw neuron scatter points and annotations on a given matplotlib axis.
    """
    if neurons_df is not None and not neurons_df.empty:
        ax.scatter(neurons_df[x_col], neurons_df[y_col], 
                   c=color, 
                   s=size, 
                   alpha=alpha, 
                   edgecolors=edgecolors,
                   label=label,
                   zorder=z_order)

        if annotate_ids:
            for i, txt in enumerate(neurons_df[neuron_id_col]):
                ax.annotate(str(txt), (neurons_df[x_col].iloc[i], neurons_df[y_col].iloc[i]),
                            textcoords="offset points", xytext=annotation_offset, ha='center', 
                            fontsize=annotation_fontsize, weight=annotation_weight, zorder=z_order + 1) # Annotations on top of their points

def _style_activity_plot_ax(ax):
    """Applies common styling to an activity plot axis."""
    ax.grid(True, linestyle='--', alpha=0.5) 
    ax.set_facecolor('white') 
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box') 
    ax.set_xticks([]) 
    ax.set_yticks([])
    
    # 设置坐标轴标签字体大小 - 使用全局配置
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=16)

def plot_single_behavior_activity_map(key_neurons_df, behavior_name, behavior_color, title, output_path=None, 
                                      all_neuron_positions_df=None, 
                                      show_background_neurons=False, 
                                      background_neuron_color='lightgray', 
                                      background_neuron_size=20, 
                                      background_neuron_alpha=0.5, 
                                      show_title=True,
                                      key_neuron_size=300,
                                      key_neuron_alpha=0.7,
                                      ax=None # New parameter
                                      ):
    """
    绘制单个行为的关键神经元空间分布图。
    如果提供了 ax 参数，则在该 ax 上绘图，否则创建新图并保存到 output_path。
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')

    background_plotted = False
    if show_background_neurons and all_neuron_positions_df is not None and not all_neuron_positions_df.empty:
        _draw_activity_on_ax(ax, all_neuron_positions_df, 
                             color=background_neuron_color, 
                             size=background_neuron_size, 
                             alpha=background_neuron_alpha, 
                             edgecolors=background_neuron_color,
                             label='All Neurons (Background)',
                             annotate_ids=False,
                             z_order=1)
        background_plotted = True

    if key_neurons_df.empty:
        # For standalone plots, print message and potentially create an empty plot text
        if fig is not None: # only print/text if it's a standalone plot
            print(f"行为 '{behavior_name}' 没有关键神经元可供绘制。")
            if not background_plotted:
                ax.text(0.5, 0.5, f'No key neurons for {behavior_name}\\n(Check threshold setting)', 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # If ax is provided (composite plot), the calling function might handle titles/messages for empty subplots
        # or we simply draw an empty styled axis.
    
    # Plot key neurons if not empty
    if not key_neurons_df.empty:
        _draw_activity_on_ax(ax, key_neurons_df,
                             color=behavior_color,
                             size=key_neuron_size,
                             alpha=key_neuron_alpha,
                             edgecolors='black',
                             label=f'Key Neurons ({behavior_name})',
                             annotate_ids=True,
                             z_order=2)
    
    if show_title: # This title becomes the subplot title in a composite
        ax.set_title(title, fontsize=24 if fig is None else 22, fontweight='bold') # 增大标题字体并加粗
    
    _style_activity_plot_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles: 
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=18, markerscale=1.5)

    if fig is not None and output_path is not None: # Save only if it's a standalone plot
        plt.savefig(output_path, bbox_inches='tight')
        print(f"图表已保存到 {output_path}")
        plt.close(fig)
    elif fig is not None and output_path is None and ax is None: # Standalone plot but no path provided (e.g. testing)
        plt.close(fig)

def plot_shared_neurons_map(behavior1_name, behavior2_name, 
                            behavior1_all_key_neurons_df, behavior2_all_key_neurons_df, 
                            shared_key_neurons_df, 
                            color1, color2, mixed_color, 
                            title, output_path=None, 
                            all_neuron_positions_df=None, 
                            show_background_neurons=False, 
                            background_neuron_color='lightgray', 
                            background_neuron_size=20, 
                            background_neuron_alpha=0.5, 
                            standard_key_neuron_alpha=0.7, 
                            use_standard_alpha_for_unshared_in_scheme_b=True, 
                            scheme='B', show_title=True, alpha_non_shared=0.3, shared_marker_size_factor=1.5,
                            ax=None # New parameter
                            ):
    """
    绘制两种行为间共享的关键神经元图。
    如果提供了 ax 参数，则在该 ax 上绘图，否则创建新图并保存到 output_path。
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')

    base_marker_size = 150

    if show_background_neurons and all_neuron_positions_df is not None and not all_neuron_positions_df.empty:
        _draw_activity_on_ax(ax, all_neuron_positions_df,
                             color=background_neuron_color, 
                             size=background_neuron_size, 
                             alpha=background_neuron_alpha,
                             edgecolors=background_neuron_color, 
                             label='All Neurons (Background)',
                             annotate_ids=False, 
                             z_order=1)

    if scheme == 'A':
        if shared_key_neurons_df.empty:
            if fig is not None: print(f"行为 {behavior1_name} 和 {behavior2_name} 之间无共享神经元可用于方案A绘制。")
            # ax.text handled by lack of points if no background
        else:
            _draw_activity_on_ax(ax, shared_key_neurons_df, color=mixed_color, size=base_marker_size,
                                 alpha=0.9, edgecolors='black', label=f'Shared ({len(shared_key_neurons_df)})',
                                 z_order=4, annotation_fontsize=9 if fig is None else 8)
    
    elif scheme == 'B':
        current_alpha_for_unshared = alpha_non_shared
        if use_standard_alpha_for_unshared_in_scheme_b:
            current_alpha_for_unshared = standard_key_neuron_alpha

        non_shared_b1_df = behavior1_all_key_neurons_df[~behavior1_all_key_neurons_df['NeuronID'].isin(shared_key_neurons_df['NeuronID'])]
        if not non_shared_b1_df.empty:
            _draw_activity_on_ax(ax, non_shared_b1_df, color=color1, size=base_marker_size,
                                 alpha=current_alpha_for_unshared, edgecolors=color1,
                                 label=f'{behavior1_name} (Unique Key: {len(non_shared_b1_df)})', z_order=2,
                                 annotation_fontsize=8, annotation_weight='normal', # Smaller for non-shared IDs
                                 annotation_offset=(0,8)) 

        non_shared_b2_df = behavior2_all_key_neurons_df[~behavior2_all_key_neurons_df['NeuronID'].isin(shared_key_neurons_df['NeuronID'])]
        if not non_shared_b2_df.empty:
            _draw_activity_on_ax(ax, non_shared_b2_df, color=color2, size=base_marker_size,
                                 alpha=current_alpha_for_unshared, edgecolors=color2,
                                 label=f'{behavior2_name} (Unique Key: {len(non_shared_b2_df)})', z_order=2,
                                 annotation_fontsize=8, annotation_weight='normal',
                                 annotation_offset=(0,8))

        if not shared_key_neurons_df.empty:
            _draw_activity_on_ax(ax, shared_key_neurons_df, color=mixed_color,
                                 size=base_marker_size * shared_marker_size_factor,
                                 alpha=1.0, edgecolors='black',
                                 label=f'Shared ({len(shared_key_neurons_df)})', z_order=4,
                                 annotation_fontsize=9 if fig is None else 8, annotation_weight='bold')
        elif fig is not None: # Only print if standalone and no shared neurons
             print(f"行为 {behavior1_name} 和 {behavior2_name} 之间无共享神经元可用于方案B高亮。")
    
    else: # Should not happen if called correctly, but good for robustness
        if fig is not None: raise ValueError(f"未知绘图方案: {scheme}。请选择 'A' 或 'B'。")
        else: ax.text(0.5,0.5, f"Error: Unknown scheme '{scheme}'", transform=ax.transAxes)

    if show_title: ax.set_title(title, fontsize=24 if fig is None else 22, fontweight='bold')
    
    _style_activity_plot_ax(ax)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=18, markerscale=1.5)

    if fig is not None and output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"图表已保存到 {output_path}")
        plt.close(fig)
    elif fig is not None and output_path is None and ax is None:
        plt.close(fig)

def plot_unique_neurons_map(unique_neurons_df, behavior_name, behavior_color, title, output_path=None, 
                              all_neuron_positions_df=None, 
                              show_background_neurons=False, 
                              background_neuron_color='lightgray', 
                              background_neuron_size=20, 
                              background_neuron_alpha=0.5, 
                              show_title=True,
                              key_neuron_size=300, # Added for consistency
                              key_neuron_alpha=0.7,  # Added for consistency
                              ax=None # New parameter
                              ):
    """
    绘制单个行为特有的关键神经元空间分布图。
    如果提供了 ax 参数，则在该 ax 上绘图，否则创建新图并保存到 output_path。
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')

    background_plotted = False
    if show_background_neurons and all_neuron_positions_df is not None and not all_neuron_positions_df.empty:
        _draw_activity_on_ax(ax, all_neuron_positions_df,
                             color=background_neuron_color, 
                             size=background_neuron_size, 
                             alpha=background_neuron_alpha,
                             edgecolors=background_neuron_color, 
                             label='All Neurons (Background)',
                             annotate_ids=False, 
                             z_order=1)
        background_plotted = True

    if unique_neurons_df.empty:
        if fig is not None: print(f"行为 '{behavior_name}' 无特有神经元可绘制。")
        # ax.text handled by lack of points if no background
    else:
        _draw_activity_on_ax(ax, unique_neurons_df, color=behavior_color, size=key_neuron_size,
                             alpha=key_neuron_alpha, edgecolors='black',
                             label=f'{behavior_name} Unique Key ({len(unique_neurons_df)})',
                             z_order=2, annotation_fontsize=18 if fig is None else 16)
    
    if show_title: ax.set_title(title, fontsize=24 if fig is None else 22, fontweight='bold')

    _style_activity_plot_ax(ax)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=18, markerscale=1.5)
    
    if fig is not None and output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"图表已保存到 {output_path}")
        plt.close(fig)
    elif fig is not None and output_path is None and ax is None:
        plt.close(fig)

def plot_combined_9_grid(plot_configurations, output_path, main_title_text):
    """
    Generates a 3x3 composite plot from 9 individual plot configurations.
    
    Args:
        plot_configurations (list): A list of 9 dictionaries. Each dictionary must contain:
            'plot_type' (str): 'single', 'shared', or 'unique'.
            'params' (dict): Parameters for the corresponding plotting function. 
                             It MUST include all necessary parameters for that plot type, 
                             EXCEPT for 'ax' and 'output_path' which are handled here.
        output_path (str): Path to save the combined plot.
        main_title_text (str): Main title for the combined plot.
    """
    if len(plot_configurations) != 9:
        print("Error: plot_combined_9_grid requires exactly 9 plot configurations. Skipping.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(24, 24)) # Adjust figsize as needed
    fig.patch.set_facecolor('white')

    for i, config in enumerate(plot_configurations):
        row, col = divmod(i, 3)
        current_ax = axes[row, col]
        
        plot_type = config.get('plot_type')
        params = config.get('params', {})
        params['ax'] = current_ax # Add the ax to the parameters

        # Ensure 'output_path' is not in params for subplot calls as it's handled by the main figure
        if 'output_path' in params:
            del params['output_path']

        if plot_type == 'single':
            plot_single_behavior_activity_map(**params)
        elif plot_type == 'shared':
            plot_shared_neurons_map(**params)
        elif plot_type == 'unique':
            plot_unique_neurons_map(**params)
        else:
            print(f"Warning: Unknown plot_type '{plot_type}' for subplot at [{row},{col}]. Skipping.")
            current_ax.text(0.5, 0.5, f"Unknown plot type: {plot_type}", ha='center', va='center', transform=current_ax.transAxes)
            _style_activity_plot_ax(current_ax)

    if main_title_text and main_title_text.strip():
      fig.suptitle(main_title_text, fontsize=20, y=0.98) # Adjust y to prevent overlap

    plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # rect=[left, bottom, right, top]
    plt.savefig(output_path, bbox_inches='tight')
    print(f"3x3组合图已保存到 {output_path}")
    plt.close(fig)

def plot_3d_combined_neuron_distribution(dataset_centers, path_config, output_path,
                                         scale_xy_mm=1.0, dataset_colors=None,
                                         marker_size=60, alpha=0.9,
                                         bg_color="#eaf2ff",
                                         z_scale=0.4):
    import os
    import numpy as np
    from data_loader import load_neuron_positions
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    if dataset_colors is None:
        dataset_colors = {'emtrace01': 'royalblue', '2980': 'orange', 'bla6250': 'seagreen'}
    all_xyz = []
    z_centers = [c[2] for c in dataset_centers.values()]
    z_mean = np.mean(z_centers) if len(z_centers) > 0 else 0.0

    for key, center in dataset_centers.items():
        cx, cy, cz = center
        position_path = None
        if hasattr(path_config, 'DATASETS') and key in path_config.DATASETS:
            position_path = path_config.DATASETS[key].get('position')
        df = None
        if position_path:
            try:
                df = load_neuron_positions(position_path)
            except Exception:
                df = None
        if df is not None and not df.empty:
            xs = cx + (df['x'].values - 0.5) * scale_xy_mm
            ys = cy + (df['y'].values - 0.5) * scale_xy_mm
            cz_scaled = (cz - z_mean) * z_scale + z_mean
            zs = np.full_like(xs, cz_scaled)
            all_xyz.append((xs, ys, zs))
            ax.scatter(xs, ys, zs, s=marker_size,
                       c=dataset_colors.get(key, 'gray'), alpha=alpha,
                       edgecolors='white', linewidths=0.3,
                       label=f"{key}")
        else:
            cz_scaled = (cz - z_mean) * z_scale + z_mean
            ax.scatter([cx], [cy], [cz_scaled], s=marker_size * 6,
                       c=dataset_colors.get(key, 'gray'), alpha=1.0,
                       edgecolors='black', linewidths=0.6,
                       marker='o', label=f"{key}")

    if all_xyz:
        xs_all = np.concatenate([p[0] for p in all_xyz])
        ys_all = np.concatenate([p[1] for p in all_xyz])
        zs_all = np.concatenate([p[2] for p in all_xyz])
        rx = xs_all.max() - xs_all.min()
        ry = ys_all.max() - ys_all.min()
        rz = zs_all.max() - zs_all.min() if zs_all.size > 0 else 1.0
        ax.set_xlim(xs_all.min() - 0.08 * rx, xs_all.max() + 0.08 * rx)
        ax.set_ylim(ys_all.min() - 0.08 * ry, ys_all.max() + 0.08 * ry)
        ax.set_zlim(zs_all.min() - 0.05 * rz, zs_all.max() + 0.05 * rz)

        try:
            ax.set_box_aspect((1, 1, z_scale))
        except Exception:
            pass

        anchor_x = xs_all.min() - 0.05 * rx
        anchor_y = ys_all.min() - 0.05 * ry
        anchor_z = zs_all.min()
        L = 0.08 * max(rx, ry, rz)
        ax.plot([anchor_x, anchor_x], [anchor_y, anchor_y], [anchor_z, anchor_z + L], color='black')
        ax.plot([anchor_x, anchor_x + L], [anchor_y, anchor_y], [anchor_z, anchor_z], color='black')
        ax.plot([anchor_x, anchor_x], [anchor_y, anchor_y + L], [anchor_z, anchor_z], color='black')
        ax.text(anchor_x + L, anchor_y, anchor_z, 'AP', color='black', fontsize=12)
        ax.text(anchor_x, anchor_y + L, anchor_z, 'ML', color='black', fontsize=12)
        ax.text(anchor_x, anchor_y, anchor_z + L, 'DV', color='black', fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=-60)
    leg = ax.legend(loc='upper right', frameon=True)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_alpha(0.95)
    plt.tight_layout()
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"3D分布图已保存到 {output_path}")
    return fig, ax

if __name__ == '__main__':
    # 此处的示例用法主要用于直接测试本模块的功能。
    # 由于函数现在可以接收 ax, 直接测试单个图的保存依然有效。
    # 测试 plot_combined_9_grid 会比较复杂，通常在主脚本中完成。
    print("plotting_utils.py 模块测试开始（主要测试独立绘图功能）。")
    # ... (Keep existing test code for single plots if desired, ensuring output_path is provided) ...
    # Example:
    dummy_neuron_data = {
        'NeuronID': [3, 9, 19, 25, 31], 'x': [0.1, 0.2, 0.3, 0.4, 0.5], 'y': [0.5, 0.4, 0.3, 0.2, 0.1],
    }
    dummy_df = pd.DataFrame(dummy_neuron_data)
    test_output_dir = "test_plots_standalone" 
    import os
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    # Test plot_single_behavior_activity_map (standalone)
    plot_single_behavior_activity_map(
        key_neurons_df=dummy_df, behavior_name="TestSingle", behavior_color="purple",
        title="Standalone Single Behavior", 
        output_path=os.path.join(test_output_dir, "standalone_single.png"),
        all_neuron_positions_df=dummy_df, show_background_neurons=True # Example with background
    )
    print(f"独立图表示例已保存到 {test_output_dir}/standalone_single.png")
    print("\nplotting_utils.py 模块测试结束。")
