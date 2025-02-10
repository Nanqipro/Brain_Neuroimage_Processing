"""
动态排序柱状图可视化模块
根据神经元钙离子浓度实时排序并生成动态柱状图动画
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.io as pio

# 设置默认绘图模板
pio.templates.default = "plotly"

def load_neuron_data(file_path):
    """加载神经元数据（复用Time_topology中的实现）"""
    data = pd.read_excel(file_path)
    neuron_cols = [col for col in data.columns if 'n' in col.lower() and col[1:].isdigit()]
    
    if not neuron_cols:
        raise ValueError("未找到神经元列!")
    if 'behavior' not in data.columns:
        raise ValueError("缺少行为列!")
    
    print(f"找到{len(neuron_cols)}个神经元列:", neuron_cols)
    data[neuron_cols] = data[neuron_cols].clip(lower=0)  # 确保所有神经元数据非负
    return data, neuron_cols

def process_sorting_data(data, neuron_cols):
    """
    处理排序数据并生成动画帧
    返回结构：
    {
        'sorted_neurons': [],  # 排序后的神经元名称列表
        'values': [],          # 对应的值列表
        'colors': [],          # 颜色值列表
        'titles': [],          # 时间标题
        'behaviors': []        # 行为标签
    }
    """
    max_value = data[neuron_cols].max().max()  # 用于颜色标准化
    frames_data = {
        'sorted_neurons': [],
        'values': [],
        'colors': [],
        'titles': [],
        'behaviors': []
    }

    for idx in tqdm(range(len(data)), desc="处理排序数据"):
        current = data.iloc[idx]
        values = current[neuron_cols]
        # 添加数据清洗步骤，确保没有负值
        values = values.clip(lower=0)  # 新增行：将负值设为0
        
        # 创建排序索引
        sorted_indices = np.argsort(-values)
        # 修改为使用iloc获取值
        sorted_neurons = [neuron_cols[i] for i in sorted_indices]
        sorted_values = [values.iloc[i] for i in sorted_indices]  # 修改这里
        
        # 生成颜色映射时添加范围限制
        normalized = (sorted_values - np.min(sorted_values)) / (np.max(sorted_values) - np.min(sorted_values))
        normalized = np.clip(normalized, 0, 1)  # 强制限制在0-1之间
        colors = [f'rgba(0, 0, 255, {v:.4f})' for v in normalized]  # 添加精度限制
        
        # 存储数据
        frames_data['sorted_neurons'].append(sorted_neurons)
        frames_data['values'].append(sorted_values)
        frames_data['colors'].append(colors)
        frames_data['titles'].append(f"神经元活性排序 - 时间: {current['stamp']}")
        frames_data['behaviors'].append(current['behavior'])
    
    return frames_data

def create_sorting_animation(frames_data, output_path):
    """创建排序动画"""
    fig = go.Figure(
        data=[create_bar_trace(frames_data, 0)],
        layout=create_layout(frames_data, 0)
    )
    
    # 添加动画帧
    fig.frames = [
        go.Frame(
            data=[create_bar_trace(frames_data, k)],
            name=f"frame_{k}",
            layout=go.Layout(
                title=frames_data['titles'][k],
                xaxis=dict(
                    categoryorder='array',
                    categoryarray=frames_data['sorted_neurons'][k]
                ),
                annotations=[
                    dict(
                        x=0.02, y=0.98,
                        xref='paper', yref='paper',
                        text=f"当前行为: {frames_data['behaviors'][k]}",
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
        )
        for k in range(len(frames_data['titles']))
    ]
    
    # 修改动画控件配置
    fig.update_layout(
        updatemenus=[
            # 播放/暂停按钮
            {
                "buttons": [
                    {
                        "label": "播放",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    },
                    {
                        "label": "暂停",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.9,
                "y": 1.1,
                "xanchor": "right",
                "yanchor": "top"
            }
        ],
        # 添加速度控制滑块
        sliders=[
            # 时间轴滑块
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "时间点: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 50, "t": 50},
                "len": 1.0,
                "x": 0,
                "y": -0.1,
                "steps": [
                    {
                        "args": [
                            [f"frame_{k}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": f"{k}" if k % 50 == 0 else "",
                        "method": "animate"
                    }
                    for k in range(len(frames_data['titles']))
                ]
            },
            # 速度控制滑块
            {
                "active": 2,  # 默认选择正常速度
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "播放速度: ",
                    "suffix": "x",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.3,  # 滑块长度
                "x": 0.6,    # 位置
                "y": 1.1,    # 位置
                "steps": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": int(200/speed), "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": str(speed),
                        "method": "animate"
                    }
                    for speed in [0.25, 0.5, 1, 2, 4, 8]  # 速度选项
                ]
            }
        ]
    )
    
    fig.write_html(output_path)

def create_bar_trace(frames_data, k):
    """创建单个柱状图trace"""
    return go.Bar(
        x=frames_data['sorted_neurons'][k],
        y=frames_data['values'][k],
        marker_color=frames_data['colors'][k],
        width=0.8,
        text=[f"{v:.2f}" for v in frames_data['values'][k]],
        textposition='outside'
    )

def create_behavior_annotations(behaviors):
    """创建行为区间标记（上下交替分布）"""
    shapes = []
    annotations = []
    current_behavior = behaviors[0]
    start_idx = 0
    total_frames = len(behaviors)
    
    # 参数设置
    timeline_y = -0.1      # 时间轴线Y坐标
    upper_y = -0.05       # 上方标签Y坐标
    lower_y = -0.15       # 下方标签Y坐标
    line_width = 1        # 线条宽度
    is_upper = True       # 标记是否为上方标签

    for i in range(1, total_frames):
        if behaviors[i] != current_behavior:
            # 计算归一化位置
            x_pos = i / total_frames
            
            # 添加竖线
            shapes.append({
                'type': 'line',
                'x0': x_pos, 'x1': x_pos,
                'y0': timeline_y - 0.02,
                'y1': timeline_y + 0.02,
                'xref': 'paper',
                'yref': 'paper',
                'line': {'color': 'black', 'width': line_width}
            })
            
            # 添加行为标签（上下交替）
            mid_pos = (start_idx + i) / (2 * total_frames)
            annotations.append({
                'x': mid_pos,
                'y': upper_y if is_upper else lower_y,
                'xref': 'paper',
                'yref': 'paper',
                'text': current_behavior,
                'showarrow': False,
                'font': {'size': 12},
                'xanchor': 'center',
                'yanchor': 'bottom' if is_upper else 'top'
            })
            
            current_behavior = behaviors[i]
            start_idx = i
            is_upper = not is_upper  # 切换上下位置
    
    # 处理最后一个行为区间
    mid_pos = (start_idx + total_frames) / (2 * total_frames)
    annotations.append({
        'x': mid_pos,
        'y': upper_y if is_upper else lower_y,
        'xref': 'paper',
        'yref': 'paper',
        'text': current_behavior,
        'showarrow': False,
        'font': {'size': 12},
        'xanchor': 'center',
        'yanchor': 'bottom' if is_upper else 'top'
    })
    
    # 添加时间轴线
    shapes.append({
        'type': 'line',
        'x0': 0, 'x1': 1,
        'y0': timeline_y,
        'y1': timeline_y,
        'xref': 'paper',
        'yref': 'paper',
        'line': {'color': 'black', 'width': line_width}
    })
    
    return shapes, annotations

def create_layout(frames_data, initial_frame):
    """创建布局配置"""
    behavior_shapes, behavior_annotations = create_behavior_annotations(frames_data['behaviors'])
    
    return go.Layout(
        title=frames_data['titles'][initial_frame],
        xaxis=dict(
            title='神经元',
            tickangle=45,
            type='category',
        ),
        yaxis=dict(
            title='钙离子浓度',
            range=[0, max(frames_data['values'][0])*1.1]
        ),
        plot_bgcolor='white',
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=f"当前行为: {frames_data['behaviors'][initial_frame]}",
                showarrow=False,
                font=dict(size=14)
            )
        ] + behavior_annotations,
        shapes=behavior_shapes,
        margin=dict(b=100, t=150),  # 增加顶部边距，为速度控制按钮留出空间
        sliders=[{
            "active": initial_frame,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "时间点:",
                "visible": True,
                "xanchor": "right"
            },
            "pad": {"b": 50, "t": 50},  # 调整滑块位置
            "len": 1.0,  # 确保滑块长度与时间轴一致
            "steps": [
                {
                    "args": [
                        [f"frame_{k}"],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ],
                    "label": f"{k}" if k % 50 == 0 else "",
                    "method": "animate"
                }
                for k in range(len(frames_data['titles']))
            ]
        }]
    )

def main():
    """主函数"""
    # 加载数据
    data_path = '../datasets/Day6_with_behavior_labels_filled.xlsx'
    data, neuron_cols = load_neuron_data(data_path)
    
    # 处理排序数据
    frames_data = process_sorting_data(data, neuron_cols)
    
    # 创建动画
    output_path = '../graph/Day6_Dynamic_Sorting.html'
    create_sorting_animation(frames_data, output_path)
    print(f"动画已保存至: {output_path}")

if __name__ == "__main__":
    main()
