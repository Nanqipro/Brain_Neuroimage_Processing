import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# 加载数据
day1_data = pd.read_excel('Day3.xlsx')
day2_data = pd.read_excel('Day6.xlsx')
day3_data = pd.read_excel('Day9.xlsx')
mapping_table = pd.read_excel('mapping_table.xlsx')

# 用于存储最终的拼接数据（允许空挡）
combined_traces = {}

# 遍历每个神经元编号，以第一天为标准进行拼接
for i, neuron_day1 in enumerate(mapping_table['day1'].dropna()):
    neuron_day2 = mapping_table['day2'].iloc[i] if i < len(mapping_table['day2'].dropna()) else np.nan
    neuron_day3 = mapping_table['day3'].iloc[i] if i < len(mapping_table['day3'].dropna()) else np.nan

    # 提取三天对应的神经元数据，允许空挡
    trace_day1 = day1_data[neuron_day1].values if neuron_day1 in day1_data.columns else np.array([np.nan] * len(day1_data))
    trace_day2 = day2_data[neuron_day2].values if neuron_day2 in day2_data.columns else np.array([np.nan] * len(day2_data))
    trace_day3 = day3_data[neuron_day3].values if neuron_day3 in day3_data.columns else np.array([np.nan] * len(day3_data))

    # 拼接三天的数据成一条曲线，空值部分自动作为空挡
    combined_trace = np.concatenate([trace_day1, trace_day2, trace_day3])
    combined_traces[neuron_day1] = combined_trace  # 使用第一天的编号作为统一的神经元标识

# 绘制 Plotly 图形
fig = go.Figure()

# 设置偏移以避免曲线重叠
offset = np.linspace(-10, 10, len(combined_traces))

for i, (neuron, trace) in enumerate(combined_traces.items()):
    trace_with_offset = trace + offset[i]  # 为每个神经元增加偏移量
    fig.add_trace(go.Scatter(
        x=np.arange(len(trace_with_offset)),
        y=trace_with_offset,
        mode='lines',
        name=f'Neuron {neuron}',
        connectgaps=False  # 不连接缺失数据
    ))

# 创建按钮列表，每个按钮控制一个神经元的显示
buttons = []
for i, neuron in enumerate(combined_traces.keys()):
    # 设置所有神经元的可见性，初始全部不可见
    visibility = [False] * len(combined_traces)
    visibility[i] = True  # 仅显示选中的神经元

    buttons.append(dict(
        label=f'Neuron {neuron}',
        method='update',
        args=[{'visible': visibility},  # 更新每条线的可见性
              {'title': f'Calcium Trace for {neuron}'}]
    ))

# 添加一个“Show All”按钮来显示所有线条
buttons.append(dict(
    label='Show All',
    method='update',
    args=[{'visible': [True] * len(combined_traces)},
          {'title': 'Calcium Trace for All Neurons'}]
))

# 更新图表布局并添加下拉菜单
fig.update_layout(
    title="Neuron Calcium Trace Over Three Days (Combined with Gaps)",
    xaxis_title="Time Points",
    yaxis_title="Calcium Ion Concentration (with Offset)",
    showlegend=True,
    updatemenus=[dict(
        type='dropdown',
        showactive=True,
        buttons=buttons,
        x=1.15,
        y=0.5
    )]
)

fig.show()


