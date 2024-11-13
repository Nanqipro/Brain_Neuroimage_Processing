# import pandas as pd
# import plotly.express as px
#
# # 读取聚类结果
# file_path = 'calcium_window_emd_cluster_single_sheet.xlsx'
# sheet_name = 'Window_30_Step_5'  # 指定要读取的工作表名称
# df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
#
# # 使用 Plotly 绘制 3D 散点图
# fig = px.scatter_3d(
#     df,
#     x='Mean',  # 使用均值作为 x 轴
#     y='StdDev',  # 使用标准差作为 y 轴
#     z='Peak',  # 使用峰值作为 z 轴
#     color='Cluster',  # 根据聚类标签着色
#     title='3D Scatter Plot of Neuron Activity Clustering',
#     labels={'Mean': 'Mean', 'StdDev': 'StdDev', 'Peak': 'Peak'}
# )
#
# # 显示图表
# fig.update_layout(scene=dict(
#     xaxis_title='Mean',
#     yaxis_title='StdDev',
#     zaxis_title='Peak'
# ))
# fig.show()

# import pandas as pd
# import plotly.express as px
#
# # 读取聚类结果
# file_path = 'calcium_window_emd_cluster_single_sheet.xlsx'
# sheet_name = 'Window_30_Step_5'  # 指定要读取的工作表名称
# df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
#
# # 使用 Plotly 绘制 3D 散点图
# fig = px.scatter_3d(
#     df,
#     x='Mean',  # 使用均值作为 x 轴
#     y='StdDev',  # 使用标准差作为 y 轴
#     z='Peak',  # 使用峰值作为 z 轴
#     color='Cluster',  # 根据聚类标签着色
#     symbol='Neuron',  # 使用不同的符号表示不同神经元
#     hover_name='Neuron',  # 悬停时显示神经元名称
#     hover_data={'Start Time': True, 'Cluster': True},  # 悬停时显示起始时间和类别
#     title='3D Scatter Plot of Neuron Activity Clustering by Window',
#     labels={'Mean': 'Mean', 'StdDev': 'StdDev', 'Peak': 'Peak'}
# )
#
# # 显示图表
# fig.update_layout(scene=dict(
#     xaxis_title='Mean',
#     yaxis_title='StdDev',
#     zaxis_title='Peak'
# ))
# fig.show()

import pandas as pd
import plotly.express as px

# 读取聚类结果
file_path = 'calcium_window_emd_cluster_single_sheet.xlsx'
sheet_name = 'Window_30_Step_5'  # 指定要读取的工作表名称
df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

# 使用 Plotly 绘制 3D 散点图，展示不同神经元在不同时间窗口内的类别
fig = px.scatter_3d(
    df,
    x='Start Time',  # 使用时间窗口的起始时间作为 x 轴
    y='Neuron',  # 不同神经元作为 y 轴
    z='Cluster',  # 聚类类别作为 z 轴
    color='Cluster',  # 根据聚类标签着色
    title='Neuron Clustering across Time Windows',
    labels={'Start Time': 'Time Window Start', 'Neuron': 'Neuron', 'Cluster': 'Cluster'}
)

# 更新图表布局
fig.update_layout(scene=dict(
    xaxis_title='Time Window Start',
    yaxis_title='Neuron',
    zaxis_title='Cluster'
))

fig.show()
