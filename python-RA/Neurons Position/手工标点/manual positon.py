import matplotlib
matplotlib.use('TkAgg')  # 使用支持交互的图形后端

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# （可选）设置中文显示支持，如不需要显示中文可忽略
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

image_path = r'C:\Users\PAN\Desktop\RA\数据集\NO2980\240924EM\2980240924EM.png'  # 替换为你的实际图像路径
img = mpimg.imread(image_path)
img_height, img_width = img.shape[0], img.shape[1]

clicked_points = []   # 用于存储已标记的点 (x, y)
point_plots = []      # 用于存储点的plot对象（line2D对象）
text_labels = []       # 用于存储文本序号对象（text对象）

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title('请用左键标点，右键撤销上一次标点 (关闭窗口结束)')

def onclick(event):
    # 左键点击添加标点
    if event.button == 1 and event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        # 绘制点
        p = ax.plot(x, y, 'ro', markersize=5)[0]
        point_plots.append(p)
        # 序号为当前点的数量
        idx = len(clicked_points)
        t = ax.text(x+5, y, str(idx), color='white', fontsize=8, backgroundcolor='black')
        text_labels.append(t)
        plt.draw()

    # 右键点击撤销上一次标点
    elif event.button == 3:
        if len(clicked_points) > 0:
            # 移除最后一个点
            clicked_points.pop()
            # 移除对应的点对象和文字对象
            last_point_plot = point_plots.pop()
            last_text_label = text_labels.pop()

            last_point_plot.remove()
            last_text_label.remove()
            plt.draw()


# 连接鼠标点击事件
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# 使用阻塞的 show，使窗口保持打开，直到手动关闭
plt.show(block=True)

# 窗口关闭后处理数据
relative_points = []
for (x, y) in clicked_points:
    rel_x = x / img_width
    rel_y = y / img_height
    relative_points.append((rel_x, rel_y))

if len(relative_points) > 0:
    relative_points = np.array(relative_points)
    # 保存标注点相对坐标
    output_file = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\2980 测试集\clicked_points 2980EM.csv'
    np.savetxt(output_file, relative_points, delimiter=',', header='relative_x,relative_y', comments='')
    print(f"已保存标记点相对坐标至 {output_file}")

    # 绘制相对坐标系的散点分布
    fig2, ax2 = plt.subplots()
    sc = ax2.scatter(relative_points[:,0], relative_points[:,1], c='r', s=20)
    ax2.set_title('标记点在相对坐标系中的分布')
    ax2.set_xlabel('relative x')
    ax2.set_ylabel('relative y')
    ax2.set_xlim([0,1])
    ax2.set_ylim([1,0])
    ax2.grid(True)
    # 添加序号
    for i, (rx, ry) in enumerate(relative_points, start=1):
        ax2.text(rx+0.01, ry, str(i), color='red', fontsize=8)
    plt.show(block=True)
else:
    print("未标记任何点，不生成输出文件及相对坐标系散点图。")
