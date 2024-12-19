import matplotlib
matplotlib.use('TkAgg')  # 使用支持交互的图形后端

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# （可选）设置中文显示支持，如不需要显示中文可忽略
# 请确保你的系统中安装了适合的中文字体（如黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

image_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\watershed_result.png'  # 替换为你的实际图像路径
img = mpimg.imread(image_path)
img_height, img_width = img.shape[0], img.shape[1]

clicked_points = []

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title('请在图像上点击标注点 (关闭窗口结束)')

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        ax.plot(x, y, 'ro', markersize=5)
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
    output_file = 'clicked_points.csv'
    np.savetxt(output_file, relative_points, delimiter=',', header='relative_x,relative_y', comments='')
    print(f"已保存标记点相对坐标至 {output_file}")

    # 绘制相对坐标系的散点分布
    fig2, ax2 = plt.subplots()
    ax2.scatter(relative_points[:,0], relative_points[:,1], c='r', s=20)
    ax2.set_title('标记点在相对坐标系中的分布')
    ax2.set_xlabel('relative x')
    ax2.set_ylabel('relative y')
    ax2.set_xlim([0,1])
    # 如果你希望 y 轴从上到下增大，可以保持下面的设定
    ax2.set_ylim([1,0])
    ax2.grid(True)
    plt.show(block=True)
else:
    print("未标记任何点，不生成输出文件及相对坐标系散点图。")
