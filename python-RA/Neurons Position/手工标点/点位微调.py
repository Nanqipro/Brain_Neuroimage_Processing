import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv

# 假设你已有点坐标csv文件，如 clicked_points.csv，格式为:
# relative_x,relative_y
input_csv = 'clicked_points day6 test.csv'
output_csv = 'adjusted_points.csv'
background_image_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\240924EM\240924EM.jpg'  # 可选背景图路径，如不需要可不使用

# 读取点坐标
points = []
with open(input_csv, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # 跳过表头
    for row in reader:
        if len(row) < 2:
            continue
        x, y = map(float, row)
        points.append([x, y])
points = np.array(points)

# 如果要显示背景图(相对坐标0~1范围)，请确保图像加载后使用 imshow() 在 [0,1]x[0,1] 范围内
# 假设你的相对坐标也是 0~1 范围, 那可以这样:
img = mpimg.imread(background_image_path)
fig, ax = plt.subplots()
ax.imshow(img, extent=[0,1,1,0])  # extent=[x_min,x_max,y_min,y_max]，这里将y_min>y_max反转垂直坐标

sc = ax.scatter(points[:,0], points[:,1], c='red', s=50, picker=True)
# picker=True 用于选中点

selected = None  # 用于记录当前被拖拽的点的索引

def on_pick(event):
    # 当点击到点时，记录该点的索引
    if event.mouseevent.button == 1:  # 左键选中点
        ind = event.ind[0]
        global selected
        selected = ind

def on_motion(event):
    # 当鼠标移动且有点被选中时，更新点坐标
    global selected
    if selected is not None:
        if event.xdata is not None and event.ydata is not None:
            # 更新点的位置
            points[selected,0] = event.xdata
            points[selected,1] = event.ydata
            sc.set_offsets(points)

            # 重绘
            fig.canvas.draw_idle()

def on_release(event):
    # 鼠标释放时取消选中
    global selected
    selected = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.title("Drag points to adjust positions, close window to finish")
plt.xlim(0,1)
plt.ylim(1,0)  # 因为背景图是y从上到下增大
plt.gca().invert_yaxis()  # 或使用extent设定也可以，让点的方向与图像对应
plt.show()

# 窗口关闭后，points已经是调整后的坐标
# 将其保存回CSV文件
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['relative_x', 'relative_y'])
    for p in points:
        writer.writerow(p)

print(f"调整后的点坐标已保存至 {output_csv}")
