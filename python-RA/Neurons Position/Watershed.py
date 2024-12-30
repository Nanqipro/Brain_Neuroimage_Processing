import cv2
import numpy as np

# 读取红色掩膜（灰度）
red_mask = cv2.imread(r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\red_mask_preview.png", cv2.IMREAD_GRAYSCALE)
if red_mask is None:
    raise IOError("无法读取red_mask_preview.png，请检查文件路径和文件存在性。")

# 读取原始图像（以彩色方式读取）
img = cv2.imread(r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\240924EM\240924EM.jpg", cv2.IMREAD_COLOR)
if img is None:
    raise IOError("无法读取240924EM.jpg，请检查文件路径和文件存在性。")

# 对二值图（red_mask）进行形态学开运算，清理噪点
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域（膨胀操作）
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 距离变换找到前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
thresh_val = 0.5 * dist_transform.max()
_, sure_fg = cv2.threshold(dist_transform, thresh_val, 255, 0)

# sure_fg 是float类型，将其转化为 uint8类型
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连通组件
num_labels, markers = cv2.connectedComponents(sure_fg)

# 所有背景标签加1，以区分背景和前景
markers = markers + 1

# 未知区域标记为0
markers[unknown == 255] = 0

# 确保markers为int32类型，以符合watershed输入要求
markers = markers.astype(np.int32)

# 使用分水岭算法
markers = cv2.watershed(img, markers)

# 分割后，markers中标记了不同对象
unique_labels = np.unique(markers)
# 对象的标记值大于1
object_labels = [lbl for lbl in unique_labels if lbl > 1]

neurons_positions = []
for lbl in object_labels:
    # 提取该label对应的区域
    mask_obj = (markers == lbl).astype(np.uint8)
    # 计算该区域的质心
    M = cv2.moments(mask_obj)
    if M['m00'] != 0:
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        neurons_positions.append((cx, cy))

print("分割得到的神经元坐标：", neurons_positions)

# 可选：将结果以图像形式可视化
# 在原图上绘制质心
for (x, y) in neurons_positions:
    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

cv2.imwrite("watershed_result.png", img)
print("已在watershed_result.png中绘制结果标记。")
