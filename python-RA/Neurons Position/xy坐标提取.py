import cv2
import numpy as np
import pytesseract
import csv
import math
import os

# 如果没有在系统变量中配置Tesseract路径，请在此处指定Tesseract的安装路径
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 输入图像文件名（请根据实际路径进行修改）
image_path = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\Neurons Position\240924EM\240924EM.jpg'


# 读取图像
img = cv2.imread(image_path)


# ----------------------------
# 1. 提取红色标记（神经元位置）
# ----------------------------

# 将图像从BGR转为HSV空间，方便颜色分割
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义红色范围（根据实际情况调整）
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# 使用形态学操作清理噪点
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 找红点的轮廓
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

neurons_positions = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    # 过滤面积过小的噪点
    if area < 5:
        continue
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        neurons_positions.append((cx, cy))

# ----------------------------
# 2. 提取白色文字区域进行OCR
# ----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 针对白字尝试高阈值
_, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# 形态学处理，让文字更清晰（可根据情况调整）
text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# 找出文字块轮廓
text_contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 用于存放 (编号, 中心x, 中心y)
number_positions = []

for cnt in text_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # 对很小的区域进行过滤，避免噪点
    if w < 5 or h < 5:
        continue

    roi = img[y:y + h, x:x + w]

    # 使用pytesseract进行OCR，只识别数字（可尝试其他config参数）
    text = pytesseract.image_to_string(roi, config='--psm 6 -c tessedit_char_whitelist=0123456789')
    text = text.strip()

    # 检查是否是纯数字
    if text.isdigit():
        cx = x + w / 2
        cy = y + h / 2
        number_positions.append((text, cx, cy))


# ----------------------------
# 3. 将编号与神经元位置匹配 (最近邻匹配)
# ----------------------------
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


matched_results = []
used_indices = set()

for (num, tx, ty) in number_positions:
    # 找最近的神经元位置
    nearest_idx = -1
    nearest_dist = float('inf')
    for i, (nx, ny) in enumerate(neurons_positions):
        if i in used_indices:
            continue
        dist = distance((tx, ty), (nx, ny))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = i
    if nearest_idx != -1:
        used_indices.add(nearest_idx)
        matched_results.append((num, neurons_positions[nearest_idx][0], neurons_positions[nearest_idx][1]))

# 按编号排序（可选）
matched_results.sort(key=lambda x: int(x[0]))

cv2.imwrite("red_mask_preview.png", red_mask)



# ----------------------------
# # 4. 输出结果到CSV
# # ----------------------------
# output_csv_path = 'neuron_positions.csv'
# with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Neuron_ID", "X", "Y"])
#     for res in matched_results:
#         writer.writerow(res)
#
# print(f"神经元坐标提取完成，结果已输出至 {output_csv_path}")
