import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import random

def generate_neuron_mask(image_path, output_path="../../processed_data/neuron_mask.png"):
    # 1. 读取图片并转换为灰度图
    # 注意：如果你的输入是那张包含左右两部分的截图，请先裁剪出左边的图保存为 input.png
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法读取图片，请检查路径。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 图像增强 (CLAHE - 限制对比度自适应直方图均衡化)
    # 这步是为了让暗处的神经元也能亮起来，便于提取
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. 图像去噪
    # 使用高斯模糊去除高频噪声
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 4. 二值化 (使用 Otsu 自动阈值)
    # 将图像转换为黑白，白色为细胞，黑色为背景
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 形态学操作 (去噪与平滑)
    # 开运算：先腐蚀后膨胀，去除背景中的微小噪点
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 确定的背景区域 (膨胀操作让细胞边界向外扩展，剩下的肯定是背景)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 6. 距离变换 (Distance Transform)
    # 计算每个前景像素到最近背景像素的距离。细胞中心距离最远，亮度最高。
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 7. 寻找种子点 (Markers)
    # 只有距离变换值大于最大值 0.4 倍的地方才被认为是确定的细胞中心
    # 这个 0.4 参数可以调整：越大细胞越少（只留最亮的中心），越小细胞越多
    ret, sure_fg = cv2.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
    
    # 转换为 uint8 格式
    sure_fg = np.uint8(sure_fg)
    
    # 未知区域 (边缘区域) = 背景 - 前景
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 8. 标记连通区域 (Labeling)
    ret, markers = cv2.connectedComponents(sure_fg)

    # 将所有标记 +1，保证背景是 1，而不是 0
    markers = markers + 1
    # 将未知区域标记为 0
    markers[unknown == 255] = 0

    # 9. 分水岭算法 (Watershed)
    # 从种子点开始灌水，直到填满边缘
    markers = cv2.watershed(img, markers)

    # ---------------------------------------------------------
    # 10. 生成结果图 (生成好看协调的随机颜色)
    # ---------------------------------------------------------
    
    # 创建一个纯白背景的画布
    h, w = gray.shape
    result_viz = np.ones((h, w, 3), dtype=np.uint8) * 255 

    # 获取所有独立的细胞标签 (去掉 -1:边界, 1:背景)
    unique_labels = np.unique(markers)
    
    # 为了颜色好看，我们生成一个颜色列表
    # 使用 HSV 空间生成颜色，保证颜色鲜艳 (Saturation高, Value高)
    colors = {}
    for label in unique_labels:
        if label <= 1: # 忽略背景和边界
            continue
        
        # 随机生成柔和但区分度高的颜色
        # Hue: 0-179, Saturation: 100-200 (不过于刺眼), Value: 150-255
        hue = random.randint(0, 179)
        sat = random.randint(120, 220) 
        val = random.randint(180, 255)
        
        # 转回 BGR 用于 OpenCV
        bgr_color = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors[label] = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

    # 填色
    for y in range(h):
        for x in range(w):
            idx = markers[y, x]
            if idx > 1: # 如果是细胞区域
                result_viz[y, x] = colors[idx]
            # 如果想要描边效果，可以把 idx == -1 的像素设为黑色
            # if idx == -1: 
            #    result_viz[y, x] = (0, 0, 0)

    # 11. 显示与保存
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original (Left)")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Generated Segmentation Mask")
    plt.imshow(cv2.cvtColor(result_viz, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite(output_path, result_viz)
    print(f"处理完成！结果已保存至 {output_path}")

# ================= 使用示例 =================
# 请将下面的 'input.png' 替换为你实际的左图文件名
# 你需要先手动截图把左边那个黑白的图存下来
if __name__ == "__main__":
    # 假设你有一个名为 raw_calcium.png 的左侧原图
    # 这里为了演示，请确保目录下有图片
    try:
        generate_neuron_mask('../../raw_data/test.jpg') 
    except Exception as e:
        print(f"运行出错 (可能是没找到图片): {e}")