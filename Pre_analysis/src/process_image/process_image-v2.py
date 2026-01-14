# pip install cellpose
import os
import numpy as np
from cellpose import models, io, plot

# 加载模型 (cyto3 是通用的细胞模型)
pretrained_model = os.environ.get("CELLPOSE_PRETRAINED_MODEL", "cpsam")
model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)

# 读取图片3
img = io.imread('../../raw_data/test.jpg')

# 自动预测 masks
channel_axis = None
if getattr(img, "ndim", 0) == 3:
    channel_axis = -1
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3, 4):
        channel_axis = 0
masks, flows, styles = model.eval(img, diameter=None, channel_axis=channel_axis)

img_vis = img
if channel_axis == 0:
    img_vis = np.transpose(img, (1, 2, 0))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../processed_data/neuron_mask.png"))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
neuron_mask = plot.mask_overlay(img_vis, masks)

# cellpose自带绘图工具可以直接生成你右边那种图
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,8))
plot.show_segmentation(fig, img, masks, flows[0])
plt.imsave(output_path, neuron_mask)
plt.show()
