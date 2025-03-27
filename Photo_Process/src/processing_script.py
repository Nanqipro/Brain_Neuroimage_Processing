#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
神经元图像处理脚本
用于对神经元显微图像进行降噪处理，强化神经元轮廓，忽略雪花散点
"""

import cv2
import numpy as np
from skimage import exposure, filters, morphology, transform, feature
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

class NeuronImageProcessor:
    """神经元图像处理类，专注于神经元轮廓增强与背景噪点抑制"""
    
    def __init__(self, image_path):
        """
        初始化处理器并加载图像
        
        Parameters
        ----------
        image_path : str
            输入图像文件路径
        """
        self.image_path = image_path
        self.original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # 检查图像是否加载成功
        if self.original is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 根据图像通道数处理
        if len(self.original.shape) == 2:  # 灰度图像
            self.image = self.original
        elif len(self.original.shape) == 3 and self.original.shape[2] == 4:  # 带透明通道
            # 保留透明通道信息
            self.alpha = self.original[:, :, 3]
            # 转为灰度处理
            self.image = cv2.cvtColor(self.original[:, :, 0:3], cv2.COLOR_BGR2GRAY)
        else:  # 彩色图像转为灰度
            self.image = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        self.processed = self.image.copy()
    
    def apply_strong_denoising(self, h=20, template_window_size=7, search_window_size=21):
        """
        应用强力非局部均值去噪，移除雪花散点
        
        Parameters
        ----------
        h : int
            过滤强度参数，值越大去噪越强
        template_window_size : int
            模板窗口大小
        search_window_size : int
            搜索窗口大小
        """
        self.processed = cv2.fastNlMeansDenoising(
            self.processed, 
            None, 
            h, 
            template_window_size, 
            search_window_size
        )
        return self
    
    def apply_bilateral_filter(self, d=13, sigma_color=75, sigma_space=75):
        """
        应用双边滤波，保持神经元边缘的同时降噪
        
        Parameters
        ----------
        d : int
            滤波直径
        sigma_color : int
            颜色空间标准差
        sigma_space : int
            坐标空间标准差
        """
        self.processed = cv2.bilateralFilter(self.processed, d, sigma_color, sigma_space)
        return self
    
    def apply_median_filter(self, kernel_size=5):
        """
        应用中值滤波去除椒盐噪声
        
        Parameters
        ----------
        kernel_size : int
            内核大小，必须为奇数
        """
        # 确保kernel_size是奇数
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.processed = cv2.medianBlur(self.processed, kernel_size)
        return self
    
    def remove_small_spots(self, kernel_size=5, iterations=2, threshold=40):
        """
        使用形态学操作和阈值处理移除小噪点
        
        Parameters
        ----------
        kernel_size : int
            内核大小，必须为奇数
        iterations : int
            形态学操作的迭代次数
        threshold : int
            阈值，小于此值的像素将被视为背景 (0-255)
        """
        # 确保kernel_size是奇数
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            
        # 先应用阈值处理，如果设置了阈值
        if threshold > 0:
            _, thresholded = cv2.threshold(self.processed, threshold, 255, cv2.THRESH_TOZERO)
            self.processed = thresholded
            
        # 创建形态学内核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # 形态学开运算：先腐蚀后膨胀，可以移除小于内核的结构
        self.processed = cv2.morphologyEx(self.processed, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return self
    
    def suppress_small_spots(self, spot_size=3, strength=0.8):
        """
        抑制小亮点，同时保留大结构
        
        Parameters
        ----------
        spot_size : int
            要抑制的亮点的尺寸
        strength : float
            抑制强度，0-1之间
        """
        # 创建形态学内核
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spot_size, spot_size))
        
        # 提取小亮点
        small_spots = cv2.morphologyEx(self.processed, cv2.MORPH_TOPHAT, small_kernel)
        
        # 抑制小亮点
        self.processed = cv2.addWeighted(self.processed, 1.0, small_spots, -strength, 0)
        return self
    
    def enhance_neuron_structures(self, min_size=15, enhancement_factor=2.5):
        """
        专门增强神经元结构，忽略小噪点
        
        Parameters
        ----------
        min_size : int
            神经元结构的最小尺寸
        enhancement_factor : float
            增强系数
        """
        # 创建形态学内核，尺寸对应神经元结构
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_size, min_size))
        
        # 应用顶帽变换突出亮结构
        neuron_structures = cv2.morphologyEx(self.processed, cv2.MORPH_TOPHAT, large_kernel)
        
        # 将神经元结构叠加到原图像
        self.processed = cv2.addWeighted(self.processed, 1.0, neuron_structures, enhancement_factor, 0)
        return self
    
    def enhance_large_spots(self, min_size=20, enhancement_factor=2.0):
        """
        专门增强大尺寸的神经元结构，忽略小亮点
        
        Parameters
        ----------
        min_size : int
            要增强的亮块的最小尺寸
        enhancement_factor : float
            亮块增强系数
        """
        # 创建形态学内核，尺寸对应大亮块
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_size, min_size))
        
        # 提取大亮块
        large_spots = cv2.morphologyEx(self.processed, cv2.MORPH_TOPHAT, large_kernel)
        
        # 将大亮块叠加到原图像
        self.processed = cv2.addWeighted(self.processed, 1.0, large_spots, enhancement_factor, 0)
        return self
    
    def apply_contrast_enhancement(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        使用自适应直方图均衡增强对比度，使神经元轮廓更加明显
        
        Parameters
        ----------
        clip_limit : float
            对比度限制参数
        tile_grid_size : tuple
            网格尺寸
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.processed = clahe.apply(self.processed)
        return self
    
    def edge_preserving_smooth(self, sigma_s=50, sigma_r=0.3):
        """
        使用边缘保持平滑过滤小细节，保留神经元轮廓
        
        Parameters
        ----------
        sigma_s : float
            空间邻域大小
        sigma_r : float
            范围邻域大小，值越小保留的边缘越多
        """
        # 计算迭代次数，模拟大sigma_s的效果
        iterations = max(1, int(sigma_s / 10))
        d = 11  # 双边滤波的直径
        
        # 计算双边滤波的sigma_color参数
        sigma_color = int(75 * (1 - sigma_r))  # sigma_r越小，sigma_color越大
        sigma_color = max(10, min(150, sigma_color))  # 限制在合理范围内
        
        # 转换为三通道图像以应用双边滤波
        if len(self.processed.shape) == 2:
            temp = cv2.cvtColor(self.processed, cv2.COLOR_GRAY2BGR)
        else:
            temp = self.processed.copy()
        
        # 迭代应用双边滤波
        for _ in range(iterations):
            temp = cv2.bilateralFilter(temp, d, sigma_color, sigma_color)
        
        # 如果原图是灰度图，转回灰度
        if len(self.processed.shape) == 2:
            self.processed = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        else:
            self.processed = temp
            
        return self
    
    def adaptive_bright_spot_enhancement(self, window_size=21, sensitivity=1.5, min_contrast=15):
        """
        根据局部区域亮度自适应地增强亮块
        
        这个方法通过分析每个像素周围的局部区域亮度来判断是否为亮块。
        如果一个像素比其周围区域的平均亮度高出一定程度，则认为它是一个亮块。
        
        Parameters
        ----------
        window_size : int
            局部区域窗口大小，必须是奇数
        sensitivity : float
            灵敏度系数，值越大，越容易被判定为亮块
        min_contrast : int
            最小对比度差异，低于此值的差异将被忽略
        """
        # 确保window_size是奇数
        window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
        
        # 创建一个和原图相同大小的输出图像
        enhanced = self.processed.copy()
        
        # 使用高斯滤波获取局部平均亮度
        local_mean = cv2.GaussianBlur(self.processed, (window_size, window_size), 0)
        
        # 计算每个像素与其局部平均的差异
        diff = self.processed.astype(np.float32) - local_mean.astype(np.float32)
        
        # 创建增强系数矩阵
        enhancement_mask = np.zeros_like(diff)
        
        # 根据差异计算增强系数
        # 只有当像素亮度高于局部平均，且差异超过最小对比度时才增强
        bright_spots = (diff > min_contrast)
        enhancement_mask[bright_spots] = diff[bright_spots] * sensitivity
        
        # 应用增强
        enhanced = np.clip(self.processed + enhancement_mask, 0, 255).astype(np.uint8)
        
        # 更新处理后的图像
        self.processed = enhanced
        return self
    
    def adaptive_region_enhancement(self, region_size=64, overlap=0.5, enhancement_factor=1.5):
        """
        基于区域分块的自适应增强
        
        将图像分成多个重叠的区块，对每个区块独立进行分析和增强。
        这种方法特别适合处理图像中不同区域具有不同亮度特征的情况。
        
        Parameters
        ----------
        region_size : int
            分块的大小
        overlap : float
            区块重叠比例，0-1之间
        enhancement_factor : float
            增强系数
        """
        # 将图像转换为浮点型以进行计算
        img_float = self.processed.astype(np.float32)
        
        # 计算步长（考虑重叠）
        stride = int(region_size * (1 - overlap))
        
        # 获取图像尺寸
        height, width = self.processed.shape[:2]
        
        # 创建结果图像和权重图像
        result = np.zeros_like(img_float)
        weights = np.zeros_like(img_float)
        
        # 对每个区块进行处理
        for y in range(0, height - region_size + 1, stride):
            for x in range(0, width - region_size + 1, stride):
                # 提取当前区块
                region = img_float[y:y+region_size, x:x+region_size]
                
                # 计算区块的统计特征
                mean = np.mean(region)
                std = np.std(region)
                
                # 创建高斯权重窗口（使边缘平滑过渡）
                y_coords, x_coords = np.ogrid[:region_size, :region_size]
                y_center = region_size / 2
                x_center = region_size / 2
                weight_window = np.exp(-((x_coords - x_center)**2 + (y_coords - y_center)**2) / (2 * (region_size/4)**2))
                
                # 根据局部统计特征增强区块
                if std > 0:
                    # 计算增强后的区块
                    enhanced_region = mean + (region - mean) * enhancement_factor
                    
                    # 应用权重窗口
                    result[y:y+region_size, x:x+region_size] += enhanced_region * weight_window
                    weights[y:y+region_size, x:x+region_size] += weight_window
        
        # 归一化结果
        weights[weights == 0] = 1  # 避免除以零
        result = result / weights
        
        # 裁剪到有效范围并转回uint8
        self.processed = np.clip(result, 0, 255).astype(np.uint8)
        return self
    
    def apply_discretize(self, levels=4, preserve_bright=True, bright_threshold=180):
        """
        将图像亮度离散化为指定数量的层级，有助于增强对比度
        
        Parameters
        ----------
        levels : int
            离散化的层级数量，2-16之间
        preserve_bright : bool
            是否保留高亮区域的原始亮度
        bright_threshold : int
            高亮区域的亮度阈值 (0-255)
        """
        # 确保levels参数在合理范围内
        levels = max(2, min(16, levels))
        
        # 计算每个层级的值
        level_values = np.linspace(0, 255, levels, dtype=np.uint8)
        
        # 创建查找表进行离散化
        lut = np.zeros(256, dtype=np.uint8)
        step = 256 // levels
        
        for i in range(256):
            # 如果保留亮区域且亮度超过阈值，则不离散化
            if preserve_bright and i >= bright_threshold:
                lut[i] = i
            else:
                lut[i] = level_values[min(i // step, levels - 1)]
        
        # 应用查找表进行离散化
        self.processed = cv2.LUT(self.processed, lut)
        return self
    
    def enhance_edges(self, sigma=1.0, amount=1.8):
        """
        使用锐化蒙版增强神经元边缘
        
        Parameters
        ----------
        sigma : float
            高斯模糊的标准差
        amount : float
            增强程度
        """
        blurred = filters.gaussian(self.processed, sigma=sigma, preserve_range=True)
        self.processed = np.clip(self.processed + amount * (self.processed - blurred), 0, 255).astype(np.uint8)
        return self
    
    def apply_adaptive_threshold(self, block_size=11, c=2):
        """
        应用自适应阈值处理，用于二值化神经元结构
        
        Parameters
        ----------
        block_size : int
            计算阈值的区域大小，必须为奇数
        c : int
            从平均值或加权平均值减去的常数
        """
        # 确保block_size是奇数
        block_size = max(3, block_size if block_size % 2 == 1 else block_size + 1)
        
        self.processed = cv2.adaptiveThreshold(
            self.processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )
        return self
    
    def adjust_gamma(self, gamma=1.3):
        """
        调整图像的伽马值，增强亮部区域（神经元）
        
        Parameters
        ----------
        gamma : float
            伽马值，大于1增强亮部，小于1增强暗部
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        self.processed = cv2.LUT(self.processed, table)
        return self
    
    def apply_ridge_filter(self, sigma=1.0, lower_threshold=0.1, upper_threshold=0.3):
        """
        应用脊线滤波器突出神经元轴突状结构
        
        Parameters
        ----------
        sigma : float
            高斯滤波器的标准差
        lower_threshold : float
            Canny边缘检测的低阈值
        upper_threshold : float
            Canny边缘检测的高阈值
        """
        # 归一化处理图像
        img_norm = self.processed.astype(np.float32) / 255.0
        
        # 应用脊线检测
        # 先使用hessian_matrix计算Hessian矩阵，然后传递给hessian_matrix_eigvals
        hessian_matrices = feature.hessian_matrix(img_norm, sigma=sigma)
        result = feature.hessian_matrix_eigvals(hessian_matrices)
        
        # 结果是两个特征值图像，取其中一个（神经元结构通常在第二个特征值中更明显）
        ridge_response = result[1]
        
        # 归一化到0-255范围
        ridge_response = np.clip(ridge_response, 0, None)
        ridge_response = 255 * (ridge_response / np.max(ridge_response))
        
        # 转换回uint8
        ridge_img = ridge_response.astype(np.uint8)
        
        # 将脊线结果与原图像融合
        self.processed = cv2.addWeighted(self.processed, 0.7, ridge_img, 0.3, 0)
        return self
    
    def apply_threshold_separation(self, threshold=150, neuron_boost=1.5, background_reduce=0.5):
        """
        基于阈值分离神经元和背景，增强神经元区域并降低背景亮度
        
        Parameters
        ----------
        threshold : int
            区分神经元和背景的亮度阈值 (0-255)
        neuron_boost : float
            神经元区域的增强系数
        background_reduce : float
            背景区域的减弱系数 (0-1之间)
        """
        # 创建掩码，标记超过阈值的区域为神经元
        _, neuron_mask = cv2.threshold(self.processed, threshold, 255, cv2.THRESH_BINARY)
        
        # 膨胀神经元区域，确保边缘部分也被包含
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        neuron_mask = cv2.dilate(neuron_mask, kernel, iterations=1)
        
        # 反向掩码用于背景
        background_mask = cv2.bitwise_not(neuron_mask)
        
        # 分别处理神经元和背景区域
        neuron_region = cv2.bitwise_and(self.processed, self.processed, mask=neuron_mask)
        background_region = cv2.bitwise_and(self.processed, self.processed, mask=background_mask)
        
        # 增强神经元区域
        neuron_enhanced = cv2.convertScaleAbs(neuron_region, alpha=neuron_boost, beta=0)
        
        # 降低背景区域亮度
        background_reduced = cv2.convertScaleAbs(background_region, alpha=background_reduce, beta=0)
        
        # 合并处理后的区域
        self.processed = cv2.add(neuron_enhanced, background_reduced)
        
        return self
    
    def enhance_neuron_boundaries(self, kernel_size=3, sigma=1.4, amount=2.0):
        """
        专门增强神经元边界，使轮廓更加清晰
        
        Parameters
        ----------
        kernel_size : int
            边缘检测的内核大小
        sigma : float
            高斯滤波的标准差
        amount : float
            增强程度
        """
        # 使用高斯滤波平滑图像
        smoothed = cv2.GaussianBlur(self.processed, (kernel_size, kernel_size), sigma)
        
        # 使用Sobel算子检测边缘
        sobelx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算边缘幅度
        magnitude = cv2.magnitude(sobelx, sobely)
        
        # 归一化到0-255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 对边缘进行阈值处理，获取强边缘
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        
        # 细化边缘
        edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        
        # 将边缘叠加到原图像上
        self.processed = cv2.addWeighted(self.processed, 1.0, edges, amount * 0.01, 0)
        
        return self
    
    def highlight_neurons_by_local_contrast(self, window_size=25, contrast_threshold=20, amplification=1.8):
        """
        通过局部对比度高亮神经元结构
        
        找出局部区域中高对比度的部分（可能是神经元），并增强它们
        
        Parameters
        ----------
        window_size : int
            局部区域窗口大小
        contrast_threshold : int
            判定为高对比度区域的阈值
        amplification : float
            高对比度区域的增强系数
        """
        # 确保window_size是奇数
        window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
        
        # 对图像进行高斯模糊，获取局部平均亮度
        local_mean = cv2.GaussianBlur(self.processed, (window_size, window_size), 0)
        
        # 计算局部方差/标准差，表示对比度
        local_variance = cv2.GaussianBlur(
            np.square(self.processed.astype(np.float32) - local_mean.astype(np.float32)), 
            (window_size, window_size), 0
        )
        local_std = np.sqrt(local_variance)
        
        # 创建高对比度区域的掩码（可能是神经元）
        high_contrast_mask = (local_std > contrast_threshold).astype(np.uint8) * 255
        
        # 对掩码进行形态学处理，去除噪点并连接断开的区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        high_contrast_mask = cv2.morphologyEx(high_contrast_mask, cv2.MORPH_CLOSE, kernel)
        high_contrast_mask = cv2.morphologyEx(high_contrast_mask, cv2.MORPH_OPEN, kernel)
        
        # 分离高对比度区域和背景
        high_contrast_region = cv2.bitwise_and(self.processed, self.processed, mask=high_contrast_mask)
        background_region = cv2.bitwise_and(
            self.processed, self.processed, 
            mask=cv2.bitwise_not(high_contrast_mask)
        )
        
        # 增强高对比度区域
        high_contrast_enhanced = cv2.convertScaleAbs(high_contrast_region, alpha=amplification, beta=0)
        
        # 合并结果
        self.processed = cv2.add(high_contrast_enhanced, background_region)
        
        return self
    
    def save_result(self, output_path=None):
        """
        保存处理后的图像
        
        Parameters
        ----------
        output_path : str, optional
            输出图像路径，如果不指定则自动生成
        
        Returns
        -------
        str
            保存后的文件路径
        """
        if output_path is None:
            base_name = os.path.basename(self.image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(os.path.dirname(self.image_path), f"{name}_processed{ext}")
        
        cv2.imwrite(output_path, self.processed)
        print(f"处理后的图像已保存至: {output_path}")
        return output_path
    
    def show_comparison(self):
        """显示原始图像和处理后的图像对比"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.processed, cmap='gray')
        plt.title('Processed Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def process_neuron_image(image_path, output_path=None, show_result=False):
    """
    处理神经元图像的便捷函数，采用改进的神经元增强流程
    
    Parameters
    ----------
    image_path : str
        输入图像路径
    output_path : str, optional
        输出图像路径
    show_result : bool
        是否显示处理结果
        
    Returns
    -------
    str
        处理后的图像路径
    """
    processor = NeuronImageProcessor(image_path)
    
    # 应用改进的处理流程，专注于神经元轮廓强化与背景噪点抑制
    (processor
     .apply_strong_denoising(h=20)                # 加强去噪，去除雪花散点
     .apply_bilateral_filter(d=13, sigma_color=75, sigma_space=75)  # 保留边缘的平滑
     .apply_median_filter(kernel_size=5)          # 去除剩余的椒盐噪声
     .remove_small_spots(kernel_size=5, iterations=2, threshold=40)  # 去除小噪点
     .suppress_small_spots(spot_size=3, strength=0.8)  # 专门抑制小亮点
     .adjust_gamma(gamma=1.3)                     # 调整伽马值增强亮部
     .apply_contrast_enhancement(clip_limit=3.0)  # 增强对比度
     .edge_preserving_smooth(sigma_s=50, sigma_r=0.3)  # 边缘保持的平滑
     .adaptive_bright_spot_enhancement(window_size=21, sensitivity=1.5, min_contrast=15)  # 自适应亮点增强
     .enhance_neuron_structures(min_size=15, enhancement_factor=2.5)  # 增强神经元结构
     .enhance_large_spots(min_size=20, enhancement_factor=2.0)  # 专门增强大尺寸结构
     .enhance_edges(sigma=1.0, amount=1.8)        # 增强边缘
     .apply_ridge_filter(sigma=1.0)               # 突出神经元轴突结构
     .adaptive_region_enhancement(region_size=64, overlap=0.5, enhancement_factor=1.5)  # 区域自适应增强
    )
    
    # 保存结果
    result_path = processor.save_result(output_path)
    
    # 显示对比图
    if show_result:
        processor.show_comparison()
    
    return result_path


def process_neuron_image_simplified(image_path, output_path=None, show_result=False):
    """
    处理神经元图像的简化函数，使用核心处理步骤
    
    Parameters
    ----------
    image_path : str
        输入图像路径
    output_path : str, optional
        输出图像路径
    show_result : bool
        是否显示处理结果
        
    Returns
    -------
    str
        处理后的图像路径
    """
    processor = NeuronImageProcessor(image_path)
    
    # 使用简化但高效的处理流程
    (processor
     .apply_strong_denoising(h=20)                # 强力去噪
     .apply_median_filter(kernel_size=5)          # 去除椒盐噪声
     .remove_small_spots(kernel_size=5, iterations=2, threshold=40)  # 去除小噪点
     .apply_contrast_enhancement(clip_limit=3.0)  # 增强对比度
     .enhance_large_spots(min_size=20, enhancement_factor=2.0)  # 增强神经元结构
     .enhance_edges(sigma=1.0, amount=1.8)        # 增强边缘
     .apply_discretize(levels=5, preserve_bright=True, bright_threshold=180)  # 亮度离散化增强对比
    )
    
    # 保存结果
    result_path = processor.save_result(output_path)
    
    # 显示对比图
    if show_result:
        processor.show_comparison()
    
    return result_path


def process_neuron_image_advanced(image_path, output_path=None, show_result=False):
    """
    处理神经元图像的高级函数，专注于神经元分离与轮廓增强
    
    Parameters
    ----------
    image_path : str
        输入图像路径
    output_path : str, optional
        输出图像路径
    show_result : bool
        是否显示处理结果
        
    Returns
    -------
    str
        处理后的图像路径
    """
    processor = NeuronImageProcessor(image_path)
    
    # 应用改进的处理流程，重点是阈值分离和轮廓增强
    (processor
     .apply_strong_denoising(h=20)                # 强力去噪，去除雪花散点
     .apply_bilateral_filter(d=13, sigma_color=75, sigma_space=75)  # 保留边缘的平滑
     .apply_median_filter(kernel_size=5)          # 去除椒盐噪声
     .adjust_gamma(gamma=1.3)                     # 调整伽马值增强亮部
     .apply_contrast_enhancement(clip_limit=3.5)  # 增强对比度
     .highlight_neurons_by_local_contrast(window_size=25, contrast_threshold=20, amplification=1.8)  # 通过局部对比度高亮神经元
     .apply_threshold_separation(threshold=145, neuron_boost=1.7, background_reduce=0.3)  # 阈值分离神经元和背景
     .enhance_large_spots(min_size=20, enhancement_factor=2.0)  # 增强神经元结构
     .enhance_neuron_boundaries(kernel_size=3, sigma=1.4, amount=2.0)  # 增强神经元边界
     .enhance_edges(sigma=1.0, amount=1.5)        # 锐化边缘
    )
    
    # 保存结果
    result_path = processor.save_result(output_path)
    
    # 显示对比图
    if show_result:
        processor.show_comparison()
    
    return result_path


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='神经元图像处理工具')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    parser.add_argument('--show', '-s', action='store_true', help='显示处理结果')
    parser.add_argument('--method', '-m', choices=['full', 'simple', 'advanced'], default='advanced', 
                        help='处理方法: full=完整处理 simple=简化处理 advanced=高级处理(默认)')
    parser.add_argument('--denoise', '-d', type=int, default=20, help='去噪强度 (默认: 20)')
    parser.add_argument('--threshold', '-t', type=int, default=145, help='神经元分离阈值 (默认: 145)')
    parser.add_argument('--enhance', '-e', type=float, default=2.5, help='神经元结构增强系数 (默认: 2.5)')
    parser.add_argument('--background', '-b', type=float, default=0.3, help='背景减弱系数 (默认: 0.3)')
    
    args = parser.parse_args()
    
    if args.method == 'advanced':
        # 创建处理器
        processor = NeuronImageProcessor(args.input)
        
        # 应用高级处理流程
        (processor
         .apply_strong_denoising(h=args.denoise)
         .apply_bilateral_filter(d=13, sigma_color=75, sigma_space=75)
         .apply_median_filter(kernel_size=5)
         .adjust_gamma(gamma=1.3)
         .apply_contrast_enhancement(clip_limit=3.5)
         .highlight_neurons_by_local_contrast(window_size=25, contrast_threshold=20, amplification=1.8)
         .apply_threshold_separation(threshold=args.threshold, neuron_boost=args.enhance, background_reduce=args.background)
         .enhance_large_spots(min_size=20, enhancement_factor=2.0)
         .enhance_neuron_boundaries(kernel_size=3, sigma=1.4, amount=2.0)
         .enhance_edges(sigma=1.0, amount=1.5)
        )
    elif args.method == 'full':
        # 创建处理器
        processor = NeuronImageProcessor(args.input)
        
        # 应用完整处理流程
        (processor
         .apply_strong_denoising(h=args.denoise)
         .apply_bilateral_filter(d=13, sigma_color=75, sigma_space=75)
         .apply_median_filter(kernel_size=5)
         .remove_small_spots(kernel_size=5, iterations=2, threshold=args.threshold)
         .suppress_small_spots(spot_size=3, strength=0.8)
         .adjust_gamma(gamma=1.3)
         .apply_contrast_enhancement(clip_limit=3.0)
         .edge_preserving_smooth(sigma_s=50, sigma_r=0.3)
         .adaptive_bright_spot_enhancement(window_size=21, sensitivity=1.5, min_contrast=15)
         .enhance_neuron_structures(min_size=15, enhancement_factor=args.enhance)
         .enhance_large_spots(min_size=20, enhancement_factor=2.0)
         .enhance_edges(sigma=1.0, amount=1.8)
         .apply_ridge_filter(sigma=1.0)
         .adaptive_region_enhancement(region_size=64, overlap=0.5, enhancement_factor=1.5)
        )
    else:
        # 创建处理器
        processor = NeuronImageProcessor(args.input)
        
        # 应用简化处理流程
        (processor
         .apply_strong_denoising(h=args.denoise)
         .apply_median_filter(kernel_size=5)
         .remove_small_spots(kernel_size=5, iterations=2, threshold=args.threshold)
         .apply_contrast_enhancement(clip_limit=3.0)
         .enhance_large_spots(min_size=20, enhancement_factor=args.enhance)
         .enhance_edges(sigma=1.0, amount=1.8)
         .apply_discretize(levels=5, preserve_bright=True, bright_threshold=180)
        )
    
    # 保存结果
    result_path = processor.save_result(args.output)
    
    # 显示结果
    if args.show:
        processor.show_comparison()
    
    return 0


if __name__ == "__main__":
    # 处理示例图像
    input_path = "../datasets/P1.png"
    # 检查路径是否存在
    if os.path.exists(input_path):
        # 使用高级处理方法
        result_path = process_neuron_image_advanced(input_path, show_result=True)
        print(f"高级处理完成，结果保存至: {result_path}")
        
        # 使用原始方法进行对比
        output_path = os.path.splitext(input_path)[0] + "_original_processed.png"
        process_neuron_image(input_path, output_path=output_path, show_result=True)
        
        # 也可以尝试简化处理流程
        # output_path = os.path.splitext(input_path)[0] + "_simple_processed.png"
        # process_neuron_image_simplified(input_path, output_path=output_path, show_result=True)
    else:
        print(f"找不到图像文件: {input_path}")
        print("请提供正确的神经元图像路径，或者使用命令行参数运行脚本")
        print("示例: python processing_script.py ../datasets/P1.png -s")
