#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage import exposure, restoration, filters, morphology
import matplotlib.pyplot as plt
import os
import sys
import json

class MedicalImageProcessor:
    """医学影像处理类，用于医学图像的去噪和增强"""
    
    def __init__(self, image_path):
        """初始化处理器并加载图像"""
        self.image_path = image_path
        self.original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # 根据图像通道数处理
        if len(self.original.shape) == 2:  # 灰度图像
            self.image = self.original
        else:  # 彩色图像转为灰度
            self.image = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        self.processed = self.image.copy()
    
    def apply_gaussian_filter(self, kernel_size=5):
        """应用高斯滤波进行初步降噪"""
        self.processed = cv2.GaussianBlur(self.processed, (kernel_size, kernel_size), 0)
        return self
    
    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """应用双边滤波，保持边缘的同时降噪"""
        self.processed = cv2.bilateralFilter(self.processed, d, sigma_color, sigma_space)
        return self
    
    def apply_median_filter(self, kernel_size=5):
        """应用中值滤波去除椒盐噪声"""
        self.processed = cv2.medianBlur(self.processed, kernel_size)
        return self
    
    def apply_non_local_means(self, h=10, template_window_size=7, search_window_size=21):
        """应用非局部均值去噪，保留更多细节"""
        self.processed = cv2.fastNlMeansDenoising(
            self.processed, 
            None, 
            h, 
            template_window_size, 
            search_window_size
        )
        return self
    
    def apply_contrast_enhancement(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """使用自适应直方图均衡增强对比度"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.processed = clahe.apply(self.processed)
        return self
    
    def apply_unsharp_masking(self, sigma=1.0, amount=1.0):
        """使用锐化蒙版增强边缘和细节"""
        blurred = filters.gaussian(self.processed, sigma=sigma, preserve_range=True)
        self.processed = np.clip(self.processed + amount * (self.processed - blurred), 0, 255).astype(np.uint8)
        return self
    
    def enhance_bright_spots(self, kernel_size=3, iterations=1, enhancement_factor=1.5):
        """增强亮点
        
        参数:
            kernel_size (int): 形态学内核大小，值越大增强的亮点越大
            iterations (int): 形态学操作的迭代次数
            enhancement_factor (float): 亮点增强系数
        """
        # 创建形态学内核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # 形态学顶帽操作，突出亮点
        tophat = cv2.morphologyEx(self.processed, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
        # 将顶帽结果添加回原图像，使用权重控制增强程度
        self.processed = cv2.addWeighted(self.processed, 1.0, tophat, enhancement_factor, 0)
        return self
    
    def enhance_large_spots(self, min_size=15, enhancement_factor=1.5):
        """专门增强大尺寸的亮块，忽略小亮点
        
        参数:
            min_size (int): 要增强的亮块的最小尺寸
            enhancement_factor (float): 亮块增强系数
        """
        # 创建形态学内核，尺寸对应大亮块
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_size, min_size))
        
        # 提取大亮块
        large_spots = cv2.morphologyEx(self.processed, cv2.MORPH_TOPHAT, large_kernel)
        
        # 将大亮块叠加到原图像
        self.processed = cv2.addWeighted(self.processed, 1.0, large_spots, enhancement_factor, 0)
        return self
    
    def remove_small_spots(self, kernel_size=3, iterations=1, threshold=0):
        """使用形态学操作和阈值处理移除小噪点
        
        参数:
            kernel_size (float): 内核大小，可以是小数值，会被转换为最接近的奇数
            iterations (int): 形态学操作的迭代次数
            threshold (int): 阈值，小于此值的像素将被视为背景 (0-255)
        """
        # 确保kernel_size是奇数
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
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
        """抑制小亮点，同时保留大结构
        
        参数:
            spot_size (int): 要抑制的亮点的尺寸
            strength (float): 抑制强度，0-1之间
        """
        # 创建形态学内核
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spot_size, spot_size))
        
        # 提取小亮点
        small_spots = cv2.morphologyEx(self.processed, cv2.MORPH_TOPHAT, small_kernel)
        
        # 抑制小亮点
        self.processed = cv2.addWeighted(self.processed, 1.0, small_spots, -strength, 0)
        return self
    
    def edge_preserving_smooth(self, sigma_s=60, sigma_r=0.4):
        """使用边缘保持平滑过滤小细节，保留大结构
        
        参数:
            sigma_s (float): 空间邻域大小
            sigma_r (float): 范围邻域大小，值越小保留的边缘越多
        """
        # OpenCV的edgePreservingFilter在某些情况下可能会导致图像尺寸变化
        # 使用双边滤波作为替代方案，它也能保持边缘同时平滑图像
        
        # 计算迭代次数，模拟大sigma_s的效果
        iterations = max(1, int(sigma_s / 10))
        d = 9  # 双边滤波的直径
        
        # 计算双边滤波的sigma_color参数，基于边缘保持的sigma_r
        sigma_color = int(75 * (1 - sigma_r))  # sigma_r越小，sigma_color越大
        sigma_color = max(10, min(150, sigma_color))  # 限制在合理范围内
        
        # 转换为三通道图像以防万一
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
    
    def apply_discretize(self, levels=4, preserve_bright=False, bright_threshold=200):
        """将图像亮度离散化为指定数量的层级
        
        参数:
            levels (int): 离散化的层级数量，2-16之间
            preserve_bright (bool): 是否保留高亮区域的原始亮度
            bright_threshold (int): 高亮区域的亮度阈值 (0-255)
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
    
    def adjust_gamma(self, gamma=1.2):
        """调整图像的伽马值，增强亮部区域"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        self.processed = cv2.LUT(self.processed, table)
        return self
    
    def adaptive_bright_spot_enhancement(self, window_size=15, sensitivity=1.2, min_contrast=10):
        """根据局部区域亮度自适应地增强亮块
        
        这个方法通过分析每个像素周围的局部区域亮度来判断是否为亮块。
        如果一个像素比其周围区域的平均亮度高出一定程度，则认为它是一个亮块。
        
        参数:
            window_size (int): 局部区域窗口大小，必须是奇数
            sensitivity (float): 灵敏度系数，值越大，越容易被判定为亮块
            min_contrast (int): 最小对比度差异，低于此值的差异将被忽略
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
    
    def adaptive_region_enhancement(self, region_size=64, overlap=0.5, enhancement_factor=1.3):
        """基于区域分块的自适应增强
        
        将图像分成多个重叠的区块，对每个区块独立进行分析和增强。
        这种方法特别适合处理图像中不同区域具有不同亮度特征的情况。
        
        参数:
            region_size (int): 分块的大小
            overlap (float): 区块重叠比例，0-1之间
            enhancement_factor (float): 增强系数
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
    
    def save_result(self, output_path=None):
        """保存处理后的图像"""
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
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.processed, cmap='gray')
        plt.title('处理后的图像')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def process_medical_image(image_path, output_path=None, show_result=False):
    """处理医学图像的便捷函数"""
    processor = MedicalImageProcessor(image_path)
    
    # 应用一系列图像处理技术
    (processor
     .apply_gaussian_filter(kernel_size=3)        # 轻度高斯滤波
     .apply_bilateral_filter(d=9, sigma_color=75, sigma_space=75)  # 保留边缘的平滑
     .apply_non_local_means(h=10)                # 非局部均值去噪
     .apply_contrast_enhancement(clip_limit=2.0)  # 增强对比度
     .enhance_bright_spots(kernel_size=5)         # 增强亮点
     .adjust_gamma(gamma=1.2)                     # 调整伽马值
     .apply_unsharp_masking(sigma=1.0, amount=1.5)  # 增强细节
    )
    
    # 保存结果
    result_path = processor.save_result(output_path)
    
    # 显示对比图
    if show_result:
        processor.show_comparison()
    
    return result_path


def process_cell_image(image_path, output_path=None, show_result=False):
    """专门处理细胞图像的函数，去除小噪点并突出细胞结构"""
    processor = MedicalImageProcessor(image_path)
    
    # 应用一系列图像处理技术，特别针对细胞图像调整
    (processor
     .apply_gaussian_filter(kernel_size=3)         # 轻度高斯滤波
     .apply_bilateral_filter(d=13, sigma_color=75, sigma_space=75)  # 加强平滑但保留边缘
     .apply_non_local_means(h=18)                  # 加强非局部均值去噪
     .remove_small_spots(kernel_size=3, iterations=2)  # 去除小噪点
     .apply_contrast_enhancement(clip_limit=2.5)   # 增强对比度
     .enhance_bright_spots(kernel_size=9)          # 增强较大的亮点(细胞)
     .adjust_gamma(gamma=1.3)                      # 调整伽马值
     .apply_unsharp_masking(sigma=1.0, amount=0.8) # 轻微锐化
    )
    
    # 保存结果
    result_path = processor.save_result(output_path)
    
    # 显示对比图
    if show_result:
        processor.show_comparison()
    
    return result_path


def process_image_with_steps(image_path, output_path=None, show_result=False, steps=None):
    """使用自定义步骤处理图像
    
    参数:
        image_path (str): 输入图像路径
        output_path (str, 可选): 输出图像路径
        show_result (bool): 是否显示处理结果
        steps (list): 处理步骤列表，每个步骤是一个包含方法名和参数的字典
            例如：[{'method': 'apply_gaussian_filter', 'params': {'kernel_size': 3}}]
    """
    processor = MedicalImageProcessor(image_path)
    
    # 应用自定义处理步骤
    if steps:
        for step in steps:
            method_name = step.get('method')
            params = step.get('params', {})
            
            # 获取处理方法
            method = getattr(processor, method_name, None)
            if method and callable(method):
                # 应用处理方法
                method(**params)
    
    # 保存结果
    result_path = processor.save_result(output_path)
    
    # 显示对比图
    if show_result:
        processor.show_comparison()
    
    return result_path


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='医学影像处理工具')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    parser.add_argument('--show', '-s', action='store_true', help='显示处理结果')
    parser.add_argument('--cell', '-c', action='store_true', help='使用细胞图像专用处理模式')
    parser.add_argument('--template', '-t', help='使用JSON模板文件定义处理步骤')
    parser.add_argument('--gaussian', '-g', type=int, default=3, help='高斯滤波核大小 (默认: 3)')
    parser.add_argument('--bilateral', '-b', type=int, default=9, help='双边滤波d值 (默认: 9)')
    parser.add_argument('--nlm', '-n', type=int, default=10, help='非局部均值h值 (默认: 10)')
    parser.add_argument('--contrast', '-ct', type=float, default=2.0, help='对比度增强限制 (默认: 2.0)')
    parser.add_argument('--bright', '-br', type=int, default=5, help='亮点增强核大小 (默认: 5)')
    parser.add_argument('--gamma', '-ga', type=float, default=1.2, help='伽马值 (默认: 1.2)')
    parser.add_argument('--unsharp', '-u', type=float, default=1.5, help='锐化强度 (默认: 1.5)')
    parser.add_argument('--remove-spots', '-rs', type=float, default=0, help='小噪点移除强度 (0表示不移除)')
    parser.add_argument('--threshold', '-th', type=int, default=0, help='应用阈值处理，小于此值的像素将被视为背景 (0-255, 0表示不应用)')
    
    args = parser.parse_args()
    
    if args.template:
        # 使用JSON模板文件定义的处理步骤
        try:
            with open(args.template, 'r', encoding='utf-8') as f:
                steps = json.load(f)
            process_image_with_steps(args.input, args.output, args.show, steps)
        except Exception as e:
            print(f"使用模板处理图像时出错: {str(e)}")
            return 1
    elif args.cell:
        # 使用细胞图像专用处理模式
        process_cell_image(args.input, args.output, args.show)
    else:
        # 创建处理器
        processor = MedicalImageProcessor(args.input)
        
        # 应用处理步骤
        processor.apply_gaussian_filter(kernel_size=args.gaussian)
        processor.apply_bilateral_filter(d=args.bilateral)
        processor.apply_non_local_means(h=args.nlm)
        
        # 可选的阈值处理
        if args.threshold > 0:
            processor.apply_threshold(threshold=args.threshold, threshold_type=cv2.THRESH_TOZERO)
        
        # 可选的小噪点移除
        if args.remove_spots > 0:
            processor.remove_small_spots(
                kernel_size=args.remove_spots, 
                iterations=1, 
                threshold=0
            )
            
        processor.apply_contrast_enhancement(clip_limit=args.contrast)
        processor.enhance_bright_spots(kernel_size=args.bright)
        processor.adjust_gamma(gamma=args.gamma)
        processor.apply_unsharp_masking(amount=args.unsharp)
        
        # 保存结果
        result_path = processor.save_result(args.output)
        
        # 显示结果
        if args.show:
            processor.show_comparison()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 