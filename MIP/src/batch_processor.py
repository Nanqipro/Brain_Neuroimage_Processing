#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import sys
from tqdm import tqdm
from medical_image_processor import MedicalImageProcessor

def process_directory(input_dir, output_dir, file_pattern="*.png", **kwargs):
    """处理目录中的所有医学图像
    
    参数:
        input_dir (str): 输入图像所在目录
        output_dir (str): 输出处理后图像的目录
        file_pattern (str): 文件匹配模式，默认为所有PNG文件
        **kwargs: 传递给处理器的参数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有匹配的文件
    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not image_files:
        print(f"在 {input_dir} 目录中未找到匹配 '{file_pattern}' 的文件")
        return
    
    print(f"找到 {len(image_files)} 个文件，开始处理...")
    
    # 处理每个文件
    for image_path in tqdm(image_files):
        # 构建输出文件路径
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_processed{ext}")
        
        # 处理图像
        try:
            # 创建处理器
            processor = MedicalImageProcessor(image_path)
            
            # 应用处理步骤
            if kwargs.get('use_gaussian', True):
                processor.apply_gaussian_filter(kernel_size=kwargs.get('gaussian_kernel', 3))
            
            if kwargs.get('use_bilateral', True):
                processor.apply_bilateral_filter(
                    d=kwargs.get('bilateral_d', 9),
                    sigma_color=kwargs.get('bilateral_sigma_color', 75),
                    sigma_space=kwargs.get('bilateral_sigma_space', 75)
                )
            
            if kwargs.get('use_nlm', True):
                processor.apply_non_local_means(h=kwargs.get('nlm_h', 10))
            
            if kwargs.get('use_contrast', True):
                processor.apply_contrast_enhancement(clip_limit=kwargs.get('contrast_clip', 2.0))
            
            if kwargs.get('use_bright', True):
                processor.enhance_bright_spots(kernel_size=kwargs.get('bright_kernel', 5))
            
            if kwargs.get('use_gamma', True):
                processor.adjust_gamma(gamma=kwargs.get('gamma', 1.2))
            
            if kwargs.get('use_unsharp', True):
                processor.apply_unsharp_masking(amount=kwargs.get('unsharp_amount', 1.5))
            
            # 保存结果
            processor.save_result(output_path)
            print(f"已处理: {filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
    
    print(f"所有图像处理完成。结果保存在 {output_dir}")
    return True

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='批量处理医学图像')
    parser.add_argument('input_dir', help='输入图像目录')
    parser.add_argument('--output_dir', '-o', help='输出图像目录', default=None)
    parser.add_argument('--pattern', '-p', help='文件匹配模式', default="*.png")
    
    # 添加处理参数
    parser.add_argument('--gaussian', '-g', type=int, default=3, help='高斯滤波核大小 (默认: 3)')
    parser.add_argument('--bilateral', '-b', type=int, default=9, help='双边滤波d值 (默认: 9)')
    parser.add_argument('--nlm', '-n', type=int, default=10, help='非局部均值h值 (默认: 10)')
    parser.add_argument('--contrast', '-c', type=float, default=2.0, help='对比度增强限制 (默认: 2.0)')
    parser.add_argument('--bright', '-br', type=int, default=5, help='亮点增强核大小 (默认: 5)')
    parser.add_argument('--gamma', '-ga', type=float, default=1.2, help='伽马值 (默认: 1.2)')
    parser.add_argument('--unsharp', '-u', type=float, default=1.5, help='锐化强度 (默认: 1.5)')
    
    # 添加处理步骤开关
    parser.add_argument('--no-gaussian', action='store_true', help='禁用高斯滤波')
    parser.add_argument('--no-bilateral', action='store_true', help='禁用双边滤波')
    parser.add_argument('--no-nlm', action='store_true', help='禁用非局部均值去噪')
    parser.add_argument('--no-contrast', action='store_true', help='禁用对比度增强')
    parser.add_argument('--no-bright', action='store_true', help='禁用亮点增强')
    parser.add_argument('--no-gamma', action='store_true', help='禁用伽马调整')
    parser.add_argument('--no-unsharp', action='store_true', help='禁用锐化蒙版')
    
    args = parser.parse_args()
    
    # 默认输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "processed")
    
    # 构建处理参数
    process_params = {
        'gaussian_kernel': args.gaussian,
        'bilateral_d': args.bilateral,
        'nlm_h': args.nlm,
        'contrast_clip': args.contrast,
        'bright_kernel': args.bright,
        'gamma': args.gamma,
        'unsharp_amount': args.unsharp,
        'use_gaussian': not args.no_gaussian,
        'use_bilateral': not args.no_bilateral,
        'use_nlm': not args.no_nlm,
        'use_contrast': not args.no_contrast,
        'use_bright': not args.no_bright,
        'use_gamma': not args.no_gamma,
        'use_unsharp': not args.no_unsharp
    }
    
    # 处理目录
    success = process_directory(args.input_dir, args.output_dir, args.pattern, **process_params)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 