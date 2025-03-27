#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from medical_image_processor import MedicalImageProcessor

class MedicalImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("医学图像处理器")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.image_path = None
        self.processor = None
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        
        # 设置界面
        self._setup_ui()
    
    def _setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 添加控件到控制面板
        ttk.Button(control_frame, text="打开图像", command=self._open_image).grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(control_frame, text="处理图像", command=self._process_image).grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(control_frame, text="保存结果", command=self._save_image).grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        ttk.Button(control_frame, text="重置参数", command=self._reset_params).grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        
        # 参数控制区
        params_frame = ttk.LabelFrame(control_frame, text="参数设置")
        params_frame.grid(row=4, column=0, padx=5, pady=5, sticky='nsew')
        
        # 高斯滤波参数
        ttk.Label(params_frame, text="高斯滤波核大小:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.gaussian_kernel = tk.IntVar(value=3)
        ttk.Spinbox(params_frame, from_=1, to=15, increment=2, textvariable=self.gaussian_kernel, width=5).grid(row=0, column=1, padx=5, pady=5)
        
        # 双边滤波参数
        ttk.Label(params_frame, text="双边滤波强度:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.bilateral_d = tk.IntVar(value=9)
        ttk.Spinbox(params_frame, from_=5, to=15, increment=2, textvariable=self.bilateral_d, width=5).grid(row=1, column=1, padx=5, pady=5)
        
        # 非局部均值参数
        ttk.Label(params_frame, text="非局部均值h值:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.nlm_h = tk.IntVar(value=10)
        ttk.Spinbox(params_frame, from_=1, to=30, increment=1, textvariable=self.nlm_h, width=5).grid(row=2, column=1, padx=5, pady=5)
        
        # 对比度增强参数
        ttk.Label(params_frame, text="对比度增强限制:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.contrast_clip = tk.DoubleVar(value=2.0)
        ttk.Spinbox(params_frame, from_=0.5, to=5.0, increment=0.5, textvariable=self.contrast_clip, width=5).grid(row=3, column=1, padx=5, pady=5)
        
        # 亮点增强参数
        ttk.Label(params_frame, text="亮点增强核大小:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.bright_kernel = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=3, to=11, increment=2, textvariable=self.bright_kernel, width=5).grid(row=4, column=1, padx=5, pady=5)
        
        # 伽马调整参数
        ttk.Label(params_frame, text="伽马值:").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.gamma = tk.DoubleVar(value=1.2)
        ttk.Spinbox(params_frame, from_=0.5, to=2.0, increment=0.1, textvariable=self.gamma, width=5).grid(row=5, column=1, padx=5, pady=5)
        
        # 锐化蒙版参数
        ttk.Label(params_frame, text="锐化强度:").grid(row=6, column=0, padx=5, pady=5, sticky='w')
        self.unsharp_amount = tk.DoubleVar(value=1.5)
        ttk.Spinbox(params_frame, from_=0.5, to=3.0, increment=0.1, textvariable=self.unsharp_amount, width=5).grid(row=6, column=1, padx=5, pady=5)
        
        # 处理步骤选择
        steps_frame = ttk.LabelFrame(control_frame, text="处理步骤")
        steps_frame.grid(row=5, column=0, padx=5, pady=5, sticky='nsew')
        
        self.use_gaussian = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps_frame, text="高斯滤波", variable=self.use_gaussian).grid(row=0, column=0, padx=5, pady=2, sticky='w')
        
        self.use_bilateral = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps_frame, text="双边滤波", variable=self.use_bilateral).grid(row=1, column=0, padx=5, pady=2, sticky='w')
        
        self.use_nlm = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps_frame, text="非局部均值去噪", variable=self.use_nlm).grid(row=2, column=0, padx=5, pady=2, sticky='w')
        
        self.use_contrast = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps_frame, text="对比度增强", variable=self.use_contrast).grid(row=3, column=0, padx=5, pady=2, sticky='w')
        
        self.use_bright = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps_frame, text="亮点增强", variable=self.use_bright).grid(row=4, column=0, padx=5, pady=2, sticky='w')
        
        self.use_gamma = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps_frame, text="伽马调整", variable=self.use_gamma).grid(row=5, column=0, padx=5, pady=2, sticky='w')
        
        self.use_unsharp = tk.BooleanVar(value=True)
        ttk.Checkbutton(steps_frame, text="锐化蒙版", variable=self.use_unsharp).grid(row=6, column=0, padx=5, pady=2, sticky='w')
        
        # 右侧图像显示区
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 原始图像和处理后图像的标签
        self.original_label = ttk.Label(image_frame, text="原始图像")
        self.original_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.processed_label = ttk.Label(image_frame, text="处理后图像")
        self.processed_label.grid(row=0, column=1, padx=5, pady=5)
        
        # 图像显示框
        self.original_canvas = tk.Canvas(image_frame, bg="black", width=500, height=500)
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5)
        
        self.processed_canvas = tk.Canvas(image_frame, bg="black", width=500, height=500)
        self.processed_canvas.grid(row=1, column=1, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _open_image(self):
        """打开图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择医学图像",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.dcm"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.image_path = file_path
            self.processor = MedicalImageProcessor(file_path)
            self.original_image = self.processor.image
            self.current_image = self.original_image
            
            # 显示原始图像
            self._display_image(self.original_image, self.original_canvas)
            self.status_var.set(f"已加载图像: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载图像时出错: {str(e)}")
    
    def _process_image(self):
        """处理图像"""
        if self.processor is None:
            messagebox.showinfo("提示", "请先打开图像")
            return
        
        try:
            # 重置处理器
            self.processor.processed = self.processor.image.copy()
            
            # 根据用户选择的步骤应用处理
            if self.use_gaussian.get():
                self.processor.apply_gaussian_filter(kernel_size=self.gaussian_kernel.get())
            
            if self.use_bilateral.get():
                self.processor.apply_bilateral_filter(d=self.bilateral_d.get())
            
            if self.use_nlm.get():
                self.processor.apply_non_local_means(h=self.nlm_h.get())
            
            if self.use_contrast.get():
                self.processor.apply_contrast_enhancement(clip_limit=self.contrast_clip.get())
            
            if self.use_bright.get():
                self.processor.enhance_bright_spots(kernel_size=self.bright_kernel.get())
            
            if self.use_gamma.get():
                self.processor.adjust_gamma(gamma=self.gamma.get())
            
            if self.use_unsharp.get():
                self.processor.apply_unsharp_masking(amount=self.unsharp_amount.get())
            
            # 更新显示
            self.processed_image = self.processor.processed
            self._display_image(self.processed_image, self.processed_canvas)
            self.status_var.set("图像处理完成")
        except Exception as e:
            messagebox.showerror("错误", f"处理图像时出错: {str(e)}")
    
    def _save_image(self):
        """保存处理后的图像"""
        if self.processed_image is None:
            messagebox.showinfo("提示", "请先处理图像")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存处理后的图像",
            defaultextension=".png",
            filetypes=[
                ("PNG 图像", "*.png"),
                ("JPEG 图像", "*.jpg"),
                ("TIFF 图像", "*.tif"),
                ("BMP 图像", "*.bmp"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            cv2.imwrite(file_path, self.processed_image)
            self.status_var.set(f"图像已保存至: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图像时出错: {str(e)}")
    
    def _reset_params(self):
        """重置所有参数到默认值"""
        self.gaussian_kernel.set(3)
        self.bilateral_d.set(9)
        self.nlm_h.set(10)
        self.contrast_clip.set(2.0)
        self.bright_kernel.set(5)
        self.gamma.set(1.2)
        self.unsharp_amount.set(1.5)
        
        self.use_gaussian.set(True)
        self.use_bilateral.set(True)
        self.use_nlm.set(True)
        self.use_contrast.set(True)
        self.use_bright.set(True)
        self.use_gamma.set(True)
        self.use_unsharp.set(True)
        
        self.status_var.set("参数已重置为默认值")
    
    def _display_image(self, img, canvas):
        """在画布上显示图像"""
        # 清空画布
        canvas.delete("all")
        
        # 调整图像大小以适应画布
        height, width = img.shape[:2]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # 如果画布还没有被正确初始化，使用默认尺寸
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 500
        
        # 计算缩放比例
        scale = min(canvas_width / width, canvas_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 将OpenCV图像转换为PIL格式
        if len(img.shape) == 2:  # 灰度图像
            pil_img = Image.fromarray(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        else:  # BGR格式转RGB
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 调整大小
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # 转换为PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # 保存引用，防止被垃圾回收
        canvas.image = photo
        
        # 在画布中央显示图像
        canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2,
                            anchor=tk.NW, image=photo)

def main():
    """GUI程序入口点"""
    root = tk.Tk()
    app = MedicalImageGUI(root)
    root.mainloop()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 