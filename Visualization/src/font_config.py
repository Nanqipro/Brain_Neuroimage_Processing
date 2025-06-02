#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
matplotlib字体配置模块

解决中文显示和Unicode符号问题
"""

import matplotlib.pyplot as plt
import matplotlib
import platform
import warnings

def configure_matplotlib_fonts():
    """
    配置matplotlib字体设置，解决中文显示和Unicode符号问题
    """
    
    # 获取系统信息
    system = platform.system()
    
    # 设置字体优先级列表
    if system == "Windows":
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans', 'Arial']
    elif system == "Darwin":  # macOS
        font_list = ['PingFang SC', 'Hiragino Sans GB', 'STSong', 'DejaVu Sans', 'Arial']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Liberation Sans', 'Arial']
    
    # 基础字体配置
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['font.family'] = ['sans-serif']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 数学公式字体配置
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.default'] = 'regular'
    
    # 图形质量设置
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # 网格和样式设置
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.axisbelow'] = True
    
    # 禁用部分字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    print(f"字体配置完成 - 系统: {system}")
    print(f"字体优先级: {font_list[:3]}")

def test_font_display():
    """
    测试字体显示效果
    """
    import numpy as np
    
    # 创建测试图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 测试中文显示
    x = np.linspace(-5, 5, 100)
    y = np.sin(x)
    
    ax1.plot(x, y, label='正弦函数')
    ax1.set_title('中文字体测试')
    ax1.set_xlabel('X轴 (负数测试)')
    ax1.set_ylabel('Y轴 (数值范围)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 测试数学符号
    x2 = np.linspace(-3, 3, 50)
    y2 = x2**2 - 2*x2 - 1
    
    ax2.plot(x2, y2, 'r-', label='y = x² - 2x - 1')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('数学符号测试')
    ax2.set_xlabel('X轴')
    ax2.set_ylabel('Y轴')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('font_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("字体测试完成，已保存为 font_test.png")

if __name__ == "__main__":
    configure_matplotlib_fonts()
    test_font_display() 