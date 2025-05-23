"""
Python版本的SCN数据处理工具模块

该模块包含了与MATLAB src目录等效的Python实现，用于脑神经元钙成像数据的
相空间分析和数据处理。

模块包含：
- mutual_information: 时间延迟互信息计算
- phase_space_reconstruction: 相空间重构  
- cellset_trim: 数据裁剪
- format_convert: 格式转换

作者: SCN研究小组（Python版本）
日期: 2024
"""

from .mutual import mutual_information
from .phasespace import phase_space_reconstruction
from .cellset2trim import cellset_trim
from .formatConvert import format_convert

__all__ = [
    'mutual_information',
    'phase_space_reconstruction', 
    'cellset_trim',
    'format_convert'
]

__version__ = '1.0.0' 