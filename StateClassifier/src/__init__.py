#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经图像处理状态分类器模块

该包提供了从MATLAB转换而来的神经图像处理功能，包括：
- 数据格式转换
- 时间延迟互信息计算
- 相空间重构
- 细胞数组裁剪

Author: Converted from MATLAB to Python
"""

from .format_convert import format_convert
from .mutual import mutual, find_optimal_delay
from .phasespace import phasespace, estimate_embedding_params
from .cellset2trim import cellset2trim, cellset2trim_dict, get_dataset_stats, validate_trim_length

__version__ = "1.0.0"
__author__ = "Converted from MATLAB"

__all__ = [
    'format_convert',
    'mutual',
    'find_optimal_delay', 
    'phasespace',
    'estimate_embedding_params',
    'cellset2trim',
    'cellset2trim_dict',
    'get_dataset_stats',
    'validate_trim_length'
] 