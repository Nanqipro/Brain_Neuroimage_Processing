# EMtrace01 分析的配置文件

# 基于效应大小识别关键神经元的阈值
EFFECT_SIZE_THRESHOLD = 0.4407

# 不同行为的颜色配置
BEHAVIOR_COLORS = {
    'Close': 'red',    # "靠近"行为
    'Middle': 'green', # "中间"行为
    'Open': 'blue',     # "打开/远离"行为
}

# 行为对共享神经元的混合颜色配置 (键: 按字母顺序排序的行为名称元组)
MIXED_BEHAVIOR_COLORS = {
    ('Close', 'Middle'): 'yellow',  # Close & Middle 的共享神经元颜色
    ('Close', 'Open'): 'magenta', # Close & Open 的共享神经元颜色
    ('Middle', 'Open'): 'cyan',    # Middle & Open 的共享神经元颜色
    # ('Close', 'Middle', 'Open'): 'lightgray' # 未来若需分析三者共享，可启用此颜色
}

# 目标神经元数量 (供参考，在阈值建议功能中使用过)
TARGET_MIN_NEURONS = 5
TARGET_MAX_NEURONS = 10 