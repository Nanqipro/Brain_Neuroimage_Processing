# 时间区间设置示例文件
# 此文件展示如何在heatmap_sort-EM.py中设置不同的时间区间

# 示例1：设置0-100秒的时间区间
def example_0_to_100_seconds():
    """
    设置0到100秒的时间区间
    """
    start_seconds = 0.0
    end_seconds = 100.0
    sampling_rate = 4.8  # Hz
    
    STAMP_MIN = start_seconds * sampling_rate  # 0 * 4.8 = 0
    STAMP_MAX = end_seconds * sampling_rate    # 100 * 4.8 = 480
    
    print(f"时间区间: {start_seconds}s - {end_seconds}s")
    print(f"对应时间戳: {STAMP_MIN} - {STAMP_MAX}")
    
    # 在heatmap_sort-EM.py的Config类中设置:
    # STAMP_MIN = 0.0
    # STAMP_MAX = 480.0
    
    return STAMP_MIN, STAMP_MAX

# 示例2：设置50-150秒的时间区间
def example_50_to_150_seconds():
    """
    设置50到150秒的时间区间
    """
    start_seconds = 50.0
    end_seconds = 150.0
    sampling_rate = 4.8  # Hz
    
    STAMP_MIN = start_seconds * sampling_rate  # 50 * 4.8 = 240
    STAMP_MAX = end_seconds * sampling_rate    # 150 * 4.8 = 720
    
    print(f"时间区间: {start_seconds}s - {end_seconds}s")
    print(f"对应时间戳: {STAMP_MIN} - {STAMP_MAX}")
    
    # 在heatmap_sort-EM.py的Config类中设置:
    # STAMP_MIN = 240.0
    # STAMP_MAX = 720.0
    
    return STAMP_MIN, STAMP_MAX

# 示例3：只设置起始时间，不限制结束时间
def example_from_30_seconds():
    """
    从30秒开始，不限制结束时间
    """
    start_seconds = 30.0
    sampling_rate = 4.8  # Hz
    
    STAMP_MIN = start_seconds * sampling_rate  # 30 * 4.8 = 144
    STAMP_MAX = None  # 不限制结束时间
    
    print(f"时间区间: 从{start_seconds}s开始到数据结束")
    print(f"对应时间戳: 从{STAMP_MIN}开始")
    
    # 在heatmap_sort-EM.py的Config类中设置:
    # STAMP_MIN = 144.0
    # STAMP_MAX = None
    
    return STAMP_MIN, STAMP_MAX

# 示例4：只设置结束时间，从数据开始
def example_until_80_seconds():
    """
    从数据开始到80秒结束
    """
    end_seconds = 80.0
    sampling_rate = 4.8  # Hz
    
    STAMP_MIN = None  # 从数据开始
    STAMP_MAX = end_seconds * sampling_rate  # 80 * 4.8 = 384
    
    print(f"时间区间: 从数据开始到{end_seconds}s")
    print(f"对应时间戳: 到{STAMP_MAX}结束")
    
    # 在heatmap_sort-EM.py的Config类中设置:
    # STAMP_MIN = None
    # STAMP_MAX = 384.0
    
    return STAMP_MIN, STAMP_MAX

if __name__ == "__main__":
    print("=== 时间区间设置示例 ===\n")
    
    print("示例1: 0-100秒")
    example_0_to_100_seconds()
    print()
    
    print("示例2: 50-150秒")
    example_50_to_150_seconds()
    print()
    
    print("示例3: 从30秒开始")
    example_from_30_seconds()
    print()
    
    print("示例4: 到80秒结束")
    example_until_80_seconds()
    print()
    
    print("=== 如何在代码中应用 ===")
    print("""
    在heatmap_sort-EM.py文件中，找到Config类，修改以下参数：
    
    class Config:
        # ... 其他参数 ...
        
        # 时间戳区间设置
        STAMP_MIN = 240.0   # 起始时间戳（例如：50秒 * 4.8 = 240）
        STAMP_MAX = 720.0   # 结束时间戳（例如：150秒 * 4.8 = 720）
        
        # 采样率
        SAMPLING_RATE = 4.8  # Hz
        
        # ... 其他参数 ...
    """) 