import pandas as pd
import numpy as np
from collections import Counter

def process_behavior_data(input_path, output_path):
    # 读取原始数据
    data = pd.read_csv(input_path)
    
    # 定义有效行为列表（仅保留open/close/middle三类）
    valid_behaviors = [
        'Open-Armed-Exp', 'Open-arm-probe', 'Open-arm',
        'Middle-Zone', 'Middle-Zone-stiff',
        'Closed-Armed-Exp', 'Closed-arm', 'Closed-arm-stiff'
    ]
    
    # 过滤无效行为行（非open/close/middle三类）
    filtered_data = data[data['behavior'].isin(valid_behaviors)].copy()
    
    # 定义严格的三类映射
    behavior_mapping = {
        'Open-Armed-Exp': 'open-arm',
        'Open-arm-probe': 'open-arm',
        'Open-arm': 'open-arm',
        'Closed-Armed-Exp': 'close-arm',
        'Closed-arm': 'close-arm',
        'Closed-arm-stiff': 'close-arm',
        'Middle-Zone': 'middle',
        'Middle-Zone-stiff': 'middle'
    }
    
    # 应用映射并过滤可能存在的未知类别
    filtered_data['behavior'] = filtered_data['behavior'].map(behavior_mapping)
    filtered_data = filtered_data.dropna(subset=['behavior'])  # 删除映射失败的行
    
    # 打印处理后的类别统计
    print("最终类别分布:")
    print(Counter(filtered_data['behavior']))
    
    # 保存处理后的数据
    filtered_data.to_csv(output_path, index=False)
    print(f"\n数据已保存至: {output_path}")

if __name__ == "__main__":
    input_file = r"c:\Users\Jim\Desktop\gcn\dataset\Emtrace\EMtrace.csv"
    output_file = r"c:\Users\Jim\Desktop\gcn\dataset\Emtrace\processed.csv"  # 修改输出路径
    process_behavior_data(input_file, output_file)