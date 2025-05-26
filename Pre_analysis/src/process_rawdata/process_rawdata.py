import pandas as pd
import numpy as np
import os

def process_data(input_file_path: str, output_file_path: str = None) -> pd.DataFrame:
    """
    处理原始数据，确保FrameLost列每行都有对应标签
    
    Parameters
    ----------
    input_file_path : str
        输入Excel文件路径
    output_file_path : str, optional
        输出Excel文件路径，默认为None
        
    Returns
    -------
    pd.DataFrame
        处理后的数据框
    """
    # 加载原始数据
    data = pd.read_excel(input_file_path, sheet_name='Sheet1')
    
    # 确保数据列存在
    required_columns = ['stamp', 'FrameLost']
    for col in required_columns:
        if col not in data.columns:
            # 如果列不存在，创建空列
            data[col] = np.nan
            print(f"警告: 创建了缺失的列 '{col}'")
    
    # 将空字符串替换为NaN以便正确填充
    data['FrameLost'] = data['FrameLost'].replace('', np.nan)
    # 使用前向填充填充空值
    data['FrameLost'] = data['FrameLost'].ffill()
    
    # 将所有仍为NaN的值替换为'NULL'字符串
    data = data.fillna('NULL')
    
    # 如果提供了输出路径，保存到Excel
    if output_file_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        data.to_excel(output_file_path, index=False)
        print(f"文件已保存为: {output_file_path}")
    
    return data

if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_path = '../../raw_data/Day3EMtrace.xlsx'
    output_path = '../../processed_data/Day3EMtrace_processed.xlsx'
    
    # 处理数据
    processed_data = process_data(input_path, output_path)
    
    # 打印处理结果统计信息
    print(f"处理完成，共处理 {len(processed_data)} 行数据")
    print(f"FrameLost列非空值数量: {processed_data['FrameLost'].ne('NULL').sum()}")