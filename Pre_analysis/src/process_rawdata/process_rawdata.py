import pandas as pd
import numpy as np
import os

def process_data(input_file_path: str, output_file_path: str = None) -> pd.DataFrame:
    """
    处理原始数据，确保FrameLost列每行都有对应标签
    支持Excel文件(.xlsx, .xls)和CSV文件(.csv)的自动识别
    
    Parameters
    ----------
    input_file_path : str
        输入文件路径，支持Excel或CSV格式
    output_file_path : str, optional
        输出Excel文件路径，默认为None
        
    Returns
    -------
    pd.DataFrame
        处理后的数据框
        
    Raises
    ------
    FileNotFoundError
        当输入文件不存在时抛出
    ValueError
        当文件格式不支持时抛出
    """
    # 检查文件是否存在
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"输入文件不存在: {input_file_path}")
    
    # 获取文件扩展名并转换为小写
    file_extension = os.path.splitext(input_file_path)[1].lower()
    
    # 根据文件扩展名选择合适的读取函数
    try:
        if file_extension == '.csv':
            print(f"正在读取CSV文件: {input_file_path}")
            data = pd.read_csv(input_file_path)
        elif file_extension in ['.xlsx', '.xls']:
            print(f"正在读取Excel文件: {input_file_path}")
            data = pd.read_excel(input_file_path, sheet_name='Sheet1')
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}. 支持的格式: .csv, .xlsx, .xls")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        raise
    
    print(f"成功读取文件，共 {len(data)} 行，{len(data.columns)} 列")
    
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
        output_dir = os.path.dirname(output_file_path)
        if output_dir:  # 只有当输出目录不为空时才创建
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            data.to_excel(output_file_path, index=False)
            print(f"文件已保存为: {output_file_path}")
        except Exception as e:
            print(f"保存文件时发生错误: {e}")
            raise
    
    return data

if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_path = '../../raw_data/bla6250EM0626goodtrace.xlsx'
    output_path = '../../processed_data/bla6250EM0626goodtrace_processed.xlsx'
    
    try:
        # 处理数据
        processed_data = process_data(input_path, output_path)
        
        # 打印处理结果统计信息
        print(f"处理完成，共处理 {len(processed_data)} 行数据")
        print(f"FrameLost列非空值数量: {processed_data['FrameLost'].ne('NULL').sum()}")
        
    except Exception as e:
        print(f"程序执行失败: {e}")