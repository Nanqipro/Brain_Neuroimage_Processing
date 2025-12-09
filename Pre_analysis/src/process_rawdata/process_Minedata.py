import pandas as pd
import os

def fill_behavior_column(input_path, output_path):
    """
    读取文件（支持csv或xlsx），根据 In, In out, In corner 列的值补齐 behavior 列。
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 '{input_path}'")
        return

    print(f"正在读取文件: {input_path} ...")
    
    try:
        # 根据文件后缀判断读取方式
        if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
            # 读取 Excel 文件
            # 注意: 需要安装 openpyxl (pip install openpyxl)
            df = pd.read_excel(input_path)
        else:
            # 读取 CSV 文件
            # 尝试使用 utf-8 读取，如果失败则尝试 gbk (处理中文常见编码)
            try:
                df = pd.read_csv(input_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8 解码失败，尝试使用 GBK 编码读取...")
                df = pd.read_csv(input_path, encoding='gbk')
        
        # 定义需要检查的列名 (确保与文件中的列名完全一致)
        check_columns = ['In', 'In out', 'In corner']
        
        # 检查这些列是否存在于数据中
        missing_cols = [col for col in check_columns if col not in df.columns]
        if missing_cols:
            print(f"错误: 数据中缺少以下列: {missing_cols}")
            # 打印一下当前的列名，方便调试
            print(f"当前文件包含的列名: {df.columns.tolist()}")
            return

        # 定义一个函数来决定 behavior 的值
        def get_behavior_label(row):
            # 遍历三个状态列
            for col in check_columns:
                # 检查值是否为 1 (兼容整数1, 浮点1.0, 或字符串'1')
                val = row[col]
                # 处理可能的浮点数或字符串情况
                try:
                    if float(val) == 1:
                        return col
                except (ValueError, TypeError):
                    # 如果转换失败（比如是空字符串），检查是否是字符 '1'
                    if str(val).strip() == '1':
                        return col
            
            # 如果三列都不是1，保留原本 behavior 的值
            return row['behavior']

        # 应用逻辑：对每一行 (axis=1) 应用函数
        print("正在处理数据...")
        df['behavior'] = df.apply(get_behavior_label, axis=1)

        # 保存结果到新文件
        # 即使输入是 excel，输出通常存为 csv 比较方便后续处理，或者也可以存为 excel
        if output_path.endswith('.xlsx'):
             df.to_excel(output_path, index=False)
        else:
             df.to_csv(output_path, index=False, encoding='utf-8-sig')
             
        print(f"处理成功! 结果已保存至: {output_path}")

    except Exception as e:
        print(f"发生错误: {e}")
        # 如果提示缺少 openpyxl，提示用户安装
        if "openpyxl" in str(e):
            print("提示: 读取 .xlsx 文件需要安装 openpyxl 库。请运行: pip install openpyxl")

# ==========================================
# 在这里配置你的文件路径
# ==========================================

if __name__ == "__main__":
    # 输入文件的路径
    input_csv_path = '../../raw_data/no.2980240924openfield_CellVideo0_corrected_0_cell_trace.xlsx'
    
    # 输出文件的路径
    output_csv_path = '../../processed_data/no.2980240924openfield_CellVideo0_corrected_0_cell_trace.csv'

    # 执行函数
    fill_behavior_column(input_csv_path, output_csv_path)