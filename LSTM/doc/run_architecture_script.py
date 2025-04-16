import sys
import os
import subprocess

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # 导入并执行架构图生成函数
    from enhanced_lstm_architecture import create_enhanced_lstm_architecture
    
    print("开始生成增强型LSTM模型架构图...")
    image_path = create_enhanced_lstm_architecture()
    print(f"架构图已成功生成: {image_path}")
    
except Exception as e:
    print(f"生成架构图时出错: {str(e)}")