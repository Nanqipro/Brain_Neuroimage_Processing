#!/bin/bash
# 批量实验运行脚本

echo "开始批量GCN实验..."
echo "当前目录: $(pwd)"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查必要文件
if [ ! -f "run.py" ]; then
    echo "错误: 未找到run.py文件，请确保在GCNs目录下运行"
    exit 1
fi

if [ ! -f "batch_experiments.py" ]; then
    echo "错误: 未找到batch_experiments.py文件"
    exit 1
fi

# 创建结果目录
mkdir -p batch_results

# 运行批量实验
echo "开始运行批量实验..."
python batch_experiments.py "$@"

echo "批量实验完成!"
echo "结果保存在 batch_results/ 目录中"