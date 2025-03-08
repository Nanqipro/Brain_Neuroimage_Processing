#!/bin/bash

# 安装GNN依赖的脚本
echo "开始安装GNN依赖项..."

# 检测CUDA版本
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | cut -d ' ' -f 3)
    echo "检测到CUDA版本: $CUDA_VERSION"
else
    echo "未检测到NVIDIA GPU或drivers，将安装CPU版本"
    CUDA_VERSION="CPU"
fi

# 检测PyTorch版本
if python -c "import torch; print(torch.__version__)" &> /dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    TORCH_CUDA=$(python -c "import torch; print(torch.__version__.split('+')[1] if '+' in torch.__version__ else 'cpu')")
    echo "检测到PyTorch版本: $TORCH_VERSION, CUDA标识: $TORCH_CUDA"
else
    echo "未检测到PyTorch，请先安装PyTorch"
    exit 1
fi

# 安装PyTorch Geometric
echo "安装PyTorch Geometric..."
pip install torch-geometric
if [ $? -ne 0 ]; then
    echo "PyTorch Geometric安装失败"
    exit 1
fi

# 根据CUDA版本安装兼容的依赖项
if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "为CUDA 12.x安装依赖项..."
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu121.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu121.html
elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
    echo "为CUDA 11.8安装依赖项..."
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu118.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu118.html
elif [[ "$CUDA_VERSION" == "11.7"* ]] || [[ "$CUDA_VERSION" == "11.6"* ]]; then
    echo "为CUDA 11.7/11.6安装依赖项..."
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu117.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu117.html
else
    echo "安装CPU版本依赖项..."
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
fi

# 验证安装
echo "验证安装..."
python -c "import torch_geometric; import torch_scatter; import torch_sparse; print('所有GNN依赖项安装成功')" || { echo "安装验证失败"; exit 1; }

echo "GNN依赖项安装完成！"
echo "运行检查函数验证..."
python -c "from analysis_results import check_gnn_dependencies; status, missing = check_gnn_dependencies(); print(f'验证结果: {\"成功\" if status else \"失败\"}'); print(f'缺失组件: {missing}' if missing else '')" || { echo "运行检查函数失败"; }

echo "现在您可以运行分析脚本使用GNN功能："
echo "python analysis_results.py" 