# GNN依赖安装指南

## 简介

本项目的GNN分析功能依赖于PyTorch Geometric及其相关组件。正确安装这些依赖项对于启用完整的GNN分析功能至关重要。

## 必需的依赖项

- PyTorch (`torch`)
- PyTorch Geometric (`torch-geometric`)
- Torch Scatter (`torch-scatter`)
- Torch Sparse (`torch-sparse`)

## 安装步骤

### 1. 检查CUDA版本

首先，确认系统的CUDA版本：

```bash
nvidia-smi
# 或
nvcc -V
```

### 2. 根据CUDA版本安装依赖

**CUDA 12.1版本**（适用于DGX工作站）：

```bash
# 安装PyTorch Geometric
pip install torch-geometric

# 安装依赖项（根据您的PyTorch版本和CUDA版本）
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

**CUDA 11.8版本**：

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
```

**CUDA 11.7版本**：

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu117.html
```

**CPU版本**（无GPU）：

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

### 3. 验证安装

安装后，运行以下命令验证依赖项是否正确安装：

```bash
python -c "import torch_geometric; import torch_scatter; import torch_sparse; print('所有GNN依赖项安装成功')"
```

如果无错误输出"所有GNN依赖项安装成功"，则表示安装正确。

### 4. 运行检查脚本

您还可以通过运行项目的依赖检查功能来验证安装：

```bash
python -c "from analysis_results import check_gnn_dependencies; check_gnn_dependencies()"
```

此脚本将检查所有必需的GNN依赖项并报告任何缺失项。

## 故障排除

如果您遇到与CUDA兼容性相关的问题，请尝试：

1. 确保已安装CUDA工具包
2. 确保PyTorch版本与您的CUDA版本兼容
3. 尝试重建虚拟环境并重新安装依赖项

## 参考资料

- PyTorch Geometric官方安装指南：[https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- PyTorch官网：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 