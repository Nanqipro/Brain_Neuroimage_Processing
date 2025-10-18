#!/bin/bash
# 模型对比脚本 - 在同一数据集上测试多个模型

# 使用说明：
# ./compare_models.sh <dataset_name>
# 例如: ./compare_models.sh MUTAG

DATASET=${1:-MUTAG}

echo "================================================================"
echo "在数据集 $DATASET 上对比不同模型"
echo "================================================================"

# 定义要测试的模型
MODELS=("gcn" "gat" "gin" "sage" "gatv2" "chebnet")

echo ""
echo "将测试以下模型："
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "数据集: $DATASET"
echo ""

read -p "确认开始测试? [y/N]: " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 创建临时配置文件
CONFIG_FILE="temp_model_comparison_$DATASET.json"

cat > "$CONFIG_FILE" << EOF
{
    "description": "在 $DATASET 数据集上对比不同模型",
    "experiments": [
EOF

# 添加每个模型的配置
first=true
for model in "${MODELS[@]}"; do
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "$CONFIG_FILE"
    fi
    
    cat >> "$CONFIG_FILE" << EOF
        {
            "data_source": "tudataset",
            "dataset": "$DATASET",
            "model": "$model",
            "hidden_dim": 64,
            "dropout": 0.5,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0005,
            "epochs": 200,
            "patience": 20,
            "seed": 42,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "print_every": 10
        }
EOF
done

cat >> "$CONFIG_FILE" << EOF

    ]
}
EOF

echo "配置文件已生成: $CONFIG_FILE"
echo ""
echo "开始运行实验..."
echo ""

# 运行实验
python run.py --config "$CONFIG_FILE"

# 清理临时文件
echo ""
read -p "是否删除临时配置文件? [y/N]: " cleanup
if [[ $cleanup =~ ^[Yy]$ ]]; then
    rm "$CONFIG_FILE"
    echo "已删除 $CONFIG_FILE"
fi

echo ""
echo "================================================================"
echo "模型对比完成！"
echo "结果保存在 result/batch_summary/ 目录"
echo "================================================================"

