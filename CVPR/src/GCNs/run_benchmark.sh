#!/bin/bash
# 快速基准测试脚本 - 在常用数据集上测试模型

# 使用说明：
# ./run_benchmark.sh <model_name>
# 例如: ./run_benchmark.sh gcn

MODEL=${1:-gcn}

echo "================================================================"
echo "在标准基准数据集上测试模型: $MODEL"
echo "================================================================"

# 常用的小型数据集（快速测试）
SMALL_DATASETS="MUTAG,ENZYMES,PROTEINS"

# 常用的中型数据集（标准测试）
MEDIUM_DATASETS="MUTAG,ENZYMES,PROTEINS,DD,COLLAB"

# 所有常用数据集（完整测试）
ALL_DATASETS="MUTAG,ENZYMES,PROTEINS,DD,COLLAB,IMDB-BINARY,IMDB-MULTI"

echo ""
echo "请选择测试级别："
echo "1) 小型测试 (MUTAG, ENZYMES, PROTEINS) - 约10-20分钟"
echo "2) 中型测试 (添加DD, COLLAB) - 约30-60分钟"
echo "3) 完整测试 (所有数据集) - 约1-2小时"
echo ""
read -p "请输入选择 [1-3]: " choice

case $choice in
    1)
        DATASETS=$SMALL_DATASETS
        echo "运行小型测试..."
        ;;
    2)
        DATASETS=$MEDIUM_DATASETS
        echo "运行中型测试..."
        ;;
    3)
        DATASETS=$ALL_DATASETS
        echo "运行完整测试..."
        ;;
    *)
        echo "无效选择，使用小型测试"
        DATASETS=$SMALL_DATASETS
        ;;
esac

echo ""
echo "开始在以下数据集上测试模型 $MODEL:"
echo "$DATASETS"
echo ""

python run.py \
    --data_source tudataset \
    --dataset "$DATASETS" \
    --model "$MODEL" \
    --hidden_dim 64 \
    --dropout 0.5 \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 200 \
    --patience 20 \
    --save_results

echo ""
echo "================================================================"
echo "基准测试完成！"
echo "结果保存在 result/summary/$MODEL/ 目录"
echo "================================================================"

