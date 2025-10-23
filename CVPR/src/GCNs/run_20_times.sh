#!/bin/bash

# ============================================================================
# 20次重复训练脚本
# 数据集划分比例: 训练集60%, 验证集20%, 测试集20%
# 每次训练使用不同的seed，确保数据划分不同
# ============================================================================

echo "========================================"
echo "20次重复训练脚本"
echo "数据集划分: 训练集60% | 验证集20% | 测试集20%"
echo "========================================"

# 默认参数（可根据需要修改）
DATA_SOURCE="tudataset"     # 数据源: csv, tudataset, ogb
DATASET="MUTAG"             # 数据集名称
MODEL="gcn"                 # 模型: gcn, gat, gin, etc.
SEED=42                     # 初始seed
NUM_RUNS=20                 # 训练次数
GPU_ID=3                    # GPU ID
EPOCHS=200                  # 训练轮数
BATCH_SIZE=32               # 批大小

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "训练配置："
echo "  - 数据源: $DATA_SOURCE"
echo "  - 数据集: $DATASET"
echo "  - 模型: $MODEL"
echo "  - 初始Seed: $SEED"
echo "  - 训练次数: $NUM_RUNS"
echo "  - GPU ID: $GPU_ID"
echo "  - 最大训练轮数: $EPOCHS"
echo "  - 批大小: $BATCH_SIZE"
echo "  - 数据划分: 训练60%, 验证20%, 测试20%"
echo ""

# 运行训练
python run.py \
    --data_source $DATA_SOURCE \
    --dataset $DATASET \
    --model $MODEL \
    --num_runs $NUM_RUNS \
    --seed $SEED \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --gpu_id $GPU_ID \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --save_results

echo ""
echo "训练完成！"
echo "========================================"

