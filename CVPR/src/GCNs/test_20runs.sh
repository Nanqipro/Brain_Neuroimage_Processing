#!/bin/bash

# ============================================================================
# 快速测试脚本 - 验证20次训练功能
# ============================================================================

echo "========================================"
echo "快速测试：3次训练验证"
echo "数据集: MUTAG (小数据集，快速测试)"
echo "模型: GCN"
echo "训练轮数: 20轮 (快速验证)"
echo "========================================"

cd /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/src/GCNs

echo ""
echo "开始测试..."
echo ""

python run.py \
    --data_source tudataset \
    --dataset MUTAG \
    --model gcn \
    --num_runs 3 \
    --seed 42 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --epochs 20 \
    --batch_size 32 \
    --gpu_id 3 \
    --save_results

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
echo ""
echo "请检查以下内容："
echo "1. 是否进行了3次训练（Seed=42, 43, 44）"
echo "2. 数据集划分是否为 6:2:2"
echo "3. 是否显示了汇总结果（平均值±标准差）"
echo "4. 是否保存了汇总JSON文件"
echo ""
echo "如果以上都正常，说明功能运行正确！"
echo "您可以使用以下命令进行完整的20次训练："
echo ""
echo "  ./run_20_times.sh"
echo ""
echo "或者："
echo ""
echo "  python run.py --data_source tudataset --dataset MUTAG --model gcn \\"
echo "                --num_runs 20 --seed 42 --save_results"
echo ""

