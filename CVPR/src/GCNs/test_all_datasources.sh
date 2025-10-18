#!/bin/bash
# 测试所有三种数据源是否正常工作

echo "======================================================================"
echo "测试 run.py 的三种数据源支持"
echo "======================================================================"
echo ""

# 测试 1: TUDataset
echo "【测试 1/3】 TUDataset - MUTAG"
echo "----------------------------------------------------------------------"
python run.py \
    --data_source tudataset \
    --dataset MUTAG \
    --model gin \
    --gpu_id 3 \
    --epochs 1 \
    --print_every 1 \
    2>&1 | grep -A 5 "测试集性能"
echo ""

# 测试 2: OGB
echo "【测试 2/3】 OGB - ogbg-molhiv"
echo "----------------------------------------------------------------------"
python run.py \
    --data_source ogb \
    --dataset ogbg-molhiv \
    --model gin \
    --gpu_id 3 \
    --epochs 1 \
    --batch_size 128 \
    --print_every 1 \
    2>&1 | grep -A 5 "测试集性能"
echo ""

# 测试 3: TUDataset (另一个数据集)
echo "【测试 3/3】 TUDataset - PROTEINS"
echo "----------------------------------------------------------------------"
python run.py \
    --data_source tudataset \
    --dataset PROTEINS \
    --model gat \
    --gpu_id 3 \
    --epochs 1 \
    --print_every 1 \
    2>&1 | grep -A 5 "测试集性能"
echo ""

echo "======================================================================"
echo "所有测试完成！"
echo "======================================================================"

