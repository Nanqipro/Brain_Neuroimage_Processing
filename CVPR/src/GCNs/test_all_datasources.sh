#!/bin/bash
# 测试所有11个模型在三种数据源上是否正常工作

# 定义所有模型
MODELS=("gcn" "gat" "sage" "hybrid" "gin" "chebnet" "edgeconv" "gunet" "pna" "gatv2" "deepergcn")

# 定义数据集配置
# 格式: "data_source:dataset:batch_size"
DATASETS=(
    "tudataset:MUTAG:32"
    "ogb:ogbg-molhiv:128"
    "tudataset:PROTEINS:32"
)

echo "======================================================================"
echo "测试 run.py 的三种数据源 × 11个模型 = 33项测试"
echo "======================================================================"
echo ""

# 统计变量
TOTAL_TESTS=$((${#DATASETS[@]} * ${#MODELS[@]}))
CURRENT_TEST=0
PASSED=0
FAILED=0
FAILED_TESTS=()

# 开始时间
START_TIME=$(date +%s)

# 遍历所有数据集
for dataset_config in "${DATASETS[@]}"; do
    # 解析数据集配置
    IFS=':' read -r data_source dataset batch_size <<< "$dataset_config"
    
    echo ""
    echo "======================================================================"
    echo "数据集: $data_source - $dataset"
    echo "======================================================================"
    
    # 遍历所有模型
    for model in "${MODELS[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        
        echo ""
        echo "----------------------------------------------------------------------"
        echo "【测试 $CURRENT_TEST/$TOTAL_TESTS】 $dataset + $model"
        echo "----------------------------------------------------------------------"
        
        # 运行测试
        OUTPUT=$(python run.py \
            --data_source $data_source \
            --dataset $dataset \
            --model $model \
            --gpu_id 3 \
            --epochs 1 \
            --batch_size $batch_size \
            --print_every 1 \
            2>&1)
        
        # 检查是否成功
        if echo "$OUTPUT" | grep -q "测试集性能"; then
            echo "✅ 通过"
            echo "$OUTPUT" | grep -A 3 "测试集性能"
            PASSED=$((PASSED + 1))
        else
            echo "❌ 失败"
            echo "$OUTPUT" | tail -20  # 显示最后20行错误信息
            FAILED=$((FAILED + 1))
            FAILED_TESTS+=("$dataset + $model")
        fi
    done
done

# 结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "======================================================================"
echo "所有测试完成！"
echo "======================================================================"
echo "总测试数: $TOTAL_TESTS"
echo "✅ 通过: $PASSED"
echo "❌ 失败: $FAILED"
echo "⏱️  总耗时: ${ELAPSED}秒"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "失败的测试:"
    for failed_test in "${FAILED_TESTS[@]}"; do
        echo "  - $failed_test"
    done
    exit 1
else
    echo ""
    echo "🎉 所有测试均通过！"
    exit 0
fi
