#!/bin/bash
# 运行所有11个模型在三种数据集上的完整实验
# 3个数据集 × 11个模型 = 33个实验

# 定义所有模型
MODELS=("gcn" "gat" "sage" "hybrid" "gin" "chebnet" "edgeconv" "gunet" "pna" "gatv2" "deepergcn")

# 定义数据集配置
# 格式: "data_source:dataset:batch_size:epochs"
DATASETS=(
    "tudataset:MUTAG:32:300"
    "ogb:ogbg-molhiv:128:300"
    "tudataset:PROTEINS:32:300"
)

# GPU设置
GPU_ID=3

# 创建统一的日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="experiment_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 创建总日志文件
MAIN_LOG="$LOG_DIR/experiment_summary.log"

echo "======================================================================"
echo "运行完整实验：3个数据集 × 11个模型 = 33个实验"
echo "GPU: cuda:$GPU_ID"
echo "日志目录: $LOG_DIR"
echo "======================================================================"
echo ""

# 同时输出到终端和日志文件
echo "======================================================================"  | tee -a "$MAIN_LOG"
echo "实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
echo "GPU: cuda:$GPU_ID" | tee -a "$MAIN_LOG"
echo "日志目录: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "======================================================================"  | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 统计变量
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#MODELS[@]}))
CURRENT_EXP=0
COMPLETED=0
FAILED=0
FAILED_EXPS=()

# 开始时间
START_TIME=$(date +%s)

# 遍历所有数据集
for dataset_config in "${DATASETS[@]}"; do
    # 解析数据集配置
    IFS=':' read -r data_source dataset batch_size epochs <<< "$dataset_config"
    
    echo "" | tee -a "$MAIN_LOG"
    echo "======================================================================" | tee -a "$MAIN_LOG"
    echo "数据集: $data_source - $dataset" | tee -a "$MAIN_LOG"
    echo "======================================================================" | tee -a "$MAIN_LOG"
    
    # 遍历所有模型
    for model in "${MODELS[@]}"; do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        
        echo "" | tee -a "$MAIN_LOG"
        echo "----------------------------------------------------------------------" | tee -a "$MAIN_LOG"
        echo "【实验 $CURRENT_EXP/$TOTAL_EXPERIMENTS】$dataset + $model" | tee -a "$MAIN_LOG"
        echo "配置: epochs=$epochs, batch_size=$batch_size, gpu=$GPU_ID" | tee -a "$MAIN_LOG"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
        echo "----------------------------------------------------------------------" | tee -a "$MAIN_LOG"
        
        # 记录单个实验开始时间
        EXP_START=$(date +%s)
        
        # 为每个实验创建独立的日志文件
        EXP_LOG="$LOG_DIR/${dataset}_${model}.log"
        
        # 运行实验并保存日志
        python run.py \
            --data_source $data_source \
            --dataset $dataset \
            --model $model \
            --gpu_id $GPU_ID \
            --epochs $epochs \
            --batch_size $batch_size \
            --hidden_dim 64 \
            --dropout 0.5 \
            --print_every 20 \
            --save_results \
            2>&1 | tee "$EXP_LOG"
        
        # 检查退出状态
        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -eq 0 ]; then
            EXP_END=$(date +%s)
            EXP_TIME=$((EXP_END - EXP_START))
            echo "✅ 完成 (耗时: ${EXP_TIME}秒)" | tee -a "$MAIN_LOG"
            echo "日志: $EXP_LOG" | tee -a "$MAIN_LOG"
            COMPLETED=$((COMPLETED + 1))
        else
            EXP_END=$(date +%s)
            EXP_TIME=$((EXP_END - EXP_START))
            echo "❌ 失败 (耗时: ${EXP_TIME}秒)" | tee -a "$MAIN_LOG"
            echo "错误日志: $EXP_LOG" | tee -a "$MAIN_LOG"
            FAILED=$((FAILED + 1))
            FAILED_EXPS+=("$dataset + $model")
        fi
        
        echo "" | tee -a "$MAIN_LOG"
    done
done

# 结束时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "所有实验完成！" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
echo "总实验数: $TOTAL_EXPERIMENTS" | tee -a "$MAIN_LOG"
echo "✅ 成功: $COMPLETED" | tee -a "$MAIN_LOG"
echo "❌ 失败: $FAILED" | tee -a "$MAIN_LOG"
echo "⏱️  总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

if [ $FAILED -gt 0 ]; then
    echo "失败的实验:" | tee -a "$MAIN_LOG"
    for failed_exp in "${FAILED_EXPS[@]}"; do
        echo "  - $failed_exp" | tee -a "$MAIN_LOG"
    done
    echo "" | tee -a "$MAIN_LOG"
    echo "📁 所有日志已保存到: $LOG_DIR" | tee -a "$MAIN_LOG"
    echo "   - 总日志: $MAIN_LOG" | tee -a "$MAIN_LOG"
    echo "   - 各实验日志: $LOG_DIR/<dataset>_<model>.log" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    exit 1
else
    echo "🎉 所有实验均成功完成！" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    echo "📁 结果保存位置：" | tee -a "$MAIN_LOG"
    echo "   - 实验日志: $LOG_DIR/" | tee -a "$MAIN_LOG"
    echo "   - 训练结果: result/tudataset_MUTAG_*/" | tee -a "$MAIN_LOG"
    echo "              result/ogb_ogbg-molhiv_*/" | tee -a "$MAIN_LOG"
    echo "              result/tudataset_PROTEINS_*/" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    exit 0
fi

