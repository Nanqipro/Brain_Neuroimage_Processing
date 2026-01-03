#!/bin/bash
# 运行所有模型在所有数据集上的8次训练实验
# 数据划分: 训练60%, 验证20%, 测试20%
# 每次训练seed递增，确保数据划分不同
# 并行执行：5个数据集在4个GPU上运行

# 定义所有模型
# MODELS=("gcn" "gat" "sage" "gin" "chebnet" "edgeconv"  "pna" "gatv2" "deepergcn")
MODELS=("gatv2" "deepergcn")
# "gunet"  "hybrid" 
# 定义数据集配置
# 格式: "data_source:dataset:batch_size:epochs:gpu_id"
DATASETS=(
    # "tudataset:MUTAG:32:300:0"                                                      # GPU 0
    # "ogb:ogbg-molhiv:128:300:1"                                                     # GPU 1
    # "tudataset:PROTEINS:32:300:0"                                                   # GPU 2
    # "custom:../../data/random/graphs_100:32:300:0"                                  # GPU 0 - Random 100图 (并行)
    # "custom:../../data/random/graphs_1000:32:300:0"                                 # GPU 1 - Random 1000图 (并行)
    # "custom:../../data/random/graphs_10000:64:300:1"                                # GPU 2 - Random 10000图 (并行)
    # "custom:../../data/random/graphs_100000:128:300:2"                              # GPU 3 - Random 100000图 (并行)
    # "custom:../../data/random/graphs_1000000:128:300:3"                             # GPU 3 - Random 1000000图 (顺序执行)
    "ogb:ogbg-ppa:256:300:1"                                                      # GPU 3 (备用)
)

# 训练设置
NUM_RUNS=8           # 每个模型-数据集组合训练8次
INITIAL_SEED=42      # 初始seed

# 创建统一的日志目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="multi_run_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 创建总日志文件
MAIN_LOG="$LOG_DIR/experiment_summary.log"

echo "======================================================================"
echo "🚀 运行多次训练实验 (并行模式，8次重复)"
echo "======================================================================"
echo "配置: ${#DATASETS[@]}个数据集 × ${#MODELS[@]}个模型 × ${NUM_RUNS}次训练"
echo "数据划分: 训练60%, 验证20%, 测试20%"
echo "并行策略: ${#DATASETS[@]}个数据集在4个GPU上并行运行"
echo ""
echo "数据集分配策略:"
echo "  - GPU 0: graphs_100 + graphs_1000 (两个小数据集并行)"
echo "  - GPU 1: graphs_10000 (中等数据集)"
echo "  - GPU 2: graphs_100000 (大数据集)"
echo "  - GPU 3: graphs_1000000 (超大数据集)"
echo ""
echo "日志目录: $LOG_DIR"
echo "======================================================================"
echo ""

# 同时输出到终端和日志文件
echo "======================================================================"  | tee -a "$MAIN_LOG"
echo "🚀 多次训练实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
echo "配置: ${#DATASETS[@]}个数据集 × ${#MODELS[@]}个模型 × ${NUM_RUNS}次训练" | tee -a "$MAIN_LOG"
echo "数据划分: 训练60%, 验证20%, 测试20%" | tee -a "$MAIN_LOG"
echo "⚡ 并行模式: ${#DATASETS[@]}个数据集在4个GPU上并行运行" | tee -a "$MAIN_LOG"
echo "  - GPU 0: graphs_100 + graphs_1000 (并行)" | tee -a "$MAIN_LOG"
echo "  - GPU 1: graphs_10000" | tee -a "$MAIN_LOG"
echo "  - GPU 2: graphs_100000" | tee -a "$MAIN_LOG"
echo "  - GPU 3: graphs_1000000" | tee -a "$MAIN_LOG"
echo "======================================================================"  | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 统计变量
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#MODELS[@]}))

# 开始时间
START_TIME=$(date +%s)

# 用于存储每个数据集的统计信息
declare -A DATASET_COMPLETED
declare -A DATASET_FAILED
declare -A DATASET_FAILED_EXPS

# 定义单个数据集的训练函数
run_dataset_experiments() {
    local dataset_config=$1
    local dataset_index=$2
    
    # 解析数据集配置
    IFS=':' read -r data_source dataset batch_size epochs gpu_id <<< "$dataset_config"
    
    # 获取数据集名称（处理路径）
    local dataset_name=$(basename "$dataset")
    
    # 为该数据集创建专属日志
    local DATASET_LOG="$LOG_DIR/${dataset_name}_dataset.log"
    
    echo "======================================================================" | tee -a "$DATASET_LOG"
    echo "📊 数据集: $data_source - $dataset" | tee -a "$DATASET_LOG"
    echo "🎮 GPU: cuda:$gpu_id" | tee -a "$DATASET_LOG"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$DATASET_LOG"
    echo "======================================================================" | tee -a "$DATASET_LOG"
    echo "" | tee -a "$DATASET_LOG"
    
    local completed=0
    local failed=0
    local failed_exps=()
    
    # 遍历所有模型
    for model in "${MODELS[@]}"; do
        echo "" | tee -a "$DATASET_LOG"
        echo "----------------------------------------------------------------------" | tee -a "$DATASET_LOG"
        echo "🔧 模型: $model (${NUM_RUNS}次训练)" | tee -a "$DATASET_LOG"
        echo "配置: epochs=$epochs, batch_size=$batch_size, gpu=$gpu_id" | tee -a "$DATASET_LOG"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$DATASET_LOG"
        echo "----------------------------------------------------------------------" | tee -a "$DATASET_LOG"
        
        # 记录单个实验开始时间
        local exp_start=$(date +%s)
        
        # 为每个实验创建独立的日志文件
        local exp_log="$LOG_DIR/${dataset_name}_${model}_multi.log"
        
        # 运行多次训练实验并保存日志
        python run.py \
            --data_source $data_source \
            --dataset $dataset \
            --model $model \
            --num_runs $NUM_RUNS \
            --seed $INITIAL_SEED \
            --train_ratio 0.6 \
            --val_ratio 0.2 \
            --gpu_id $gpu_id \
            --epochs $epochs \
            --batch_size $batch_size \
            --hidden_dim 64 \
            --dropout 0.5 \
            --patience 50 \
            --print_every 20 \
            --save_results \
            2>&1 | tee "$exp_log"
        
        # 检查退出状态
        local exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            local exp_end=$(date +%s)
            local exp_time=$((exp_end - exp_start))
            local hours=$((exp_time / 3600))
            local minutes=$(((exp_time % 3600) / 60))
            local seconds=$((exp_time % 60))
            
            # 从日志中提取关键指标
            echo "" | tee -a "$DATASET_LOG"
            echo "📊 性能指标:" | tee -a "$DATASET_LOG"
            
            # 提取测试集性能指标
            local test_acc=$(grep "准确率:" "$exp_log" | tail -1 | awk '{print $3}')
            local test_f1=$(grep "F1分数:" "$exp_log" | tail -1 | awk '{print $3}')
            local test_auc=$(grep "AUC-ROC:" "$exp_log" | tail -1 | awk '{print $3}')
            
            # 提取训练时间统计
            local total_train_time=$(grep "总训练时间（仅训练集）:" "$exp_log" | tail -1 | awk '{print $3}' | sed 's/秒//')
            local avg_train_time=$(grep "平均训练时间（每epoch，仅训练集）:" "$exp_log" | tail -1 | awk '{print $3}' | sed 's/秒//')
            
            # 提取推理时间统计
            local test_infer_time=$(grep "测试集推理时间:" "$exp_log" | tail -1 | awk '{print $3}' | sed 's/秒//')
            local test_infer_per_graph=$(grep "测试集每个图推理时间:" "$exp_log" | tail -1 | awk '{print $3}' | sed 's/毫秒\/图//')
            
            # 打印提取的指标
            [ -n "$test_acc" ] && echo "  - 测试准确率: $test_acc" | tee -a "$DATASET_LOG"
            [ -n "$test_f1" ] && echo "  - 测试F1: $test_f1" | tee -a "$DATASET_LOG"
            [ -n "$test_auc" ] && echo "  - 测试AUC-ROC: $test_auc" | tee -a "$DATASET_LOG"
            [ -n "$total_train_time" ] && echo "  - 总训练时间: ${total_train_time}秒" | tee -a "$DATASET_LOG"
            [ -n "$avg_train_time" ] && echo "  - 平均每epoch训练时间: ${avg_train_time}秒" | tee -a "$DATASET_LOG"
            [ -n "$test_infer_time" ] && echo "  - 测试集推理时间: ${test_infer_time}秒" | tee -a "$DATASET_LOG"
            [ -n "$test_infer_per_graph" ] && echo "  - 每个图推理时间: ${test_infer_per_graph}ms" | tee -a "$DATASET_LOG"
            
            echo "✅ 完成 (耗时: ${hours}h ${minutes}m ${seconds}s)" | tee -a "$DATASET_LOG"
            echo "日志: $exp_log" | tee -a "$DATASET_LOG"
            ((completed++))
        else
            local exp_end=$(date +%s)
            local exp_time=$((exp_end - exp_start))
            echo "❌ 失败 (耗时: ${exp_time}秒)" | tee -a "$DATASET_LOG"
            echo "错误日志: $exp_log" | tee -a "$DATASET_LOG"
            ((failed++))
            failed_exps+=("$dataset_name + $model")
        fi
        
        echo "" | tee -a "$DATASET_LOG"
    done
    
    # 数据集完成总结
    local dataset_end=$(date +%s)
    local dataset_time=$((dataset_end - START_TIME))
    local hours=$((dataset_time / 3600))
    local minutes=$(((dataset_time % 3600) / 60))
    local seconds=$((dataset_time % 60))
    
    echo "======================================================================" | tee -a "$DATASET_LOG"
    echo "📊 数据集 $dataset_name 完成！" | tee -a "$DATASET_LOG"
    echo "✅ 成功: $completed" | tee -a "$DATASET_LOG"
    echo "❌ 失败: $failed" | tee -a "$DATASET_LOG"
    echo "⏱️  耗时: ${hours}h ${minutes}m ${seconds}s" | tee -a "$DATASET_LOG"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$DATASET_LOG"
    echo "======================================================================" | tee -a "$DATASET_LOG"
    
    # 保存统计信息到文件（供主进程读取）
    echo "$completed" > "$LOG_DIR/${dataset_name}_completed.txt"
    echo "$failed" > "$LOG_DIR/${dataset_name}_failed.txt"
    if [ $failed -gt 0 ]; then
        printf "%s\n" "${failed_exps[@]}" > "$LOG_DIR/${dataset_name}_failed_exps.txt"
    fi
    
    return $failed
}

# 并行执行所有数据集（每个数据集在不同GPU上）
echo "⚡ 启动并行任务..." | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 存储后台进程PID
PIDS=()
DATASET_NAMES=()

# 启动每个数据集的训练任务
dataset_index=0
for dataset_config in "${DATASETS[@]}"; do
    # 解析数据集名称（仅用于显示）
    IFS=':' read -r data_source dataset batch_size epochs gpu_id <<< "$dataset_config"
    dataset_name=$(basename "$dataset")
    
    echo "🚀 启动数据集 $dataset_name 在 GPU $gpu_id 上..." | tee -a "$MAIN_LOG"
    
    # 在后台运行数据集训练
    run_dataset_experiments "$dataset_config" $dataset_index &
    
    # 记录PID和数据集名称
    PIDS+=($!)
    DATASET_NAMES+=("$dataset_name")
    
    ((dataset_index++))
done

echo "" | tee -a "$MAIN_LOG"
echo "✅ 所有 ${#DATASETS[@]} 个数据集已启动，正在并行训练..." | tee -a "$MAIN_LOG"
echo "⏳ 等待所有任务完成..." | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 等待所有后台任务完成
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    dataset_name=${DATASET_NAMES[$i]}
    echo "⏳ 等待数据集 $dataset_name (PID: $pid) 完成..." | tee -a "$MAIN_LOG"
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 数据集 $dataset_name 成功完成" | tee -a "$MAIN_LOG"
    else
        echo "⚠️  数据集 $dataset_name 有部分失败 (退出码: $exit_code)" | tee -a "$MAIN_LOG"
    fi
done

echo "" | tee -a "$MAIN_LOG"
echo "🎉 所有并行任务已完成！" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 汇总所有数据集的统计信息
TOTAL_COMPLETED=0
TOTAL_FAILED=0
ALL_FAILED_EXPS=()

for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r data_source dataset _ _ _ <<< "$dataset_config"
    dataset_name=$(basename "$dataset")
    
    # 读取统计文件
    if [ -f "$LOG_DIR/${dataset_name}_completed.txt" ]; then
        completed=$(cat "$LOG_DIR/${dataset_name}_completed.txt")
        TOTAL_COMPLETED=$((TOTAL_COMPLETED + completed))
    fi
    
    if [ -f "$LOG_DIR/${dataset_name}_failed.txt" ]; then
        failed=$(cat "$LOG_DIR/${dataset_name}_failed.txt")
        TOTAL_FAILED=$((TOTAL_FAILED + failed))
        
        if [ -f "$LOG_DIR/${dataset_name}_failed_exps.txt" ]; then
            while IFS= read -r line; do
                ALL_FAILED_EXPS+=("$line")
            done < "$LOG_DIR/${dataset_name}_failed_exps.txt"
        fi
    fi
done

# 结束时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "🎊 所有多次训练实验完成！" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
echo "总实验数: $TOTAL_EXPERIMENTS (每个${NUM_RUNS}次训练)" | tee -a "$MAIN_LOG"
echo "并行数据集: ${#DATASETS[@]} 个" | tee -a "$MAIN_LOG"
echo "✅ 成功: $TOTAL_COMPLETED" | tee -a "$MAIN_LOG"
echo "❌ 失败: $TOTAL_FAILED" | tee -a "$MAIN_LOG"
echo "⏱️  总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒 (并行)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 显示每个数据集的统计
echo "📊 各数据集统计:" | tee -a "$MAIN_LOG"
for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r data_source dataset _ _ gpu_id <<< "$dataset_config"
    dataset_name=$(basename "$dataset")
    if [ -f "$LOG_DIR/${dataset_name}_completed.txt" ] && [ -f "$LOG_DIR/${dataset_name}_failed.txt" ]; then
        completed=$(cat "$LOG_DIR/${dataset_name}_completed.txt")
        failed=$(cat "$LOG_DIR/${dataset_name}_failed.txt")
        echo "  - $dataset_name (GPU $gpu_id): ✅ $completed  ❌ $failed" | tee -a "$MAIN_LOG"
    fi
done
echo "" | tee -a "$MAIN_LOG"

if [ $TOTAL_FAILED -gt 0 ]; then
    echo "失败的实验:" | tee -a "$MAIN_LOG"
    for failed_exp in "${ALL_FAILED_EXPS[@]}"; do
        echo "  - $failed_exp" | tee -a "$MAIN_LOG"
    done
    echo "" | tee -a "$MAIN_LOG"
fi

echo "📁 结果保存位置：" | tee -a "$MAIN_LOG"
echo "   - 实验日志: $LOG_DIR/" | tee -a "$MAIN_LOG"
echo "   - 各数据集日志: $LOG_DIR/<dataset>_dataset.log" | tee -a "$MAIN_LOG"
echo "   - 训练结果: result/<data_source>/<dataset>/<model>/" | tee -a "$MAIN_LOG"
echo "   - 训练历史CSV: result/<data_source>/<dataset>/<model>/<timestamp>/training_history.csv" | tee -a "$MAIN_LOG"
echo "   - 多次训练汇总: result/multi_run_summary_*.json" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 生成综合性能报告
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "📈 综合性能报告（基于多次训练汇总）" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"

# 查找并解析所有 multi_run_summary JSON 文件
for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r data_source dataset _ _ _ <<< "$dataset_config"
    dataset_name=$(basename "$dataset")
    
    # 查找最近的汇总文件
    summary_json=$(ls -t result/multi_run_summary_*.json 2>/dev/null | head -1)
    
    if [ -n "$summary_json" ] && [ -f "$summary_json" ]; then
        echo "" | tee -a "$MAIN_LOG"
        echo "数据集: $dataset_name" | tee -a "$MAIN_LOG"
        echo "汇总文件: $summary_json" | tee -a "$MAIN_LOG"
        
        # 提取关键统计数据（需要 jq 工具，如果没有则跳过）
        if command -v jq &> /dev/null; then
            local mean_acc=$(jq -r '.statistics.accuracy.mean' "$summary_json" 2>/dev/null)
            local std_acc=$(jq -r '.statistics.accuracy.std' "$summary_json" 2>/dev/null)
            local mean_f1=$(jq -r '.statistics.f1.mean' "$summary_json" 2>/dev/null)
            local std_f1=$(jq -r '.statistics.f1.std' "$summary_json" 2>/dev/null)
            local mean_auc=$(jq -r '.statistics.auc_roc.mean' "$summary_json" 2>/dev/null)
            local std_auc=$(jq -r '.statistics.auc_roc.std' "$summary_json" 2>/dev/null)
            local mean_train_time=$(jq -r '.statistics.total_training_time.mean' "$summary_json" 2>/dev/null)
            local std_train_time=$(jq -r '.statistics.total_training_time.std' "$summary_json" 2>/dev/null)
            local mean_infer_time=$(jq -r '.statistics.test_inference_time.mean' "$summary_json" 2>/dev/null)
            local std_infer_time=$(jq -r '.statistics.test_inference_time.std' "$summary_json" 2>/dev/null)
            
            [ "$mean_acc" != "null" ] && echo "  - 准确率: ${mean_acc} ± ${std_acc}" | tee -a "$MAIN_LOG"
            [ "$mean_f1" != "null" ] && echo "  - F1分数: ${mean_f1} ± ${std_f1}" | tee -a "$MAIN_LOG"
            [ "$mean_auc" != "null" ] && echo "  - AUC-ROC: ${mean_auc} ± ${std_auc}" | tee -a "$MAIN_LOG"
            [ "$mean_train_time" != "null" ] && echo "  - 训练时间: ${mean_train_time} ± ${std_train_time} 秒" | tee -a "$MAIN_LOG"
            [ "$mean_infer_time" != "null" ] && echo "  - 推理时间: ${mean_infer_time} ± ${std_infer_time} 秒" | tee -a "$MAIN_LOG"
        else
            echo "  (需要 jq 工具来解析 JSON 统计数据)" | tee -a "$MAIN_LOG"
        fi
    fi
done

echo "" | tee -a "$MAIN_LOG"
echo "======================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

if [ $TOTAL_FAILED -gt 0 ]; then
    echo "⚠️  有部分实验失败" | tee -a "$MAIN_LOG"
    exit 1
else
    echo "🎉 所有实验均成功完成！" | tee -a "$MAIN_LOG"
    exit 0
fi

