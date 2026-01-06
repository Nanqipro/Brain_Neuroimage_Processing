#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

GPU_ID="${GPU_ID:-0}"
DEFAULT_MODEL_PATH="${DEFAULT_MODEL_PATH:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/example/run_20_seed61/best_model.pth}"

MODEL_29800930_T0="${MODEL_29800930_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/no.29800930openfield_CellVideo0_corrected_0_cell_trace_20260104_220849_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_29800930_T05="${MODEL_29800930_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/no.29800930openfield_CellVideo0_corrected_0_cell_trace_20260104_220849_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_29800930_T05_LT="${MODEL_29800930_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/no.29800930openfield_CellVideo0_corrected_0_cell_trace_20260104_220849_阈值0.5_反向/run_20_seed61/best_model.pth}"

MODEL_2980240924_T0="${MODEL_2980240924_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/no.2980240924openfield_CellVideo0_corrected_0_cell_trace_20260104_220849_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_2980240924_T05="${MODEL_2980240924_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/no.2980240924openfield_CellVideo0_corrected_0_cell_trace_20260104_220849_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_2980240924_T05_LT="${MODEL_2980240924_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/no.2980240924openfield_CellVideo0_corrected_0_cell_trace_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_5355_T0="${MODEL_5355_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/NO5355EM20251106_cell_trace_20260105_012729_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_5355_T05="${MODEL_5355_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/NO5355EM20251106_cell_trace_20260105_012729_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_5355_T05_LT="${MODEL_5355_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/NO5355EM20251106_cell_trace_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_5355_3REGION_T0="${MODEL_5355_3REGION_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/NO5355EM20251106_cell_trace-三区域_20260105_012729_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_5355_3REGION_T05="${MODEL_5355_3REGION_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/NO5355EM20251106_cell_trace-三区域_20260105_012729_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_5355_3REGION_T05_LT="${MODEL_5355_3REGION_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/NO5355EM20251106_cell_trace-三区域_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_6250_T0="${MODEL_6250_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/bla6250EM0626goodtrace_20260105_071334_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_6250_T05="${MODEL_6250_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/bla6250EM0626goodtrace_20260105_071334_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_6250_T05_LT="${MODEL_6250_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/bla6250EM0626goodtrace_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_6250_PLUS_T0="${MODEL_6250_PLUS_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/bla6250EM0626goodtrace_plus_20260105_071334_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_6250_PLUS_T05="${MODEL_6250_PLUS_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/bla6250EM0626goodtrace_plus_20260105_071334_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_6250_PLUS_T05_LT="${MODEL_6250_PLUS_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/bla6250EM0626goodtrace_plus_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_EMTRACE01_T0="${MODEL_EMTRACE01_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace01_20260105_095431_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_EMTRACE01_T05="${MODEL_EMTRACE01_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace01_20260105_095431_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_EMTRACE01_T05_LT="${MODEL_EMTRACE01_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace01_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_EMTRACE02_T0="${MODEL_EMTRACE02_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace02_20260105_095431_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_EMTRACE02_T05="${MODEL_EMTRACE02_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace02_20260105_095431_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_EMTRACE02_T05_LT="${MODEL_EMTRACE02_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace02_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_EMTRACE01_PLUS_T0="${MODEL_EMTRACE01_PLUS_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace01_plus_20260105_140711_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_EMTRACE01_PLUS_T05="${MODEL_EMTRACE01_PLUS_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace01_plus_20260105_140710_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_EMTRACE01_PLUS_T05_LT="${MODEL_EMTRACE01_PLUS_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace01_plus_thr0p5_lt/run_20_seed61/best_model.pth}"

MODEL_EMTRACE02_PLUS_T0="${MODEL_EMTRACE02_PLUS_T0:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace02_plus_20260105_140711_阈值0.0/run_20_seed61/best_model.pth}"
MODEL_EMTRACE02_PLUS_T05="${MODEL_EMTRACE02_PLUS_T05:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace02_plus_20260105_140711_阈值0.5/run_20_seed61/best_model.pth}"
MODEL_EMTRACE02_PLUS_T05_LT="${MODEL_EMTRACE02_PLUS_T05_LT:-/home/nanqipro01/gitlocal/Brain_Neuroimage_Processing/bettergcn/results/EMtrace02_plus_thr0p5_lt/run_20_seed61/best_model.pth}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="../eval_results_batch/${RUN_TS}"

COMMON_ARGS=(
  --seed 42
  --train_ratio 0.6
  --val_ratio 0.2
  --gpu_id "${GPU_ID}"
  --balance_strategy comprehensive
)

PIDS=()

run_eval_three() {
  local name="$1"
  local data_file="$2"
  local position_file="$3"
  local min_samples="$4"
  local effect_size_file="$5"
  local model_t0="$6"
  local model_t05="$7"
  local model_t05_lt="$8"

  python eval_saved_model.py \
    --data_file "$data_file" \
    --position_file "$position_file" \
    --min_samples "$min_samples" \
    --effect_size_file "$effect_size_file" \
    --output_dir "${OUT_ROOT}/${name}/thr0p0_gt" \
    --model_path "$model_t0" \
    --eval_modes single \
    --effect_threshold 0.0 \
    --effect_filter_mode gt \
    "${COMMON_ARGS[@]}"

  python eval_saved_model.py \
    --data_file "$data_file" \
    --position_file "$position_file" \
    --min_samples "$min_samples" \
    --effect_size_file "$effect_size_file" \
    --output_dir "${OUT_ROOT}/${name}/thr0p5_gt" \
    --model_path "$model_t05" \
    --eval_modes single \
    --effect_threshold 0.5 \
    --effect_filter_mode gt \
    "${COMMON_ARGS[@]}"

  python eval_saved_model.py \
    --data_file "$data_file" \
    --position_file "$position_file" \
    --min_samples "$min_samples" \
    --effect_size_file "$effect_size_file" \
    --output_dir "${OUT_ROOT}/${name}/thr0p5_lt" \
    --model_path "$model_t05_lt" \
    --eval_modes single \
    --effect_threshold 0.5 \
    --effect_filter_mode lt \
    "${COMMON_ARGS[@]}"
}


# 2980
run_eval_three \
  "no.29800930openfield_CellVideo0_corrected_0_cell_trace" \
  "../datasets/no.29800930openfield_CellVideo0_corrected_0_cell_trace.xlsx" \
  "../datasets/no.29800930openfield神经元编号位置图.csv" \
  "50" \
  "../datasets/effect_sizes_no.29800930openfield_CellVideo0_corrected_0_cell_trace.csv" \
  "${MODEL_29800930_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_29800930_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_29800930_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

run_eval_three \
  "no.2980240924openfield_CellVideo0_corrected_0_cell_trace" \
  "../datasets/no.2980240924openfield_CellVideo0_corrected_0_cell_trace.xlsx" \
  "../datasets/no.2980240924openfield神经元编号位置图.csv" \
  "50" \
  "../datasets/effect_sizes_no.2980240924openfield_CellVideo0_corrected_0_cell_trace.csv" \
  "${MODEL_2980240924_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_2980240924_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_2980240924_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
PIDS=()


# 5355

run_eval_three \
  "NO5355EM20251106_cell_trace" \
  "../datasets/NO5355EM20251106_cell_trace.xlsx" \
  "../datasets/NO5355EM20251106_cell_trace.csv" \
  "50" \
  "../datasets/effect_sizes_NO5355EM20251106_cell_trace.csv" \
  "${MODEL_5355_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_5355_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_5355_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

run_eval_three \
  "NO5355EM20251106_cell_trace-三区域" \
  "../datasets/NO5355EM20251106_cell_trace-三区域.xlsx" \
  "../datasets/NO5355EM20251106_cell_trace.csv" \
  "50" \
  "../datasets/effect_sizes_NO5355EM20251106_cell_trace-三区域.csv" \
  "${MODEL_5355_3REGION_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_5355_3REGION_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_5355_3REGION_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
PIDS=()

# 6250
run_eval_three \
  "bla6250EM0626goodtrace" \
  "../datasets/bla6250EM0626goodtrace.xlsx" \
  "../datasets/6250_Max_position.csv" \
  "50" \
  "../datasets/effect_sizes_bla6250EM0626goodtrace.csv" \
  "${MODEL_6250_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_6250_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_6250_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

run_eval_three \
  "bla6250EM0626goodtrace_plus" \
  "../datasets/bla6250EM0626goodtrace_plus.xlsx" \
  "../datasets/6250_Max_position.csv" \
  "50" \
  "../datasets/effect_sizes_bla6250EM0626goodtrace_plus.csv" \
  "${MODEL_6250_PLUS_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_6250_PLUS_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_6250_PLUS_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
PIDS=()

# 2979
run_eval_three \
  "EMtrace01" \
  "../datasets/EMtrace01.xlsx" \
  "../datasets/EMtrace01_Max_position.csv" \
  "50" \
  "../datasets/effect_sizes_EMtrace01.csv" \
  "${MODEL_EMTRACE01_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE01_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE01_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

run_eval_three \
  "EMtrace02" \
  "../datasets/EMtrace02.xlsx" \
  "../datasets/EMtrace02_Max_position.csv" \
  "50" \
  "../datasets/effect_sizes_EMtrace02.csv" \
  "${MODEL_EMTRACE02_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE02_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE02_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

for pid in "${PIDS[@]}"; do
  wait "$pid"
done
PIDS=()

run_eval_three \
  "EMtrace01_plus" \
  "../datasets/EMtrace01_plus.xlsx" \
  "../datasets/EMtrace01_Max_position.csv" \
  "50" \
  "../datasets/effect_sizes_EMtrace01_plus.csv" \
  "${MODEL_EMTRACE01_PLUS_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE01_PLUS_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE01_PLUS_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

run_eval_three \
  "EMtrace02_plus" \
  "../datasets/EMtrace02_plus.xlsx" \
  "../datasets/EMtrace02_Max_position.csv" \
  "50" \
  "../datasets/effect_sizes_EMtrace02_plus.csv" \
  "${MODEL_EMTRACE02_PLUS_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE02_PLUS_T0:-$DEFAULT_MODEL_PATH}" \
  "${MODEL_EMTRACE02_PLUS_T0:-$DEFAULT_MODEL_PATH}" & PIDS+=($!)

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "Batch evaluation done. Results in: ${OUT_ROOT}"
