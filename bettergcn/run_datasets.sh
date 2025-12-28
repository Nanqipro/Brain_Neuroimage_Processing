#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

python main.py --data_file ../datasets/no.29800930openfield_CellVideo0_corrected_0_cell_trace.xlsx --position_file ../datasets/no.29800930openfield神经元编号位置图.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_no.29800930openfield_CellVideo0_corrected_0_cell_trace.csv --effect_threshold 0.5

python main.py --data_file ../datasets/no.29800930openfield_CellVideo0_corrected_0_cell_trace.xlsx --position_file ../datasets/no.29800930openfield神经元编号位置图.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_no.29800930openfield_CellVideo0_corrected_0_cell_trace.csv --effect_threshold 0.5 --effect_filter_mode lt

python main.py --data_file ../datasets/no.2980240924openfield_CellVideo0_corrected_0_cell_trace.xlsx --position_file ../datasets/no.2980240924openfield神经元编号位置图.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_no.2980240924openfield_CellVideo0_corrected_0_cell_trace.csv --effect_threshold 0.5

python main.py --data_file ../datasets/no.2980240924openfield_CellVideo0_corrected_0_cell_trace.xlsx --position_file ../datasets/no.2980240924openfield神经元编号位置图.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_no.2980240924openfield_CellVideo0_corrected_0_cell_trace.csv --effect_threshold 0.5 --effect_filter_mode lt

python main.py --data_file ../datasets/NO5355EM20251106_cell_trace.xlsx --position_file ../datasets/NO5355EM20251106_cell_trace.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_NO5355EM20251106_cell_trace.csv --effect_threshold 0.5

python main.py --data_file ../datasets/NO5355EM20251106_cell_trace.xlsx --position_file ../datasets/NO5355EM20251106_cell_trace.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_NO5355EM20251106_cell_trace.csv --effect_threshold 0.5 --effect_filter_mode lt

python main.py --data_file ../datasets/NO5355EM20251106_cell_trace-三区域.xlsx --position_file ../datasets/NO5355EM20251106_cell_trace.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_NO5355EM20251106_cell_trace-三区域.csv --effect_threshold 0.5

python main.py --data_file ../datasets/NO5355EM20251106_cell_trace-三区域.xlsx --position_file ../datasets/NO5355EM20251106_cell_trace.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_NO5355EM20251106_cell_trace-三区域.csv --effect_threshold 0.5 --effect_filter_mode lt

python main.py --data_file ../datasets/no.29800930openfield_CellVideo0_corrected_0_cell_trace.xlsx --position_file ../datasets/no.29800930openfield神经元编号位置图.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_no.29800930openfield_CellVideo0_corrected_0_cell_trace.csv --effect_threshold 0.0

python main.py --data_file ../datasets/no.2980240924openfield_CellVideo0_corrected_0_cell_trace.xlsx --position_file ../datasets/no.2980240924openfield神经元编号位置图.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_no.2980240924openfield_CellVideo0_corrected_0_cell_trace.csv --effect_threshold 0.0

python main.py --data_file ../datasets/NO5355EM20251106_cell_trace.xlsx --position_file ../datasets/NO5355EM20251106_cell_trace.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_NO5355EM20251106_cell_trace.csv --effect_threshold 0.0

python main.py --data_file ../datasets/NO5355EM20251106_cell_trace-三区域.xlsx --position_file ../datasets/NO5355EM20251106_cell_trace.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_NO5355EM20251106_cell_trace-三区域.csv --effect_threshold 0.0


# 6250
python main.py --data_file ../datasets/bla6250EM0626goodtrace.xlsx --position_file ../datasets/6250_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_bla6250EM0626goodtrace.csv --effect_threshold 0.0

python main.py --data_file ../datasets/bla6250EM0626goodtrace_plus.xlsx --position_file ../datasets/6250_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_bla6250EM0626goodtrace_plus.csv --effect_threshold 0.0

python main.py --data_file ../datasets/bla6250EM0626goodtrace.xlsx --position_file ../datasets/6250_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_bla6250EM0626goodtrace.csv --effect_threshold 0.5

python main.py --data_file ../datasets/bla6250EM0626goodtrace.xlsx --position_file ../datasets/6250_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_bla6250EM0626goodtrace.csv --effect_threshold 0.5 --effect_filter_mode lt

python main.py --data_file ../datasets/bla6250EM0626goodtrace_plus.xlsx --position_file ../datasets/6250_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_bla6250EM0626goodtrace_plus.csv --effect_threshold 0.5

python main.py --data_file ../datasets/bla6250EM0626goodtrace_plus.xlsx --position_file ../datasets/6250_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_bla6250EM0626goodtrace_plus.csv --effect_threshold 0.5 --effect_filter_mode lt

# 2979 多标签
python main.py --data_file ../datasets/EMtrace01.xlsx --position_file ../datasets/EMtrace01_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace01.csv --effect_threshold 0.0

python main.py --data_file ../datasets/EMtrace02.xlsx --position_file ../datasets/EMtrace02_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace02.csv --effect_threshold 0.0

python main.py --data_file ../datasets/EMtrace01.xlsx --position_file ../datasets/EMtrace01_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace01.csv --effect_threshold 0.5

python main.py --data_file ../datasets/EMtrace01.xlsx --position_file ../datasets/EMtrace01_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace01.csv --effect_threshold 0.5 --effect_filter_mode lt

python main.py --data_file ../datasets/EMtrace02.xlsx --position_file ../datasets/EMtrace02_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace02.csv --effect_threshold 0.5

python main.py --data_file ../datasets/EMtrace02.xlsx --position_file ../datasets/EMtrace02_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace02.csv --effect_threshold 0.5 --effect_filter_mode lt

# 2979 单标签
python main.py --data_file ../datasets/EMtrace01_plus.xlsx --position_file ../datasets/EMtrace01_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace01_plus.csv --effect_threshold 0.0

python main.py --data_file ../datasets/EMtrace02_plus.xlsx --position_file ../datasets/EMtrace02_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace02_plus.csv --effect_threshold 0.0

python main.py --data_file ../datasets/EMtrace01_plus.xlsx --position_file ../datasets/EMtrace01_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace01_plus.csv --effect_threshold 0.5

python main.py --data_file ../datasets/EMtrace01_plus.xlsx --position_file ../datasets/EMtrace01_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace01_plus.csv --effect_threshold 0.5 --effect_filter_mode lt

python main.py --data_file ../datasets/EMtrace02_plus.xlsx --position_file ../datasets/EMtrace02_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace02_plus.csv --effect_threshold 0.5

python main.py --data_file ../datasets/EMtrace02_plus.xlsx --position_file ../datasets/EMtrace02_Max_position.csv --min_samples 50 --effect_size_file ../datasets/effect_sizes_EMtrace02_plus.csv --effect_threshold 0.5 --effect_filter_mode lt
