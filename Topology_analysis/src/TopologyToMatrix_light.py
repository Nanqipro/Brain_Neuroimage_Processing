"""
Neuron Activity State Analysis Module

This module analyzes the activation states of neurons by:
1. Reading calcium concentration data from Excel files
2. Computing average calcium concentrations for each neuron
3. Classifying neurons into ON/OFF states (1/0)
4. Saving the activation states matrix to Excel
"""

import pandas as pd
import logging
import re
from typing import List, Tuple
from dataclasses import dataclass
import os

# ===================== 配置部分 =====================
# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets')

# 文件路径配置
DEFAULT_INPUT_FILES = {
    'day3': os.path.join(DATA_DIR, 'Day3_with_behavior_labels_filled.xlsx'),
    'day6': os.path.join(DATA_DIR, 'Day6_with_behavior_labels_filled.xlsx'),
    'day9': os.path.join(DATA_DIR, 'Day9_with_behavior_labels_filled.xlsx')
}

DEFAULT_OUTPUT_FILES = {
    'day3': os.path.join(OUTPUT_DIR, 'Day3_neuron_states.xlsx'),
    'day6': os.path.join(OUTPUT_DIR, 'Day6_neuron_states.xlsx'),
    'day9': os.path.join(OUTPUT_DIR, 'Day9_neuron_states.xlsx')
}
# ===================== 配置结束 =====================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def identify_neuron_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify neuron columns in the DataFrame.
    Assumes neuron columns are named as 'n1', 'n2', etc.

    Args:
        df: DataFrame containing calcium concentration data

    Returns:
        List of column names corresponding to neurons
    """
    neuron_pattern = re.compile(r'^n\d+$')
    neuron_cols = [col for col in df.columns if neuron_pattern.match(str(col))]
    
    if not neuron_cols:
        logging.warning("No neuron columns found in the data")
        return []
    
    # Sort by neuron number
    neuron_cols.sort(key=lambda x: int(x[1:]))
    logging.info(f"Found {len(neuron_cols)} neuron columns")
    return neuron_cols

def read_calcium_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read calcium concentration data from an Excel file and identify neuron columns.

    Args:
        file_path: Path to the Excel file containing calcium data

    Returns:
        Tuple containing:
            - DataFrame containing calcium concentration data
            - List of neuron column names
    """
    try:
        df = pd.read_excel(file_path)
        neuron_cols = identify_neuron_columns(df)
        if not neuron_cols:
            raise ValueError("No neuron columns found in the data")
        logging.info(f"Successfully read file: {file_path}")
        return df, neuron_cols
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise

def compute_average(df: pd.DataFrame, neuron_cols: List[str]) -> pd.Series:
    """
    Compute average calcium concentration for each neuron.

    Args:
        df: DataFrame containing calcium concentration data
        neuron_cols: List of column names corresponding to neurons

    Returns:
        Series containing average values for each neuron
    """
    avg_series = df[neuron_cols].mean(axis=0)
    logging.info("Computed average calcium concentration for each neuron")
    return avg_series

def classify_on_off(df: pd.DataFrame, avg_series: pd.Series, neuron_cols: List[str]) -> pd.DataFrame:
    """
    Classify neuron states as ON/OFF based on their average values.

    Args:
        df: DataFrame containing calcium concentration data
        avg_series: Series containing average values for each neuron
        neuron_cols: List of column names corresponding to neurons

    Returns:
        DataFrame with binary values (1 for ON, 0 for OFF)
    """
    on_off_df = df[neuron_cols] > avg_series
    on_off_df = on_off_df.astype(int)
    logging.info("Classified ON/OFF states")
    return on_off_df

def save_activation_matrix(
    activation_matrix: pd.DataFrame,
    save_path: str
) -> None:
    """
    Save neuron activation matrix to Excel.

    Args:
        activation_matrix: DataFrame containing neuron activation states
        save_path: Path to save the Excel file
    """
    try:
        # Add time stamp column
        activation_matrix = activation_matrix.reset_index()
        activation_matrix.index.name = 'Time_Stamp'
        activation_matrix.index = activation_matrix.index + 1  # Start from 1
        
        # Save to Excel
        activation_matrix.to_excel(save_path, index=True)
        logging.info(f"Saved activation matrix to '{save_path}'")
    except Exception as e:
        logging.error(f"Error saving activation matrix to Excel: {e}")
        raise

def main(
    file_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Execute the complete neuron state analysis workflow.

    Args:
        file_path: Path to the Excel file containing calcium data
        output_path: Path to save the activation states matrix. If None, will generate based on input filename.

    Returns:
        DataFrame containing neuron activation states
    """
    try:
        # 如果没有指定输出路径，根据输入文件名生成输出路径
        if output_path is None:
            input_filename = os.path.basename(file_path)
            day_match = re.search(r'Day(\d+)', input_filename)
            if day_match:
                day_num = day_match.group(1)
                output_path = os.path.join(OUTPUT_DIR, f'Day{day_num}_neuron_states.xlsx')
            else:
                output_path = os.path.join(OUTPUT_DIR, 'neuron_states.xlsx')
        
        # Read and process data
        df, neuron_cols = read_calcium_data(file_path)
        avg_series = compute_average(df, neuron_cols)
        activation_matrix = classify_on_off(df, avg_series, neuron_cols)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        save_activation_matrix(activation_matrix, output_path)
        
        return activation_matrix
    except Exception as e:
        logging.error(f"Error in main workflow: {e}")
        raise

if __name__ == "__main__":
    try:
        # 使用配置中的默认路径
        excel_file = DEFAULT_INPUT_FILES['day3']  # 可以改为 'day3' 或 'day6'
        activation_matrix = main(
            file_path=excel_file,
            output_path=DEFAULT_OUTPUT_FILES['day3']  # 可选参数，如果不指定会自动生成
        )
    except Exception as e:
        logging.error(f"Program execution failed: {e}")
        raise