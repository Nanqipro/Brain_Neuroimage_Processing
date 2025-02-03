"""
Time Topology Plus Analysis Module

This module analyzes the temporal topology of neuron activity patterns by:
1. Reading calcium concentration data from Excel files
2. Computing average calcium concentrations for each neuron
3. Classifying neurons into ON/OFF states
4. Generating topology structures and recording edge information
5. Creating and saving topology matrices

The module uses a minimalist approach to topology generation, where:
- The first active neuron becomes the root node
- Each subsequent active neuron connects to the earliest activated neuron
"""

import pandas as pd
import networkx as nx
import os
import logging
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class EdgeRecord:
    """Data class for storing edge information."""
    time_stamp: int
    neuron1: str
    neuron2: str

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

def extract_index(neuron_name: str) -> int:
    """
    Extract the numeric index from a neuron name.

    Args:
        neuron_name: Name of the neuron containing a numeric index

    Returns:
        int: Extracted numeric index, or infinity if no index found
    """
    match = re.search(r'\d+', neuron_name)
    return int(match.group()) if match else float('inf')

def read_calcium_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read calcium concentration data from an Excel file and identify neuron columns.

    Args:
        file_path: Path to the Excel file containing calcium data

    Returns:
        Tuple containing:
            - DataFrame containing calcium concentration data
            - List of neuron column names

    Raises:
        FileNotFoundError: If the specified file does not exist
        Exception: If there's an error reading the file
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

def generate_topologies(on_off_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate topology structures for each timestamp and record edge information.

    Args:
        on_off_df: DataFrame containing ON/OFF states

    Returns:
        List of dictionaries containing edge information
    """
    edge_records = []
    neurons = on_off_df.columns.tolist()
    sorted_neurons = sorted(neurons, key=extract_index)

    for idx, row in on_off_df.iterrows():
        time_stamp = idx + 1
        on_neurons = row[row == 1].index.tolist()
        on_neurons_sorted = sorted(on_neurons, key=extract_index)

        if not on_neurons_sorted:
            continue

        connected_neurons = [on_neurons_sorted[0]]
        for neuron in on_neurons_sorted[1:]:
            edge = (connected_neurons[0], neuron)
            edge_sorted = tuple(sorted(edge, key=extract_index))
            edge_records.append({
                'Time_Stamp': time_stamp,
                'Neuron1': edge_sorted[0],
                'Neuron2': edge_sorted[1]
            })
            connected_neurons.append(neuron)

        if (idx + 1) % 100 == 0:
            logging.info(f"Generated {idx + 1} topology structures")

    logging.info("Completed topology structure generation")
    return edge_records

def save_topology_matrix(
    edge_records: List[Dict[str, Any]],
    save_path: str = '../datasets/Day6_topology_matrix.xlsx'
) -> None:
    """
    Generate topology connection matrix and save to Excel.

    Args:
        edge_records: List of dictionaries containing edge information
        save_path: Path to save the Excel file

    Raises:
        Exception: If there's an error saving the matrix
    """
    try:
        edge_df = pd.DataFrame(edge_records)
        if edge_df.empty:
            logging.warning("Edge records empty, no topology matrix generated.")
            return

        edge_df['Connection'] = edge_df.apply(
            lambda row: f"{row['Neuron1']}_{row['Neuron2']}", axis=1
        )

        unique_connections = edge_df['Connection'].unique()
        unique_connections_sorted = sorted(
            unique_connections,
            key=lambda x: (
                extract_index(x.split('_')[0]),
                extract_index(x.split('_')[1])
            )
        )

        time_stamps = sorted(edge_df['Time_Stamp'].unique())
        topology_matrix = pd.DataFrame(
            0,
            index=time_stamps,
            columns=unique_connections_sorted
        )
        topology_matrix.index.name = 'Time_Stamp'

        for _, row in edge_df.iterrows():
            topology_matrix.at[row['Time_Stamp'], row['Connection']] = 1

        topology_matrix = topology_matrix.reset_index()
        topology_matrix.to_excel(save_path, index=False)
        logging.info(f"Saved topology matrix to '{save_path}'")
    except Exception as e:
        logging.error(f"Error saving topology matrix to Excel: {e}")
        raise

def main(
    file_path: str,
    should_save_topology_matrix: bool = True,
    topology_matrix_path: str = '../datasets/Day6_topology_matrix.xlsx',
    initial_edges: Optional[List[tuple]] = None
) -> List[Dict[str, Any]]:
    """
    Execute the complete topology analysis workflow.

    Args:
        file_path: Path to the Excel file containing calcium data
        should_save_topology_matrix: Whether to save topology matrix to Excel
        topology_matrix_path: Path to save the topology matrix
        initial_edges: Optional list of initial edges defining neuron connections

    Returns:
        List of dictionaries containing edge information

    Raises:
        Exception: If any step in the workflow fails
    """
    try:
        df, neuron_cols = read_calcium_data(file_path)
        avg_series = compute_average(df, neuron_cols)
        on_off_df = classify_on_off(df, avg_series, neuron_cols)
        edge_records = generate_topologies(on_off_df)

        if should_save_topology_matrix:
            save_topology_matrix(edge_records, save_path=topology_matrix_path)

        return edge_records
    except Exception as e:
        logging.error(f"Error in main workflow: {e}")
        raise

if __name__ == "__main__":
    try:
        excel_file = '../datasets/Day6_with_behavior_labels_filled.xlsx'
        edge_records = main(
            file_path=excel_file,
            should_save_topology_matrix=True,
            topology_matrix_path='../datasets/Day6_topology_matrix.xlsx',
            initial_edges=None
        )
    except Exception as e:
        logging.error(f"Program execution failed: {e}")
        raise