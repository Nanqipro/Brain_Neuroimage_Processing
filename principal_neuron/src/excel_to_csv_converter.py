import argparse
from pathlib import Path

import pandas as pd
import numpy as np

def simplify_label(label):
    """Simplifies a behavior label based on keywords."""
    label_lower = str(label).lower()
    if 'open' in label_lower:
        return 'Open'
    elif 'close' in label_lower: # Covers "closed-arm", "closed-armed-exp"
        return 'Close'
    elif 'middle' in label_lower or 'center' in label_lower: # Covers "middle-zone"
        return 'Middle'
    # Add more specific rules if needed, based on unique_labels output
    # For example, if "Stretching" means something specific or should be dropped.
    # For now, return original if no keyword matches, or a placeholder like 'Other'
    return 'Other' # Or np.nan if unclassified should be treated as missing

def process_and_simplify_excel(excel_file_path, output_csv_path, label_column_name='Behavior'):
    """
    Reads an Excel file, simplifies a specified label column including contextual fill for 'Other',
    and saves it to CSV. Also prints column names, unique labels, and info about the simplified labels.
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path)
        print(f"Successfully read '{excel_file_path}'.")
        
        # Print column names
        print("\nColumn names:")
        print(list(df.columns))
        
        if label_column_name not in df.columns:
            print(f"\nError: Label column '{label_column_name}' not found in the Excel file.")
            print(f"Please specify the correct label column name.")
            return

        # Print unique values in the original label column
        unique_labels = df[label_column_name].unique()
        print(f"\nUnique values in the '{label_column_name}' column (before simplification):")
        print(unique_labels)
        
        # Initial label simplification
        df['Simplified_Label'] = df[label_column_name].apply(simplify_label)
        print("\nSimplified Label counts (before contextual fill for 'Other'):")
        print(df['Simplified_Label'].value_counts(dropna=False))
        
        # Contextual fill for 'Other' labels
        # Create a series for filling, replacing 'Other' with NaN
        labels_to_fill = df['Simplified_Label'].replace('Other', np.nan).copy()
        
        # Forward fill
        labels_to_fill = labels_to_fill.ffill()
        # Backward fill for any 'Other' at the very beginning
        labels_to_fill = labels_to_fill.bfill()
        
        # Assign back to the DataFrame
        df['Simplified_Label'] = labels_to_fill
        
        # If any NaNs remain (e.g., if all original labels were 'Other'), mark them explicitly.
        # This is unlikely in the current scenario given previous outputs.
        df['Simplified_Label'] = df['Simplified_Label'].fillna(
            'Other_Unresolved_By_Context'
        )
        
        print("\nSimplified Label counts (after contextual fill for 'Other'):")
        print(df['Simplified_Label'].value_counts(dropna=False))
        
        # Save the processed DataFrame to a new CSV file
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully processed data and saved to '{output_csv_path}'")
        
        print("\nFirst 5 rows of the processed data with simplified labels:")
        print(df.head())
        
    except FileNotFoundError:
        print(f"Error: The file '{excel_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Simplify behavioral labels in an Excel workbook and export CSV."
    )
    parser.add_argument("input", type=Path, help="Input Excel workbook")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "processed_trace_simplified.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--label-column",
        default="behavior",
        help="Name of the behavior-label column",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process_and_simplify_excel(
        args.input,
        args.output,
        label_column_name=args.label_column,
    )


if __name__ == '__main__':
    main()
