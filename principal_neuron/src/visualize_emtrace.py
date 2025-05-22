import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import io
from upsetplot import UpSet, from_contents

# --- Function to Generate UpSet Plot and Summary ---
def generate_upset_plot_with_summary(neuron_sets_dict, top_n, plot_title_prefix, dataset_source_name, output_filename):
    """
    Generates an UpSet plot for neuron overlap and prints a textual summary of intersections.

    Args:
        neuron_sets_dict (dict): Keys are behavior names, values are lists/sets of neuron IDs.
        top_n (int): The N value for Top N analysis.
        plot_title_prefix (str): The main title prefix for the plot.
        dataset_source_name (str): A descriptor for the data source (e.g., '(EMtrace Data)').
        output_filename (str): Filename to save the UpSet plot.
    """
    if not neuron_sets_dict or not any(neuron_sets_dict.values()):
        print("\nNo neuron data provided to generate UpSet plot and summary.")
        return

    # Ensure all values are sets for consistent processing, convert keys to string for safety
    processed_sets = {str(k): set(v) for k, v in neuron_sets_dict.items()}
    set_names = list(processed_sets.keys())

    print(f"\n--- Generating UpSet Plot and Summary for {dataset_source_name} (Top {top_n}) ---")

    try:
        upset_data_df = from_contents(processed_sets)

        if not upset_data_df.empty:
            plt.figure(figsize=(10, 7))
            upset_instance = UpSet(upset_data_df,
                                   min_subset_size=0,
                                   show_counts=True,
                                   sort_by='cardinality',
                                   # orientation='horizontal' # Optional: if preferred
                                  )
            upset_instance.plot()
            full_plot_title = f'{plot_title_prefix} Top {top_n} ({dataset_source_name})'
            plt.suptitle(full_plot_title, y=1.02)
            
            plt.savefig(output_filename, bbox_inches='tight')
            print(f"UpSet plot saved as {output_filename}")
            plt.show()

            # --- Generate Textual Summary from Upset Data ---
            print("\n--- Key Findings from the Plot ---")
            # upset_instance.intersections is a Pandas Series with MultiIndex
            # The MultiIndex levels correspond to the sets, and are boolean (True if in intersection)
            # The values are the sizes of these intersections.
            
            if hasattr(upset_instance, 'intersections') and not upset_instance.intersections.empty:
                for i, count in upset_instance.intersections.items(): # i is the tuple of booleans, count is the size
                    if count == 0 and upset_instance.min_subset_size > 0: # Skip empty if min_subset_size was > 0
                        continue

                    current_intersection_sets = []
                    # The index `i` directly maps to the order of `set_names` used in `from_contents`
                    for idx, is_member in enumerate(i):
                        if is_member:
                            current_intersection_sets.append(set_names[idx])
                    
                    if not current_intersection_sets: # Should not happen if count > 0
                        description = "(Empty set or error in intersection logic)"
                    elif len(current_intersection_sets) == len(set_names) and len(set_names) > 1:
                        description = f"Shared by ALL ({', '.join(current_intersection_sets)})"
                    elif len(current_intersection_sets) == 1:
                        description = f"Specific to '{current_intersection_sets[0]}'"
                    else:
                        description = f"Shared by ({', '.join(current_intersection_sets)})"
                    
                    print(f"{description}: {count} neuron(s)")
            else:
                print("Could not extract intersection data for summary.")

        else:
            print("Could not generate valid data for UpSet plot (e.g., all sets are empty).")
    except Exception as e:
        print(f"Error generating UpSet plot or summary: {e}")
        print("Please ensure 'upsetplot' library is installed and data is correctly formatted.")

# Data for EMtrace_3标签版.csv
# In a real scenario, you would load this from the CSV file:
# df_emtrace = pd.read_csv('/home/torpedo/Workspace/主神经元ID/data/EMtrace_3标签版.csv')
# For this example, using the string data directly:
csv_data_emtrace = """Behavior,Rank_1,Rank_2,Rank_3,Rank_4,Rank_5,Rank_6,Rank_7,Rank_8,Rank_9,Rank_10,Rank_11,Rank_12,Rank_13,Rank_14,Rank_15,Rank_16,Rank_17,Rank_18,Rank_19,Rank_20,Rank_21,Rank_22,Rank_23,Rank_24,Rank_25,Rank_26,Rank_27,Rank_28,Rank_29,Rank_30,Rank_31,Rank_32,Rank_33,Rank_34,Rank_35,Rank_36,Rank_37,Rank_38,Rank_39,Rank_40,Rank_41,Rank_42,Rank_43
Close,3,25,42,32,39,41,9,31,19,11,20,34,7,13,18,1,26,16,17,5,2,10,12,36,14,33,35,28,23,40,27,6,4,24,43,37,30,29,38,8,15,22,21
Middle,3,42,25,18,13,19,20,17,43,7,40,41,31,1,9,32,4,6,33,30,35,14,5,38,34,22,11,29,2,10,37,28,12,26,15,21,24,23,36,16,39,27,8
Open,5,39,30,10,22,18,27,43,16,40,32,17,38,11,4,29,13,26,21,19,6,15,36,34,25,2,20,33,37,42,23,9,8,12,14,24,35,31,41,28,1,7,3
"""
df_emtrace = pd.read_csv(io.StringIO(csv_data_emtrace))

top_n_emtrace = 10 # Defined specifically for this dataset analysis
core_neurons_emtrace = {}

for index, row in df_emtrace.iterrows():
    behavior_name = row['Behavior']
    # Rank columns are Rank_1, Rank_2, ...
    # Select the first top_n rank columns
    # Ensure we don't go out of bounds if less than top_n ranks exist
    num_rank_cols = min(top_n_emtrace, len(df_emtrace.columns) -1) # -1 for 'Behavior' column
    # Get values from Rank_1 up to Rank_N, converting to int
    ranked_neurons = row[1:num_rank_cols+1].astype(int).tolist()
    core_neurons_emtrace[behavior_name] = ranked_neurons

print("Extracted Top 10 Core Neurons for EMtrace:")
print(core_neurons_emtrace)

# --- Calculate and Print Specific and Overlapping Neurons (Console Output - still useful for quick check) ---
# This section can remain or be commented out if the function's summary is preferred
close_neurons_set = set(core_neurons_emtrace.get('Close', []))
middle_neurons_set = set(core_neurons_emtrace.get('Middle', []))
open_neurons_set = set(core_neurons_emtrace.get('Open', []))

if close_neurons_set or middle_neurons_set or open_neurons_set:
    print("\n--- Quick Analysis of Top 10 Neuron Overlap and Specificity (Console) ---")
    shared_all_three = list(close_neurons_set.intersection(middle_neurons_set).intersection(open_neurons_set))
    print(f"Neurons in Top 10 for Close, Middle, AND Open: {shared_all_three}")
    # ... (rest of this specific/overlap print block can be kept or removed) ...
    # For brevity in this edit, I'm assuming it's kept or the user can manage it.
    # The key is to replace the old UpSet plotting block with a call to the new function.
    # Let's re-add the specific printouts for clarity, as they are useful for comparison
    shared_close_middle = list(close_neurons_set.intersection(middle_neurons_set) - set(shared_all_three))
    print(f"Neurons in Top 10 for Close AND Middle (but not Open): {shared_close_middle}")
    shared_close_open = list(close_neurons_set.intersection(open_neurons_set) - set(shared_all_three))
    print(f"Neurons in Top 10 for Close AND Open (but not Middle): {shared_close_open}")
    shared_middle_open = list(middle_neurons_set.intersection(open_neurons_set) - set(shared_all_three))
    print(f"Neurons in Top 10 for Middle AND Open (but not Close): {shared_middle_open}")
    specific_close = list(close_neurons_set - middle_neurons_set - open_neurons_set)
    print(f"Top 10 Neurons specific to Close (not in Middle or Open Top 10): {specific_close}")
    specific_middle = list(middle_neurons_set - close_neurons_set - open_neurons_set)
    print(f"Top 10 Neurons specific to Middle (not in Close or Open Top 10): {specific_middle}")
    specific_open = list(open_neurons_set - close_neurons_set - middle_neurons_set)
    print(f"Top 10 Neurons specific to Open (not in Close or Middle Top 10): {specific_open}")
else:
    print("\nNo neuron data found to perform overlap analysis (Console quick check).")

# --- Bar Charts for Top N Neurons (Can also be functionalized if needed later) ---
# This plotting code can remain as is for now, or also be refactored into a function.
# For this change, we are focusing on refactoring the Upset plot.
for behavior, neurons in core_neurons_emtrace.items():
    plt.figure(figsize=(12, 7))
    neuron_ids_str = [str(n) for n in neurons]
    y_values = [top_n_emtrace - i for i in range(len(neurons))]
    plt.bar(neuron_ids_str, y_values)
    plt.ylabel(f'Importance Score (Higher is better, based on Top {top_n_emtrace} Ranking)')
    plt.xlabel('Neuron ID')
    plt.title(f'Top {top_n_emtrace} Neurons for Behavior: {behavior} (EMtrace Data)')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(ticks=range(0, top_n_emtrace + 1), labels=range(top_n_emtrace, -1, -1))
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

# --- Venn Diagram for Overlap (Can also be functionalized) ---
if close_neurons_set or middle_neurons_set or open_neurons_set:
    plt.figure(figsize=(8, 8))
    venn3([close_neurons_set, middle_neurons_set, open_neurons_set], ('Close Behavior', 'Middle Behavior', 'Open Behavior'))
    plt.title(f'Overlap of Top {top_n_emtrace} Neurons in EMtrace Behaviors')
    plt.show()
else:
    print("Not enough data for Venn diagram (e.g., missing behaviors).")

# --- Generate UpSet Plot and Summary using the new function ---
# The old direct UpSet plot code block should be removed and replaced by this call:
generate_upset_plot_with_summary(
    neuron_sets_dict=core_neurons_emtrace, 
    top_n=top_n_emtrace, 
    plot_title_prefix="Neuron Overlap", 
    dataset_source_name="EMtrace Data", 
    output_filename="upset_plot_emtrace_top10_functionalized.png"
)

print("\nScript finished.") 