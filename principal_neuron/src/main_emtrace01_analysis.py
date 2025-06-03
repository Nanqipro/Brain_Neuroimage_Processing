import pandas as pd
import os
from itertools import combinations # Add this import for combinations

# Assuming data_loader, config, and plotting_utils are in the same directory (src)
from data_loader import load_effect_sizes, load_neuron_positions
from config import (
    EFFECT_SIZE_THRESHOLD, BEHAVIOR_COLORS, MIXED_BEHAVIOR_COLORS,
    SHOW_BACKGROUND_NEURONS, BACKGROUND_NEURON_COLOR, 
    BACKGROUND_NEURON_SIZE, BACKGROUND_NEURON_ALPHA,
    STANDARD_KEY_NEURON_ALPHA, USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B # New config imports
)
from plotting_utils import (
    plot_single_behavior_activity_map, 
    plot_shared_neurons_map,
    plot_unique_neurons_map,
    plot_combined_9_grid # Import the new 3x3 grid plotting function
)

import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "output_plots"

def analyze_effect_sizes(df_effect_sizes_long):
    """
    Analyzes the effect size data (already in long format) to help determine a threshold.
    Prints descriptive statistics and plots a histogram and boxplot.
    Saves plots to the OUTPUT_DIR.
    Assumes df_effect_sizes_long has columns: 'Behavior', 'NeuronID', 'EffectSize'.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print("Descriptive statistics for effect sizes:")
    # The describe() on the long format will include NeuronID if not careful.
    # We are interested in the distribution of EffectSize values.
    print(df_effect_sizes_long['EffectSize'].describe())

    # Plot histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_effect_sizes_long, x='EffectSize', hue='Behavior', kde=True, element="step")
    plt.title('Distribution of Effect Sizes by Behavior')
    plt.xlabel('Effect Size')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    hist_path = os.path.join(OUTPUT_DIR, 'effect_size_histogram.png')
    plt.savefig(hist_path)
    print(f"\nHistogram of effect sizes saved to {hist_path}")
    # plt.show()

    # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_effect_sizes_long, x='Behavior', y='EffectSize')
    plt.title('Box Plot of Effect Sizes by Behavior')
    plt.xlabel('Behavior')
    plt.ylabel('Effect Size')
    plt.grid(axis='y', alpha=0.75)
    box_path = os.path.join(OUTPUT_DIR, 'effect_size_boxplot.png')
    plt.savefig(box_path)
    print(f"Boxplot of effect sizes saved to {box_path}")
    # plt.show()
    
    print("\nConsider the overall distribution, the spread within each behavior,")
    print("and any natural breaks or clusters when choosing a threshold.")
    print("You might want to choose a threshold that captures the upper quartile, for example,")
    print("or a value that seems to separate 'strong' effects from weaker ones based on the plots.")

def suggest_threshold_for_neuron_count(df_effects, min_neurons=5, max_neurons=10):
    print(f"\nAnalyzing effect sizes to find a threshold that yields {min_neurons}-{max_neurons} neurons per behavior.")

    potential_t_values = set()
    # Add effect sizes around the Nth neuron mark as candidates
    for behavior in df_effects['Behavior'].unique():
        behavior_df = df_effects[df_effects['Behavior'] == behavior].copy()
        behavior_df.sort_values(by='EffectSize', ascending=False, inplace=True)
        
        if len(behavior_df) >= min_neurons:
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[min_neurons - 1], 4)) # N_min_th neuron
        if len(behavior_df) > min_neurons -1 and min_neurons > 1 :
            # Add value slightly above (N_min-1)th neuron's ES to catch exactly N_min
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[min_neurons - 2], 4) + 0.00001) 

        if len(behavior_df) >= max_neurons:
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[max_neurons - 1], 4)) # N_max_th neuron
        if len(behavior_df) > max_neurons:
            # Add value slightly above (N_max+1)th neuron's ES to ensure <= N_max neurons
            potential_t_values.add(round(behavior_df['EffectSize'].iloc[max_neurons], 4) + 0.00001)
    
    # Add some generic sensible thresholds
    generic_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    for gt in generic_thresholds:
        potential_t_values.add(gt)
    
    candidate_thresholds = sorted([val for val in list(potential_t_values) if val > 0])

    best_t = None
    best_t_score = float('inf')
    best_t_counts = {}

    print(f"\nTesting {len(candidate_thresholds)} candidate thresholds...") # ({', '.join(f'{x:.3f}' for x in candidate_thresholds)}) 

    for t in candidate_thresholds:
        current_score_penalty = 0
        counts_for_t = {}
        all_behaviors_in_desired_range = True
        
        for behavior in df_effects['Behavior'].unique():
            behavior_df = df_effects[df_effects['Behavior'] == behavior]
            count = len(behavior_df[behavior_df['EffectSize'] >= t])
            counts_for_t[behavior] = count
            
            if not (min_neurons <= count <= max_neurons):
                all_behaviors_in_desired_range = False
            
            if count < min_neurons:
                current_score_penalty += (min_neurons - count) * 2 # Heavier penalty for too few
            elif count > max_neurons:
                current_score_penalty += (count - max_neurons)
        
        current_full_score = current_score_penalty
        if all_behaviors_in_desired_range:
            # If all counts are in range, prefer solutions that are more 'balanced'
            # (e.g., sum of squared deviations from the midpoint of the desired range)
            mid_point = (min_neurons + max_neurons) / 2.0
            balance_score = sum((c - mid_point)**2 for c in counts_for_t.values())
            current_full_score = balance_score # Override penalty, use balance score for 'good' thresholds
        
        if current_full_score < best_t_score:
            best_t_score = current_full_score
            best_t = t
            best_t_counts = counts_for_t
        elif current_full_score == best_t_score and (best_t is None or t < best_t):
             # Prefer smaller threshold if scores are identical to be slightly more inclusive
            if all_behaviors_in_desired_range == all(min_neurons <= c <= max_neurons for c in best_t_counts.values()): # only if new one is also 'good'
                best_t = t
                best_t_counts = counts_for_t

    if best_t is not None:
        print(f"\nRecommended threshold: T = {best_t:.4f}") # Using 4 decimal places for threshold
        print("Neuron counts for this threshold:")
        all_final_counts_in_range = True
        for b, c in best_t_counts.items():
            print(f"  {b}: {c} neurons")
            if not (min_neurons <= c <= max_neurons):
                all_final_counts_in_range = False
        if not all_final_counts_in_range:
             print(f"  Note: This threshold aims for the best balance, but some behaviors might be slightly outside the {min_neurons}-{max_neurons} range.")
        return best_t
    else:
        print("\nCould not automatically determine a suitable threshold from the candidates.")
        overall_75th = df_effects['EffectSize'].quantile(0.75)
        print(f"The overall 75th percentile of effect sizes is {overall_75th:.4f}. This could be a starting point for manual selection.")
        return None

def get_key_neurons(df_effects, threshold):
    """Identifies key neurons for each behavior based on the effect size threshold."""
    key_neurons_by_behavior = {}
    for behavior in df_effects['Behavior'].unique():
        behavior_df = df_effects[df_effects['Behavior'] == behavior]
        key_neuron_ids = behavior_df[behavior_df['EffectSize'] >= threshold]['NeuronID'].tolist()
        key_neurons_by_behavior[behavior] = sorted(list(set(key_neuron_ids)))
        print(f"Behavior: {behavior}, Threshold >= {threshold}, Key Neurons ({len(key_neuron_ids)}): {key_neurons_by_behavior[behavior]}")
    return key_neurons_by_behavior

if __name__ == "__main__":
    # Ensure the output directory for plots exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Define paths for data files (relative to workspace root)
    effect_data_identifier = 'data/EMtrace01-3标签版.csv'
    position_data_identifier = 'data/EMtrace01_Max_position.csv'

    # Load data
    print(f"\nLoading effect sizes from: {effect_data_identifier}")
    df_effect_sizes_transformed = load_effect_sizes(effect_data_identifier)
    
    print(f"\nLoading neuron positions from: {position_data_identifier}")
    df_neuron_positions = load_neuron_positions(position_data_identifier)

    if df_effect_sizes_transformed is not None and df_neuron_positions is not None:
        print(f"\nUsing effect size threshold: {EFFECT_SIZE_THRESHOLD} (from config.py)")
        
        # Get key neurons based on the threshold
        key_neurons_by_behavior = get_key_neurons(df_effect_sizes_transformed, EFFECT_SIZE_THRESHOLD)
        
        # --- Prepare data for 3x3 Combined Plot ---
        print("\nPreparing data for 3x3 combined plot...")
        plot_configurations_for_3x3 = []

        # Common parameters for many plots
        common_plot_params = {
            'all_neuron_positions_df': df_neuron_positions,
            'show_background_neurons': SHOW_BACKGROUND_NEURONS,
            'background_neuron_color': BACKGROUND_NEURON_COLOR,
            'background_neuron_size': BACKGROUND_NEURON_SIZE,
            'background_neuron_alpha': BACKGROUND_NEURON_ALPHA,
            'show_title': True # Titles in subplots are desired
        }

        # Parameters specific to single and unique plots (they have key_neuron_size and key_neuron_alpha)
        single_unique_plot_params = {
            **common_plot_params,
            'key_neuron_size': 150,
            'key_neuron_alpha': STANDARD_KEY_NEURON_ALPHA
        }

        # Parameters specific to shared plots (they don't have key_neuron_size and key_neuron_alpha)
        shared_plot_params = {
            'all_neuron_positions_df': df_neuron_positions,
            'show_background_neurons': SHOW_BACKGROUND_NEURONS,
            'background_neuron_color': BACKGROUND_NEURON_COLOR,
            'background_neuron_size': BACKGROUND_NEURON_SIZE,
            'background_neuron_alpha': BACKGROUND_NEURON_ALPHA,
            'show_title': True,
            'standard_key_neuron_alpha': STANDARD_KEY_NEURON_ALPHA,
            'use_standard_alpha_for_unshared_in_scheme_b': USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B,
            'alpha_non_shared': 0.3,
            'shared_marker_size_factor': 1.5
        }

        # Ensure a consistent order for behaviors (e.g., Close, Middle, Open)
        ordered_behavior_names = [b for b in BEHAVIOR_COLORS.keys() if b in key_neurons_by_behavior]
        if len(ordered_behavior_names) < 3 and len(key_neurons_by_behavior.keys()) ==3:
             # Fallback if BEHAVIOR_COLORS doesn't cover all, though it should
             ordered_behavior_names = list(key_neurons_by_behavior.keys())[:3]
        elif len(ordered_behavior_names) != len(key_neurons_by_behavior.keys()):
            print("Warning: Behavior order for 3x3 grid might be inconsistent or incomplete based on BEHAVIOR_COLORS keys.")
            # If partial, fill up to 3 with remaining from key_neurons_by_behavior
            missing_behaviors = [b for b in key_neurons_by_behavior.keys() if b not in ordered_behavior_names]
            ordered_behavior_names.extend(missing_behaviors)
        
        temp_key_dfs = {} # To store key DFs for behaviors
        for behavior_name in ordered_behavior_names:
            neuron_ids = key_neurons_by_behavior.get(behavior_name, [])
            df = df_neuron_positions[df_neuron_positions['NeuronID'].isin(neuron_ids)] if neuron_ids else pd.DataFrame(columns=['NeuronID', 'x', 'y'])
            temp_key_dfs[behavior_name] = df

        # Row 1: Single behavior plots
        for behavior_name in ordered_behavior_names:
            params_single = {
                **single_unique_plot_params,
                'key_neurons_df': temp_key_dfs[behavior_name],
                'behavior_name': behavior_name,
                'behavior_color': BEHAVIOR_COLORS.get(behavior_name, 'gray'),
                'title': f'{behavior_name} Key' # Simpler title for subplot
            }
            plot_configurations_for_3x3.append({'plot_type': 'single', 'params': params_single})

        # Row 2: Shared neuron plots (e.g., Close-Middle, Close-Open, Middle-Open)
        # Ensure consistent pairing order for title and mixed_color_key
        behavior_pairs = list(combinations(ordered_behavior_names, 2))
        for b1, b2 in behavior_pairs: # This generates 3 pairs if ordered_behavior_names has 3 items
            ids1 = set(key_neurons_by_behavior.get(b1, []))
            ids2 = set(key_neurons_by_behavior.get(b2, []))
            shared_ids_list = sorted(list(ids1.intersection(ids2)))
            
            df_b1_all_key = temp_key_dfs[b1]
            df_b2_all_key = temp_key_dfs[b2]
            df_shared_key = df_neuron_positions[df_neuron_positions['NeuronID'].isin(shared_ids_list)]
            
            mixed_color_key = tuple(sorted((b1, b2)))
            params_shared = {
                **shared_plot_params,
                'behavior1_name': b1,
                'behavior2_name': b2,
                'behavior1_all_key_neurons_df': df_b1_all_key,
                'behavior2_all_key_neurons_df': df_b2_all_key,
                'shared_key_neurons_df': df_shared_key,
                'color1': BEHAVIOR_COLORS.get(b1, 'pink'),
                'color2': BEHAVIOR_COLORS.get(b2, 'lightblue'),
                'mixed_color': MIXED_BEHAVIOR_COLORS.get(mixed_color_key, 'purple'),
                'title': f'{b1}-{b2} Shared',
                'scheme': 'B' # Assuming Scheme B is standard for these subplots
            }
            plot_configurations_for_3x3.append({'plot_type': 'shared', 'params': params_shared})
        
        # Fill remaining shared plots if less than 3 behaviors (won't happen with 3 behaviors)
        while len(plot_configurations_for_3x3) < 6 and len(ordered_behavior_names) <2: # Max 3 single + 3 shared
             # Add placeholder for shared if not enough behaviors to make 3 pairs
            plot_configurations_for_3x3.append({
                'plot_type': 'placeholder', # Need to handle this in plot_combined_9_grid or ensure 9 configs
                'params': {'title': 'N/A'}
            })

        # Row 3: Unique neuron plots
        all_behavior_sets_for_unique = {name: set(key_neurons_by_behavior.get(name,[])) for name in ordered_behavior_names}
        for b_name in ordered_behavior_names:
            other_behaviors_neurons = set()
            for other_b_name in ordered_behavior_names:
                if b_name == other_b_name: continue
                other_behaviors_neurons.update(all_behavior_sets_for_unique.get(other_b_name, set()))
            
            unique_ids = list(all_behavior_sets_for_unique.get(b_name, set()) - other_behaviors_neurons)
            df_unique_key = df_neuron_positions[df_neuron_positions['NeuronID'].isin(unique_ids)] if unique_ids else pd.DataFrame(columns=['NeuronID', 'x', 'y'])

            params_unique = {
                **single_unique_plot_params,
                'unique_neurons_df': df_unique_key,
                'behavior_name': b_name,
                'behavior_color': BEHAVIOR_COLORS.get(b_name, 'gray'),
                'title': f'{b_name} Unique'
            }
            plot_configurations_for_3x3.append({'plot_type': 'unique', 'params': params_unique})

        # Ensure we have exactly 9 configurations for the 3x3 grid
        # If there were fewer than 3 behaviors, some slots might be empty or need placeholders.
        # The logic above tries to fill based on ordered_behavior_names. If still not 9, add placeholders.
        # This placeholder handling should ideally be more robust or data generation should guarantee data for 9 plots.
        while len(plot_configurations_for_3x3) < 9:
            print(f"Warning: Not enough plot configurations for 3x3 grid (currently {len(plot_configurations_for_3x3)}). Adding placeholder(s).")
            plot_configurations_for_3x3.append({
                'plot_type': 'placeholder', 
                'params': {'title': 'Empty Slot'} # Placeholder title
            })
        
        if len(plot_configurations_for_3x3) > 9:
             print("Warning: More than 9 plot configurations generated. Truncating to 9 for 3x3 grid.")
             plot_configurations_for_3x3 = plot_configurations_for_3x3[:9]

        # --- Generate 3x3 Combined Plot ---
        if len(plot_configurations_for_3x3) == 9:
            print("\nGenerating 3x3 combined plot...")
            combined_plot_filename = "plot_all_behaviors_3x3_grid.png"
            combined_output_path = os.path.join(OUTPUT_DIR, combined_plot_filename)
            
            plot_combined_9_grid(
                plot_configurations=plot_configurations_for_3x3,
                output_path=combined_output_path,
                main_title_text=f"Comprehensive View: Neuron Activity Patterns (Effect Size >= {EFFECT_SIZE_THRESHOLD})"
            )
        else:
            print("Error: Could not prepare exactly 9 plot configurations for the 3x3 grid. Skipping combined plot.")

        print("\nAll plots generated.")

    else:
        if df_effect_sizes_transformed is None:
            print("Could not load effect sizes. Please check 'data_loader.py' and the CSV data.")
        if df_neuron_positions is None:
            print("Could not load neuron positions. Please check 'data_loader.py' and the CSV data.")
        # print("Error: Could not load data.") # Covered by more specific messages above

    # ... (suggest_threshold_for_neuron_count function definition if kept for reference) ... 
    # ... (suggest_threshold_for_neuron_count function definition if kept for reference) ... 