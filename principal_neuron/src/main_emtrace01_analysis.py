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
    plot_unique_neurons_map # Import the new plotting function for unique neurons
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
        
        # --- Plots 1, 2, 3: Single Behavior Key Neurons ---
        print("\nGenerating plots for single behavior key neurons...")
        for behavior_name, neuron_ids in key_neurons_by_behavior.items():
            if not neuron_ids: # Check if the list of neuron IDs is empty
                print(f"No key neurons for '{behavior_name}' at threshold {EFFECT_SIZE_THRESHOLD}. Skipping plot.")
                # Optionally, call plotting function which handles empty df to create a placeholder plot
                empty_df_for_plot = pd.DataFrame(columns=['NeuronID', 'x', 'y'])
                plot_title = f"Key Neurons for '{behavior_name}' (No neurons found)"
                file_name = f"plot_{behavior_name.lower()}_key_neurons.png"
                output_path = os.path.join(OUTPUT_DIR, file_name)
                plot_single_behavior_activity_map(
                    key_neurons_df=empty_df_for_plot,
                    behavior_name=behavior_name,
                    behavior_color=BEHAVIOR_COLORS.get(behavior_name, 'gray'),
                    title=plot_title,
                    output_path=output_path,
                    all_neuron_positions_df=df_neuron_positions,
                    show_background_neurons=SHOW_BACKGROUND_NEURONS,
                    background_neuron_color=BACKGROUND_NEURON_COLOR,
                    background_neuron_size=BACKGROUND_NEURON_SIZE,
                    background_neuron_alpha=BACKGROUND_NEURON_ALPHA
                )
                continue

            # Filter positions for the current behavior's key neurons
            current_behavior_key_neurons_df = df_neuron_positions[df_neuron_positions['NeuronID'].isin(neuron_ids)]
            
            if current_behavior_key_neurons_df.empty and neuron_ids:
                 print(f"Warning: Neuron IDs {neuron_ids} found for behavior '{behavior_name}', but no positions found in position data.")
                 # Plot with a message or skip

            plot_title = f"Key Neurons for '{behavior_name}' Behavior (Effect Size >= {EFFECT_SIZE_THRESHOLD})"
            file_name = f"plot_{behavior_name.lower()}_key_neurons.png" # e.g., plot_close_key_neurons.png
            output_path = os.path.join(OUTPUT_DIR, file_name)
            
            plot_single_behavior_activity_map(
                key_neurons_df=current_behavior_key_neurons_df,
                behavior_name=behavior_name,
                behavior_color=BEHAVIOR_COLORS.get(behavior_name, 'gray'),
                title=plot_title,
                output_path=output_path,
                all_neuron_positions_df=df_neuron_positions,
                show_background_neurons=SHOW_BACKGROUND_NEURONS,
                background_neuron_color=BACKGROUND_NEURON_COLOR,
                background_neuron_size=BACKGROUND_NEURON_SIZE,
                background_neuron_alpha=BACKGROUND_NEURON_ALPHA
            )

        # --- Identify Shared Neurons & Generate Plots 4, 5, 6 ---
        print("\nIdentifying shared key neurons and generating plots (Scheme B by default)...")
        behavior_names = list(key_neurons_by_behavior.keys())
        
        for pair in combinations(behavior_names, 2):
            b1, b2 = pair
            ids1 = set(key_neurons_by_behavior[b1])
            ids2 = set(key_neurons_by_behavior[b2])
            shared_ids_list = sorted(list(ids1.intersection(ids2)))
            print(f"Shared neurons between '{b1}' and '{b2}': {len(shared_ids_list)} {shared_ids_list}")

            b1_all_key_neuron_ids = key_neurons_by_behavior[b1]
            b2_all_key_neuron_ids = key_neurons_by_behavior[b2]

            df_b1_all_key_neurons = df_neuron_positions[df_neuron_positions['NeuronID'].isin(b1_all_key_neuron_ids)]
            df_b2_all_key_neurons = df_neuron_positions[df_neuron_positions['NeuronID'].isin(b2_all_key_neuron_ids)]
            df_shared_key_neurons = df_neuron_positions[df_neuron_positions['NeuronID'].isin(shared_ids_list)]

            # Ensure consistent key for MIXED_BEHAVIOR_COLORS
            mixed_color_key = tuple(sorted((b1, b2)))
            mixed_color = MIXED_BEHAVIOR_COLORS.get(mixed_color_key, 'purple') # Default to purple if not in config

            # For Scheme B (default)
            scheme_to_use = 'B'
            file_name_shared = f"plot_shared_{b1.lower()}_{b2.lower()}_scheme{scheme_to_use}.png" 
            output_path_shared = os.path.join(OUTPUT_DIR, file_name_shared)
            plot_title_shared = f"Key Neurons: {b1} & {b2} (Shared Highlighted - Scheme {scheme_to_use})"
            
            plot_shared_neurons_map(
                behavior1_name=b1, behavior2_name=b2,
                behavior1_all_key_neurons_df=df_b1_all_key_neurons, 
                behavior2_all_key_neurons_df=df_b2_all_key_neurons,
                shared_key_neurons_df=df_shared_key_neurons,
                color1=BEHAVIOR_COLORS[b1],
                color2=BEHAVIOR_COLORS[b2],
                mixed_color=mixed_color,
                title=plot_title_shared,
                output_path=output_path_shared,
                scheme=scheme_to_use,
                all_neuron_positions_df=df_neuron_positions,
                show_background_neurons=SHOW_BACKGROUND_NEURONS,
                background_neuron_color=BACKGROUND_NEURON_COLOR,
                background_neuron_size=BACKGROUND_NEURON_SIZE,
                background_neuron_alpha=BACKGROUND_NEURON_ALPHA,
                standard_key_neuron_alpha=STANDARD_KEY_NEURON_ALPHA, # Pass new config
                use_standard_alpha_for_unshared_in_scheme_b=USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B # Pass new config
            )
            
            # # To generate Scheme A as well, uncomment and modify this block:
            # scheme_to_use_A = 'A'
            # file_name_shared_A = f"plot_shared_{b1.lower()}_{b2.lower()}_scheme{scheme_to_use_A}.png"
            # output_path_shared_A = os.path.join(OUTPUT_DIR, file_name_shared_A)
            # plot_title_shared_A = f"Key Neurons: {b1} & {b2} (Only Shared - Scheme {scheme_to_use_A})"
            # plot_shared_neurons_map(
            #     behavior1_name=b1, behavior2_name=b2,
            #     behavior1_all_key_neurons_df=df_b1_all_key_neurons, # Not strictly needed for scheme A but passed for consistency
            #     behavior2_all_key_neurons_df=df_b2_all_key_neurons, # Not strictly needed for scheme A
            #     shared_key_neurons_df=df_shared_key_neurons,
            #     color1=BEHAVIOR_COLORS[b1], # Not used in scheme A plot itself
            #     color2=BEHAVIOR_COLORS[b2], # Not used in scheme A plot itself
            #     mixed_color=mixed_color,
            #     title=plot_title_shared_A,
            #     output_path=output_path_shared_A,
            #     scheme=scheme_to_use_A
            # )

        # --- Identify Unique Neurons & Generate Plots 7, 8, 9 ---
        print("\nIdentifying unique key neurons for each behavior...")
        unique_neurons_by_behavior = {}
        all_behavior_sets = {name: set(ids) for name, ids in key_neurons_by_behavior.items()}
        behavior_names = list(key_neurons_by_behavior.keys()) # Ensure behavior_names is defined here if not earlier

        for b_name in behavior_names:
            other_behaviors_neurons = set()
            for other_b_name in behavior_names:
                if b_name == other_b_name:
                    continue
                other_behaviors_neurons.update(all_behavior_sets[other_b_name])
            unique_ids = list(all_behavior_sets[b_name] - other_behaviors_neurons)
            unique_neurons_by_behavior[b_name] = sorted(unique_ids)
            print(f"Unique neurons for '{b_name}': {len(unique_neurons_by_behavior[b_name])} {unique_neurons_by_behavior[b_name]}")

        print("\nGenerating plots for unique key neurons (Plots 7, 8, 9)...")
        for behavior_name, unique_ids_list in unique_neurons_by_behavior.items():
            df_unique_key_neurons = pd.DataFrame(columns=['NeuronID', 'x', 'y']) # Default to empty
            if not unique_ids_list:
                print(f"No unique neurons for '{behavior_name}'. Plotting empty placeholder.")
            else:
                df_unique_key_neurons = df_neuron_positions[df_neuron_positions['NeuronID'].isin(unique_ids_list)]
            
            if df_unique_key_neurons.empty and unique_ids_list: # Check if still empty even if IDs existed (positions missing)
                print(f"Warning: Unique Neuron IDs {unique_ids_list} found for '{behavior_name}', but no positions found. Plotting empty.")

            file_name_unique = f"plot_unique_{behavior_name.lower()}_neurons.png"
            output_path_unique = os.path.join(OUTPUT_DIR, file_name_unique)
            plot_title_unique = f"Unique Key Neurons for '{behavior_name}' Behavior (Effect Size >= {EFFECT_SIZE_THRESHOLD})"

            plot_unique_neurons_map(
                unique_neurons_df=df_unique_key_neurons,
                behavior_name=behavior_name,
                behavior_color=BEHAVIOR_COLORS.get(behavior_name, 'grey'), # Default to grey if color not in config
                title=plot_title_unique,
                output_path=output_path_unique,
                all_neuron_positions_df=df_neuron_positions,
                show_background_neurons=SHOW_BACKGROUND_NEURONS,
                background_neuron_color=BACKGROUND_NEURON_COLOR,
                background_neuron_size=BACKGROUND_NEURON_SIZE,
                background_neuron_alpha=BACKGROUND_NEURON_ALPHA
            )
        
        print("\nAll plots generated.")

    else:
        if df_effect_sizes_transformed is None:
            print("Could not load effect sizes. Please check 'data_loader.py' and the CSV data.")
        if df_neuron_positions is None:
            print("Could not load neuron positions. Please check 'data_loader.py' and the CSV data.")
        # print("Error: Could not load data.") # Covered by more specific messages above

    # ... (suggest_threshold_for_neuron_count function definition if kept for reference) ... 