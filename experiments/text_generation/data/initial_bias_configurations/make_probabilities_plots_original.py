import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches

import json

OUTPUT_ROOT_DIR = '/home/coder/autogoal/experiments/text_generation/data/initial_bias_configurations/plots'

# Utility to load all configs from a directory of jsons
def load_config_data_from_dir(dir_path):
    data = {}
    for fname in os.listdir(dir_path):
        if fname.endswith('.json'):
            config_name = os.path.splitext(fname)[0]
            fpath = os.path.join(dir_path, fname)
            with open(fpath, 'r') as f:
                try:
                    config_data = json.load(f)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")
                    continue
            data[config_name] = config_data
    return data

def get_custom_colors(sorted_configs, variability, thresholds=None, margins=None):
    # Define group colors - use RGBA format for consistent handling of transparency
    group_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # blue, green, orange, red
    
    # Define static thresholds for groups
    tv_thresholds = [0.0, 0.005, 0.1, 0.3]  # Blue: 0-0.07, Green: 0.07-0.1, Orange: 0.1-0.4, Red: 0.4+
    
    # Ensure uniform is first
    if 'uniform' not in sorted_configs:
        sorted_configs = ['uniform'] + [c for c in sorted_configs if c != 'uniform']
    
    # Create groups for candidate selection
    groups = [[] for _ in range(4)]  # [blue, green, orange, red]
    
    # Classify configs by TV using static thresholds
    color_map = {}
    
    for config in sorted_configs:
        tv = variability.get(config, 0)
        # Determine which group this config belongs to
        group_idx = 0  # Default to blue group
        for i, threshold in enumerate(tv_thresholds):
            if tv >= threshold:
                group_idx = i
        # Ensure group_idx is at most 3 (red group)
        group_idx = min(group_idx, 3)
        color_map[config] = group_colors[group_idx]
        groups[group_idx].append(config)
    
    # Choose one candidate per group:
    # For blue group: always select baseline ('uniform')
    # For other groups: select the last config in each non-empty group
    candidates = []
    
    # Blue group candidate is always baseline ('uniform')
    if 'uniform' in groups[0]:
        candidates.append('uniform')
    elif groups[0]:  # If uniform isn't there but group isn't empty
        candidates.append(groups[0][0])  # First config in blue group
    
    # For other groups (green, orange, red), use last config as candidate
    for group in groups[1:]:
        if group:  # If group is not empty
            candidates.append(group[-1])  # Last config in the group
    
    return color_map, candidates, groups

def plot_probabilities(data, title):
    # Ensure the output root directory exists
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    

    # Convert weights to probabilities
    probabilities = {}
    for config, methods in data.items():
        total_weight = sum(methods.values())
        if total_weight == 0:
            probs = {method: 0 for method in methods}
        else:
            probs = {method: weight / total_weight for method, weight in methods.items()}
        probabilities[config] = probs

    # Extract the order of methods from the first configuration
    first_config = next(iter(data))
    methods_order = list(data[first_config].keys())

    # Ensure all methods from other configurations are included
    all_methods = methods_order.copy()
    for config, methods in data.items():
        for method in methods:
            if method not in all_methods:
                all_methods.append(method)

    # Add uniform baseline if not present
    if 'uniform' not in probabilities:
        n_methods = len(all_methods)
        uniform_prob = 1.0 / n_methods if n_methods > 0 else 0
        probabilities['uniform'] = {method: uniform_prob for method in all_methods}

    num_methods = len(all_methods)
    num_configs = len(probabilities)
    x = np.arange(num_methods)  # the label locations

    # Compute variability for each configuration compared to the base config
    base_config = 'uniform' if 'uniform' in probabilities else first_config
    base_probs = probabilities[base_config]

    variability = {}
    for config, probs in probabilities.items():
        if config == base_config:
            variability[config] = 0  # Base config has zero variability
            continue
        # Compute sum of absolute differences as a variability metric
        var = sum(abs(probs.get(method, 0) - base_probs.get(method, 0)) for method in all_methods)
        variability[config] = var

    # Sort configurations based on variability (ascending order)
    # Base config is first, followed by others sorted by variability
    sorted_configs = sorted(probabilities.keys(), key=lambda x: variability[x])
    
    # Prepare data for plotting
    num_methods = len(all_methods)
    num_configs = len(sorted_configs)
    x = np.arange(num_methods)  # the label locations (one per method)

    # Set the width of each bar (per config) within each method group
    total_width = 0.8
    bar_width = total_width / num_configs

    # Compute color map, candidates, and groups
    color_map, candidates, groups = get_custom_colors(sorted_configs, variability)
    
    # Create a mapping from each config to its group color (for reference)
    config_groups = {}
    group_colors = ['blue', 'green', 'orange', 'red']
    for i, group in enumerate(groups):
        for config in group:
            config_groups[config] = group_colors[i]

    # For patterns: only candidates get a hatch
    pattern_map = {cfg: ('///' if cfg in candidates else '') for cfg in sorted_configs}

    # Plotting: grouped bar plot (each method is a group, each config is a bar in the group)
    fig, ax = plt.subplots(figsize=(max(10, 2*num_methods), 7))
    rects_list = []
    for j, config in enumerate(sorted_configs):
        # For each method, plot this config's value at the correct position
        values = [probabilities[config].get(method, 0) for method in all_methods]
        # Bar positions for this config: offset within each method group
        # Set alpha: 1.0 for representatives, 0.7 for others
        if config in candidates:
            this_bar_width = bar_width * 2.0  # Reduce width from 3.0 to 2.0
            bar_alpha = 1.0
            bar_hatch = '///'
            # For PDF hatch to show, use facecolor and set edgecolor to black
            bar_edgecolor = 'black'
            bar_linewidth = 0.8
        else:
            this_bar_width = bar_width
            bar_alpha = 0.5
            bar_hatch = ''
            bar_edgecolor = None
            bar_linewidth = 0

        # Calculate position to prevent clipping
        if config in candidates:
            offset = (this_bar_width - bar_width) / 2
            position_offset = (j + 0.5) * bar_width - offset
        else:
            position_offset = (j + 0.5) * bar_width

        # Convert color to RGBA if it's a hex string
        bar_color = color_map[config]
        if isinstance(bar_color, str):
            from matplotlib.colors import to_rgba
            bar_color = to_rgba(bar_color, alpha=bar_alpha)

        rects = ax.bar(
            x - (total_width/2) + position_offset,
            values,
            this_bar_width,
            label=f"{j} - {config} (Var: {variability[config]:.3f})" if config in candidates else None,
            color=bar_color,
            hatch=bar_hatch,
            edgecolor=bar_edgecolor,
            linewidth=bar_linewidth
        )
        rects_list.append(rects)

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Probability')
    # Extract dataset name from title
    dataset_name = title.split('(')[-1].split(')')[0] if '(' in title else 'LIAR'
    ax.set_title(f'Probabilities of Fine-tuning Method/Model Type ({dataset_name})')
    ax.set_xticks(x)

    # Mapping for simpler x labels
    simple_label_map = {
        "FineTuneGenLLMTask": "Fine-tune Gen",
        "LoraGenLLMTask": "Lora Gen",
        "PartialFineTuneGenLLMTask": "Partial Fine-tune Gen"
    }
    simple_labels = [simple_label_map.get(m, m) for m in all_methods]
    ax.set_xticklabels(simple_labels, ha='center')
    
    # Add legend back to the main figure (top left)
    legend_handles = []
    for config in candidates:
        # Convert color to RGBA with full opacity for legend
        if isinstance(color_map[config], str):
            from matplotlib.colors import to_rgba
            color = to_rgba(color_map[config], alpha=1.0)
        else:
            color = color_map[config]
            
        patch = mpatches.Patch(
            facecolor=color,
            edgecolor='black',
            hatch='///',
            label=f"{config} (Var: {variability[config]:.3f})"
        )
        legend_handles.append(patch)
    ax.legend(handles=legend_handles, loc='upper left', frameon=True)

    # Function to attach a text label above each bar
    def autolabel(rects, config_index):
        """Attach a text label above each bar displaying its height."""
        for rect in rects:
            height = rect.get_height()  
            if height > 0:
                # Add two lines: probability and config index
                ax.annotate(
                    f'{height:.3f}\n({config_index})',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', 
                    va='bottom', 
                    fontsize=8
                )

    # Apply autolabel to all bars, passing the config's index
    # for i, rects in enumerate(rects_list):
    #     autolabel(rects, i)

    fig.tight_layout()
    
    # Adjust layout to make room for the legend
    # plt.subplots_adjust(right=0.8)


    plt.show()

    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    import re
    safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', title.lower())
    fig_path = os.path.join(OUTPUT_ROOT_DIR, safe_title)
    try:
        fig.savefig(f"{fig_path}.svg", dpi=300)
        fig.savefig(f"{fig_path}.png", dpi=300)
        # Save as PDF with tight bounding box and no transparency issues
        fig.savefig(f"{fig_path}.pdf", bbox_inches='tight', transparent=False)
        print(f"Plot saved successfully at {fig_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)

    # We no longer need a separate legend plot, legend is now in main figure
        

if __name__=='__main__':
    # Paths for drop and squad
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = ["drop", "squad"]
    for dataset in datasets:
        data_dir = os.path.join(base_dir, dataset)
        if not os.path.isdir(data_dir):
            print(f"Directory not found: {data_dir}")
            continue
        data = load_config_data_from_dir(data_dir)
        if not data:
            print(f"No data loaded for {dataset}.")
            continue
        plot_probabilities(data, f"Starting Prob. of Fine-tuning Method/Model Type ({dataset.upper()})")