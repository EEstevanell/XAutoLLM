import os
import json
from typing import List
from matplotlib.axes import Axes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from pathlib import Path

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        # Existing settings
        "font.family": "serif",  # Force serif font family
        "font.serif": [
            "Times New Roman",
            "Linux Libertine O",
            "DejaVu Serif",
            "serif"
        ],  # Academic standard first, fallback to generic
        "mathtext.fontset": "stix",  # Math font matching Times
        "axes.formatter.use_mathtext": True,  # Math-style number formatting
        # Keep your existing size settings
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.title_fontsize": 10,
        "legend.fontsize": 10,
        "figure.constrained_layout.use": True,
    }
)


class ExperimentPlotData:
    def __init__(
        self,
        alias_name: str,
        dataframe: pd.DataFrame,
        color: str,
        linestyle: str,
        markerstyle: str,
    ):
        self.alias_name = alias_name
        self.dataframe = dataframe
        self.color = color
        self.linestyle = linestyle
        self.markerstyle = markerstyle


# Define the root directory for experiences and output
EXPERIENCE_ROOT_DIR = "experience_store"
OUTPUT_ROOT_DIR = "/home/coder/autogoal/experiments/output/plots"


def parse_timestamp(date_folder, json_filename):
    """
    Combine date and time from folder and file to create a datetime object.
    """
    date_part = date_folder
    time_part = json_filename.split("-")[0]  # 'hh:mm:ss'
    datetime_str = f"{date_part} {time_part}"
    try:
        return datetime.strptime(datetime_str, "%Y-%m-%d %H_%M_%S")
    except ValueError as ve:
        print(f"Timestamp parsing error for {datetime_str}: {ve}")
        return None


def load_data_for_alias(alias):
    """
    Load and process JSON data for a given alias.
    Returns a pandas DataFrame sorted by timestamp with f1, accuracy, and evaluation_time.
    """
    alias_path = os.path.join(EXPERIENCE_ROOT_DIR, alias)
    data = []

    if not os.path.isdir(alias_path):
        print(f"Alias directory not found: {alias_path}")
        return pd.DataFrame()

    # Iterate through date folders
    for date_folder in sorted(os.listdir(alias_path)):
        date_path = os.path.join(alias_path, date_folder)
        if not os.path.isdir(date_path):
            continue  # Skip non-directory files

        # Iterate through JSON files
        for json_file in sorted(os.listdir(date_path)):
            if not json_file.endswith(".json"):
                continue  # Skip non-JSON files

            json_path = Path(os.path.join(date_path, json_file))
            try:
                with open(json_path, "r") as f:
                    content = json.load(f)

                timestamp = parse_timestamp(date_folder, json_file)
                if timestamp is None:
                    continue  # Skip if timestamp parsing failed

                # Extract metrics
                f1 = content.get("f1", np.nan)
                accuracy = content.get("accuracy", np.nan)
                evaluation_time = content.get("evaluation_time", np.nan)

                # finetuning_method will always be first algorithm here
                algorithm = dict(content["algorithms"][0])
                finetuning_method = list(algorithm)[0]
                llm = list(algorithm[finetuning_method]["inner_model"]["value"])[0]
                params = list(algorithm[finetuning_method])[1:]

                # Append to data
                data.append(
                    {
                        "alias": alias,
                        "timestamp": timestamp,
                        "f1": f1,
                        "accuracy": accuracy,
                        "evaluation_time": evaluation_time,
                        "finetuning_method": finetuning_method,
                        "llm": str(llm)
                        .removeprefix("WORD_EMB_")
                        .removeprefix("TEXT_GEN_")
                        .replace("_", " "),
                        "parameters": params,
                    }
                )

            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue  # Skip corrupted or unreadable files

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_data():
    """
    Load data for all aliases and return a dictionary mapping alias names to DataFrames.
    """
    data = dict()

    # List all aliases
    aliases = [
        alias
        for alias in os.listdir(EXPERIENCE_ROOT_DIR)
        if os.path.isdir(os.path.join(EXPERIENCE_ROOT_DIR, alias))
    ]

    if not aliases:
        print(f"No aliases found in the directory: {EXPERIENCE_ROOT_DIR}")
        return data

    for alias in aliases:
        print(f"Processing alias: {alias}")
        data[alias] = load_data_for_alias(alias)

    return data


def find_alias_pairs(aliases):
    """
    Identifies and returns pairs of aliases in the format <alias> and <alias>_warmstart_id_<id>.

    :param aliases: List of all alias names.
    :return: List of tuples, each containing a pair of aliases.
    """
    base_aliases = [alias for alias in aliases if "_warmstart_id_" not in alias]
    alias_pairs = []

    for base in base_aliases:
        # Find all warmstart aliases corresponding to the base alias
        warmstart_aliases = [
            alias for alias in aliases if alias.startswith(f"{base}_warmstart_id_")
        ]
        for warm_alias in warmstart_aliases:
            alias_pairs.append((base, warm_alias))

    return alias_pairs


def plot_data_with_metrics(df, alias, figure_name, output_dir="plots"):
    """
    Plot the progression of cumulative mean Macro F1 scores and standard deviation over steps.
    Plot the cumulative mean errors over time as slightly transparent red bars normalized between 0 and 1.
    Change the evaluation time color to green.
    Additionally, plot the normalized cumulative mean errors in an inset plot.

    :param df: DataFrame containing 'step', 'f1', 'accuracy', 'evaluation_time', and other relevant columns.
    :param alias: Identifier for the current dataset/experiment.
    :param figure_name: Name of the output file (e.g., 'combined_metrics.png').
    :param output_dir: Directory where the plot will be saved.
    """
    # Ensure 'step' is present
    if "step" not in df.columns:
        df = df.reset_index().rename(columns={"index": "step"})

    # Sort by step to ensure correct plotting order
    df.sort_values("step", inplace=True)

    # Identify error steps (steps with invalid 'f1', 'accuracy', or 'evaluation_time')
    error_conditions = (
        df[["f1", "accuracy", "evaluation_time"]].isna().any(axis=1)
    )
    error_steps = df[error_conditions]["step"].tolist()

    # Total steps
    total_steps = len(df)

    # Number of error steps
    num_errors = len(error_steps)

    # Create an 'error' column: 1 if error occurred at that step, else 0
    df["error"] = error_conditions.astype(int)

    # Compute cumulative sum and cumulative mean of errors
    df["cumulative_errors"] = df["error"].cumsum()
    df["cumulative_mean_errors"] = df["cumulative_errors"] / (
        df["step"] + 1
    )  # Adding 1 to avoid division by zero

    # Normalize cumulative mean errors to range [0, 1]
    df["normalized_cumulative_mean_errors"] = (
        df["cumulative_mean_errors"] / 1
    )  # Already a ratio, but kept for clarity

    # Valid data (non-error steps)
    valid_df = df[~error_conditions].copy()
    valid_df.reset_index(drop=True, inplace=True)

    # Compute cumulative mean and standard deviation of 'f1'
    valid_df["cumulative_mean_f1"] = valid_df["f1"].expanding().mean()
    valid_df["cumulative_std_f1"] = (
        valid_df["f1"].expanding().std().fillna(0)
    )

    # Compute cumulative max of 'f1'
    valid_df["cumulative_max_f1"] = valid_df["f1"].cummax()

    # Compute cumulative mean of 'evaluation_time'
    valid_df["cumulative_mean_eval_time"] = (
        valid_df["evaluation_time"].expanding().mean()
    )

    # Initialize the matplotlib figure and gridspec for main plot and inset
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])

    # Apply spline smoothing to cumulative metrics
    spline_k = 3  # Spline degree
    # Ensure there are enough points for spline
    if len(valid_df["step"]) > spline_k:
        spline_f1 = UnivariateSpline(
            valid_df["step"], valid_df["cumulative_mean_f1"], k=spline_k, s=0
        )
        spline_eval_time = UnivariateSpline(
            valid_df["step"], valid_df["cumulative_mean_eval_time"], k=spline_k, s=0
        )
    else:
        # If not enough points, skip smoothing
        spline_f1 = valid_df["cumulative_mean_f1"]
        spline_eval_time = valid_df["cumulative_mean_eval_time"]

    # Plot cumulative max Macro F1
    ax1.plot(
        valid_df["step"],
        valid_df["cumulative_max_f1"],
        label="Cumulative Max F1",
        color="blue",
        linewidth=2,
    )

    # Plot cumulative mean Macro F1 with smoothing
    if isinstance(spline_f1, pd.Series):
        ax1.plot(
            valid_df["step"],
            spline_f1,
            label="Cumulative Mean F1",
            color="blue",
            linestyle="--",
            linewidth=2,
        )
    else:
        ax1.plot(
            valid_df["step"],
            spline_f1(valid_df["step"]),
            label="Cumulative Mean F1",
            color="blue",
            linestyle="--",
            linewidth=2,
        )

    # Plot standard deviation shading with smoothing
    upper_bound = (
        valid_df["cumulative_mean_f1"] + valid_df["cumulative_std_f1"]
    )
    lower_bound = (
        valid_df["cumulative_mean_f1"] - valid_df["cumulative_std_f1"]
    )

    if isinstance(spline_f1, pd.Series):
        ax1.fill_between(
            valid_df["step"],
            lower_bound,
            upper_bound,
            color="blue",
            alpha=0.2,
            label="Std Dev",
        )
    else:
        smoothed_upper = UnivariateSpline(
            valid_df["step"], upper_bound, k=spline_k, s=0
        )(valid_df["step"])
        smoothed_lower = UnivariateSpline(
            valid_df["step"], lower_bound, k=spline_k, s=0
        )(valid_df["step"])
        ax1.fill_between(
            valid_df["step"],
            smoothed_lower,
            smoothed_upper,
            color="blue",
            alpha=0.2,
            label="Std Dev",
        )

    # Set labels and title for primary y-axis with error statistics
    title_text = (
        f"Model Performance and Evaluation Time Over Steps "
        f"(Errors: {num_errors}/{total_steps})"
    )
    ax1.set_xlabel("Step Number", fontsize=16)
    ax1.set_ylabel("Macro F1 Score", fontsize=16, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title(title_text, fontsize=20)

    # Plot cumulative mean evaluation time on secondary y-axis
    ax2 = ax1.twinx()
    eval_time_color = "green"

    if isinstance(spline_eval_time, pd.Series):
        ax2.plot(
            valid_df["step"],
            spline_eval_time,
            label="Cumulative Mean Eval Time",
            color=eval_time_color,
            linewidth=2,
        )
    else:
        ax2.plot(
            valid_df["step"],
            spline_eval_time(valid_df["step"]),
            label="Cumulative Mean Eval Time",
            color=eval_time_color,
            linewidth=2,
        )

    ax2.set_ylabel(
        "Cumulative Mean Evaluation Time (s)", fontsize=16, color=eval_time_color
    )
    ax2.tick_params(axis="y", labelcolor=eval_time_color)

    # Plot normalized cumulative mean errors as bars in the inset
    ax_inset = fig.add_subplot(gs[1])
    ax_inset.bar(
        df["step"],
        df["normalized_cumulative_mean_errors"],
        color="red",
        alpha=0.3,
        label="Cumulative Mean Error Ratio",
    )
    ax_inset.set_xlabel("Step Number", fontsize=12)
    ax_inset.set_ylabel("Error Ratio", fontsize=12, color="red")
    ax_inset.set_ylim(0, 1)
    ax_inset.tick_params(axis="y", labelcolor="red")
    ax_inset.legend(loc="upper left", fontsize=10)
    ax_inset.set_title("Cumulative Mean Error Ratio Over Steps", fontsize=14)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=12)

    # Improve layout
    fig.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, figure_name)
    try:
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully at {fig_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)


def preprocess_data(df: pd.DataFrame):
    if "step" not in df.columns:
        df = df.reset_index(drop=True).reset_index().rename(columns={"index": "step"})

    df.sort_values("step", inplace=True)

    error_conditions = (
        df[["f1", "accuracy", "evaluation_time"]].isna().any(axis=1)
    )
    error_steps = df[error_conditions]["step"].tolist()
    num_errors = len(error_steps)
    df["error"] = error_conditions.astype(int)

    df["cumulative_errors"] = df["error"].cumsum()
    df["cumulative_mean_errors"] = df["cumulative_errors"] / (
        df["step"] + 1
    )  # Adding 1 to avoid division by zero

    # Normalize cumulative mean errors to range [0, 1]
    df["normalized_cumulative_mean_errors"] = (
        df["cumulative_mean_errors"] / 1
    )  # Already a ratio, but kept for clarity

    # Set 'f1' to 0 where errors occur
    valid_df = df[~error_conditions].copy()
    # valid_df = df.copy()
    valid_df.loc[error_conditions, ["f1", "accuracy", "evaluation_time"]] = 0.0

    # Compute cumulative mean and standard deviation of 'f1'
    valid_df["cumulative_mean_f1"] = valid_df["f1"].expanding().mean()
    valid_df["cumulative_std_f1"] = (
        valid_df["f1"].expanding().std().fillna(0)
    )

    # Compute cumulative maximum of 'f1'
    valid_df["cumulative_max_f1"] = valid_df["f1"].cummax()

    # Compute cumulative mean of 'evaluation_time' but removing errors as time cannot be infinite if we want to plot
    eval_time_df = df[~error_conditions].copy()
    eval_time_df.reset_index(drop=True, inplace=True)

    # Compute cumulative minimum of 'evaluation_time'
    eval_time_df["cumulative_min_eval_time"] = eval_time_df["evaluation_time"].cummin()
    eval_time_df["cumulative_mean_eval_time"] = (
        eval_time_df["evaluation_time"].expanding().mean()
    )

    return valid_df, eval_time_df


def plot_experiments(
    data: List[ExperimentPlotData], figure_name="", output_dir="plots", plot_mean=True
):
    def prepare_timestamps(df: pd.DataFrame):
        # Ensure 'timestamp' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Calculate the relative time in hours since the first evaluation
        df["relative_time"] = (
            df["timestamp"] - first_timestamp
        ).dt.total_seconds() / 3600.0  # Convert to hours

        # Filter data to include only the first 24 hours
        df_24h = df[df["relative_time"] <= 24]

        # Sort the data by relative time
        df_24h.sort_values("relative_time", inplace=True)
        return df_24h

    # Create a single figure with three subplots in a column
    fig, (ax1, ax1_time) = plt.subplots(nrows=2, ncols=1, figsize=(10, 16), sharex=True)

    group_name = data[0].alias_name.split("|")[0]
    for exp_data in data:
        alias_name = (
            exp_data.alias_name.split("|")[1]
            if "|" in exp_data.alias_name
            else exp_data.alias_name
        )
        df = exp_data.dataframe.copy()
        first_timestamp = df["timestamp"].min()

        df, eval_time_df = preprocess_data(df)
        error_df = df.copy()

        df_24h = prepare_timestamps(df)
        eval_time_df_24h = prepare_timestamps(eval_time_df)
        error_df_24h = prepare_timestamps(error_df)

        # Create time grid for interpolation
        time_grid = np.linspace(0, 24, num=1000)

        # Interpolate cumulative_mean_f1 and cumulative_max_f1 onto time_grid
        f1_interp = np.interp(
            time_grid,
            df_24h["relative_time"],
            df_24h[f'cumulative_{"mean" if plot_mean else "max"}_f1'],
        )

        # Interpolate cumulative_mean_eval_time onto time_grid
        eval_time_interp = np.interp(
            time_grid,
            eval_time_df_24h["relative_time"],
            eval_time_df_24h[f'cumulative_{"mean" if plot_mean else "min"}_eval_time'],
        )

        # Plot cumulative mean macro F1 score
        ax1.plot(
            time_grid,
            f1_interp,
            label=f"{alias_name}",
            color=exp_data.color,
            linestyle=exp_data.linestyle,
            # marker=exp_data.markerstyle
        )

        # Plot cumulative mean evaluation time on ax2
        ax1_time.plot(
            time_grid,
            eval_time_interp,
            label=f"{alias_name}",
            color=exp_data.color,
            linestyle=exp_data.linestyle,
            # marker=exp_data.markerstyle
        )

    # Customize the first plot (Macro F1 Score)
    ax1.set_xlim(0, 24)
    ax1.set_ylabel("Macro F1")
    ax1.set_title(f"Mean Macro F1 Score ({group_name})")
    ax1.grid(True)
    ax1.set_ylim(auto=True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1)

    # Customize the second plot (Evaluation Time)
    ax1_time.set_xlim(0, 24)
    ax1_time.set_xlabel("Time (Hours)")
    ax1_time.set_ylabel("Evaluation Time (Seconds)")
    ax1_time.set_title("Mean Evaluation Time")
    ax1_time.grid(True)
    ax1_time.set_ylim(auto=True)
    lines1_time, labels1_time = ax1_time.get_legend_handles_labels()
    ax1_time.legend(lines1_time, labels1_time)

    # Improve layout
    # fig.tight_layout()

    # Display the combined figure
    # plt.show()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, figure_name)
    try:
        fig.savefig(f"{fig_path}.svg", dpi=300)
        fig.savefig(f"{fig_path}.pdf", dpi=300)
        fig.savefig(f"{fig_path}.png", dpi=300)
        print(f"Plot saved successfully at {fig_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)


import os
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.visualization.scatter import Scatter


def is_dominated(baseline_points, candidate):
    """
    Check if candidate point is dominated by any baseline point.
    baseline_points: array of [evaluation_time, f1] points
    candidate: [evaluation_time, f1] point
    Returns True if candidate is dominated by any baseline point
    """
    eval_time_candidate = candidate[1]
    f1_candidate = candidate[0]

    if f1_candidate == np.inf or f1_candidate == -np.inf or f1_candidate == np.nan:
        return True

    for baseline in baseline_points:
        eval_time_baseline = baseline[1]
        f1_baseline = baseline[0]

        if eval_time_candidate > eval_time_baseline and f1_candidate > f1_baseline:
            return True
    return False


def create_stepwise_points(points):
    """
    Convert a set of points into a stepwise pattern by adding intermediate points.
    Points should be sorted by evaluation_time (x-axis).
    """
    if len(points) < 2:
        return points

    stepwise_points = []
    for i in range(len(points) - 1):
        current_point = points[i]
        next_point = points[i + 1]

        # Add current point
        stepwise_points.append(current_point)
        # Add horizontal line point (same y as current, x as next)
        stepwise_points.append([next_point[0], current_point[1]])

    # Add the last point
    stepwise_points.append(points[-1])

    return np.array(stepwise_points)


def plot_pareto_front(
    experiments_data: List[ExperimentPlotData],
    dataset_name: str = "",
    figure_name="pareto_front",
    output_dir="plots",
):
    """
    Plot Pareto fronts for multiple experiments using pymoo for non-dominated sorting
    and scatter plot visualization. Each ExperimentPlotData's DataFrame should have
    columns "f1" and "evaluation_time". The objective is to maximize f1 and
    minimize evaluation_time.

    Parameters:
      experiments_data (List[ExperimentPlotData]): List of objects that include attributes:
          - alias_name: str, may be formatted as "group|name".
          - dataframe: pandas DataFrame with at least "f1" and "evaluation_time".
          - color: color for plotting.
          - linestyle: (unused in pymoo scatter) can be used if needed.
          - markerstyle: marker style string for plotting.
      figure_name (str): Base name for saving the figure.
      output_dir (str): Directory where plot files will be saved.
    """

    # Assume the first experiment is the baseline.
    baseline_exp = experiments_data[0]
    # Derive group name and update alias if formatted as "group|alias".
    group_name = baseline_exp.alias_name.split("|")[0]
    if "|" in baseline_exp.alias_name:
        baseline_alias = baseline_exp.alias_name.split("|")[1]
    else:
        baseline_alias = baseline_exp.alias_name

    # Prepare the baseline DataFrame and filter out NaNs.
    baseline_df = baseline_exp.dataframe.copy()
    baseline_df = baseline_df[
        ~baseline_df["f1"].isna() & ~baseline_df["evaluation_time"].isna()
    ]

    # Create objectives for baseline: (minimize -f1, evaluation_time)
    baseline_objectives = np.column_stack(
        (-baseline_df["f1"].values, baseline_df["evaluation_time"].values)
    )

    # Compute the Pareto front indices for baseline.
    pareto_indices = NonDominatedSorting().do(
        baseline_objectives, only_non_dominated_front=True
    )
    baseline_pf = baseline_df.iloc[pareto_indices].copy()
    # Sort baseline Pareto front by evaluation_time (ascending)
    baseline_pf.sort_values(by="evaluation_time", inplace=True)

    # Get baseline objectives for dominance checks.
    baseline_pf_objectives = np.column_stack(
        (-baseline_pf["f1"].values, baseline_pf["evaluation_time"].values)
    )

    # Setup the pymoo scatter plot.
    plot = Scatter(
        title=f"Pareto Front {dataset_name}",
        labels=["Evaluation Time (Seconds)", "Macro F1" if dataset_name != "DROP" and dataset_name != "SQUAD" else "F1"],
        tight_layout=True,
        legend=True,
    )

    # For baseline, extract points for plotting and create stepwise pattern
    base_points = baseline_pf[["evaluation_time", "f1"]].values
    stepwise_points = create_stepwise_points(base_points)

    # Plot baseline as a stepwise line and original points as markers
    plot.add(
        stepwise_points,
        plot_type="line",
        color=baseline_exp.color,
        linewidth=2,
        label=baseline_alias,
    )
    plot.add(
        base_points, s=100, marker=baseline_exp.markerstyle, color=baseline_exp.color
    )

    # Iterate through the rest of the experiments.
    for exp_data in experiments_data[1:]:
        # Determine alias (if "group|alias", take second part).
        alias_name = (
            exp_data.alias_name.split("|")[1]
            if "|" in exp_data.alias_name
            else exp_data.alias_name
        )

        # Process the experiment's DataFrame: filter out NaNs.
        df = exp_data.dataframe.copy()
        df = df[~df["f1"].isna() & ~df["evaluation_time"].isna()]
        if df.empty:
            continue

        # For each candidate point, prepare objectives.
        candidates = np.column_stack(
            (-df["f1"].values, df["evaluation_time"].values)
        )

        # Filter candidates that are NOT dominated by any baseline Pareto point.
        non_dominated_candidates = []
        for idx, cand in enumerate(candidates):
            if not is_dominated(baseline_pf_objectives, cand):
                non_dominated_candidates.append(df.iloc[idx])

        if non_dominated_candidates:
            cand_df = pd.DataFrame(non_dominated_candidates)
            points = cand_df[["evaluation_time", "f1"]].values
            # Plot only markers for these non-dominated points.
            plot.add(
                points,
                s=80,
                marker=exp_data.markerstyle,
                alpha=0.7,
                color=exp_data.color,
                label=alias_name,
            )

    # Show the plot.
    # plot.show()

    plot.plot_if_not_done_yet()

    # Save the plot.
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, figure_name)
    try:
        plot.fig.savefig(f"{fig_path}.svg", dpi=300, pad_inches=0)
        plot.fig.savefig(f"{fig_path}.pdf", dpi=300, pad_inches=0)
        plot.fig.savefig(f"{fig_path}.png", dpi=300, pad_inches=0)
        print(f"Plot saved successfully at {fig_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")


def plot_wins_by_configuration(tasks, config_names, output_dir="plots", figure_name="wins_by_config"):
    """
    Plot grouped bar chart showing the winning ratio (wins divided by total generated solutions)
    for each configuration, with adjustable padding between configuration groups and between
    configurations within each group.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    plt.figure(figsize=(12, 8))
    
    # Layout parameters
    bar_width = 0.25
    group_spacing = 0.1       # Space between configuration groups (Low, Mod, High)
    config_padding = 0.05      # Padding between configurations within each group
    category_labels = ["Low", "Mod", "High"]

    categories = len(category_labels)
    bars_per_category = 3     # Three configurations per group (e.g., LIAR, Med, Max)
    # Total width of a group = (bar_width + config_padding) for each configuration minus the extra padding at the end.
    group_width = bars_per_category * (bar_width + config_padding) - config_padding

    # Calculate the starting x position for each group.
    x_base = np.arange(categories) * (group_width + group_spacing)
    # For each group, compute the x positions of the three configurations with the added padding.
    x_indices = np.concatenate([
        x_base[i] + np.arange(bars_per_category) * (bar_width + config_padding)
        for i in range(categories)
    ])
    
    # Store all winning ratios for CSV export
    all_data = []
    
    for task_idx, task in enumerate(tasks):
        win_ratios = []
        # Compute baseline Pareto front for the task (using the actual zero-shot baseline)
        zeroshot_baseline_exp = task.get('zeroshot_baseline_exp', None)
        if zeroshot_baseline_exp is None:
            print(f"Warning: No zero-shot baseline found for {task['label']}, using default empty baseline")
            baseline_pf_objectives = np.zeros((1,2))
        else:
            baseline_df = zeroshot_baseline_exp.dataframe.copy()
            baseline_df = baseline_df[~baseline_df["f1"].isna() & ~baseline_df["evaluation_time"].isna()]
            if baseline_df.empty:
                baseline_pf_objectives = np.zeros((1,2))
            else:
                baseline_objectives = np.column_stack((
                    -baseline_df["f1"].values,
                    baseline_df["evaluation_time"].values
                ))
                pareto_indices = NonDominatedSorting().do(baseline_objectives, only_non_dominated_front=True)
                baseline_pf = baseline_df.iloc[pareto_indices]
                baseline_pf_objectives = np.column_stack((
                    -baseline_pf["f1"].values,
                    baseline_pf["evaluation_time"].values
                ))

        # Build a mapping from alias_name to ExperimentPlotData for this task
        exp_map = {exp.alias_name: exp for exp in task['experiments_data']}
        # For each config in config_names, compute win ratio or set to 0 if missing
        for config in config_names:
            exp_data = exp_map.get(config)
            if exp_data is not None:
                df = exp_data.dataframe.copy()
                df = df[~df["f1"].isna() & ~df["evaluation_time"].isna()]
                if not df.empty:
                    total_solutions = df.shape[0]
                    win_count = sum(
                        1 for cand in np.column_stack((-df["f1"], df["evaluation_time"]))
                        if not is_dominated(baseline_pf_objectives, cand)
                    )
                    win_ratio = win_count / total_solutions
                else:
                    win_ratio = 0
            else:
                win_ratio = 0
            win_ratios.append(win_ratio)
            all_data.append({
                "task": task['label'],
                "configuration": config,
                "win_ratio": win_ratio
            })

        # Plot bars with a task-specific offset so that bars from different tasks are not completely overlapping.
        task_offset = bar_width * task_idx / len(tasks)
        plt.bar(
            x_indices + task_offset,
            win_ratios,
            width=bar_width / len(tasks),
            color=task['color'],
            label=task['label'],
            edgecolor='white',
            linewidth=0.5
        )

    # Set x-axis ticks at the center of each bar with font size 18 for configuration names.
    plt.xticks(x_indices + bar_width/2, config_names, rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-bar_width, x_indices[-1] + bar_width + group_spacing)
    
    # Draw vertical lines to separate configuration groups.
    for i in range(1, categories):
        plt.axvline(x_base[i] - group_spacing/2, color='gray', linestyle=':', alpha=0.4)

    plt.title("Winning Ratio by WS Prior")
    # Place the legend inside the plot in the upper left.
    plt.legend(loc="upper left")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save the outputs
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, figure_name)
    plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{fig_path}.pdf", dpi=300, bbox_inches='tight')
    pd.DataFrame(all_data).to_csv(f"{fig_path}_data.csv", index=False)
    
    # plt.show()

# --- Modernized main_combined using DataLoader and config structure ---
from text_classification.src.data_loading import DataLoader

def build_experiment_plot_data(data_dict, color_map, marker_map, linestyle_map):
    plot_data = []
    configs = []
    
    alias_name_dict = {
        'baseline': "Zero-shot",
        'low': "Low",
        'moderate': "Mod",
        'high': "High"
    }
    
    # Process baseline first to ensure it's at the beginning of plot_data
    if "baseline" in data_dict:
        methods = data_dict["baseline"]
        for method, df in methods.items():
            alias_name = f"{alias_name_dict['baseline'].capitalize()}"
            color = color_map.get("baseline", "blue")
            marker = marker_map.get(method, "o")
            linestyle = linestyle_map.get(method, "-")
            plot_data.append(ExperimentPlotData(
                alias_name=alias_name,
                dataframe=df,
                color=color,
                linestyle=linestyle,
                markerstyle=marker,
            ))
            break  # Only add the first baseline config to the configs list, as it is the zero-shot baseline
            # Don't add baseline configs to the configs list
    
    # Then process low, moderate, high for the actual configs to plot
    for complexity in ["low", "moderate", "high"]:
        if complexity not in data_dict:
            continue
        
        methods = data_dict[complexity]
        for method, df in methods.items():
            alias_name = f"{alias_name_dict[complexity].capitalize()} ({method.capitalize()})"
            color = color_map.get(complexity, "black")
            marker = marker_map.get(method, "o")
            linestyle = linestyle_map.get(method, "-")
            plot_data.append(ExperimentPlotData(
                alias_name=alias_name,
                dataframe=df,
                color=color,
                linestyle=linestyle,
                markerstyle=marker,
            ))
            configs.append(alias_name)
    return plot_data, configs

def main_combined():

    # Ensure the output root directory exists
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    # --- Loader and config setup for both domains ---
    # Classification
    classification_config_path = "/home/coder/autogoal/experiments/text_classification/configs/multi-objective/candidates.yaml"
    classification_data_root = "/home/coder/autogoal/experiments/text_classification/data/experience_store"
    classification_loader = DataLoader(classification_config_path, classification_data_root)
    # Generation
    generation_config_path = "/home/coder/autogoal/experiments/text_generation/configs/multi-objective/candidates.yaml"
    generation_data_root = "/home/coder/autogoal/experiments/text_generation/data/experience_store"
    # Import DataLoader from text_generation (assume same interface)
    try:
        from text_generation.src.data_loading import DataLoader as GenerationDataLoader
    except ImportError:
        GenerationDataLoader = DataLoader  # fallback if not available
    generation_loader = GenerationDataLoader(generation_config_path, generation_data_root)

    color_map = {
        "baseline": "blue",
        "low": "green",
        "moderate": "orange",
        "high": "red",
    }
    # Dataset-level color map for bars
    dataset_color_map = {
        "liar": "#1f77b4",
        "sst2": "#ff7f0e",
        "meld": "#2ca02c",
        "ag_news": "#d62728",
        "squad": "#9467bd",  # purple
        "drop": "#8c564b",   # brown
        # Add more datasets here as needed
    }
    marker_map = {
        "zeroshot": "o",
        "knn25": "s",
        "knn50": "D",
        "liar": "v",
        "median": "*",
        "max": "d",
    }
    linestyle_map = {
        "zeroshot": "-",
        "knn25": "-.",
        "knn50": "--",
        "liar": "-",
        "median": "-.",
        "max": "--",
    }

    # Datasets by domain
    classification_datasets = ["liar", "sst2", "meld", "ag_news"]
    generation_datasets = ["squad", "drop"]
    datasets = classification_datasets + generation_datasets

    all_tasks = []
    configs_master = None

    for dataset in datasets:
        # Select loader based on dataset
        if dataset in classification_datasets:
            loader = classification_loader
        else:
            loader = generation_loader

        data_dict = loader.load_all_data_for_dataset(dataset)
        plot_data, configs = build_experiment_plot_data(data_dict, color_map, marker_map, linestyle_map)

        # Find the zero-shot baseline experiment from the plot_data
        zeroshot_baseline_exp = None
        for exp in plot_data:
            if exp.alias_name.lower() == "zero-shot":
                zeroshot_baseline_exp = exp
                break

        all_tasks.append({
            "experiments_data": plot_data,
            "color": dataset_color_map.get(dataset, "black"),
            "label": dataset.upper(),
            "zeroshot_baseline_exp": zeroshot_baseline_exp,
        })
        if configs_master is None:
            configs_master = configs

    # --- Sort datasets by mean winning rate (using correct baseline logic) ---
    # For each dataset, compute mean win rate using the actual zero-shot baseline
    mean_win_rates = []
    for task in all_tasks:
        exp_map = {exp.alias_name: exp for exp in task['experiments_data']}
        # Use the zero-shot baseline for this dataset
        zeroshot_baseline_exp = task.get('zeroshot_baseline_exp', None)
        if zeroshot_baseline_exp is None:
            print(f"Warning: No zero-shot baseline found for {task['label']}")
            mean_win_rates.append(0)
            continue
        
        baseline_df = zeroshot_baseline_exp.dataframe.copy()
        baseline_df = baseline_df[~baseline_df["f1"].isna() & ~baseline_df["evaluation_time"].isna()]
        if baseline_df.empty:
            print(f"Warning: Zero-shot baseline data is empty for {task['label']}")
            baseline_pf_objectives = np.zeros((1,2))
        else:
            baseline_objectives = np.column_stack((-baseline_df["f1"].values, baseline_df["evaluation_time"].values))
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            pareto_indices = NonDominatedSorting().do(baseline_objectives, only_non_dominated_front=True)
            baseline_pf = baseline_df.iloc[pareto_indices]
            baseline_pf_objectives = np.column_stack((-baseline_pf["f1"].values, baseline_pf["evaluation_time"].values))

        win_rates = []
        for config in configs_master:
            exp_data = exp_map.get(config)
            if exp_data is not None:
                df = exp_data.dataframe.copy()
                df = df[~df["f1"].isna() & ~df["evaluation_time"].isna()]
                if not df.empty:
                    total_solutions = df.shape[0]
                    win_count = sum(
                        1 for cand in np.column_stack((-df["f1"], df["evaluation_time"]))
                        if not is_dominated(baseline_pf_objectives, cand)
                    )
                    win_ratio = win_count / total_solutions
                else:
                    win_ratio = 0
            else:
                win_ratio = 0
            win_rates.append(win_ratio)
        mean_win_rates.append(np.mean(win_rates))

    # Sort all_tasks by mean_win_rates (ascending, so highest is last)
    sorted_indices = np.argsort(mean_win_rates)
    all_tasks = [all_tasks[i] for i in sorted_indices]


    # Use the configs from the first dataset as x-axis labels (assumes all datasets have same configs)
    # plot_wins_by_configuration(all_tasks, configs_master)

    # --- Plot Pareto fronts for each task (dataset) ---
    # for task in all_tasks:
    #     # Use the experiments_data for this task
    #     experiments_data = task["experiments_data"]
    #     # Use the dataset label for the figure name
    #     dataset_label = task["label"]
    #     figure_name = f"pareto_front_{dataset_label}"
    #     plot_pareto_front(
    #         experiments_data,
    #         dataset_label,
    #         figure_name=figure_name,
    #         output_dir=OUTPUT_ROOT_DIR
    #     )

if __name__ == "__main__":
    main_combined()
