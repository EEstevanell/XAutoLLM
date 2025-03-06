import itertools
import os
import json
from typing import Dict, List
from matplotlib.axes import Axes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import statsmodels.stats.multitest as multitest
from scipy import stats
from scipy.stats import mannwhitneyu

class ExperimentGroupPlot():
    def __init__(
        self,
        alias_name: str,
        linestyle: str,
        color: str,
        dataframes: List[pd.DataFrame],
    ):
        self.alias_name = alias_name
        self.linestyle = linestyle
        self.dataframes = dataframes
        self.color = color

# Define the root directory for experiences and output
EXPERIENCE_ROOT_DIR = 'experience_store'

def parse_timestamp(date_folder, json_filename):
    """
    Combine date and time from folder and file to create a datetime object.
    """
    date_part = date_folder
    time_part = json_filename.split('-')[0]  # 'hh:mm:ss'
    datetime_str = f"{date_part} {time_part}"
    try:
        return datetime.strptime(datetime_str, "%Y-%m-%d %H_%M_%S")
    except ValueError as ve:
        print(f"Timestamp parsing error for {datetime_str}: {ve}")
        return None

def load_data_for_alias(alias):
    """
    Load and process JSON data for a given alias.
    Returns a pandas DataFrame sorted by timestamp with macro_f1, accuracy, and evaluation_time.
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
            if not json_file.endswith('.json'):
                continue  # Skip non-JSON files

            json_path = os.path.join(date_path, json_file)
            try:
                with open(json_path, 'r') as f:
                    content = json.load(f)

                timestamp = parse_timestamp(date_folder, json_file)
                if timestamp is None:
                    continue  # Skip if timestamp parsing failed

                # Extract metrics
                macro_f1 = content.get('f1', np.nan)
                accuracy = content.get('accuracy', np.nan)
                evaluation_time = content.get('evaluation_time', np.nan)

                # finetuning_method will always be first algorithm here
                algorithm = dict(content["algorithms"][0])
                finetuning_method = list(algorithm)[0]
                llm = list(algorithm[finetuning_method]["inner_model"]["value"])[0]
                params = list(algorithm[finetuning_method])[1:]

                # Append to data
                data.append({
                    'alias': alias,
                    'timestamp': timestamp,
                    'macro_f1': macro_f1,
                    'accuracy': accuracy,
                    'evaluation_time': evaluation_time,
                    'finetuning_method': finetuning_method,
                    'llm': str(llm).removeprefix("WORD_EMB_").removeprefix("TEXT_GEN_").replace("_", " "),
                    'parameters': params,
                })

            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue  # Skip corrupted or unreadable files

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_data():
    """
    Load data for all aliases and return a dictionary mapping alias names to DataFrames.
    """
    data = dict()
    
    # List all aliases
    aliases = [alias for alias in os.listdir(EXPERIENCE_ROOT_DIR) 
               if os.path.isdir(os.path.join(EXPERIENCE_ROOT_DIR, alias))]

    if not aliases:
        print(f"No aliases found in the directory: {EXPERIENCE_ROOT_DIR}")
        return data
    
    for alias in aliases:
        print(f"Processing alias: {alias}")
        data[alias] = load_data_for_alias(alias)
        
    return data

##################################################
# Additional Metric Computations
##################################################

def compute_auc_over_time(valid_df: pd.DataFrame) -> float:
    """
    Computes the area under the 'macro_f1' vs. time curve using the trapezoidal rule.
    Assumes 'timestamp' is in datetime format and sorted.
    Returns the integral over the first 24 hours.
    If no valid rows, returns np.nan.
    """
    if valid_df.empty:
        return np.nan
    # Sort by timestamp
    valid_df = valid_df.sort_values('timestamp')
    valid_df['relative_time_hours'] = (valid_df['timestamp'] - valid_df['timestamp'].min()).dt.total_seconds() / 3600.0
    # Truncate beyond 24 hours if needed
    valid_df = valid_df[valid_df["relative_time_hours"] <= 24]
    if valid_df.empty:
        return np.nan
    x = valid_df['relative_time_hours'].values
    y = valid_df['macro_f1'].values
    # Compute AUC using trapezoidal rule
    return trapezoid(y, x)

def compute_convergence_rate(valid_df: pd.DataFrame) -> float:
    """
    A simple measure of how quickly improvements happen.
    For example, compute the slope of macro_f1 with respect to time in the first half of training.
    Return np.nan if insufficient data.
    """
    if valid_df.empty:
        return np.nan
    # Sort by timestamp
    valid_df = valid_df.sort_values('timestamp')
    valid_df['relative_time_hours'] = (valid_df['timestamp'] - valid_df['timestamp'].min()).dt.total_seconds() / 3600.0
    half_time = 12.0  # first half = first 12 hours
    df_half = valid_df[valid_df["relative_time_hours"] <= half_time]
    if len(df_half) < 2:
        return np.nan
    # Basic linear approximation: slope = (f1_end - f1_start) / (time_end - time_start)
    f1_start = df_half.iloc[0]['macro_f1']
    f1_end = df_half.iloc[-1]['macro_f1']
    t_start = df_half.iloc[0]['relative_time_hours']
    t_end = df_half.iloc[-1]['relative_time_hours']
    if (t_end - t_start) == 0:
        return np.nan
    return (f1_end - f1_start) / (t_end - t_start)

def calculate_time_to_threshold(valid_df: pd.DataFrame, global_max_f1: float, threshold_percent: float) -> float:
    """
    Calculate time (in hours) until macro_f1 reaches threshold_percent% of global_max_f1.
    Returns None if the threshold is never reached.
    """
    if valid_df.empty or global_max_f1 is None or np.isnan(global_max_f1):
        return None
    
    valid_df = valid_df.sort_values("timestamp")
    start_time = valid_df["timestamp"].min()
    target_f1 = global_max_f1 * (threshold_percent / 100.0)
    
    # Find if/when we cross the threshold
    threshold_rows = valid_df[valid_df["macro_f1"] >= target_f1].copy()
    if threshold_rows.empty:
        return None  # Threshold never reached
    
    first_threshold_time = threshold_rows["timestamp"].min()
    time_to_threshold = (first_threshold_time - start_time).total_seconds() / 3600.0
    return time_to_threshold

def apply_correction(results_df: pd.DataFrame, method: str = 'fdr_bh') -> pd.DataFrame:
    """
    Applies multiple comparison correction to p-values in results DataFrame
    
    Parameters:
        results_df: DataFrame with columns ['group1', 'group2', 'statistic', 'p_value', 'effect_size']
        method: Correction method ('fdr_bh', 'bonferroni', 'holm')
    """
    if results_df.empty:
        return results_df
        
    # Extract p-values
    p_values = results_df['p_value'].values
    
    # Apply correction
    reject, p_corrected, _, _ = multitest.multipletests(
        p_values,
        alpha=0.05,
        method=method
    )
    
    # Add corrected p-values and significance
    results_df['p_corrected'] = p_corrected
    results_df['significant'] = reject
    
    return results_df

def test_normality(df: pd.DataFrame, column: str = 'max_f1') -> bool:
    """
    Enhanced normality test using Shapiro-Wilk test
    Parameters:
        df: DataFrame with the data
        column: Column to test for normality (default: max_f1)
    Returns:
        bool: True if data is normally distributed
    """
    # Add Q-Q plot visualization
    from scipy.stats import probplot
    import matplotlib.pyplot as plt
    
    # for group in df["group_name"].unique():
    #     group_data = df[df["group_name"] == group][column].dropna()
    #     if len(group_data) >= 3:
    #         probplot(group_data, dist="norm", plot=plt)
    #         plt.title(f"Q-Q Plot for {group}")
    #         plt.show()
            
    from scipy.stats import shapiro
    results = []
    
    for group in df["group_name"].unique():
        group_data = df[df["group_name"] == group][column].dropna()
        if len(group_data) >= 3:  # Minimum sample size for Shapiro-Wilk
            _, p_value = shapiro(group_data)
            results.append(p_value > 0.05)
    
    return all(results)

import numpy as np
import pandas as pd

def compute_confidence_intervals(group_data: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    """
    Computes mean +/- t-based confidence intervals for max_f1 and mean_f1 per group_name.
    """
    from scipy.stats import t
    results = []
    for group_name, df_g in group_data.groupby('group_name'):
        max_f1_vals = df_g['max_f1'].dropna().values
        mean_f1_vals = df_g['mean_f1'].dropna().values
        
        n_max = len(max_f1_vals)
        n_mean = len(mean_f1_vals)
        max_f1_lower = max_f1_upper = np.nan
        mean_f1_lower = mean_f1_upper = np.nan
        
        if n_max >= 2:
            t_crit = t.ppf(1 - alpha/2, df=n_max - 1)
            mean_max_f1 = np.mean(max_f1_vals)
            std_max_f1 = np.std(max_f1_vals, ddof=1)
            se_max_f1 = std_max_f1 / np.sqrt(n_max)
            margin_max_f1 = t_crit * se_max_f1
            max_f1_lower = mean_max_f1 - margin_max_f1
            max_f1_upper = mean_max_f1 + margin_max_f1
        else:
            mean_max_f1 = np.nan
        
        if n_mean >= 2:
            t_crit = t.ppf(1 - alpha/2, df=n_mean - 1)
            mean_mean_f1 = np.mean(mean_f1_vals)
            std_mean_f1 = np.std(mean_f1_vals, ddof=1)
            se_mean_f1 = std_mean_f1 / np.sqrt(n_mean)
            margin_mean_f1 = t_crit * se_mean_f1
            mean_f1_lower = mean_mean_f1 - margin_mean_f1
            mean_f1_upper = mean_mean_f1 + margin_mean_f1
        else:
            mean_mean_f1 = np.nan
        
        results.append({
            'group_name': group_name,
            'mean_max_f1': mean_max_f1,
            'max_f1_CI_lower': max_f1_lower,
            'max_f1_CI_upper': max_f1_upper,
            'mean_mean_f1': mean_mean_f1,
            'mean_f1_CI_lower': mean_f1_lower,
            'mean_f1_CI_upper': mean_f1_upper,
        })
    return pd.DataFrame(results)

def run_friedman_test(summary_df: pd.DataFrame, metric: str = 'max_f1'):
    from scipy.stats import friedmanchisquare
    pivot_df = summary_df.pivot(index='df_index', columns='group_name', values=metric)
    pivot_df.dropna(axis=0, inplace=True)
    
    if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2:
        print(f"Not enough data to run Friedman test for {metric}.")
        return None
    
    groups = [pivot_df[col].values for col in pivot_df.columns]
    stat, p = friedmanchisquare(*groups)
    print(f"Friedman test for '{metric}': stat={stat:.3f}, p-value={p:.3g}")
    return stat, p

def pairwise_wilcoxon(summary_df: pd.DataFrame, metric: str = 'max_f1'):
    from scipy.stats import wilcoxon
    pivot_df = summary_df.pivot(index='df_index', columns='group_name', values=metric)
    pivot_df.dropna(axis=0, inplace=True)
    group_names = pivot_df.columns
    
    results = []
    for g1, g2 in itertools.combinations(group_names, 2):
        data1 = pivot_df[g1].values
        data2 = pivot_df[g2].values
        if len(data1) < 2:
            continue
        stat, p = wilcoxon(data1, data2, alternative='two-sided')
        results.append((g1, g2, stat, p))
    
    # Apply Bonferroni correction
    if results:
        p_values = [r[3] for r in results]
        reject, pvals_corrected, _, _ = multitest.multipletests(p_values, alpha=0.05, method='bonferroni')
        for i, (g1, g2, stat, p) in enumerate(results):
            print(f"Wilcoxon test for '{metric}' comparing {g1} vs {g2} -> "
                  f"stat={stat:.3f}, raw p-value={p:.3g}, corrected p-value={pvals_corrected[i]:.3g}, "
                  f"significant={reject[i]}")
    else:
        print(f"No valid pairwise comparisons for '{metric}'.")
    return results

def cliffs_delta(x, y):
    x = np.array(x)
    y = np.array(y)
    wins = 0
    losses = 0
    ties = 0
    
    for xi in x:
        for yi in y:
            if xi > yi:
                wins += 1
            elif xi < yi:
                losses += 1
            else:
                ties += 1
    
    n_x = len(x)
    n_y = len(y)
    delta = (wins - losses) / (n_x * n_y)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return delta, magnitude

def pairwise_cliffs_delta(summary_df: pd.DataFrame, metric: str = 'max_f1'):
    pivot_df = summary_df.pivot(index='df_index', columns='group_name', values=metric)
    pivot_df.dropna(axis=0, inplace=True)
    group_names = pivot_df.columns
    for g1, g2 in itertools.combinations(group_names, 2):
        data1 = pivot_df[g1].values
        data2 = pivot_df[g2].values
        if len(data1) == 0 or len(data2) == 0:
            continue
        delta, magnitude = cliffs_delta(data1, data2)
        print(f"Cliff’s Delta ({g1} vs {g2}, '{metric}'): {delta:.3f}, magnitude: {magnitude}")
        
def run_pairwise_tests(df: pd.DataFrame, parametric: bool = False, 
                      correction: str = "fdr_bh") -> pd.DataFrame:
    """
    Unified pairwise comparison function
    """
    from itertools import combinations
    from scipy.stats import ttest_rel, wilcoxon
    
    test_func = ttest_rel if parametric else wilcoxon
    results = []
    
    for g1, g2 in combinations(df["group_name"].unique(), 2):
        data1 = df[df["group_name"] == g1]["max_f1"]
        data2 = df[df["group_name"] == g2]["max_f1"]
        
        stat, p_val = test_func(data1, data2)
        
        # Replace simple effect size with comprehensive version
        effect_sizes = compute_comprehensive_effect_sizes(data1, data2)
        
        results.append({
            "group1": g1,
            "group2": g2,
            "statistic": stat,
            "p_value": p_val,
            "parametric_effects": effect_sizes["parametric"],
            "nonparametric_effects": effect_sizes["nonparametric"]
        })
    
    return apply_correction(pd.DataFrame(results), correction)

def run_repeated_measures_anova(summary_df: pd.DataFrame) -> dict:
    """
    Performs repeated measures ANOVA with effect sizes
    """
    from statsmodels.stats.anova import AnovaRM
    
    results = {}
    
    # Prepare data for repeated measures
    pivot_df = summary_df.pivot(
        index='df_index', 
        columns='group_name', 
        values=['max_f1', 'mean_f1']
    )
    
    # Remove any rows with missing values
    pivot_df.dropna(axis=0, inplace=True)
    
    if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2:
        return {"error": "Insufficient data for repeated measures ANOVA"}[3]
    
    # Run ANOVA for max_f1
    rm_anova_max = AnovaRM(
        data=summary_df,
        depvar='max_f1',
        subject='df_index',
        within=['group_name']
    ).fit()
    
    # Run ANOVA for mean_f1
    rm_anova_mean = AnovaRM(
        data=summary_df,
        depvar='mean_f1',
        subject='df_index',
        within=['group_name']
    ).fit()
    
    results["max_f1"] = {
        "f_value": rm_anova_max.anova_table["F Value"][0],
        "p_value": rm_anova_max.anova_table["Pr > F"][0],
        "df": (rm_anova_max.anova_table["Num DF"][0], 
               rm_anova_max.anova_table["Den DF"][0])
    }
    
    results["mean_f1"] = {
        "f_value": rm_anova_mean.anova_table["F Value"][0],
        "p_value": rm_anova_mean.anova_table["Pr > F"][0],
        "df": (rm_anova_mean.anova_table["Num DF"][0], 
               rm_anova_mean.anova_table["Den DF"][0])
    }
    
    return results

def compute_effect_sizes(group1_data: np.array, group2_data: np.array) -> dict:
    n1, n2 = len(group1_data), len(group2_data)
    pooled_std = np.sqrt(((n1-1)*np.var(group1_data) + (n2-1)*np.var(group2_data)) / (n1+n2-2))
    cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
    hedges_g = cohens_d * (1 - (3 / (4 * (n1 + n2) - 9)))
    return {'cohens_d': cohens_d, 'hedges_g': hedges_g}

def compute_comprehensive_effect_sizes(group1_data: np.array, 
                                     group2_data: np.array) -> dict:
    return {
        "parametric": compute_effect_sizes(group1_data, group2_data),
        "nonparametric": cliffs_delta(group1_data, group2_data)
    }

def perform_statistical_analysis(summary_df: pd.DataFrame) -> Dict:
    # Add sample size checks
    if len(summary_df) < 30:  # Rule of thumb for normality
        print("Warning: Small sample size, consider non-parametric tests")
    
    # Add Levene's test for homogeneity of variance
    from scipy.stats import levene
    groups = [group for _, group in summary_df.groupby('group_name')['max_f1']]
    levene_stat, levene_p = levene(*groups)
    
    results = {
        "descriptive_stats": {},
        "normality": {},
        "parametric": {},
        "nonparametric": {},
        "effect_sizes": {},
        "pairwise_comparisons": {}
    }
    
    # Add descriptive statistics
    for group in summary_df["group_name"].unique():
        group_data = summary_df[summary_df["group_name"] == group]
        results["descriptive_stats"][group] = {
            "mean_max_f1": float(group_data["max_f1"].mean()),
            "mean_mean_f1": float(group_data["mean_f1"].mean()),
            "std_max_f1": float(group_data["max_f1"].std()),
            "std_mean_f1": float(group_data["mean_f1"].std()),
            "median_max_f1": float(group_data["max_f1"].median()),
            "median_mean_f1": float(group_data["mean_f1"].median()),
            "iqr_max_f1": float(group_data["max_f1"].quantile(0.75) - 
                           group_data["max_f1"].quantile(0.25)),
            "iqr_mean_f1": float(group_data["mean_f1"].quantile(0.75) - 
                           group_data["mean_f1"].quantile(0.25))
        }
    
    # Both ANOVA and Friedman
    results["parametric"]["anova"] = run_repeated_measures_anova(summary_df)
    results["nonparametric"]["friedman"] = run_friedman_test(summary_df)
    
    # Add pairwise tests with multiple comparison corrections
    results["pairwise_comparisons"]["parametric"] = run_pairwise_tests(
        summary_df, parametric=True, correction="fdr_bh")
    results["pairwise_comparisons"]["nonparametric"] = run_pairwise_tests(
        summary_df, parametric=False, correction="fdr_bh")
    
    return results
        
def summarize_experiment_groups(data: List[ExperimentGroupPlot], output_json: str = None) -> pd.DataFrame:
    # Compute global max_f1 for each df_index
    global_max_dict = {}
    for group in data:
        for i, df in enumerate(group.dataframes):
            if "macro_f1" not in df.columns:
                continue
            df_valid = df[~df["macro_f1"].isin([np.nan, np.inf, -np.inf])]
            local_max = df_valid["macro_f1"].max() if not df_valid.empty else None
            if i not in global_max_dict:
                global_max_dict[i] = local_max
            else:
                if local_max is not None:
                    existing_val = global_max_dict[i]
                    global_max_dict[i] = max(existing_val, local_max) if existing_val else local_max

    rows = []
    for group in data:
        for i, df in enumerate(group.dataframes):
            df = df.copy()
            df["error"] = False
            for col in ["macro_f1", "accuracy"]:
                if col in df.columns:
                    df["error"] |= df[col].isnull() | df[col].isin([np.inf, -np.inf])

            # Filter out invalid macro_f1 rows
            if "macro_f1" in df.columns:
                valid_df = df[~df["macro_f1"].isnull() & ~df["macro_f1"].isin([np.inf, -np.inf])].copy()
            else:
                valid_df = pd.DataFrame()

            # Keep only valid timestamps
            df.dropna(subset=["timestamp"], inplace=True)
            if valid_df.empty or "timestamp" not in valid_df.columns:
                rows.append({
                    "group_name": group.alias_name,
                    "df_index": i,
                    "max_f1": None,
                    "mean_f1": None,
                    "std_f1": None,
                    "auc": None,
                    "convergence_rate": None,
                    "time_to_max_hours": None,
                    "time_to_50_hours": None,
                    "time_to_75_hours": None,
                    "time_to_90_hours": None,
                    "num_evaluations": 0,
                    "num_errors": df["error"].sum()
                })
                continue

            valid_df["timestamp"] = pd.to_datetime(valid_df["timestamp"])
            max_f1 = valid_df["macro_f1"].max()
            mean_f1 = valid_df["macro_f1"].mean()
            std_f1 = valid_df["macro_f1"].std()
            auc_val = compute_auc_over_time(valid_df)
            convergence = compute_convergence_rate(valid_df)
            
            # Calculate time to max within threshold
            global_max_f1 = global_max_dict.get(i, None)
            if not valid_df["macro_f1"].isnull().all() and global_max_f1 is not None:
                threshold = 0.015
                close_to_max = valid_df[valid_df["macro_f1"] >= (global_max_f1 - threshold)]
                if not close_to_max.empty:
                    first_max_time = close_to_max["timestamp"].min()
                    start_time = valid_df["timestamp"].min()
                    time_to_max = (first_max_time - start_time).total_seconds() / 3600
                else:
                    time_to_max = None
            else:
                time_to_max = None

            rows.append({
                "group_name": group.alias_name,
                "df_index": i,
                "max_f1": max_f1,
                "mean_f1": mean_f1,
                "std_f1": std_f1 if not np.isnan(std_f1) else None,
                "auc": auc_val if not np.isnan(auc_val) else None,
                "convergence_rate": convergence if not np.isnan(convergence) else None,
                "time_to_max_hours": time_to_max if time_to_max is not None else 24,
                "success_run": time_to_max is not None,
                "time_to_50_hours": calculate_time_to_threshold(valid_df, global_max_f1, 50),
                "time_to_75_hours": calculate_time_to_threshold(valid_df, global_max_f1, 75),
                "time_to_90_hours": calculate_time_to_threshold(valid_df, global_max_f1, 90),
                "num_evaluations": len(df),
                "num_errors": df["error"].sum()
            })
    
    # Build summary by group
    summary_df = pd.DataFrame(rows)
    group_avg = summary_df.groupby("group_name", as_index=False).agg({
        "max_f1": "mean",
        "mean_f1": "mean",
        "std_f1": "mean",
        "auc": "mean",
        "convergence_rate": "mean",
        "time_to_max_hours": "mean",
        "time_to_50_hours": "mean",
        "time_to_75_hours": "mean",
        "time_to_90_hours": "mean",
        "num_evaluations": "mean",
        "num_errors": "mean"
    })
    
    # Perform statistical tests
    statistical_results = perform_statistical_analysis(summary_df)
    
    report = generate_statistical_report(summary_df, statistical_results)
    
    print(report)
    
    if output_json:
        save_to_json(statistical_results, output_json)
    
    return summary_df, group_avg

def save_to_json(data: Dict, filename: str) -> None:
    """
    Safely save statistical results to JSON
    """
    try:
        # Flatten nested structures and convert to serializable format
        flattened_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened_data[f"{key}_{subkey}"] = subvalue
            else:
                flattened_data[key] = value
        
        # Convert to DataFrame and save
        df = pd.DataFrame([flattened_data])
        df.to_json(filename, orient='records', indent=2)
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")

def generate_statistical_report(summary_df: pd.DataFrame, results: Dict) -> str:
    report = []
    report.append("# Statistical Analysis Report\n")
    
    # Normality test results
    report.append("## Normality Tests")
    report.append(f"Data {'is' if test_normality(summary_df) else 'is not'} normally distributed\n")
    
    # Main test results
    if "anova" in results:
        report.append("## Repeated Measures ANOVA Results")
        report.append(f"F-value: {results['anova']['max_f1']['f_value']:.3f}")
        report.append(f"p-value: {results['anova']['max_f1']['p_value']:.3e}\n")
    
    return "\n".join(report)

if __name__ == '__main__':
    aliases = ['liar', 'meld', 'sst2', 'ag_news']
    counts = {alias: {'positive': 0, 'negative': 0, 'total': 0} for alias in aliases}
    others_counts = {alias: {'positive': 0, 'negative': 0, 'total': 0} for alias in aliases}
    
    data_dict = load_data()
    
    experiments_data_liar = [
        ExperimentGroupPlot(
            "LIAR",
            "-",
            "blue",
            [
                data_dict['liar_id_9 (rng_42)'],
                data_dict['liar_id_9 (rnd_123)'],
                data_dict['liar_id_9-rd_2024 (rnd_2024)'],
                data_dict['liar_id_9-rd_101 (rn_101)'],
                data_dict['liar_id_9-rd_707 (rn_707)'],
                data_dict['liar_id_9-rd_2029 (rn_2029)'],
            ]
        ), 
        ExperimentGroupPlot(
            "LIAR (f-pos cos k=0.5)",
            "-",
            "green",
            [
                data_dict['liar_warmstart_id_(f-pos cos k=0.5) (rn_42)'],
                data_dict['liar_warmstart_id_(f-pos cos k=0.5) (rn_123)'],
                data_dict['liar_warmstart_id_(f-pos cos k=0.5) (rn_2024)'],
                data_dict['liar_warmstart_id_(f-pos cos k=0.5)-rd_101 (rn_101)'],
                data_dict['liar_warmstart_id_(f-pos cos k=0.5)-rd_707 (rn_707)'],
                data_dict['liar_warmstart_id_(f-pos cos k=0.5)-rd_2029 (rn_2029)'],
            ]
        ), 
        ExperimentGroupPlot(
            "LIAR (f-pos + a-neg)",
            "-",
            "orange",
            [
                data_dict['liar_warmstart_id_(f-pos + a-neg) (rn_42)'],
                data_dict['liar_warmstart_id_(f-pos + a-neg) (rn_123)'],
                data_dict['liar_warmstart_id_(f-pos + a-neg) (rn_2024)'],
                data_dict['liar_warmstart_id_(f-pos + a-neg)-rd_101 (rn_101)'],
                data_dict['liar_warmstart_id_(f-pos + a-neg)-rd_707 (rn_707)'],
                data_dict['liar_warmstart_id_(f-pos + a-neg)-rd_2029 (rn_2029)'],
            ]
        ),
        ExperimentGroupPlot(
            "LIAR (f-pos + f-neg)",
            "-",
            "red",
            [
                data_dict['liar_warmstart_id_(f-pos f-neg cos k=0.5)-rd_42 (rnd_42)'],
                data_dict['liar_warmstart_id_(f-pos f-neg cos k=0.5)-rd_123 (rnd_123)'],
                data_dict['liar_warmstart_id_(f-pos f-neg cos k=0.5)-rd_2024 (rnd_2024)'],
                data_dict['liar_warmstart_id_(f-pos f-neg)-rd_101 (rn_101)'],
                data_dict['liar_warmstart_id_(f-pos f-neg)-rd_707 (rn_707)'],
                data_dict['liar_warmstart_id_(f-pos f-neg)-rd_2029 (rn_2029)'],
            ]
        ),
    ]
    experiments_data_sst2 = [
        ExperimentGroupPlot(
            "SST2",
            "-",
            "blue",
            [
                data_dict['sst2_id_9-rd_42 (rnd_42)'],
                data_dict['sst2_id_9-rd_123 (rnd_123)'],
                data_dict['sst2_id_9-rd_2024 (rnd_2024)'],
                data_dict['sst2_id_9-rd_101 (rn_101)'],
                data_dict['sst2_id_9-rd_707 (rn_707)'],
                data_dict['sst2_id_9-rd_2029 (rn_2029)'],
            ]
        ), 
        ExperimentGroupPlot(
            "SST2 (f-pos cos k=0.5)",
            "-",
            "green",
            [
                data_dict['sst2_warmstart_id_(f-pos cos k=0.5)-rd_42 (rn_42)'],
                data_dict['sst2_warmstart_id_(f-pos cos k=0.5)-rd_123 (rn_123)'],
                data_dict['sst2_warmstart_id_(f-pos cos k=0.5)-rd_2024 (rn_2024)'],
                data_dict['sst2_warmstart_id_(f-pos cos k=0.5)-rd_101 (rn_101)'],
                data_dict['sst2_warmstart_id_(f-pos cos k=0.5)-rd_707 (rn_707)'],
                data_dict['sst2_warmstart_id_(f-pos cos k=0.5)-rd_2029 (rn_2029)'],
            ]
        ), 
        ExperimentGroupPlot(
            "SST2 (f-pos + a-neg)",
            "-",
            "orange",
            [
                data_dict['sst2_warmstart_id_(f-pos + a-neg)-rd_42 (rnd_42)'],
                data_dict['sst2_warmstart_id_(f-pos + a-neg)-rd_123 (rnd_123)'],
                data_dict['sst2_warmstart_id_(f-pos + a-neg)-rd_2024 (rnd_2024)'],
                data_dict['sst2_warmstart_id_(f-pos + a-neg)-rd_101 (rn_101)'],
                data_dict['sst2_warmstart_id_(f-pos + a-neg)-rd_707 (rn_707)'],
                data_dict['sst2_warmstart_id_(f-pos + a-neg)-rd_2029 (rn_2029)'],
            ]
        ),
        ExperimentGroupPlot(
            "SST2 (f-pos + f-neg)",
            "-",
            "red",
            [
                data_dict['sst2_warmstart_id_(f-pos f-neg)-rd_42 (rnd_42)'],
                data_dict['sst2_warmstart_id_(f-pos f-neg)-rd_123 (rnd_123)'],
                data_dict['sst2_warmstart_id_(f-pos f-neg)-rd_2024 (rnd_2024)'],
                data_dict['sst2_warmstart_id_(f-pos f-neg)-rd_101 (rn_101)'],
                data_dict['sst2_warmstart_id_(f-pos f-neg)-rd_707 (rn_707)'],
                data_dict['sst2_warmstart_id_(f-pos f-neg)-rd_2029 (rn_2029)'],
            ]
        ),
    ]
    
    summary_df_liar, group_avg_liar = summarize_experiment_groups(experiments_data_liar, "liar_experiment_summary.json")
    summary_df_sst2, group_avg_sst2 = summarize_experiment_groups(experiments_data_sst2, "sst2_experiment_summary.json")
