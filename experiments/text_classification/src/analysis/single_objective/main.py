import itertools
from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from src.data_loading.data_loader import DataLoader
import statsmodels.stats.multitest as multitest
from scipy.integrate import trapezoid

def compute_auc_over_time(valid_df: pd.DataFrame, normalize: bool = True) -> float:
    """
    Computes the area under the 'macro_f1' vs. time curve using the trapezoidal rule.
    
    Args:
        valid_df: DataFrame containing 'timestamp' and 'macro_f1' columns.
        normalize: Whether to normalize the AUC to the time range (default: True).
                  When True, the result represents the average f1 score over time.
    
    Returns:
        The area under curve for the first 24 hours, or np.nan if insufficient data.
        When normalized, this value will be between 0 and 1 (representing average f1).
    """
    # Early validation
    if valid_df.empty or 'macro_f1' not in valid_df.columns or 'timestamp' not in valid_df.columns:
        return np.nan
        
    # Prepare the data (sort and convert timestamps)
    valid_df = valid_df.copy()
    valid_df = valid_df[~valid_df['macro_f1'].isnull() & ~valid_df['macro_f1'].isin([np.inf, -np.inf])]
    
    if len(valid_df) < 2:  # Need at least two points for AUC
        return np.nan
        
    valid_df = valid_df.sort_values('timestamp')
    start_time = valid_df['timestamp'].min()
    valid_df['relative_time_hours'] = (valid_df['timestamp'] - start_time).dt.total_seconds() / 3600.0
    
    # Truncate beyond 24 hours if needed
    max_hours = 24
    valid_df = valid_df[valid_df["relative_time_hours"] <= max_hours]
    if len(valid_df) < 2:
        return np.nan
        
    # Add boundary point at t=0 if missing
    if valid_df['relative_time_hours'].min() > 0:
        # First evaluation might not be at exactly t=0, assume initial f1=0
        first_row = valid_df.iloc[0].copy()
        first_row['relative_time_hours'] = 0
        first_row['macro_f1'] = 0
        valid_df = pd.concat([pd.DataFrame([first_row]), valid_df])
    
    # Add boundary point at t=24 if missing
    if valid_df['relative_time_hours'].max() < max_hours:
        # If experiment stopped before 24h, use the last f1 value until the end
        last_row = valid_df.iloc[-1].copy()
        last_row['relative_time_hours'] = max_hours
        valid_df = pd.concat([valid_df, pd.DataFrame([last_row])])
    
    # Extract arrays for integration
    x = valid_df['relative_time_hours'].values
    y = valid_df['macro_f1'].values
    
    # Compute AUC using trapezoidal rule
    auc = trapezoid(y, x)
    
    # Normalize if requested (this gives average performance)
    if normalize:
        time_span = x[-1] - x[0]
        if time_span > 0:
            auc = auc / time_span
    
    return auc

def calculate_time_to_threshold(valid_df: pd.DataFrame, global_max_f1: float, threshold_percent: float) -> Optional[float]:
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
        # Use 24.0 hours as the default when the threshold isn’t reached
        return 24  # Threshold never reached
    
    first_threshold_time = threshold_rows["timestamp"].min()
    time_to_threshold = (first_threshold_time - start_time).total_seconds() / 3600.0
    return time_to_threshold

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


class EffectSizeCalculator:
    """Handles the calculation of various effect size metrics"""
    @staticmethod
    def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + 
                            (ny-1)*np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std
    
    @staticmethod
    def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected version of Cohen's d)"""
        d = EffectSizeCalculator.cohens_d(x, y)
        n = len(x) + len(y)
        correction = 1 - (3 / (4 * (n - 2) - 1))
        return d * correction
    
    @staticmethod
    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Cliff's delta non-parametric effect size"""
        delta = 2 * (stats.mannwhitneyu(x, y).statistic / (len(x) * len(y))) - 1
        magnitude = np.select(
            [abs(delta) < 0.147, abs(delta) < 0.33, abs(delta) < 0.474],
            ['negligible', 'small', 'medium'],
            default='large'
        )
        return delta, magnitude


class MultipleComparisonHandler:
    """Handles multiple comparison corrections"""
    @staticmethod
    def adjust_pvalues(df: pd.DataFrame, p_col: str = 'p_value', 
                    method: str = 'fdr_bh') -> pd.DataFrame:
        """Apply multiple-testing correction to p-values"""
        pvals = df[p_col].values
        adjusted = multitest.multipletests(pvals, method=method)[1]
        df[p_col + '_adjusted'] = adjusted
        return df
    
    @staticmethod
    def get_significance_matrix(pvalues: np.ndarray, 
                              alpha: float = 0.05) -> np.ndarray:
        """Convert p-values to a boolean significance matrix"""
        return pvalues < alpha


class ExperimentHandler:
    """
        Main handler for single-objective experiment analysis.
        Expects experiments_data as a dictionary mapping candidate names to a list of DataFrames
        (each corresponding to a seed run).
    """
    def __init__(self, dataset_name: str, experiments_data: dict, alpha: float = 0.05):
        self.dataset_name = dataset_name
        self.experiments_data = experiments_data # dict: candidate -> list[pd.DataFrame]
        self.alpha = alpha
        self.summary_df = None
        self.group_avg = None
        self.statistical_results = None
        self.results_cache = {}
        
    def _bootstrap_ci(self, x: np.ndarray, y: np.ndarray, 
                 effect_func, n_iterations: int = 1000, 
                 ci_level: float = 0.95) -> Tuple[float, float]:
        """Computes bootstrap confidence intervals for effect sizes"""
        bootstrap_effects = []
        nx, ny = len(x), len(y)
        
        for _ in range(n_iterations):
            # Resample with replacement
            boot_x = np.random.choice(x, size=nx, replace=True)
            boot_y = np.random.choice(y, size=ny, replace=True)
            
            # Calculate effect size for this bootstrap sample
            effect = effect_func(boot_x, boot_y)
            bootstrap_effects.append(effect)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_effects, (1 - ci_level) * 100 / 2)
        ci_upper = np.percentile(bootstrap_effects, (1 + ci_level) * 100 / 2)
        
        return (ci_lower, ci_upper)
    
    def process_experiments(self) -> None:
        """Main pipeline to process all experiments"""
        self._build_summary_df()
        self._compute_group_averages()
        self._run_statistical_analysis()
    
    def _build_summary_df(self) -> None:
        rows = []
        global_max_dict = self._compute_global_maxima()
        for candidate, df_list in self.experiments_data.items():
            for i, df in enumerate(df_list):
                try:
                    processed_row = self._process_single_experiment(df, candidate, i, global_max_dict.get(i))
                except Exception as e:
                    print(f"Error processing candidate {candidate}, run {i}: {e}")
                    continue
                if processed_row:
                    rows.append(processed_row)
        self.summary_df = pd.DataFrame(rows)
    
    def _compute_global_maxima(self) -> dict:
        global_max_dict = {}
        for candidate, df_list in self.experiments_data.items():
            for i, df in enumerate(df_list):
                if "macro_f1" not in df.columns:
                    continue
                df_valid = df[~df["macro_f1"].isin([np.nan, np.inf, -np.inf])]
                if df_valid.empty:
                    continue
                local_max = df_valid["macro_f1"].max()
                if i not in global_max_dict or local_max > global_max_dict[i]:
                    global_max_dict[i] = local_max
        return global_max_dict

    def _process_single_experiment(self, df: pd.DataFrame, candidate: str, run_idx: int, global_max: float) -> dict:
        """Process a single experiment DataFrame"""
        if df.empty or "macro_f1" not in df.columns:
            raise Exception("Invalid DataFrame")
            
        valid_df = df[~df["macro_f1"].isnull() & 
                     ~df["macro_f1"].isin([np.inf, -np.inf])].copy()
        
        if valid_df.empty:
            raise Exception("Invalid DataFrame")
            
        valid_df["timestamp"] = pd.to_datetime(valid_df["timestamp"])
        
        return {
            "candidate": candidate,
            "run_idx": run_idx,
            "max_f1": valid_df["macro_f1"].max(),
            "mean_f1": valid_df["macro_f1"].mean(),
            "std_f1": valid_df["macro_f1"].std(),
            "auc": compute_auc_over_time(valid_df),
            "convergence_rate": compute_convergence_rate(valid_df),
            "time_to_50_hours": calculate_time_to_threshold(valid_df, global_max, 50),
            "time_to_75_hours": calculate_time_to_threshold(valid_df, global_max, 75),
            "time_to_90_hours": calculate_time_to_threshold(valid_df, global_max, 90),
            "num_evaluations": len(df),
            "num_errors": len(df) - len(valid_df),
            "error_ratio": (len(df) - len(valid_df)) / len(df)
        }
    
    def _compute_group_averages(self) -> None:
        if self.summary_df is not None and not self.summary_df.empty:
            self.group_avg = self.summary_df.groupby("candidate", as_index=False).agg({
                "max_f1": ["mean", "std"],
                "mean_f1": ["mean", "std"],
                "auc": ["mean", "std"],
                "convergence_rate": ["mean", "std"],
                "time_to_50_hours": ["mean", "std"],
                "time_to_75_hours": ["mean", "std"],
                "time_to_90_hours": ["mean", "std"],
                "num_evaluations": ["mean"],
                "num_errors": ["mean"],
                "error_ratio": ["mean"],
            }).fillna(0)
    
    def _run_statistical_analysis(self) -> None:
        """Executes all statistical tests with proper error control and validation"""
        metrics_to_analyze = {
            'performance': ['max_f1', 'mean_f1', 'std_f1'],
            'efficiency': ['auc', 'convergence_rate'],
            'time_metrics': ['time_to_50_hours', 'time_to_75_hours', 'time_to_90_hours'],
            'reliability': ['num_evaluations', 'num_errors', 'error_ratio']
        }

        self.statistical_results = {}
        all_pvalues = []

        for metric_category, metrics in metrics_to_analyze.items():
            self.statistical_results[metric_category] = {}
            for metric in metrics:
                # Validate independence assumption
                self._validate_independence(metric)
                
                # Compute power analysis
                power_results = self._compute_power_analysis(metric)
                
                # Run tests and store results
                test_results = {
                    "normality": self._test_normality(column=metric),
                    "overall_tests": self._run_overall_tests(metric),
                    "pairwise_tests": self._run_pairwise_tests(metric),
                    "power_analysis": power_results
                }
                
                # Collect p-values for family-wise error control
                if 'pairwise_tests' in test_results:
                    df = test_results['pairwise_tests']
                    all_pvalues.extend(df['p_value'].values)
                
                self.statistical_results[metric_category][metric] = test_results

        # Apply family-wise error correction across all tests
        adjusted_pvalues = self._adjust_family_wise_error()
        
        # Update all p-values with adjusted values
        for metric_category in self.statistical_results:
            for metric in self.statistical_results[metric_category]:
                if 'pairwise_tests' in self.statistical_results[metric_category][metric]:
                    df['significant'] = df['p_value_adjusted'] < self.alpha
    
    def _run_repeated_measures_anova(self, metric: str) -> Dict[str, Any]:
        """Run repeated measures ANOVA test"""
        from statsmodels.stats.anova import AnovaRM
        try:
            rm_anova = AnovaRM(
                data=self.summary_df,
                depvar=metric,
                subject='run_idx',
                within=['candidate']
            ).fit()
            
            return {
                'f_value': float(rm_anova.anova_table.iloc[0]['F Value']),
                'p_value': float(rm_anova.anova_table.iloc[0]['Pr > F']),
                'df': (
                    float(rm_anova.anova_table.iloc[0]['Num DF']),
                    float(rm_anova.anova_table.iloc[0]['Den DF'])
                ),
                'significant': float(rm_anova.anova_table.iloc[0]['Pr > F']) < self.alpha
            }
        except Exception as e:
            return {
                'error': f"Failed to compute repeated measures ANOVA: {str(e)}"
            }
            
    def _test_normality(self, column: str = 'max_f1') -> Dict[str, Any]:
        """Test normality of data by group"""
        results = {}
        for group in self.summary_df['candidate'].unique():
            group_data = self.summary_df[self.summary_df['candidate'] == group][column].dropna()
            
            # Proceed with normality tests for adequate sample sizes
            shapiro_stat, shapiro_p = stats.shapiro(group_data)
            
            results[group] = {
                'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'is_normal': shapiro_p > self.alpha
            }
        return results

    def _run_overall_tests(self, metric: str) -> Dict[str, Any]:
        """Run both parametric and non-parametric overall tests"""
        is_normal = all(result.get('is_normal', False)
                    for result in self._test_normality(metric).values())
        results = {}
        
        if is_normal:
            results['anova'] = self._run_repeated_measures_anova(metric)
        results['friedman'] = self._run_friedman_test(metric)
        return results
    
    def _run_friedman_test(self, metric: str) -> Dict[str, Any]:
        """Run Friedman test for specified metric"""
        groups = [group for _, group in 
                self.summary_df.groupby('candidate')[metric]]
        statistic, pvalue = stats.friedmanchisquare(*groups)
        return {
            'statistic': statistic,
            'p_value': pvalue,
            'significant': pvalue < self.alpha
        }
        
    def _run_pairwise_tests(self, metric: str) -> pd.DataFrame:
        """Run pairwise comparisons for specified metric"""
        results = []
        groups = self.summary_df['candidate'].unique()
        n_comparisons = len(groups) * (len(groups) - 1) // 2
        adjusted_alpha = self.alpha / n_comparisons
        
        for g1, g2 in itertools.combinations(groups, 2):
            data1 = self.summary_df[self.summary_df['candidate'] == g1][metric]
            data2 = self.summary_df[self.summary_df['candidate'] == g2][metric]
            
            if len(data1) < 2 or len(data2) < 2:
                continue
                
            stat, p = stats.wilcoxon(data1, data2)
            effect_sizes = self._calculate_effect_sizes(data1, data2)
            results.append({
                'metric': metric,
                'group1': g1,
                'group2': g2,
                'statistic': stat,
                'p_value': p,
                'significant': p < adjusted_alpha,
                **effect_sizes
            })
            
        # Create DataFrame and apply correction
        df_results = pd.DataFrame(results)
        df_results = MultipleComparisonHandler.adjust_pvalues(
            df_results, p_col='p_value', method='fdr_bh'
        )
        return df_results
    
    def _calculate_effect_sizes(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Calculate multiple effect size measures"""
        d = EffectSizeCalculator.cohens_d(x, y)
        g = EffectSizeCalculator.hedges_g(x, y)
        delta = EffectSizeCalculator.cliffs_delta(x, y)[0]
        
        # Add bootstrap CI
        ci_d = self._bootstrap_ci(x, y, EffectSizeCalculator.cohens_d)
        return {
            'cohens_d': d,
            'hedges_g': g,
            'cliffs_delta': delta,
            'cohens_d_ci': ci_d
        }
    
    def save_results(self, output_path: str) -> None:
        """Saves all results to JSON with improved structure for group averages."""
        def convert_numpy(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        def convert_to_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # and with columns like ('max_f1', 'mean') and ('max_f1', 'std'), etc.
        # First, create an empty dictionary for the restructured group averages.
        group_averages = {}
        # Define the desired candidate order.
        candidate_order = ["baseline", "low", "moderate", "high"]
        
        # Ensure the 'group_name' column exists and then set it as index to iterate rows easily.
        df = self.group_avg.copy()
        # When grouping, the candidate names should be in a column (not an index)
        # So we assume self.group_avg['group_name'] contains the candidate names.
        for candidate in candidate_order:
            candidate_df = df[df['candidate'] == candidate]
            if candidate_df.empty:
                continue
            candidate_row = candidate_df.iloc[0]
            metrics = {}
            # Iterate over the columns; skip the candidate column itself.
            for col in df.columns:
                if col == 'candidate':
                    continue
                # If the metric columns are multi-index (e.g., ('max_f1', 'mean')), unpack them.
                if isinstance(col, tuple):
                    metric, stat = col
                    metrics.setdefault(metric, {})[stat] = candidate_row[col]
                else:
                    metrics[col] = candidate_row[col]
            group_averages[candidate] = metrics

        # Build the final results dictionary.
        results = {
            "dataset": self.dataset_name,
            "group_averages": group_averages,
            "statistical_analysis": self.statistical_results
        }
        
        results = convert_numpy(results)
        results = convert_to_serializable(results)

        # Write out the JSON file in an indented format.
        import json
        with open(output_path, 'w') as fout:
            json.dump(results, fout, indent=2)

        
    def generate_report(self) -> str:
        """Generate a comprehensive statistical report"""
        return self._format_statistical_report()
    
    def _format_statistical_report(self) -> str:
        """Format the statistical results into a readable report"""
        if self.statistical_results is None:
            return "No statistical analysis has been performed yet."
            
        report = []
        report.append(f"Statistical Analysis Report for {self.dataset_name}\n")
        
        for category, metrics in self.statistical_results.items():
            report.append(f"\n## {category.title()} Metrics")
            
            for metric, analyses in metrics.items():
                report.append(f"\n### {metric}")
                
                # Normality Tests
                report.append("\nNormality Tests:")
                for group, results in analyses['normality'].items():
                    if 'warning' in results:
                        report.append(f"Warning: {results['warning']}")
                        continue
                    shapiro = results['shapiro']
                    report.append(f"- Group {group}: Shapiro-Wilk W={shapiro['statistic']:.3f}, p={shapiro['p_value']:.3f}")
                
                # Overall Tests
                if 'overall_tests' in analyses:
                    report.append("\nOverall Tests:")
                    if 'friedman' in analyses['overall_tests']:
                        friedman = analyses['overall_tests']['friedman']
                        report.append(f"- Friedman: χ²={friedman['statistic']:.3f}, p={friedman['p_value']:.3f}")
                    if 'anova' in analyses['overall_tests']:
                        for test_type, results in analyses['overall_tests']['anova'].items():
                            if 'error' in results:
                                report.append(f"- ANOVA ({test_type}): {results['error']}")
                            else:
                                report.append(f"- ANOVA ({test_type}): F={results['f_value']:.3f}, p={results['p_value']:.3f}")
                
                # Pairwise Tests
                if 'pairwise_tests' in analyses and not analyses['pairwise_tests'].empty:
                    report.append("\nPairwise Comparisons:")
                    for _, row in analyses['pairwise_tests'].iterrows():
                        report.append(
                            f"- {row['group1']} vs {row['group2']}:"
                            f"\n  Wilcoxon: W={row['statistic']:.3f}, p={row['p_value']:.3f}"
                            f"\n  Effect sizes: d={row['cohens_d']:.3f}, g={row['hedges_g']:.3f}, δ={row['cliffs_delta']:.3f}"
                        )
        
        return "\n".join(report)

    def _compute_power_analysis(self, metric: str) -> dict:
        """Compute power analysis for the given metric"""
        from statsmodels.stats.power import TTestPower
        effect_size = self.summary_df[metric].std() / 2  # Minimum detectable effect
        analysis = TTestPower()
        power = analysis.solve_power(
            effect_size=effect_size,
            nobs=len(self.summary_df) // 4,  # per group
            alpha=0.05,
            power=None
        )
        return {'power': power, 'effect_size': effect_size}
    
    def _adjust_family_wise_error(self) -> np.ndarray:
        """Apply family-wise error correction across all tests"""
        all_pvalues = []
        for category in self.statistical_results:
            for metric in self.statistical_results[category]:
                if 'pairwise_tests' in self.statistical_results[category][metric]:
                    df = self.statistical_results[category][metric]['pairwise_tests']
                    all_pvalues.extend(df['p_value'].values)
        
        # Only take the corrected p-values (second return value)
        _, adjusted_pvals, _, _ = multitest.multipletests(all_pvalues, method='holm')
        return adjusted_pvals
    
    def _validate_independence(self, metric: str) -> None:
        """Check for independence between samples"""
        correlations = []
        for g1, g2 in itertools.combinations(self.summary_df['candidate'].unique(), 2):
            data1 = self.summary_df[self.summary_df['candidate'] == g1][metric]
            data2 = self.summary_df[self.summary_df['candidate'] == g2][metric]
            corr = stats.pearsonr(data1, data2)[0]
            correlations.append(corr)
        
        if np.mean(np.abs(correlations)) > 0.3:
            warnings.warn("High correlation between groups detected. Consider using dependent samples tests.")

def main():
    """Main function to run the analysis pipeline"""
    # Initialize the DataLoader with the single-objective candidates configuration.
    loader = DataLoader("/home/coder/autogoal/experiments/configs/single-objective/candidates.yaml",
                      "/home/coder/autogoal/experiments/data/experience_store")
    
    datasets = ['liar', 'sst2']
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        # Directly load a dictionary mapping candidate names to their list of DataFrames (one per seed)
        experiments_data = loader.load_all_data_for_single_objective_dataset(dataset)
        # Create ExperimentHandler using the loaded experiments_data
        handler = ExperimentHandler(dataset, experiments_data)
        handler.process_experiments()
        results_path = f"/home/coder/autogoal/experiments/output/single-objective-analysis/{dataset}_experiment_summary.json"
        handler.save_results(results_path)

if __name__ == "__main__":
    main()