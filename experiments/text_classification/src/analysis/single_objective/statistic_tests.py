import itertools
from typing import Any, Dict, List, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from text_classification.src.data_loading.data_loader import DataLoader
import statsmodels.stats.multitest as multitest
from scipy.integrate import trapezoid

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


class ExperimentGroupPlot:
    """A group of experiments with visualization properties"""
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
    """Main handler for experiment analysis"""
    def __init__(self, dataset_name: str, experiments_data: List[ExperimentGroupPlot], alpha: float = 0.05):
        self.dataset_name = dataset_name
        self.experiments_data = experiments_data
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
        """Builds the summary DataFrame from experiment data"""
        rows = []
        global_max_dict = self._compute_global_maxima()
        
        for group in self.experiments_data:
            for i, df in enumerate(group.dataframes):
                processed_row = self._process_single_experiment(
                    df, group.alias_name, i, global_max_dict.get(i)
                )
                if processed_row:
                    rows.append(processed_row)
        
        self.summary_df = pd.DataFrame(rows)
    
    def _compute_global_maxima(self) -> Dict[int, float]:
        """Computes global maximum F1 scores for each experiment index"""
        global_max_dict = {}
        for group in self.experiments_data:
            for i, df in enumerate(group.dataframes):
                if "macro_f1" not in df.columns:
                    continue
                df_valid = df[~df["macro_f1"].isin([np.nan, np.inf, -np.inf])]
                if df_valid.empty:
                    continue
                    
                local_max = df_valid["macro_f1"].max()
                if i not in global_max_dict or local_max > global_max_dict[i]:
                    global_max_dict[i] = local_max
        return global_max_dict
    
    def _process_single_experiment(self, df: pd.DataFrame, 
                                 group_name: str, idx: int, 
                                 global_max: float) -> Dict[str, Any]:
        """Process a single experiment DataFrame"""
        if df.empty or "macro_f1" not in df.columns:
            raise Exception("Invalid DataFrame")
            
        valid_df = df[~df["macro_f1"].isnull() & 
                     ~df["macro_f1"].isin([np.inf, -np.inf])].copy()
        
        if valid_df.empty:
            raise Exception("Invalid DataFrame")
            
        valid_df["timestamp"] = pd.to_datetime(valid_df["timestamp"])
        
        return {
            "group_name": group_name,
            "df_index": idx,
            "max_f1": valid_df["macro_f1"].max(),
            "mean_f1": valid_df["macro_f1"].mean(),
            "std_f1": valid_df["macro_f1"].std(),
            "auc": compute_auc_over_time(valid_df),
            "convergence_rate": compute_convergence_rate(valid_df),
            "time_to_max_hours": calculate_time_to_threshold(valid_df, global_max, 100.0),
            "time_to_50_hours": calculate_time_to_threshold(valid_df, global_max, 50),
            "time_to_75_hours": calculate_time_to_threshold(valid_df, global_max, 75),
            "time_to_90_hours": calculate_time_to_threshold(valid_df, global_max, 90),
            "num_evaluations": len(valid_df),
            "num_errors": len(df) - len(valid_df)
        }
    
    def _compute_group_averages(self) -> None:
        """Computes average metrics for each experiment group"""
        self.group_avg = self.summary_df.groupby("group_name", as_index=False).agg({
            "max_f1": ["mean", "std"],
            "mean_f1": ["mean", "std"],
            "auc": ["mean", "std"],
            "convergence_rate": ["mean", "std"],
            "time_to_max_hours": ["mean", "std"],
            "time_to_50_hours": ["mean", "std"],
            "time_to_75_hours": ["mean", "std"],
            "time_to_90_hours": ["mean", "std"]
        }).fillna(0)
    
    def _run_statistical_analysis(self) -> None:
        """Executes all statistical tests with proper error control and validation"""
        metrics_to_analyze = {
            'performance': ['max_f1', 'mean_f1', 'std_f1'],
            'efficiency': ['auc', 'convergence_rate'],
            'time_metrics': ['time_to_max_hours', 'time_to_50_hours',
                            'time_to_75_hours', 'time_to_90_hours'],
            'reliability': ['num_evaluations', 'num_errors']
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
        p_value_idx = 0
        for metric_category in self.statistical_results:
            for metric in self.statistical_results[metric_category]:
                if 'pairwise_tests' in self.statistical_results[metric_category][metric]:
                    df = self.statistical_results[metric_category][metric]['pairwise_tests']
                    n_tests = len(df)
                    df['adjusted_p_value'] = adjusted_pvalues[p_value_idx:p_value_idx + n_tests]
                    df['significant'] = df['adjusted_p_value'] < self.alpha
                    p_value_idx += n_tests
    
    def _run_repeated_measures_anova(self, metric: str) -> Dict[str, Any]:
        """Run repeated measures ANOVA test"""
        from statsmodels.stats.anova import AnovaRM
        try:
            rm_anova = AnovaRM(
                data=self.summary_df,
                depvar=metric,
                subject='df_index',
                within=['group_name']
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
        for group in self.summary_df['group_name'].unique():
            group_data = self.summary_df[self.summary_df['group_name'] == group][column].dropna()
            
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
                self.summary_df.groupby('group_name')[metric]]
        statistic, pvalue = stats.friedmanchisquare(*groups)
        return {
            'statistic': statistic,
            'p_value': pvalue,
            'significant': pvalue < self.alpha
        }
        
    def _run_pairwise_tests(self, metric: str) -> pd.DataFrame:
        """Run pairwise comparisons for specified metric"""
        results = []
        groups = self.summary_df['group_name'].unique()
        n_comparisons = len(groups) * (len(groups) - 1) // 2
        adjusted_alpha = self.alpha / n_comparisons
        
        for g1, g2 in itertools.combinations(groups, 2):
            data1 = self.summary_df[self.summary_df['group_name'] == g1][metric]
            data2 = self.summary_df[self.summary_df['group_name'] == g2][metric]
            
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
        """Save all results to JSON with proper key conversion"""
        results = {
            "dataset": self.dataset_name,
            "group_averages": self.group_avg.to_dict(),
            "statistical_analysis": self.statistical_results
        }
        
        flattened_data = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened_data[f"{key}_{subkey}"] = subvalue
            else:
                flattened_data[key] = value
        
        df = pd.DataFrame([flattened_data])
        df.to_json(output_path, orient='records', indent=2)
        
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
        for g1, g2 in itertools.combinations(self.summary_df['group_name'].unique(), 2):
            data1 = self.summary_df[self.summary_df['group_name'] == g1][metric]
            data2 = self.summary_df[self.summary_df['group_name'] == g2][metric]
            corr = stats.pearsonr(data1, data2)[0]
            correlations.append(corr)
        
        if np.mean(np.abs(correlations)) > 0.3:
            warnings.warn("High correlation between groups detected. Consider using dependent samples tests.")


def main():
    """Main function to run the analysis pipeline"""
    # Initialize the DataLoader with the single-objective candidates configuration.
    loader = DataLoader("experiments/configs/single-objective/candidates.yaml",
                      "experiments/data/experience_store")
    
    for dataset in ['liar', 'sst2', 'meld', 'ag_news']:
        print(f"Processing dataset: {dataset}")
        dataset_dict = loader.load_all_data_for_dataset(dataset)
        
        # Prepare experiment groups for analysis
        experiments_data = []
        for bias_level, candidate_dict in dataset_dict.items():
            if bias_level == 'baseline':
                experiments_data.append(ExperimentGroupPlot(bias_level, "-", "blue", [candidate_dict]))
            else:
                for candidate, candidate_data in candidate_dict.items():
                    alias_name = f"{bias_level} - {candidate}"
                    experiments_data.append(ExperimentGroupPlot(alias_name, "-", "green", [candidate_data]))
        
        # Process experiments and save results
        handler = ExperimentHandler(dataset, experiments_data)
        handler.process_experiments()
        
        results_path = f"experiments/output/single-objective-analysis/{dataset}_experiment_summary.json"
        handler.save_results(results_path)
        print(handler.generate_report())


if __name__ == "__main__":
    main()