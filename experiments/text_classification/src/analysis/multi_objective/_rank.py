#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Import the DataLoader and WarmstartConfigParser modules from the suggested project structure.
from text_classification.src.data_loading import DataLoader

@dataclass
class AlgorithmRunMetrics:
    algo_name: str
    metrics: np.ndarray

class AutoMLMetrics:
    def __init__(self):
        self.metrics = ['hypervolume', 'max_f1', 'min_eval_time']
        self.spacing = SpacingIndicator()

    def calculate_all(self, current_front: np.ndarray, reference_front: np.ndarray, df: pd.DataFrame = None) -> dict:
        metrics_dict = {}
        objectives_meta = {
            'f1': {'minimize': False},
            'evaluation_time': {'minimize': True}
        }
        scaler = MinMaxScaler()
        if len(current_front) > 0 and len(reference_front) > 0:
            combined = np.vstack([current_front, reference_front])
            scaler.fit(combined)
            current_front_norm = scaler.transform(current_front)
            reference_front_norm = scaler.transform(reference_front)

            metrics_dict['hypervolume'] = self.calculate_hypervolume(current_front_norm, objectives_meta)
            metrics_dict['generational_distance'] = self.calculate_gd(current_front_norm, reference_front_norm)
            metrics_dict['inverted_generational_distance'] = self.calculate_igd(current_front_norm, reference_front_norm)
            metrics_dict['spacing'] = self.calculate_spacing(current_front_norm)
            metrics_dict['coverage'] = self.calculate_coverage(current_front_norm, reference_front_norm)
        if df is not None:
            metrics_dict['max_f1'] = df['f1'].max()
            metrics_dict['mean_f1'] = df['f1'].mean()
            metrics_dict['min_eval_time'] = df['evaluation_time'].min()
            metrics_dict['mean_eval_time'] = df['evaluation_time'].mean()
        return metrics_dict

    def calculate_hypervolume(self, front: np.ndarray, objectives_meta: dict) -> float:
        if len(front) == 0:
            return 0.0
        ref_point = np.zeros(front.shape[1])
        for i, (name, meta) in enumerate(objectives_meta.items()):
            if meta['minimize']:
                ref_point[i] = np.max(front[:, i]) * 1.1
            else:
                ref_point[i] = np.min(front[:, i]) * 0.9
                front[:, i] = -front[:, i]
                ref_point[i] = -ref_point[i]
        hv = Hypervolume(ref_point=ref_point)
        return float(hv.do(front))

    def calculate_gd(self, current: np.ndarray, reference: np.ndarray) -> float:
        if len(current) == 0 or len(reference) == 0:
            return float('inf')
        distances = np.min(np.linalg.norm(current[:, None] - reference, axis=2), axis=1)
        return float(np.mean(distances))

    def calculate_igd(self, current: np.ndarray, reference: np.ndarray) -> float:
        if len(current) == 0 or len(reference) == 0:
            return float('inf')
        distances = np.min(np.linalg.norm(reference[:, None] - current, axis=2), axis=1)
        return float(np.mean(distances))

    def calculate_spacing(self, front: np.ndarray) -> float:
        if len(front) < 2:
            return 0.0
        return float(self.spacing.do(front))

    def calculate_coverage(self, current: np.ndarray, reference: np.ndarray) -> float:
        if len(current) == 0 or len(reference) == 0:
            return 0.0
        dominated = 0
        for ref_point in reference:
            if np.any(np.all(current <= ref_point, axis=1)):
                dominated += 1
        return float(dominated / len(reference))


class AutoMLComparator:
    def __init__(self, data_dict: dict, metrics_calculator: AutoMLMetrics, ranking_method: str = 'olympic'):
        self.data = data_dict
        self.metrics_calculator = metrics_calculator
        self.ranking_method = ranking_method.lower()
        self.all_metric_vectors = []
        self.ranking_results = {}

    def extract_objectives(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([])
        df = df.dropna(subset=['f1', 'evaluation_time'])
        if df.empty:
            return np.array([])
        front = df[['f1', 'evaluation_time']].to_numpy(copy=True)
        front[:, 0] = -1.0 * front[:, 0]
        return front

    def get_local_pareto_front(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([])
        nds = NonDominatedSorting()
        fronts = nds.do(points)
        return points[fronts[0]]

    def build_global_reference_set(self) -> np.ndarray:
        all_points = []
        for algo_name, df in self.data.items():
            front_2d = self.extract_objectives(df)
            if len(front_2d) > 0:
                all_points.append(front_2d)
        if not all_points:
            return np.array([])
        return np.vstack(all_points)

    def compute_performance_vector_for_algorithm(self, algo_name: str, local_front: np.ndarray, global_ref: np.ndarray, df: pd.DataFrame = None) -> np.ndarray:
        if len(local_front) == 0 or len(global_ref) == 0:
            return np.array([])
        result_dict = self.metrics_calculator.calculate_all(local_front, global_ref, df)
        print(f"Algorithm: {algo_name}, HV: {result_dict['hypervolume']}")
        metric_order = self.metrics_calculator.metrics
        vect = [result_dict.get(m, 0) for m in metric_order]
        return np.array([vect])

    def gather_all_metrics(self):
        global_ref_2d = self.build_global_reference_set()
        for algo_name, df in self.data.items():
            points_2d = self.extract_objectives(df)
            local_front_2d = self.get_local_pareto_front(points_2d)
            if len(local_front_2d) > 0:
                perf_vec = self.compute_performance_vector_for_algorithm(algo_name, local_front_2d, global_ref_2d, df)
                if perf_vec.size > 0:
                    self.all_metric_vectors.append(AlgorithmRunMetrics(algo_name, perf_vec[0]))

    def do_multi_metric_nds(self):
        if not self.all_metric_vectors:
            return
        big_matrix = np.array([run.metrics for run in self.all_metric_vectors])
        algo_names = [run.algo_name for run in self.all_metric_vectors]
        for i in range(len(self.metrics_calculator.metrics)):
            metric = self.metrics_calculator.metrics[i]
            if metric in ['hypervolume', 'spacing', 'coverage', 'max_f1', 'mean_f1']:
                big_matrix[:, i] = -1.0 * big_matrix[:, i]
        nds = NonDominatedSorting()
        fronts = nds.do(big_matrix, only_non_dominated_front=False)
        self.front_counts = {}
        L = len(fronts)
        for algo in set(algo_names):
            self.front_counts[algo] = [0] * L
        for level_idx, front_indices in enumerate(fronts):
            for idx in front_indices:
                algo = algo_names[idx]
                self.front_counts[algo][level_idx] += 1

    def rank_algorithms(self):
        self.ranking_results = {
            'olympic': {},
            'linear': {},
            'exponential': {},
            'adaptive': {}
        }
        L = 0
        if self.front_counts:
            L = len(next(iter(self.front_counts.values())))

        def olympic_score(algo):
            return tuple([-n for n in self.front_counts[algo]])

        def linear_score(algo):
            score = 0
            for i, c in enumerate(self.front_counts[algo]):
                score += c * (L - i)
            return -score

        def exponential_score(algo):
            score = 0.0
            for i, c in enumerate(self.front_counts[algo]):
                score += c * (2.0 ** (-i))
            return -score

        def adaptive_score(algo, total_cw):
            cum = 0
            score = 0.0
            for l in range(L):
                cum += self.front_counts[algo][l]
                if total_cw[l] > 0:
                    score += (cum / total_cw[l])
            return -score

        total_cw = [0] * L
        front_sums = {}
        for algo in self.front_counts:
            c = 0
            front_sums[algo] = []
            for l in range(L):
                c += self.front_counts[algo][l]
                front_sums[algo].append(c)
        for l in range(L):
            total_cw[l] = sum(front_sums[algo][l] for algo in self.front_counts)
        algo_list = list(self.front_counts.keys())

        sorted_olympic = sorted(algo_list, key=lambda A: olympic_score(A))
        sorted_linear = sorted(algo_list, key=lambda A: linear_score(A))
        sorted_expo = sorted(algo_list, key=lambda A: exponential_score(A))
        adaptive_map = {algo: adaptive_score(algo, total_cw) for algo in algo_list}
        sorted_adaptive = sorted(algo_list, key=lambda A: adaptive_map[A])

        # Helper function to produce dense ranks via tie handling.
        # Candidates with the same score receive the same rank.
        def to_rank_dict(sorted_algos, score_func):
            rank_dict = {}
            current_rank = 0
            last_score = None
            for algo in sorted_algos:
                score = score_func(algo)
                if last_score is None or score != last_score:
                    current_rank += 1
                rank_dict[algo] = current_rank
                last_score = score
            return rank_dict

        self.ranking_results['olympic'] = to_rank_dict(sorted_olympic, olympic_score)
        self.ranking_results['linear'] = to_rank_dict(sorted_linear, linear_score)
        self.ranking_results['exponential'] = to_rank_dict(sorted_expo, exponential_score)
        self.ranking_results['adaptive'] = to_rank_dict(sorted_adaptive, lambda algo: adaptive_map[algo])
        
    def run_full_pipeline(self):
        self.gather_all_metrics()
        self.do_multi_metric_nds()
        self.rank_algorithms()
        if self.ranking_method == 'all':
            return self.ranking_results
        return self.ranking_results.get(self.ranking_method, {})

def clean_data(data: dict):
    for config_name, df in data.items():
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['f1', 'evaluation_time'])
        data[config_name] = df

def main():
    # Initialize the DataLoader with the multi-objective candidates configuration.
    loader = DataLoader('autogoal/experiments/text_classification/configs/multi-objective/candidates.yaml', '/home/coder/autogoal/experiments/text_classification/data/experience_store')
    
    # Run the analysis for all datasets.
    for dataset in ['liar', 'sst2', 'meld', 'ag_news']:
        dataset_dict = loader.load_all_data_for_dataset(dataset)
        print('-' * 50)
        print(f"Dataset: {dataset}")
        
        combined_data_dict = {}
        for bias_level, candidate_dict in dataset_dict.items():
            for candidate, candidate_data in candidate_dict.items():
                combined_key = f"{bias_level} - {candidate}"
                combined_data_dict[combined_key] = candidate_data
                
        
        clean_data(combined_data_dict)
        metrics_calculator = AutoMLMetrics()
        comparator = AutoMLComparator(
            data_dict=combined_data_dict,
            metrics_calculator=metrics_calculator,
            ranking_method='all'
        )
        results = comparator.run_full_pipeline()

        for method_name, rank_map in results.items():
            print(f"\n{method_name.title()} Ranking:")
            for algo, rank_val in sorted(rank_map.items(), key=lambda x: x[1]):
                print(f"{rank_val}. {algo}")
        print('-' * 50)
        print()

if __name__ == "__main__":
    main()
