import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class AutoMLMetrics:
    def __init__(self):
        self.metrics = ['hypervolume', 'generational_distance', 'inverted_generational_distance', 'spacing', 'coverage']
        self.spacing = SpacingIndicator()

    def calculate_all(self, current_front: np.ndarray, reference_front: np.ndarray, df: pd.DataFrame = None) -> dict:
        metrics_dict = {}
        objectives_meta = {
            'macro_f1': {'minimize': False},
            'evaluation_time': {'minimize': True}
        }
        scaler = MinMaxScaler()
        if len(current_front) > 0 and len(reference_front) > 0:
            combined = np.vstack([current_front, reference_front])
            scaler.fit(combined)
            current_front_norm = scaler.transform(current_front)
            reference_front_norm = scaler.transform(reference_front)
        else:
            current_front_norm, reference_front_norm = current_front, reference_front

        metrics_dict['hypervolume'] = self.calculate_hypervolume(current_front_norm, objectives_meta)
        metrics_dict['generational_distance'] = self.calculate_gd(current_front_norm, reference_front_norm)
        metrics_dict['inverted_generational_distance'] = self.calculate_igd(current_front_norm, reference_front_norm)
        metrics_dict['spacing'] = self.calculate_spacing(current_front_norm)
        metrics_dict['coverage'] = self.calculate_coverage(current_front_norm, reference_front_norm)

        if df is not None:
            metrics_dict['max_f1'] = df['macro_f1'].max()
            metrics_dict['mean_f1'] = df['macro_f1'].mean()
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
        self.front_counts = {}
        self.ranking_results = {}

    def extract_objectives(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([])
        df = df.dropna(subset=['macro_f1', 'evaluation_time'])
        if df.empty:
            return np.array([])
        front = df[['macro_f1', 'evaluation_time']].to_numpy(copy=True)
        front[:, 0] = -front[:, 0]  # Convert maximization to minimization for F1.
        return front

    def get_local_pareto_front(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([])
        nds = NonDominatedSorting()
        fronts = nds.do(points)
        return points[fronts[0]]

    def build_global_reference_set(self) -> np.ndarray:
        all_points = []
        for df in self.data.values():
            pts = self.extract_objectives(df)
            if len(pts) > 0:
                all_points.append(pts)
        if not all_points:
            return np.array([])
        return np.vstack(all_points)

    def compute_performance_vector_for_algorithm(self, algo_name: str, local_front: np.ndarray,
                                                 global_ref: np.ndarray, df: pd.DataFrame = None) -> np.ndarray:
        if len(local_front) == 0 or len(global_ref) == 0:
            return np.array([])
        result_dict = self.metrics_calculator.calculate_all(local_front, global_ref, df)
        # Use a consistent metric order
        metric_order = self.metrics_calculator.metrics
        vect = [result_dict[m] for m in metric_order]
        return np.array([vect])

    def gather_all_metrics(self):
        global_ref_2d = self.build_global_reference_set()
        for algo_name, df in self.data.items():
            pts = self.extract_objectives(df)
            local_front = self.get_local_pareto_front(pts)
            if len(local_front) > 0:
                perf_vec = self.compute_performance_vector_for_algorithm(algo_name, local_front, global_ref_2d, df)
                if perf_vec.size > 0:
                    self.all_metric_vectors.append((algo_name, perf_vec[0]))

    def do_multi_metric_nds(self):
        if not self.all_metric_vectors:
            return
        algo_names, all_vectors = zip(*self.all_metric_vectors)
        big_matrix = np.array(all_vectors)
        # Convert maximization metrics to minimization if needed (e.g., flip sign)
        for i in range(big_matrix.shape[1]):
            big_matrix[:, i] = -big_matrix[:, i]
        nds = NonDominatedSorting()
        fronts = nds.do(big_matrix, only_non_dominated_front=False)
        L = len(fronts)
        for algo in algo_names:
            self.front_counts[algo] = [0] * L
        for level_idx, front_indices in enumerate(fronts):
            for idx in front_indices:
                algo = algo_names[idx]
                self.front_counts[algo][level_idx] += 1

    def rank_algorithms(self):
        self.ranking_results = {'olympic': {}, 'linear': {}, 'exponential': {}, 'adaptive': {}}
        if not self.front_counts:
            return
        L = len(next(iter(self.front_counts.values())))
        def olympic_score(algo):
            return tuple([-n for n in self.front_counts[algo]])
        def linear_score(algo):
            counts = self.front_counts[algo]
            return -sum(c * (L - i) for i, c in enumerate(counts))
        def exponential_score(algo):
            counts = self.front_counts[algo]
            return -sum(c * (2 ** (-i)) for i, c in enumerate(counts))
        algo_list = list(self.front_counts.keys())
        sorted_olympic = sorted(algo_list, key=olympic_score)
        sorted_linear = sorted(algo_list, key=linear_score)
        sorted_exponential = sorted(algo_list, key=exponential_score)
        def to_rank_dict(sorted_algos):
            return {algo: rank for rank, algo in enumerate(sorted_algos, start=1)}
        self.ranking_results['olympic'] = to_rank_dict(sorted_olympic)
        self.ranking_results['linear'] = to_rank_dict(sorted_linear)
        self.ranking_results['exponential'] = to_rank_dict(sorted_exponential)

    def run_full_pipeline(self):
        self.gather_all_metrics()
        self.do_multi_metric_nds()
        self.rank_algorithms()
        if self.ranking_method == 'all':
            return self.ranking_results
        return self.ranking_results.get(self.ranking_method, {})

# Utility function to clean DataFrames if necessary.
def assign_ranks(data_dict: dict):
    for key, df in data_dict.items():
        df.replace([np.inf, -np.inf], pd.NA, inplace=True)
        df.dropna(subset=['macro_f1', 'evaluation_time'], inplace=True)
