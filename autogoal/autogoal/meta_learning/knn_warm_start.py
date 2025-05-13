\
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from autogoal.meta_learning._experience import Experience
# Ensure MetricSpec is imported if it's used for type hinting here, or rely on parent class's handling
# from autogoal.meta_learning.utils import MetricSpec 
from autogoal.meta_learning.warm_start import WarmStart

logger = logging.getLogger(__name__)


class KNNWarmStart(WarmStart):
    """
    A KNN-like warm-starting strategy.

    Selects experiences based on a combined distance metric (features + deviation from best metrics)
    for positive examples, and feature-only distance for negative examples.
    The 'best' or ideal value for each metric is now expected to be part of the MetricSpec definition.
    """

    def __init__(
        self,
        # best_metrics_values: Dict[str, float], # Removed: now part of MetricSpec
        feature_ensemble_weight: float = 0.7,
        metrics_ensemble_weight: float = 0.3,
        *args,
        **kwargs,
    ):
        """
        Initializes KNNWarmStart.

        Args:
            feature_ensemble_weight (float): Weight for feature distance in combined score.
            metrics_ensemble_weight (float): Weight for metrics distance in combined score.
            *args, **kwargs: Arguments for WarmStart. self.metrics (List[MetricSpec]) is populated by parent.
        """
        super().__init__(*args, **kwargs)
        # self.best_metrics_values = best_metrics_values # Removed
        self.feature_ensemble_weight = feature_ensemble_weight
        self.metrics_ensemble_weight = metrics_ensemble_weight
        logger.info(
            f"KNNWarmStart initialized with "
            f"feature_ensemble_weight: {self.feature_ensemble_weight}, "
            f"metrics_ensemble_weight: {self.metrics_ensemble_weight}. "
            f"Best metric values will be sourced from MetricSpec definitions."
        )

    def _normalize_scalar_list(self, values: List[float]) -> List[float]:
        """Normalizes a list of scalar values to the [0, 1] range."""
        if not values:
            return []
        # Filter out None values before min/max, if any (should not happen with current logic upstream)
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return [0.0] * len(values) # Or handle as error

        min_val = min(valid_values)
        max_val = max(valid_values)
        range_val = max_val - min_val

        if abs(range_val) < 1e-9:  # Effectively zero range
            # All valid values are the same, map them to 0.0 or 0.5.
            # For distances, 0.0 is common if they are all minimal.
            return [0.0 if v is not None else 1.0 for v in values] # Penalize None if it slips through

        return [(v - min_val) / range_val if v is not None else 1.0 for v in values]


    def compute_distances(
        self,
        current_task_features: Dict[str, Optional[np.ndarray]],
        current_system_features: Optional[np.ndarray],
        experiences: List[Experience],
    ) -> Tuple[List[float], List[float]]: # Returns (combined_distances, feature_distances_normalized)
        """
        Computes two lists of distances for experiences:
        1. Combined distances: For non-error experiences, this is a weighted sum of
           normalized feature-based distance and normalized metrics-based distance.
           For error experiences (exp.metrics is None), this is their normalized feature-based distance.
        2. Normalized feature distances: Purely feature-based distances, normalized.
        """
        if not experiences:
            return [], []

        # 1. Get raw feature-based distances from parent and normalize them
        raw_feature_distances = super().compute_distances(
            current_task_features, current_system_features, experiences
        )
        norm_feature_distances_list = self._normalize_scalar_list(raw_feature_distances)

        # 2. Calculate overall metrics-based distance component for all experiences
        # (Error experiences will get a high penalty here due to missing metrics)
        metric_diffs_collect: Dict[str, List[Optional[float]]] = {
            metric_spec.name: [] for metric_spec in self.metrics
        }

        for exp in experiences:
            # exp.metrics being None implies an error or unsuitability for metric evaluation
            exp_metric_map = {m.name: m.value for m in (exp.metrics or [])}
            for metric_spec in self.metrics: # self.metrics is List[MetricSpec]
                exp_val = exp_metric_map.get(metric_spec.name)
                best_val = metric_spec.best_value # Changed: Get best_value from MetricSpec
                diff: Optional[float] = None
                if exp_val is not None and best_val is not None:
                    # Ensure metric direction is considered if 'maximize' flag is available
                    # For simplicity, using abs difference, assuming higher is better for best_val
                    # and we want to be close to it.
                    # If a metric is "lower is better", best_val should be low.
                    diff = abs(exp_val - best_val)
                metric_diffs_collect[metric_spec.name].append(diff)
        
        norm_metric_diffs_per_metric: Dict[str, List[float]] = {}
        for name, diffs_list in metric_diffs_collect.items():
            valid_diffs = [d for d in diffs_list if d is not None]
            max_abs_diff = 0.0
            if valid_diffs:
                max_abs_diff = max(valid_diffs) if valid_diffs else 0.0
            
            normalized_diffs_for_current_metric = []
            for d_val in diffs_list:
                if d_val is not None:
                    normalized_diffs_for_current_metric.append(
                        d_val / max_abs_diff if max_abs_diff > 1e-9 else 0.0
                    )
                else:
                    # This experience is missing this metric or it couldn't be compared
                    normalized_diffs_for_current_metric.append(1.0)  # Max penalty
            norm_metric_diffs_per_metric[name] = normalized_diffs_for_current_metric

        exp_total_metrics_component_list = []
        for i, _ in enumerate(experiences):
            current_exp_metric_distance_sum = 0.0
            for metric_spec in self.metrics:
                if metric_spec.name in norm_metric_diffs_per_metric and i < len(norm_metric_diffs_per_metric[metric_spec.name]):
                    norm_diff = norm_metric_diffs_per_metric[metric_spec.name][i]
                    current_exp_metric_distance_sum += norm_diff * metric_spec.weight
                else: # Should not happen if logic is correct
                    current_exp_metric_distance_sum += metric_spec.weight # Max penalty
            exp_total_metrics_component_list.append(current_exp_metric_distance_sum)
        
        norm_overall_metrics_dist_list = self._normalize_scalar_list(exp_total_metrics_component_list)

        # 3. Construct the final combined distances
        final_combined_distances_list = []
        for i, exp in enumerate(experiences):
            is_error_experience = exp.metrics is None # Define error experience

            norm_fd = norm_feature_distances_list[i]
            
            if is_error_experience:
                final_combined_distances_list.append(norm_fd)
            else:
                norm_md = norm_overall_metrics_dist_list[i]
                combined_dist = (
                    self.feature_ensemble_weight * norm_fd +
                    self.metrics_ensemble_weight * norm_md
                )
                final_combined_distances_list.append(combined_dist)
        
        logger.info(f"Computed combined and feature distances for {len(experiences)} experiences.")
        return final_combined_distances_list, norm_feature_distances_list

    def select_experiences(
        self, 
        experiences: List[Experience], 
        # These are the combined distances (feature+metric for non-errors, feature-only for errors)
        distances_for_positive_ranking: List[float], 
        # These are pure normalized feature distances for all experiences
        pure_feature_distances: List[float]
    ) -> Tuple[List[Experience], List[float], List[Experience], List[float]]:
        """
        Selects positive and negative experiences.
        - Positive: k_pos non-error experiences with smallest combined distances.
        - Negative: k_neg experiences (errors or non-selected non-errors) with largest feature distances.
        Returns (selected_pos_exp, pos_distances_for_alpha, selected_neg_exp, neg_distances_for_alpha)
        """
        if not experiences:
            logger.warning("Cannot select experiences: no experiences provided.")
            return [], [], [], []
        
        num_experiences = len(experiences)
        if num_experiences != len(distances_for_positive_ranking) or num_experiences != len(pure_feature_distances):
            logger.error("Mismatch in lengths of experiences and distance lists. Cannot select.")
            # Potentially return an error or empty lists
            return [], [], [], []

        # Positive Experience Selection
        positive_candidates_data = []
        for i, exp in enumerate(experiences):
            if exp.metrics is not None: # Non-error experiences
                positive_candidates_data.append((exp, distances_for_positive_ranking[i], i)) # Store original index if needed

        # Sort by combined distance (smaller is better)
        positive_candidates_data.sort(key=lambda x: x[1])
        
        actual_k_pos = min(self.k_pos, len(positive_candidates_data))
        selected_positive_tuples = positive_candidates_data[:actual_k_pos]
        
        selected_positive_experiences = [exp_tuple[0] for exp_tuple in selected_positive_tuples]
        # These are the combined distances for the selected positive experiences
        positive_distances_for_alpha = [exp_tuple[1] for exp_tuple in selected_positive_tuples]
        
        selected_positive_exp_set = set(selected_positive_experiences)

        # Negative Experience Selection
        negative_candidates_data = []
        for i, exp in enumerate(experiences):
            is_error = exp.metrics is None
            if is_error or exp not in selected_positive_exp_set:
                # Use pure feature distance for ranking negative candidates
                negative_candidates_data.append((exp, pure_feature_distances[i]))
        
        # Sort by pure feature distance (larger is "more different", so sort ascending and pick from end)
        negative_candidates_data.sort(key=lambda x: x[1]) 
        
        actual_k_neg = min(self.k_neg, len(negative_candidates_data))
        # Select k_neg experiences with the largest feature distances
        selected_negative_tuples = negative_candidates_data[len(negative_candidates_data)-actual_k_neg:]
        
        selected_negative_experiences = [exp_tuple[0] for exp_tuple in selected_negative_tuples]
        # These are the feature distances for the selected negative experiences
        negative_distances_for_alpha = [exp_tuple[1] for exp_tuple in selected_negative_tuples]

        logger.info(
            f"Selected {len(selected_positive_experiences)} positive and "
            f"{len(selected_negative_experiences)} negative experiences."
        )
        return selected_positive_experiences, positive_distances_for_alpha, selected_negative_experiences, negative_distances_for_alpha

    def compute_learning_rates(
        self,
        selected_positive_experiences: List[Experience],
        positive_distances: List[float], # Combined distances for these
        selected_negative_experiences: List[Experience],
        negative_distances: List[float], # Feature distances for these
    ) -> Dict[Experience, float]:
        """
        Computes learning rates (alphas).
        Positive experiences: alpha based on their combined distance.
        Negative experiences: alpha based on their feature distance.
        """
        alphas: Dict[Experience, float] = {}

        # Positive experiences: smaller distance = higher alpha (closer to max_alpha)
        if selected_positive_experiences and positive_distances:
            # Normalize distances within the selected positive set to [0,1]
            # where 0 is best (smallest original distance in this set)
            norm_pos_dists = self._normalize_scalar_list(positive_distances)
            for i, exp in enumerate(selected_positive_experiences):
                # (1 - norm_dist) makes alpha larger for smaller original distances
                alphas[exp] = self.max_alpha * (1 - norm_pos_dists[i])

        # Negative experiences: smaller feature distance (among negatives) = less negative alpha
        # Larger feature distance (among negatives) = more negative alpha (closer to min_alpha)
        if selected_negative_experiences and negative_distances:
            # Normalize feature distances within the selected negative set to [0,1]
            # where 0 is best (smallest original feature distance in this set)
            norm_neg_dists = self._normalize_scalar_list(negative_distances)
            for i, exp in enumerate(selected_negative_experiences):
                # norm_neg_dists[i] = 0 means it's the "closest" of the negatives (smallest feature distance)
                # norm_neg_dists[i] = 1 means it's the "farthest" of the negatives (largest feature distance)
                # min_alpha is negative. So, largest feature distance gets alpha closest to min_alpha.
                alphas[exp] = self.min_alpha * norm_neg_dists[i] 
        
        logger.info(f"Computed learning rates for {len(alphas)} experiences.")
        return alphas
