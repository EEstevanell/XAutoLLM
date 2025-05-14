import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from autogoal.meta_learning._experience import Experience
from autogoal.meta_learning.utils import MetricSpec
from autogoal.meta_learning.warm_start import WarmStart

logger = logging.getLogger(__name__)


class KNNWarmStart(WarmStart):
    """
    Pure kNN baseline warm-start:
     - compute pos: combined(feature,metrics) distances for exps with metrics
     - compute neg: feature-only distances for exps without metrics
     - uniform alphas: sum of positives = max_alpha; sum of negatives = min_alpha
    """

    def __init__(
        self,
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
        # 1) automatically set best_value per metric from past experiences
        for m in self.metrics:
            # collect all past values for this metric
            vals = [exp_val.value
                    for exp in self._experiences
                    for exp_val in (exp.metrics or [])
                    if exp_val.name == m.name]
            if vals:
                m.best_value = max(vals) if m.maximize else min(vals)
            else:
                # no data → fall back to “current default”
                m.best_value = 1 if m.maximize else 0

        self.feature_ensemble_weight = feature_ensemble_weight
        self.metrics_ensemble_weight = metrics_ensemble_weight
        logger.info(
            f"KNNWarmStart initialized with "
            f"feature_ensemble_weight: {self.feature_ensemble_weight}, "
            f"metrics_ensemble_weight: {self.metrics_ensemble_weight}. "
        )
        
    def compute_distances(
        self,
        current_task_features: Dict[str, Optional[np.ndarray]],
        current_system_features: Optional[np.ndarray],
        experiences: List[Experience],
    ) -> Tuple[Dict[Experience, float], Dict[Experience, float]]:
        """
        Returns:
          pos_distances: combined normalized distance (feat+metric)
                         for experiences WITH metrics
          neg_distances: normalized feature-only distance
                         for experiences WITHOUT metrics
        """
        # --- 1) flatten features into vectors ---
        def flatten(feat_dict, sys_feat):
            parts = [v.ravel() for v in feat_dict.values() if v is not None]
            if sys_feat is not None:
                parts.append(sys_feat.ravel())
            return np.concatenate(parts) if parts else np.zeros(0)

        curr_vec = flatten(current_task_features, current_system_features)[None, :]
        exp_matrix = np.vstack([flatten(e.task_features, e.system_features) for e in experiences])

        # --- 2) compute & normalize feature distances ---
        metric = "cosine"
        if curr_vec.shape[1] == 0:
            raw_fd = np.ones(len(experiences))
        else:
            raw_fd = pairwise_distances(curr_vec, exp_matrix, metric=metric).ravel()
        norm_fd = self._normalize_scalar_list(raw_fd.tolist())

        # --- 3) compute & normalize metric distances for positives ---
        unnorm_md: List[Optional[float]] = []
        for e in experiences:
            if e.metrics is None:
                unnorm_md.append(None)
            else:
                s = 0.0
                for spec in self.metrics:
                    val = next((m.value for m in e.metrics if m.name == spec.name), None)
                    if val is None or spec.best_value is None:
                        s += spec.weight  # max penalty
                    else:
                        s += abs(val - spec.best_value) * spec.weight
                unnorm_md.append(s)

        # normalize only the non-None entries
        md_vals = [v for v in unnorm_md if v is not None]
        norm_md_vals = self._normalize_scalar_list(md_vals) if md_vals else []
        # rebuild full-length list (None → 1.0)
        norm_md: List[float] = []
        idx = 0
        for v in unnorm_md:
            if v is None:
                norm_md.append(1.0)
            else:
                norm_md.append(norm_md_vals[idx])
                idx += 1

        # --- 4) split into two dicts ---
        pos_distances: Dict[Experience, float] = {}
        neg_distances: Dict[Experience, float] = {}
        for i, e in enumerate(experiences):
            if e.metrics is None:
                neg_distances[e] = norm_fd[i]
            else:
                pos_distances[e] = (
                    self.feature_ensemble_weight * norm_fd[i]
                    + self.metrics_ensemble_weight * norm_md[i]
                )

        return pos_distances, neg_distances
    
    def compute_learning_rates(
        self,
        selected_positive: List[Experience],
        selected_negative: List[Experience],
    ) -> Dict[Experience, float]:
        """
        Uniform allocation:
          α_pos_each = max_alpha / len(selected_positive)
          α_neg_each = min_alpha / len(selected_negative)
        """
        alphas: Dict[Experience, float] = {}
        np_ = len(selected_positive)
        nn_ = len(selected_negative)
        if np_ > 0:
            a_pos = (self.adaptative_positive_alpha_limit or 1) / np_
            for e in selected_positive:
                alphas[e] = a_pos
        if nn_ > 0:
            a_neg = (self.adaptative_negative_alpha_limit or -1) / nn_
            for e in selected_negative:
                alphas[e] = a_neg
        logger.info(
            f"Assigned α={a_pos if np_>0 else '—'} to {np_} positives, "
            f"α={a_neg if nn_>0 else '—'} to {nn_} negatives"
        )
        return alphas
    
    def select_experiences(
        self,
        pos_distances: Dict[Experience, float],
        neg_distances: Dict[Experience, float],
    ) -> Tuple[List[Experience], List[Experience]]:
        """
        Pick k_pos experiences with smallest pos_distances,
        and k_neg experiences with largest neg_distances.
        """
        # sort positives ascending
        pos_sorted = sorted(pos_distances.items(), key=lambda kv: kv[1])
        selected_pos = [e for e, _ in pos_sorted[: self.k_pos]]

        # sort negatives descending
        neg_sorted = sorted(neg_distances.items(), key=lambda kv: kv[1], reverse=True)
        selected_neg = [e for e, _ in neg_sorted[: self.k_neg]]

        logger.info(
            f"Selected {len(selected_pos)} positives, {len(selected_neg)} negatives."
        )
        return selected_pos, selected_neg
    
    def warm_up(self, generator_fn: Callable) -> Optional[Dict]:
        # 1) Extract current features
        self.pre_warm_up(generator_fn)

        # 2) Filter experiences
        exps = self.filter_experiences(self._experiences)
        if not exps:
            logger.info("No experiences to warm-start from.")
            return None

        # 3) Compute distances
        pos_dists, neg_dists = self.compute_distances(
            self.current_task_features,
            self.current_system_features,
            exps,
        )

        # 4) Select experiences
        pos_exps, neg_exps = self.select_experiences(pos_dists, neg_dists)
        if not pos_exps and not neg_exps:
            logger.info("No experiences selected.")
            return None
        
        # 5) Compute alphas & replay
        alphas = self.compute_learning_rates(pos_exps, neg_exps)
        logger.info(f"Replaying {len(alphas)} experiences.")
        self.adjust_model(alphas)

        return self._model
    
    def _normalize_scalar_list(self, values: List[float]) -> List[float]:
        """Normalize to [0,1]."""
        if not values:
            return []
        min_v, max_v = min(values), max(values)
        diff = max_v - min_v
        if abs(diff) < 1e-9:
            return [0.0 for _ in values]
        return [(v - min_v) / diff for v in values]