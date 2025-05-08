import numpy as np
from typing import List, Tuple
from autogoal.meta_learning.distance import DistanceMetric, EuclideanDistance, CosineDistance

class HybridDistanceMetricBase(DistanceMetric):
    """
    Base class for hybrid distance metrics between generative tasks.
    Computes a weighted sum of distances between regular and semantic feature vectors.
    """
    def __init__(
        self,
        regular_metric: DistanceMetric,
        semantic_metric: DistanceMetric = CosineDistance(),
        weight_regular: float = 0.5,
        weight_semantic: float = 0.5,
    ):
        self.regular_metric = regular_metric
        self.semantic_metric = semantic_metric
        self.weight_regular = weight_regular
        self.weight_semantic = weight_semantic

    def compute(
        self,
        task_features1: Tuple[np.ndarray, np.ndarray],
        task_features2: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        if not (isinstance(task_features1, tuple) and len(task_features1) == 2 and isinstance(task_features2, tuple) and len(task_features2) == 2):
            raise ValueError("task_features must be tuples of length 2 (regular_vec, semantic_vec).")
        
        reg_vec1, sem_vec1 = task_features1
        reg_vec2, sem_vec2 = task_features2
        dist_reg = self.regular_metric.compute(reg_vec1, reg_vec2)
        dist_sem = self.semantic_metric.compute(sem_vec1, sem_vec2)
        return (self.weight_regular * dist_reg) + (self.weight_semantic * dist_sem)

    def compute_pairwise(
        self,
        task_feature_tuples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        regular_vectors = np.stack([t[0] for t in task_feature_tuples])
        semantic_vectors = np.stack([t[1] for t in task_feature_tuples])
        reg_matrix = self.regular_metric.compute_pairwise(regular_vectors)
        sem_matrix = self.semantic_metric.compute_pairwise(semantic_vectors)
        return (self.weight_regular * reg_matrix) + (self.weight_semantic * sem_matrix)

class HybridEuclideanCosineDistance(HybridDistanceMetricBase):
    """
    Hybrid distance metric using Euclidean for regular features and Cosine for semantic features.
    """
    def __init__(self, weight_regular: float = 0.5, weight_semantic: float = 0.5):
        super().__init__(regular_metric=EuclideanDistance(), semantic_metric=CosineDistance(),
                         weight_regular=weight_regular, weight_semantic=weight_semantic)

class HybridCosineDistance(HybridDistanceMetricBase):
    """
    Hybrid distance metric using Cosine for both regular and semantic features.
    """
    def __init__(self, weight_regular: float = 0.5, weight_semantic: float = 0.5):
        super().__init__(regular_metric=CosineDistance(), semantic_metric=CosineDistance(),
                         weight_regular=weight_regular, weight_semantic=weight_semantic)
