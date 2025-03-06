import numpy as np
from scipy.spatial.distance import (
    pdist,
    squareform,
    euclidean,
    cosine,
    cityblock,
    minkowski,
    chebyshev,
    mahalanobis,
)
from abc import ABC, abstractmethod


class DistanceMetric(ABC):
    @abstractmethod
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the distance between two feature vectors.

        Parameters:
        - vector1: The first feature vector.
        - vector2: The second feature vector.

        Returns:
        - The distance between the two vectors as a float.
        """
        pass

    @abstractmethod
    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Computes pairwise distances among a set of feature vectors.

        Parameters:
        - feature_vectors: A 2D numpy array where each row is a feature vector.

        Returns:
        - A 2D numpy array representing the pairwise distance matrix.
        """
        pass


class EuclideanDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return euclidean(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Euclidean distances among a set of feature vectors.
        """
        distances = pdist(feature_vectors, metric="euclidean")
        distance_matrix = squareform(distances)
        return distance_matrix


class CosineDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return cosine(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="cosine")
        distance_matrix = squareform(distances)
        return distance_matrix


class ManhattanDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return cityblock(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="cityblock")
        distance_matrix = squareform(distances)
        return distance_matrix


class MinkowskiDistance(DistanceMetric):
    def __init__(self, p: int = 3):
        if p < 1:
            raise ValueError("Parameter p must be greater than or equal to 1.")
        self.p = p

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return minkowski(vector1, vector2, p=self.p)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="minkowski", p=self.p)
        distance_matrix = squareform(distances)
        return distance_matrix


class ChebyshevDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return chebyshev(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="chebyshev")
        distance_matrix = squareform(distances)
        return distance_matrix


class MahalanobisDistance(DistanceMetric):
    def __init__(self):
        """
        Initializes the MahalanobisDistance instance without the inverse covariance matrix.
        """
        self.VI = None

    def set_VI(self, VI: np.ndarray):
        """
        Sets the inverse covariance matrix for distance computations.

        Parameters:
        - VI: The inverse covariance matrix.
        """
        self.VI = VI

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Mahalanobis distance between two feature vectors using the precomputed inverse covariance matrix.

        Parameters:
        - vector1: The first feature vector.
        - vector2: The second feature vector.

        Returns:
        - The Mahalanobis distance as a float.

        Raises:
        - ValueError: If the inverse covariance matrix VI has not been set.
        """
        if self.VI is None:
            raise ValueError(
                "Inverse covariance matrix VI has not been set. Call set_VI() before computing distances."
            )
        return mahalanobis(vector1, vector2, self.VI)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Mahalanobis distances among a set of feature vectors.

        Parameters:
        - feature_vectors: A 2D numpy array where each row is a feature vector.

        Returns:
        - A 2D numpy array representing the pairwise distance matrix.
        """
        covariance = np.cov(feature_vectors, rowvar=False)
        try:
            self.VI = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            # Regularize covariance matrix to make it invertible
            regularization_term = 1e-6 * np.eye(covariance.shape[0])
            self.VI = np.linalg.inv(covariance + regularization_term)
        distances = pdist(feature_vectors, metric="mahalanobis", VI=self.VI)
        distance_matrix = squareform(distances)
        return distance_matrix
 