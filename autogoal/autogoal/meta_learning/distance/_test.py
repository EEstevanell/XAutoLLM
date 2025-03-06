import unittest
import numpy as np
from scipy.spatial.distance import (
    euclidean,
    cosine,
    cityblock,
    minkowski,
    chebyshev,
    mahalanobis,
    pdist,
    squareform,
)
from numpy.linalg import inv

# Assuming the distance metric classes are defined in a module named distance_metrics
from autogoal.meta_learning.distance import (
    EuclideanDistance,
    CosineDistance,
    ManhattanDistance,
    MinkowskiDistance,
    ChebyshevDistance,
    MahalanobisDistance,
)

class TestDistanceMetrics(unittest.TestCase):
    def setUp(self):
        # Define sample vectors (original)
        self.v1 = np.array([1, 2, 3])
        self.v2 = np.array([4, 5, 6])
        self.v3 = np.array([1, 2, 3])  # Same as v1
        self.v4 = np.array([-1, -2, -3])
        self.v5 = np.array([7, 8, 9])
        self.feature_vectors = np.array([
            self.v1,
            self.v2,
            self.v3,
            self.v4,
            self.v5
        ])
        
        # Define linearly independent vectors for Mahalanobis tests
        self.independent_v1 = np.array([1, 0, 0])
        self.independent_v2 = np.array([0, 1, 0])
        self.independent_v3 = np.array([0, 0, 1])
        self.dependent = np.array([1, 1, 1])  # Added a fourth independent vector
        self.independent_feature_vectors = np.array([
            self.independent_v1,
            self.independent_v2,
            self.independent_v3,
            self.dependent
        ])

    def test_euclidean_distance_compute(self):
        metric = EuclideanDistance()
        expected = euclidean(self.v1, self.v2)
        result = metric.compute(self.v1, self.v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Euclidean compute failed.")

    def test_euclidean_distance_compute_pairwise(self):
        metric = EuclideanDistance()
        expected = squareform(pdist(self.feature_vectors, metric="euclidean"))
        result = metric.compute_pairwise(self.feature_vectors)
        np.testing.assert_array_almost_equal(result, expected, decimal=7, err_msg="Euclidean compute_pairwise failed.")

    def test_cosine_distance_compute(self):
        metric = CosineDistance()
        expected = cosine(self.v1, self.v2)
        result = metric.compute(self.v1, self.v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Cosine compute failed.")

    def test_cosine_distance_compute_pairwise(self):
        metric = CosineDistance()
        expected = squareform(pdist(self.feature_vectors, metric="cosine"))
        result = metric.compute_pairwise(self.feature_vectors)
        np.testing.assert_array_almost_equal(result, expected, decimal=7, err_msg="Cosine compute_pairwise failed.")

    def test_manhattan_distance_compute(self):
        metric = ManhattanDistance()
        expected = cityblock(self.v1, self.v2)
        result = metric.compute(self.v1, self.v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Manhattan compute failed.")

    def test_manhattan_distance_compute_pairwise(self):
        metric = ManhattanDistance()
        expected = squareform(pdist(self.feature_vectors, metric="cityblock"))
        result = metric.compute_pairwise(self.feature_vectors)
        np.testing.assert_array_almost_equal(result, expected, decimal=7, err_msg="Manhattan compute_pairwise failed.")

    def test_minkowski_distance_compute_p3(self):
        p = 3
        metric = MinkowskiDistance(p=p)
        expected = minkowski(self.v1, self.v2, p=p)
        result = metric.compute(self.v1, self.v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Minkowski compute failed for p=3.")

    def test_minkowski_distance_compute_p2(self):
        p = 2
        metric = MinkowskiDistance(p=p)
        expected = minkowski(self.v1, self.v2, p=p)
        result = metric.compute(self.v1, self.v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Minkowski compute failed for p=2.")

    def test_minkowski_distance_compute_pairwise_p3(self):
        p = 3
        metric = MinkowskiDistance(p=p)
        expected = squareform(pdist(self.feature_vectors, metric="minkowski", p=p))
        result = metric.compute_pairwise(self.feature_vectors)
        np.testing.assert_array_almost_equal(result, expected, decimal=7, err_msg="Minkowski compute_pairwise failed for p=3.")

    def test_chebyshev_distance_compute(self):
        metric = ChebyshevDistance()
        expected = chebyshev(self.v1, self.v2)
        result = metric.compute(self.v1, self.v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Chebyshev compute failed.")

    def test_chebyshev_distance_compute_pairwise(self):
        metric = ChebyshevDistance()
        expected = squareform(pdist(self.feature_vectors, metric="chebyshev"))
        result = metric.compute_pairwise(self.feature_vectors)
        np.testing.assert_array_almost_equal(result, expected, decimal=7, err_msg="Chebyshev compute_pairwise failed.")

    def test_mahalanobis_distance_with_set_VI(self):
        metric = MahalanobisDistance()
        # Use linearly independent vectors to ensure invertible covariance
        covariance = np.cov(self.independent_feature_vectors, rowvar=False)
        VI = inv(covariance)
        metric.set_VI(VI)
        expected = mahalanobis(self.independent_v1, self.independent_v2, VI)
        result = metric.compute(self.independent_v1, self.independent_v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Mahalanobis compute with set VI failed.")

    def test_mahalanobis_distance_compute_without_VI(self):
        metric = MahalanobisDistance()
        with self.assertRaises(ValueError):
            metric.compute(self.v1, self.v2)

    def test_mahalanobis_distance_compute_pairwise_with_singular_covariance(self):
        # Create feature vectors with linearly dependent vectors to cause singular covariance
        feature_vectors = np.array([
            [1, 2, 3],
            [2, 4, 6],  # 2 * v1
            [3, 6, 9],  # 3 * v1
        ])
        metric = MahalanobisDistance()
        # This should trigger regularization
        distance_matrix = metric.compute_pairwise(feature_vectors)
        # Verify that VI is set
        self.assertIsNotNone(metric.VI, "Inverse covariance matrix VI was not set.")
        # Compute distance between first and second vector
        expected = mahalanobis(feature_vectors[0], feature_vectors[1], metric.VI)
        result = metric.compute(feature_vectors[0], feature_vectors[1])
        self.assertAlmostEqual(result, expected, places=7, msg="Mahalanobis compute with singular covariance failed.")

    def test_mahalanobis_distance_compute_pairwise_regularization(self):
        # Create feature vectors with near-singular covariance
        feature_vectors = np.array([
            [1, 2, 3],
            [1.000001, 2.000001, 3.000001],
            [1.000002, 2.000002, 3.000002],
        ])
        metric = MahalanobisDistance()
        distance_matrix = metric.compute_pairwise(feature_vectors)
        # Verify that VI is set
        self.assertIsNotNone(metric.VI, "Inverse covariance matrix VI was not set.")
        # Compute distance between first and second vector
        expected = mahalanobis(feature_vectors[0], feature_vectors[1], metric.VI)
        result = metric.compute(feature_vectors[0], feature_vectors[1])
        self.assertAlmostEqual(result, expected, places=7, msg="Mahalanobis compute with regularization failed.")

    def test_minkowski_distance_invalid_p(self):
        with self.assertRaises(ValueError):
            MinkowskiDistance(p=0)  # p must be >=1

    def test_compute_same_vectors(self):
        metrics = [
            EuclideanDistance(),
            CosineDistance(),
            ManhattanDistance(),
            MinkowskiDistance(p=3),
            ChebyshevDistance(),
            # MahalanobisDistance requires at least two vectors to compute VI
        ]
        for metric in metrics:
            with self.subTest(metric=metric.__class__.__name__):
                distance = metric.compute(self.v1, self.v3)  # v1 and v3 are identical
                if isinstance(metric, CosineDistance):
                    # Cosine distance between identical vectors should be 0
                    self.assertAlmostEqual(distance, 0.0, places=7, msg=f"{metric.__class__.__name__} compute failed for identical vectors.")
                else:
                    # Other distances should be 0
                    self.assertAlmostEqual(distance, 0.0, places=7, msg=f"{metric.__class__.__name__} compute failed for identical vectors.")

    def test_compute_pairwise_same_vectors(self):
        metric = EuclideanDistance()
        distance_matrix = metric.compute_pairwise(self.feature_vectors)
        # Distance between v1 and v3 should be zero
        self.assertAlmostEqual(distance_matrix[0, 2], 0.0, places=7, msg="Euclidean compute_pairwise failed for identical vectors.")
        
        metric = CosineDistance()
        distance_matrix = metric.compute_pairwise(self.feature_vectors)
        self.assertAlmostEqual(distance_matrix[0, 2], 0.0, places=7, msg="Cosine compute_pairwise failed for identical vectors.")
        
        metric = ManhattanDistance()
        distance_matrix = metric.compute_pairwise(self.feature_vectors)
        self.assertAlmostEqual(distance_matrix[0, 2], 0.0, places=7, msg="Manhattan compute_pairwise failed for identical vectors.")
        
        metric = MinkowskiDistance(p=3)
        distance_matrix = metric.compute_pairwise(self.feature_vectors)
        self.assertAlmostEqual(distance_matrix[0, 2], 0.0, places=7, msg="Minkowski compute_pairwise failed for identical vectors.")
        
        metric = ChebyshevDistance()
        distance_matrix = metric.compute_pairwise(self.feature_vectors)
        self.assertAlmostEqual(distance_matrix[0, 2], 0.0, places=7, msg="Chebyshev compute_pairwise failed for identical vectors.")

    def test_mahalanobis_distance_with_set_VI(self):
        # Create covariance matrix and its inverse
        covariance = np.cov(self.feature_vectors, rowvar=False)
        VI = inv(covariance)
        metric = MahalanobisDistance()
        metric.set_VI(VI)
        expected = mahalanobis(self.v1, self.v2, VI)
        result = metric.compute(self.v1, self.v2)
        self.assertAlmostEqual(result, expected, places=7, msg="Mahalanobis compute with set VI failed.")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
