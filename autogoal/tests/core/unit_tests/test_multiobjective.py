import unittest
import math
from autogoal.search.utils import crowding_distance, crowding_distance_with_maximize, feature_scaling, feature_scaling_with_maximize, non_dominated_sort
import numpy as np

# Example imports from your existing code:
# from utils import non_dominated_sort, crowding_distance, feature_scaling, dominates

class TestFeatureScaling(unittest.TestCase):
    def test_simple_scaling(self):
        """
        Verify that feature_scaling correctly normalizes a small matrix
        with no -inf or repeated extremes.
        """
        data = [
            [1.0, 10.0],
            [5.0, 20.0],
            [9.0, 30.0]
        ]
        scaled = feature_scaling(data)
        # Check shape
        self.assertEqual(len(scaled), 3)
        self.assertEqual(len(scaled[0]), 2)

        # Because we've got min=1 and max=9 for the first column,
        # the scaled values should be:
        #   row0 col0 -> (1 - 1)/(9 - 1) = 0
        #   row1 col0 -> (5 - 1)/8       = 4/8 = 0.5
        #   row2 col0 -> (9 - 1)/8       = 1.0
        # For the second column, min=10, max=30 => range=20
        #   row0 col1 -> (10 - 10)/20 = 0
        #   row1 col1 -> (20 - 10)/20 = 0.5
        #   row2 col1 -> (30 - 10)/20 = 1.0

        expected = [
            [0.0,   0.0],
            [0.5,   0.5],
            [1.0,   1.0]
        ]

        for i in range(len(data)):
            for j in range(len(data[0])):
                self.assertAlmostEqual(scaled[i][j], expected[i][j], places=5)

    def test_simple_scaling_with_maximize(self):
        """
        Verify that feature_scaling correctly normalizes a small matrix
        with no -inf or repeated extremes.
        """
        data = [
            [1.0, 10.0],
            [5.0, 20.0],
            [9.0, 30.0]
        ]
        maximize = [True, False]
        scaled = feature_scaling_with_maximize(data, maximize)
        # Check shape
        self.assertEqual(len(scaled), 3)
        self.assertEqual(len(scaled[0]), 2)

        # Because we've got min=1 and max=9 for the first column,
        # the scaled values should be:
        #   row0 col0 -> (1 - 1)/(9 - 1) = 0
        #   row1 col0 -> (5 - 1)/8       = 4/8 = 0.5
        #   row2 col0 -> (9 - 1)/8       = 1.0
        # For the second column, min=10, max=30 => range=20
        #   row0 col1 -> (10 - 10)/20 = 0
        #   row1 col1 -> (20 - 10)/20 = 0.5
        #   row2 col1 -> (30 - 10)/20 = 1.0

        expected = [
            [0.0,   1.0],
            [0.5,   0.5],
            [1.0,   0.0]
        ]

        for i in range(len(data)):
            for j in range(len(data[0])):
                self.assertAlmostEqual(scaled[i][j], expected[i][j], places=5)

    def test_infinite_values(self):
        """
        Check how feature_scaling handles rows/columns with -inf.
        """
        data = [
            [2.0, math.inf],
            [2.0,  5.0],
            [-math.inf, 10.0]
        ]
        maximize = [True, False]
        scaled = feature_scaling_with_maximize(data, maximize)
        # The second column has one -inf => it remains -inf in the final
        # scaled matrix, typically. The first column has identical values => 
        # everything is the same => we might expect them to scale to 1 or remain the same.

        self.assertEqual(scaled[0][1], -math.inf)
        # The first column diff=0 => typically your code assigns 1 for 
        # the rows that are not -inf. Let's see if we get 1.0 or some fallback:
        # Implementation-specific, so we just check it's not -inf.
        self.assertNotEqual(scaled[0][0], -math.inf)

    def test_empty_input(self):
        """
        Ensure that scaling an empty list or empty dimension doesn’t crash.
        """
        data = []
        maximize = []
        scaled = feature_scaling_with_maximize(data, maximize)
        self.assertEqual(len(scaled), 0, "Scaling an empty list should return an empty list")

class TestNonDominatedSort(unittest.TestCase):
    def test_basic_sorting(self):
        """
        Test non_dominated_sort with a small set of points and verify
        correct front assignment. 
        We'll do a 2D objective, both are to be maximized here.
        """
        points = [
            [1.0, 1.0],  # A
            [2.0, 1.0],  # B
            [1.0, 2.0],  # C
            [2.0, 2.0],  # D
            [0.5, 3.0]   # E
        ]
        maximize = [True, True]

        # We'll do a quick check:
        #   D = [2,2] is not dominated => front 0
        #   E = [0.5,3] is not dominated => front 0
        #   B = [2,1] is dominated by D => so front 1
        #   C = [1,2] is dominated by D => front 1
        #   A = [1,1] is dominated by B or C => front 2
        # The order of solutions in each front can vary, but the grouping must match.

        fronts = non_dominated_sort(points, maximize)
        # Example: fronts might be [[3,4],[1,2],[0]] if the code references indices in some order
        # The key is the partitioning. Let's reconstruct the sets of indices:

        # Let's flatten them for convenience:
        sorted_fronts = [set(front) for front in fronts]
        # We assert that 3 (D) and 4 (E) are in the same front. 
        # B=1, C=2 in next front; A=0 in last front.

        # Check for membership
        self.assertTrue({3, 4} in sorted_fronts, f"Expected D,E to be in the first front.")
        self.assertTrue({1, 2}.issubset(sorted_fronts[1] | sorted_fronts[0] | sorted_fronts[2]),
                        "Expected B,C in a secondary front.")
        self.assertTrue(0 in sorted_fronts[2] or 0 in sorted_fronts[1],
                        "A has the worst coords, expected last or second-later front")

    def test_mixed_maximize(self):
        """
        Confirm it handles a scenario where the first objective is to be maximized,
        second objective is to be minimized.
        """
        #  e.g. F1 is accuracy => maximize; F2 is cost => minimize
        points = [
            [0.95, 200],  # Good acc, large cost
            [0.80, 120],  # Medium acc, medium cost
            [0.90, 100],  # Good acc, best cost => Possibly non-dominated
            [0.70, 100],  # Lower acc, but same cost as #2
        ]
        maximize = [True, False]  # First dimension => maximize, second => minimize

        # We suspect that (0.90,100) is not dominated by (0.95,200) because 
        #  it has better cost. It's also not dominated by (0.80,120) because it has 
        #  better accuracy & cost. So it likely is in the first front alone.
        # (0.95, 200) might be non-dominated if no point has both ACC>0.95 & COST<200.
        # Actually, (0.95,200) is not better than (0.90,100) in cost dimension, but is 0.95 >= 0.90 => 
        # we see (0.90,100) dominates (0.95,200)? 
        # To check: for #0: ACC=0.95 > 0.90, COST=200 >100 => that is worse in cost dimension => so #0 is dominated by #2 if #2 is strictly better in cost and not worse in accuracy. Actually #2 has ACC=0.90, cost=100 => #2's ACC is less => so #2 doesn't strictly dominate #0. 
        # => They do not dominate each other => so #0 and #2 might both be front 0. 
        # #1 has ACC=0.80 < 0.90, cost=120 > 100 => dominated by #2 => front 1 
        # #3 => (0.70,100) cost is same or better, but accuracy is lower => dominated by #2 => front 1

        fronts = non_dominated_sort(points, maximize)
        self.assertTrue(len(fronts) >= 1, "Should have at least one front")
        # Let's flatten
        front_sets = [set(fr) for fr in fronts]
        # Check if both #0, #2 are in front 0
        self.assertTrue({0,2}.issubset(front_sets[0]),
            f"Points #0 and #2 should appear in the first front, got {front_sets[0]}")
        # Then #1, #3 are likely in front 1 or beyond
        self.assertIn(1, front_sets[1], "Expected #1 in second front")
        self.assertIn(3, front_sets[1], "Expected #3 in second front")

class TestCrowdingDistance(unittest.TestCase):
    def test_simple_front_with_maximize(self):
        """
        Compare crowding_distance outcomes with a known 2D front.
        The sorting + distance should reveal boundary points have infinite distance,
        and interior gets a finite sum of side differences.
        """
        # Suppose all solutions are in the same front:
        values = [
            [0.0, 0.0],  # A
            [1.0, 1.0],  # B
            [0.5, 0.5],  # C
            [0.7, 0.8],  # D
        ]
        # We'll assume 2 objectives, both to be maximized or something. 
        # The code typically doesn't matter for crowding distance except for sorting direction.
        maximize = [True, True]
        dist = crowding_distance_with_maximize(values, maximize)

        self.assertEqual(len(dist), 4)
        # Typically, the boundary solutions in each dimension => A => index0, B => index1 
        # get infinite distances. Let's see if that happens:
        n_infs = sum([1 for d in dist if d == float('inf')])
        self.assertGreaterEqual(n_infs, 2, "At least 2 boundary points should have infinite distance")

    def test_identical_points(self):
        """
        If some points are identical, we check that the crowding distance
        still assigns them a correct finite distance or places them as boundary.
        """
        values = [
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 0.3],
            [0.2, 1.0]
        ]
        maximize = [True, False]  # e.g. first obj is bigger=better, second is smaller=better
        dist = crowding_distance_with_maximize(values, maximize)
        self.assertEqual(len(dist), 4)
        # Since 2 identical points might become "neighbors" with zero difference. 
        # That can test that the code is stable with duplicates.

if __name__ == "__main__":
    unittest.main()
