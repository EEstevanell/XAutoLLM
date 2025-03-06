import unittest

class TestComputeLearningRates(unittest.TestCase):
    def test_basic_positive_and_negative(self):
        try:
            from autogoal.meta_learning.warm_start import WarmStart
            from autogoal.meta_learning._experience import Experience
            from autogoal.search.utils import crowding_distance_with_maximize, non_dominated_sort
        except ImportError as e:
            raise ImportError("Missing required packages for warm-start tests. Please install autogoal: " + str(e))

        """
        Checks that basic positive experiences get higher alphas
        (due to better ND-sorting/crowding) and negative experiences
        get alpha = min_alpha * weight.
        """
        ws = WarmStart(min_alpha=-0.02, max_alpha=0.05)
        # Suppose we have 2 positive experiences, 1 negative
        # Distances arranged so that Epos1 is closer => bigger alpha
        # Epos2 is slightly further, negative is even further
        Epos1 = Experience(f1=0.9, evaluation_time=10.0, alias="A")
        Epos2 = Experience(f1=0.7, evaluation_time=12.0, alias="B")
        Epos3 = Experience(f1=0.85, evaluation_time=13.0)
        Epos4 = Experience(f1=0.95, evaluation_time=15.0)
        Epos5 = Experience(f1=0.5, evaluation_time=22.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4, Epos5]
        positive_distances = [0.1, 0.3, 0.2, 0.5, 0]

        Eneg1 = Experience(f1=None, evaluation_time=None, alias="Err")

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[Eneg1],
            negative_distances=[0.5],
        )
        self.assertEqual(len(alpha_dict), 6)

        alpha_pos1 = alpha_dict[Epos1]
        alpha_pos2 = alpha_dict[Epos2]
        alpha_neg1 = alpha_dict[Eneg1]

        # Check positivity
        self.assertTrue(alpha_pos1 <= ws.max_alpha and alpha_pos1 >= 0.0)
        self.assertTrue(alpha_pos2 <= ws.max_alpha and alpha_pos2 >= 0.0)

        # Check negativity
        self.assertTrue(alpha_neg1 >= ws.min_alpha and alpha_neg1 <= 0.0)

        # Typically, Epos1 is closer => alpha_pos1 > alpha_pos2
        self.assertGreaterEqual(alpha_pos1, alpha_pos2)

    def test_adaptive_alpha_limits(self):
        try:
            from autogoal.meta_learning.warm_start import WarmStart
            from autogoal.meta_learning._experience import Experience
            from autogoal.search.utils import crowding_distance_with_maximize, non_dominated_sort
        except ImportError as e:
            raise ImportError("Missing required packages for warm-start tests. Please install autogoal: " + str(e))

        """
        Confirm that adaptative alpha limits change min_alpha & max_alpha
        and hence produce smaller or bigger alphas for positives/negatives.
        """
        ws = WarmStart()
        ws.adaptative_negative_alpha_limit = -1
        ws.adaptative_positive_alpha_limit = 1

        Epos = Experience(f1=0.8, evaluation_time=5.0)
        Eneg = Experience(f1=None, evaluation_time=None)

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=[Epos],
            positive_distances=[0.0],
            selected_negative_experiences=[Eneg],
            negative_distances=[1.0],
        )
        # With 1 negative => min_alpha = -0.1 / 1 => -0.1
        # With 1 positive => max_alpha = 0.1 / 1 => 0.1
        self.assertAlmostEqual(ws.min_alpha, -1)
        self.assertAlmostEqual(ws.max_alpha, 1)

        # The positive is distance=0 => weight=exp(0)=1 => alpha=0.1*(some ND utility)
        # The ND utility => front=0 => rank=0 => norm_rank=0 => crowd might be inf => normalized? 
        # For a single solution, crowd=inf => after normalizing => you might get something => let's just check alpha <= 0.1
        alpha_pos = alpha_dict[Epos]
        self.assertLessEqual(alpha_pos, 0.1)

        # Negative => alpha = min_alpha * weight => -0.1 * exp(-1.0) => ~ -0.0367 
        # Then alpha = max(-0.0367, -0.1) => => -0.0367 is bigger => so alpha= -0.0367
        alpha_neg = alpha_dict[Eneg]
        self.assertAlmostEqual(alpha_neg, -0.0367879, places=4)

    def test_logarithmic_front(self):
        try:
            from autogoal.meta_learning.warm_start import WarmStart
            from autogoal.meta_learning._experience import Experience
            from autogoal.search.utils import crowding_distance_with_maximize, non_dominated_sort
        except ImportError as e:
            raise ImportError("Missing required packages for warm-start tests. Please install autogoal: " + str(e))

        """
        If there are no negative experiences, we skip negative logic
        but still compute alpha for positives.
        """
        # same Eval Time and distances
        ws = WarmStart(utility_function="logarithmic_front")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.1, evaluation_time=10.0)
        Epos2 = Experience(f1=0.2, evaluation_time=10.0)
        Epos3 = Experience(f1=0.3, evaluation_time=10.0)
        Epos4 = Experience(f1=0.4, evaluation_time=10.0)
        Epos5 = Experience(f1=0.5, evaluation_time=10.0)
        Epos6 = Experience(f1=0.6, evaluation_time=10.0)
        Epos7 = Experience(f1=0.7, evaluation_time=10.0)
        Epos8 = Experience(f1=0.8, evaluation_time=10.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4, Epos5, Epos6, Epos7, Epos8]
        positive_distances = [1, 1, 1, 1, 1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 8)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        alpha5 = alpha_dict[Epos5]
        alpha6 = alpha_dict[Epos6]
        alpha7 = alpha_dict[Epos7]
        alpha8 = alpha_dict[Epos8]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)
        self.assertTrue(alpha5 >= 0 and alpha5 <= ws.max_alpha)
        self.assertTrue(alpha6 >= 0 and alpha6 <= ws.max_alpha)
        self.assertTrue(alpha7 >= 0 and alpha7 <= ws.max_alpha)
        self.assertTrue(alpha8 >= 0 and alpha8 <= ws.max_alpha)

        self.assertTrue(alpha8 >= alpha7 >= alpha6 >= alpha5 >= alpha4 >= alpha3 >= alpha2 >= alpha1)

        # same F1 and distances
        ws = WarmStart(utility_function="logarithmic_front")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.7, evaluation_time=11.0)
        Epos3 = Experience(f1=0.7, evaluation_time=12.0)
        Epos4 = Experience(f1=0.7, evaluation_time=13.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 <= alpha3 <= alpha2 <= alpha1)

        
        # same Eval Time
        ws = WarmStart(utility_function="logarithmic_front")
        ws.beta_scale = 1
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.8, evaluation_time=10.0)
        Epos3 = Experience(f1=0.85, evaluation_time=10.0)
        Epos4 = Experience(f1=0.95, evaluation_time=10.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [0.01, 0.1, 1, 10]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 <= alpha3 and alpha4 <= alpha2 and alpha4 <= alpha1)

        # same distances different metrics
        ws = WarmStart(utility_function="logarithmic_front")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.8, evaluation_time=13.0)
        Epos3 = Experience(f1=0.9, evaluation_time=14.0)
        Epos4 = Experience(f1=0.99, evaluation_time=15.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        # same front
        self.assertTrue(alpha4 == alpha3 == alpha2 == alpha1)

    def test_linear_front(self):
        try:
            from autogoal.meta_learning.warm_start import WarmStart
            from autogoal.meta_learning._experience import Experience
            from autogoal.search.utils import crowding_distance_with_maximize, non_dominated_sort
        except ImportError as e:
            raise ImportError("Missing required packages for warm-start tests. Please install autogoal: " + str(e))

        """
        If there are no negative experiences, we skip negative logic
        but still compute alpha for positives.
        """
        # same Eval Time and distances
        ws = WarmStart(utility_function="linear_front")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.1, evaluation_time=10.0)
        Epos2 = Experience(f1=0.2, evaluation_time=10.0)
        Epos3 = Experience(f1=0.3, evaluation_time=10.0)
        Epos4 = Experience(f1=0.4, evaluation_time=10.0)
        Epos5 = Experience(f1=0.5, evaluation_time=10.0)
        Epos6 = Experience(f1=0.6, evaluation_time=10.0)
        Epos7 = Experience(f1=0.7, evaluation_time=10.0)
        Epos8 = Experience(f1=0.8, evaluation_time=10.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4, Epos5, Epos6, Epos7, Epos8]
        positive_distances = [1, 1, 1, 1, 1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 8)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        alpha5 = alpha_dict[Epos5]
        alpha6 = alpha_dict[Epos6]
        alpha7 = alpha_dict[Epos7]
        alpha8 = alpha_dict[Epos8]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)
        self.assertTrue(alpha5 >= 0 and alpha5 <= ws.max_alpha)
        self.assertTrue(alpha6 >= 0 and alpha6 <= ws.max_alpha)
        self.assertTrue(alpha7 >= 0 and alpha7 <= ws.max_alpha)
        self.assertTrue(alpha8 >= 0 and alpha8 <= ws.max_alpha)

        self.assertTrue(alpha8 >= alpha7 >= alpha6 >= alpha5 >= alpha4 >= alpha3 >= alpha2 >= alpha1)

        # same F1 and distances
        ws = WarmStart(utility_function="linear_front")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.7, evaluation_time=11.0)
        Epos3 = Experience(f1=0.7, evaluation_time=12.0)
        Epos4 = Experience(f1=0.7, evaluation_time=13.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 <= alpha3 <= alpha2 <= alpha1)

        
        # same Eval Time and F1 to check distance
        ws = WarmStart(utility_function="linear_front")
        ws.beta_scale = 1
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.7, evaluation_time=10.0)
        Epos3 = Experience(f1=0.7, evaluation_time=10.0)
        Epos4 = Experience(f1=0.7, evaluation_time=10.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [0.01, 0.1, 1, 10]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 <= alpha3 <= alpha2 <= alpha1)

        # same distances different metrics
        ws = WarmStart(utility_function="linear_front")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.8, evaluation_time=13.0)
        Epos3 = Experience(f1=0.9, evaluation_time=14.0)
        Epos4 = Experience(f1=0.99, evaluation_time=15.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        # same front as there are no dominated solutions
        self.assertTrue(alpha4 == alpha3 == alpha2 == alpha1)

    def test_weighted_sum(self):
        try:
            from autogoal.meta_learning.warm_start import WarmStart
            from autogoal.meta_learning._experience import Experience
            from autogoal.search.utils import crowding_distance_with_maximize, non_dominated_sort
        except ImportError as e:
            raise ImportError("Missing required packages for warm-start tests. Please install autogoal: " + str(e))

        """
        If there are no negative experiences, we skip negative logic
        but still compute alpha for positives.
        """
        # same Eval Time and distances
        ws = WarmStart(utility_function="weighted_sum")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.8, evaluation_time=10.0)
        Epos3 = Experience(f1=0.85, evaluation_time=10.0)
        Epos4 = Experience(f1=0.95, evaluation_time=10.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 >= alpha3 >= alpha2 >= alpha1)

        # same F1 and distances
        ws = WarmStart(utility_function="weighted_sum")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.7, evaluation_time=11.0)
        Epos3 = Experience(f1=0.7, evaluation_time=12.0)
        Epos4 = Experience(f1=0.7, evaluation_time=13.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 <= alpha3 <= alpha2 <= alpha1)

        
        # same Eval Time
        ws = WarmStart(utility_function="weighted_sum")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.8, evaluation_time=10.0)
        Epos3 = Experience(f1=0.85, evaluation_time=10.0)
        Epos4 = Experience(f1=0.95, evaluation_time=10.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [0.01, 0.1, 1, 10]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 <= alpha3 and alpha4 <= alpha2 and alpha4 <= alpha1)

        # same distances different metrics
        ws = WarmStart(utility_function="weighted_sum")
        ws.beta_scale = 0.5
        Epos1 = Experience(f1=0.7, evaluation_time=10.0)
        Epos2 = Experience(f1=0.8, evaluation_time=13.0)
        Epos3 = Experience(f1=0.9, evaluation_time=14.0)
        Epos4 = Experience(f1=0.99, evaluation_time=15.0)
        positive_experiences = [Epos1, Epos2, Epos3, Epos4]
        positive_distances = [1, 1, 1, 1]

        alpha_dict = ws.compute_learning_rates(
            selected_positive_experiences=positive_experiences,
            positive_distances=positive_distances,
            selected_negative_experiences=[],
            negative_distances=[],
        )

        self.assertEqual(len(alpha_dict), 4)
        self.assertIn(Epos1, alpha_dict)
        self.assertIn(Epos2, alpha_dict)
        self.assertIn(Epos3, alpha_dict)
        self.assertIn(Epos4, alpha_dict)

        # With no negatives, min_alpha stays -0.02
        # Also check that alpha is actually computed
        alpha1 = alpha_dict[Epos1]
        alpha2 = alpha_dict[Epos2]
        alpha3 = alpha_dict[Epos3]
        alpha4 = alpha_dict[Epos4]
        self.assertTrue(alpha1 >= 0 and alpha1 <= ws.max_alpha)
        self.assertTrue(alpha2 >= 0 and alpha2 <= ws.max_alpha)
        self.assertTrue(alpha3 >= 0 and alpha3 <= ws.max_alpha)
        self.assertTrue(alpha4 >= 0 and alpha4 <= ws.max_alpha)

        self.assertTrue(alpha4 != alpha3 != alpha2 != alpha1)

if __name__ == "__main__":
    unittest.main()

# Manual test execution
# if __name__ == "__main__":
#     test_suite = TestComputeLearningRates()
#     test_suite.test_basic_positive_and_negative()
#     test_suite.test_adaptive_alpha_limits()
#     test_suite.test_weighted_sum()
#     test_suite.test_linear_front()
#     test_suite.test_logarithmic_front()
