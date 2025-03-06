from datetime import date
import math
from typing import Callable, Dict, List, Optional, Union
from autogoal.meta_learning._experience import Experience, ExperienceStore
from autogoal.meta_learning.feature_extraction.text_classification import (
    TextClassificationFeatureExtractor,
)
from autogoal.meta_learning.normalization import Normalizer
from autogoal.meta_learning.distance import (
    DistanceMetric,
    EuclideanDistance,
    MahalanobisDistance,
)
from autogoal.meta_learning.feature_extraction.system_feature_extractor import (
    SystemFeatureExtractor,
)
from autogoal.meta_learning.sampling import ExperienceReplayModelSampler
from autogoal.meta_learning import FeatureExtractor
from autogoal.sampling import (
    UnormalizedWeightParam,
    update_model,
)
import numpy as np
from autogoal.search.utils import non_dominated_sort, crowding_distance_with_maximize


class WarmStart:
    """
    A class to adjust the internal probabilistic model of an AutoML process when starting
    based on relevant past experiences (warm starting).

    This class "warm-starting" method follows 5 steps to adjust the internal
    the AutoML process. Namely:

    1. Extracting meta-features from the current dataset/system.
    2. Computing distances between the current dataset/system and past experiences.
    3. Selecting the most relevant experiences based on distance and accuracy threshold.
    4. Computing learning rates (alphas) for adjusting the sampler.
    5. Adjusting the internal probabilistic model accordingly for each experience in their order of relevance (alpha).

    Parameters:
        threshold (float, optional): Minimum accuracy threshold for considering an experience.
            Experiences with accuracy below this threshold will be ignored. Default is `0.2`.
        k (int, optional): The maximum number of past experiences to consider.
            Default is `20`.
        max_alpha (float, optional): The maximum learning rate (alpha) used when adjusting
            the model. Default is `0.5`.
        normalizers (Optional[List[Normalizer]], optional): A list of normalizer instances
            to apply to the features before computing distances. Default is an empty list.
        distance (DistanceMetric, optional): The distance metric class to use when computing
            distances between feature vectors. Default is `EuclideanDistance`.
        dataset_feature_extractor (Optional[FeatureExtractor], optional): The feature extractor
            class to use for extracting dataset features. Default is `TextClassificationFeatureExtractor`.
        system_feature_extractor (Optional[FeatureExtractor], optional): The feature extractor
            class to use for extracting system features. Default is `SystemFeatureExtractor`.

    Attributes:
        _model (Dict): The internal probabilistic model that will be adjusted.
        generator_fn (callable): The function used to generate configurations during the warm-up.
        threshold (float): The accuracy threshold.
        k (int): The maximum number of experiences to consider.
        max_alpha (float): The maximum learning rate.
        normalizers (List[Normalizer]): List of normalizers for feature normalization.
        distance (DistanceMetric): The distance metric instance.
        dataset_feature_extractor_class (FeatureExtractor): Class for dataset feature extraction.
        system_feature_extractor_class (FeatureExtractor): Class for system feature extraction.
        X_train: Training data features of the current dataset.
        y_train: Training data labels of the current dataset.
    """

    def __init__(
        self,
        # Experience Selection Parameters
        positive_min_threshold=0.2,
        k_pos=20,
        k_neg=20,
        # Optimization Parameters
        max_alpha=0.05,
        min_alpha=-0.02,
        adaptative_positive_alpha_limit=None,  # If some, max_alpha will be computed dynamically based on the amount of positive experiences and this value
        adaptative_negative_alpha_limit=None,  # If some, min_alpha will be computed dynamically based on the amount of negative experiences and this value
        beta_scale=1.0,  # Defines how much the beta is scaled based on the distances
        beta=None,  # If None, it will be computed dynamically based on the distances by beta_scale
        # Utility Function Parameters
        utility_function="weighted_sum",  # Possible values: "weighted_sum", "linear_front", "logarithmic_front"
        f1_weight=0.5,  # Weight of the F1 score in the utility function. Only used if utility_function is "weighted_sum"
        evaluation_time_weight=0.5,  # Weight of the evaluation time in the utility function. Only used if utility_function is "weighted_sum"
        # Normalization and Distance Parameters
        normalizers: Optional[List[Normalizer]] = None,
        distance: DistanceMetric = EuclideanDistance,
        # Experience Matching and Filtering Parameters
        dataset_feature_extractor: Optional[
            FeatureExtractor
        ] = TextClassificationFeatureExtractor,
        system_feature_extractor: Optional[FeatureExtractor] = SystemFeatureExtractor,
        from_date: Optional[Union[str, date]] = None,
        to_date: Optional[Union[str, date]] = None,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        # Debugging and Testing Parameters
        exit_after_warmup: bool = False,
        on_warmup_exit: Optional[Callable[[Dict[str, float]], None]] = None,
    ):
        self._model: Dict = {}
        self.generator_fn = None
        self.positive_min_threshold = positive_min_threshold
        self.k_pos = k_pos
        self.k_neg = k_neg

        self.max_alpha = max_alpha
        self.min_alpha = min_alpha

        self.adaptative_negative_alpha_limit = adaptative_negative_alpha_limit
        if (
            self.adaptative_negative_alpha_limit is not None
            and self.adaptative_negative_alpha_limit > 0
        ):
            raise ValueError(
                "adaptative_negative_alpha_limit must be a non-positive value."
            )

        self.adaptative_positive_alpha_limit = adaptative_positive_alpha_limit
        if (
            self.adaptative_positive_alpha_limit is not None
            and self.adaptative_positive_alpha_limit < 0
        ):
            raise ValueError(
                "adaptative_negative_alpha_limit must be a non-negative value."
            )

        self.beta = beta
        self.beta_scale = beta_scale

        self.utility_function = utility_function
        total_weight = f1_weight + evaluation_time_weight
        self.f1_weight = f1_weight / total_weight  # ratio of f1 weight
        self.evaluation_time_weight = (
            evaluation_time_weight / total_weight
        )  # ratio of evaluation time weight

        if self.utility_function == "logarithmic_front":
            print("Using logarithmic front utility function.")

        if self.utility_function == "linear_front":
            print("Using linear front utility function.")

        if self.utility_function == "weighted_sum":
            print("Using weighted sum utility function.")
            print(
                f"F1 Weight: {self.f1_weight}, Evaluation Time Weight: {self.evaluation_time_weight}"
            )

        self.normalizers = normalizers or []
        self.distance = distance() if distance else EuclideanDistance()
        self.dataset_feature_extractor_class = dataset_feature_extractor
        self.system_feature_extractor_class = system_feature_extractor
        self.from_date = from_date
        self.to_date = to_date
        self.include = include
        self.exclude = exclude
        self.exit_after_warmup = exit_after_warmup
        self.on_warmup_exit = on_warmup_exit

    def pre_warm_up(self, X_train, y_train):
        """
        Stores the training data for later use during warm-up.

        Parameters:
            X_train: Training data features of the current dataset.
            y_train: Training data labels of the current dataset.

        Returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train

        # Step 1: Extract meta-features of the current dataset and current system
        self.current_dataset_features = self._extract_meta_features(
            self.X_train, self.y_train
        )
        self.current_system_features = self._extract_system_features()

    def warm_up(self, generator_fn):
        """
        Adjusts the internal probabilistic model based on relevant past experiences.

        This method performs the following steps:
        1. Extracts meta-features from the current dataset.
        2. Computes distances between the current dataset/system and past experiences.
        3. Selects the most relevant experiences based on distance and accuracy threshold.
        4. Computes learning rates (alphas) for adjusting the sampler.
        5. Adjusts the internal probabilistic model accordingly.

        Parameters:
            generator_fn (callable): A function that, given a sampler, generates configurations
                (e.g., the function that defines the search space).

        Returns:
            Dict: The updated internal probabilistic model.
        """
        self.generator_fn = generator_fn

        # Step 2: Load experiences
        experiences = ExperienceStore.load_all_experiences(
            self.from_date, self.to_date, include=self.include, exclude=self.exclude
        )

        # Step 2.1: Filter experiences based on feature extractors
        experiences = self.filter_experiences_by_feature_extractors(experiences)

        if not experiences:
            # No relevant experiences found
            return  # No need to adjust the model_sampler

        # Step 3: Compute distances and select relevant experiences
        distances = self.compute_distances(
            self.current_dataset_features, self.current_system_features, experiences
        )

        (
            selected_positive_experiences,
            positive_distances,
            selected_negative_experiences,
            negative_distances,
        ) = self.select_experiences(experiences, distances)

        if not selected_positive_experiences and not selected_negative_experiences:
            # No experiences to adjust with
            return

        # Step 4: Compute learning rates (alphas)
        alpha_experiences = self.compute_learning_rates(
            selected_positive_experiences,
            positive_distances,
            selected_negative_experiences,
            negative_distances,
        )

        print(
            f"Learning from {len(selected_positive_experiences)} positive experiences and {len(selected_negative_experiences)} negative experiences."
        )

        # Step 5: Adjust the internal probabilistic model
        self.adjust_model(alpha_experiences)

        if self.exit_after_warmup:
            print("Exiting after warm-up.")
            if self.on_warmup_exit:
                print("Executing on exit callback")
                self.on_warmup_exit(self._model)
            else:
                model_info = {
                    "FineTuneGenLLMClassifier": self._model["FineTuneGenLLMClassifier"].value,
                    "LoraGenLLMClassifier": self._model["LoraGenLLMClassifier"].value,
                    "PartialFineTuneGenLLMClassifier": self._model["PartialFineTuneGenLLMClassifier"].value,
                    "FineTuneLLMEmbeddingClassifier": self._model["FineTuneLLMEmbeddingClassifier"].value,
                    "LoraLLMEmbeddingClassifier": self._model["LoraLLMEmbeddingClassifier"].value,
                    "PartialFineTuneLLMEmbeddingClassifier": self._model["PartialFineTuneLLMEmbeddingClassifier"].value
                }
                import json
                with open('test-output.json', 'w') as f:
                    json.dump(model_info, f, indent=4)
                    
            raise ValueError("Exiting after warm-up.")
        return self._model

    def _normalize_features(
        self, feature_vectors_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Applies the sequence of normalizers to a list of feature vectors.

        Parameters:
            feature_vectors_list (List[np.ndarray]): A list of feature vectors (numpy arrays).

        Returns:
            List[np.ndarray]: A list of normalized feature vectors.
        """
        # Stack feature vectors for fitting
        feature_matrix = np.vstack(feature_vectors_list)

        # Apply each normalizer sequentially
        for normalizer in self.normalizers:
            feature_matrix = normalizer.fit_transform(feature_matrix)

        # Split back into individual feature vectors
        num_vectors = len(feature_vectors_list)
        normalized_features_list = np.vsplit(feature_matrix, num_vectors)

        # Flatten each array in the list
        normalized_features_list = [vec.flatten() for vec in normalized_features_list]

        return normalized_features_list

    def _extract_meta_features(self, X_train, y_train):
        """
        Extracts meta-features from the current dataset using the specified feature extractor.

        Parameters:
            X_train: Training data features.
            y_train: Training data labels.

        Returns:
            np.ndarray: Extracted dataset meta-features.
        """
        extractor = self.dataset_feature_extractor_class()
        return extractor.extract_features(X_train, y_train)

    def _extract_system_features(self):
        """
        Extracts system features using the specified system feature extractor.

        Returns:
            np.ndarray: Extracted system features.
        """
        extractor = self.system_feature_extractor_class()
        return extractor.extract_features()

    def compute_distances(
        self,
        current_dataset_features,
        current_system_features,
        experiences: List[Experience],
    ):
        """
        Computes the total distances between the current dataset/system and each past experience,
        using the specified distance metric.

        Parameters:
            current_dataset_features (np.ndarray): Meta-features of the current dataset.
            current_system_features (np.ndarray): System features of the current system.
            experiences (List[Experience]): List of past experiences.

        Returns:
            List[float]: Distances corresponding to each experience.
        """
        # Step 1: Combine dataset and system features for each experience
        combined_features = []
        for exp in experiences:
            combined = np.concatenate((exp.dataset_features, exp.system_features))
            combined_features.append(combined)

        # Step 2: Combine current dataset and system features
        current_combined = np.concatenate(
            (current_dataset_features, current_system_features)
        )
        combined_features.append(current_combined)  # This is the last element

        # Step 3: Normalize all combined features together
        normalized_combined_features = self._normalize_features(combined_features)

        # Step 4: Update experiences with normalized features
        for i, exp in enumerate(experiences):
            combined = normalized_combined_features[i]

            # Assuming you want to keep dataset and system features separate
            dataset_length = len(exp.dataset_features)
            exp.dataset_features = combined[:dataset_length]
            exp.system_features = combined[dataset_length:]

        # Step 5: Get normalized current combined features
        normalized_current_combined = normalized_combined_features[-1]
        current_features = normalized_current_combined

        # Step 6: Prepare the distance metric (e.g., compute and set VI for Mahalanobis)
        self._prepare_distance_metric(normalized_combined_features)

        # Step 7: Compute distances
        distances = []
        for exp in experiences:
            exp_features = np.concatenate((exp.dataset_features, exp.system_features))
            distance = self.distance.compute(current_features, exp_features)
            distances.append(distance)

        return distances

    def _prepare_distance_metric(self, combined_features: np.ndarray):
        """
        Prepares the distance metric by computing and setting necessary parameters,
        such as the inverse covariance matrix for MahalanobisDistance.

        Parameters:
            combined_features (np.ndarray): Combined normalized features from datasets and systems.

        Returns:
            None
        """
        if isinstance(self.distance, MahalanobisDistance):
            covariance = np.cov(combined_features, rowvar=False)
            try:
                VI = np.linalg.inv(covariance)
                print("Computed VI first try")
            except np.linalg.LinAlgError:
                # Regularize covariance matrix to make it invertible
                regularization_term = 1e-6 * np.eye(covariance.shape[0])
                VI = np.linalg.inv(covariance + regularization_term)
                print("Computed VI after regularization")

            self.distance.set_VI(VI)

    def select_experiences(self, experiences: List[Experience], distances):
        # Pair each experience with its distance
        experience_distance_pairs = list(zip(experiences, distances))

        # Filter positive experiences based on F1 score threshold
        positive_experiences = [
            (exp, dist)
            for exp, dist in experience_distance_pairs
            if exp.f1 is not None
            and exp.f1 > -np.Infinity
            and exp.evaluation_time is not None
            and exp.evaluation_time < np.Infinity
            and exp.f1 >= self.positive_min_threshold
        ]

        # Filter negative experiences where F1 is None or evaluation_time is None
        negative_experiences = [
            (exp, dist)
            for exp, dist in experience_distance_pairs
            if exp.f1 is None
            or exp.f1 == -np.Infinity
            or exp.evaluation_time is None
            or exp.evaluation_time == np.Infinity
        ]

        # Sort positive and negative experiences by distance
        sorted_positive_experiences = sorted(positive_experiences, key=lambda x: x[1])
        sorted_negative_experiences = sorted(negative_experiences, key=lambda x: x[1])

        # Select top-k positive and negative experiences
        selected_positive = (
            sorted_positive_experiences[: self.k_pos]
            if self.k_pos is not None
            else sorted_positive_experiences
        )
        selected_negative = (
            sorted_negative_experiences[: self.k_neg]
            if self.k_neg is not None
            else sorted_negative_experiences
        )

        # Separate experiences and distances
        selected_positive_experiences = [exp for exp, dist in selected_positive]
        positive_distances = [dist for exp, dist in selected_positive]

        selected_negative_experiences = [exp for exp, dist in selected_negative]
        negative_distances = [dist for exp, dist in selected_negative]

        return (
            selected_positive_experiences,
            positive_distances,
            selected_negative_experiences,
            negative_distances,
        )

    def filter_experiences_by_feature_extractors(
        self, experiences: List[Experience]
    ) -> List[Experience]:
        """
        Filters experiences to include only those that used the same feature extractors.

        Parameters:
            experiences (List[Experience]): A list of past experiences.

        Returns:
            List[Experience]: A list of experiences that used the same feature extractors.
        """
        dataset_extractor_name = self.dataset_feature_extractor_class.__name__
        system_extractor_name = self.system_feature_extractor_class.__name__

        filtered_experiences = [
            exp
            for exp in experiences
            if exp.dataset_feature_extractor_name == dataset_extractor_name
            and exp.system_feature_extractor_name == system_extractor_name
        ]

        return filtered_experiences

    def __linear_front_utility(self, front_idx, num_fronts):
        return (num_fronts - front_idx) / num_fronts

    def __logarithmic_front_utility(self, front_idx, num_fronts):
        return math.log(num_fronts - front_idx + 1) / math.log(num_fronts + 1)

    def __compute_front_utility(self, front_idx, num_fronts):
        return (
            self.__linear_front_utility(front_idx, num_fronts)
            if self.utility_function == "linear_front"
            else (
                self.__logarithmic_front_utility(front_idx, num_fronts)
                if self.utility_function == "logarithmic_front"
                else None
            )
        )

    def compute_learning_rates(
        self,
        selected_positive_experiences: List["Experience"],
        positive_distances: List[float],
        selected_negative_experiences: List["Experience"],
        negative_distances: List[float],
    ) -> Dict["Experience", float]:
        """
        Computes and returns the learning rates (alphas) for each relevant experience.
        """
        experience_alphas = {}

        # -------------------------
        # 1) Adaptative +/- alpha logic
        # -------------------------
        if (
            self.adaptative_negative_alpha_limit is not None
            and len(selected_negative_experiences) > 0
        ):
            self.min_alpha = self.adaptative_negative_alpha_limit / len(
                selected_negative_experiences
            )
            print(
                f"Using adaptative negative alpha limit ({self.adaptative_negative_alpha_limit}). "
                f"Computed min_alpha: {self.min_alpha}"
            )

        if (
            self.adaptative_positive_alpha_limit is not None
            and len(selected_positive_experiences) > 0
        ):
            self.max_alpha = self.adaptative_positive_alpha_limit / len(
                selected_positive_experiences
            )
            print(
                f"Using adaptative positive alpha limit ({self.adaptative_positive_alpha_limit}). "
                f"Computed max_alpha: {self.max_alpha}"
            )

        # ------------------------------------------------------
        # 2) Combine positive + negative experiences for distance stats
        # ------------------------------------------------------
        all_distances = positive_distances + negative_distances
        beta = (
            self.beta
            if self.beta is not None
            else self.compute_distance_decay_beta(all_distances)
        )

        # We'll map them for convenience
        experience_distance = {}
        for exp, dist_ in zip(selected_positive_experiences, positive_distances):
            experience_distance[exp] = dist_
        for exp, dist_ in zip(selected_negative_experiences, negative_distances):
            experience_distance[exp] = dist_

        # Initialize utilities. We'll compute them based on the utility function
        utilities = [0.0] * len(selected_positive_experiences)

        if (
            self.utility_function == "linear_front"
            or self.utility_function == "logarithmic_front"
        ):
            # ------------------------------------------------------
            # 3) Non-dominated sort for POSITIVE experiences
            # ------------------------------------------------------
            # Build a list of [f1, eval_time] for each positive experience
            objectives = [
                [exp.f1, exp.evaluation_time] for exp in selected_positive_experiences
            ]

            # Non-dominated sort: F1 => maximize=True, eval_time => maximize=False
            fronts = non_dominated_sort(objectives, maximize=[True, False])
            num_fronts = len(fronts)

            # Arrays to store rank + crowd-dist for each positive experience
            for front_idx, front in enumerate(fronts):
                # Linear scaling based on front index
                front_utility = self.__compute_front_utility(front_idx, num_fronts)
                for idx in front:
                    utilities[idx] = front_utility
        elif self.utility_function == "weighted_sum":
            # ------------------------------------------------------
            # 3) Weighted-Sum utility for POSITIVE experiences
            # ------------------------------------------------------

            # First group experiences by aliases
            # Group experiences by alias
            experience_groups = {}
            for i, exp in enumerate(selected_positive_experiences):
                alias = exp.alias or "Unknown"

                if alias not in experience_groups:
                    experience_groups[alias] = {"experiences": [], "indexes": []}

                experience_groups[alias]["experiences"].append(exp)
                experience_groups[alias]["indexes"].append(i)

            for alias, group_experiences in experience_groups.items():
                group_positive_experiences = group_experiences["experiences"]
                group_positive_indexes = group_experiences["indexes"]

                if (
                    group_positive_experiences is None
                    or len(group_positive_experiences) == 0
                ):
                    continue

                # Get F1 scores and evaluation times
                f1_scores = [exp.f1 for exp in group_positive_experiences]
                eval_times = [exp.evaluation_time for exp in group_positive_experiences]

                # Normalize F1 scores within the group
                max_f1 = max(f1_scores) or 1.0  # Prevent division by zero
                normalized_f1 = [f1 / max_f1 for f1 in f1_scores]

                # Normalize evaluation times within the group (lower is better)
                min_time = min(eval_times)
                max_time = max(eval_times)
                time_range = max_time - min_time if max_time != min_time else 1.0
                normalized_time = [
                    (time - min_time) / time_range for time in eval_times
                ]
                normalized_time_inv = [1 - t for t in normalized_time]

                # Compute utility scores per experience
                utility_scores = [
                    self.f1_weight * f1 + self.evaluation_time_weight * t_inv
                    for f1, t_inv in zip(normalized_f1, normalized_time_inv)
                ]

                for i, utility in enumerate(utility_scores):
                    utilities[group_positive_indexes[i]] = utility
        else:
            raise ValueError("Invalid utility function.")

        # ------------------------------------------------------
        # 4) Compute alpha for each positive experience based on utility and distances
        # ------------------------------------------------------
        for i, exp in enumerate(selected_positive_experiences):
            distance = experience_distance[exp]

            # Validate parameters
            if beta is None:
                raise ValueError(
                    f"Exponential decay requires a beta value. Received beta={beta}."
                )
            
            if beta < 0:
                raise ValueError(
                    f"Decay rate beta must be non-negative. Received beta={beta}."
                )

            if distance < 0:
                raise ValueError(
                    f"Distance must be non-negative. Received distance={distance}."
                )

            weight = math.exp(-beta * distance)
            alpha = self.max_alpha * utilities[i] * weight
            experience_alphas[exp] = alpha

        # ----------------------------------------
        # 5) Negative experiences is distance-based only
        # ----------------------------------------
        for exp in selected_negative_experiences:
            distance = experience_distance[exp]
            weight = math.exp(-beta * distance)

            alpha = self.min_alpha * weight
            alpha = max(alpha, self.min_alpha)
            experience_alphas[exp] = alpha

        return experience_alphas

    def compute_distance_decay_beta(self, all_distances: List[float]):
        # Compute statistics of distances
        if not all_distances:
            return self.beta_scale  # Default fallback
            
        if self.beta_scale is None:
            raise ValueError("Beta scale must be set for dynamically set the distance decay beta.")

        mean_dist = np.mean(all_distances)
        std_dist = np.std(all_distances)
        epsilon = 1e-6
        
        print(
            "Initialized beta with mean and std of distances:",
            mean_dist,
            std_dist,
        )
        print("Beta Scale:", self.beta_scale)

        beta = self.beta_scale / (max(std_dist, epsilon) + mean_dist)
        print("Computed beta:", beta)
        return beta

    def handle_error_experiences(self, experiences: List[Experience], alphas):
        """
        Assigns negative learning rates to experiences with errors (missing accuracy).

        The negative learning rate is equal in magnitude to the smallest positive learning rate
        among the successful experiences.

        Parameters:
        - experiences: A list of experiences (some may have 'accuracy' as None).
        - alphas: A list of computed learning rates for the experiences.

        Returns:
        - error_experience_alphas: A dictionary mapping error experiences to their negative learning rates.
        """
        # Find the minimum positive learning rate
        min_positive_alpha = min([alpha for alpha in alphas if alpha > 0], default=0)

        # Handle experiences with errors (missing accuracy)
        error_experience_alphas = {}
        for exp in experiences:
            if exp.accuracy is None:
                # Assign negative learning rate equal to the smallest positive alpha
                error_experience_alphas[exp] = -min_positive_alpha

        return error_experience_alphas

    def adjust_model(self, alpha_experiences: Dict[Experience, float]):
        """
        Adjusts the internal probabilistic model based on external experiences.

        For each experience, it uses its learning rate (alpha) to update the model parameters.

        Parameters:
            alpha_experiences (Dict[Experience, float]): A dictionary mapping experiences to their learning rates.

        Returns:
            None
        """

        # Extract experiences and alphas
        experience_alpha_pairs = list(alpha_experiences.items())
        experience_alpha_pairs.sort(key=lambda x: x[1], reverse=True)

        for experience, alpha in experience_alpha_pairs:
            # intialize the model sampler with the experience
            sampler = ExperienceReplayModelSampler(self._model)
            sampler.set_replicate_mode(experience)

            # generate the probabilistic model for the experience
            self.generator_fn(sampler)

            # update the warmstart model with the experience model
            self._model = update_model(self._model, sampler.updates, alpha)

            for item, value in self._model.items():
                if isinstance(value, UnormalizedWeightParam) and value.value == 0:
                    # Clip the value to a minimum of 0.001
                    self._model[item] = UnormalizedWeightParam(value=0.001)
