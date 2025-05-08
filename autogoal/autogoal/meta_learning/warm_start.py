from datetime import date
import math
from typing import Callable, Dict, List, Optional, Union
from autogoal.meta_learning._experience import Experience, ExperienceStore, Metric

# --- MetricSpec class for user input of metric configuration ---
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
from autogoal.meta_learning.utils import MetricSpec
from autogoal.sampling import (
    UnormalizedWeightParam,
    update_model,
)
import numpy as np
from autogoal.search.utils import non_dominated_sort, crowding_distance_with_maximize


class WarmStart:
    """
    Implements warm-starting for AutoML by adjusting the internal probabilistic model using relevant past experiences.

    The warm-starting process consists of the following main steps:

    1. **Meta-feature Extraction**: Extract meta-features from the current dataset and system using configurable feature extractors.
    2. **Experience Filtering**: Filter past experiences to retain only those that used the same feature extractors and contain all required metrics.
    3. **Distance Computation**: Compute distances between the current (dataset, system) and each filtered experience using a configurable distance metric and optional normalization.
    4. **Experience Selection**: Select the most relevant positive and negative experiences based on metric thresholds and distance, up to configurable limits.
    5. **Learning Rate Computation**: Compute learning rates (alphas) for each selected experience, using utility functions and distance-based decay.
    6. **Model Adjustment**: Adjust the internal probabilistic model by replaying the selected experiences, weighted by their computed learning rates.

    The process is robust to the absence of relevant experiences and will skip warm-starting gracefully if no suitable experiences are found.

    Parameters:
        positive_min_threshold (float): Minimum value for the main metric to consider an experience as positive. Default is 0.2.
        k_pos (int): Maximum number of positive experiences to use. Default is 20.
        k_neg (int): Maximum number of negative experiences to use. Default is 20.
        max_alpha (float): Maximum learning rate for positive experiences. Default is 0.05.
        min_alpha (float): Minimum (negative) learning rate for negative experiences. Default is -0.02.
        adaptative_positive_alpha_limit (float or None): If set, max_alpha is dynamically computed as this value divided by the number of positive experiences.
        adaptative_negative_alpha_limit (float or None): If set, min_alpha is dynamically computed as this value divided by the number of negative experiences.
        beta_scale (float): Scaling factor for distance-based learning rate decay. Default is 1.0.
        beta (float or None): If set, overrides dynamic computation of the decay rate.
        metrics (various): Metrics to use for utility computation. Accepts None, str, list[str], dict, list[dict], or list[MetricSpec].
        utility_function (str): Utility function for positive experience weighting. Options: 'weighted_sum', 'linear_front', 'logarithmic_front'.
        normalizers (list[Normalizer]): List of normalizers to apply to features before distance computation.
        distance (DistanceMetric): Distance metric instance for comparing feature vectors.
        dataset_feature_extractor (FeatureExtractor): Class for extracting dataset meta-features.
        system_feature_extractor (FeatureExtractor): Class for extracting system meta-features.
        from_date, to_date, include, exclude: Experience filtering options by date or alias.
        exit_after_warmup (bool): If True, exits after warm-up and optionally calls a callback.
        on_warmup_exit (callable): Callback to execute on exit after warm-up.

    Attributes:
        _model (dict): The internal probabilistic model adjusted by warm-starting.
        metrics (list[MetricSpec]): List of metrics with weights and maximize flags.
        _experiences (list[Experience]): Filtered list of relevant past experiences.
        generator_fn (callable): Function to generate configurations for the model.
        X_train, y_train: Training data for the current dataset.
        current_dataset_features, current_system_features: Extracted meta-features for the current run.
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
        metrics=None,  # Accepts None, str, list[str], dict, list[dict], or list[MetricSpec]
        utility_function="weighted_sum",
        # Normalization and Distance Parameters
        normalizers: Optional[List[Normalizer]] = None,
        distance: DistanceMetric = EuclideanDistance,
        # Experience Matching and Filtering Parameters
        dataset_feature_extractor: Optional[FeatureExtractor] = TextClassificationFeatureExtractor,
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
        # Process metrics (names/weights only)
        self.metrics = self._process_metrics(metrics)

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

        # Load and filter experiences to infer maximize flags
        all_experiences = ExperienceStore.load_all_experiences(
            self.from_date, self.to_date, include=self.include, exclude=self.exclude
        )
        filtered_experiences = self.filter_experiences_by_feature_extractors(all_experiences)
        self._experiences = filtered_experiences
        self._infer_metric_maximize_flags()

        print(f"Using '{self.utility_function}' utility function.")
        print(f"Metrics: {[m.name for m in self.metrics]}, Weights: {[m.weight for m in self.metrics]}, Maximize: {[m.maximize for m in self.metrics]}")

    def _process_metrics(self, metrics):
        """
        Process the user input for metrics and return a list of MetricSpec objects with maximize inferred from experiences.
        Accepts:
            - None: uses default metrics (f1 and evaluation_time)
            - str: single metric name
            - list[str]: list of metric names
            - dict: single metric spec as dict (only name and optionally weight)
            - list[dict]: list of metric specs as dicts
            - list[MetricSpec]: already processed
        Returns:
            List[MetricSpec]: List of MetricSpec objects with normalized weights and maximize inferred from experiences.
        """
        from autogoal.meta_learning.utils import MetricSpec

        # Default metrics (no maximize set yet)
        if metrics is None:
            metrics = [
                MetricSpec(name="f1", weight=1.0),
                MetricSpec(name="evaluation_time", weight=1.0),
            ]
        elif isinstance(metrics, MetricSpec):
            metrics = [metrics]
        elif isinstance(metrics, str):
            metrics = [MetricSpec(name=metrics, weight=1.0)]
        elif isinstance(metrics, dict):
            metrics = [MetricSpec.from_dict(metrics)]
        elif isinstance(metrics, list):
            if all(isinstance(m, MetricSpec) for m in metrics):
                pass  # Already MetricSpec
            elif all(isinstance(m, str) for m in metrics):
                metrics = [MetricSpec(name=m, weight=1.0) for m in metrics]
            elif all(isinstance(m, dict) for m in metrics):
                metrics = [MetricSpec.from_dict(m) for m in metrics]
            else:
                raise ValueError("All elements in the metrics list must be either MetricSpec, dict, or str.")
        else:
            raise ValueError("metrics must be None, a string, a dict, a MetricSpec, or a list thereof.")

        # Assign equal weights if not specified, then normalize
        n = len(metrics)
        for m in metrics:
            if not hasattr(m, "weight") or m.weight is None:
                m.weight = 1.0
        total_weight = sum(m.weight for m in metrics)
        for m in metrics:
            m.weight = m.weight / total_weight if total_weight > 0 else 1.0 / n
        return metrics

    def _infer_metric_maximize_flags(self):
        """
        Infers the `maximize` flag for each metric in `self.metrics` by inspecting the filtered experiences.
        If a metric is not found in any experience, defaults its `maximize` flag to True.
        This ensures that the system is robust to missing metrics in the experience store.
        """
        if not hasattr(self, '_experiences'):
            self._experiences = []
        for metric in self.metrics:
            found = False
            for exp in self._experiences:
                for m in (exp.metrics or []):
                    if hasattr(m, "name") and m.name == metric.name:
                        metric.maximize = m.maximize
                        found = True
                        break
                    elif isinstance(m, dict) and m.get("name") == metric.name:
                        metric.maximize = m.get("maximize", True)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Default to maximize True if not found in any experience
                metric.maximize = True

    def pre_warm_up(self, X_train, y_train, current_dataset_meta_features=None):
        """
        Prepares the current dataset and system for warm-starting by extracting and storing meta-features.

        Args:
            X_train: Features of the current training dataset.
            y_train: Labels of the current training dataset.
            current_dataset_meta_features (optional): Precomputed meta-features for the dataset. If not provided, they are extracted.

        Side Effects:
            Sets `self.X_train`, `self.y_train`, `self.current_dataset_features`, and `self.current_system_features` for use in warm-up.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.current_dataset_features = current_dataset_meta_features
        self.current_system_features = self._extract_system_features()
        if self.current_dataset_features is None:
            self.current_dataset_features = self._extract_meta_features(
                self.X_train, self.y_train
            )

    def warm_up(self, generator_fn):
        """
        Performs the full warm-starting process, adjusting the internal probabilistic model using relevant past experiences.

        Steps:
            1. Filters experiences to retain only those with matching feature extractors and all required metrics.
            2. If no relevant experiences are found, exits gracefully without adjustment.
            3. Computes distances between the current (dataset, system) and each filtered experience.
            4. Selects the most relevant positive and negative experiences based on the main metric and distance.
            5. Computes learning rates (alphas) for each selected experience using the configured utility function and distance decay.
            6. Adjusts the internal probabilistic model by replaying the selected experiences, weighted by their alphas.
            7. Optionally exits after warm-up, calling a callback or saving the model if configured.

        Args:
            generator_fn (callable): Function that, given a sampler, generates configurations for the model.

        Returns:
            dict or None: The updated internal probabilistic model, or None if no relevant experiences were found.
        """
        self.generator_fn = generator_fn
        experiences = self._experiences

        # Step 1: Filter experiences by feature extractors and required metrics
        experiences = self.filter_experiences_by_feature_extractors(experiences)

        if not experiences:
            # No relevant experiences found, skip warmstart gracefully
            return  # No need to adjust the model_sampler

        # Step 2: Compute distances and select relevant experiences
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
            # No experiences to adjust with, skip warmstart gracefully
            return

        # Step 3: Compute learning rates (alphas)
        alpha_experiences = self.compute_learning_rates(
            selected_positive_experiences,
            positive_distances,
            selected_negative_experiences,
            negative_distances,
        )

        print(
            f"Learning from {len(selected_positive_experiences)} positive experiences and {len(selected_negative_experiences)} negative experiences."
        )

        # Step 4: Adjust the internal probabilistic model
        self.adjust_model(alpha_experiences)

        if self.exit_after_warmup:
            print("Exiting after warm-up.")
            if self.on_warmup_exit:
                print("Executing on exit callback")
                self.on_warmup_exit(self._model)
            else:
                # Log the distributions for generative LLM tasks
                model_info = {}
                for key in [
                    "FineTuneGenLLMTask",
                    "LoraGenLLMTask",
                    "PartialFineTuneGenLLMTask",
                ]:
                    if key in self._model and hasattr(self._model[key], "value"):
                        model_info[key] = self._model[key].value
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

        def get_metric_value(exp, name):
            for m in (exp.metrics or []):
                # Support both Metric and dict for backward compatibility
                if hasattr(m, "name") and m.name == name:
                    return m.value
                elif isinstance(m, dict) and m.get("name") == name:
                    return m.get("value")
            return None

        # Use the first metric in self.metrics for positive/negative filtering
        main_metric = self.metrics[0]
        threshold = self.positive_min_threshold

        positive_experiences = []
        negative_experiences = []
        for exp, dist in experience_distance_pairs:
            val = get_metric_value(exp, main_metric.name)
            if val is not None and val != -np.Infinity:
                if (main_metric.maximize and val >= threshold) or (not main_metric.maximize and val <= threshold):
                    positive_experiences.append((exp, dist))
                else:
                    negative_experiences.append((exp, dist))
            else:
                negative_experiences.append((exp, dist))

        sorted_positive_experiences = sorted(positive_experiences, key=lambda x: x[1])
        sorted_negative_experiences = sorted(negative_experiences, key=lambda x: x[1])

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
        Filters experiences to include only those that:
        - Used the same feature extractors as the current configuration.
        - Contain all required metrics as specified in self.metrics.

        Parameters:
            experiences (List[Experience]): A list of past experiences.

        Returns:
            List[Experience]: A list of experiences that used the same feature extractors and have all required metrics.
        """
        dataset_extractor_name = self.dataset_feature_extractor_class.__name__
        system_extractor_name = self.system_feature_extractor_class.__name__
        required_metric_names = {m.name for m in self.metrics}

        def has_all_required_metrics(exp):
            # exp.metrics may be None, a list of Metric or dict
            if not exp.metrics:
                return False
            found = set()
            for m in exp.metrics:
                if hasattr(m, "name"):
                    found.add(m.name)
                elif isinstance(m, dict) and "name" in m:
                    found.add(m["name"])
            return required_metric_names.issubset(found)

        filtered_experiences = [
            exp
            for exp in experiences
            if exp.dataset_feature_extractor_name == dataset_extractor_name
            and exp.system_feature_extractor_name == system_extractor_name
            and has_all_required_metrics(exp)
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
            # Build objectives matrix for non-dominated sort
            objectives = []
            for exp in selected_positive_experiences:
                obj = []
                for metric in self.metrics:
                    val = None
                    for m in (exp.metrics or []):
                        if hasattr(m, "name") and m.name == metric.name:
                            val = m.value
                            break
                        elif isinstance(m, dict) and m.get("name") == metric.name:
                            val = m.get("value")
                            break
                    obj.append(val if val is not None else (-np.Infinity if metric.maximize else np.Infinity))
                objectives.append(obj)
            fronts = non_dominated_sort(objectives, maximize=[m.maximize for m in self.metrics])
            num_fronts = len(fronts)
            for front_idx, front in enumerate(fronts):
                front_utility = self.__compute_front_utility(front_idx, num_fronts)
                for idx in front:
                    utilities[idx] = front_utility
        elif self.utility_function == "weighted_sum":
            # Group by alias
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
                if not group_positive_experiences:
                    continue
                # For each metric, collect values
                metric_values = [[] for _ in self.metrics]
                for exp in group_positive_experiences:
                    for j, metric in enumerate(self.metrics):
                        val = None
                        for m in (exp.metrics or []):
                            if hasattr(m, "name") and m.name == metric.name:
                                val = m.value
                                break
                            elif isinstance(m, dict) and m.get("name") == metric.name:
                                val = m.get("value")
                                break
                        metric_values[j].append(val if val is not None else 0.0)
                # Normalize each metric
                normalized_metrics = []
                for j, vals in enumerate(metric_values):
                    maximize = self.metrics[j].maximize
                    if maximize:
                        max_v = max(vals) or 1.0
                        norm = [v / max_v for v in vals]
                    else:
                        min_v = min(vals)
                        max_v = max(vals)
                        rng = max_v - min_v if max_v != min_v else 1.0
                        norm = [1 - ((v - min_v) / rng) for v in vals]
                    normalized_metrics.append(norm)
                # Weighted sum
                for i, idx in enumerate(group_positive_indexes):
                    utility = sum(self.metrics[j].weight * normalized_metrics[j][i] for j in range(len(self.metrics)))
                    utilities[idx] = utility
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
