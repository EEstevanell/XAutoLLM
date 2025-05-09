from datetime import date
import math
from typing import Callable, Dict, List, Optional, Union
import logging # Add logging import

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
    CosineDistance, # Added
)
from autogoal.meta_learning.feature_extraction.system_feature_extractor import (
    SystemFeatureExtractor,
)
from autogoal.meta_learning.sampling import ExperienceReplayModelSampler
from autogoal.meta_learning import FeatureExtractor
from autogoal.meta_learning.utils import MetricSpec, Metric # Ensure Metric is imported if not already
from autogoal.sampling import UnormalizedWeightParam, update_model
import numpy as np
from autogoal.search.utils import non_dominated_sort, crowding_distance_with_maximize


logger = logging.getLogger(__name__) # Initialize logger for the module

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
        distance (DistanceMetric): Distance metric instance for comparing task_meta and system feature vectors.
        semantic_distance (DistanceMetric): Distance metric instance for comparing semantic feature vectors.
        dataset_feature_extractor (FeatureExtractor): Class for extracting dataset task features (must return a dict e.g. {"meta":..., "semantic":...}).
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
        current_task_features (dict): Extracted task meta-features for the current run (e.g. {"meta":..., "semantic":...}).
        current_system_features: Extracted system features for the current run.
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
        adaptative_positive_alpha_limit=None,
        adaptative_negative_alpha_limit=None,
        beta_scale=1.0,
        beta=None,
        # Utility Function Parameters
        metrics=None,
        utility_function="weighted_sum",
        # New Feature Weights
        task_meta_weight=0.4,
        semantic_weight=0.3,
        system_weight=0.3,
        # Normalization and Distance Parameters
        normalizers: Optional[List[Normalizer]] = None,
        distance: Optional[DistanceMetric] = None, # Default to EuclideanDistance if None
        semantic_distance: Optional[DistanceMetric] = None, # Default to CosineDistance if None
        # Experience Matching and Filtering Parameters
        dataset_feature_extractor: Optional[FeatureExtractor] = TextClassificationFeatureExtractor, # Must return a dict
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
                "adaptative_positive_alpha_limit must be a non-negative value." # Corrected from adaptative_negative_alpha_limit
            )

        self.beta = beta
        self.beta_scale = beta_scale
        self.utility_function = utility_function
        self.metrics = self._process_metrics(metrics)

        # Store feature weights
        self.task_meta_weight = task_meta_weight
        self.semantic_weight = semantic_weight
        self.system_weight = system_weight

        self.normalizers = normalizers or []
        # Use provided distance or default to EuclideanDistance for meta/system
        self.distance_metric = distance() if distance else EuclideanDistance()
        # Use provided semantic_distance or default to CosineDistance for semantic features
        self.semantic_distance_metric = semantic_distance() if semantic_distance else CosineDistance()

        self.dataset_feature_extractor_class = dataset_feature_extractor
        self.system_feature_extractor_class = system_feature_extractor
        self.from_date = from_date
        self.to_date = to_date
        self.include = include
        self.exclude = exclude
        self.exit_after_warmup = exit_after_warmup
        self.on_warmup_exit = on_warmup_exit

        all_experiences = ExperienceStore.load_all_experiences(
            self.from_date, self.to_date, include=self.include, exclude=self.exclude
        )
        filtered_experiences = self.filter_experiences(all_experiences)
        self._experiences = filtered_experiences
        self._infer_metric_maximize_flags()

        print(f"Using '{self.utility_function}' utility function.")
        print(f"Metrics: {[m.name for m in self.metrics]}, Weights: {[m.weight for m in self.metrics]}, Maximize: {[m.maximize for m in self.metrics]}")
        print(f"Feature weights: TaskMeta={self.task_meta_weight}, Semantic={self.semantic_weight}, System={self.system_weight}")
        print(f"Distance for Meta/System: {self.distance_metric.__class__.__name__}, Distance for Semantic: {self.semantic_distance_metric.__class__.__name__}")


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
        This ensures that the process is robust to missing metrics in the experience store.
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

    def pre_warm_up(self, X_train, y_train, current_task_features=None):
        """
        Prepares the current dataset and system for warm-starting by extracting and storing meta-features.

        Args:
            X_train: Features of the current training dataset.
            y_train: Labels of the current training dataset.
            current_task_features (optional): Precomputed task features for the dataset (a dict e.g. {"meta":..., "semantic":...}).
                                             If not provided, or if invalid, they are extracted.

        Side Effects:
            Sets `self.X_train`, `self.y_train`, `self.current_task_features`,
            and `self.current_system_features` for use in warm-up.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.current_task_features = current_task_features
        self.current_system_features = self._extract_system_features()

        recompute_task_features = False
        if self.current_task_features is None:
            logger.info("No precomputed task features provided or found in cache. Will extract.")
            recompute_task_features = True
        elif isinstance(self.current_task_features, dict):
            # Check if essential keys like 'meta' are None. 'semantic' can sometimes be None.
            if self.current_task_features.get("meta") is None:
                logger.info("Provided/cached task features have 'meta' component as None. Recomputing task features.")
                recompute_task_features = True
            # Optionally, add more checks, e.g., if semantic is critical and is None
            # elif "semantic" in self.current_task_features and self.current_task_features.get("semantic") is None:
            #     logger.info("Provided/cached task features have 'semantic' component as None. Recomputing task features.")
            #     recompute_task_features = True
        else:
            logger.warning(
                f"Provided/cached task features are not a dictionary (type: {type(self.current_task_features)}). Recomputing."
            )
            recompute_task_features = True
        
        if recompute_task_features:
            logger.info("Extracting task features for the current dataset.")
            self.current_task_features = self._extract_task_features(X_train, y_train)
        else:
            logger.info("Using provided/cached task features.")

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
        experiences = self._experiences # already filtered experiences

        if not experiences:
            # No relevant experiences found, skip warmstart gracefully
            return  # No need to adjust the model_sampler

        # Step 2: Compute distances and select relevant experiences
        distances = self.compute_distances(
            self.current_task_features, # Changed: now a dict
            self.current_system_features,
            experiences
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
        self, feature_vectors_list: List[Optional[np.ndarray]], feature_key_for_logging: str = "unknown"
    ) -> List[Optional[np.ndarray]]:
        """
        Applies the sequence of normalizers to a list of feature vectors (for a specific feature type).
        Handles None or empty arrays in the input list.
        Parameters:
            feature_vectors_list (List[Optional[np.ndarray]]): A list of 1D feature vectors (numpy arrays) of the same type, or None.
            feature_key_for_logging (str): Key name for logging purposes.
        Returns:
            List[Optional[np.ndarray]]: A list of normalized feature vectors, with Nones preserved.
        """
        if not feature_vectors_list:
            return []

        valid_feature_vectors = []
        valid_indices = []
        for i, fv in enumerate(feature_vectors_list):
            if fv is not None and isinstance(fv, np.ndarray) and fv.size > 0:
                if fv.ndim == 0: # handle 0-d arrays by reshaping
                    fv = fv.reshape(1) 
                elif fv.ndim > 1: # flatten if more than 1D, assuming it should be 1D
                    fv = fv.flatten()
                valid_feature_vectors.append(fv)
                valid_indices.append(i)

        if not valid_feature_vectors:
            # print(f"Warning: No valid feature vectors to normalize for key '{feature_key_for_logging}'. Returning original list.")
            return feature_vectors_list # All were None or empty

        try:
            # Check for consistent length among valid feature vectors before vstack
            first_len = -1
            if valid_feature_vectors:
                first_len = len(valid_feature_vectors[0])
                if not all(len(vec) == first_len for vec in valid_feature_vectors):
                    # print(f"Warning: Inconsistent feature vector lengths for key '{feature_key_for_logging}'. Shapes: {[fv.shape for fv in valid_feature_vectors]}. Skipping normalization for this group.")
                    return feature_vectors_list


            feature_matrix = np.vstack(valid_feature_vectors)
        except ValueError as e:
            # print(f"Error stacking feature vectors for key '{feature_key_for_logging}': {e}. Shapes: {[fv.shape for fv in valid_feature_vectors]}. Skipping normalization.")
            return feature_vectors_list # Return original list if stacking fails

        # Apply each normalizer sequentially
        normalized_matrix = feature_matrix
        for normalizer in self.normalizers:
            normalized_matrix = normalizer.fit_transform(normalized_matrix)

        # Split back into individual feature vectors
        normalized_valid_features = [vec.flatten() for vec in np.vsplit(normalized_matrix, len(valid_feature_vectors))]

        # Reconstruct the original list structure with Nones
        result_list = [None] * len(feature_vectors_list)
        for i, norm_fv in enumerate(normalized_valid_features):
            result_list[valid_indices[i]] = norm_fv

        return result_list

    def _extract_task_features(self, X_train, y_train) -> Dict[str, Optional[np.ndarray]]:
        if self.dataset_feature_extractor_class:
            extractor = self.dataset_feature_extractor_class()
            features = extractor.extract_features(X_train, y_train) # Expected to be a Dict[str, Optional[np.ndarray]]

            if not isinstance(features, dict):
                raise ValueError(
                    f"Dataset feature extractor {extractor.__class__.__name__} must return a dictionary. "
                    f"Got: {type(features)}."
                )

            # Ensure 'meta' and 'semantic' keys exist and values are np.ndarray or None
            for key in ["meta", "semantic"]:
                if key not in features:
                    # Consider how to handle if a key is truly optional vs. an error
                    print(f"Warning: Dataset features dictionary from {extractor.__class__.__name__} missing key: '{key}'. Assuming None.")
                    features[key] = None # Default to None if missing
                
                val = features[key]
                if val is not None and not isinstance(val, np.ndarray):
                    try:
                        # Attempt conversion if it's list-like, otherwise error
                        # This assumes feature extractors might sometimes return lists that need conversion
                        features[key] = np.array(val, dtype=float) 
                    except Exception as e:
                        raise ValueError(
                            f"Dataset features for key '{key}' from {extractor.__class__.__name__} must be np.ndarray or convertible. "
                            f"Got type: {type(val)}. Error: {e}"
                        )
            return features
        return {"meta": None, "semantic": None} # Default if no extractor

    def _extract_system_features(self) -> Optional[np.ndarray]:
        if self.system_feature_extractor_class:
            extractor = self.system_feature_extractor_class()
            features_dict = extractor.extract_features() # This returns a dict {"meta": ndarray|None, "semantic": ndarray|None}

            if features_dict is None: # Graceful handling if extractor itself returns None
                 print(f"Warning: System feature extractor {extractor.__class__.__name__} returned None overall.")
                 return None

            if not isinstance(features_dict, dict) or "meta" not in features_dict:
                raise ValueError(
                    f"System feature extractor {extractor.__class__.__name__} must return a dictionary "
                    f"with a 'meta' key. Got: {type(features_dict)}."
                )

            meta_features = features_dict.get("meta")

            if meta_features is None:
                # This is acceptable, system might not have meta features or they couldn't be extracted
                return None 

            if not isinstance(meta_features, np.ndarray):
                try:
                    # This conversion is a fallback; ideally, the extractor returns an ndarray directly for 'meta'
                    meta_features = np.array(meta_features, dtype=float)
                except Exception as e:
                    raise ValueError(
                        f"The 'meta' component of system features from {extractor.__class__.__name__} "
                        f"must be a np.ndarray or convertible to one. Got: {type(meta_features)}. Error: {e}"
                    )
            return meta_features
        return None

    def compute_distances(
        self,
        current_task_features: Dict[str, Optional[np.ndarray]],
        current_system_features: Optional[np.ndarray],
        experiences: List[Experience],
    ) -> List[float]:
        num_experiences = len(experiences)
        if num_experiences == 0:
            return []

        # To store final weighted distances for each experience
        final_distances = np.zeros(num_experiences)
        
        # --- 1. System Features ---
        if self.system_weight > 0 and current_system_features is not None and current_system_features.size > 0:
            all_system_features = [exp.system_features for exp in experiences] + [current_system_features]
            norm_all_system_features = self._normalize_features(all_system_features, "system")
            norm_current_system = norm_all_system_features[-1]

            if norm_current_system is not None:
                # Prepare distance metric (e.g., Mahalanobis)
                valid_exp_norm_system = [nf for nf in norm_all_system_features[:-1] if nf is not None]
                if valid_exp_norm_system:
                    self._prepare_distance_metric(self.distance_metric, valid_exp_norm_system + [norm_current_system])
                
                system_dists = np.full(num_experiences, np.inf)
                for i, exp_norm_system in enumerate(norm_all_system_features[:-1]):
                    if exp_norm_system is not None:
                        system_dists[i] = self.distance_metric.compute(norm_current_system, exp_norm_system)
                final_distances += self.system_weight * system_dists
        elif self.system_weight > 0: # Current system features are None/empty but weight > 0
            final_distances += self.system_weight * np.full(num_experiences, np.inf)


        # --- 2. Task Features (Iterate through keys like "meta", "semantic") ---
        # Expected keys in current_task_features: "meta", "semantic" (can be None)
        feature_configs = {
            "meta": {"weight": self.task_meta_weight, "metric": self.distance_metric},
            "semantic": {"weight": self.semantic_weight, "metric": self.semantic_distance_metric},
        }

        for key, config in feature_configs.items():
            weight = config["weight"]
            dist_metric = config["metric"]
            current_feature_vec = current_task_features.get(key)

            if weight > 0 and current_feature_vec is not None and current_feature_vec.size > 0:
                all_task_type_features = [
                    (exp.task_features.get(key) if exp.task_features else None) for exp in experiences
                ] + [current_feature_vec]
                
                norm_all_task_type_features = self._normalize_features(all_task_type_features, f"task_{key}")
                norm_current_task_vec = norm_all_task_type_features[-1]

                if norm_current_task_vec is not None:
                    # Prepare distance metric if it's the main one (for Mahalanobis on "meta")
                    if dist_metric == self.distance_metric:
                        valid_exp_norm_task_type = [nf for nf in norm_all_task_type_features[:-1] if nf is not None]
                        if valid_exp_norm_task_type:
                             self._prepare_distance_metric(dist_metric, valid_exp_norm_task_type + [norm_current_task_vec])
                    
                    task_dists = np.full(num_experiences, np.inf)
                    for i, exp_norm_task_vec in enumerate(norm_all_task_type_features[:-1]):
                        if exp_norm_task_vec is not None:
                            task_dists[i] = dist_metric.compute(norm_current_task_vec, exp_norm_task_vec)
                    final_distances += weight * task_dists
            elif weight > 0: # Current feature for this key is None/empty but weight > 0
                final_distances += weight * np.full(num_experiences, np.inf)
        
        return final_distances.tolist()

    def _prepare_distance_metric(self, distance_metric_instance: DistanceMetric, feature_vectors_for_metric: List[np.ndarray]):
        """
        Prepares a specific distance metric instance (e.g., Mahalanobis) using the given feature vectors.
        Assumes feature_vectors_for_metric contains only valid, non-None, 1D np.ndarrays of consistent length.
        """
        if not isinstance(distance_metric_instance, MahalanobisDistance):
            return # Only Mahalanobis needs this preparation

        if not feature_vectors_for_metric:
            # print("Warning: MahalanobisDistance preparation received no feature vectors. Skipping VI computation.")
            return

        # Further checks (already somewhat handled by _normalize_features, but good for direct calls)
        if not all(isinstance(fv, np.ndarray) and fv.ndim == 1 for fv in feature_vectors_for_metric):
            # print("Warning: MahalanobisDistance requires a list of 1D np.ndarrays. Skipping VI computation.")
            return
        
        first_len = feature_vectors_for_metric[0].shape[0]
        if not all(fv.shape[0] == first_len for fv in feature_vectors_for_metric):
            # print("Warning: MahalanobisDistance requires all feature vectors to have the same length. Skipping VI computation.")
            return

        try:
            combined_features_matrix = np.vstack(feature_vectors_for_metric)
            # Check if there are enough samples for the number of features
            if combined_features_matrix.shape[0] <= combined_features_matrix.shape[1]:
                # print(f"Warning: Not enough samples ({combined_features_matrix.shape[0]}) for Mahalanobis "
                #       f"distance with ({combined_features_matrix.shape[1]}) features. Covariance matrix may be singular. Skipping VI computation.")
                return

            covariance = np.cov(combined_features_matrix, rowvar=False)
            # Add a small regularization term to the diagonal to improve stability
            reg_term = 1e-6 * np.eye(covariance.shape[0])
            VI = np.linalg.inv(covariance + reg_term)
            # print(f"Computed VI for {distance_metric_instance.__class__.__name__}")
            distance_metric_instance.set_VI(VI)
        except np.linalg.LinAlgError:
            # print(f"LinAlgError computing VI for {distance_metric_instance.__class__.__name__} even after regularization. Distance may not be reliable.")
            # Optionally, fall back to Euclidean or skip setting VI
            pass # VI will not be set, Mahalanobis will raise error or use identity if VI is None
        except ValueError as e: # Can happen if combined_features_matrix is empty or other issues
            # print(f"ValueError preparing Mahalanobis for {distance_metric_instance.__class__.__name__}: {e}. Skipping VI computation.")
            pass

    def select_experiences(self, experiences: List[Experience], distances: List[float]): # Added type hint for distances
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

        # Use all metrics in self.metrics for positive/negative filtering
        positive_experiences = []
        negative_experiences = []

        for exp, dist in experience_distance_pairs:
            all_metrics_valid = True
            if not self.metrics: # If no metrics are defined, consider all experiences as potentially positive (or handle as an error/warning)
                all_metrics_valid = True # Or False, depending on desired behavior for empty self.metrics
            else:
                for metric_spec in self.metrics:
                    val = get_metric_value(exp, metric_spec.name)
                    if val is None or val == -np.Infinity or val == np.Infinity:
                        all_metrics_valid = False
                        break
            
            if all_metrics_valid:
                positive_experiences.append((exp, dist))
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

    def filter_experiences(
        self, experiences: List[Experience]
    ) -> List[Experience]:
        """
        Filters experiences to include only those that:
        - Used the same feature extractors as the current configuration.
        - Contain all required metrics as specified in self.metrics.
        For negative/error experiences, if no metrics are specified, allow inclusion if error is present.

        Parameters:
            experiences (List[Experience]): A list of past experiences.

        Returns:
            List[Experience]: A list of experiences that used the same feature extractors and have all required metrics (or error for negative).
        """
        dataset_extractor_name = self.dataset_feature_extractor_class.__name__
        system_extractor_name = self.system_feature_extractor_class.__name__
        required_metric_names = {m.name for m in self.metrics}

        def has_all_required_metrics(exp):
            if required_metric_names:
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
            # If no metrics specified ignore the experience
            return False

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

    @staticmethod
    def _get_metric_value_from_exp(exp: "Experience", metric_name: str) -> Optional[float]:
        if exp.metrics is None:
            return None
        for m in exp.metrics:
            # Support both Metric object and dict
            if hasattr(m, "name") and m.name == metric_name:
                return m.value
            elif isinstance(m, dict) and m.get("name") == metric_name:
                return m.get("value")
        return None

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

        if not selected_positive_experiences: # No positive experiences to compute utility for
            pass # utilities will remain all 0.0
        elif (
            self.utility_function == "linear_front"
            or self.utility_function == "logarithmic_front"
        ):
            # Build objectives matrix for non-dominated sort
            objectives = []
            for exp in selected_positive_experiences:
                obj_vector = []
                for metric_spec in self.metrics: # metric_spec is MetricSpec
                    val = WarmStart._get_metric_value_from_exp(exp, metric_spec.name)
                    
                    actual_val_for_sort: float
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        actual_val_for_sort = -np.Infinity if metric_spec.maximize else np.Infinity
                    elif isinstance(val, float) and math.isinf(val):
                        actual_val_for_sort = val 
                    else: # Finite number
                        actual_val_for_sort = float(val) # Ensure it's float
                    obj_vector.append(actual_val_for_sort)
                objectives.append(obj_vector)
            
            if objectives: # Ensure objectives is not empty before calling non_dominated_sort
                # Assuming non_dominated_sort is available in the scope
                # from autogoal.search._nsga2 import non_dominated_sort # Example import
                fronts = non_dominated_sort(objectives, maximize=[m.maximize for m in self.metrics])
                num_fronts = len(fronts)
                for front_idx, front in enumerate(fronts):
                    front_utility = self.__compute_front_utility(front_idx, num_fronts)
                    for original_exp_idx_in_objectives_list in front:
                        utilities[original_exp_idx_in_objectives_list] = front_utility
        
        elif self.utility_function == "weighted_sum":
            experience_groups = {}
            for i, exp in enumerate(selected_positive_experiences):
                alias = exp.alias or "Unknown" # Group by alias
                if alias not in experience_groups:
                    experience_groups[alias] = {"experiences": [], "original_indices": []}
                experience_groups[alias]["experiences"].append(exp)
                experience_groups[alias]["original_indices"].append(i)

            for alias, group_data in experience_groups.items():
                group_exp_list = group_data["experiences"]
                group_original_indices = group_data["original_indices"]

                if not group_exp_list:
                    continue

                # Stores normalized scores for each metric for all experiences in this group
                # Outer list: metrics, Inner list: scores for experiences in group_exp_list
                all_metrics_normalized_scores_for_group = []

                for metric_spec in self.metrics:
                    metric_name = metric_spec.name
                    maximize = metric_spec.maximize
                    
                    current_metric_values_in_group = [WarmStart._get_metric_value_from_exp(exp, metric_name) for exp in group_exp_list]
                    
                    # Filter out None, NaN, Inf to find min/max from valid numbers
                    valid_numeric_values = [
                        v for v in current_metric_values_in_group 
                        if v is not None and isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
                    ]
                    
                    normalized_scores_for_this_metric = [0.0] * len(group_exp_list)

                    if not valid_numeric_values:
                        # All values are None, NaN, Inf, or list is empty; all get normalized score 0
                        pass # Already initialized to 0.0
                    else:
                        min_val = min(valid_numeric_values)
                        max_val = max(valid_numeric_values)
                        range_val = max_val - min_val

                        for i, value in enumerate(current_metric_values_in_group):
                            if value is None or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                                normalized_scores_for_this_metric[i] = 0.0  # Worst score
                            elif range_val == 0:
                                # All valid values are the same.
                                normalized_scores_for_this_metric[i] = 0.5 # Neutral score
                            else:
                                if maximize:
                                    normalized_scores_for_this_metric[i] = (value - min_val) / range_val
                                else: # Minimize
                                    normalized_scores_for_this_metric[i] = (max_val - value) / range_val
                            # Clamp to [0,1] just in case of floating point issues, though theoretically should be within
                            normalized_scores_for_this_metric[i] = np.clip(normalized_scores_for_this_metric[i], 0.0, 1.0)


                    all_metrics_normalized_scores_for_group.append(normalized_scores_for_this_metric)
                
                # Calculate final utility for each experience in the group
                for i, original_idx in enumerate(group_original_indices): # i is index within the group
                    utility_score = 0.0
                    if self.metrics: # Ensure there are metrics to sum
                        for metric_idx, metric_spec in enumerate(self.metrics):
                            utility_score += metric_spec.weight * all_metrics_normalized_scores_for_group[metric_idx][i]
                        utilities[original_idx] = utility_score / sum(m.weight for m in self.metrics) if sum(m.weight for m in self.metrics) > 0 else 0 # Normalize by sum of weights
                    else:
                        utilities[original_idx] = 0 # No metrics, no utility


        else:
            raise ValueError(f"Invalid utility function: {self.utility_function}.")

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

    def compute_distance_decay_beta(self, all_distances: List[float]): # Added type hint
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
