# Meta-Learning in AutoGOAL: Technical Documentation

## Overview

The meta-learning module in AutoGOAL provides a framework for leveraging past experiment experiences to accelerate and improve the search for optimal machine learning pipelines, especially in the context of large language models (LLMs) and complex NLP tasks. This system enables **warm-starting** of AutoML searches by reusing knowledge from previous tasks, using meta-features to characterize tasks and systems, and adapting the search process based on task similarity.

Key features include:
- **Meta-feature extraction** for datasets and systems
- **Experience storage and retrieval**
- **Distance metrics** for task similarity
- **Normalization** of meta-features
- **Warm-starting** the search process using relevant past experiences
- **Flexible extension points** for new tasks, feature extractors, and distance metrics

## Architecture and Components

The meta-learning system is organized into several submodules:


### 1. Feature Extraction (`feature_extraction`)
Feature extraction is the foundation of the meta-learning system. It enables the characterization of both datasets (tasks) and computational environments (systems) using numerical meta-features, which are essential for comparing tasks, measuring similarity, and transferring knowledge.

- **`FeatureExtractor` (base class):**
  - Abstract interface for extracting meta-features from datasets or systems.
  - Defines the method `extract_features`, which must be implemented by subclasses.

- **`TextClassificationFeatureExtractor`:**
  - Extracts a rich set of meta-features for text classification tasks, such as:
    - Number of classes
    - Class imbalance ratio
    - Vocabulary size
    - Average, min, max, and standard deviation of document length (in tokens/words)
    - Average word length
    - Proportion of stopwords, punctuation, digits, etc.
    - Label entropy
    - Proportion of unique tokens
    - Any other dataset-level statistics relevant to NLP
  - These features are used to represent the "nature" of a dataset in a way that is comparable across tasks.

- **`SystemFeatureExtractor`:**
  - Extracts meta-features about the computational environment, such as:
    - Number of CPU cores
    - Total and available RAM
    - GPU presence, type, and memory
    - Operating system, Python version
    - Installed library versions (e.g., PyTorch, Transformers)
    - Any other relevant system-level property
  - This allows the meta-learning system to account for hardware/software differences when transferring experience.

#### Extending Feature Extraction
To support a new task type, implement a new subclass of `FeatureExtractor` and override the `extract_features` method to return a 1D numpy array of features. Register your extractor in the relevant parts of the code (e.g., in `WarmStart`).


### 2. Experience Management (`_experience`)
The experience management subsystem is responsible for recording, storing, and retrieving the results of past AutoML runs, along with all relevant context.

- **`Experience`:**
  - Represents a single experiment (AutoML run) and contains:
    - The pipeline structure (model choices, hyperparameters, etc.)
    - The meta-features of the dataset and system (as extracted above)
    - Performance metrics (e.g., F1, accuracy, loss, evaluation time)
    - The full configuration used for the run
    - Timestamps, random seed, and any other relevant metadata
    - Optionally, error information if the run failed

- **`ExperienceStore`:**
  - Handles persistent storage and retrieval of `Experience` objects.
  - Typically stores each experience as a JSON or similar file in a structured directory (e.g., by dataset, date, or experiment type).
  - Supports filtering by date, inclusion/exclusion patterns, dataset, system, or any meta-feature.
  - Enables efficient querying for relevant past experiences during warm start.


### 3. Distance Metrics and Computation (`distance`)

Distance computation is a core part of the meta-learning system, as it determines which past experiences are most relevant to the current task. The process involves comparing the meta-features of the current dataset/system to those of all stored experiences, using a configurable distance metric.

#### DistanceMetric Interface
- **`DistanceMetric` (abstract base):**
  - Defines the interface for computing the distance between two feature vectors (`compute`) and for computing all pairwise distances in a set (`compute_pairwise`).
  - All distance metrics inherit from this class.

#### Implemented Metrics
- `EuclideanDistance`: Standard L2 norm; good general-purpose metric.
- `CosineDistance`: Measures angle between vectors; useful for high-dimensional, sparse data.
- `ManhattanDistance`: L1 norm; robust to outliers.
- `MinkowskiDistance`: Generalization of L1/L2; parameterizable.
- `ChebyshevDistance`: Maximum coordinate difference; sensitive to single large differences.
- `MahalanobisDistance`: Accounts for feature correlation; requires covariance matrix.

The choice of metric can significantly affect which experiences are considered "close" to a new task. For example, Mahalanobis distance is useful when meta-features are correlated, while Euclidean is a good default for most cases.

#### How Distance Computation Works
The distance computation process in `WarmStart` typically follows these steps:

1. **Feature Concatenation and Normalization**
   - For each experience, concatenate its dataset and system meta-features into a single vector.
   - Do the same for the current task.
   - Normalize all feature vectors together (using the configured normalizers) to ensure comparability.

2. **Distance Calculation**
   - For each experience, compute the distance between its normalized feature vector and that of the current task.
   - The result is a list of distances, one per experience.

3. **(Optional) Distance Metric Preparation**
   - Some metrics, like Mahalanobis, require fitting parameters (e.g., the inverse covariance matrix) on the set of all feature vectors.

#### Example: Distance Computation in WarmStart

Below is a simplified code excerpt from the `WarmStart` class, showing how distances are computed:

```python
def compute_distances(self, current_dataset_features, current_system_features, experiences):
    # Step 1: Combine features
    combined_features = [np.concatenate((exp.dataset_features, exp.system_features)) for exp in experiences]
    current_combined = np.concatenate((current_dataset_features, current_system_features))
    combined_features.append(current_combined)

    # Step 2: Normalize all features together
    normalized_combined_features = self._normalize_features(combined_features)

    # Step 3: Prepare distance metric if needed (e.g., Mahalanobis)
    self._prepare_distance_metric(np.vstack(normalized_combined_features))

    # Step 4: Compute distances
    distances = []
    current_features = normalized_combined_features[-1]
    for i, exp in enumerate(experiences):
        exp_features = normalized_combined_features[i]
        distance = self.distance.compute(current_features, exp_features)
        distances.append(distance)
    return distances
```

#### Example: EuclideanDistance Implementation

```python
class EuclideanDistance(DistanceMetric):
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        return np.linalg.norm(vector1 - vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(feature_vectors, metric="euclidean")
        return squareform(distances)
```

#### Extending Distance Metrics
To add a new metric, subclass `DistanceMetric` and implement `compute` (for two vectors) and `compute_pairwise` (for a matrix of vectors). Register your metric where appropriate (e.g., in the `WarmStart` constructor).


### 4. Normalization (`normalization`)
Meta-features may have different scales, distributions, or units. Normalization ensures that all features contribute comparably to distance calculations and model adjustments.

- **`Normalizer` (base class):**
  - Interface for normalizing feature vectors. Defines `fit`, `transform`, and `fit_transform` methods.

- **Implemented Normalizers:**
  - `LogNormalizer`: Applies a logarithmic transformation to reduce skewness in features (e.g., vocabulary size).
  - `MinMaxNormalizer`: Scales features to the [0, 1] range.
  - `MinMaxLogNormalizer`: Combines log and min-max scaling for robust normalization.

Multiple normalizers can be chained to handle complex feature distributions.


### 5. Warm Start (`warm_start`)
The `WarmStart` class is the heart of the meta-learning system. It enables the AutoML process to "warm start" by biasing the search towards regions of the configuration space that have performed well on similar tasks in the past.

#### Core Steps
1. **Meta-feature Extraction**
   - Uses the configured feature extractors to compute meta-features for the current dataset and system.
   - These features are stored as `current_dataset_features` and `current_system_features`.

2. **Distance Computation**
   - Computes the distance between the current task (dataset+system) and all stored experiences using the selected distance metric and normalizers.
   - Both dataset and system features are concatenated and normalized before distance calculation.

3. **Experience Selection**
   - Selects the most relevant experiences for warm start:
     - **Positive experiences**: Top `k_pos` closest experiences with performance above `positive_min_threshold` (e.g., F1 > 0.2).
     - **Negative experiences**: Top `k_neg` closest experiences with poor or failed performance.
   - Filtering can be further refined by date, inclusion/exclusion patterns, or custom logic.

4. **Computation of Learning Rates (Alphas)**
   - For each selected experience, computes a learning rate (`alpha`) that determines how much it should influence the initial model.
   - Alphas are computed based on:
     - **Utility**: How good was the experience? (e.g., F1, or a weighted sum of objectives)
     - **Distance**: How similar is the experience to the current task?
     - **Decay**: Influence decays exponentially with distance, controlled by `beta` (higher `beta` = more local influence).
     - **Sign**: Positive alphas encourage similar configurations; negative alphas discourage (for negative experiences).

5. **Adjustment of the Model**
   - The internal probabilistic model (used by the search algorithm) is updated by replaying the selected experiences, weighted by their alphas.
   - This biases the search towards promising regions and away from poor ones, based on prior knowledge.

#### Utility Functions
- The system supports different utility functions for multi-objective optimization:
  - **Weighted sum**: Combine objectives (e.g., F1, evaluation time) with user-defined weights.
  - **Linear/logarithmic front**: Assign utility based on Pareto front position.
  - Custom utility functions can be implemented for new objectives.

#### Learning Rate Decay
- The `beta` parameter controls how quickly the influence of an experience decays with distance. It can be set manually or computed dynamically based on the distribution of distances.

#### Key Methods
- `pre_warm_up(X_train, y_train)`: Extracts and stores meta-features for the current task.
- `warm_up(generator_fn)`: Loads experiences, computes distances, selects relevant ones, computes learning rates, and adjusts the model.
- `compute_distances(...)`: Computes distances between the current task and past experiences.
- `compute_learning_rates(...)`: Assigns learning rates to experiences based on utility and distance.
- `adjust_model(...)`: Updates the internal model using the selected experiences.


### 6. Experience Replay Sampler (`sampling`)
The `ExperienceReplayModelSampler` is a specialized sampler that enables the search algorithm to:
- **Replicate**: Directly sample configurations from a specific past experience (for model adjustment).
- **Regular Sample**: Sample new configurations, but with probabilities influenced by the warm-started model.

This class is essential for translating the warm-start bias into concrete pipeline configurations during search. It supports both exploration (regular sampling) and exploitation (replaying successful past configurations).


### 7. Integration with Search Algorithms
The meta-learning system is designed to be search-algorithm agnostic, but it is tightly integrated with the `NSPEWarmStartSearch` algorithm:
- **`NSPEWarmStartSearch`**: A variant of the NSPE (Non-dominated Sorting Probabilistic Evolution) search algorithm that calls `warm_up` before starting the search. This ensures that the initial probabilistic model is already biased by relevant past experiences, leading to faster convergence and better solutions, especially for new but similar tasks.
- The integration is seamless: simply pass a `WarmStart` object to the `AutoML` constructor with `search_algorithm=NSPEWarmStartSearch`.


## Usage Example

A typical usage pattern (see `execute_experiments.py`):

```python
from autogoal.meta_learning.warm_start import WarmStart
from autogoal.meta_learning.distance import EuclideanDistance
from autogoal.meta_learning.normalization import LogNormalizer, MinMaxNormalizer
from autogoal.meta_learning.feature_extraction import TextClassificationFeatureExtractor, SystemFeatureExtractor

# Prepare warm start object
warm_start = WarmStart(
    positive_min_threshold=0.2,
    k_pos=10,
    k_neg=10,
    distance=EuclideanDistance,
    normalizers=[LogNormalizer(), MinMaxNormalizer()],
    dataset_feature_extractor=TextClassificationFeatureExtractor,
    system_feature_extractor=SystemFeatureExtractor,
    beta_scale=1.0,
    max_alpha=0.05,
    min_alpha=-0.02,
)

# Extract meta-features for the current task
dataset = ...  # e.g., sst2
X_train, y_train, X_test, y_test = dataset.load()
warm_start.pre_warm_up(X_train, y_train)

# Use in AutoML search
from autogoal.ml import AutoML
from autogoal.search._warm_start_pge import NSPEWarmStartSearch

model = AutoML(
    ...,
    search_algorithm=NSPEWarmStartSearch,
    warm_start=warm_start,
)
model.fit(X_train, y_train)
```


## How to Extend the Meta-Learning System

The meta-learning system is designed for extensibility. You can adapt it to new domains, tasks, or objectives by extending its components:

- **Add a new feature extractor:**
  - Subclass `FeatureExtractor`, implement `extract_features`, and use it in `WarmStart`.
  - Register your extractor in the relevant experiment or pipeline code.
- **Add a new distance metric:**
  - Subclass `DistanceMetric`, implement `compute` and `compute_pairwise`.
  - Use your metric in the `WarmStart` constructor.
- **Add a new normalizer:**
  - Subclass `Normalizer`, implement `fit`, `transform`, and `fit_transform`.
  - Add it to the `normalizers` list in `WarmStart`.
- **Customize experience selection or utility:**
  - Modify or extend the selection/utility logic in `WarmStart` (e.g., new utility functions, new selection criteria).
- **Support new task types:**
  - Implement new feature extractors and update experience storage to include new meta-features.
  - Optionally, add new objectives and utility functions.


## Experience Store

- Experiences are stored in a directory (see `EXPERIENCE_STORE`), with each experience containing:
  - Meta-features (dataset and system)
  - Full configuration (pipeline, hyperparameters)
  - Performance metrics (F1, accuracy, evaluation time, etc.)
  - Timestamps, random seed, and metadata
  - Error information if applicable
- The system supports filtering experiences by date, inclusion/exclusion patterns, dataset, system, or any meta-feature.
- Experience files are typically stored as JSON or similar, organized by experiment type, dataset, or date for efficient retrieval.


## References and Further Reading
- See `experiments/src/execute_experiments.py` for a full example of how the meta-learning system is used in practice.
- See `autogoal/autogoal/meta_learning/` for implementation details and source code.
- For more on the theory and motivation, see the project paper and documentation in the main repository.

---


---

*This document describes the meta-learning system in AutoGOAL as implemented in the XAutoLLM project. For further details, see the code and experiment scripts in the repository. For questions or contributions, please refer to the main project README or open an issue on GitHub.*
