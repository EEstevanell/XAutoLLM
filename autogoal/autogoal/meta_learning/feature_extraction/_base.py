import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Mapping # Removed List

class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, X_train: any = None, y_train: any = None, X_test: any = None, y_test: any = None) -> Mapping[str, Optional[np.ndarray]]: # Changed List[float] to np.ndarray
        """Extracts feature vector from the dataset."""
        pass

