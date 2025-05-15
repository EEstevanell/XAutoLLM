# --- Metric class for structured, future-proof metrics ---
from math import inf
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
from autogoal.meta_learning.utils import Metric, sanitize_for_json
import numpy as np
import json
import uuid
import re # Added import
import logging

# Path to store experiences
DATA_PATH = Path.home() / ".autogoal" / "experience_store" # Changed path

class Experience:
    def __init__(
        self,
        algorithms: List[Dict[str, Any]] = None,
        task_features: Optional[Dict[str, Optional[np.ndarray]]] = None,  # Changed type hint
        system_features: Optional[np.ndarray] = None, # Changed type hint
        dataset_feature_extractor_name: str = None, 
        system_feature_extractor_name: str = None,
        timestamp: str = None,
        alias: str = None,
        cross_val_steps: Optional[int] = None,
        metrics: Optional[list] = None,
        error: Optional[str] = None,
    ):
        self.algorithms = algorithms
        self.task_features = task_features if task_features is not None else {} # Ensure it\'s a dict
        self.system_features = system_features
        self.dataset_feature_extractor_name = dataset_feature_extractor_name
        self.system_feature_extractor_name = system_feature_extractor_name
        self.timestamp = timestamp
        self.alias = alias
        self.cross_val_steps = cross_val_steps
        # Accepts list of Metric, dict, or dicts (for backward compatibility)
        self.metrics = []
        if metrics is not None:
            for m in metrics:
                if isinstance(m, Metric):
                    self.metrics.append(m)
                elif isinstance(m, dict):
                    self.metrics.append(Metric.from_dict(m))
                else:
                    raise ValueError(f"Invalid metric type: {type(m)}")
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithms': sanitize_for_json(self.algorithms),
            'task_features': sanitize_for_json(self.task_features), # Changed
            'system_features': sanitize_for_json(self.system_features),
            'dataset_feature_extractor_name': sanitize_for_json(self.dataset_feature_extractor_name),
            'system_feature_extractor_name': sanitize_for_json(self.system_feature_extractor_name),
            'timestamp': sanitize_for_json(self.timestamp),
            'alias': sanitize_for_json(self.alias),
            'cross_val_steps': sanitize_for_json(self.cross_val_steps),
            'metrics': [m.to_dict() for m in self.metrics],
            'error': sanitize_for_json(self.error),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        task_features_raw = data.get('task_features')

        # Backward compatibility for old 'dataset_features'
        if task_features_raw is None and 'dataset_features' in data:
            old_dataset_features = data.get('dataset_features')
            if old_dataset_features is not None:
                # Assuming old dataset_features were a flat list (potentially features)
                # Convert to np.ndarray if it's a list
                processed_old_features = np.array(old_dataset_features) if isinstance(old_dataset_features, list) else old_dataset_features
                task_features_raw = {"meta": processed_old_features, "semantic": None}
            else:
                task_features_raw = {} 
        elif task_features_raw is None: 
            task_features_raw = {}

        # Convert lists in task_features_raw to np.ndarray
        processed_task_features: Dict[str, Optional[np.ndarray]] = {}
        if isinstance(task_features_raw, dict):
            for key, value in task_features_raw.items():
                if isinstance(value, list):
                    processed_task_features[key] = np.array(value)
                elif isinstance(value, np.ndarray): # Already an ndarray (e.g. if called internally not from JSON)
                    processed_task_features[key] = value
                elif value is None:
                    processed_task_features[key] = None
                else:
                    # Potentially handle other types or raise error if unexpected type
                    processed_task_features[key] = value 
        else:
            # Handle cases where task_features_raw might not be a dict as expected
            # For safety, initialize to empty if structure is unexpected after migration logic
            processed_task_features = {}


        system_features_raw = data.get('system_features')
        processed_system_features: Optional[np.ndarray] = None
        if isinstance(system_features_raw, list):
            processed_system_features = np.array(system_features_raw)
        elif isinstance(system_features_raw, np.ndarray):
            processed_system_features = system_features_raw
        elif system_features_raw is None:
            processed_system_features = None
        # else: handle other types or raise error

        metrics = data.get('metrics', [])
        if isinstance(metrics, dict): # Backward compatibility
            metrics = [Metric(name=k, maximize=True, value=v) for k, v in metrics.items()]
        elif isinstance(metrics, list):
            metrics = [Metric.from_dict(m) if not isinstance(m, Metric) else m for m in metrics]
        return cls(
            algorithms=data['algorithms'],
            task_features=processed_task_features, # Pass processed features
            system_features=processed_system_features, # Pass processed features
            dataset_feature_extractor_name=data.get('dataset_feature_extractor_name', 'Unknown'),
            system_feature_extractor_name=data.get('system_feature_extractor_name', 'Unknown'),
            timestamp=data['timestamp'],
            alias=data.get('alias', 'Unknown'),
            cross_val_steps=data.get('cross_val_steps'),
            metrics=metrics,
            error=data.get('error'),
        )


# Updated ExperienceStore class
class ExperienceStore:
    logger = logging.getLogger(__name__)
    DATA_PATH = DATA_PATH

    @staticmethod
    def save_experience(experience: Experience):
        """
        Saves an experience to disk as a JSON file, grouping experiences by alias and date.
        """
        # Ensure the base data directory exists
        ExperienceStore.DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Parse the timestamp to get the date (YYYY-MM-DD)
        timestamp = experience.timestamp
        try:
            date_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            date_str = date_obj.strftime("%Y-%m-%d")
            time_str = date_obj.strftime("%H:%M:%S")
        except ValueError:
            # Handle different timestamp formats or default to today's date
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")

        # Create a directory for the alias and date
        alias_dir = ExperienceStore.DATA_PATH / experience.alias
        date_dir = alias_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename
        experience_id = str(uuid.uuid4())
        file_path = date_dir / f"{time_str}-{experience_id}.json"

        # Serialize the experience to JSON
        exp_dict = experience.to_dict()

        with open(file_path, 'w') as f:
            json.dump(exp_dict, f, indent=4)

    @staticmethod
    def load_all_experiences(
        from_date: Optional[Union[str, date]] = None,
        to_date: Optional[Union[str, date]] = None,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> List[Experience]:
        """
        Loads all experiences from disk, traversing all alias and date folders, with optional filtering.

        Args:
            from_date (Optional[Union[str, date]]): The start date in "YYYY-MM-DD" format or a date object.
            to_date (Optional[Union[str, date]]): The end date in "YYYY-MM-DD" format or a date object.
            include (Optional[str]): Regex pattern for aliases to include.
            exclude (Optional[str]): Regex pattern for aliases to exclude.

        Returns:
            A list of Experience instances.
        """
        experiences = []
        
        # Use the class attribute DATA_PATH
        current_data_path = ExperienceStore.DATA_PATH
        ExperienceStore.logger.info(f"Loading experiences from: {current_data_path}")

        if not current_data_path.exists():
            ExperienceStore.logger.warning(f"DATA_PATH does not exist: {current_data_path}")
            # No experiences saved yet
            return experiences

        # Parse the date filters if provided
        if isinstance(from_date, str):
            from_date_obj = datetime.strptime(from_date, "%Y-%m-%d").date()
        elif isinstance(from_date, date):
            from_date_obj = from_date
        else:
            from_date_obj = None

        if isinstance(to_date, str):
            to_date_obj = datetime.strptime(to_date, "%Y-%m-%d").date()
        elif isinstance(to_date, date):
            to_date_obj = to_date
        else:
            to_date_obj = None
            
        # Compile regex patterns if provided
        include_pattern = re.compile(include) if include else None
        exclude_pattern = re.compile(exclude) if exclude else None

        ExperienceStore.logger.debug(f"Filters: from_date={from_date_obj}, to_date={to_date_obj}, include_pattern={'set' if include_pattern else 'None'}, exclude_pattern={'set' if exclude_pattern else 'None'}")

        # Traverse all alias directories
        for alias_dir in current_data_path.iterdir():
            if alias_dir.is_dir():
                alias = alias_dir.name
                
                # Apply alias filtering
                if include_pattern and not include_pattern.search(alias):
                    ExperienceStore.logger.debug(f"Skipping alias '{alias}': No match for include pattern.")
                    continue
                
                if exclude_pattern and exclude_pattern.search(alias):
                    ExperienceStore.logger.debug(f"Skipping alias '{alias}': Match for exclude pattern.")
                    continue
                
                ExperienceStore.logger.debug(f"Processing alias: {alias}")    
                
                alias_exp_count = 0
                alias_exp_pos_count = 0
                alias_exp_neg_count = 0
                # Traverse all date directories within the alias directory
                for date_dir in alias_dir.iterdir():
                    if date_dir.is_dir():
                        # Parse the directory name as a date
                        try:
                            dir_date_obj = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                        except ValueError:
                            # Skip directories that do not match the date format
                            continue

                        # Apply date filtering
                        if from_date_obj and dir_date_obj < from_date_obj:
                            continue
                        if to_date_obj and dir_date_obj > to_date_obj:
                            continue

                        # Iterate over all JSON files in the date directory
                        for file_path in date_dir.glob('*.json'):
                            with open(file_path, 'r') as f:
                                try:
                                    data = json.load(f)
                                    experience = Experience.from_dict(data)
                                    experiences.append(experience)
                                    alias_exp_count += 1
                                    
                                    # Count positive/negative by checking for at least one metric with a value
                                    if experience.metrics and any(
                                        (m.value is not None and m.value != -inf)
                                        for m in experience.metrics
                                    ):
                                        alias_exp_pos_count += 1
                                    else:
                                        alias_exp_neg_count += 1
                                except Exception as e:
                                    ExperienceStore.logger.error(f"Error loading {file_path}: {e}")
                                    continue
                                
                ExperienceStore.logger.info(f"loaded {alias_exp_count} experiences for alias {alias} ({alias_exp_pos_count} positive, {alias_exp_neg_count} negative)")
        return experiences
