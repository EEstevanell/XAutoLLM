#!/usr/bin/env python
import os
from typing import Dict, List, Union
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, config_path: str, data_root: str):
        self.config_path = config_path
        self.data_root = data_root
        self.config = self._load_yaml_config()

    def _load_yaml_config(self) -> Dict:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_dataset_configs(self, dataset: str) -> Dict:
        return self.config.get(dataset, {})

    def load_data_for_alias(self, alias: str) -> pd.DataFrame:
        alias_path = os.path.join(self.data_root, alias)
        data = []
        if not os.path.isdir(alias_path):
            print(f"Alias directory not found: {alias_path}")
            return pd.DataFrame()

        # Iterate through date folders
        for date_folder in sorted(os.listdir(alias_path)):
            date_path = os.path.join(alias_path, date_folder)
            if not os.path.isdir(date_path):
                continue

            # Iterate through JSON files
            for json_file in sorted(os.listdir(date_path)):
                if not json_file.endswith('.json'):
                    continue
                json_path = os.path.join(date_path, json_file)
                try:
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                    timestamp = self.parse_timestamp(date_folder, json_file)
                    if timestamp is None:
                        continue

                    # --- Extract metrics from the new Experience format ---
                    # Metrics are now in content['metrics'] as a list of dicts with 'name' and 'value'
                    metrics_list = content.get('metrics', [])
                    metrics_dict = {m.get('name'): m.get('value', np.nan) for m in metrics_list if isinstance(m, dict) and 'name' in m}
                    f1 = metrics_dict.get('f1', np.nan)
                    accuracy = metrics_dict.get('accuracy', metrics_dict.get('Accuracy', np.nan))
                    evaluation_time = metrics_dict.get('evaluation_time', np.nan)

                    # --- Algorithm extraction remains the same ---
                    algorithm = dict(content["algorithms"][0])
                    finetuning_method = list(algorithm)[0]
                    llm = list(algorithm[finetuning_method]["inner_model"]["value"])[0]
                    params = list(algorithm[finetuning_method])[1:]

                    # Append to data list
                    data.append({
                        'alias': alias,
                        'timestamp': timestamp,
                        'f1': f1,
                        'accuracy': accuracy,
                        'evaluation_time': evaluation_time,
                        'finetuning_method': finetuning_method,
                        'llm': str(llm).removeprefix("WORD_EMB_").removeprefix("TEXT_GEN_").replace("_", " "),
                        'parameters': params,
                    })
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
                    continue

        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def parse_timestamp(self, date_folder: str, json_filename: str) -> Union[datetime, None]:
        date_part = date_folder
        time_part = json_filename.split('-')[0]  # Assuming format is 'hh:mm:ss'
        datetime_str = f"{date_part} {time_part}"
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H_%M_%S")
        except ValueError as ve:
            print(f"Timestamp parsing error for {datetime_str}: {ve}")
            return None

    def load_all_data_for_dataset(self, dataset: str) -> Dict[str, dict]:
        """
        Loads all data for a dataset, returning a nested dict: {complexity: {method: DataFrame}} for all levels, including baseline.
        This ensures a consistent structure for downstream analysis.
        """
        dataset_configs = self.get_dataset_configs(dataset)
        all_data = {}
        for complexity, methods in dataset_configs.items():
            all_data[complexity] = {}
            # methods should always be a dict mapping method names to aliases
            if isinstance(methods, dict):
                for method, alias_name in methods.items():
                    all_data[complexity][method] = self.load_data_for_alias(alias_name)
            else:
                # fallback: if not a dict, treat as a single alias (should not happen in your config)
                all_data[complexity][str(methods)] = self.load_data_for_alias(methods)
        return all_data

    def load_all_data_for_single_objective_dataset(self, dataset: str) -> Dict[str, List[pd.DataFrame]]:
        """
        This method is dedicated to single-objective experiments.
        In single-objective analyses, each candidate is evaluated over multiple seeds.
        The expected YAML configuration for a dataset should define each candidate with either a single alias (str)
        or a list of aliases (List[str]) corresponding to each seed run.

        Returns a dictionary mapping candidate names to a list of DataFrames (one per seed).
        """
        dataset_configs = self.get_dataset_configs(dataset)
        all_data = {}
        for candidate, aliases in dataset_configs.items():
            candidate_data = []
            if isinstance(aliases, list):
                for alias in aliases:
                    df = self.load_data_for_alias(alias)
                    candidate_data.append(df)
            else:
                df = self.load_data_for_alias(aliases)
                candidate_data.append(df)
            all_data[candidate] = candidate_data
        return all_data

if __name__ == "__main__":
    # Example usage for multi-objective experiments
    multi_objective_loader = DataLoader('/home/coder/autogoal/experiments/text_classification/configs/multi-objective/candidates.yaml', '/home/coder/autogoal/experiments/text_classification/data/experience_store')
    liar_data_all_configs = multi_objective_loader.load_all_data_for_dataset('liar')
    print("Multi-objective data loaded:")
    print(liar_data_all_configs)
    
    # Example usage for single-objective experiments
    single_objective_loader = DataLoader('/home/coder/autogoal/experiments/text_classification/configs/single-objective/candidates.yaml', '/home/coder/autogoal/experiments/text_classification/data/experience_store')
    liar_single_objective_data = single_objective_loader.load_all_data_for_single_objective_dataset('liar')
    print("Single-objective data loaded:")
    print(liar_single_objective_data)
