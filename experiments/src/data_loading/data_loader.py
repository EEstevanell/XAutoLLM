import os
from typing import Dict
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

    def _load_yaml_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_dataset_configs(self, dataset: str):
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

                    # Extract metrics from content
                    macro_f1 = content.get('f1', np.nan)
                    accuracy = content.get('accuracy', np.nan)
                    evaluation_time = content.get('evaluation_time', np.nan)

                    algorithm = dict(content["algorithms"][0])
                    finetuning_method = list(algorithm)[0]
                    llm = list(algorithm[finetuning_method]["inner_model"]["value"])[0]
                    params = list(algorithm[finetuning_method])[1:]

                    # Append to data list
                    data.append({
                        'alias': alias,
                        'timestamp': timestamp,
                        'macro_f1': macro_f1,
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

    def parse_timestamp(self, date_folder: str, json_filename: str) -> datetime:
        date_part = date_folder
        time_part = json_filename.split('-')[0]  # Assuming format is 'hh:mm:ss'
        datetime_str = f"{date_part} {time_part}"
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H_%M_%S")
        except ValueError as ve:
            print(f"Timestamp parsing error for {datetime_str}: {ve}")
            return None

    def load_all_data_for_dataset(self, dataset: str) -> Dict[str, pd.DataFrame]:
        dataset_configs = self.get_dataset_configs(dataset)
        all_data = {}
        
        for complexity in dataset_configs:
            all_data[complexity] = {}
            if complexity == "baseline":
                alias_name = dataset_configs[complexity]
                all_data[complexity] = self.load_data_for_alias(alias_name)
            else:
                for method in dataset_configs[complexity]:
                    alias_name = dataset_configs[complexity][method]
                    all_data[complexity][method] = self.load_data_for_alias(alias_name)

        return all_data

if __name__ == "__main__":
    multi_objective_loader = DataLoader('experiments\configs\multi-objective\candidates.yaml', 'data/experience_store')
    single_objective_loader = DataLoader('experiments\configs\single-objective', 'data/experience_store')

    # Example usage to load all data for a specific dataset (e.g., "liar")
    liar_data_all_configs = multi_objective_loader.load_all_data_for_dataset('liar')
    
    # Print or process the loaded data as needed
    print(liar_data_all_configs)
