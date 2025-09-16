#!/usr/bin/env python3
"""
Experiment configuration generator for AutoGOAL framework.

This script loads candidate configurations from YAML files and creates appropriate 
experiment configurations using the WarmstartConfigParser. It supports both 
single-objective and multi-objective experiments, with single-objective as default.

Usage:
    python run_experiment.py [--experiment_type single|multi]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import re

import yaml

from text_classification.src.data_loading import WarmstartConfigParser
from text_classification.src.data_loading.data_loader import DataLoader


# Configure logging properly
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)


class ExperimentConfigGenerator:
    """Generator for experiment configurations for both single-objective and multi-objective experiments."""
    
    def __init__(self, experiment_type: str = "single"):
        """
        Initialize the experiment configuration generator.
        
        Args:
            experiment_type: Type of experiment ('single' or 'multi'), defaults to 'single'
        """
        self.experiment_type = experiment_type
        self.config_path = Path(f"/home/coder/autogoal/experiments/configs/{experiment_type}-objective/candidates.yaml")
        self.data_root = Path("/home/coder/autogoal/experiments/data/experience_store")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Initialize DataLoader
        self.loader = DataLoader(str(self.config_path), str(self.data_root))
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_candidate_configs(self, dataset: str, bias: str, method: Optional[str] = None) -> List[str]:
        """
        Get candidate configuration strings for the specified parameters.
        
        Args:
            dataset: Name of the dataset
            bias: Bias level ('baseline', 'low', 'moderate', 'high')
            method: Method name, required only for multi-objective experiments
            
        Returns:
            List of candidate configuration strings
        """
        try:
            if self.experiment_type == 'single':
                # For single-objective, all bias levels (including baseline) have lists of candidates
                candidates = self.config[dataset][bias]
                return candidates if isinstance(candidates, list) else [candidates]
            else:  # multi-objective
                if bias == "baseline":
                    # For multi-objective baseline, we have a simple string
                    baseline_value = self.config[dataset][bias]
                    return [baseline_value] if baseline_value else []
                else:
                    # For other bias levels in multi-objective, we need the method
                    if method is None:
                        raise ValueError("Method parameter is required for multi-objective experiments")
                    candidate = self.config[dataset][bias][method]
                    return [candidate] if candidate else []
        except KeyError as e:
            logger.error(f"Configuration not found for {dataset}/{bias}/{method if method else ''}: {e}")
            return []
    
    def extract_seed_from_candidate(self, candidate_str: str) -> int:
        """
        Extract the random seed value from a candidate string.
        
        Args:
            candidate_str: Candidate configuration string
            
        Returns:
            Random seed as integer, or 42 as default
        """
        seed_match = re.search(r'seed_(\d+)', candidate_str)
        if seed_match:
            return int(seed_match.group(1))
        return 42  # Default seed if not found
    
    def create_experiment_config(self, candidate_str: str, dataset_name: str, is_baseline: bool = False) -> Dict[str, Any]:
        """
        Generate experiment configuration dictionary from config string.
        
        Args:
            candidate_str: Configuration string from the YAML file
            dataset_name: Name of the dataset
            is_baseline: Whether this is a baseline configuration
            
        Returns:
            Complete experiment configuration dictionary
        """
        if is_baseline and self.experiment_type == 'multi':
            # For multi-objective baseline, create a minimal config with just the seed
            random_seed = self.extract_seed_from_candidate(candidate_str)
            return {"random_seed": random_seed}
        
        # Get the full configuration from WarmstartConfigParser
        config = WarmstartConfigParser.parse(candidate_str, dataset_name)
        
        # For single-objective experiments, we enforce the weights regardless of parser output
        if self.experiment_type == 'single':
            config['f1_weight'] = 1
            config['evaluation_time_weight'] = 0
            
        return config
    
    def generate_dataset_configs(self, dataset: str, bias: str, method: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate configurations for a dataset with specific bias level.
        
        Args:
            dataset: Name of the dataset
            bias: Bias level ('baseline', 'low', 'moderate', 'high')
            method: Method name, required only for multi-objective experiments
            
        Returns:
            List of experiment data dictionaries
        """
        # Get candidate configurations
        candidates = self.get_candidate_configs(dataset, bias, method)
        is_baseline = (bias == "baseline")
        
        experiment_data_list = []
        
        # Process each candidate and generate experiment data
        for candidate_str in candidates:
            logger.info(f"Processing candidate: {candidate_str}")
            
            # Use the candidate string as the experiment ID
            experiment_id = candidate_str
            
            # Generate config for this candidate
            config = self.create_experiment_config(candidate_str, dataset, is_baseline)
            
            # Create experiment data dictionary with is_baseline flag
            experiment_data = {
                "dataset": dataset,
                "experiment_id": experiment_id,
                "config": config,
                "is_baseline": is_baseline
            }
            
            experiment_data_list.append(experiment_data)
            
        return experiment_data_list
    
    def generate_all_experiment_configs(self) -> List[Dict[str, Any]]:
        """
        Generate configurations for all experiments available in the configuration file.
        
        Returns:
            List of experiment data dictionaries, each containing:
                - dataset: Name of the dataset
                - experiment_id: Unique identifier for the experiment
                - config: Configuration dictionary
                - is_baseline: Boolean indicating if this is a baseline experiment
        """
        all_experiment_data = []
        
        for dataset, bias_levels in self.config.items():
            logger.info(f"Processing dataset: {dataset}")
            
            for bias in bias_levels:
                logger.info(f"  Processing bias level: {bias}")
                
                if self.experiment_type == 'single':
                    experiment_data_list = self.generate_dataset_configs(dataset, bias)
                    all_experiment_data.extend(experiment_data_list)
                else:  # multi-objective
                    if bias == "baseline":
                        experiment_data_list = self.generate_dataset_configs(dataset, bias)
                        all_experiment_data.extend(experiment_data_list)
                    else:
                        for method in self.config[dataset][bias]:
                            logger.info(f"    Processing method: {method}")
                            experiment_data_list = self.generate_dataset_configs(dataset, bias, method)
                            all_experiment_data.extend(experiment_data_list)
        
        return all_experiment_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate AutoGOAL experiment configurations'
    )
    
    parser.add_argument(
        '--experiment_type',
        type=str,
        choices=['single', 'multi'],
        default='single',  # Set single-objective as default
        help='Type of experiment (single-objective or multi-objective, defaults to single)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    try:
        # Initialize experiment configuration generator
        generator = ExperimentConfigGenerator(args.experiment_type)
        
        # Generate all experiment configurations
        all_experiment_data = generator.generate_all_experiment_configs()
        
        # Log experiment summary
        logger.info(f"Generated {len(all_experiment_data)} experiment configurations:")
        
        # Group experiments by dataset and baseline status for better readability
        datasets = set(data["dataset"] for data in all_experiment_data)
        for dataset in datasets:
            dataset_experiments = [data for data in all_experiment_data if data["dataset"] == dataset]
            baselines = [data for data in dataset_experiments if data["is_baseline"]]
            non_baselines = [data for data in dataset_experiments if not data["is_baseline"]]
            
            logger.info(f"  {dataset}: {len(dataset_experiments)} experiment(s) - {len(baselines)} baseline(s), {len(non_baselines)} non-baseline(s)")
            
            # Show sample configuration for each dataset
            if non_baselines and non_baselines[0]["config"] is not None:
                logger.info(f"    Sample non-baseline configuration for {dataset}:")
                sample_config = non_baselines[0]["config"]
                for key, value in sorted(sample_config.items()):
                    if key not in ["normalizers"]:  # Skip complex objects
                        logger.info(f"      {key}: {value}")
            
            if baselines:
                logger.info(f"    Sample baseline configuration for {dataset}:")
                sample_config = baselines[0]["config"]
                for key, value in sorted(sample_config.items()):
                    logger.info(f"      {key}: {value}")
        
        # Return success
        return 0
    
    except Exception as e:
        logger.error(f"Error generating experiment configurations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())