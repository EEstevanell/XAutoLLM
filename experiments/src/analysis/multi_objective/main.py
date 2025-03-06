#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd

# Import the necessary modules from your project structure.
# Adjust the module paths as needed.
from src.data_loading import DataLoader
from src.analysis.multi_objective._rank import AutoMLMetrics, AutoMLComparator, clean_data
from src.analysis.multi_objective._summary import summarize_all_experiments

def main():
    # List of datasets to analyze
    datasets = ['liar', 'sst2', 'meld', 'ag_news']
    loader = DataLoader('autogoal/experiments/configs/multi-objective/candidates.yaml', '/home/coder/autogoal/experiments/data/experience_store')
    overall_output = {}
    for dataset in datasets:
        # Load the dataset configurations and data
        dataset_dict = loader.load_all_data_for_dataset(dataset)
        
        # Generate summary data (one row per candidate) using the summary utility
        summary_df = summarize_all_experiments(dataset_dict)
        summary_data = summary_df.to_dict(orient='records')
        
        # Prepare combined candidate data for ranking analysis.
        # For the case of "baseline" the value is a DataFrame, and for others it is a dict.
        combined_data_dict = {}
        for key, value in dataset_dict.items():
            if key == 'baseline':
                combined_data_dict[key] = value
            else:
                for candidate, candidate_df in value.items():
                    combined_key = f"{key} - {candidate}"
                    combined_data_dict[combined_key] = candidate_df
        
        # Clean the data to remove any errors or invalid metrics
        clean_data(combined_data_dict)
        
        # Initialize metrics calculator and run the ranking analysis.
        metrics_calculator = AutoMLMetrics()
        comparator = AutoMLComparator(
            data_dict=combined_data_dict,
            metrics_calculator=metrics_calculator,
            ranking_method='all'
        )
        ranking_results = comparator.run_full_pipeline()
        
        # The AutoMLComparator stores results in all_metric_vectors, where
        # each AlgorithmRunMetrics has 'metrics' with hypervolume as the first element.
        hypervolume_results = {}
        for run in comparator.all_metric_vectors:
            # Extract the hypervolume (first element)
            hypervolume_results[run.algo_name] = run.metrics[0]
        
        # Save the collated results for the current dataset.
        overall_output[dataset] = {
            "summary": summary_data,
            "ranking": ranking_results,
            "hypervolume": hypervolume_results
        }
    
    # Create the output directory if it does not exist.
    output_path = "/home/coder/autogoal/experiments/output/multi-objective-analysis/overall_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the overall results into a single JSON file.
    with open(output_path, 'w') as f:
        json.dump(overall_output, f, indent=2)
    
    print(f"Analysis result saved to {output_path}")

if __name__ == "__main__":
    main()
