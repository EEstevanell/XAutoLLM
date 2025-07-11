import numpy as np
import pandas as pd

# Import the DataLoader from your current project structure.
from text_classification.src.analysis.multi_objective._summary import (
    DataLoader,
    summarize_all_experiments,
    export_summary_to_json,
)

def main():
    # Initialize the DataLoader with the multi-objective candidates configuration.
    loader = DataLoader(
        "autogoal/experiments/text_generation/configs/multi-objective/candidates.yaml",
        "/home/coder/autogoal/experiments/text_generation/data/experience_store",
    )

    # Run the analysis for all datasets.
    for dataset in ["drop", "squad"]:
        data = loader.load_all_data_for_dataset(dataset)
        print("-" * 50)
        print(f"Dataset: {dataset}")

        summary_df = summarize_all_experiments(data)
        print(summary_df)

        export_summary_to_json(
            summary_df,
            f"autogoal/experiments/output/text_generation/multi-objective-analysis/summary-{dataset}.json",
        )

        print("-" * 50)
        print()


if __name__ == "__main__":
    main()
