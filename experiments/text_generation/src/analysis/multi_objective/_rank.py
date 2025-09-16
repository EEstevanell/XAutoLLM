#!/usr/bin/env python

# Import the DataLoader and WarmstartConfigParser modules from the suggested project structure.
from text_classification.src.analysis.multi_objective._rank import (
    clean_data,
    DataLoader,
    AutoMLMetrics,
    AutoMLComparator,
)


def main():
    # Initialize the DataLoader with the multi-objective candidates configuration.
    loader = DataLoader(
        "autogoal/experiments/text_generation/configs/multi-objective/candidates.yaml",
        "/home/coder/autogoal/experiments/text_generation/data/experience_store",
    )

    # Run the analysis for all datasets.
    for dataset in ["drop", "squad"]:
        dataset_dict = loader.load_all_data_for_dataset(dataset)
        print("-" * 50)
        print(f"Dataset: {dataset}")

        combined_data_dict = {}
        for bias_level, candidate_dict in dataset_dict.items():
            for candidate, candidate_data in candidate_dict.items():
                combined_key = f"{bias_level} - {candidate}"
                combined_data_dict[combined_key] = candidate_data

        clean_data(combined_data_dict)
        metrics_calculator = AutoMLMetrics()
        comparator = AutoMLComparator(
            data_dict=combined_data_dict,
            metrics_calculator=metrics_calculator,
            ranking_method="all",
        )
        results = comparator.run_full_pipeline()

        for method_name, rank_map in results.items():
            print(f"\n{method_name.title()} Ranking:")
            for algo, rank_val in sorted(rank_map.items(), key=lambda x: x[1]):
                print(f"{rank_val}. {algo}")
        print("-" * 50)
        print()


if __name__ == "__main__":
    main()
