import os
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.preprocessing import MinMaxScaler

# Import the DataLoader from your current project structure.
from src.data_loading import DataLoader

def summarize_dataframe(alias_name: str, df: pd.DataFrame) -> dict:
    """
    Summarizes metrics for a single configuration DataFrame.
    """
    df = df.copy()
    df["error"] = False
    for col in ["macro_f1", "accuracy"]:
        if col in df.columns:
            df["error"] |= df[col].isnull() | df[col].isin([np.inf, -np.inf])
    
    valid_df = df[~df["error"]].copy() if "macro_f1" in df.columns else pd.DataFrame()

    if not valid_df.empty and "timestamp" in valid_df.columns:
        valid_df["timestamp"] = pd.to_datetime(valid_df["timestamp"], errors="coerce")

    max_f1 = valid_df["macro_f1"].max() if not valid_df.empty else None
    mean_f1 = valid_df["macro_f1"].mean() if not valid_df.empty else None
    min_eval_time = valid_df["evaluation_time"].min() if "evaluation_time" in valid_df.columns else None
    mean_eval_time = valid_df["evaluation_time"].mean() if "evaluation_time" in valid_df.columns else None

    return {
        "alias_name": alias_name,
        "max_f1": max_f1,
        "mean_f1": mean_f1,
        "min_evaluation_time": min_eval_time,
        "mean_evaluation_time": mean_eval_time,
        "num_evaluations": len(df),
        "num_errors": df["error"].sum(),
        "error_rate": df["error"].mean(),
    }

def summarize_all_experiments(data_dict: dict) -> pd.DataFrame:
    """
    Iterates through all configurations loaded from the DataLoader.
    """
    rows = []
    
    for key, content in data_dict.items():
        if isinstance(content, pd.DataFrame):
            # Baseline configuration case
            summary = summarize_dataframe(key, content)
            rows.append(summary)
        elif isinstance(content, dict):
            for subkey, df in content.items():
                alias = f"{key} - {subkey}"
                summary = summarize_dataframe(alias, df)
                rows.append(summary)
    
    return pd.DataFrame(rows)

def export_summary_to_json(summary_df: pd.DataFrame, output_path: str) -> None:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Export the DataFrame to JSON with indent for readability
    summary_df.to_json(output_path, orient='records', indent=2)
    print(f"Summary JSON saved to '{output_path}'")

def main():
    # Initialize the DataLoader with the multi-objective candidates configuration.
    loader = DataLoader('autogoal/experiments/configs/multi-objective/candidates.yaml', '/home/coder/autogoal/experiments/data/experience_store')
    
    # Run the analysis for all datasets.
    for dataset in ['liar', 'sst2', 'meld', 'ag_news']:
        data = loader.load_all_data_for_dataset(dataset)
        print('-' * 50)
        print(f"Dataset: {dataset}")
        
        summary_df = summarize_all_experiments(data)
        print(summary_df)
        
        export_summary_to_json(summary_df, f"autogoal/experiments/output/multi-objective-analysis/summary-{dataset}.json")
        
        print('-' * 50)
        print()

if __name__ == "__main__":
    main()
