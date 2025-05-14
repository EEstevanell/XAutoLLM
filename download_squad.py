import pandas as pd
import json
from datasets import load_dataset

def download_and_save_drop_as_csv():
    """
    Downloads the ucinlp/drop dataset from Hugging Face.
    Saves the train split as 'train.csv' and the validation split as 'test.csv'.
    The CSVs contain columns: 'passage', 'question', and 'answers_spans' (JSON string).
    """
    try:
        # 1. Load the DROP dataset
        dataset_name = "ucinlp/drop"
        print(f"Loading {dataset_name} dataset...")
        drop_dataset = load_dataset(dataset_name)
        print("Dataset loaded successfully.")

        # 2. Process train split
        print("Processing 'train' split...")
        train = drop_dataset['train']
        train_df = pd.DataFrame({
            'passage': train['passage'],
            'question': train['question'],
            'answers_spans': [json.dumps(ans) for ans in train['answers_spans']]
        })
        train_csv = 'train.csv'
        print(f"Saving train split to '{train_csv}'...")
        train_df.to_csv(train_csv, index=False, encoding='utf-8')
        print(f"Saved {len(train_df)} rows to '{train_csv}'.")

        # 3. Process validation split as test
        print("Processing 'validation' split...")
        val = drop_dataset['validation']
        test_df = pd.DataFrame({
            'passage': val['passage'],
            'question': val['question'],
            'answers_spans': [json.dumps(ans) for ans in val['answers_spans']]
        })
        test_csv = 'test.csv'
        print(f"Saving validation split as '{test_csv}'...")
        test_df.to_csv(test_csv, index=False, encoding='utf-8')
        print(f"Saved {len(test_df)} rows to '{test_csv}'.")

        print("All specified splits processed and saved.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'datasets' and 'pandas' libraries installed.")
        print("You can install them using: pip install datasets pandas")

if __name__ == "__main__":
    download_and_save_drop_as_csv()
