import pandas as pd
from datasets import load_dataset

def download_and_save_cnn_dailymail_as_csv():
    """
    Downloads the abisee/cnn_dailymail dataset (version 3.0.0) from Hugging Face.
    It concatenates the 'train' and 'validation' splits and saves them to 'train.csv'.
    The 'test' split is saved to 'test.csv'.
    The saved CSVs will contain all columns from the dataset, including 'article' and 'highlights'.
    """
    try:
        # 1. Load the CNN Dailymail dataset from Hugging Face
        dataset_name = "abisee/cnn_dailymail"
        # Version 3.0.0 is specified as it's a common version for this dataset [4, 5].
        # This dataset contains 'article' and 'highlights' columns and has 'train', 'validation', and 'test' splits [3, 4, 5].
        dataset_version = "3.0.0" 
        print(f"Loading {dataset_name} dataset (version {dataset_version})...")
        cnn_dailymail_dataset = load_dataset(dataset_name, dataset_version)
        print("Dataset loaded successfully.")

        # 2. Process and save train and validation splits
        print("Processing 'train' and 'validation' splits...")
        train_df = cnn_dailymail_dataset['train'].to_pandas()
        validation_df = cnn_dailymail_dataset['validation'].to_pandas()

        # Concatenate train and validation DataFrames
        combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)
        
        # The 'abisee/cnn_dailymail' dataset directly provides 'article' and 'highlights' columns [4, 5].
        # The SQuAD-specific processing for an 'answers' column is not needed and has been removed.
        
        output_train_csv = 'train.csv'
        print(f"Saving combined 'train' and 'validation' splits to '{output_train_csv}'...")
        # The DataFrame will contain columns like 'article', 'highlights', and 'id' [4, 5].
        combined_train_df.to_csv(output_train_csv, index=False, encoding='utf-8')
        print(f"Saved {len(combined_train_df)} rows to '{output_train_csv}'.")

        # 3. Process and save test split
        print("Processing 'test' split...")
        test_df = cnn_dailymail_dataset['test'].to_pandas()
        
        output_test_csv = 'test.csv'
        print(f"Saving 'test' split to '{output_test_csv}'...")
        test_df.to_csv(output_test_csv, index=False, encoding='utf-8')
        print(f"Saved {len(test_df)} rows to '{output_test_csv}'.")

        print("All specified splits processed and saved.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'datasets' and 'pandas' libraries installed.")
        print("You can install them using: pip install datasets pandas")

if __name__ == "__main__":
    # To run the script, make sure you have the required libraries installed:
    # pip install datasets pandas
    download_and_save_cnn_dailymail_as_csv()
