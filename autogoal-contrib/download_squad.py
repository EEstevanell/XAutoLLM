import pandas as pd
from datasets import load_dataset
import os # Import os for creating directory if needed

def download_and_save_squad_splits_as_csv(output_dir="squad_csv_splits"):
    """
    Downloads the SQuAD dataset ('rajpurkar/squad') from Hugging Face,
    processes the 'answers' column for CSV compatibility by taking the first answer,
    and saves the 'train' and 'validation' splits into separate CSV files
    in the specified output directory.

    Args:
        output_dir (str): The directory where the CSV files will be saved.
                          Defaults to "squad_csv_splits".
    """
    try:
        # 1. Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # 2. Load the SQuAD dataset from Hugging Face
        print("Loading SQuAD dataset (rajpurkar/squad)...")
        squad_dataset = load_dataset("rajpurkar/squad")
        print("Dataset loaded successfully.")
        print(f"Available splits: {list(squad_dataset.keys())}") # Should show ['train', 'validation']

        # 3. Process and save each split separately
        for split_name in squad_dataset.keys():
            print(f"\nProcessing '{split_name}' split...")
            split_data = squad_dataset[split_name]

            # Convert the Hugging Face Dataset split to a Pandas DataFrame [1, 2]
            df = split_data.to_pandas()
            print(f"Converted '{split_name}' split to DataFrame with {len(df)} rows.")

            # --- Process the 'answers' column robustly ---
            # Extract the *first* answer text if available
            df['answer_text'] = df['answers'].apply(
                lambda x: x['text'][0] if isinstance(x, dict) and 'text' in x and len(x['text']) > 0 else None
            )
            # Extract the *first* answer start position if available
            df['answer_start'] = df['answers'].apply(
                lambda x: x['answer_start'][0] if isinstance(x, dict) and 'answer_start' in x and len(x['answer_start']) > 0 else None
            )
            # Remove the original complex 'answers' column
            df = df.drop(columns=['answers'])
            print(f"Processed 'answers' column for '{split_name}' split.")

            # --- Save the split DataFrame to a CSV file ---
            output_filename = os.path.join(output_dir, f"squad_{split_name}.csv")
            print(f"Saving '{split_name}' split to '{output_filename}'...")
            # index=False prevents pandas from writing the DataFrame index as a column
            df.to_csv(output_filename, index=False, encoding='utf-8')
            print(f"'{split_name}' split successfully saved to '{output_filename}'.")

            # Display structure and head of the saved CSV for verification
            print(f"\nStructure of '{split_name}' DataFrame:")
            df.info(memory_usage='deep') # Show memory usage for potentially large strings
            print(f"\nFirst 3 rows of '{split_name}.csv':")
            print(df.head(3))

        print("\nAll splits processed and saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'datasets' and 'pandas' libraries installed.")
        print("You can install them using: pip install datasets pandas")

if __name__ == "__main__":
    # To run the script, make sure you have the required libraries installed:
    # pip install datasets pandas
    download_and_save_squad_splits_as_csv()
