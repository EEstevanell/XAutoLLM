import pandas as pd
from datasets import load_dataset

def download_and_save_squad_as_csv(output_filename="squad_dataset.csv"):
    """
    Downloads the SQuAD dataset ('rajpurkar/squad') from Hugging Face
    and saves its train and validation splits into a single CSV file.

    Args:
        output_filename (str): The name of the CSV file to save the dataset to.
                               Defaults to "squad_dataset.csv".
    """
    try:
        # 1. Load the SQuAD dataset from Hugging Face
        print("Loading SQuAD dataset (rajpurkar/squad)...")
        squad_dataset = load_dataset("rajpurkar/squad")
        print("Dataset loaded successfully.")

        # 2. For each split, process and save to CSV

        for split_name, split_data in squad_dataset.items():
            print(f"Processing split: {split_name}")
            df = split_data.to_pandas()

            # Extraction logic that works for both lists and numpy arrays
            import numpy as np
            def get_first_answer_text(answers):
                if isinstance(answers, dict) and 'text' in answers:
                    texts = answers['text']
                    if isinstance(texts, (list, np.ndarray)) and len(texts) > 0:
                        return texts[0]
                return None

            def get_first_answer_start(answers):
                if isinstance(answers, dict) and 'answer_start' in answers:
                    starts = answers['answer_start']
                    if isinstance(starts, (list, np.ndarray)) and len(starts) > 0:
                        return starts[0]
                return None

            print("Processing the 'answers' column for CSV compatibility...")
            df['answer_text'] = df['answers'].apply(get_first_answer_text)
            df['answer_start'] = df['answers'].apply(get_first_answer_start)
            df = df.drop(columns=['answers'])

            # Save to appropriate CSV
            if split_name == 'train':
                out_csv = 'train.csv'
            elif split_name == 'validation':
                out_csv = 'test.csv'
            else:
                out_csv = f"{split_name}.csv"

            print(f"Saving {split_name} split to '{out_csv}'...")
            df.to_csv(out_csv, index=False, encoding='utf-8')
            print(f"Saved {len(df)} rows to '{out_csv}'.")

        print("All splits processed and saved.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'datasets' and 'pandas' libraries installed.")
        print("You can install them using: pip install datasets pandas")

if __name__ == "__main__":
    # To run the script, make sure you have the required libraries installed:
    # pip install datasets pandas
    download_and_save_squad_as_csv()
