import csv
from typing import Tuple, List, Optional

from autogoal.datasets import download, datapath

def load(make_prompt: bool, *args, **kwargs):
    try:
        download("cnn_dailymail")
    except Exception as e:
        print(
            "Error loading data. This may be caused due to bad connection. "
            "Please delete badly downloaded data and retry."
        )
        raise e

    path = datapath("cnn_dailymail/cnn_dailymail")

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Helper function to read CSV files
    def read_csv(file_path: str) -> Tuple[List[str], List[str]]:
        X = []
        y = []
        with open(file_path, "r", encoding="utf-8") as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                # Ensure required fields exist and are not empty
                article = row.get("article", "")
                highlights = row.get("highlights", "")
                if article and highlights:
                    X.append(article)
                    y.append(highlights)
        return X, y

    # Read training and testing data
    X_train, y_train = read_csv(str(path / "train.csv"))
    X_test, y_test = read_csv(str(path / "test.csv"))

    if (make_prompt):
        # Convert to prompt format: article -> highlights
        X_train = [f"Article: {article}\nHighlights:" for article in X_train]
        X_test = [f"Article: {article}\nHighlights:" for article in X_test]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Example usage with ordinal encoding
    X_train, y_train, X_test, y_test = load()
    print("Training articles len):", len(X_train))
    print("Training articles 1st:", X_train[1])
    print("Training highlights 1st:", y_train[:51])
    print("Testing articles len:", len(X_test))
    print("Testing articles 1st:", X_test[1])
    print("Testing highlights 1st:", y_test[:1])
