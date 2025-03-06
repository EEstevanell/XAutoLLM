import csv
from typing import Tuple, List, Optional

from autogoal.datasets import download, datapath


def load(
    *args, encoding: Optional[str] = "ordinal", **kwargs
) -> Tuple[List[str], List, List[str], List]:
    try:
        download("meld")
    except Exception as e:
        print(
            "Error loading data. This may be caused due to bad connection. "
            "Please delete badly downloaded data and retry."
        )
        raise e

    path = datapath("meld/meld")

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Helper function to read CSV files
    def read_csv(file_path: str) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        with open(file_path, "r", encoding="utf-8") as fd:
            reader = csv.reader(fd)
            header = next(reader, None)  # Skip header
            for row in reader:
                if len(row) < 3:
                    continue  # Skip malformed rows
                texts.append(row[1])
                labels.append(row[3])
        return texts, labels

    # Read training and testing data
    X_train, y_train = read_csv(str(path / "train.csv"))
    X_test, y_test = read_csv(str(path / "test.csv"))

    # Encode labels if encoding is specified
    if encoding:
        if encoding.lower() == "ordinal":
            try:
                from sklearn.preprocessing import LabelEncoder
            except ImportError as e:
                print("Please install scikit-learn to use ordinal encoding.")
                raise e

            label_encoder = LabelEncoder()
            all_labels = y_train + y_test
            label_encoder.fit(all_labels)
            y_train = label_encoder.transform(y_train).tolist()
            y_test = label_encoder.transform(y_test).tolist()

            print("Encoded Classes and Mappings:")
            for class_label, class_index in zip(
                label_encoder.classes_, label_encoder.transform(label_encoder.classes_)
            ):
                print(f"{class_label}: {class_index}")
        else:
            raise ValueError(f"Unsupported encoding type: {encoding}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Example usage with ordinal encoding
    X_train, y_train, X_test, y_test = load(encoding="ordinal")
    print("Training samples:", len(X_train))
    print("Training labels:", y_train[:5])
    print("Testing samples:", len(X_test))
    print("Testing labels:", y_test[:5])
