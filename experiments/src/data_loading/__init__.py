from autogoal.meta_learning import (
    StandardScalerNormalizer,
    EuclideanDistance,
    CosineDistance,
)
from src.data_loading.data_loader import DataLoader
import re

class WarmstartConfigParser:
    @staticmethod
    def parse(config_str: str, dataset_name: str = None) -> dict:
        """
        Parse a warmstart candidate configuration string and return a
        configuration dictionary.

        Rules:
          - If "no-pos" is found then k_pos = 0; if "f-pos" or "a-pos" is found, k_pos = 10000.
          - Similarly, if there is any token with "neg", k_neg = 10000.
          - If "a-pos" is present, then adaptative_positive_alpha_limit = 1.
          - If "a-neg" is present, then adaptative_negative_alpha_limit = -1.
          - If token "k=<value>" is present, then beta_scale is set to that value; otherwise beta_scale = 0.
          - If "cos" is present, distance is CosineDistance; if "euc" is present, then EuclideanDistance.
          - Utility function is extracted from the second square-bracket token (defaults to "weighted_sum").
          - Fixed keys are added and an exclusion pattern is always present.
        """
        s = config_str.strip()
        # Extract tokens inside the first parenthesis, e.g., from "..._(f-pos cos k=0.5)"
        paren_match = re.search(r"\(([^)]+)\)", s)
        tokens_str = paren_match.group(1) if paren_match else ""
        tokens = []
        for token in tokens_str.split():
            tokens.extend(token.split("+"))
        tokens = [token.strip() for token in tokens if token.strip() != ""]

        # Positive experience
        has_positive = any(token in ["f-pos", "a-pos"] for token in tokens)
        k_pos = 10000 if has_positive else 0
        adaptative_positive_alpha_limit = 1 if "a-pos" in tokens else None

        # Negative experience
        has_negative = any("neg" in token for token in tokens)
        k_neg = 10000 if has_negative else 0
        adaptative_negative_alpha_limit = -1 if "a-neg" in tokens else None

        # Beta scale (distance factor)
        beta_scale = 0
        for token in tokens:
            if token.startswith("k="):
                try:
                    beta_scale = float(token.split("=")[1])
                except ValueError:
                    beta_scale = 0

        # Distance:
        if any("cos" in token for token in tokens):
            distance = CosineDistance
        elif any("euc" in token for token in tokens):
            distance = EuclideanDistance
        else:
            distance = None  # or set a default

        # Extract utility function from square brackets (ignoring the first one)
        square_matches = re.findall(r"\[([^\]]+)\]", s)
        utility_function = (
            square_matches[1].strip() if len(square_matches) >= 2 else "weighted_sum"
        )

        # Build final warmstart configuration dictionary
        config_dict = {
            "k_pos": k_pos,
            "k_neg": k_neg,
            "distance": distance,
            "normalizers": [StandardScalerNormalizer()],
            "positive_min_threshold": 0,
            "adaptative_positive_alpha_limit": adaptative_positive_alpha_limit,
            "adaptative_negative_alpha_limit": adaptative_negative_alpha_limit,
            "max_alpha": 0.05,
            "min_alpha": -0.02,
            "beta_scale": beta_scale,
            "f1_weight": 0.5,
            "evaluation_time_weight": 0.5,
            "exclude": f"{(dataset_name + '|') if dataset_name is not None else ''}warmstart",
            "utility_function": utility_function,
        }
        return config_dict

def main():
    # Use the multi-objective candidate configuration file.
    config_path = "/home/coder/autogoal/experiments/configs/multi-objective/candidates.yaml"
    data_root = "/home/coder/autogoal/experiments/data/experience_store"

    # Initialize DataLoader
    loader = DataLoader(config_path, data_root)

    # Specify the dataset and configuration keys, e.g., for 'liar' dataset, "low" complexity, "liar" method.
    dataset = "liar"
    bias = "low"
    method = "liar"

    # Retrieve candidate configuration string from the YAML file.
    candidate_str = loader.config.get(dataset, {}).get(bias, {}).get(method)
    if candidate_str is None:
        print("Candidate configuration not found in YAML.")
        return
    print("Candidate configuration string:")
    print(candidate_str)

    # Parse the candidate configuration string to build the warmstart configuration dictionary.
    warmstart_config = WarmstartConfigParser.parse(candidate_str)
    print("\nParsed Warmstart Configuration:")
    for key, val in warmstart_config.items():
        print(f"{key}: {val}")

    # Load experience data using the alias from the candidate configuration string.
    # Here, we assume the candidate string also corresponds to a directory under data/experience_store.
    df = loader.load_data_for_alias(candidate_str)
    if df.empty:
        print("\nNo data loaded for this configuration alias.")
    else:
        print("\nLoaded Data (head):")
        print(df.head())

if __name__ == "__main__":
    main()
