import os
import json

EXPERIENCE_DIR = "/home/coder/autogoal/experiments/text_classification/data/experience_store"
METRIC_FIELDS = ["accuracy", "f1", "evaluation_time"]

def migrate_experience_file(filepath):
    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            return

    # Only migrate if any of the old metric fields are present and 'metrics' is not already present
    if any(field in data for field in METRIC_FIELDS) and "metrics" not in data:
        metrics = []
        for field in METRIC_FIELDS:
            if field in data:
                value = data[field]
                # Decide maximize based on metric name
                maximize = field != "evaluation_time"
                metrics.append({
                    "name": field,
                    "maximize": maximize,
                    "value": value
                })
                del data[field]
        data["metrics"] = metrics
        print(f"Migrating {filepath}...")

        # Write back the migrated data
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
    else:
        print(f"Skipping {filepath} (already migrated or not an experience file)")

def migrate_all_experiences(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                filepath = os.path.join(dirpath, filename)
                migrate_experience_file(filepath)

if __name__ == "__main__":
    migrate_all_experiences(EXPERIENCE_DIR)
    print("Migration complete.")