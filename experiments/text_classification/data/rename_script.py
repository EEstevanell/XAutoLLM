import os
import re
import sys

def process_directory_name(name):
    """
    Processes a directory name by removing any occurrence of '_id_9' and,
    if present, replacing a trailing pattern of the form:
       -(rd|rn|rng|rnd)_<number> (<token>_<number>)
    with:
       (seed_<number>)
    provided that the two numbers match.
    """
    # Remove any occurrence of '_id_9'
    modified_name = name.replace("_id_9", "")
    modified_name = modified_name.replace("_id_", " ")
    
    # Regular expression to detect the trailing pattern
    pattern = re.compile(r'-(rd|rn|rng|rnd)_([0-9]+)\s+\((rd|rn|rng|rnd)_([0-9]+)\)$')
    match = pattern.search(modified_name)
    
    if match:
        number1 = match.group(2)
        number2 = match.group(4)
        if number1 == number2:
            # Replace the matched pattern with (seed_number)
            modified_name = modified_name[:match.start()] + f" (seed_{number1})"
        else:
            print(f"Note: Pattern numbers differ in '{name}'. Pattern left unchanged after removing '_id_9'.")
    return modified_name

def rename_directories(path):
    """
    Iterates over the directories in the given path, applies the name modifications,
    and renames them if the new name differs from the original.
    """
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            new_name = process_directory_name(item)
            if new_name != item:
                new_full_path = os.path.join(path, new_name)
                print(f"Renaming '{full_path}' to '{new_full_path}'")
                os.rename(full_path, new_full_path)
            else:
                print(f"No change for '{item}'.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python rename_dirs.py <path>")
        sys.exit(1)
    directory = sys.argv[1]
    if not os.path.exists(directory):
        print("Error: Provided path does not exist.")
        sys.exit(1)
    rename_directories(directory)