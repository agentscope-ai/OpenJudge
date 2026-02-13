import json
import random
import sys
from pathlib import Path


def get_data_files():
    """Get data files list from user input"""
    print("Please enter data file paths (one file per line, press Enter on empty line to finish):")
    data_files = []
    while True:
        file_path = input().strip()
        if not file_path:  # Empty line ends input
            break
        data_files.append(file_path)

    if not data_files:
        # Exit with error if no files are entered
        print("Error: No files provided. Please enter at least one file path.")
        sys.exit(1)

    return data_files


def get_split_params():
    """Get train/validation split ratio and random seed from user"""
    print("\nEnter train/validation split ratio (default 0.8):")
    try:
        split_ratio_input = input().strip()
        if split_ratio_input:
            split_ratio = float(split_ratio_input)
        else:
            split_ratio = 0.8
    except ValueError:
        print("Invalid input, using default split ratio 0.8")
        split_ratio = 0.8

    if split_ratio <= 0 or split_ratio >= 1:
        print("Split ratio must be between 0 and 1. Using default 0.8")
        split_ratio = 0.8

    print(f"Train/Validation split ratio: {split_ratio}/{1 - split_ratio}")

    print("\nEnter random seed (default 42):")
    try:
        seed_input = input().strip()
        if seed_input:
            seed = int(seed_input)
        else:
            seed = 42
    except ValueError:
        print("Invalid input, using default seed 42")
        seed = 42

    print(f"Random seed: {seed}")

    print("\nEnter number of samples to keep (optional, press Enter for all):")
    try:
        sample_num_input = input().strip()
        if sample_num_input:
            sample_num = int(sample_num_input)
        else:
            sample_num = None
    except ValueError:
        print("Invalid input, keeping all samples")
        sample_num = None

    return split_ratio, seed, sample_num


def validate_files(data_files):
    """Validate if files exist"""
    valid_files = []
    for file_path in data_files:
        if Path(file_path).exists():
            valid_files.append(file_path)
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ File does not exist: {file_path}")

    return valid_files


def process_single_file(data_file: str, split_ratio: float, seed: int, sample_num: int) -> bool:
    """Process a single data file with train/validation split"""
    print(f"Processing file: {data_file}")

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {data_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file {data_file} - {e}")
        return False
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue in file {data_file} - {e}")
        return False
    except Exception as e:
        print(f"Error reading file {data_file}: {e}")
        return False

    # Set random seed for reproducible results
    random.seed(seed)

    output_data = []
    try:
        for item in data:
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dict item in {data_file}")
                continue

            # Check if required keys exist
            if "input" not in item or "chosen" not in item or "rejected" not in item:
                print(f"Warning: Missing required keys in item from {data_file}. Skipping item.")
                continue

            # Extract task_type from item if available, otherwise use "unknown"
            task_type = item.get("task_type", "unknown")

            output_data.append(
                {
                    "input": item["input"],
                    "answer": item["chosen"],
                    "label": 1,  # positive example
                    "score": 5.0,
                    "task_type": task_type,
                }
            )
            output_data.append(
                {
                    "input": item["input"],
                    "answer": item["rejected"],
                    "label": 0,  # negative example
                    "score": 1.0,
                    "task_type": task_type,
                }
            )
    except KeyError as e:
        print(f"Error: Missing required key {e} in file {data_file}")
        return False
    except Exception as e:
        print(f"Error processing data in file {data_file}: {e}")
        return False

    # Randomly shuffle the data
    random.shuffle(output_data)

    # Apply random sampling if specified
    if sample_num is not None and len(output_data) > sample_num:
        output_data = output_data[:sample_num]
        print(f"Sampled {len(output_data)} items from {len(output_data)} total")

    # Split data into train and validation sets
    split_idx = int(len(output_data) * split_ratio)
    train_data = output_data[:split_idx]
    val_data = output_data[split_idx:]

    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    try:
        path = Path(data_file)
        base_name = path.stem

        # Write training data
        train_output_file = path.parent.joinpath(base_name + "_train.jsonl").as_posix()
        with open(train_output_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Training file generated: {train_output_file}")

        # Write validation data
        val_output_file = path.parent.joinpath(base_name + "_val.jsonl").as_posix()
        with open(val_output_file, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Validation file generated: {val_output_file}")

        return True
    except Exception as e:
        print(f"Error writing output files: {e}")
        return False


def main():
    """Main function"""
    print("# Preprocess grader data")
    print("# Data source: https://huggingface.co/datasets/agentscope-ai/OpenJudge/tree/main")
    print()

    # Get file list from user input
    data_files = get_data_files()

    # Get split parameters
    split_ratio, seed, sample_num = get_split_params()

    # Validate files
    valid_files = validate_files(data_files)

    if not valid_files:
        print("No valid files found, please check file paths.")
        sys.exit(1)

    # Process each valid file
    success_count = 0
    for data_file in valid_files:
        if process_single_file(data_file, split_ratio, seed, sample_num):
            success_count += 1
        # Add blank line separator
        print()

    print(f"Processing completed! Successfully processed {success_count}/{len(valid_files)} files.")


if __name__ == "__main__":
    main()
