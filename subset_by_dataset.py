import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter run results by a list of dataset IDs.")
    parser.add_argument(
        "--dataset_ids", 
        type=str, 
        required=True, 
        help="A comma-separated list of dataset IDs to include (e.g., '1,2,3')."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="A specific run ID to process."
    )
    args = parser.parse_args()

    try:
        # Convert comma-separated string to a list of integers
        dataset_ids_to_keep = [int(id.strip()) for id in args.dataset_ids.split(',')]
    except ValueError:
        print("Error: Please provide a valid comma-separated list of integer dataset IDs.")
        return

    source_dir = "runs_with_expected_predictions"
    target_dir = "subsetted_runs"

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    print(f"Filtering for dataset IDs: {dataset_ids_to_keep}")

    run_id = args.run_id
    run_path = os.path.join(source_dir, run_id)

    if not os.path.isdir(run_path):
        print(f"Error: Run directory '{run_path}' not found.")
        return

    # Process the single specified run directory
    if os.path.isdir(run_path):


        print(f"\nProcessing run: {run_id}")
        output_run_path = os.path.join(target_dir, run_id)

        for filename in os.listdir(run_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(run_path, filename)
                print(f"  - Filtering file: {filename}")

                try:
                    df = pd.read_csv(file_path)
                    if 'dataset_id' not in df.columns:
                        print(f"    - Skipping {filename}: 'dataset_id' column not found.")
                        continue

                    # Filter the DataFrame
                    filtered_df = df[df['dataset_id'].isin(dataset_ids_to_keep)]

                    if filtered_df.empty:
                        print(f"    - No rows matched the dataset IDs in {filename}. Not creating an output file.")
                        continue

                    # Create target directory and save the filtered file
                    os.makedirs(output_run_path, exist_ok=True)
                    output_file_path = os.path.join(output_run_path, filename)
                    filtered_df.to_csv(output_file_path, index=False)
                    print(f"    - Saved {len(filtered_df)} rows to {output_file_path}")

                except Exception as e:
                    print(f"    - Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
