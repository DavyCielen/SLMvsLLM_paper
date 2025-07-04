import os
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict

def main():
    parser = argparse.ArgumentParser(description="Calculate classification metrics from run results.")
    parser.add_argument(
        "--run_id", 
        type=str, 
        required=True, 
        help="The ID of the run in 'subsetted_runs' to process."
    )
    args = parser.parse_args()

    run_id = args.run_id
    source_dir = os.path.join("subsetted_runs", run_id)
    results_dir = "results"

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{run_id}_metrics.csv")
    
    all_metrics: List[Dict] = []

    print(f"Processing files in: {source_dir}")

    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(source_dir, filename)
            print(f"  - Calculating metrics for: {filename}")

            try:
                df = pd.read_csv(file_path)

                # Ensure required columns exist
                if 'expected_prediction' not in df.columns or 'ensemble_prediction' not in df.columns:
                    print(f"    - Skipping {filename}: Missing 'expected_prediction' or 'ensemble_prediction' column.")
                    continue
                
                # Drop rows where labels are missing to avoid errors
                df.dropna(subset=['expected_prediction', 'ensemble_prediction'], inplace=True)

                if df.empty:
                    print(f"    - Skipping {filename}: No valid rows to process after cleaning.")
                    continue

                y_true = df['expected_prediction']
                y_pred = df['ensemble_prediction']

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                metrics = {
                    'run_id': run_id,
                    'source_file': filename,
                    'accuracy': accuracy,
                    'precision_weighted': precision,
                    'recall_weighted': recall,
                    'f1_score_weighted': f1
                }
                all_metrics.append(metrics)
                print(f"    - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            except Exception as e:
                print(f"    - Error processing {filename}: {e}")

    if not all_metrics:
        print("No metrics were calculated. No output file will be created.")
        return

    # Save all collected metrics to a single CSV file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_file, index=False)
    print(f"\nMetrics successfully saved to: {output_file}")

if __name__ == "__main__":
    main()
