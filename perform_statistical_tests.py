import os
import pandas as pd
import argparse
import re
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

def main():
    parser = argparse.ArgumentParser(description="Perform Friedman and Nemenyi tests on run results.")
    parser.add_argument(
        "--run_id", 
        type=str, 
        required=True, 
        help="The ID of the run in 'subsetted_runs' to process."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the statistical tests."
    )
    args = parser.parse_args()

    run_id = args.run_id
    alpha = args.alpha
    source_dir = os.path.join("subsetted_runs", run_id)
    results_dir = "results"

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Load and combine data --- 
    all_dfs = []
    model_names = []
    print(f"Loading data for run: {run_id}")
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            try:
                # Extract a clean model name from the filename
                match = re.search(r'ensemble_predictions_(.+)\.csv', filename)
                if match:
                    model_name = match.group(1)
                else:
                    model_name = os.path.splitext(filename)[0]
                model_names.append(model_name)

                df = pd.read_csv(os.path.join(source_dir, filename))
                # Keep only essential columns and rename prediction column
                df = df[['row_id', 'expected_prediction', 'ensemble_prediction']].rename(columns={'ensemble_prediction': model_name})
                df.set_index('row_id', inplace=True)
                all_dfs.append(df)
                print(f"  - Loaded {filename} as model '{model_name}'")
            except Exception as e:
                print(f"    - Could not process {filename}: {e}")
                continue
    
    if len(all_dfs) < 2:
        print("Error: Need at least two models to perform statistical tests.")
        return

    # Merge all dataframes on row_id
    merged_df = pd.concat(all_dfs, axis=1)
    # Remove duplicate 'expected_prediction' columns
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df.dropna(inplace=True)

    # --- 2. Prepare data for tests (0 for incorrect, 1 for correct) ---
    score_data = pd.DataFrame(index=merged_df.index)
    for model in model_names:
        score_data[model] = (merged_df[model] == merged_df['expected_prediction']).astype(int)

    # --- 3. Perform Friedman Test ---
    print("\n--- Friedman Test ---")
    stat, p_value = friedmanchisquare(*[score_data[col] for col in score_data.columns])
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Save Friedman test results
    friedman_results_path = os.path.join(results_dir, f"{run_id}_friedman_test.txt")
    with open(friedman_results_path, 'w') as f:
        f.write("Friedman Test Results\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Statistic: {stat}\n")
        f.write(f"P-value: {p_value}\n")
        if p_value < alpha:
            f.write(f"The result is significant at alpha = {alpha}.\n")
        else:
            f.write(f"The result is not significant at alpha = {alpha}.\n")
    print(f"Friedman test results saved to {friedman_results_path}")

    # --- 4. Perform Nemenyi post-hoc test if significant ---
    if p_value < alpha:
        print(f"\nFriedman test was significant (p < {alpha}). Performing Nemenyi post-hoc test...")
        print("--- Nemenyi Test (p-values) ---")
        nemenyi_results = sp.posthoc_nemenyi_friedman(score_data)
        print(nemenyi_results)

        # Save Nemenyi results to CSV
        nemenyi_results_path = os.path.join(results_dir, f"{run_id}_nemenyi_test.csv")
        nemenyi_results.to_csv(nemenyi_results_path)
        print(f"\nNemenyi test p-values saved to {nemenyi_results_path}")
    else:
        print(f"\nFriedman test was not significant (p >= {alpha}). No post-hoc test needed.")

if __name__ == "__main__":
    main()
