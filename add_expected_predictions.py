import os
import pandas as pd
from sqlalchemy import create_engine, exc as sqlalchemy_exc
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import argparse

# --- Database Functions ---

def get_db_params_from_env() -> Dict[str, str]:
    """
    Retrieves PostgreSQL database connection parameters from environment variables.
    """
    load_dotenv()  # Load .env file if present

    required_vars_map = {
        'user': 'DB_USER',
        'password': 'DB_PASSWORD',
        'host': 'DB_HOST',
        'port': 'DB_PORT',
        'dbname': 'DB_NAME'
    }
    
    db_params: Dict[str, str] = {}
    missing_vars: List[str] = []
    
    for key, env_var_name in required_vars_map.items():
        value = os.getenv(env_var_name)
        if value is None:
            missing_vars.append(env_var_name)
        else:
            db_params[key] = value
            
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    return db_params

def fetch_data_from_db(query: str, params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    """
    Connects to the PostgreSQL database and executes a query.
    """
    try:
        db_params = get_db_params_from_env()
        engine = create_engine(
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )
        
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection, params=params)
        
        return df
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None
    except sqlalchemy_exc.SQLAlchemyError as e:
        print(f"Database connection or query error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Add expected predictions to ensemble result files.")
    parser.add_argument("--run_id", type=str, required=True, help="The ID of the run to process.")
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run_id)
    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found at '{run_dir}'")
        return

    output_dir = os.path.join("runs_with_expected_predictions", args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing files in: {run_dir}")

    # Fetch all expected predictions once to be efficient
    print("Fetching all expected predictions from the database...")
    expected_predictions_df = fetch_data_from_db("SELECT row_id, dataset_id, expected_prediction FROM rows")
    if expected_predictions_df is None:
        print("Could not fetch expected predictions. Aborting.")
        return
    print(f"Successfully fetched {len(expected_predictions_df)} expected predictions.")

    for filename in os.listdir(run_dir):
        if filename.startswith("ensemble_predictions_") and filename.endswith(".csv"):
            file_path = os.path.join(run_dir, filename)
            print(f"\nProcessing file: {filename}")

            try:
                ensemble_df = pd.read_csv(file_path)
                if 'row_id' not in ensemble_df.columns:
                    print(f"  - Skipping {filename}: 'row_id' column not found.")
                    continue

                # Merge with expected predictions
                merged_df = pd.merge(ensemble_df, expected_predictions_df, on='row_id', how='left')

                output_path = os.path.join(output_dir, filename)
                merged_df.to_csv(output_path, index=False)
                print(f"  - Saved updated file to: {output_path}")

            except Exception as e:
                print(f"  - Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
