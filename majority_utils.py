import os
import pandas as pd
from sqlalchemy import create_engine, exc as sqlalchemy_exc
from dotenv import load_dotenv
import os
import datetime # For loading .env file for local development
from collections import Counter
from typing import List, Dict, Any, Optional
import argparse

def majority_vote(predictions):
    """
    Determines the majority vote from a list of predictions.

    Args:
        predictions: A list of prediction values.

    Returns:
        The most common prediction value.
    """
    if not predictions:
        return None  # Or raise an error, depending on desired behavior for empty input
    return Counter(predictions).most_common(1)[0][0]

# --- Database Functions ---

def get_db_params_from_env() -> Dict[str, str]:
    """
    Retrieves PostgreSQL database connection parameters from environment variables.

    Environment variables expected:
        POSTGRES_USER: Username for the PostgreSQL database.
        POSTGRES_PASSWORD: Password for the PostgreSQL database.
        POSTGRES_HOST: Host of the PostgreSQL database (e.g., localhost).
        POSTGRES_PORT: Port for the PostgreSQL database (e.g., 5432).
        POSTGRES_DBNAME: Name of the PostgreSQL database.

    Returns:
        A dictionary containing the database parameters.
    
    Raises:
        ValueError: If any of the required environment variables are not set.
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
    Connects to the PostgreSQL database using environment variables,
    executes the given query with parameters, and returns the results as a pandas DataFrame.

    Args:
        query: The SQL query string to execute (with placeholders like %(key)s).
        params: A dictionary of parameters to bind to the query.

    Returns:
        A pandas DataFrame containing the query results.
        Returns None if configuration, connection or query fails.
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


# --- Analysis Helper Functions ---

def calculate_ensemble_prediction(
    df: pd.DataFrame, 
    group_by_cols: List[str], 
    prediction_col: str = 'prediction', 
    ensemble_col_name: str = 'ensemble_prediction'
) -> Optional[pd.DataFrame]:
    """
    Calculates ensemble predictions by applying majority vote after grouping.

    Args:
        df: Pandas DataFrame containing the prediction data.
        group_by_cols: A list of column names to group by.
        prediction_col: The name of the column containing individual predictions.
        ensemble_col_name: The name for the new column containing ensemble predictions.

    Returns:
        A new pandas DataFrame with the group_by columns and the ensemble predictions column,
        or None if input is invalid (e.g., missing columns).
    """
    if df is None or df.empty:
        print("Input DataFrame is None or empty. Cannot calculate ensemble predictions.")
        return None

    missing_group_cols = [col for col in group_by_cols if col not in df.columns]
    if missing_group_cols:
        print(f"Error: The following group_by_cols are not in DataFrame columns: {missing_group_cols}. Available columns: {df.columns.tolist()}")
        return None
    if prediction_col not in df.columns:
        print(f"Error: prediction_col ('{prediction_col}') not in DataFrame columns: {df.columns.tolist()}")
        return None

    print(f"\nCalculating ensemble predictions, grouping by {group_by_cols} on '{prediction_col}' column...")
    
    temp_list_col = '_prediction_list_for_voting' # Temporary column for lists of predictions
    
    try:
        ensembled_df = df.groupby(group_by_cols, as_index=False).agg(
            **{temp_list_col: pd.NamedAgg(column=prediction_col, aggfunc=list)}
        )
    except Exception as e:
        print(f"Error during groupby operation: {e}")
        return None

    if ensembled_df.empty:
        print("Warning: Grouping resulted in an empty DataFrame. No ensemble predictions to calculate.")
        # Return an empty DataFrame with expected columns if possible, or just the empty ensembled_df
        expected_cols = group_by_cols + [ensemble_col_name]
        if temp_list_col in ensembled_df.columns: # if agg produced the temp list col
             expected_cols.append(temp_list_col)
        return pd.DataFrame(columns=expected_cols)

    ensembled_df[ensemble_col_name] = ensembled_df[temp_list_col].apply(majority_vote)
    
    print(f"Calculated ensemble predictions in column '{ensemble_col_name}'.")
    # By default, the temp_list_col is kept for inspection. It can be dropped if desired:
    # ensembled_df = ensembled_df.drop(columns=[temp_list_col])
    return ensembled_df

# --- Example Usage ---
if __name__ == "__main__":
    print("Attempting to fetch data and demonstrate flexible ensemble calculations...")
    print("Please ensure your .env file is set up or environment variables are exported.")
    
    parser = argparse.ArgumentParser(description="Run ensemble prediction for a specific model, dataset, and prompt.")
    parser.add_argument("--model_id", type=int, default=None, help="Optional: Model ID to filter")
    parser.add_argument("--dataset_id", type=int, default=None, help="Optional: Dataset ID to filter")
    parser.add_argument("--prompt_id", type=int, default=None, help="Optional: Prompt ID to filter")
    parser.add_argument("--library", type=str, default=None, help="Optional: Model library to filter (e.g., 'ollama')")
    parser.add_argument("--run_id", type=str, default=None, help="Optional: A specific ID for the run. If not provided, a timestamp will be used.")
    parser.add_argument(
        "--group_by",
        type=str,
        default="",
        help="Additional columns to group by for ensemble prediction, comma-separated (e.g., 'model_id,prompt_id'). 'row_id' is always included."
    )
    args = parser.parse_args()

    # Determine run_id
    run_id = args.run_id if args.run_id else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Always include 'row_id' in grouping
    additional_group_cols = [col.strip() for col in args.group_by.split(",") if col.strip()]
    group_by_cols = ['row_id'] + additional_group_cols

    # Build the WHERE clause dynamically
    where_clauses = []
    params = {}
    
    if args.model_id is not None:
        where_clauses.append("pr.model_id = %(model_id)s")
        params['model_id'] = args.model_id

    if args.dataset_id is not None:
        where_clauses.append("pr.dataset_id = %(dataset_id)s")
        params['dataset_id'] = args.dataset_id
    
    if args.prompt_id is not None:
        where_clauses.append("pr.prompt_id = %(prompt_id)s")
        params['prompt_id'] = args.prompt_id

    if args.library is not None:
        where_clauses.append("m.library = %(library)s")
        params['library'] = args.library

    where_statement = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    example_query = f"""
    SELECT 
        pr.row_id AS row_id,
        pr.dataset_id AS dataset_id,
        pr.model_id AS model_id,
        pr.prompt_id AS prompt_id,
        pr.prediction AS prediction,
        r.expected_prediction AS expected_prediction
    FROM 
        predictions pr
    JOIN
        rows r ON pr.row_id = r.row_id
    JOIN
        models m ON pr.model_id = m.model_id
    {where_statement}
    """
    
    data_df = fetch_data_from_db(example_query, params=params)
    
    if data_df is not None and not data_df.empty:
        print(f"Successfully fetched {len(data_df)} rows. DataFrame columns: {data_df.columns.tolist()}")

        # --- Calculate Ensemble Prediction ---
        ensemble_df = calculate_ensemble_prediction(
            df=data_df, 
            group_by_cols=group_by_cols, 
            prediction_col='prediction',
            ensemble_col_name='ensemble_prediction'
        )
        if ensemble_df is not None:
            print("\nEnsemble by specified group_by columns (first 5 rows):")
            print(ensemble_df.head())

            # Save the final ensemble results to a CSV file
            output_filename_parts = []
            if args.model_id is not None:
                output_filename_parts.append(f"model{args.model_id}")
            if args.dataset_id is not None:
                output_filename_parts.append(f"dataset{args.dataset_id}")
            if args.prompt_id is not None:
                output_filename_parts.append(f"prompt{args.prompt_id}")
            if args.library is not None:
                output_filename_parts.append(f"library_{args.library.replace(' ', '_')}")

            if not output_filename_parts:
                output_filename_parts.append("all")
            
            # Create directories for the output
            run_dir = os.path.join("runs", run_id)
            os.makedirs(run_dir, exist_ok=True)

            output_filename = f"ensemble_predictions_{'_'.join(output_filename_parts)}.csv"
            output_path = os.path.join(run_dir, output_filename)
            ensemble_df.to_csv(output_path, index=False)
            print(f"\nEnsemble predictions saved to: {output_path}")
            
    elif data_df is not None and data_df.empty:
        print("Query executed successfully, but no data was returned. Check your query or database content.")
    else:
        print("Failed to fetch data. Please check environment variables, database connection, and query.")
