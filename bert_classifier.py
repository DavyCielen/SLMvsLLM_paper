from transformers import pipeline
import psycopg2
import time
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load environment variables
load_dotenv()

# Database connection parameters loaded strictly from the environment
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
}

# Batch size for processing
batch_size = 5

import re

def parse_sentiment(response_content):
    """
    Parse the sentiment from the response content.
    
    Parameters:
    response_content (str): The content of the response message.
    
    Returns:
    str: 'positive', 'negative', or 'neutral' based on the content.
    """
    # Convert the content to lowercase for case-insensitive matching
    content = response_content.lower()
    
    # Use regular expressions to find the first occurrence of 'positive', 'negative', or 'neutral'
    positive_match = re.search(r'\bpositive\b', content)
    negative_match = re.search(r'\bnegative\b', content)
    neutral_match = re.search(r'\bneutral\b', content)
    
    # Return the first match in the order of positive, negative, neutral
    if positive_match:
        return 'positive'
    elif negative_match:
        return 'negative'
    elif neutral_match:
        return 'neutral'
    else:
        # If no match is found, return 'unknown' or handle it as needed
        return 'unknown'


class Model():
    def __init__(self, model_name):
        self.model = pipeline("zero-shot-classification",model = "facebook/bart-large-mnli")

    def generate(self, prompt, labels):

        response =self.model(prompt, labels)
        out = response['labels'][0]  # Since the response is sorted by score in descending order
        logging.info(response)
        out = parse_sentiment(out)
        print(out)
        logging.info(out)
        return out
    

def get_least_used_model_prompt_dataset(exclude_prompt_ids=[]):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Get the least used model-prompt-dataset combination, excluding the specified prompt IDs
        exclude_clause = ""
        if exclude_prompt_ids:
            exclude_clause = f"AND mps.prompt_id NOT IN ({','.join(map(str, exclude_prompt_ids))})"

        library = 'bert'
        
        cursor.execute(f"""
            SELECT mps.model_id, mps.prompt_id, mps.dataset_id, m.name, p.text, d.name AS dataset_name, mps.count, m.library
            FROM ModelPromptStatus mps
            JOIN Models m ON mps.model_id = m.model_id
            JOIN Prompts p ON mps.prompt_id = p.prompt_id
            JOIN Datasets d ON mps.dataset_id = d.dataset_id
            WHERE (mps.status = 'available' OR mps.status = 'in_use') 
            AND m.library = '{library}' {exclude_clause}
            ORDER BY mps.count ASC
            LIMIT 1
        """)
        result = cursor.fetchone()
        
        if result:
            model_id, prompt_id, dataset_id, model_name, prompt_text, dataset_name, count, library = result

            # Use an advisory lock to ensure only one process updates the count
            cursor.execute("SELECT pg_advisory_lock(%s)", (model_id,))
            
            # Update the count for this model-prompt-dataset combination
            cursor.execute("""
                UPDATE ModelPromptStatus
                SET count = count + 1, status = 'in_use'
                WHERE model_id = %s AND prompt_id = %s AND dataset_id = %s
            """, (model_id, prompt_id, dataset_id))
            
            cursor.execute("SELECT pg_advisory_unlock(%s)", (model_id,))
            
            conn.commit()
            return model_id, prompt_id, dataset_id, model_name, prompt_text, dataset_name, count
        
        return None
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def fetch_batch(model_id, prompt_id, dataset_id):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT row_id, content
            FROM Rows
            WHERE dataset_id = %s AND row_id IN (
                SELECT row_id
                FROM PredictionStatus
                WHERE model_id = %s AND prompt_id = %s AND dataset_id = %s AND status = 'pending'
                LIMIT %s
            )
            FOR UPDATE SKIP LOCKED
        """, (dataset_id, model_id, prompt_id, dataset_id, batch_size))
        
        rows = cursor.fetchall()
        
        cursor.execute("""
            UPDATE PredictionStatus
            SET status = 'in_progress'
            WHERE row_id = ANY(%s) AND model_id = %s AND prompt_id = %s AND dataset_id = %s
        """, ([row[0] for row in rows], model_id, prompt_id, dataset_id))
        
        conn.commit()
        return rows
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def update_prediction(row_id, model_id, prompt_id, dataset_id, prediction, prediction_time, formatted_prompt):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO Predictions (row_id, model_id, prompt_id, dataset_id, prediction, prediction_time, status, formatted_prompt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (row_id, model_id, prompt_id, dataset_id, prediction.strip().lower(), prediction_time, 'done', formatted_prompt.strip().lower()))
        
        cursor.execute("""
            UPDATE PredictionStatus
            SET status = 'done'
            WHERE row_id = %s AND model_id = %s AND prompt_id = %s AND dataset_id = %s
        """, (row_id, model_id, prompt_id, dataset_id))
        
        conn.commit()
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def revert_batch_status(rows, model_id, prompt_id, dataset_id):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE PredictionStatus
            SET status = 'pending'
            WHERE row_id = ANY(%s) AND model_id = %s AND prompt_id = %s AND dataset_id = %s
        """, ([row[0] for row in rows], model_id, prompt_id, dataset_id))
        
        conn.commit()
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def decrement_count(model_id, prompt_id, dataset_id):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Use an advisory lock to ensure only one process updates the count
        cursor.execute("SELECT pg_advisory_lock(%s)", (model_id,))
        
        cursor.execute("""
            UPDATE ModelPromptStatus
            SET count = count - 1
            WHERE model_id = %s AND prompt_id = %s AND dataset_id = %s
        """, (model_id, prompt_id, dataset_id))
        
        cursor.execute("SELECT pg_advisory_unlock(%s)", (model_id,))
        
        conn.commit()
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def main():
    exclude_prompt_ids = []

    while True:
        model_info = get_least_used_model_prompt_dataset(exclude_prompt_ids)
        
        if model_info is None:
            print("No available model-prompt-dataset combination found.")
            return

        model_id, prompt_id, dataset_id, model_name, prompt_text, dataset_name, status = model_info

        # Check the library of the model before loading
        if status == 'stop':
            print(f"Model-prompt-dataset combination {model_name} - {prompt_text} - {dataset_name} is set to stop. Moving to the next combination.")
            exclude_prompt_ids.append(prompt_id)
            continue

        model = Model(model_name)
        
        print(f"Using model: {model_name} with prompt: {prompt_text} on dataset: {dataset_name}")

        while True:
            rows = fetch_batch(model_id, prompt_id, dataset_id)
            
            if not rows:
                # Decrement the count for this model-prompt-dataset combination and fetch new prompt
                decrement_count(model_id, prompt_id, dataset_id)
                model_info = get_least_used_model_prompt_dataset(exclude_prompt_ids)
                if model_info is None:
                    print("No available model-prompt-dataset combination found.")
                    return
                model_id, prompt_id, dataset_id, model_name, prompt_text, dataset_name, status = model_info
                print(f"Switching to new prompt: {prompt_text} on dataset: {dataset_name}")
                continue

            try:
                for row_id, content in rows:
                    start_time = time.time()
                    # formatted_prompt = prompt_text.format(content=content)
                    labels = ['positive', 'negative','neutral']
                    if dataset_id == 2:
                        labels = ['positive','negative']
                    output = model.generate(content, labels=labels)
                    prediction_time = time.time() - start_time
                    update_prediction(row_id, model_id, prompt_id, dataset_id, output, prediction_time, content)
                    
                    logging.info(f"Processed row_id: {row_id} with model: {model_name}")

            except Exception as e:
                print(f"Error occurred: {e}. Reverting batch status to 'pending'.")
                revert_batch_status(rows, model_id, prompt_id, dataset_id)
                break

            # Check if the status is 'stop' after each batch
            try:
                conn = psycopg2.connect(**db_params)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT status
                    FROM ModelPromptStatus
                    WHERE model_id = %s AND prompt_id = %s AND dataset_id = %s
                """, (model_id, prompt_id, dataset_id))
                current_status = cursor.fetchone()[0]
                if current_status == 'stop':
                    print(f"Model-prompt-dataset combination {model_name} - {prompt_text} - {dataset_name} is set to stop. Moving to the next combination.")
                    exclude_prompt_ids.append(prompt_id)
                    break
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

        # Decrement the count for this model-prompt-dataset combination
        decrement_count(model_id, prompt_id, dataset_id)

        # After completing all rows for the current prompt, exclude it and fetch a new prompt for the same model
        exclude_prompt_ids.append(prompt_id)

if __name__ == "__main__":
    main()
