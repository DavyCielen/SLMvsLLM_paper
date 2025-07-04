import psycopg2
import pandas as pd

# Database connection info (you may want to read from .env instead)
DB_CONFIG = {
    "dbname": "your_db_name",
    "user": "your_username",
    "password": "your_password",
    "host": "your_host",
    "port": 5432,
}

# Mapping of paper dataset names to db names
DATASET_MAPPING = {
    "Amazon Product Review": "Amazon",
    "IMDB": "imdb",
    "SST-2 (Stanford)": "SST2",
    "Sentiment 140": "Sampled Sentiment140",
    "Yelp": "Yelp reviews",
    "SemEval2014 (Laptop)": "Semeval_2014_laptops",
    "SemEval2014 (Restaurant)": "data_restaurant_semeval_2014_2",
}

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("Dataset\t\t\tExpected\tFound")
    print("-" * 40)

    for paper_name, db_name in DATASET_MAPPING.items():
        # Get dataset ID from name
        cursor.execute("SELECT id FROM datasets WHERE name = %s", (db_name,))
        result = cursor.fetchone()
        if not result:
            print(f"{paper_name:<25}\tMISSING\tMISSING")
            continue
        dataset_id = result[0]

        # Count rows
        cursor.execute("SELECT COUNT(*) FROM rows WHERE datasetid = %s", (dataset_id,))
        count = cursor.fetchone()[0]
        print(f"{paper_name:<25}\t?????\t{count}")  # Replace ????? with actual values if desired

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
