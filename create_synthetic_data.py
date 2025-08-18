#!/usr/bin/env python3
"""create_synthetic_data.py

Sample up to 1000 rows from the `rows` table for `dataset_id` 1 and 4,
paraphrase the content using Claude 3.5 while retaining the same
sentiment, and store the results in `paraphrased_data.csv`.

Environment variables are expected in a `.env` file located in the project
root. The following variables must be present:

DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
ANTHROPIC_API_KEY (for Claude 3.5 access)
BATCH_SIZE (optional, defaults to 25)

Install dependencies (once):
    pip install psycopg2-binary python-dotenv anthropic

Usage:
    python create_synthetic_data.py --max-samples 1000 --output paraphrased_data.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Install with `pip install python-dotenv`.", file=sys.stderr)
    sys.exit(1)

try:
    from anthropic import Anthropic
except ImportError:
    print("anthropic package not installed. Install with `pip install anthropic`.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config() -> None:
    """Load environment variables from .env if present."""
    # `override=True` ensures .env overrides existing env vars when running locally
    load_dotenv(override=True)


def get_db_connection():
    """Create a new PostgreSQL connection using env variables."""
    required = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
    missing = [var for var in required if os.getenv(var) is None]
    if missing:
        raise RuntimeError(f"Missing required DB env vars: {', '.join(missing)}")

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    return conn


# ---------------------------------------------------------------------------
# Data access layer
# ---------------------------------------------------------------------------

def fetch_samples(conn, max_samples: int) -> List[Tuple[int, str, str]]:
    """Fetch `max_samples` random rows for dataset_id 1 & 4.

    Returns a list of tuples: (dataset_id, content, sentiment)
    """
    query = (
        "SELECT dataset_id, content, expected_prediction AS sentiment "
        "FROM rows WHERE dataset_id IN (1, 4) "
        "ORDER BY random() LIMIT %s;"
    )
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (max_samples,))
        rows = cur.fetchall()
    return [(row["dataset_id"], row["content"], row["sentiment"]) for row in rows]


# ---------------------------------------------------------------------------
# Paraphrasing via Claude 3.5
# ---------------------------------------------------------------------------

def init_anthropic_client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    return Anthropic(api_key=api_key)


def paraphrase(client: Anthropic, sentence: str, model: str = "claude-opus-4-20250514", max_tokens: int = 128) -> str:
    """Return a paraphrased version of *sentence* using Claude."""
    # Instruction prompt keeps sentiment unchanged
    system_prompt = (
        "You are an expert writer. Paraphrase the user's sentence while keeping "
        "the *sentiment* and *meaning* unchanged. Respond with ONLY the "
        "paraphrased sentence; do not add anything else."
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        system=system_prompt,
        messages=[{"role": "user", "content": sentence}],
    )

    # anthropic>=0.20 returns response.content list -> string
    if hasattr(response, "content"):
        if isinstance(response.content, list):
            return "".join(block.text for block in response.content).strip()
        return str(response.content).strip()
    # fallback: older SDKs
    return str(response).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create paraphrased synthetic dataset using Claude")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of samples (total)")
    parser.add_argument("--output", type=str, default="paraphrased_data.csv", help="Output CSV filename")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "25")), help="Batch size for API calls")
    args = parser.parse_args()

    load_config()

    # Connect to DB & fetch samples
    print("Connecting to DB…")
    conn = get_db_connection()
    try:
        print(f"Fetching up to {args.max_samples} random rows…")
        samples = fetch_samples(conn, args.max_samples)
    finally:
        conn.close()
    print(f"Fetched {len(samples)} rows.")

    if not samples:
        print("No samples fetched. Exiting.")
        sys.exit(0)

    # Init Anthropic client
    print("Initialising Anthropic client…")
    client = init_anthropic_client()

    # Paraphrase in batches
    output_rows: List[Tuple[int, str, str, str]] = []  # dataset_id, original, paraphrased, sentiment
    total = len(samples)
    for start in range(0, total, args.batch_size):
        batch = samples[start : start + args.batch_size]
        for dataset_id, original, sentiment in batch:
            try:
                paraphrased = paraphrase(client, original)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to paraphrase sentence: {exc}")
                paraphrased = ""
            output_rows.append((dataset_id, original, paraphrased, sentiment))
        print(f"Processed {min(start + args.batch_size, total)}/{total} rows…")

    # Write CSV
    print(f"Writing results to {args.output}…")
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset_id", "original", "paraphrased", "sentiment"])
        writer.writerows(output_rows)
    print("Done!")


if __name__ == "__main__":
    main()
