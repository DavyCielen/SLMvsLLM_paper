# Sentiment_Analysis_SML_vs_LLM

This repository contains the code and replication package for the experimental study titled *"Zero-Shot SLM Ensembles are Effective Alternatives to LLMs for Sentiment Analysis"*.

In this project, we compare ensembles of zero-shot **Small Language Models (SLMs)** with proprietary **Large Language Models (LLMs)** such as GPT-3.5 and GPT-4, evaluating their performance on multiple sentiment classification datasets. We explore the trade-offs in terms of accuracy, precision, latency, and deployment flexibility.

The full paper can be found [here](https://doi.org/10.5281/zenodo.15700568).

The final version of the code will be released under the tag `v1.0`, expected on **July 3rd, 2025**.

## Requirements

To run this project, you will need:

- A **PostgreSQL** database instance
- An **OpenAI API key**

All necessary configuration values should be placed in a `.env` file. You can start by copying the provided `.env.example` and renaming it to `.env`.

You will also need **Docker** installed to build and run the containers.

### Optional: Running in Parallel at Scale

For parallel execution at scale, the suggested approach is:

1. Build the Docker images.
2. Upload them to AWS Elastic Container Registry (ECR).
3. Create an ECS cluster and define tasks to run the Docker images.
4. Launch these tasks when you want to perform scoring.

However, the Docker setup is fully portable and will also work on any local or cloud-based machine as long as the PostgreSQL database is reachable.

## Database Setup

To initialize the PostgreSQL database, run the `setup_db.py` script. This script will create all necessary tables and define the triggers used to orchestrate the scoring workflow.

## Docker Build Instructions

Each Dockerfile in this project is designed for a specific model environment (e.g., BERT, GPT-4all, OpenAI, Ollama). As an example, we'll describe how to build the `dockerfile.ollama` image.

The `dockerfile.ollama` sets up its own **Ollama server** for local inference. This allows models like LLaMA or Mistral to be served directly within the container, without requiring external APIs.

To build the image:

```bash
docker build -f dockerfile.ollama -t ollama-runner .
```

You can then run the container (for example):

```bash
docker run --rm -it ollama-runner
```

The image includes everything needed to start the Ollama server and handle predictions locally. You can use this pattern to build the other Dockerfiles (`dockerfile.openai`, `dockerfile.bert`, etc.), adapting them as needed for your desired runtime environment.

## Running Inference

> *Note: The automated prompt generation process will be described later.*

To run inference with this system, you need to follow these steps to prepare your data and configuration:

1. **Create a dataset** in the database.
2. **Create a model entry**, including:
   - A `family` field: this must be one of `bert`, `openai`, or `ollama`
   - A `model_name`: for OpenAI or Ollama, this must exactly match the model name as listed in the respective model card.
3. **Create a prompt** that will be used for inference.
4. **Upload your input data** to the `rows` table. These are the items to be scored.

These steps can be performed via a database IDE or a script.

At this point, no work has been assigned yet.

To trigger inference:

- Insert a row into the `modelpromptstatus` table that links the `model_id`, `dataset_id`, and `prompt_id` you want to score with.
- Set the status of that row to `"available"`.

Once this is done, database triggers will automatically populate the necessary auxiliary tables to schedule prediction tasks.

This is the **generic inference workflow** and works independently of how prompts are generated.

The data used in the experiment is available in the file `data.csv` located in the root of this repository.

## Calculating Ensemble Predictions

To calculate ensemble predictions from the raw prediction data, you can use the `majority_utils.py` script. This script aggregates predictions and determines the most common (majority) prediction for each data row based on various filters.

### How to Run

The script is executed from the command line and accepts several optional arguments to filter the data. If no arguments are provided, it will process the entire dataset.

`python3 majority_utils.py [--model_id MODEL_ID] [--dataset_id DATASET_ID] [--prompt_id PROMPT_ID] [--library LIBRARY] [--run_id RUN_ID]`

### Arguments

- `--model_id`: (Optional) Filter by a specific model ID.
- `--dataset_id`: (Optional) Filter by a specific dataset ID.
- `--prompt_id`: (Optional) Filter by a specific prompt ID.
- `--library`: (Optional) Filter by a model library (e.g., `ollama`, `openai`).
- `--run_id`: (Optional) A specific ID for the run. If not provided, a timestamp will be used. This is useful for grouping related runs together.

### Output

The script will generate a `runs` directory if it doesn't already exist. Inside `runs`, a new directory will be created for each `run_id`. The ensemble prediction results will be saved as a CSV file within the corresponding `run_id` directory.

### Examples

**1. Run on the entire dataset:**

```bash
python3 majority_utils.py
```

**2. Run on predictions from the `ollama` library:**

```bash
python3 majority_utils.py --library ollama
```

**3. Run for a specific model and save to a custom run folder:**

```bash
python3 majority_utils.py --model_id 7 --run_id my_special_run
```

**4. Run with multiple filters:**

```bash
python3 majority_utils.py --library openai --dataset_id 1 --prompt_id 3
```

## Adding Expected Predictions to Results

After generating ensemble predictions, you can enrich the results with the ground truth (`expected_prediction`) for each row. The `add_expected_predictions.py` script is designed for this purpose.

### How to Run

The script takes a single mandatory argument: `run_id`.

`python3 add_expected_predictions.py --run_id RUN_ID`

### Arguments

- `--run_id`: (Required) The ID of the run you want to process. This should correspond to a folder in the `runs` directory.

### Output

The script will create a new directory named `runs_with_expected_predictions`. Inside this directory, it will create a folder with the same `run_id` as the one you provided. The original CSV files from the `runs` folder will be copied here, with the `expected_prediction` and `dataset_id` columns added.

### Example

If you have a run with the ID `20250701_101114`, you can add the expected predictions with the following command:

```bash
python3 add_expected_predictions.py --run_id 20250701_101114
```

## Subsetting Results by Dataset

To analyze a specific subset of your results, you can use the `subset_by_dataset.py` script to filter a run's output files by one or more `dataset_id`s. It is a mandatory step before running the `majority_utils.py` script. If you need all datasets, you can use the `--dataset_ids` argument with a comma-separated list of all dataset IDs.

### How to Run

The script requires both a `run_id` and a list of `dataset_ids`.

`python3 subset_by_dataset.py --run_id RUN_ID --dataset_ids ID1,ID2,...`

### Arguments

- `--run_id`: (Required) The ID of the run you want to process from the `runs_with_expected_predictions` folder.
- `--dataset_ids`: (Required) A comma-separated list of `dataset_id`s to include in the output.

### Output

The script will create a new directory named `subsetted_runs`. Inside, it will create a folder with the specified `run_id`, containing the filtered CSV files.

### Example

To filter the run `20250701_101114` to only include results for `dataset_id` 1 and 3:

```bash
python3 subset_by_dataset.py --run_id 20250701_101114 --dataset_ids 1,3
```

## Calculating Performance Metrics

Finally, to evaluate the performance of your ensemble predictions, you can use the `calculate_metrics.py` script. It computes accuracy, weighted precision, recall, and F1-score.

### How to Run

The script requires a `run_id` corresponding to a folder in the `subsetted_runs` directory.

`python3 calculate_metrics.py --run_id RUN_ID`

### Arguments

- `--run_id`: (Required) The ID of the run you want to analyze from the `subsetted_runs` folder.

### Output

The script will generate a new CSV file in the `results` directory named `<run_id>_metrics.csv`. Each row in this file contains the performance metrics for one of the input CSV files from the run.

### Example

To calculate metrics for the run `20250701_101114`:

```bash
python3 calculate_metrics.py --run_id 20250701_101114
```

## Performing Statistical Analysis

To determine if the performance differences between your models are statistically significant, you can use the `perform_statistical_tests.py` script. It runs a Friedman test, and if the results are significant, it follows up with a Nemenyi post-hoc test.

### How to Run

The script requires a `run_id` and an optional `alpha` level.

`python3 perform_statistical_tests.py --run_id RUN_ID [--alpha 0.05]`

### Arguments

- `--run_id`: (Required) The ID of the run you want to analyze from the `subsetted_runs` folder. You must have at least 3 models in the run folder for the test to work.
- `--alpha`: (Optional) The significance level for the tests. Defaults to `0.05`.

### Output

The script generates two files in the `results` directory:
- `<run_id>_friedman_test.txt`: Contains the statistic and p-value of the Friedman test.
- `<run_id>_nemenyi_test.csv`: If the Friedman test was significant, this file contains a matrix of p-values comparing each pair of models.

### Example

To run the statistical tests on the run `20250701_101114`:

```bash
python3 perform_statistical_tests.py --run_id 20250701_101114
```



