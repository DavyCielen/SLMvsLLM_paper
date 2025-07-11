
ALGORITHM: Parallelized Predictions

INPUT TABLES:
  - prompts
  - models
  - datasets
  - modelpromptdatasetstatus  (MPS)
  - predictionstatus
  - datatopredict
  - predictions

OUTPUT:
  - predictions table filled with final results
  - updated statuses in modelpromptdatasetstatus & predictionstatus

--------------------------------------------------------------------------
STEP 1: ADD DATA TO PREDICT
--------------------------------------------------------------------------
1.1 A new dataset to be predicted is stored in datatopredict (immutable).
1.2 For each row in datatopredict, ensure a matching entry in datasets.

--------------------------------------------------------------------------
STEP 2: DEFINE MODELS & PROMPTS
--------------------------------------------------------------------------
2.1 For each relevant model in models and each prompt in prompts:
    - Identify the (model, prompt, dataset) combination that must be processed.
    - Insert a row into MPS (modelpromptdatasetstatus) with status = "available"
      and workerCount = 0.

--------------------------------------------------------------------------
STEP 3: POPULATE PREDICTIONSTATUS
--------------------------------------------------------------------------
3.1 For each row r in datatopredict and each MPS combination (m, p, d):
    - Insert a record into predictionstatus:
        rowId = r.id
        mpsId = ID of the (m,p,d) from MPS
        status = "pending"
        retryCount = 0
    - This structure tracks the prediction tasks at row-level.

--------------------------------------------------------------------------
STEP 4: START WORKER CONTAINERS
--------------------------------------------------------------------------
4.1 Deploy containers (e.g., on Kubernetes/local).
4.2 Each container is assigned a "family" of models or certain MPS combos 
    it is allowed to process.

--------------------------------------------------------------------------
STEP 5: WORKER LOOP (IN EACH CONTAINER)
--------------------------------------------------------------------------
WHILE TRUE:
  5.1 Lookup MPS row where status = "available" 
      AND the container is allowed to handle (model,prompt).
      IF none found:
        BREAK (container stops)

      ELSE:
        - Set MPS.status = "in_use"
        - MPS.workerCount++

  5.2 BATCH PREDICTION:
      WHILE (there exist up to 10 rows in predictionstatus with mpsId = current
             AND status = "pending"):
        - Retrieve up to 10 pending rows
        - Update them to status = "in_progress"
        - For each row:
            * Perform <MODEL> with <PROMPT> on that row (rowId input)
            * Store result in predictions table:
                (mpsId, rowId, prediction, timeTaken, etc.)
            * When complete, set that row's status = "done"

  5.3 CHECK IF FINISHED:
      - If no rows in "pending" or "in_progress" for this MPS:
          MPS.workerCount--
          IF MPS.workerCount == 0:
            MPS.status = "done"

--------------------------------------------------------------------------
STEP 6: MANAGEMENT CONTAINER (WATCHDOG)
--------------------------------------------------------------------------
6.1 Periodically run:
    FOR each row in predictionstatus where status = "in_progress":
      IF time_in_progress > threshold:
        row.status = "pending"  (reset task)
        row.retryCount++
        IF row.retryCount > MAX_RETRIES:
          row.status = "failed"

    IF an MPS has status = "done" but 
       any associated row in predictionstatus is reset to "pending":
       - MPS.status = "in_use" or "available" (depending on logic)
         so a new worker can pick up those reset tasks

END
