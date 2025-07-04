Database Schema Overview (Core Operational Tables)

This file documents the core operational tables and gives a jump-start SQLAlchemy definition for each one. Feel free to copy-paste the code blocks into a models.py later.

Schema note  All tables are in the public schema unless explicitly noted.

⸻

1  datasets

column	type	null	default	notes
dataset_id	INTEGER	NO	nextval('datasets_dataset_id_seq')	primary-key
name	VARCHAR	NO	—	dataset display name
description	TEXT	YES	—	optional

class Datasets(Base):
    __tablename__ = "datasets"

    dataset_id  = Column(Integer, primary_key=True)
    name        = Column(String,  nullable=False)
    description = Column(Text)


⸻

2  modelpromptstatus

column	type	null	default	notes
model_id	INTEGER	NO	—	FK → models.model_id
prompt_id	INTEGER	NO	—	FK → prompts.prompt_id
dataset_id	INTEGER	NO	—	FK → datasets.dataset_id
status	VARCHAR	NO	—	pending / running / done
count	INTEGER	YES	0	processed rows

class ModelPromptStatus(Base):
    __tablename__ = "modelpromptstatus"
    __table_args__ = (
        PrimaryKeyConstraint("model_id", "prompt_id", "dataset_id"),
    )

    model_id   = Column(Integer, ForeignKey("models.model_id"),  nullable=False)
    prompt_id  = Column(Integer, ForeignKey("prompts.prompt_id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"), nullable=False)

    status = Column(String,  nullable=False)
    count  = Column(Integer, server_default=text("0"))


⸻

3  models

column	type	null	default	notes
model_id	INTEGER	NO	nextval('models_model_id_seq')	PK
name	VARCHAR	NO	—	
source	VARCHAR	NO	—	openai, ollama, …
description	TEXT	YES	—	
library	VARCHAR	NO	'unknown'	runtime lib

class Models(Base):
    __tablename__ = "models"

    model_id    = Column(Integer, primary_key=True)
    name        = Column(String,  nullable=False)
    source      = Column(String,  nullable=False)
    description = Column(Text)
    library     = Column(String,  nullable=False, server_default=text("'unknown'"))


⸻

4  predictions

column	type	null	default	notes
prediction_id	INTEGER	NO	nextval('predictions_prediction_id_seq')	PK
row_id	INTEGER	YES	—	FK → rows.row_id
model_id	INTEGER	YES	—	FK → models.model_id
prompt_id	INTEGER	YES	—	FK → prompts.prompt_id
dataset_id	INTEGER	YES	—	FK → datasets.dataset_id
prediction	VARCHAR	NO	—	model output
prediction_time	DOUBLE PRECISION	NO	—	seconds
status	VARCHAR	NO	—	success / failed
formatted_prompt	TEXT	YES	—	prompt after fill-in

class Predictions(Base):
    __tablename__ = "predictions"

    prediction_id   = Column(Integer, primary_key=True)
    row_id          = Column(Integer, ForeignKey("rows.row_id"))
    model_id        = Column(Integer, ForeignKey("models.model_id"))
    prompt_id       = Column(Integer, ForeignKey("prompts.prompt_id"))
    dataset_id      = Column(Integer, ForeignKey("datasets.dataset_id"))

    prediction      = Column(String,  nullable=False)
    prediction_time = Column(Float,   nullable=False)
    status          = Column(String,  nullable=False)
    formatted_prompt = Column(Text)


⸻

5  rows

column	type	null	default	notes
row_id	INTEGER	NO	nextval('rows_row_id_seq')	PK
dataset_id	INTEGER	YES	—	FK → datasets.dataset_id
content	TEXT	NO	—	raw row text
expected_prediction	VARCHAR	NO	—	gold label

class Rows(Base):
    __tablename__ = "rows"

    row_id              = Column(Integer, primary_key=True)
    dataset_id          = Column(Integer, ForeignKey("datasets.dataset_id"))
    content             = Column(Text,   nullable=False)
    expected_prediction = Column(String, nullable=False)


⸻

6  predictionstatus

column	type	null	default	notes
row_id	INTEGER	NO	—	FK → rows.row_id
model_id	INTEGER	NO	—	FK
prompt_id	INTEGER	NO	—	FK
dataset_id	INTEGER	YES	—	FK
status	VARCHAR	NO	—	pending / in_progress / done
in_progress_time	TIMESTAMP	YES	—	last heartbeat

class PredictionStatus(Base):
    __tablename__ = "predictionstatus"
    __table_args__ = (
        PrimaryKeyConstraint("row_id", "model_id", "prompt_id"),
    )

    row_id           = Column(Integer, ForeignKey("rows.row_id"),     nullable=False)
    model_id         = Column(Integer, ForeignKey("models.model_id"),  nullable=False)
    prompt_id        = Column(Integer, ForeignKey("prompts.prompt_id"),nullable=False)
    dataset_id       = Column(Integer, ForeignKey("datasets.dataset_id"))
    status           = Column(String,  nullable=False)
    in_progress_time = Column(DateTime)


⸻

7  prompts

column	type	null	default	notes
prompt_id	INTEGER	NO	nextval('prompts_prompt_id_seq')	PK
model_id	INTEGER	YES	—	optional FK
text	TEXT	NO	—	prompt template

class Prompts(Base):
    __tablename__ = "prompts"

    prompt_id = Column(Integer, primary_key=True)
    model_id  = Column(Integer, ForeignKey("models.model_id"))
    text      = Column(Text, nullable=False)

## Trigger Documentation  (public schema)

| Table | Trigger name | Timing / Event | Executes function | Purpose |
|-------|--------------|----------------|-------------------|---------|
| `modelpromptstatus` | **`after_insert_model_prompt_status`** | **AFTER INSERT** FOR EACH ROW | `add_prediction_status_for_model_prompt_dataset()` | Whenever a new job (`model_id × prompt_id × dataset_id`) is inserted, this function “seeds” **`predictionstatus`** with a **pending** record for *every* row in the target dataset, so workers know what must be processed. |
| `modelpromptstatus` | **`after_delete_model_prompt_status`** | **AFTER DELETE** FOR EACH ROW | `remove_prediction_status_for_model_prompt_dataset()` | If a job definition is removed/cancelled, the function deletes any orphaned `predictionstatus` rows, preventing stale progress counters. |
| `predictions` | **`predictions_after_insert`** | **AFTER INSERT** FOR EACH ROW | `update_modelpromptstatus()` | Each new prediction row increments the *rows_done* counter in **`modelpromptstatus`** and, when the dataset is fully processed, flips that job’s status to **done**. |
| `predictionstatus` | **`before_update_prediction_status`** | **BEFORE UPDATE** FOR EACH ROW | `update_in_progress_time()` | Whenever a row’s `status` transitions to **in_progress**, this function stamps `in_progress_time` with `CURRENT_TIMESTAMP`, giving a heartbeat/audit trail for worker activity. |

---

### Re-creation DDL snippets (template)

```sql
-- AFTER-INSERT: seed predictionstatus
CREATE TRIGGER after_insert_model_prompt_status
AFTER INSERT ON modelpromptstatus
FOR EACH ROW
EXECUTE FUNCTION add_prediction_status_for_model_prompt_dataset();

-- AFTER-DELETE: clean up orphan status rows
CREATE TRIGGER after_delete_model_prompt_status
AFTER DELETE ON modelpromptstatus
FOR EACH ROW
EXECUTE FUNCTION remove_prediction_status_for_model_prompt_dataset();

-- AFTER-INSERT on predictions: update job counters
CREATE TRIGGER predictions_after_insert
AFTER INSERT ON predictions
FOR EACH ROW
EXECUTE FUNCTION update_modelpromptstatus();

-- BEFORE-UPDATE on predictionstatus: timestamp in-progress
CREATE TRIGGER before_update_prediction_status
BEFORE UPDATE ON predictionstatus
FOR EACH ROW
EXECUTE FUNCTION update_in_progress_time();


## functions needed
/* ====================================================================
   Trigger Functions – definitions only (no DROP / CREATE TRIGGER)
   ==================================================================== */

-- 1. Seed predictionstatus rows whenever a new job is inserted
CREATE OR REPLACE FUNCTION public.add_prediction_status_for_model_prompt_dataset()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO public.predictionstatus (row_id, model_id, prompt_id, dataset_id, status)
    SELECT r.row_id, NEW.model_id, NEW.prompt_id, NEW.dataset_id, 'pending'
    FROM   public.rows r
    WHERE  r.dataset_id = NEW.dataset_id
    ON CONFLICT DO NOTHING;
    RETURN NEW;
END;
$$;


-- 2. Clean up predictionstatus rows when a job is deleted
CREATE OR REPLACE FUNCTION public.remove_prediction_status_for_model_prompt_dataset()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM public.predictionstatus
    WHERE model_id  = OLD.model_id
      AND prompt_id = OLD.prompt_id
      AND dataset_id = OLD.dataset_id;
    RETURN OLD;
END;
$$;


-- 3. Timestamp a row when its status becomes in_progress
CREATE OR REPLACE FUNCTION public.update_in_progress_time()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.status = 'in_progress' THEN
        NEW.in_progress_time := CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$;


-- 4. Update modelpromptstatus counters after each prediction
CREATE OR REPLACE FUNCTION public.update_modelpromptstatus()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    total_rows INTEGER;
    count_rows INTEGER;
BEGIN
    -- rows processed so far for this (model, prompt, dataset)
    SELECT COUNT(*) INTO count_rows
    FROM   public.predictions
    WHERE  model_id  = NEW.model_id
      AND  prompt_id = NEW.prompt_id
      AND  dataset_id = NEW.dataset_id;

    -- every 50th insert, check for completion
    IF count_rows % 50 = 0 THEN
        SELECT COUNT(*) INTO total_rows
        FROM   public.predictions
        WHERE  dataset_id = NEW.dataset_id;

        IF count_rows >= total_rows OR count_rows > 5000 THEN
            UPDATE public.modelpromptstatus
               SET status = 'stop'
             WHERE model_id  = NEW.model_id
               AND prompt_id = NEW.prompt_id
               AND dataset_id = NEW.dataset_id;

            INSERT INTO public.status_update_log (model_id, prompt_id, dataset_id, status)
            VALUES (NEW.model_id, NEW.prompt_id, NEW.dataset_id, 'in_use');
        END IF;
    END IF;

    RETURN NEW;
END;
$$;


⸻

How to use this file
	•	Documentation  ➡ readable in any Markdown viewer.
	•	SQLAlchemy   ➡ copy the code blocks into a module; adjust relationships & enums as needed.
	•	Migration    ➡ feed the models to Alembic autogeneration (alembic revision --autogenerate).