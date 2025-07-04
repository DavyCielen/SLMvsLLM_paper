"""
Database schema and trigger setup module.
Use create_schema(conn) to initialize tables, functions, and triggers.
"""
from sqlalchemy import text

# DDL statements for tables, functions, and triggers.
DDL = [
    """
    CREATE TABLE modelpromptstatus (
        model_id   INT  NOT NULL,
        prompt_id  INT  NOT NULL,
        dataset_id INT  NOT NULL,
        status     VARCHAR NOT NULL DEFAULT 'pending',
        PRIMARY KEY (model_id, prompt_id, dataset_id)
    );
    """,
    """
    CREATE TABLE predictionstatus (
        row_id INT NOT NULL,
        model_id INT NOT NULL,
        prompt_id INT NOT NULL,
        dataset_id INT NOT NULL,
        status VARCHAR NOT NULL,
        in_progress_time TIMESTAMP,
        PRIMARY KEY (row_id, model_id, prompt_id, dataset_id)
    );
    """,
    """
    CREATE TABLE predictions (
        prediction_id SERIAL PRIMARY KEY,
        row_id INT,
        model_id INT,
        prompt_id INT,
        dataset_id INT,
        prediction VARCHAR NOT NULL,
        prediction_time FLOAT8 NOT NULL,
        status VARCHAR NOT NULL,
        formatted_prompt TEXT
    );
    """,
    """
    CREATE TABLE rows (
        row_id INT PRIMARY KEY,
        dataset_id INT NOT NULL,
        content TEXT,
        expected_prediction VARCHAR
    );
    """,
    """
    CREATE TABLE status_update_log (
        id SERIAL PRIMARY KEY,
        model_id INT,
        prompt_id INT,
        dataset_id INT,
        status VARCHAR,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE OR REPLACE FUNCTION public.add_prediction_status_for_model_prompt_dataset()
    RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        INSERT INTO public.predictionstatus (row_id, model_id, prompt_id, dataset_id, status)
        SELECT r.row_id, NEW.model_id, NEW.prompt_id, NEW.dataset_id, 'pending'
        FROM   public.rows r
        WHERE  r.dataset_id = NEW.dataset_id
        ON CONFLICT DO NOTHING;
        RETURN NEW;
    END;$$;
    """,
    """
    CREATE OR REPLACE FUNCTION public.remove_prediction_status_for_model_prompt_dataset()
    RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        DELETE FROM public.predictionstatus
        WHERE model_id  = OLD.model_id
          AND prompt_id = OLD.prompt_id
          AND dataset_id = OLD.dataset_id;
        RETURN OLD;
    END;$$;
    """,
    """
    CREATE OR REPLACE FUNCTION public.update_in_progress_time()
    RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
        IF NEW.status = 'in_progress' THEN
            NEW.in_progress_time := CURRENT_TIMESTAMP;
        END IF;
        RETURN NEW;
    END;$$;
    """,
    """
    CREATE OR REPLACE FUNCTION public.update_modelpromptstatus()
    RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE
        total_rows INTEGER;
        count_rows INTEGER;
    BEGIN
        SELECT COUNT(*) INTO count_rows
        FROM   public.predictions
        WHERE  model_id  = NEW.model_id
          AND  prompt_id = NEW.prompt_id
          AND  dataset_id = NEW.dataset_id;

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
    END;$$;
    """,
    """
    CREATE TRIGGER after_insert_model_prompt_status
    AFTER INSERT ON modelpromptstatus
    FOR EACH ROW EXECUTE FUNCTION add_prediction_status_for_model_prompt_dataset();
    """,
    """
    CREATE TRIGGER after_delete_model_prompt_status
    AFTER DELETE ON modelpromptstatus
    FOR EACH ROW EXECUTE FUNCTION remove_prediction_status_for_model_prompt_dataset();
    """,
    """
    CREATE TRIGGER before_update_prediction_status
    BEFORE UPDATE ON predictionstatus
    FOR EACH ROW EXECUTE FUNCTION update_in_progress_time();
    """,
    """
    CREATE TRIGGER predictions_after_insert
    AFTER INSERT ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_modelpromptstatus();
    """,
]

def create_schema(conn):
    """
    Execute all DDL statements on the given SQLAlchemy connection or transaction.
    """
    for stmt in DDL:
        conn.execute(text(stmt))