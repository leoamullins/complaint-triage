import pandas as pd
from sqlalchemy import text

from complaint.config import default_data_dir
from complaint.db import get_engine

OUT_DIR = default_data_dir()
TRAIN_CAP = 15000
EVAL_N = 20000
SEED = 0.42


def sample_train(conn, cap):
    q = text("""
    SELECT complaint_id, text, label
    FROM (
        SELECT complaint_id,
               narrative AS text,
               product_group AS label,
               ROW_NUMBER() OVER (PARTITION BY product_group ORDER BY random()) AS rn
        FROM model_data
        WHERE split = 'train'
    ) t
    WHERE rn <= :cap
    """)
    return pd.read_sql(q, conn, params={"cap": cap})


def sample_eval(conn, split, n):
    q = text("""
    SELECT complaint_id, narrative AS text, product_group AS label
    FROM model_data
    WHERE split = :split
    ORDER BY random()
    LIMIT :n
    """)
    return pd.read_sql(q, conn, params={"split": split, "n": n})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with get_engine().connect() as conn:
        conn.execute(text("SELECT setseed(:s)"), {"s": SEED})

        splits = {
            "train": sample_train(conn, TRAIN_CAP),
            "val": sample_eval(conn, "val", EVAL_N),
            "test": sample_eval(conn, "test", EVAL_N),
        }

    for name, df in splits.items():
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(name, df.shape)


if __name__ == "__main__":
    main()
