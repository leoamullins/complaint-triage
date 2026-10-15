from pathlib import Path

import pandas as pd
from datasets import Dataset

from complaint.config import default_data_dir

SPLITS = ("train", "val", "test")


def load_splits(cfg, seed=42):
    """
    read train/val/test parquet files
    """
    data_dir = Path(cfg.data_dir) if cfg.data_dir else default_data_dir()
    dfs = {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in SPLITS}

    if cfg.train_limit is not None:
        train = dfs["train"]
        frac = min(1.0, cfg.train_limit / len(train))
        dfs["train"] = (
            train.groupby(cfg.label_col, group_keys=False)
            .sample(frac=frac, random_state=seed)
            .reset_index(drop=True)
        )

    for name, df in dfs.items():
        missing = {cfg.text_col, cfg.label_col} - set(df.columns)
        assert not missing, f"{name}: missing columns {missing}"

    return dfs


def make_label_maps(labels):
    """
    fixed sorted mapping so ids are identical across runs
    """
    classes = sorted(set(labels))
    label2id = {name: i for i, name in enumerate(classes)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


def add_label_ids(dfs, label2id, label_col):
    """
    add the integer labels column for the trainer
    """
    out = {}
    for name, df in dfs.items():
        df = df.copy()
        df["labels"] = df[label_col].map(label2id)
        unmapped = df["labels"].isna()
        assert not unmapped.any(), (
            f"{name}: {unmapped.sum()} rows with unknown labels "
            f"{sorted(df.loc[unmapped, label_col].unique())}"
        )
        df["labels"] = df["labels"].astype(int)
        out[name] = df
    return out


def to_datasets(dfs, tok, max_len, text_col):
    def tokenise(batch):
        return tok(batch[text_col], truncation=True, max_length=max_len)

    return {
        name: Dataset.from_pandas(df[[text_col, "labels"]], preserve_index=False).map(
            tokenise, batched=True, remove_columns=[text_col]
        )
        for name, df in dfs.items()
    }


def prepare(cfg, tok, max_len, seed):
    dfs = load_splits(cfg, seed)
    all_labels = pd.concat([df[cfg.label_col] for df in dfs.values()])
    label2id, id2label = make_label_maps(all_labels)
    dfs = add_label_ids(dfs, label2id, cfg.label_col)
    ds = to_datasets(dfs, tok, max_len, cfg.text_col)
    return dfs, ds, label2id, id2label
