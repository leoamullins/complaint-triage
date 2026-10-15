from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    data_dir = "data"
    text_col = "text"
    label_col = "label"
    train_limit = None


@dataclass
class ModelConfig:
    name = "roberta-base"
    max_len = 512


@dataclass
class TrainingConfig:
    lr = 2e-5
    epochs = 2
    batch_size = 32
    eval_batch_size = 64
    weight_decay = 0.01
    warmup_frac = 0.06
    bf16 = True
    group_by_length = True
    grad_accum = 1


@dataclass
class EvalConfig:
    tau = 1.0
    prior_source = "val"


@dataclass
class Config:
    run_name = "roberta-512"
    seed = 42
    output_dir = "outputs/roberta_512"
    data = field(default_factory=DataConfig)
    model = field(default_factory=ModelConfig)
    training = field(default_factory=TrainingConfig)
    evaluation = field(default_factory=EvalConfig)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}

    return Config(
        run_name=raw.get("run_name", "run"),
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "outputs/run"),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
    )


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
