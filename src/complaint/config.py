import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_data_dir():
    """
    DATA_DIR from the environment or .env, else data/ in the repo
    """
    return Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))


@dataclass
class DataConfig:
    data_dir: str | None = None  # None -> default_data_dir()
    text_col: str = "text"
    label_col: str = "label"
    train_limit: int | None = None


@dataclass
class ModelConfig:
    name: str = "roberta-base"
    max_len: int = 512


@dataclass
class TrainingConfig:
    lr: float = 2e-5
    epochs: int = 2
    batch_size: int = 32
    eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    bf16: bool = True
    group_by_length: bool = True
    grad_accum: int = 1


@dataclass
class EvalConfig:
    tau: float = 1.0
    prior_source: str = "val"
    review_threshold: float | None = None  # confidence below this goes to a human


@dataclass
class Config:
    run_name: str = "roberta-512"
    seed: int = 42
    output_dir: str = "outputs/roberta_512"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    default = Config()

    return Config(
        run_name=raw.get("run_name", default.run_name),
        seed=raw.get("seed", default.seed),
        output_dir=raw.get("output_dir", default.output_dir),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
    )


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
