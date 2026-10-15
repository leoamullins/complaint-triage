import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from transformers.trainer_utils import get_last_checkpoint

from complaint.config import load_config, save_config
from complaint.data import prepare
from complaint.model import build_tokeniser, build_model
from complaint.evaluation import (
    compute_metrics,
    prior_log_ratio,
    adjusted_preds,
    tau_sweep,
    save_prior_adjustment,
)
from complaint.calibration import fit_temperature


class OrderedEvalTrainer(Trainer):
    """
    group_by_length also reorders eval/predict batches in transformers 5,
    which breaks the row order of trainer.predict outputs; keep eval sequential
    """

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)


def train(cfg):
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    save_config(cfg, out / "config.yaml")

    # data
    tok = build_tokeniser(cfg.model.name)
    dfs, ds, label2id, id2label = prepare(cfg.data, tok, cfg.model.max_len, cfg.seed)

    # model
    model = build_model(cfg.model.name, label2id, id2label)

    # training args
    steps_per_epoch = math.ceil(
        len(ds["train"]) / (cfg.training.batch_size * cfg.training.grad_accum)
    )
    warmup_steps = math.ceil(
        steps_per_epoch * cfg.training.epochs * cfg.training.warmup_frac
    )
    t = cfg.training
    bf16 = t.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if t.bf16 and not bf16:
        print("warning: bf16 not supported on this device, training in fp32")
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        learning_rate=t.lr,
        num_train_epochs=t.epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.grad_accum,
        weight_decay=t.weight_decay,
        warmup_steps=warmup_steps,
        bf16=bf16,
        train_sampling_strategy="group_by_length" if t.group_by_length else "random",
        seed=cfg.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
    )

    # train
    trainer = OrderedEvalTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    # pick up from the last epoch checkpoint if a previous run was cut off
    ckpt_dir = Path(args.output_dir)
    last_ckpt = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.is_dir() else None
    if last_ckpt:
        print(f"resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # prior correction
    n_classes = len(label2id)
    train_y = dfs["train"]["labels"].to_numpy()
    val_y = dfs["val"]["labels"].to_numpy()
    source = cfg.evaluation.prior_source
    if source not in ("val", "uniform"):
        raise ValueError(f"unknown evaluation.prior_source: {source!r}")

    def target(y):
        return y if source == "val" else None

    log_ratio = prior_log_ratio(train_y, target(val_y), n_classes)

    # estimate the prior on one half of val, pick tau on the other, so the sweep isn't in-sample
    val_out = trainer.predict(ds["val"])
    idx_prior, idx_sweep = train_test_split(
        np.arange(len(val_y)), test_size=0.5, stratify=val_y, random_state=cfg.seed
    )
    sweep_ratio = prior_log_ratio(train_y, target(val_y[idx_prior]), n_classes)
    sweep = tau_sweep(val_out.predictions[idx_sweep], val_y[idx_sweep], sweep_ratio)
    for tau, f1 in sweep.items():
        print(f"tau={tau:<5} held-out val macro-F1={f1:.4f}")
    best_tau = max(sweep, key=sweep.get)
    print(f"best tau on held-out val: {best_tau} (config tau: {cfg.evaluation.tau})")

    T = fit_temperature(val_out.predictions, val_y, log_ratio, tau=cfg.evaluation.tau)
    print(f"temperature on val: {T:.3f}")

    # save
    model_dir = out / "model"
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    save_prior_adjustment(
        log_ratio, cfg.evaluation.tau, model_dir, best_tau=best_tau, temperature=T
    )
    print(f"saved model to {model_dir}")

    return trainer, dfs, ds, log_ratio


def main():
    p = argparse.ArgumentParser(description="train a complaint classifier")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--max-len", type=int, help="override model.max_len")
    p.add_argument("--train-limit", type=int, help="override data.train_limit")
    p.add_argument("--output-dir", help="override output_dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_len is not None:
        cfg.model.max_len = args.max_len
    if args.train_limit is not None:
        cfg.data.train_limit = args.train_limit
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    train(cfg)


if __name__ == "__main__":
    main()
