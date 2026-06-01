from __future__ import annotations

import json
import os
from pathlib import Path

import click
import torch
import yaml
from datasets import Dataset
from loguru import logger
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from slonik.data.chatml import Example, format_dataset


def _load_jsonl(paths: list[str]) -> list[Example]:
    rows: list[Example] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rows.append(Example(
                    schema=d["schema"],
                    question=d["question"],
                    sql=d["sql"],
                    evidence=d.get("evidence", ""),
                    db_id=d.get("db_id", ""),
                ))
    return rows


def _build_datasets(cfg: dict, tokenizer) -> tuple[Dataset, Dataset]:
    train_paths = cfg["data"]["train_path"]
    eval_paths = cfg["data"]["eval_path"]
    if isinstance(train_paths, str):
        train_paths = [train_paths]
    if isinstance(eval_paths, str):
        eval_paths = [eval_paths]

    train_rows = _load_jsonl(train_paths)
    eval_rows = _load_jsonl(eval_paths)
    logger.info(f"Loaded {len(train_rows)} train / {len(eval_rows)} eval examples")

    train_ds = Dataset.from_list(format_dataset(train_rows, tokenizer))
    eval_ds = Dataset.from_list(format_dataset(eval_rows, tokenizer))
    return train_ds, eval_ds


@click.command()
@click.option("--config", default="configs/sft_qlora.yaml", type=click.Path(exists=True))
@click.option("--resume", is_flag=True)
def main(config: str, resume: bool) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    os.environ.setdefault("WANDB_PROJECT", "slonik-7b")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["base"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else None,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
        random_state=cfg["lora"]["random_state"],
    )

    train_ds, eval_ds = _build_datasets(cfg, tokenizer)

    from trl import SFTConfig, SFTTrainer

    t = cfg["training"]
    sft_args = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        max_grad_norm=t.get("max_grad_norm", 1.0),
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        bf16=is_bfloat16_supported() and t["bf16"],
        fp16=not is_bfloat16_supported() and t["fp16"],
        seed=t["seed"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        dataset_text_field=cfg["data"]["text_field"],
        dataset_num_proc=cfg["data"]["num_proc"],
        packing=cfg["data"]["packing"],
        max_seq_length=cfg["model"]["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )

    if cfg["data"]["train_on_responses_only"]:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    stats = trainer.train(resume_from_checkpoint=resume)
    logger.info(f"Done in {stats.metrics.get('train_runtime', 0):.0f}s")

    merge_dir = cfg["merge"]["output_dir"]
    Path(merge_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(merge_dir, tokenizer, save_method=cfg["merge"]["save_method"])
    logger.info(f"Merged 16-bit model saved → {merge_dir}")


if __name__ == "__main__":
    main()
