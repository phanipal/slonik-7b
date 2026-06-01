from __future__ import annotations

import json
import os
from pathlib import Path

import click
import torch
import yaml
from datasets import Dataset
from loguru import logger
from unsloth import FastLanguageModel

from slonik.data.chatml import to_prompt
from slonik.training.rewards import format_reward, make_exec_reward, syntax_reward


def _build_prompt_dataset(path: str, tokenizer) -> Dataset:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "prompt": to_prompt(d["schema"], d["question"], tokenizer, d.get("evidence", "")),
                "db_id": d["db_id"],
                "gold_sql": d["sql"],
            })
    return Dataset.from_list(rows)


@click.command()
@click.option("--config", default="configs/grpo.yaml", type=click.Path(exists=True))
def main(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    os.environ.setdefault("WANDB_PROJECT", "slonik-7b")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["sft_checkpoint"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=torch.bfloat16,
        fast_inference=cfg["model"]["fast_inference"],
        max_lora_rank=cfg["model"]["max_lora_rank"],
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
    )

    dataset = _build_prompt_dataset(cfg["data"]["prompts_path"], tokenizer)
    logger.info(f"Loaded {len(dataset)} prompts")

    from trl import GRPOConfig, GRPOTrainer
    '''from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        top_k=cfg["generation"]["top_k"],
        max_tokens=cfg["generation"]["max_new_tokens"],
        min_tokens=cfg["generation"]["min_new_tokens"],
        repetition_penalty=cfg["generation"]["repetition_penalty"],
    )'''

    t = cfg["training"]
    grpo_args = GRPOConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        beta=t["beta"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        optim=t["optim"],
        max_grad_norm=t["max_grad_norm"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=t["bf16"],
        seed=t["seed"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        num_generations=cfg["generation"]["num_generations"],
        max_prompt_length=cfg["model"]["max_seq_length"] - cfg["generation"]["max_new_tokens"],
        max_completion_length=cfg["generation"]["max_new_tokens"],
        use_vllm=False,
        #vllm_sampling_params=sampling,
    )

    rewards = cfg["rewards"]
    exec_fn = make_exec_reward(
        databases_root=cfg["data"]["databases_root"],
        dialect=rewards["dialect"],
        timeout=rewards["timeout_seconds"],
    )

    def weighted_exec(completions, **kwargs):
        return [rewards["exec_match_weight"] * s for s in exec_fn(completions, **kwargs)]

    def weighted_syntax(completions, **_):
        return [rewards["syntax_valid_weight"] * s for s in syntax_reward(completions)]

    def weighted_format(completions, **_):
        return [rewards["format_correct_weight"] * s for s in format_reward(completions)]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[weighted_exec, weighted_syntax, weighted_format],
        train_dataset=dataset,
        args=grpo_args,
    )

    # Auto-resume from latest checkpoint if any exist
    ckpt_dir = Path(t['output_dir'])
    latest = None
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            latest = str(ckpts[-1])
            logger.info(f"Resuming GRPO from {latest}")
    trainer.train(resume_from_checkpoint=latest)
    model.save_pretrained_merged(
        f"{t['output_dir']}-merged", tokenizer, save_method="merged_16bit"
    )
    logger.info(f"GRPO merged model → {t['output_dir']}-merged")


if __name__ == "__main__":
    main()
