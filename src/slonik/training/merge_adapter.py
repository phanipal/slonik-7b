from __future__ import annotations

from pathlib import Path

import click
import torch
from loguru import logger
from unsloth import FastLanguageModel


@click.command()
@click.option("--base", required=True, help="Base model name or path")
@click.option("--adapter", required=True, help="LoRA adapter checkpoint")
@click.option("--out", required=True, type=click.Path())
@click.option("--method", default="merged_16bit", type=click.Choice(["merged_16bit", "merged_4bit", "lora"]))
def main(base: str, adapter: str, out: str, method: str) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,
        max_seq_length=2048,
        load_in_4bit=method == "merged_4bit",
        dtype=torch.bfloat16,
    )
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(out, tokenizer, save_method=method)
    logger.info(f"Merged ({method}) → {out}")


if __name__ == "__main__":
    main()
