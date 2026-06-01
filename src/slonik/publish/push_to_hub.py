from __future__ import annotations

import os
from pathlib import Path

import click
from huggingface_hub import HfApi, create_repo
from loguru import logger


@click.command()
@click.option("--model-dir", required=True, type=click.Path(exists=True))
@click.option("--repo", required=True, help="e.g. phaneendra/Slonik-7B")
@click.option("--private", is_flag=True)
@click.option("--card", default="src/slonik/publish/modelcard.md", type=click.Path(exists=True))
def main(model_dir: str, repo: str, private: bool, card: str) -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    create_repo(repo, repo_type="model", private=private, exist_ok=True, token=os.environ["HF_TOKEN"])

    card_target = Path(model_dir) / "README.md"
    card_target.write_text(Path(card).read_text())

    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo,
        repo_type="model",
        ignore_patterns=["*.tmp", "checkpoint-*", "global_step*"],
    )
    logger.info(f"Pushed {model_dir} → https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
