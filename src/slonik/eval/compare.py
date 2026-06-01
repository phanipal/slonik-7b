from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table


@click.command()
@click.option("--base", "base_path", required=True, type=click.Path(exists=True))
@click.option("--sft", "sft_path", required=True, type=click.Path(exists=True))
@click.option("--grpo", "grpo_path", type=click.Path(exists=True), default=None)
@click.option("--out", default="outputs/comparison.json", type=click.Path())
def main(base_path: str, sft_path: str, grpo_path: str | None, out: str) -> None:
    base = json.loads(Path(base_path).read_text())
    sft = json.loads(Path(sft_path).read_text())
    grpo = json.loads(Path(grpo_path).read_text()) if grpo_path else None

    table = Table(title="Model comparison")
    table.add_column("Variant"); table.add_column("N"); table.add_column("Correct"); table.add_column("Accuracy")
    for label, data in (("Base", base), ("SFT", sft)) + ((("SFT+GRPO", grpo),) if grpo else ()):
        s = data["summary"]
        table.add_row(label, str(s["n_total"]), str(s["n_correct"]), f"{s['execution_accuracy']:.2%}")
    Console().print(table)

    base_idx = {(r["db_id"], r["question"]): r for r in base["rows"]}
    sft_idx = {(r["db_id"], r["question"]): r for r in sft["rows"]}
    fixed = []
    regressed = []
    for key, sft_r in sft_idx.items():
        base_r = base_idx.get(key)
        if not base_r:
            continue
        if not base_r["correct"] and sft_r["correct"]:
            fixed.append({**sft_r, "base_pred": base_r["pred"]})
        elif base_r["correct"] and not sft_r["correct"]:
            regressed.append({**sft_r, "base_pred": base_r["pred"]})

    summary = {
        "fixed_by_sft": len(fixed),
        "regressed_by_sft": len(regressed),
        "examples_fixed": fixed[:20],
        "examples_regressed": regressed[:20],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2))
    Console().print(f"Fixed: {len(fixed)} | Regressed: {len(regressed)} | written → {out}")


if __name__ == "__main__":
    main()
