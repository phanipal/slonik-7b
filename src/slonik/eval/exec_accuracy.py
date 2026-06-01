from __future__ import annotations

from collections import Counter
from pathlib import Path

import click
import json
from rich.console import Console
from rich.table import Table


def difficulty_breakdown(rows: list[dict]) -> dict[str, float]:
    by_diff: dict[str, list[bool]] = {}
    for r in rows:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(r["correct"])
    return {k: sum(v) / len(v) if v else 0.0 for k, v in by_diff.items()}


def error_taxonomy(rows: list[dict]) -> Counter:
    bucket = Counter()
    for r in rows:
        if r["correct"]:
            continue
        err = (r.get("error") or "").lower()
        if not err:
            bucket["wrong_result"] += 1
        elif "syntax" in err or "parse" in err:
            bucket["syntax_error"] += 1
        elif "no such column" in err or "does not exist" in err:
            bucket["missing_column"] += 1
        elif "no such table" in err:
            bucket["missing_table"] += 1
        elif "timeout" in err:
            bucket["timeout"] += 1
        elif "ambiguous" in err:
            bucket["ambiguous_column"] += 1
        else:
            bucket["other"] += 1
    return bucket


@click.command()
@click.argument("results_path", type=click.Path(exists=True))
def main(results_path: str) -> None:
    data = json.loads(Path(results_path).read_text())
    rows = data["rows"]
    summary = data["summary"]

    console = Console()
    t = Table(title=f"{summary.get('model', 'model')} — execution accuracy")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("Total", str(summary["n_total"]))
    t.add_row("Correct", str(summary["n_correct"]))
    t.add_row("Accuracy", f"{summary['execution_accuracy']:.2%}")
    console.print(t)

    errs = error_taxonomy(rows)
    if errs:
        et = Table(title="Error taxonomy")
        et.add_column("Category"); et.add_column("Count")
        for k, v in errs.most_common():
            et.add_row(k, str(v))
        console.print(et)


if __name__ == "__main__":
    main()
