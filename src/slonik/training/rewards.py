from __future__ import annotations

from pathlib import Path

from slonik.data.chatml import extract_sql
from slonik.training.exec_sandbox import (
    ExecResult,
    execute_sqlite,
    is_valid_syntax,
    results_equal,
)


def format_reward(completions: list[str], **_: object) -> list[float]:
    out = []
    for c in completions:
        score = 0.0
        if "```sql" in c and "```" in c.split("```sql", 1)[1]:
            score = 1.0
        elif c.strip().lower().startswith(("select ", "with ")):
            score = 0.5
        out.append(score)
    return out


def syntax_reward(completions: list[str], **_: object) -> list[float]:
    return [1.0 if is_valid_syntax(extract_sql(c)) else 0.0 for c in completions]


def make_exec_reward(databases_root: str | Path, dialect: str = "postgres", timeout: float = 5.0):
    db_root = Path(databases_root)

    def reward(completions: list[str], db_id: list[str], gold_sql: list[str], **_: object) -> list[float]:
        scores = []
        for comp, did, gold in zip(completions, db_id, gold_sql, strict=True):
            pred = extract_sql(comp)
            db_path = db_root / did / f"{did}.sqlite"
            if not db_path.exists():
                scores.append(0.0)
                continue
            gold_res: ExecResult = execute_sqlite(db_path, gold, timeout=timeout)
            pred_res: ExecResult = execute_sqlite(db_path, pred, timeout=timeout)
            if not gold_res.ok:
                scores.append(0.0)
                continue
            if not pred_res.ok:
                scores.append(0.0)
                continue
            scores.append(1.0 if results_equal(pred_res.rows, gold_res.rows) else 0.0)
        return scores

    return reward


def combine(weights: dict[str, float]):
    def combined(completions: list[str], **kwargs: object) -> list[float]:
        f = format_reward(completions)
        s = syntax_reward(completions)
        e = kwargs.get("exec_scores") or [0.0] * len(completions)
        total = []
        for fi, si, ei in zip(f, s, e, strict=True):
            total.append(
                weights["exec_match_weight"] * ei
                + weights["syntax_valid_weight"] * si
                + weights["format_correct_weight"] * fi
            )
        return total
    return combined
