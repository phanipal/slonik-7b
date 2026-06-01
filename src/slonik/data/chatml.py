from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SYSTEM = (
    "You are a PostgreSQL expert. Given a schema and a question, write a single "
    "syntactically valid PostgreSQL query that answers the question. Use modern "
    "Postgres features (CTEs, window functions, JSONB operators, pgvector, full-text "
    "search) when appropriate. Output only the SQL inside a ```sql block."
)


@dataclass(slots=True)
class Example:
    schema: str
    question: str
    sql: str
    evidence: str = ""
    db_id: str = ""
    dialect: str = "postgresql"


def to_chatml(ex: Example, tokenizer) -> str:
    user = ex.question if not ex.evidence else f"{ex.question}\n\nHints: {ex.evidence}"
    user_block = f"### Schema\n{ex.schema.strip()}\n\n### Question\n{user.strip()}"
    assistant_block = f"```sql\n{ex.sql.strip()}\n```"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_block},
        {"role": "assistant", "content": assistant_block},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def to_prompt(schema: str, question: str, tokenizer, evidence: str = "") -> str:
    user = question if not evidence else f"{question}\n\nHints: {evidence}"
    user_block = f"### Schema\n{schema.strip()}\n\n### Question\n{user.strip()}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_block},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_dataset(rows: Iterable[Example], tokenizer) -> list[dict]:
    return [{"text": to_chatml(r, tokenizer)} for r in rows]


def extract_sql(generated: str) -> str:
    text = generated.strip()
    if "```sql" in text:
        text = text.split("```sql", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip().rstrip(";") + ";"
