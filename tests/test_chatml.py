from __future__ import annotations

from slonik.data.chatml import Example, extract_sql, to_chatml, to_prompt


class _Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def test_chatml_roundtrip():
    ex = Example(schema="CREATE TABLE t(id INT);", question="count rows", sql="SELECT COUNT(*) FROM t")
    text = to_chatml(ex, _Tok())
    assert "CREATE TABLE" in text
    assert "count rows" in text
    assert "SELECT COUNT(*)" in text


def test_to_prompt_adds_generation_marker():
    p = to_prompt("CREATE TABLE t(id INT);", "count rows", _Tok())
    assert p.endswith("assistant\n")


def test_extract_sql_handles_codeblock():
    raw = "```sql\nSELECT 1\n```\nextra"
    assert extract_sql(raw) == "SELECT 1;"


def test_extract_sql_no_codeblock():
    assert extract_sql("SELECT 1 ;") == "SELECT 1;"


def test_evidence_appended():
    ex = Example(schema="t", question="q?", sql="SELECT 1", evidence="use COUNT")
    text = to_chatml(ex, _Tok())
    assert "Hints: use COUNT" in text
