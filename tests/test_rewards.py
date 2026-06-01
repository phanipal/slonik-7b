from __future__ import annotations

from slonik.training.rewards import format_reward, syntax_reward


def test_format_reward_full_codeblock():
    s = format_reward(["```sql\nSELECT 1\n```"])
    assert s == [1.0]


def test_format_reward_partial_select():
    s = format_reward(["SELECT 1 FROM t"])
    assert s == [0.5]


def test_format_reward_garbage():
    s = format_reward(["lorem ipsum"])
    assert s == [0.0]


def test_syntax_reward_valid_postgres():
    s = syntax_reward(["```sql\nSELECT id FROM users WHERE id = 1\n```"])
    assert s == [1.0]


def test_syntax_reward_invalid():
    s = syntax_reward(["```sql\nSELEC FROM nope WHER\n```"])
    assert s == [0.0]
