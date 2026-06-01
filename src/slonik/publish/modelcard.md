---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
language:
- en
library_name: transformers
tags:
- text-to-sql
- postgresql
- code-generation
- qlora
- grpo
- pgvector
- jsonb
datasets:
- birdsql/bird23-train-filtered
- xlangai/spider
- gretelai/synthetic_text_to_sql
pipeline_tag: text-generation
---

# Slonik-7B

PostgreSQL text-to-SQL specialist. Qwen2.5-Coder-7B fine-tuned with QLoRA + GRPO on BIRD-SQL, Spider 2.0, and 2K curated PostgreSQL-modern examples covering CTEs, window functions, JSONB operators, pgvector similarity, and full-text search.

## Eval

| Benchmark | Base | SFT | SFT+GRPO |
|---|---|---|---|
| BIRD-dev (PG, exec acc) | 38.2 | 54.7 | **62.4** |
| Spider 2.0 val | 64.1 | 71.3 | **74.8** |
| PG-Modern (custom 500-q) | 41.5 | 68.9 | **76.2** |

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="phaneendra/Slonik-7B", dtype="bfloat16")
params = SamplingParams(temperature=0.0, max_tokens=512)

prompt = """### Schema
CREATE TABLE orders (id INT, customer_id INT, total NUMERIC, created_at TIMESTAMPTZ);

### Question
Total revenue per month last year, only months above 100k.

### SQL
"""
print(llm.generate([prompt], params)[0].outputs[0].text)
```

## Training

- Base: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Hardware: RTX 5080 mobile (16GB VRAM) for SFT, A100 80GB for GRPO peak
- SFT: QLoRA rank=32, 3 epochs, lr=2e-4, ~28K examples, `train_on_responses_only=True`
- GRPO: 4 rollouts per prompt, execution-feedback reward against sandboxed PG/SQLite, lr=5e-6, beta=0.04

## Limitations

- Trained primarily on read-only `SELECT` queries; not aligned for `INSERT/UPDATE/DELETE`.
- BIRD/Spider databases are SQLite-origin; some PG-only features (window frame `EXCLUDE`, `LATERAL`) are under-represented in training data.
- Not safe for direct execution against production databases without query review.

## License

Apache 2.0 (inherits from Qwen2.5-Coder-7B-Instruct base license).
