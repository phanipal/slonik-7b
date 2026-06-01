# Slonik-7B

A PostgreSQL and SQLite text-to-SQL specialist fine-tuned from Qwen2.5-Coder-7B-Instruct via QLoRA SFT followed by GRPO with execution-based rewards.

**38.20%** on BIRD Mini-Dev PostgreSQL (beats GPT-4o at 34.44%) and **45.20%** on BIRD Mini-Dev SQLite — with a 7B model trained on a single laptop GPU in about 24 hours.

## Models on Hugging Face

| Repo | Description | Use case |
|---|---|---|
| [Phani-labs/Slonik-7B-GRPO](https://huggingface.co/Phani-labs/Slonik-7B-GRPO) | Best model — full precision safetensors (15 GB) | Production inference, further fine-tuning |
| [Phani-labs/Slonik-7B-GRPO-GGUF](https://huggingface.co/Phani-labs/Slonik-7B-GRPO-GGUF) | GGUF Q4_K_M / Q5_K_M / Q8_0 quantizations | Ollama, llama.cpp, LM Studio, local laptops |
| [Phani-labs/Slonik-7B-SFT](https://huggingface.co/Phani-labs/Slonik-7B-SFT) | SFT-only baseline (15 GB) | Ablation, reproducibility |

## Results

[BIRD Mini-Dev](https://bird-bench.github.io/) 500-example official benchmarks, execution accuracy:

| Model | BIRD-PG | BIRD-SQLite | Size |
|---|---:|---:|---|
| o3-mini | 47.78% | — | reasoning |
| Claude 3.7 Sonnet | 39.26% | — | proprietary |
| **Slonik-7B-GRPO** | **38.20%** | **45.20%** | **7B** |
| GPT-4o | 34.44% | — | proprietary |
| Slonik-7B-SFT | 33.20% | — | 7B |
| Qwen2.5-Coder-32B | 22.96% | — | 32B |
| Codestral 22B | 21.11% | — | 22B |
| Qwen2.5-Coder-7B (base) | 12.22% | — | 7B |

### SFT → GRPO trajectory on BIRD-PG

| Stage | Overall | Simple | Moderate | Challenging |
|---|---:|---:|---:|---:|
| Base Qwen2.5-Coder-7B | 12.22% | — | — | — |
| Slonik-7B-SFT | 33.20% | 48.6% | 29.6% | 19.6% |
| Slonik-7B-GRPO (500 steps) | 34.60% | 49.3% | 31.2% | 21.6% |
| **Slonik-7B-GRPO (2000 steps)** | **38.20%** | **56.1%** | **33.6%** | **23.5%** |

## Method

Two stages on a single RTX 5080 Laptop GPU (16 GB VRAM, Blackwell sm_120).

**Stage 1 — QLoRA SFT (8h 13min)** on 21,847 text-to-SQL pairs from BIRD, Spider, Gretel synthetic, and a custom PostgreSQL-focused synthesis set (pgvector, JSONB, full-text search, window functions, CTEs, array ops). LoRA rank 32, alpha 64, 4-bit NF4 base, LR 1e-5, `adamw_torch_fused`. Final eval_loss 0.290.

**Stage 2 — GRPO with execution rewards (16h)** using [GRPO](https://arxiv.org/abs/2402.03300) with three reward signals: weighted execution match against BIRD SQLite databases (1.0), syntax validity via sqlglot (0.2), and code-fence formatting (0.1). 2000 steps, num_generations=2.

Total external cost: about $3 (DeepSeek API for synthetic data). Everything else ran locally.

## Project structure

```
slonik-7b/
├── configs/                  # YAML configs for SFT, GRPO, vLLM serving
├── docker/                   # Dockerfiles for training and serving
├── notebooks/                # Exploration and results notebooks
├── results/                  # BIRD eval result JSONLs (small, kept in repo)
├── scripts/                  # CLI entry points
│   ├── train_sft.py          # Stage 1 trainer wrapper
│   ├── train_grpo.py         # Stage 2 trainer wrapper
│   ├── eval_bird_pg.py       # BIRD-PG benchmark runner
│   ├── eval_bird_sqlite.py   # BIRD-SQLite benchmark runner
│   ├── prepare_bird.py       # BIRD nested-archive extractor
│   ├── synthesize_pg_modern.py  # DeepSeek-based PG synthesis
│   └── helpers/              # Shell utilities for monitoring runs
├── src/slonik/               # Library code
│   ├── data/                 # Dataset preparation
│   ├── training/             # SFT, GRPO, reward functions, exec sandbox
│   ├── eval/                 # Benchmark runners
│   ├── serve/                # FastAPI inference server
│   └── publish/              # HF upload utilities
├── tests/                    # Unit tests
├── pyproject.toml
└── requirements.txt
```

## Reproduce

### 1. Install

```bash
git clone https://github.com/Phani-labs/slonik-7b
cd slonik-7b
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare data

```bash
python scripts/prepare_bird.py            # extracts BIRD train + dev
python -m slonik.data.prepare_spider      # Spider via HF datasets
python -m slonik.data.prepare_gretel      # Gretel synthetic
python scripts/synthesize_pg_modern.py    # custom PG-Modern synth (needs DEEPSEEK_API_KEY)
```

### 3. Train

```bash
python scripts/train_sft.py    # ~8h on RTX 5080 Laptop
python scripts/train_grpo.py   # ~16h without vLLM, ~3h with vLLM
```

### 4. Evaluate

Set up Postgres + pgvector with the BIRD dump:

```bash
docker run -d --name slonik-pg -e POSTGRES_PASSWORD=slonik -p 5432:5432 pgvector/pgvector:pg16
docker exec slonik-pg psql -U postgres -c "CREATE EXTENSION vector;"
docker exec -i slonik-pg psql -U postgres -d postgres -f data/raw/minidev_pg/minidev/MINIDEV_postgresql/BIRD_dev.sql

python scripts/eval_bird_pg.py --model checkpoints/grpo-merged
python scripts/eval_bird_sqlite.py --model checkpoints/grpo-merged
```

## Stack

- **Base model**: Qwen2.5-Coder-7B-Instruct
- **Training**: Unsloth + TRL (SFT, GRPO) on PyTorch 2.11 with CUDA 12.8
- **Quantization**: 4-bit NF4 via bitsandbytes
- **Eval**: psycopg + sqlite3 against real databases
- **Quantization for GGUF**: llama.cpp `Q4_K_M`, `Q5_K_M`, `Q8_0`

## License

Apache 2.0 (same as the base model).

## Author

Phani Paladugu — Senior AI/ML Engineer

- Hugging Face: [Phani-labs](https://huggingface.co/Phani-labs)
