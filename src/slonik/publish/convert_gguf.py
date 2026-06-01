from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import click
from loguru import logger


QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]


def _ensure_llamacpp(target: Path) -> Path:
    if target.exists() and (target / "convert_hf_to_gguf.py").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp", str(target)], check=True)
    subprocess.run(["pip", "install", "-r", str(target / "requirements.txt")], check=True)
    return target


@click.command()
@click.option("--model", required=True, type=click.Path(exists=True))
@click.option("--out-dir", default="outputs/gguf", type=click.Path())
@click.option("--llamacpp", default="vendor/llama.cpp", type=click.Path())
@click.option("--quants", default=",".join(QUANT_TYPES))
def main(model: str, out_dir: str, llamacpp: str, quants: str) -> None:
    llcpp = _ensure_llamacpp(Path(llamacpp))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fp16_path = out / "model-f16.gguf"
    subprocess.run([
        "python", str(llcpp / "convert_hf_to_gguf.py"), model,
        "--outfile", str(fp16_path), "--outtype", "f16",
    ], check=True)

    quantize_bin = next(
        (p for p in [llcpp / "llama-quantize", llcpp / "build" / "bin" / "llama-quantize"] if p.exists()),
        None,
    )
    if quantize_bin is None:
        logger.error("llama-quantize binary not found. Build llama.cpp first: cd vendor/llama.cpp && cmake -B build && cmake --build build")
        return

    for q in quants.split(","):
        q = q.strip()
        target = out / f"slonik-7b-{q.lower()}.gguf"
        subprocess.run([str(quantize_bin), str(fp16_path), str(target), q], check=True)
        logger.info(f"  {q}: {target.stat().st_size / 1e9:.2f} GB → {target}")

    shutil.copy(fp16_path, out / "slonik-7b-f16.gguf")
    logger.info(f"Done. Files in {out}")


if __name__ == "__main__":
    main()
