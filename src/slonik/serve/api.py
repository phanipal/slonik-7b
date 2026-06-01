from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import click
import httpx
import yaml
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from slonik.data.chatml import extract_sql, to_prompt
from slonik.serve.observability import maybe_langfuse
from slonik.serve.sql_validator import validate_postgres


class GenerateRequest(BaseModel):
    schema_: str = Field(..., alias="schema")
    question: str
    evidence: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    validate: bool = True

    class Config:
        populate_by_name = True


class GenerateResponse(BaseModel):
    sql: str
    raw: str
    valid: Optional[bool] = None
    parse_error: Optional[str] = None
    latency_ms: int


CONFIG: dict = {}
TOK = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global TOK
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(CONFIG["model"]["path"])
    logger.info(f"Tokenizer loaded from {CONFIG['model']['path']}")
    yield


app = FastAPI(title="Slonik-7B", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": CONFIG["model"]["served_name"]}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    import time

    prompt = to_prompt(req.schema_, req.question, TOK, req.evidence)
    upstream = f"http://{CONFIG['server']['host']}:{CONFIG['server']['port']}"
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{upstream}/v1/completions",
                json={
                    "model": CONFIG["model"]["served_name"],
                    "prompt": prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "stop": CONFIG["sampling"]["stop"],
                },
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"vLLM upstream error: {e}") from e

    completion = r.json()["choices"][0]["text"]
    sql = extract_sql(completion)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    valid = None
    parse_error = None
    if req.validate:
        ok, err = validate_postgres(sql)
        valid = ok
        parse_error = err

    with maybe_langfuse() as tracer:
        if tracer:
            tracer.event(
                name="generate",
                input={"question": req.question},
                output={"sql": sql, "valid": valid},
                metadata={"latency_ms": elapsed_ms},
            )

    return GenerateResponse(sql=sql, raw=completion, valid=valid, parse_error=parse_error, latency_ms=elapsed_ms)


@click.command()
@click.option("--config", default="configs/vllm_serve.yaml", type=click.Path(exists=True))
@click.option("--port", default=8001, type=int)
def main(config: str, port: int) -> None:
    global CONFIG
    CONFIG = yaml.safe_load(open(config))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import uvicorn
    uvicorn.run("slonik.serve.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
