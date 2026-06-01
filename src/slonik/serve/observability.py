from __future__ import annotations

import os
from contextlib import contextmanager


class _NullTracer:
    def event(self, **_: object) -> None: ...
    def trace(self, **_: object) -> None: ...


@contextmanager
def maybe_langfuse():
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        yield None
        return
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        yield client
        client.flush()
    except Exception:
        yield None
