"""One thread owns the models, and it is not the event loop's.

``docs/SERVER-HANDBOOK.md`` §Concurrent Requests promises one model operation at a time.
That used to be enforced by the event loop being blocked for the whole generation — which
is also why ``GET /health`` went unanswered under load:
measured 2026-09-13, a 9.1 s non-streaming completion swallowed two of three probes and
answered the third after 2.548 s, while a streaming one answered 40 of 40 in 16-86 ms.
A supervisor reads the silence as *dead* and restarts a working server.

So every MLX touch — loading, a batch generation, one streaming step, a vision chunk, a
transcription, the cleanup — is submitted here instead, and this module owns exactly one
worker thread. The single worker is what still serializes once the loop no longer does
it by accident; no lock is needed, because two things can never be in flight at once.

**The worker is not an optimization, and a pool would be wrong.** An ``mx.array`` carries
the stream it was made on, and a model loaded on the *main* thread raises
``RuntimeError: There is no Stream(gpu, N) in current thread.`` when it is generated with
anywhere else. Measured 2026-09-14 against the pinned mlx-lm: main-thread load + worker
generation fails, worker load + generation on any worker succeeds, so it is the main
thread that poisons, not the switch. It is also silently model-dependent —
``Qwen2.5-0.5B-Instruct-4bit`` survives it, ``Llama-3.2-3B-Instruct-4bit`` does not — which
is why a green smoke run on the small model proves nothing here. The rule that follows is
simple and has no exception: **nothing that touches a model runs on the main thread**, the
lifespan preload included.

``contextvars`` are copied into the worker the way ``asyncio.to_thread`` does it, so
``request_id`` still reaches the generation's own log line.

Neither closed nor opened here: nothing holds a runner for the length of a stream. The
worker is free between two steps, so a request for another model is served there, unloads
the runner, and the stream fails at its next step — measured 2026-09-14, after 13 tokens.
On the loop the same could happen between two yields; it closes with a request-bound
generation state.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Callable, Iterator

# Lazily started: importing this module costs no thread, so the CLI and the test
# collection never grow one.
_model_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlxk-model")


async def in_worker(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run one blocking model call on the model thread, and alone.

    Arguments pass through untouched, and whatever ``fn`` raises is re-raised at the
    ``await`` — so an ``HTTPException`` or ``ContextLengthExceeded`` still lands in the
    caller's own ``except``.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(
        _model_thread, functools.partial(ctx.run, fn, *args, **kwargs)
    )


async def drive(iterator: Iterator[Any]) -> AsyncGenerator[Any, None]:
    """Step a blocking iterator on the model thread, yielding what it produces.

    One round trip per item, which is microseconds against a decode step's tens of
    milliseconds — and the loop is free in between, which is the point. A caller that
    stops early leaves the iterator suspended, exactly as a plain ``for`` loop would.
    """
    done = object()
    while True:
        item = await in_worker(next, iterator, done)
        if item is done:
            return
        yield item


def shutdown_model_thread() -> None:
    """Retire the model thread. Called from the lifespan hook after the cache is freed."""
    _model_thread.shutdown(wait=True)
