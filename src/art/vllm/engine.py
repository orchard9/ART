"""Engine and worker management for vLLM."""

import asyncio
import contextlib
import contextvars
import os
import time
from dataclasses import replace
from typing import Any, Callable, Coroutine, Generator, ParamSpec, TypeVar, cast

import cloudpickle
import vllm
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_worker import Worker
# Note: AsyncLLMEngine (V0) removed in vLLM 0.11+

from .patches import patch_allocator


async def get_llm(args: vllm.AsyncEngineArgs) -> AsyncLLM:
    """
    Create an AsyncLLM engine with model download and patches applied.

    Args:
        args: The engine arguments including model name and configuration.

    Returns:
        A configured AsyncLLM instance.
    """
    # Download model only if it's not a local path
    if not os.path.exists(args.model):
        process = await asyncio.create_subprocess_shell(
            f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {args.model}"
        )
        await process.wait()

    # Make sure we are using the V1 engine
    import vllm.envs as envs

    envs.VLLM_USE_V1 = True

    llm = AsyncLLM.from_engine_args(
        replace(
            args,
            worker_extension_cls=f"{WorkerExtension.__module__}.{WorkerExtension.__qualname__}",
            enable_sleep_mode=True,
        )
    )
    await run_on_workers(llm, patch_allocator)
    return llm


def create_engine_pause_and_resume_functions(
    engine: AsyncLLM,
) -> tuple[
    Callable[[], Coroutine[Any, Any, None]], Callable[[], Coroutine[Any, Any, None]]
]:
    """
    DEPRECATED: V1 engine uses sleep/wake_up instead of pause/resume.

    This function is kept for backward compatibility but should not be used
    with vLLM 0.11+. Use AsyncLLM.sleep() and AsyncLLM.wake_up() instead.

    Args:
        engine: The AsyncLLM engine.

    Returns:
        A tuple of (pause_engine, resume_engine) async functions that are no-ops.
    """
    import warnings
    warnings.warn(
        "create_engine_pause_and_resume_functions is deprecated for V1 engine. "
        "Use engine.sleep() and engine.wake_up() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    async def pause_engine() -> None:
        # V1 engine doesn't have the same pause semantics
        # Use sleep() instead for memory management
        pass

    async def resume_engine() -> None:
        # V1 engine doesn't have the same resume semantics
        # Use wake_up() instead
        pass

    return pause_engine, resume_engine


P = ParamSpec("P")
R = TypeVar("R")


async def run_on_workers(
    llm: AsyncLLM, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> list[R]:
    """
    Run a function on all workers in a distributed setup.

    Args:
        llm: The AsyncLLM instance with workers.
        func: The function to run on each worker.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        List of results from each worker.
    """
    return await llm.collective_rpc(
        "run", args=(cloudpickle.dumps(func), *args), kwargs=kwargs
    )


# Context variable to hold the current worker
_worker: contextvars.ContextVar["ExtendedWorker"] = contextvars.ContextVar("worker")


def get_worker() -> "ExtendedWorker":
    """Get the current worker instance"""
    return _worker.get()


class WorkerExtension:
    """Extension for running arbitrary functions on vLLM workers."""

    def run(self, pickled_func: bytes, *args: Any, **kwargs: Any) -> Any:
        func = cloudpickle.loads(pickled_func)
        token = _worker.set(cast(ExtendedWorker, self))
        try:
            return func(*args, **kwargs)
        finally:
            _worker.reset(token)

    @contextlib.contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        from vllm.v1.worker.gpu_worker import logger

        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        logger.info(f"{name}: {end_time - start_time:.2f} seconds")


class ExtendedWorker(Worker, WorkerExtension):
    pass
