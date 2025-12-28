import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncGenerator, cast

import nest_asyncio
import peft
import torch
import unsloth  # type: ignore
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import (
    GenerationMixin,
    PreTrainedModel,
)
from trl import GRPOConfig, GRPOTrainer

# vLLM 0.13+ V1 engine imports
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from ..dev.model import InternalModelConfig
from .train import gc_and_empty_cuda_cache

if TYPE_CHECKING:
    from .service import TrainInputs

nest_asyncio.apply()


class CausallLM(PreTrainedModel, GenerationMixin):
    vllm_engine: AsyncLLM  # V1 engine type


class ModelState:
    """
    A class responsible for initializing and holding references to the model and related state.
    """

    def __init__(self, config: InternalModelConfig) -> None:
        import vllm.envs as envs

        # Force V1 engine (V0 removed in vLLM 0.11+)
        envs.VLLM_USE_V1 = True

        # We can't use expandable segments with sleep mode
        enable_sleep_mode = config.get("engine_args", {}).get(
            "enable_sleep_mode", False
        )
        if enable_sleep_mode:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

        # Initialize Unsloth model
        # NOTE: We have to patch empty_cache with a no-op during model initialization
        # to avoid an allocator error.
        empty_cache = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        from_engine_args = AsyncLLM.from_engine_args

        # NOTE: We also have to patch from_engine_args to control the engine args
        # that are passed to the engine constructor.
        def _from_engine_args(
            engine_args: AsyncEngineArgs, *args: Any, **kwargs: Any
        ) -> AsyncLLM:
            return from_engine_args(
                replace(engine_args, **config.get("engine_args", {})), *args, **kwargs
            )

        AsyncLLM.from_engine_args = _from_engine_args

        self.model, self.tokenizer = cast(
            tuple[CausallLM, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**config.get("init_args", {})),
        )
        AsyncLLM.from_engine_args = from_engine_args
        torch.cuda.empty_cache = empty_cache
        torch.cuda.empty_cache()
        self.vllm = vLLMState(self.model.vllm_engine, enable_sleep_mode)
        # Initialize PEFT model
        self.peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                self.model, **config.get("peft_args", {})
            ),
        )
        self.lora_model = cast(peft.tuners.lora.LoraModel, self.peft_model.base_model)
        # Initialize trainer
        data = {"prompt": ""}
        self.trainer = GRPOTrainer(
            model=self.peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**config.get("trainer_args", {})),  # type: ignore
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=self.tokenizer,
        )
        self.inputs_queue = asyncio.Queue["TrainInputs"]()

        # Patch trainer _prepare_inputs()
        def _async_prepare_inputs(*_, **__) -> dict[str, torch.Tensor]:
            async def get_inputs() -> "TrainInputs":
                return await self.inputs_queue.get()

            # Force otherwise synchronous _prepare_inputs() to yield
            # with nested asyncio.run() call
            inputs = asyncio.run(get_inputs())

            return cast(dict[str, torch.Tensor], inputs)

        self.trainer._prepare_inputs = _async_prepare_inputs


class vLLMState:
    def __init__(self, async_engine: AsyncLLM, enable_sleep_mode: bool) -> None:
        from ..vllm import (
            patch_allocator,
            patch_get_lora_tokenizer_async,
            patch_lora_request,
        )

        if enable_sleep_mode:
            patch_allocator()
        # Unsloth patches
        patch_lora_request()
        patch_get_lora_tokenizer_async()
        self.async_engine = async_engine
        self.enable_sleep_mode = enable_sleep_mode
        # V1 engine doesn't expose driver_worker the same way
        # Worker access is handled differently in V1

    @asynccontextmanager
    async def train_mode(self) -> AsyncGenerator[None, None]:
        """
        A context manager pauses the vLLM engine and frees memory for training.
        Note: V1 engine has different sleep/wake semantics than V0.
        """
        if not self.enable_sleep_mode:
            yield
            return
        try:
            # V1 engine sleep mode
            await self.async_engine.sleep(level=2)
            gc_and_empty_cuda_cache()
            yield
        finally:
            gc_and_empty_cuda_cache()
            await asyncio.sleep(0.1)
            await self.async_engine.wake_up()
