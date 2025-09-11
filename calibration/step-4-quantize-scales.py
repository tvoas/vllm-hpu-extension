###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import os
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["PT_HPU_LAZY_MODE"] = "1"

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
import torch
import argparse
import asyncio


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--expert-parallel", action="store_true", default=False)
    parser.add_argument("--max-num-prefill-seqs", type=int, default=None)
    parser.add_argument("--distributed-executor-backend", choices=["mp", "ray"], default="mp", 
                        help="For single node calibration use the default multiprocessing backend. For multi-node calibration use ray backend")

    args = parser.parse_args()

    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        enforce_eager=args.enforce_eager,
        dtype=torch.bfloat16,
        max_num_prefill_seqs=args.max_num_prefill_seqs,
        max_model_len=128,
        trust_remote_code=True,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=args.expert_parallel,
    )

    use_async = args.pipeline_parallel_size > 1
    if not use_async:
        # Synchronous (PP=1)
        llm = vllm.LLM(
            model=args.model,
            dtype=torch.bfloat16,
            enforce_eager=args.enforce_eager,
            max_num_seqs=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            max_model_len=args.max_model_len,
            max_num_prefill_seqs=args.max_num_prefill_seqs,
            trust_remote_code=True,
            distributed_executor_backend=args.distributed_executor_backend,
            enable_expert_parallel=args.expert_parallel,
        )
    else:
        # Async engine required for PP>1
        async_engine_args = AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            dtype="bfloat16",
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            max_num_prefill_seqs=args.max_num_prefill_seqs,
            trust_remote_code=True,
            distributed_executor_backend=args.distributed_executor_backend,
            enable_expert_parallel=args.expert_parallel,
        )
        async def _build_async():
            async with build_async_engine_client_from_engine_args(
                async_engine_args,
                disable_frontend_multiprocessing=False,
            ) as async_llm:
                pass
        asyncio.run(_build_async())

    if not use_async:
        # Skip shutdown when VLLM_USE_V1 is set to "1"
        if not os.environ.get("VLLM_USE_V1") or os.environ.get("VLLM_USE_V1") != "1":
            llm.llm_engine.model_executor.shutdown()
