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
from vllm.utils import merge_async_iterators
from tqdm import tqdm
import torch
import pandas as pd
import time
import argparse
import asyncio

def get_ds(args):
    print(f"Loading dataset: {args.dataset}")
    ds = pd.read_pickle(args.dataset)

    if args.max_dataset_samples:
        ds = ds.head(args.max_dataset_samples)

    return ds

def get_dataset(args):
    def reset_seed(seed=42):
        import torch
        import random
        import numpy as np

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_prompt_token_ids(model_path, prompts, max_length=1024):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_token_ids = []
        for prompt in prompts:
            tokens = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            if len(tokens.input_ids[0]) < max_length:
                continue
            prompt_token_ids.append([x.item() for x in tokens.input_ids[0]])
        return prompt_token_ids

    def get_prompts(
        model_name,
        dataset_name="NeelNanda/pile-10k",
        num_samples=512,
        least_tokens=1024,
    ):
        print(
            f"Loading {num_samples} samples with at least {least_tokens} tokens "
            f"from {dataset_name} for model {model_name}..."
        )
        from datasets import load_dataset
        from tqdm import tqdm
        import transformers

        seed = 42

        reset_seed(seed)

        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.shuffle(seed=seed)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        num_sample = 0
        samples_lst = []
        for data in tqdm(dataset):
            prompt = data["text"]
            tokens = tokenizer(prompt, return_tensors="pt")
            if len(tokens.input_ids[0]) < least_tokens:
                continue
            num_sample += 1
            samples_lst.append(prompt)
            if num_sample >= num_samples:
                break
        return samples_lst

    least_tokens = args.sample_len
    num_samples = args.max_dataset_samples
    try:
        prompts = get_prompts(
            args.model,
            dataset_name=args.dataset,
            num_samples=num_samples,
            least_tokens=least_tokens,
        )
    except:
        raise RuntimeError(f"Failed to load prompts from dataset {args.dataset}.")
    prompt_token_ids = get_prompt_token_ids(
        args.model, prompts, least_tokens
    )
    print(f"Got {len(prompts)} prompts, length of first prompt: {len(prompt_token_ids[0])}.")
    gt = None
    return prompts, prompt_token_ids, gt


def generate_responses(llm, input_batch, args, sampling_params=None, prompt_token_ids=None):
    responses = llm.generate(
        input_batch, sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=True
    )

    total_input_tokens = 0
    total_generated_tokens = 0

    for response in responses:
        if args.verbose:
            print(
                f"Prompt: {response.prompt};\nAnswer: {response.outputs[0].text}\n")
        total_input_tokens += len(response.prompt_token_ids)
        total_generated_tokens += len(response.outputs[0].token_ids)

async def generate_responses_async(llm, input_batch, args, sampling_params):
    if not input_batch:
        return
    # Create one generator per prompt (like benchmark_throughput.py)
    generators = []
    for i, prompt in enumerate(input_batch):
        gen = llm.generate(prompt, sampling_params, request_id=f"cal-{i}")
        generators.append(gen)

    pbar = tqdm(total=len(generators), disable=not args.verbose and False)
    finished = set()
    async for idx, res in merge_async_iterators(*generators):
        if res is None:
            continue
        # res.finished exists; if not, fallback to last chunk heuristic (outputs len >0 and no next tokens)
        done = getattr(res, "finished", False)
        if done and idx not in finished:
            finished.add(idx)
            if args.verbose and res.outputs:
                print(f"Prompt[{idx}] done. Generated tokens={len(res.outputs[0].token_ids)}")
            pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-dataset-samples", type=int, default=0)
    parser.add_argument("--max-num-prefill-seqs", type=int, default=None)
    parser.add_argument("--expert-parallel", action="store_true", default=False)
    parser.add_argument(
        "--auto-process-dataset",
        action="store_true",
        default=False,
        help="Automatically generate a calibration dataset based on the provided dataset name.",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--sample-len", type=int, default=1024, help="Minimum number of tokens in each sample.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--distributed-executor-backend", choices=["mp", "ray"], default="mp", 
                        help="For single node calibration use the default multiprocessing backend. For multi-node calibration use ray backend")

    args = parser.parse_args()
    if not args.auto_process_dataset:
        calibration_ds = get_ds(args)

    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1, max_tokens=args.max_tokens)

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
    
    def _run_sync():
        if not args.auto_process_dataset:
            input_batch = []
            dataset_len = len(calibration_ds)
            batch_num = (dataset_len + args.batch_size - 1) // args.batch_size
            batch_done = 0
            for i, (_, row) in enumerate(calibration_ds.iterrows()):
                input_batch.append(row["input"])
                if len(input_batch) == args.batch_size:
                    t_start = time.perf_counter()
                    generate_responses(llm, input_batch, args)
                    t_end = time.perf_counter()
                    batch_done += 1
                    print(f"Batch finished: {(batch_done*args.batch_size)}/{calibration_ds.shape[0]} samples done; "
                          f"ETA: {int((t_end - t_start) * (batch_num - batch_done) // 60)} min")
                    input_batch = []
            if input_batch:
                generate_responses(llm, input_batch, args)
                print(f"Last batch finished: {dataset_len}/{calibration_ds.shape[0]} samples done")
        else:
            prompts, prompt_token_ids, gt = get_dataset(args)
            generate_responses(
                llm=llm,
                input_batch=None,
                args=args,
                sampling_params=sampling_params,
                prompt_token_ids=prompt_token_ids,
            )

    async def _run_async():
        async with build_async_engine_client_from_engine_args(
            async_engine_args,
            disable_frontend_multiprocessing=False,
        ) as async_llm:
            try:
                if not args.auto_process_dataset:
                    # Stream batches exactly like sync path: slice dataset into batches
                    dataset_len = len(calibration_ds)
                    for start in range(0, dataset_len, args.batch_size):
                        batch_prompts = calibration_ds.iloc[start:start + args.batch_size]["input"].tolist()
                        t_start = time.perf_counter()
                        await generate_responses_async(async_llm, batch_prompts, args, sampling_params)
                        t_end = time.perf_counter()
                        done = min(start + args.batch_size, dataset_len)
                        remaining_batches = max((dataset_len - done + args.batch_size - 1) // args.batch_size, 0)
                        eta_min = int((t_end - t_start) * remaining_batches // 60)
                        print(f"Batch finished: {done}/{dataset_len} samples done; ETA: {eta_min} min")
                else:
                    # ORIGINAL SYNC MODE: input_batch=None + prompt_token_ids (truncated)
                    # ASYNC MODE: must supply strings; recreate truncated prompts from token ids.
                    prompts, prompt_token_ids, _ = get_dataset(args)
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                    # prompt_token_ids already truncated to sample_len; decode them to safe prompts.
                    truncated_prompts = [tokenizer.decode(tids, skip_special_tokens=False)
                                         for tids in prompt_token_ids]
                    # Use truncated prompts (NOT the raw 'prompts' list) to avoid exceeding max_model_len.
                    await generate_responses_async(async_llm, truncated_prompts, args, sampling_params)
                await asyncio.sleep(0.05)
            finally:
                pass

    if use_async:
        # Manual loop for cleaner shutdown
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_async())
            # Cancel any leftover tasks to avoid late callbacks
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            loop.close()
    else:
        _run_sync()
    
    if not use_async:
        # Skip shutdown when VLLM_USE_V1 is set to "1"
        if not os.environ.get("VLLM_USE_V1") or os.environ.get("VLLM_USE_V1") != "1":
            llm.llm_engine.model_executor.shutdown()
