"""Compare multi-sample KV allocation with and without prompt sharing.

Run in a fresh process for each prompt length and sample count. The worker
records CUDA events and allocator peaks; the coordinator records wall time and
generated tokens. Logs include warmup iterations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import torch

import areno.engine.inference as inference_mod
from areno import ArenoEngine, RuntimeConfig
from areno.engine.data import SamplingParams
from areno.engine.inference import InferenceManager
from areno.engine.worker import ArenoWorker


class TimedInferenceManager(InferenceManager):
    def _timed(self, stage, method, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return method(*args, **kwargs)
        finally:
            end.record()
            self._benchmark_events.append((stage, start, end))

    def _run_prefill_payload(self, payload):
        self._benchmark_prefill_tokens += int(payload.input_ids.numel())
        return self._timed("prefill", super()._run_prefill_payload, payload)

    def _infer_next_token_tensor(self, payload):
        self._benchmark_prefill_tokens += int(payload.input_ids.numel())
        return self._timed("prefill", super()._infer_next_token_tensor, payload)

    def _infer_decode_next_token_tensor(self, *args, **kwargs):
        return self._timed("decode", super()._infer_decode_next_token_tensor, *args, **kwargs)

    def _ensure_decode_kv_blocks(self, *args, **kwargs):
        result = self._timed("cache_management", super()._ensure_decode_kv_blocks, *args, **kwargs)
        state = args[0] if args else kwargs["state"]
        counts = _cache_block_counts(state)
        for category, count in counts.items():
            self._benchmark_live_blocks[category] = max(self._benchmark_live_blocks[category], count)
        return result


def _cache_block_counts(state) -> dict[str, int]:
    """Count physical prompt, partial-tail, and extra response blocks in use."""
    full, tail, response = set(), set(), set()
    for row, blocks in state._seq_to_blocks.items():
        full_count, tail_count = divmod(len(state.prompts[row]), state.kv_block_size)
        full.update(blocks[:full_count])
        if tail_count:
            tail.update(blocks[full_count : full_count + 1])
        response.update(blocks[full_count + bool(tail_count) :])
    return {"prompt_full": len(full), "prompt_tail": len(tail), "response_extra": len(response)}


class TimedWorker(ArenoWorker):
    def __init__(self, config):
        super().__init__(config)
        self.inference = TimedInferenceManager(self)
        if os.environ.get("ARENO_PROMPT_KV_BENCH_REPLAY") == "1":
            inference_mod._sample_greedy_sharded = _fixed_tokens

    def infer_rollout(self, payload, finished_callback=None, refill_callback=None):
        self._benchmark_events = []
        self._benchmark_prefill_tokens = 0
        self._benchmark_live_blocks = {"prompt_full": 0, "prompt_tail": 0, "response_extra": 0}
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        output = super().infer_rollout(payload, finished_callback=finished_callback, refill_callback=refill_callback)
        torch.cuda.synchronize(self.device)
        stage_ms = {"prefill": 0.0, "decode": 0.0, "cache_management": 0.0}
        for stage, before, after in self._benchmark_events:
            stage_ms[stage] += before.elapsed_time(after)
        record = {
            "worker_wall_s": time.perf_counter() - start,
            "stage_gpu_ms": stage_ms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "pool_blocks": payload.num_blocks,
            "scratch_blocks": 1,
            "peak_live_blocks_by_category": self._benchmark_live_blocks,
            "block_size": payload.block_size,
            "prompt_group_size": getattr(payload, "prompt_group_size", 1),
            "decode_graph_buckets": sorted(self._decode_graphs),
            "decode_graph_skipped_buckets": sorted(self._decode_graph_skipped_buckets),
            "prefill_tokens": self._benchmark_prefill_tokens,
            "decode_scheduled_tokens": output.metrics.get("decode_scheduled_tokens") if output is not None else None,
        }
        with Path(os.environ["ARENO_PROMPT_KV_BENCH_WORKER_LOG"]).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return output


def _fixed_tokens(logits, vocab_size, tp_size, *, sample_step, **kwargs):
    """Force different known tokens per row while retaining policy logprobs."""
    del tp_size, kwargs
    rows = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
    return (rows * 17 + sample_step + 100) % vocab_size


def git_revision(path: str) -> str | None:
    result = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--model-sha256", default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--attention-backend", choices=("native", "flash"), default="native")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_log = output_dir / "worker.jsonl"
    if worker_log.exists():
        raise FileExistsError(f"refusing to overwrite {worker_log}")
    os.environ["ARENO_PROMPT_KV_BENCH_WORKER_LOG"] = str(worker_log.resolve())
    os.environ["ARENO_PROMPT_KV_BENCH_REPLAY"] = "1" if args.replay else "0"
    runtime = RuntimeConfig(attn_backend=args.attention_backend, compile_model=False, eager_decode=not args.cuda_graph)
    if args.shared:
        if not hasattr(runtime, "share_prompt_kv"):
            raise RuntimeError("this checkout does not support shared prompt KV")
        runtime.share_prompt_kv = True
    metadata = {
        "commit": git_revision("."),
        "model_revision": args.model_revision or git_revision(args.model_path),
        "model_sha256": args.model_sha256,
        "model_path": args.model_path,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_length": args.prompt_length,
        "samples": args.samples,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "steps": args.steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "replay": args.replay,
        "shared": args.shared,
        "attention_backend": args.attention_backend,
        "compile_model": False,
        "eager_decode": not args.cuda_graph,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    engine = ArenoEngine.from_pretrained(
        args.model_path, tp_size=1, dp_size=1, devices=[0], runtime_config=runtime, role="rollout", start=False
    )
    engine.config.model.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    engine.cluster.worker_cls = TimedWorker
    engine.cluster.start()
    try:
        prompt = [1 + (index % 10000) for index in range(args.prompt_length)]
        prompts = [prompt for _ in range(args.samples)]
        sampling = SamplingParams(temperature=0.0, seed=args.seed)
        options = {
            "max_new_tokens": args.max_new_tokens,
            "max_running_prompts": args.samples,
            "max_prompt_len": args.prompt_length,
            "sampling_params": sampling,
        }
        if "prompt_group_size" in inspect.signature(engine.generate_rollout).parameters:
            options["prompt_group_size"] = args.samples
        with (output_dir / "coordinator.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(args.warmup + args.steps):
                start = time.perf_counter()
                output = engine.generate_rollout(prompts, **options)
                record = {
                    "step": step,
                    "warmup": step < args.warmup,
                    "wall_s": time.perf_counter() - start,
                    "response_ids": output.response_ids,
                    "logprobs": [
                        output.logprobs[row, : len(tokens)].tolist() for row, tokens in enumerate(output.response_ids)
                    ],
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps({"step": step, "wall_s": record["wall_s"]}), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
