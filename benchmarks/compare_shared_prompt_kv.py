"""Compare two runs of shared_prompt_kv.py and fail on numerical differences."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _summarize(metadata: dict, coordinator: list[dict], worker: list[dict]) -> dict:
    expected = metadata["warmup"] + metadata["steps"]
    if len(coordinator) != expected or len(worker) != expected:
        raise ValueError(f"expected {expected} iterations, got {len(coordinator)} coordinator and {len(worker)} worker")
    warmup = metadata["warmup"]
    samples = metadata["samples"]
    prompt_length = metadata["prompt_length"]
    selected_coordinator = coordinator[warmup:]
    selected_worker = worker[warmup:]
    if any(row["stage_gpu_ms"]["prefill"] <= 0 or row["stage_gpu_ms"]["decode"] <= 0 for row in selected_worker):
        raise ValueError("missing prefill or decode timing")
    logical_prefill = [samples * prompt_length / (row["stage_gpu_ms"]["prefill"] / 1000) for row in selected_worker]
    physical_prefill = [row["prefill_tokens"] / (row["stage_gpu_ms"]["prefill"] / 1000) for row in selected_worker]
    decode = [row["decode_scheduled_tokens"] / (row["stage_gpu_ms"]["decode"] / 1000) for row in selected_worker]
    block_counts = None
    if all("peak_live_blocks_by_category" in row for row in selected_worker):
        block_counts = {
            category: max(row["peak_live_blocks_by_category"][category] for row in selected_worker)
            for category in ("prompt_full", "prompt_tail", "response_extra")
        }
    return {
        "median_wall_s": statistics.median(row["wall_s"] for row in selected_coordinator),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in selected_worker),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in selected_worker),
        "pool_blocks": sorted({row["pool_blocks"] for row in selected_worker}),
        "peak_live_blocks_by_category": block_counts,
        "median_logical_prefill_tokens_per_s": statistics.median(logical_prefill),
        "median_physical_prefill_tokens_per_s": statistics.median(physical_prefill),
        "median_decode_tokens_per_s": statistics.median(decode),
        "median_cache_management_ms": statistics.median(
            row["stage_gpu_ms"]["cache_management"] for row in selected_worker
        ),
    }


def compare(baseline_dir: Path, candidate_dir: Path, *, atol: float, rtol: float) -> dict:
    baseline_meta = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))
    candidate_meta = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in (
        "model_revision",
        "model_sha256",
        "prompt_length",
        "samples",
        "max_new_tokens",
        "warmup",
        "steps",
        "seed",
        "dtype",
        "replay",
        "attention_backend",
        "compile_model",
        "eager_decode",
        "torch",
        "torch_cuda",
        "gpu",
    ):
        if baseline_meta[key] != candidate_meta[key]:
            raise ValueError(f"mismatched {key}: {baseline_meta[key]!r} != {candidate_meta[key]!r}")
    baseline_rows = _read_lines(baseline_dir / "coordinator.jsonl")
    candidate_rows = _read_lines(candidate_dir / "coordinator.jsonl")
    baseline_worker = _read_lines(baseline_dir / "worker.jsonl")
    candidate_worker = _read_lines(candidate_dir / "worker.jsonl")
    baseline = _summarize(baseline_meta, baseline_rows, baseline_worker)
    candidate = _summarize(candidate_meta, candidate_rows, candidate_worker)
    max_abs_error = 0.0
    compared_tokens = 0
    for first, second in zip(baseline_rows, candidate_rows, strict=True):
        if first["response_ids"] != second["response_ids"]:
            raise AssertionError(f"response token mismatch at step {first['step']}")
        for first_row, second_row in zip(first["logprobs"], second["logprobs"], strict=True):
            for expected, actual in zip(first_row, second_row, strict=True):
                if not math.isfinite(expected) or not math.isfinite(actual):
                    raise AssertionError("non-finite logprob")
                error = abs(expected - actual)
                max_abs_error = max(max_abs_error, error)
                compared_tokens += 1
                if error > atol + rtol * abs(expected):
                    raise AssertionError(f"logprob mismatch: expected={expected} actual={actual} abs_error={error}")
    if not compared_tokens:
        raise AssertionError("no logprob values compared")
    return {
        "baseline_commit": baseline_meta["commit"],
        "candidate_commit": candidate_meta["commit"],
        "atol": atol,
        "rtol": rtol,
        "max_abs_logprob_error": max_abs_error,
        "compared_logprobs": compared_tokens,
        "baseline": baseline,
        "candidate": candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline_dir, args.candidate_dir, atol=args.atol, rtol=args.rtol), indent=2))


if __name__ == "__main__":
    main()
