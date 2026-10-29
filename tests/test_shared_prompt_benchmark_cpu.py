from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.compare_shared_prompt_kv import compare
from benchmarks.shared_prompt_kv import _cache_block_counts
from benchmarks.trainer_shared_prompt_kv import _compare_rollouts


def _write_run(path, *, responses, logprobs, revision="same", group_size=1):
    path.mkdir()
    metadata = {
        "commit": "commit",
        "model_revision": revision,
        "model_sha256": "sha",
        "prompt_length": 8,
        "samples": 2,
        "max_new_tokens": 2,
        "warmup": 0,
        "steps": 1,
        "seed": 1,
        "dtype": "bf16",
        "replay": True,
        "attention_backend": "native",
        "compile_model": False,
        "eager_decode": True,
        "torch": "version",
        "torch_cuda": "version",
        "gpu": "gpu",
    }
    worker = {
        "stage_gpu_ms": {"prefill": 1.0, "decode": 2.0, "cache_management": 0.5},
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 120,
        "pool_blocks": 4,
        "prompt_group_size": group_size,
        "prefill_tokens": 16 if group_size == 1 else 8,
        "decode_scheduled_tokens": 2,
    }
    coordinator = {"step": 0, "wall_s": 0.1, "response_ids": responses, "logprobs": logprobs}
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "worker.jsonl").write_text(json.dumps(worker) + "\n", encoding="utf-8")
    (path / "coordinator.jsonl").write_text(json.dumps(coordinator) + "\n", encoding="utf-8")


def test_comparison_checks_token_and_logprob_consistency(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, responses=[[1, 2], [3, 4]], logprobs=[[-1.0, -2.0], [-3.0, -4.0]])
    _write_run(candidate, responses=[[1, 2], [3, 4]], logprobs=[[-1.001, -2.0], [-3.0, -4.0]], group_size=2)
    result = compare(baseline, candidate, atol=1e-2, rtol=1e-2)
    assert result["compared_logprobs"] == 4
    assert result["baseline"]["median_physical_prefill_tokens_per_s"] == 16000
    assert result["candidate"]["median_physical_prefill_tokens_per_s"] == 8000
    with pytest.raises(AssertionError, match="logprob mismatch"):
        compare(baseline, candidate, atol=1e-5, rtol=1e-5)


def test_comparison_rejects_different_model_revisions(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, responses=[[1]], logprobs=[[-1.0]])
    _write_run(candidate, responses=[[1]], logprobs=[[-1.0]], revision="other")
    with pytest.raises(ValueError, match="mismatched model_revision"):
        compare(baseline, candidate, atol=1e-2, rtol=1e-2)


def test_trainer_comparison_records_all_reload_mismatches():
    before = [SimpleNamespace(resp_tokens=[1, 2], resp_logprobs=[-0.1, -2.0])]
    after = [SimpleNamespace(resp_tokens=[1, 2], resp_logprobs=[-0.12, -2.1])]
    result = _compare_rollouts(before, after)
    assert result["tokens_match"]
    assert result["compared_logprobs"] == 2
    assert result["logprob_mismatches"] == 2
    assert result["max_abs_logprob_error"] == pytest.approx(0.1)


def test_benchmark_counts_shared_prompt_tail_and_response_blocks():
    state = SimpleNamespace(
        prompts=[[1, 2, 3, 4, 5, 6]] * 2,
        kv_block_size=4,
        _seq_to_blocks={0: [0, 1, 2], 1: [0, 3, 4]},
    )
    assert _cache_block_counts(state) == {"prompt_full": 1, "prompt_tail": 2, "response_extra": 2}
