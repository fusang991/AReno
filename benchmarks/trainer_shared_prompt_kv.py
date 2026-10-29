"""Run two synthetic GSPO steps and a model-weight save/reload check."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from areno import Trainer
from areno.api import CUDA, CudaConfig, SamplingParams, TrainSequence, gspo_loss_fn


def _create_trainer(model_path: str, output: Path, shared: bool) -> Trainer:
    runtime = {"attn_backend": "native", "compile_model": False, "eager_decode": True}
    if shared:
        runtime["share_prompt_kv"] = True
    trainer = Trainer(
        world_size=1,
        model_path=model_path,
        backend_type=CUDA,
        custom_config=CudaConfig(tp_size=1, max_running_prompts=4, runtime=runtime),
        metrics_log_dir=str(output),
    )
    trainer.init()
    return trainer


def _rollout(trainer: Trainer, prompt: list[int], *, seed: int, greedy: bool):
    sampling = SamplingParams(max_new_tokens=16, greedy=greedy, temperature=0.8, top_k=50, seed=seed)
    trainer.begin_rollout_session()
    try:
        return trainer.rollout_token_batch([prompt], n_samples=4, sampling_params=sampling)[0].sequences
    finally:
        trainer.end_rollout_session()


def _step(trainer: Trainer, prompt: list[int], seed: int) -> dict:
    sequences = _rollout(trainer, prompt, seed=seed, greedy=False)
    if len(sequences) != 4 or any(not sequence.resp_tokens for sequence in sequences):
        raise AssertionError("expected four non-empty sampled responses")
    # A deterministic synthetic reward exercises a genuine update without
    # claiming task quality on the tiny smoke-validation dataset.
    rewards = [sum(token % 97 for token in seq.resp_tokens) / (96 * len(seq.resp_tokens)) for seq in sequences]
    mean = sum(rewards) / len(rewards)
    std = (sum((value - mean) ** 2 for value in rewards) / len(rewards)) ** 0.5
    if std <= 1e-8:
        raise AssertionError("synthetic rewards have no variance")
    eos = trainer.get_tokenizer().eos_token_id
    batch = [
        TrainSequence(
            tokens=prompt + seq.resp_tokens,
            prompt_len=len(prompt),
            logprobs=[0.0] * len(prompt) + seq.resp_logprobs,
            scalar_advantage=(reward - mean) / std,
            reward=reward,
            eos_token_id=int(eos),
        )
        for seq, reward in zip(sequences, rewards, strict=True)
    ]
    result = trainer.train(batch, gspo_loss_fn, mini_bs=4)
    for key in ("loss", "grad_norm", "logp_abs_diff_mean"):
        if key not in result or not math.isfinite(float(result[key])):
            raise AssertionError(f"missing or non-finite training metric: {key}")
    if result["grad_norm"] <= 0:
        raise AssertionError("training did not produce a nonzero gradient")
    return {
        "seed": seed,
        "metrics": result,
        "rewards": rewards,
        "responses": [trainer.get_tokenizer().decode(seq.resp_tokens) for seq in sequences],
    }


def _compare_rollouts(before, after) -> dict:
    maximum = 0.0
    mismatches = 0
    compared = 0
    tokens_match = len(before) == len(after)
    for first, second in zip(before, after):
        if first.resp_tokens != second.resp_tokens:
            tokens_match = False
        if len(first.resp_logprobs) != len(second.resp_logprobs):
            tokens_match = False
        for expected, actual in zip(first.resp_logprobs, second.resp_logprobs):
            if not math.isfinite(expected) or not math.isfinite(actual):
                raise AssertionError("greedy rollout produced a non-finite logprob")
            error = abs(expected - actual)
            maximum = max(maximum, error)
            compared += 1
            if error > 1e-2 + 1e-2 * abs(expected):
                mismatches += 1
    return {
        "tokens_match": tokens_match,
        "max_abs_logprob_error": maximum,
        "logprob_mismatches": mismatches,
        "compared_logprobs": compared,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = json.loads(args.dataset_path.read_text(encoding="utf-8").splitlines()[0])
    trainer = _create_trainer(args.model_path, args.output_dir / "metrics_before", args.shared)
    try:
        prompt = trainer.get_tokenizer().encode(record["prompt"])
        steps = [_step(trainer, prompt, args.seed + step) for step in range(2)]
        before = _rollout(trainer, prompt, seed=args.seed, greedy=True)
        repeated = _rollout(trainer, prompt, seed=args.seed, greedy=True)
        checkpoint = trainer.save_checkpoint(str(args.output_dir / "checkpoint"))
    finally:
        trainer.close()
    resumed = _create_trainer(checkpoint, args.output_dir / "metrics_after", args.shared)
    try:
        after = _rollout(resumed, prompt, seed=args.seed, greedy=True)
        repeat_comparison = _compare_rollouts(before, repeated)
        reload_comparison = _compare_rollouts(before, after)
        resumed_step = _step(resumed, prompt, args.seed + 2)
    finally:
        resumed.close()
    result = {
        "shared": args.shared,
        "dataset": record,
        "steps": steps,
        "repeat_comparison": repeat_comparison,
        "reload_comparison": reload_comparison,
        "resumed_step": resumed_step,
        "checkpoint": checkpoint,
        "resume_note": "model weights reloaded; CUDA Trainer checkpoint does not restore optimizer state",
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not reload_comparison["tokens_match"] or reload_comparison["logprob_mismatches"]:
        raise AssertionError("greedy output changed beyond tolerance after saving and reloading")


if __name__ == "__main__":
    main()
