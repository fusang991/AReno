Shared prompt KV: A100 measurements
=====================================

The measured source commit was ``74cdb6be6860ce15b0590ff22f1b0fe0aa005298``.
Both sides used this exact commit with sharing disabled or enabled. The
public model was ModelScope ``Qwen/Qwen3-0.6B``, revision
``09b42cad3d112e832108974449ccb5e8e0f5b5d1``; the model safetensors SHA-256
was ``f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b``.
The host had one NVIDIA A100-SXM4 40 GB, CUDA 13.0, PyTorch 2.13.0+cu130,
Triton 3.7.1, Python 3.12.12, and native attention. Inputs and 16-token
responses were fixed; dtype BF16, seed 1234, eager decode, 2 warmups and 10
measured steps were identical. These numbers are a fixed-token replay, not a
measurement of end-to-end training throughput.

.. list-table:: A/B values for sharing disabled / enabled
   :header-rows: 1

   * - Prompt tokens / samples
     - Peak allocated GiB
     - Peak reserved GiB
     - Median wall seconds
     - Pool blocks
   * - 1024 / 1
     - 2.67 / 2.67
     - 2.81 / 2.81
     - 1.053 / 1.063
     - 5 / 5
   * - 1024 / 4
     - 3.96 / 3.63
     - 4.18 / 3.80
     - 3.007 / 3.010
     - 20 / 8
   * - 1024 / 8
     - 5.67 / 4.91
     - 6.08 / 5.26
     - 5.555 / 5.562
     - 40 / 12
   * - 4096 / 1
     - 3.87 / 3.87
     - 4.07 / 4.07
     - 10.527 / 10.526
     - 17 / 17
   * - 4096 / 4
     - 8.77 / 7.46
     - 9.53 / 8.20
     - 40.452 / 40.408
     - 68 / 20
   * - 4096 / 8
     - 15.30 / 12.23
     - 16.72 / 13.62
     - 80.690 / 80.629
     - 136 / 24

All 4,992 compared response log-probabilities across the six cases, including
warmups, matched exactly (maximum absolute difference 0). The 4096-token,
8-sample run used 128 / 16 live full prompt blocks and 8 / 8 response blocks;
each 256-token block is represented per layer. The allocator also used one
scratch block. Partial tails are covered separately by the CUDA graph replay
of a 257-token prompt: 4 / 1 full prompt blocks and 4 / 4 private tail blocks,
with 64 matching log-probabilities and no skipped graph bucket.

An additional FP32 fixed-token replay on source commit
``3401172ea37c4cb4783c4b9736066ba293258249`` used a 257-token prompt,
four samples, no warmup, and one measured step. With sharing disabled /
enabled, it used 4 / 1 full prompt blocks and 4 / 4 private tail blocks;
all 64 response log-probabilities and response token IDs matched exactly.
This is a correctness check, not a performance measurement. Runtime and
benchmark sources are unchanged between that commit and this report.

The implementation preserves the baseline prefill batch shape to avoid BF16
matrix-kernel rounding changes. It stores one immutable prompt copy but still
computes duplicate prefill rows. Thus this experiment establishes a memory
saving, not prefill compute elimination or a throughput speedup. The
``worker.jsonl`` files separately record prefill/decode CUDA times, logical
and physical token counts, cache-management time and block categories.

The public Trainer smoke test used two GSPO updates and a weight save/reload,
then another update, on a small synthetic greeting task. Native
``train/logp_abs_diff_mean`` for the first two updates was 0.034207, 0.018548 (control)
and 0.034207, 0.020724 (shared); after reload it was 0.023471 and
0.012508 respectively. Reloaded greedy response tokens and all 64
log-probabilities matched exactly for each mode. These synthetic rewards are
not a task-quality evaluation. The sampled updates use temperature and top-k;
AReno records raw model log-probabilities rather than the renormalized
sampling distribution. The existing CUDA AdamW checkpoint restores
model weights but does not restore its optimizer state; this is not a full
optimizer-resume result.

On the local CPU suite, the candidate had 1552 passed, 34 skipped, and six
failures. The same six failures reproduced on the unchanged target branch:
Gemma4 dependency, MLX log formatting, three MoE fixture mismatches, and a
setuptools metadata ordering check. The relevant candidate CPU subset passed
259 tests. CUDA and Trainer results above were measured on the remote A100.
In a subsequent local Python 3.12 run, the candidate CPU suite had 1555
passed, 34 skipped, and three failures. The same three MoE fixture failures
reproduced individually on the unchanged target branch. The repository-wide
pre-commit checks passed; the full CPU suite is not green on the candidate.

Reproduce the matrix after resolving the exact ModelScope snapshot locally::

   python -m benchmarks.run_shared_prompt_kv_matrix \
     --model-path "$MODEL_SNAPSHOT" \
     --model-revision 09b42cad3d112e832108974449ccb5e8e0f5b5d1 \
     --model-sha256 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b \
     --output-dir "$RESULTS_DIR" --max-new-tokens 16 --warmup 2 --steps 10

Raw per-iteration records and comparison JSON are in
``benchmarks/results/issue592/matrix-v1``; trainer smoke records are in the
adjacent ``trainer`` directory. Only local model/checkpoint paths were
redacted in these copies. The later positional-compatibility change and this
report do not change the measured attention or cache behavior.

Reproduce the FP32 correctness replay on the same model snapshot::

   for MODE in baseline shared; do
     SHARED_FLAG=
     if [ "$MODE" = shared ]; then SHARED_FLAG=--shared; fi
     python -m benchmarks.shared_prompt_kv \
       --model-path "$MODEL_SNAPSHOT" \
       --model-revision 09b42cad3d112e832108974449ccb5e8e0f5b5d1 \
       --model-sha256 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b \
       --output-dir "$RESULTS_DIR/fp32-$MODE" --prompt-length 257 \
       --samples 4 --max-new-tokens 16 --warmup 0 --steps 1 --seed 1234 \
       --dtype fp32 --replay ${SHARED_FLAG:+$SHARED_FLAG}
   done
   python -m benchmarks.compare_shared_prompt_kv \
     "$RESULTS_DIR/fp32-baseline" "$RESULTS_DIR/fp32-shared" --atol 0 --rtol 0

The redacted replay records and comparison output are in
``benchmarks/results/issue592/fp32-replay``.
