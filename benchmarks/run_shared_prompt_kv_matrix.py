"""Run the issue-592 fixed-token matrix in isolated worker processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.compare_shared_prompt_kv import compare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--attention-backend", choices=("native", "flash"), default="native")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    summary = {}
    for prompt_length in (1024, 4096):
        for samples in (1, 4, 8):
            case = f"prompt{prompt_length}-samples{samples}"
            for name, extra in (("baseline", []), ("shared", ["--shared"])):
                active = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if active:
                    raise RuntimeError(f"GPU already has compute processes: {active}")
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], check=True
                )
                subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True)
                command = [
                    sys.executable,
                    str(root / "benchmarks/shared_prompt_kv.py"),
                    "--model-path",
                    args.model_path,
                    "--model-revision",
                    args.model_revision,
                    "--model-sha256",
                    args.model_sha256,
                    "--output-dir",
                    str(output_dir / case / name),
                    "--prompt-length",
                    str(prompt_length),
                    "--samples",
                    str(samples),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                    "--attention-backend",
                    args.attention_backend,
                    "--replay",
                    *extra,
                ]
                print(json.dumps({"case": case, "mode": name, "command": command}), flush=True)
                subprocess.run(command, cwd=root, env=env, check=True)
            comparison = compare(output_dir / case / "baseline", output_dir / case / "shared", atol=1e-2, rtol=1e-2)
            (output_dir / case / "comparison.json").write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
            summary[case] = comparison
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"case": case, "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()
