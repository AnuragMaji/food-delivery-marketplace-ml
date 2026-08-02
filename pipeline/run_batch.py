"""Pipeline orchestrator: runs one full accelerated tick end-to-end, wiring
together every stage already built:

  1. Generate ~1 accelerated week of new orders (data-gen/realtime_generator.py)
  2. Publish to Redpanda, then consume + aggregate rolling metrics (streaming/)
  3. Validate against the data contract (contracts/validate_batch.py)
  4. Clean + load into fact_deliveries (pipeline/preprocess.py)
  5. Recompute features into Redis + feature_snapshots (features/batch_features.py)
  6. Score ETA predictions for a sample of the newly-landed deliveries (model_api)

Each stage runs as its own subprocess — the exact same commands already
validated by hand — rather than importing everything into one process, so
each stage stays independently testable and this mirrors how GitHub Actions
(task #12) will actually invoke things.

Usage:
    python pipeline/run_batch.py
    python pipeline/run_batch.py --weeks 2 --max-predictions 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable
DEFAULT_MAX_PREDICTIONS = 300


def run_step(name: str, script_args: list[str]):
    print(f"\n=== {name} ===")
    result = subprocess.run([PYTHON] + script_args, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Step '{name}' failed (exit code {result.returncode})")


def score_predictions(delivery_ids: list[int], max_predictions: int):
    model_api_url = os.environ.get("MODEL_API_URL", "http://localhost:8000")

    try:
        health = requests.get(f"{model_api_url}/health", timeout=5)
        health.raise_for_status()
    except requests.RequestException as e:
        print(f"  model_api unreachable at {model_api_url} ({e}) — skipping prediction scoring.")
        return

    sample = delivery_ids if len(delivery_ids) <= max_predictions else random.sample(delivery_ids, max_predictions)
    print(f"  scoring {len(sample)} of {len(delivery_ids)} new deliveries...")

    n_success, n_failed = 0, 0
    for delivery_id in sample:
        try:
            resp = requests.post(f"{model_api_url}/predict", json={"delivery_id": delivery_id}, timeout=10)
            resp.raise_for_status()
            n_success += 1
        except requests.RequestException:
            n_failed += 1

    print(f"  scored {n_success} deliveries ({n_failed} failed)")


def main():
    parser = argparse.ArgumentParser(description="Run one full pipeline tick end-to-end.")
    parser.add_argument("--weeks", type=int, default=1, help="Accelerated weeks to generate this tick (default 1)")
    parser.add_argument("--max-predictions", type=int, default=DEFAULT_MAX_PREDICTIONS,
                         help="Max new deliveries to score with model_api per tick (default 300)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the generator (default: nondeterministic)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_batch = tmp / "realtime_batch.json"
        consumed_batch = tmp / "consumed_batch.json"
        validated_batch = tmp / "validated_batch.json"
        inserted_ids_path = tmp / "inserted_ids.json"
        batch_id = f"tick-{uuid.uuid4().hex[:8]}"

        gen_args = ["data-gen/realtime_generator.py", "--weeks", str(args.weeks), "--output", str(raw_batch)]
        if args.seed is not None:
            gen_args += ["--seed", str(args.seed)]
        run_step("Step 1: Generate new orders", gen_args)

        run_step("Step 2a: Publish to Redpanda", ["streaming/producer.py", "--input", str(raw_batch)])
        run_step("Step 2b: Consume + aggregate rolling metrics", ["streaming/consumer.py", "--output", str(consumed_batch)])
        run_step("Step 3: Validate against the data contract", [
            "contracts/validate_batch.py", "--input", str(consumed_batch),
            "--output", str(validated_batch), "--batch-id", batch_id,
        ])
        run_step("Step 4: Clean + load into fact_deliveries", [
            "pipeline/preprocess.py", "--input", str(validated_batch), "--output-ids", str(inserted_ids_path),
        ])
        run_step("Step 5: Recompute features", ["features/batch_features.py"])

        print("\n=== Step 6: Score ETA predictions ===")
        if inserted_ids_path.exists():
            with open(inserted_ids_path) as f:
                delivery_ids = json.load(f)
            score_predictions(delivery_ids, args.max_predictions)
        else:
            print("  no inserted-ids file found — skipping scoring.")

    print(f"\nTick '{batch_id}' complete.")


if __name__ == "__main__":
    main()
