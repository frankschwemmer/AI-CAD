#!/usr/bin/env python3
"""Performance budget checks for the Rust SDF pipeline."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


def prepare_binary(workspace_root: Path) -> Path:
    subprocess.run(
        ["cargo", "build", "--quiet", "--release", "-p", "sdf-cli"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return workspace_root / "target" / "release" / "sdf-cli"


def run_benchmark(binary: Path, workspace_root: Path, resolution: int, runs: int) -> dict[str, float]:
    cmd = [
        str(binary),
        "benchmark-pipeline",
        "--resolution",
        str(resolution),
        "--runs",
        str(runs),
    ]
    completed = subprocess.run(
        cmd,
        cwd=workspace_root,
        check=True,
        capture_output=True,
        text=True,
    )
    metrics: dict[str, float] = {}
    for line in completed.stdout.strip().splitlines():
        key, value = line.split(maxsplit=1)
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


@dataclass(frozen=True)
class BudgetCheck:
    metric: str
    limit: float
    resolution: int
    runs: int = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit non-zero when any budget fails.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    checks = [
        BudgetCheck(metric="eval_ms", limit=50.0, resolution=64),
        BudgetCheck(metric="mesh_ms", limit=100.0, resolution=64),
        BudgetCheck(metric="total_ms", limit=500.0, resolution=64),
        BudgetCheck(metric="mesh_ms", limit=500.0, resolution=128, runs=1),
    ]
    binary = prepare_binary(workspace_root)

    cache: dict[tuple[int, int], dict[str, float]] = {}
    failures: list[str] = []

    for check in checks:
        key = (check.resolution, check.runs)
        if key not in cache:
            cache[key] = run_benchmark(binary, workspace_root, check.resolution, check.runs)
        metrics = cache[key]
        value = metrics.get(check.metric)
        if value is None:
            failures.append(
                f"missing metric '{check.metric}' for resolution {check.resolution}"
            )
            continue

        passed = value <= check.limit
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] res={check.resolution} runs={check.runs} {check.metric}={value:.3f}ms "
            f"(limit {check.limit:.3f}ms)"
        )
        if not passed:
            failures.append(
                f"{check.metric} at resolution {check.resolution}: {value:.3f}ms > {check.limit:.3f}ms"
            )

    if failures:
        print("\nBudget failures:")
        for failure in failures:
            print(f" - {failure}")
        if args.enforce:
            raise SystemExit(1)
    else:
        print("\nAll performance budgets passed.")


if __name__ == "__main__":
    main()
