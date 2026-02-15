#!/usr/bin/env python3
"""Generate deterministic 3D point sets for oracle comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate_test_points(n: int = 10_000, bounds: float = 3.0, seed: int = 42) -> np.ndarray:
    """Deterministically generate points in [-bounds, bounds]^3 plus structured points."""
    rng = np.random.default_rng(seed)
    random_points = rng.uniform(-bounds, bounds, size=(n, 3))
    structured_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.5, 0.5],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    return np.vstack([structured_points, random_points])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--bounds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points = generate_test_points(args.count, args.bounds, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"points": points.tolist()}), encoding="utf-8")


if __name__ == "__main__":
    main()
