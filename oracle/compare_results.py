#!/usr/bin/env python3
"""Cross-language oracle comparison: Rust CLI vs Python implementation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from evaluate_oracle import (
    evaluate_box,
    evaluate_capped_cone,
    evaluate_capped_cylinder,
    evaluate_capsule,
    evaluate_cylinder,
    evaluate_operation_scene,
    evaluate_plane,
    evaluate_rounded_box,
    evaluate_rounded_cylinder,
    evaluate_sphere,
    evaluate_transform_scene,
    evaluate_torus,
)
from generate_test_points import generate_test_points


def compare(rust_values: np.ndarray, python_values: np.ndarray, tolerance: float, name: str) -> None:
    if rust_values.shape != python_values.shape:
        raise AssertionError(
            f"{name}: value length mismatch (rust={rust_values.shape}, python={python_values.shape})"
        )
    diff = np.abs(rust_values - python_values)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    failures = int(np.sum(diff > tolerance))
    print(
        f"[{name}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
        f"failures={failures}/{len(diff)}"
    )
    if failures != 0:
        idx = int(np.argmax(diff))
        raise AssertionError(
            f"{name}: {failures} points exceed tolerance {tolerance}. "
            f"Max diff: {max_diff:.2e} at index {idx}."
        )


def run_rust_primitive(
    workspace_root: Path, command: str, flags: list[str], points: np.ndarray
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        points_file = Path(tmp_dir) / "points.txt"
        points_file.write_text(
            "".join(f"{x} {y} {z}\n" for x, y, z in points),
            encoding="utf-8",
        )

        cmd = [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "sdf-cli",
            "--",
            command,
            *flags,
            "--points-file",
            str(points_file),
        ]

        completed = subprocess.run(
            cmd,
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
        )
        values = [
            float(line.strip())
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class PrimitiveCase:
    name: str
    command: str
    flags: list[str]
    oracle: Callable[[np.ndarray], np.ndarray]
    tolerance: float = 1e-12


def primitive_cases() -> list[PrimitiveCase]:
    box_half_extents = np.array([1.0, 1.5, 0.75], dtype=np.float64)
    rounded_box_half_extents = np.array([1.2, 0.9, 0.8], dtype=np.float64)
    plane_normal = np.array([0.5, -1.0, 0.25], dtype=np.float64)
    capsule_a = np.array([-0.5, 0.2, -1.0], dtype=np.float64)
    capsule_b = np.array([1.25, -0.3, 0.8], dtype=np.float64)

    return [
        PrimitiveCase(
            name="sphere",
            command="evaluate-sphere",
            flags=["--radius", "1.0"],
            oracle=lambda pts: evaluate_sphere(1.0, pts),
        ),
        PrimitiveCase(
            name="box",
            command="evaluate-box",
            flags=["--hx", "1.0", "--hy", "1.5", "--hz", "0.75"],
            oracle=lambda pts: evaluate_box(box_half_extents, pts),
        ),
        PrimitiveCase(
            name="rounded_box",
            command="evaluate-rounded-box",
            flags=["--hx", "1.2", "--hy", "0.9", "--hz", "0.8", "--radius", "0.2"],
            oracle=lambda pts: evaluate_rounded_box(rounded_box_half_extents, 0.2, pts),
        ),
        PrimitiveCase(
            name="cylinder",
            command="evaluate-cylinder",
            flags=["--radius", "1.1", "--height", "2.4"],
            oracle=lambda pts: evaluate_cylinder(1.1, 2.4, pts),
        ),
        PrimitiveCase(
            name="capped_cylinder",
            command="evaluate-capped-cylinder",
            flags=["--radius", "0.8", "--half-height", "1.3"],
            oracle=lambda pts: evaluate_capped_cylinder(0.8, 1.3, pts),
        ),
        PrimitiveCase(
            name="torus",
            command="evaluate-torus",
            flags=["--major-radius", "2.0", "--minor-radius", "0.4"],
            oracle=lambda pts: evaluate_torus(2.0, 0.4, pts),
        ),
        PrimitiveCase(
            name="plane",
            command="evaluate-plane",
            flags=["--nx", "0.5", "--ny", "-1.0", "--nz", "0.25", "--offset", "0.2"],
            oracle=lambda pts: evaluate_plane(plane_normal, 0.2, pts),
        ),
        PrimitiveCase(
            name="capsule",
            command="evaluate-capsule",
            flags=[
                "--ax",
                "-0.5",
                "--ay",
                "0.2",
                "--az",
                "-1.0",
                "--bx",
                "1.25",
                "--by",
                "-0.3",
                "--bz",
                "0.8",
                "--radius",
                "0.35",
            ],
            oracle=lambda pts: evaluate_capsule(capsule_a, capsule_b, 0.35, pts),
        ),
        PrimitiveCase(
            name="capped_cone",
            command="evaluate-capped-cone",
            flags=["--radius1", "0.8", "--radius2", "0.3", "--height", "2.2"],
            oracle=lambda pts: evaluate_capped_cone(0.8, 0.3, 2.2, pts),
        ),
        PrimitiveCase(
            name="rounded_cylinder",
            command="evaluate-rounded-cylinder",
            flags=["--radius", "1.2", "--height", "2.4", "--edge-radius", "0.2"],
            oracle=lambda pts: evaluate_rounded_cylinder(1.2, 2.4, 0.2, pts),
        ),
    ]


def operation_cases() -> list[PrimitiveCase]:
    return [
        PrimitiveCase(
            name="op_union",
            command="evaluate-operation-scene",
            flags=["--operation", "union"],
            oracle=lambda pts: evaluate_operation_scene("union", pts),
        ),
        PrimitiveCase(
            name="op_intersection",
            command="evaluate-operation-scene",
            flags=["--operation", "intersection"],
            oracle=lambda pts: evaluate_operation_scene("intersection", pts),
        ),
        PrimitiveCase(
            name="op_difference",
            command="evaluate-operation-scene",
            flags=["--operation", "difference"],
            oracle=lambda pts: evaluate_operation_scene("difference", pts),
        ),
        PrimitiveCase(
            name="op_smooth_union",
            command="evaluate-operation-scene",
            flags=["--operation", "smooth_union", "--k", "0.2"],
            oracle=lambda pts: evaluate_operation_scene("smooth_union", pts, k=0.2),
            tolerance=1e-11,
        ),
        PrimitiveCase(
            name="op_smooth_intersection",
            command="evaluate-operation-scene",
            flags=["--operation", "smooth_intersection", "--k", "0.2"],
            oracle=lambda pts: evaluate_operation_scene("smooth_intersection", pts, k=0.2),
            tolerance=1e-11,
        ),
        PrimitiveCase(
            name="op_smooth_difference",
            command="evaluate-operation-scene",
            flags=["--operation", "smooth_difference", "--k", "0.2"],
            oracle=lambda pts: evaluate_operation_scene("smooth_difference", pts, k=0.2),
            tolerance=1e-11,
        ),
        PrimitiveCase(
            name="op_negate",
            command="evaluate-operation-scene",
            flags=["--operation", "negate"],
            oracle=lambda pts: evaluate_operation_scene("negate", pts),
        ),
        PrimitiveCase(
            name="op_shell",
            command="evaluate-operation-scene",
            flags=["--operation", "shell", "--thickness", "0.1"],
            oracle=lambda pts: evaluate_operation_scene("shell", pts, thickness=0.1),
        ),
        PrimitiveCase(
            name="op_elongate",
            command="evaluate-operation-scene",
            flags=[
                "--operation",
                "elongate",
                "--hx",
                "0.6",
                "--hy",
                "0.2",
                "--hz",
                "0.0",
            ],
            oracle=lambda pts: evaluate_operation_scene(
                "elongate", pts, half_size=np.array([0.6, 0.2, 0.0], dtype=np.float64)
            ),
        ),
        PrimitiveCase(
            name="op_repeat",
            command="evaluate-operation-scene",
            flags=[
                "--operation",
                "repeat",
                "--px",
                "1.5",
                "--py",
                "1.0",
                "--pz",
                "0.0",
            ],
            oracle=lambda pts: evaluate_operation_scene(
                "repeat", pts, period=np.array([1.5, 1.0, 0.0], dtype=np.float64)
            ),
        ),
    ]


def transform_cases() -> list[PrimitiveCase]:
    return [
        PrimitiveCase(
            name="tx_translate",
            command="evaluate-transform-scene",
            flags=["--transform", "translate"],
            oracle=lambda pts: evaluate_transform_scene("translate", pts),
        ),
        PrimitiveCase(
            name="tx_rotate",
            command="evaluate-transform-scene",
            flags=["--transform", "rotate", "--angle", "0.6"],
            oracle=lambda pts: evaluate_transform_scene("rotate", pts, angle=0.6),
        ),
        PrimitiveCase(
            name="tx_scale",
            command="evaluate-transform-scene",
            flags=["--transform", "scale", "--factor", "1.5"],
            oracle=lambda pts: evaluate_transform_scene("scale", pts, factor=1.5),
        ),
        PrimitiveCase(
            name="tx_orient",
            command="evaluate-transform-scene",
            flags=["--transform", "orient", "--tx", "1.0", "--ty", "0.0", "--tz", "0.0"],
            oracle=lambda pts: evaluate_transform_scene(
                "orient", pts, target_axis=np.array([1.0, 0.0, 0.0], dtype=np.float64)
            ),
        ),
        PrimitiveCase(
            name="tx_mirror",
            command="evaluate-transform-scene",
            flags=["--transform", "mirror", "--nx", "0.0", "--ny", "1.0", "--nz", "0.0", "--offset", "0.0"],
            oracle=lambda pts: evaluate_transform_scene(
                "mirror",
                pts,
                normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                offset=0.0,
            ),
        ),
        PrimitiveCase(
            name="tx_twist",
            command="evaluate-transform-scene",
            flags=["--transform", "twist", "--rate", "0.2"],
            oracle=lambda pts: evaluate_transform_scene("twist", pts, rate=0.2),
        ),
        PrimitiveCase(
            name="tx_bend",
            command="evaluate-transform-scene",
            flags=["--transform", "bend", "--rate", "0.15"],
            oracle=lambda pts: evaluate_transform_scene("bend", pts, rate=0.15),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--bounds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--primitive",
        choices=[
            "all",
            "primitives",
            "operations",
            "transforms",
            "sphere",
            "box",
            "rounded_box",
            "cylinder",
            "capped_cylinder",
            "torus",
            "plane",
            "capsule",
            "capped_cone",
            "rounded_cylinder",
            "op_union",
            "op_intersection",
            "op_difference",
            "op_smooth_union",
            "op_smooth_intersection",
            "op_smooth_difference",
            "op_negate",
            "op_shell",
            "op_elongate",
            "op_repeat",
            "tx_translate",
            "tx_rotate",
            "tx_scale",
            "tx_orient",
            "tx_mirror",
            "tx_twist",
            "tx_bend",
        ],
        default="all",
    )
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    points = generate_test_points(args.count, args.bounds, args.seed)
    cases = primitive_cases() + operation_cases() + transform_cases()

    if args.primitive == "primitives":
        cases = primitive_cases()
    elif args.primitive == "operations":
        cases = operation_cases()
    elif args.primitive == "transforms":
        cases = transform_cases()
    elif args.primitive != "all":
        cases = [case for case in cases if case.name == args.primitive]

    for case in cases:
        rust_values = run_rust_primitive(workspace_root, case.command, case.flags, points)
        python_values = case.oracle(points)
        tolerance = args.tolerance if args.tolerance is not None else case.tolerance
        compare(rust_values, python_values, tolerance, name=case.name)

    print(f"Cross-language comparison passed for {len(cases)} case(s).")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.CalledProcessError) as err:
        print(err, file=sys.stderr)
        raise SystemExit(1) from err
