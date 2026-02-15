#!/usr/bin/env python3
"""Mesh-level oracle checks for marching cubes and export formats."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import sdf
import trimesh


def run_cli(workspace_root: Path, args: list[str]) -> str:
    cmd = ["cargo", "run", "--quiet", "-p", "sdf-cli", "--", *args]
    completed = subprocess.run(
        cmd,
        cwd=workspace_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def parse_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in output.strip().splitlines():
        name, value = line.split(maxsplit=1)
        metrics[name] = float(value)
    return metrics


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(loaded)}")
    return loaded


def generate_python_mesh_sphere(path: Path, bounds: tuple[tuple[float, float, float], tuple[float, float, float]], step: float) -> trimesh.Trimesh:
    sdf.sphere(1.0).save(str(path), bounds=bounds, step=step, sparse=False, workers=1, verbose=False)
    return load_mesh(path)


def generate_python_mesh_union(path: Path, bounds: tuple[tuple[float, float, float], tuple[float, float, float]], step: float) -> trimesh.Trimesh:
    shape = sdf.union(
        sdf.sphere(1.0),
        sdf.box(size=(1.6, 1.2, 1.8), center=(0.35, -0.15, 0.0)),
    )
    shape.save(str(path), bounds=bounds, step=step, sparse=False, workers=1, verbose=False)
    return load_mesh(path)


def assert_close_ratio(name: str, rust_value: float, py_value: float, tolerance: float) -> None:
    if abs(py_value) <= np.finfo(np.float64).eps:
        raise AssertionError(f"{name}: python reference value is zero")
    rel = abs(rust_value - py_value) / abs(py_value)
    print(f"[{name}] rust={rust_value:.6f} python={py_value:.6f} rel={rel:.4%}")
    if rel > tolerance:
        raise AssertionError(f"{name}: relative error {rel:.4%} exceeds {tolerance:.2%}")


def assert_triangle_count_within(name: str, rust_count: int, py_count: int, tolerance: float) -> None:
    if py_count == 0:
        raise AssertionError(f"{name}: python triangle count is zero")
    rel = abs(rust_count - py_count) / py_count
    print(f"[{name}] rust={rust_count} python={py_count} rel={rel:.4%}")
    if rel > tolerance:
        raise AssertionError(f"{name}: triangle count delta {rel:.4%} exceeds {tolerance:.2%}")


def assert_mesh_quality(mesh: trimesh.Trimesh, name: str) -> None:
    if len(mesh.faces) == 0:
        raise AssertionError(f"{name}: empty mesh")
    if not mesh.is_watertight:
        raise AssertionError(f"{name}: mesh is not watertight")
    min_area = float(mesh.area_faces.min())
    if min_area <= 1e-10:
        raise AssertionError(f"{name}: degenerate triangle area {min_area:e}")


def compare_sphere(
    workspace_root: Path,
    tmp: Path,
    resolution: int,
    bounds: float,
) -> None:
    rust_path = tmp / "rust_sphere.stl"
    run_cli(
        workspace_root,
        [
            "export-mesh",
            "--scene",
            "sphere",
            "--resolution",
            str(resolution),
            "--bounds",
            str(bounds),
            "--format",
            "binary-stl",
            "--name",
            "sphere",
            "--output",
            str(rust_path),
        ],
    )
    rust_mesh = load_mesh(rust_path)
    assert_mesh_quality(rust_mesh, "rust_sphere")

    step = (2.0 * bounds) / (resolution - 1)
    py_path = tmp / "py_sphere.stl"
    py_mesh = generate_python_mesh_sphere(
        py_path,
        ((-bounds, -bounds, -bounds), (bounds, bounds, bounds)),
        step,
    )
    assert_mesh_quality(py_mesh, "python_sphere")

    assert_triangle_count_within("sphere_triangle_count", len(rust_mesh.faces), len(py_mesh.faces), 0.05)
    assert_close_ratio("sphere_volume", abs(float(rust_mesh.volume)), abs(float(py_mesh.volume)), 0.01)
    assert_close_ratio("sphere_area", float(rust_mesh.area), float(py_mesh.area), 0.02)


def compare_union(
    workspace_root: Path,
    tmp: Path,
    resolution: int,
    bounds: float,
) -> None:
    rust_path = tmp / "rust_union.stl"
    run_cli(
        workspace_root,
        [
            "export-mesh",
            "--scene",
            "union_sphere_box",
            "--resolution",
            str(resolution),
            "--bounds",
            str(bounds),
            "--format",
            "binary-stl",
            "--name",
            "union",
            "--output",
            str(rust_path),
        ],
    )
    rust_mesh = load_mesh(rust_path)
    assert_mesh_quality(rust_mesh, "rust_union")

    step = (2.0 * bounds) / (resolution - 1)
    py_path = tmp / "py_union.stl"
    py_mesh = generate_python_mesh_union(
        py_path,
        ((-bounds, -bounds, -bounds), (bounds, bounds, bounds)),
        step,
    )
    assert_mesh_quality(py_mesh, "python_union")

    assert_close_ratio("union_volume", abs(float(rust_mesh.volume)), abs(float(py_mesh.volume)), 0.01)


def validate_exports(
    workspace_root: Path,
    tmp: Path,
    resolution: int,
    bounds: float,
) -> None:
    formats = [
        ("binary-stl", "mesh.stl"),
        ("ascii-stl", "mesh_ascii.stl"),
        ("obj", "mesh.obj"),
    ]
    for fmt, filename in formats:
        out_path = tmp / filename
        run_cli(
            workspace_root,
            [
                "export-mesh",
                "--scene",
                "sphere",
                "--resolution",
                str(resolution),
                "--bounds",
                str(bounds),
                "--format",
                fmt,
                "--name",
                "sphere",
                "--output",
                str(out_path),
            ],
        )
        mesh = load_mesh(out_path)
        if len(mesh.faces) == 0:
            raise AssertionError(f"{fmt}: exported mesh has no faces")
        print(f"[{fmt}] loaded faces={len(mesh.faces)} vertices={len(mesh.vertices)}")

    a = tmp / "deterministic_a.stl"
    b = tmp / "deterministic_b.stl"
    shared_args = [
        "export-mesh",
        "--scene",
        "sphere",
        "--resolution",
        str(resolution),
        "--bounds",
        str(bounds),
        "--format",
        "binary-stl",
        "--name",
        "deterministic",
    ]
    run_cli(workspace_root, [*shared_args, "--output", str(a)])
    run_cli(workspace_root, [*shared_args, "--output", str(b)])
    if a.read_bytes() != b.read_bytes():
        raise AssertionError("binary STL export is not deterministic")
    print("[binary_stl_determinism] passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--bounds", type=float, default=1.5)
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        compare_sphere(workspace_root, tmp, args.resolution, args.bounds)
        compare_union(workspace_root, tmp, args.resolution, args.bounds)
        validate_exports(workspace_root, tmp, args.resolution, args.bounds)

    print("Mesh oracle checks passed.")


if __name__ == "__main__":
    main()
