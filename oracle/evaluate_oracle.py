#!/usr/bin/env python3
"""Python oracle evaluation for SDF primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


def evaluate_sphere(radius: float, points: np.ndarray) -> np.ndarray:
    """Evaluate sphere SDF at each point."""
    return np.linalg.norm(points, axis=1) - radius


def evaluate_box(half_extents: np.ndarray, points: np.ndarray) -> np.ndarray:
    q = np.abs(points) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def evaluate_rounded_box(half_extents: np.ndarray, radius: float, points: np.ndarray) -> np.ndarray:
    return evaluate_box(half_extents, points) - radius


def evaluate_capped_cylinder(radius: float, half_height: float, points: np.ndarray) -> np.ndarray:
    radial = np.linalg.norm(points[:, :2], axis=1) - radius
    axial = np.abs(points[:, 2]) - half_height
    outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside


def evaluate_cylinder(radius: float, height: float, points: np.ndarray) -> np.ndarray:
    return evaluate_capped_cylinder(radius, height * 0.5, points)


def evaluate_torus(major_radius: float, minor_radius: float, points: np.ndarray) -> np.ndarray:
    qx = np.linalg.norm(points[:, :2], axis=1) - major_radius
    return np.sqrt(qx * qx + points[:, 2] * points[:, 2]) - minor_radius


def evaluate_plane(normal: np.ndarray, offset: float, points: np.ndarray) -> np.ndarray:
    return points @ normal - offset


def evaluate_capsule(a: np.ndarray, b: np.ndarray, radius: float, points: np.ndarray) -> np.ndarray:
    pa = points - a
    ba = b - a
    ba_dot = float(np.dot(ba, ba))
    if ba_dot <= np.finfo(np.float64).eps:
        return np.linalg.norm(pa, axis=1) - radius
    h = np.clip((pa @ ba) / ba_dot, 0.0, 1.0)
    return np.linalg.norm(pa - np.outer(h, ba), axis=1) - radius


def evaluate_capped_cone(radius1: float, radius2: float, height: float, points: np.ndarray) -> np.ndarray:
    h = height * 0.5
    q = np.column_stack((np.linalg.norm(points[:, :2], axis=1), points[:, 2]))
    k1 = np.array([radius2, h], dtype=np.float64)
    k2 = np.array([radius2 - radius1, 2.0 * h], dtype=np.float64)

    cap_radius = np.where(q[:, 1] < 0.0, radius1, radius2)
    ca = np.column_stack((q[:, 0] - np.minimum(q[:, 0], cap_radius), np.abs(q[:, 1]) - h))

    k2_dot = float(np.dot(k2, k2))
    if k2_dot <= np.finfo(np.float64).eps:
        h_proj = np.zeros(len(points), dtype=np.float64)
    else:
        h_proj = np.clip(
            ((k1[0] - q[:, 0]) * k2[0] + (k1[1] - q[:, 1]) * k2[1]) / k2_dot,
            0.0,
            1.0,
        )
    cb = np.column_stack((q[:, 0] - k1[0] + k2[0] * h_proj, q[:, 1] - k1[1] + k2[1] * h_proj))

    s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
    ca_dot = np.sum(ca * ca, axis=1)
    cb_dot = np.sum(cb * cb, axis=1)
    return s * np.sqrt(np.minimum(ca_dot, cb_dot))


def evaluate_rounded_cylinder(
    radius: float, height: float, edge_radius: float, points: np.ndarray
) -> np.ndarray:
    core_radius = max(radius - edge_radius, 0.0)
    core_half_height = max(height * 0.5 - edge_radius, 0.0)
    return evaluate_capped_cylinder(core_radius, core_half_height, points) - edge_radius


def op_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def op_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def op_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, -b)


def _clamp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(values, low), high)


def _mix(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a * (1.0 - t) + b * t


def op_smooth_union(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 0.0:
        return op_union(a, b)
    h = _clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return _mix(b, a, h) - k * h * (1.0 - h)


def op_smooth_intersection(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -op_smooth_union(-a, -b, k)


def op_smooth_difference(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -op_smooth_union(-a, b, k)


def op_negate(distance: np.ndarray) -> np.ndarray:
    return -distance


def op_shell(distance: np.ndarray, thickness: float) -> np.ndarray:
    return np.abs(distance) - thickness


def op_repeat_points(points: np.ndarray, period: np.ndarray) -> np.ndarray:
    q = points.copy()
    for axis in range(3):
        c = abs(float(period[axis]))
        if c > np.finfo(np.float64).eps:
            q[:, axis] = np.remainder(q[:, axis] + 0.5 * c, c) - 0.5 * c
    return q


def op_elongate(points: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    q = np.abs(points) - half_size
    local = np.maximum(q, 0.0)
    base = evaluate_sphere(0.6, local)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return base + inside


def evaluate_operation_scene(
    operation: str,
    points: np.ndarray,
    *,
    k: float = 0.2,
    thickness: float = 0.1,
    half_size: np.ndarray | None = None,
    period: np.ndarray | None = None,
) -> np.ndarray:
    a = evaluate_sphere(1.0, points)
    shifted = points.copy()
    shifted[:, 0] = shifted[:, 0] - 0.35
    shifted[:, 1] = shifted[:, 1] + 0.15
    b = evaluate_box(np.array([0.8, 0.6, 0.9], dtype=np.float64), shifted)

    if operation == "union":
        return op_union(a, b)
    if operation == "intersection":
        return op_intersection(a, b)
    if operation == "difference":
        return op_difference(a, b)
    if operation == "smooth_union":
        return op_smooth_union(a, b, k)
    if operation == "smooth_intersection":
        return op_smooth_intersection(a, b, k)
    if operation == "smooth_difference":
        return op_smooth_difference(a, b, k)
    if operation == "negate":
        return op_negate(a)
    if operation == "shell":
        return op_shell(a, thickness)
    if operation == "elongate":
        if half_size is None:
            half_size = np.array([0.6, 0.2, 0.0], dtype=np.float64)
        return op_elongate(points, half_size)
    if operation == "repeat":
        if period is None:
            period = np.array([1.5, 1.0, 0.0], dtype=np.float64)
        repeated = op_repeat_points(points, period)
        return evaluate_sphere(0.55, repeated)

    raise ValueError(f"unsupported operation: {operation}")


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= np.finfo(np.float64).eps:
        return np.zeros(3, dtype=np.float64)
    return v / norm


def _rotate_axis(points: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    a = _normalize(axis)
    if np.linalg.norm(a) <= np.finfo(np.float64).eps:
        return points.copy()
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    term1 = points * cos_t
    term2 = np.cross(np.broadcast_to(a, points.shape), points) * sin_t
    dot_ap = points @ a
    term3 = np.outer(dot_ap * (1.0 - cos_t), a)
    return term1 + term2 + term3


def inverse_translate(points: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return points - offset


def inverse_rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    x = c * points[:, 0] + s * points[:, 1]
    y = -s * points[:, 0] + c * points[:, 1]
    return np.column_stack((x, y, points[:, 2]))


def inverse_rotate_x(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    y = c * points[:, 1] + s * points[:, 2]
    z = -s * points[:, 1] + c * points[:, 2]
    return np.column_stack((points[:, 0], y, z))


def inverse_orient(points: np.ndarray, from_axis: np.ndarray, to_axis: np.ndarray) -> np.ndarray:
    from_n = _normalize(from_axis)
    to_n = _normalize(to_axis)
    axis = np.cross(from_n, to_n)
    axis_len = float(np.linalg.norm(axis))
    cosine = float(np.clip(np.dot(from_n, to_n), -1.0, 1.0))

    if axis_len <= 1e-12:
        if cosine > 0.0:
            return points.copy()
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(to_n[0]) < 0.9 else np.array(
            [0.0, 1.0, 0.0], dtype=np.float64
        )
        ortho = _normalize(np.cross(to_n, helper))
        return _rotate_axis(points, ortho, np.pi)

    angle = np.arccos(cosine)
    return _rotate_axis(points, axis, -angle)


def mirror_points(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    n = _normalize(normal)
    d = points @ n - offset
    return points - 2.0 * np.outer(d, n)


def inverse_twist_z(points: np.ndarray, rate: float) -> np.ndarray:
    angles = rate * points[:, 2]
    return inverse_rotate_z_variable(points, angles)


def inverse_rotate_z_variable(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
    c = np.cos(angles)
    s = np.sin(angles)
    x = c * points[:, 0] + s * points[:, 1]
    y = -s * points[:, 0] + c * points[:, 1]
    return np.column_stack((x, y, points[:, 2]))


def inverse_bend_x(points: np.ndarray, rate: float) -> np.ndarray:
    angles = rate * points[:, 0]
    return inverse_rotate_x(points, angles)


def evaluate_transform_scene(
    transform: str,
    points: np.ndarray,
    *,
    angle: float = 0.6,
    factor: float = 1.5,
    target_axis: np.ndarray | None = None,
    normal: np.ndarray | None = None,
    offset: float = 0.0,
    rate: float = 0.2,
) -> np.ndarray:
    if transform == "translate":
        local = inverse_translate(points, np.array([0.7, -0.4, 0.5], dtype=np.float64))
        return evaluate_sphere(1.0, local)
    if transform == "rotate":
        return evaluate_sphere(1.0, inverse_rotate_z(points, angle))
    if transform == "scale":
        local = points / factor
        return evaluate_sphere(1.0, local) * factor
    if transform == "orient":
        if target_axis is None:
            target_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        local = inverse_orient(points, np.array([0.0, 1.0, 0.0], dtype=np.float64), target_axis)
        return evaluate_sphere(1.0, local)
    if transform == "mirror":
        if normal is None:
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        local = mirror_points(points, normal, offset)
        return evaluate_sphere(1.0, local)
    if transform == "twist":
        local = inverse_twist_z(points, rate)
        return evaluate_sphere(1.0, local)
    if transform == "bend":
        local = inverse_bend_x(points, rate)
        return evaluate_sphere(1.0, local)
    raise ValueError(f"unsupported transform: {transform}")


def _require(name: str, value: float | None) -> float:
    if value is None:
        raise ValueError(f"missing required argument: {name}")
    return float(value)


def _primitive_evaluator(args: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
    if args.primitive == "sphere":
        radius = _require("--radius", args.radius)
        return lambda pts: evaluate_sphere(radius, pts)
    if args.primitive == "box":
        half_extents = np.array(
            [_require("--hx", args.hx), _require("--hy", args.hy), _require("--hz", args.hz)],
            dtype=np.float64,
        )
        return lambda pts: evaluate_box(half_extents, pts)
    if args.primitive == "rounded_box":
        half_extents = np.array(
            [_require("--hx", args.hx), _require("--hy", args.hy), _require("--hz", args.hz)],
            dtype=np.float64,
        )
        radius = _require("--radius", args.radius)
        return lambda pts: evaluate_rounded_box(half_extents, radius, pts)
    if args.primitive == "cylinder":
        radius = _require("--radius", args.radius)
        height = _require("--height", args.height)
        return lambda pts: evaluate_cylinder(radius, height, pts)
    if args.primitive == "capped_cylinder":
        radius = _require("--radius", args.radius)
        half_height = _require("--half-height", args.half_height)
        return lambda pts: evaluate_capped_cylinder(radius, half_height, pts)
    if args.primitive == "torus":
        major_radius = _require("--major-radius", args.major_radius)
        minor_radius = _require("--minor-radius", args.minor_radius)
        return lambda pts: evaluate_torus(major_radius, minor_radius, pts)
    if args.primitive == "plane":
        normal = np.array(
            [_require("--nx", args.nx), _require("--ny", args.ny), _require("--nz", args.nz)],
            dtype=np.float64,
        )
        offset = _require("--offset", args.offset)
        return lambda pts: evaluate_plane(normal, offset, pts)
    if args.primitive == "capsule":
        a = np.array(
            [_require("--ax", args.ax), _require("--ay", args.ay), _require("--az", args.az)],
            dtype=np.float64,
        )
        b = np.array(
            [_require("--bx", args.bx), _require("--by", args.by), _require("--bz", args.bz)],
            dtype=np.float64,
        )
        radius = _require("--radius", args.radius)
        return lambda pts: evaluate_capsule(a, b, radius, pts)
    if args.primitive == "capped_cone":
        radius1 = _require("--radius1", args.radius1)
        radius2 = _require("--radius2", args.radius2)
        height = _require("--height", args.height)
        return lambda pts: evaluate_capped_cone(radius1, radius2, height, pts)
    if args.primitive == "rounded_cylinder":
        radius = _require("--radius", args.radius)
        height = _require("--height", args.height)
        edge_radius = _require("--edge-radius", args.edge_radius)
        return lambda pts: evaluate_rounded_cylinder(radius, height, edge_radius, pts)
    raise ValueError(f"unsupported primitive: {args.primitive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primitive",
        choices=[
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
        ],
        required=True,
    )
    parser.add_argument("--radius", type=float)
    parser.add_argument("--hx", type=float)
    parser.add_argument("--hy", type=float)
    parser.add_argument("--hz", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--half-height", dest="half_height", type=float)
    parser.add_argument("--major-radius", dest="major_radius", type=float)
    parser.add_argument("--minor-radius", dest="minor_radius", type=float)
    parser.add_argument("--nx", type=float)
    parser.add_argument("--ny", type=float)
    parser.add_argument("--nz", type=float)
    parser.add_argument("--offset", type=float)
    parser.add_argument("--ax", type=float)
    parser.add_argument("--ay", type=float)
    parser.add_argument("--az", type=float)
    parser.add_argument("--bx", type=float)
    parser.add_argument("--by", type=float)
    parser.add_argument("--bz", type=float)
    parser.add_argument("--radius1", type=float)
    parser.add_argument("--radius2", type=float)
    parser.add_argument("--edge-radius", dest="edge_radius", type=float)
    parser.add_argument("--points-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.points_file.read_text(encoding="utf-8"))
    points = np.asarray(payload["points"], dtype=np.float64)
    evaluator = _primitive_evaluator(args)
    values = evaluator(points)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"values": values.tolist()}), encoding="utf-8")


if __name__ == "__main__":
    main()
