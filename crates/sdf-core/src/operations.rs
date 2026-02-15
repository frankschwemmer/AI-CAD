use crate::primitives::{Point3, Sdf3};

#[inline]
fn max_component(v: Point3) -> f64 {
    v[0].max(v[1]).max(v[2])
}

#[inline]
fn clamp(value: f64, low: f64, high: f64) -> f64 {
    value.max(low).min(high)
}

#[inline]
fn mix(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

/// Exact CSG union for true SDFs.
#[inline]
pub fn union(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Exact CSG intersection for true SDFs.
#[inline]
pub fn intersection(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Exact CSG difference for true SDFs.
#[inline]
pub fn difference(a: f64, b: f64) -> f64 {
    a.max(-b)
}

/// Smooth CSG union using a polynomial smooth-min blend radius `k`.
#[inline]
pub fn smooth_union(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return union(a, b);
    }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    mix(b, a, h) - k * h * (1.0 - h)
}

/// Smooth CSG intersection (dual of smooth union).
#[inline]
pub fn smooth_intersection(a: f64, b: f64, k: f64) -> f64 {
    -smooth_union(-a, -b, k)
}

/// Smooth CSG difference (dual form based on smooth union).
#[inline]
pub fn smooth_difference(a: f64, b: f64, k: f64) -> f64 {
    -smooth_union(-a, b, k)
}

/// Sign inversion for SDF values.
#[inline]
pub fn negate(distance: f64) -> f64 {
    -distance
}

/// Shell operation around a surface with fixed thickness.
#[inline]
pub fn shell(distance: f64, thickness: f64) -> f64 {
    assert!(thickness >= 0.0, "shell thickness must be non-negative");
    distance.abs() - thickness
}

/// Domain repetition for infinite tiling.
#[inline]
pub fn repeat_point(point: Point3, period: Point3) -> Point3 {
    fn repeat_axis(value: f64, axis_period: f64) -> f64 {
        let c = axis_period.abs();
        if c <= f64::EPSILON {
            value
        } else {
            (value + 0.5 * c).rem_euclid(c) - 0.5 * c
        }
    }

    [
        repeat_axis(point[0], period[0]),
        repeat_axis(point[1], period[1]),
        repeat_axis(point[2], period[2]),
    ]
}

/// Axis elongation operation.
#[inline]
pub fn elongate<S>(sdf: &S, point: Point3, half_size: Point3) -> f64
where
    S: Sdf3,
{
    assert!(
        half_size[0] >= 0.0 && half_size[1] >= 0.0 && half_size[2] >= 0.0,
        "elongation half-size must be non-negative"
    );

    let q = [
        point[0].abs() - half_size[0],
        point[1].abs() - half_size[1],
        point[2].abs() - half_size[2],
    ];
    let local = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    sdf.evaluate(local) + max_component(q).min(0.0)
}

#[cfg(test)]
mod tests {
    use crate::primitives::{Sdf3, box3, sphere};

    use super::{
        difference, elongate, intersection, negate, repeat_point, shell, smooth_difference,
        smooth_intersection, smooth_union, union,
    };

    fn sample_points() -> &'static [[f64; 3]] {
        &[
            [0.0, 0.0, 0.0],
            [0.3, -0.7, 0.2],
            [1.2, -0.5, 0.8],
            [-1.5, 0.4, -0.9],
            [2.0, 1.0, -1.2],
        ]
    }

    fn scene_values(point: [f64; 3]) -> (f64, f64) {
        let a = sphere(1.0).evaluate(point);
        let shifted = [point[0] - 0.35, point[1] + 0.15, point[2]];
        let b = box3([0.8, 0.6, 0.9]).evaluate(shifted);
        (a, b)
    }

    fn assert_close(actual: f64, expected: f64, eps: f64) {
        assert!(
            (actual - expected).abs() <= eps,
            "expected {expected}, got {actual}, eps={eps}"
        );
    }

    #[test]
    fn csg_scalar_ops_match_definitions() {
        assert_eq!(union(1.0, -2.0), -2.0);
        assert_eq!(intersection(1.0, -2.0), 1.0);
        assert_eq!(difference(1.0, -2.0), 2.0);
    }

    #[test]
    fn boolean_identities_hold_on_sample_scene() {
        for point in sample_points() {
            let (a, b) = scene_values(*point);
            assert_close(union(a, a), a, 1e-12);
            assert_close(intersection(a, a), a, 1e-12);
            assert!(difference(a, a) >= -1e-12);
            assert_close(union(a, b), union(b, a), 1e-12);
            assert_close(intersection(a, b), intersection(b, a), 1e-12);
            assert_close(negate(negate(a)), a, 1e-12);
        }
    }

    #[test]
    fn smooth_operations_converge_to_sharp_for_small_k() {
        for point in sample_points() {
            let (a, b) = scene_values(*point);
            let smooth = smooth_union(a, b, 0.001);
            let sharp = union(a, b);
            assert!((smooth - sharp).abs() < 0.01);
        }
    }

    #[test]
    fn smooth_duals_are_consistent() {
        for point in sample_points() {
            let (a, b) = scene_values(*point);
            assert_close(smooth_intersection(a, b, 0.2), -smooth_union(-a, -b, 0.2), 1e-12);
            assert_close(smooth_difference(a, b, 0.2), -smooth_union(-a, b, 0.2), 1e-12);
        }
    }

    #[test]
    fn shell_creates_band_around_surface() {
        assert_close(shell(-0.2, 0.1), 0.1, 1e-12);
        assert_close(shell(0.1, 0.1), 0.0, 1e-12);
        assert_close(shell(0.35, 0.1), 0.25, 1e-12);
    }

    #[test]
    fn repeat_point_is_periodic() {
        let period = [2.0, 3.0, 0.0];
        let p = [2.3, -4.1, 1.5];
        let repeated = repeat_point(p, period);
        let repeated_shifted = repeat_point([p[0] + 2.0, p[1] - 3.0, p[2]], period);
        assert_close(repeated[0], repeated_shifted[0], 1e-12);
        assert_close(repeated[1], repeated_shifted[1], 1e-12);
        assert_close(repeated[2], repeated_shifted[2], 1e-12);
    }

    #[test]
    fn elongate_extends_shape_along_axes() {
        let base = sphere(0.6);
        assert_close(elongate(&base, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), -0.6, 1e-12);
        assert_close(elongate(&base, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), -0.6, 1e-12);
        assert!(elongate(&base, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]) > 0.0);
    }
}
