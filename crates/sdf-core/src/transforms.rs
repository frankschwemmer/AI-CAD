use crate::primitives::{Point3, Sdf3};

#[inline]
fn dot(a: Point3, b: Point3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: Point3, b: Point3) -> Point3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn length(v: Point3) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn normalize(v: Point3) -> Point3 {
    let len = length(v);
    if len <= f64::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[inline]
fn rotate_axis(point: Point3, axis: Point3, angle: f64) -> Point3 {
    let a = normalize(axis);
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    let term1 = [
        point[0] * cos_theta,
        point[1] * cos_theta,
        point[2] * cos_theta,
    ];
    let cross_term = cross(a, point);
    let term2 = [
        cross_term[0] * sin_theta,
        cross_term[1] * sin_theta,
        cross_term[2] * sin_theta,
    ];
    let dot_term = dot(a, point) * (1.0 - cos_theta);
    let term3 = [a[0] * dot_term, a[1] * dot_term, a[2] * dot_term];

    [
        term1[0] + term2[0] + term3[0],
        term1[1] + term2[1] + term3[1],
        term1[2] + term2[2] + term3[2],
    ]
}

/// Applies inverse translation to a point for transformed SDF evaluation.
#[inline]
pub fn inverse_translate(point: Point3, offset: Point3) -> Point3 {
    [
        point[0] - offset[0],
        point[1] - offset[1],
        point[2] - offset[2],
    ]
}

/// Applies inverse rotation around X to a point.
#[inline]
pub fn inverse_rotate_x(point: Point3, angle: f64) -> Point3 {
    let c = angle.cos();
    let s = angle.sin();
    [point[0], c * point[1] + s * point[2], -s * point[1] + c * point[2]]
}

/// Applies inverse rotation around Y to a point.
#[inline]
pub fn inverse_rotate_y(point: Point3, angle: f64) -> Point3 {
    let c = angle.cos();
    let s = angle.sin();
    [c * point[0] - s * point[2], point[1], s * point[0] + c * point[2]]
}

/// Applies inverse rotation around Z to a point.
#[inline]
pub fn inverse_rotate_z(point: Point3, angle: f64) -> Point3 {
    let c = angle.cos();
    let s = angle.sin();
    [c * point[0] + s * point[1], -s * point[0] + c * point[1], point[2]]
}

/// Uniform scaling transform that preserves the signed-distance property.
#[inline]
pub fn scale<S>(sdf: &S, point: Point3, scale: f64) -> f64
where
    S: Sdf3,
{
    assert!(scale > 0.0, "uniform scale must be positive");
    let local = [point[0] / scale, point[1] / scale, point[2] / scale];
    sdf.evaluate(local) * scale
}

/// Applies inverse orientation from `to_axis` back to `from_axis`.
#[inline]
pub fn inverse_orient(point: Point3, from_axis: Point3, to_axis: Point3) -> Point3 {
    let from = normalize(from_axis);
    let to = normalize(to_axis);
    let axis = cross(from, to);
    let axis_len = length(axis);
    let cosine = dot(from, to).clamp(-1.0, 1.0);

    if axis_len <= 1e-12 {
        if cosine > 0.0 {
            return point;
        }
        // 180 degree flip around any axis orthogonal to `to`.
        let helper = if to[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let ortho = normalize(cross(to, helper));
        return rotate_axis(point, ortho, std::f64::consts::PI);
    }

    let angle = cosine.acos();
    // Inverse orientation uses negative angle.
    rotate_axis(point, axis, -angle)
}

/// Reflects point across plane `dot(p, normal) = offset`.
#[inline]
pub fn mirror_point(point: Point3, normal: Point3, offset: f64) -> Point3 {
    let n = normalize(normal);
    let distance = dot(point, n) - offset;
    [
        point[0] - 2.0 * distance * n[0],
        point[1] - 2.0 * distance * n[1],
        point[2] - 2.0 * distance * n[2],
    ]
}

/// Twist around Z as a function of height (`z`).
#[inline]
pub fn inverse_twist_z(point: Point3, rate: f64) -> Point3 {
    inverse_rotate_z(point, rate * point[2])
}

/// Bend around X where the bend angle scales with `x`.
#[inline]
pub fn inverse_bend_x(point: Point3, rate: f64) -> Point3 {
    inverse_rotate_x(point, rate * point[0])
}

#[cfg(test)]
mod tests {
    use crate::primitives::{Sdf3, sphere};

    use super::{
        inverse_bend_x, inverse_orient, inverse_rotate_z, inverse_translate, inverse_twist_z,
        mirror_point, scale,
    };

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() <= eps,
            "expected {b}, got {a}, eps={eps}"
        );
    }

    #[test]
    fn inverse_translate_moves_point_into_local_space() {
        let point = [10.0, 5.0, -1.0];
        let offset = [2.0, 3.0, -4.0];
        let translated = inverse_translate(point, offset);
        assert_eq!(translated, [8.0, 2.0, 3.0]);
    }

    #[test]
    fn rotate_preserves_sphere_distances() {
        let sdf = sphere(1.0);
        let p = [0.2, 0.8, -0.3];
        let rotated_world = [0.8, -0.2, -0.3];
        let rotated_back = inverse_rotate_z(rotated_world, std::f64::consts::FRAC_PI_2);
        assert_close(
            sdf.evaluate(rotated_back),
            sdf.evaluate(p),
            1e-12,
        );
    }

    #[test]
    fn uniform_scale_preserves_sdf_relation() {
        let sdf = sphere(1.0);
        let value = scale(&sdf, [0.0, 0.0, 0.0], 2.0);
        assert_close(value, -2.0, 1e-12);
    }

    #[test]
    fn inverse_orient_aligns_axes() {
        let point = [0.0, 1.0, 0.0];
        let from = [0.0, 1.0, 0.0];
        let to = [1.0, 0.0, 0.0];
        let local = inverse_orient(point, from, to);
        assert_close(local[0], -1.0, 1e-12);
        assert_close(local[1], 0.0, 1e-12);
    }

    #[test]
    fn mirror_reflects_across_plane() {
        let mirrored = mirror_point([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], 0.0);
        assert_eq!(mirrored, [1.0, -2.0, 3.0]);
    }

    #[test]
    fn twist_and_bend_modify_points_continuously() {
        let p = [1.0, 0.5, 2.0];
        let twisted = inverse_twist_z(p, 0.2);
        let bent = inverse_bend_x(p, 0.15);
        assert!(twisted[0].is_finite() && twisted[1].is_finite() && twisted[2].is_finite());
        assert!(bent[0].is_finite() && bent[1].is_finite() && bent[2].is_finite());
    }
}
