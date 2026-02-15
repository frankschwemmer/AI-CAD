/// Cartesian point used for SDF evaluation.
pub type Point3 = [f64; 3];

/// Trait for 3D signed distance fields.
pub trait Sdf3 {
    fn evaluate(&self, point: Point3) -> f64;
}

#[inline]
fn dot(a: Point3, b: Point3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn length(v: Point3) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn length2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

#[inline]
fn sub(a: Point3, b: Point3) -> Point3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn mul_scalar(v: Point3, scalar: f64) -> Point3 {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

#[inline]
fn max_component(v: Point3) -> f64 {
    v[0].max(v[1]).max(v[2])
}

#[inline]
fn clamp(value: f64, low: f64, high: f64) -> f64 {
    value.max(low).min(high)
}

/// Sphere SDF primitive.
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    radius: f64,
}

impl Sphere {
    /// Creates a sphere with a non-negative radius.
    pub fn new(radius: f64) -> Self {
        assert!(radius >= 0.0, "sphere radius must be non-negative");
        Self { radius }
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }
}

impl Sdf3 for Sphere {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        length(point) - self.radius
    }
}

/// Convenience constructor for a sphere SDF.
#[inline]
pub fn sphere(radius: f64) -> Sphere {
    Sphere::new(radius)
}

/// Axis-aligned box SDF primitive using half extents per axis.
#[derive(Debug, Clone, Copy)]
pub struct Box3 {
    half_extents: Point3,
}

impl Box3 {
    pub fn new(half_extents: Point3) -> Self {
        assert!(
            half_extents[0] >= 0.0 && half_extents[1] >= 0.0 && half_extents[2] >= 0.0,
            "box half extents must be non-negative"
        );
        Self { half_extents }
    }

    pub fn half_extents(&self) -> Point3 {
        self.half_extents
    }
}

impl Sdf3 for Box3 {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        let q = [
            point[0].abs() - self.half_extents[0],
            point[1].abs() - self.half_extents[1],
            point[2].abs() - self.half_extents[2],
        ];
        let outside = length([q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)]);
        let inside = max_component(q).min(0.0);
        outside + inside
    }
}

#[inline]
pub fn box3(half_extents: Point3) -> Box3 {
    Box3::new(half_extents)
}

/// Rounded box SDF primitive.
#[derive(Debug, Clone, Copy)]
pub struct RoundedBox {
    half_extents: Point3,
    radius: f64,
}

impl RoundedBox {
    pub fn new(half_extents: Point3, radius: f64) -> Self {
        assert!(radius >= 0.0, "rounded box radius must be non-negative");
        Self {
            half_extents,
            radius,
        }
    }
}

impl Sdf3 for RoundedBox {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        box3(self.half_extents).evaluate(point) - self.radius
    }
}

#[inline]
pub fn rounded_box(half_extents: Point3, radius: f64) -> RoundedBox {
    RoundedBox::new(half_extents, radius)
}

/// Finite cylinder centered at origin, aligned with Z axis.
#[derive(Debug, Clone, Copy)]
pub struct Cylinder {
    radius: f64,
    height: f64,
}

impl Cylinder {
    pub fn new(radius: f64, height: f64) -> Self {
        assert!(radius >= 0.0, "cylinder radius must be non-negative");
        assert!(height >= 0.0, "cylinder height must be non-negative");
        Self { radius, height }
    }
}

impl Sdf3 for Cylinder {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        capped_cylinder(self.radius, self.height * 0.5).evaluate(point)
    }
}

#[inline]
pub fn cylinder(radius: f64, height: f64) -> Cylinder {
    Cylinder::new(radius, height)
}

/// Finite cylinder parameterized by radius and half-height.
#[derive(Debug, Clone, Copy)]
pub struct CappedCylinder {
    radius: f64,
    half_height: f64,
}

impl CappedCylinder {
    pub fn new(radius: f64, half_height: f64) -> Self {
        assert!(radius >= 0.0, "capped cylinder radius must be non-negative");
        assert!(
            half_height >= 0.0,
            "capped cylinder half-height must be non-negative"
        );
        Self {
            radius,
            half_height,
        }
    }
}

impl Sdf3 for CappedCylinder {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        let d = [
            (point[0] * point[0] + point[1] * point[1]).sqrt() - self.radius,
            point[2].abs() - self.half_height,
        ];
        let outside = length2([d[0].max(0.0), d[1].max(0.0)]);
        let inside = d[0].max(d[1]).min(0.0);
        outside + inside
    }
}

#[inline]
pub fn capped_cylinder(radius: f64, half_height: f64) -> CappedCylinder {
    CappedCylinder::new(radius, half_height)
}

/// Torus centered at origin, aligned with Z axis.
#[derive(Debug, Clone, Copy)]
pub struct Torus {
    major_radius: f64,
    minor_radius: f64,
}

impl Torus {
    pub fn new(major_radius: f64, minor_radius: f64) -> Self {
        assert!(major_radius >= 0.0, "torus major radius must be non-negative");
        assert!(minor_radius >= 0.0, "torus minor radius must be non-negative");
        Self {
            major_radius,
            minor_radius,
        }
    }
}

impl Sdf3 for Torus {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        let qx = (point[0] * point[0] + point[1] * point[1]).sqrt() - self.major_radius;
        (qx * qx + point[2] * point[2]).sqrt() - self.minor_radius
    }
}

#[inline]
pub fn torus(major_radius: f64, minor_radius: f64) -> Torus {
    Torus::new(major_radius, minor_radius)
}

/// Plane SDF: dot(point, normal) - offset.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    normal: Point3,
    offset: f64,
}

impl Plane {
    pub fn new(normal: Point3, offset: f64) -> Self {
        Self { normal, offset }
    }
}

impl Sdf3 for Plane {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        dot(point, self.normal) - self.offset
    }
}

#[inline]
pub fn plane(normal: Point3, offset: f64) -> Plane {
    Plane::new(normal, offset)
}

/// Capsule defined by segment [a, b] and radius.
#[derive(Debug, Clone, Copy)]
pub struct Capsule {
    a: Point3,
    b: Point3,
    radius: f64,
}

impl Capsule {
    pub fn new(a: Point3, b: Point3, radius: f64) -> Self {
        assert!(radius >= 0.0, "capsule radius must be non-negative");
        Self { a, b, radius }
    }
}

impl Sdf3 for Capsule {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        let pa = sub(point, self.a);
        let ba = sub(self.b, self.a);
        let ba_dot = dot(ba, ba);
        if ba_dot <= f64::EPSILON {
            return length(pa) - self.radius;
        }
        let h = clamp(dot(pa, ba) / ba_dot, 0.0, 1.0);
        length(sub(pa, mul_scalar(ba, h))) - self.radius
    }
}

#[inline]
pub fn capsule(a: Point3, b: Point3, radius: f64) -> Capsule {
    Capsule::new(a, b, radius)
}

/// Capped cone centered at origin, aligned with Z axis.
/// `radius1` applies at z = -height/2 and `radius2` at z = +height/2.
#[derive(Debug, Clone, Copy)]
pub struct CappedCone {
    radius1: f64,
    radius2: f64,
    height: f64,
}

impl CappedCone {
    pub fn new(radius1: f64, radius2: f64, height: f64) -> Self {
        assert!(radius1 >= 0.0, "capped cone radius1 must be non-negative");
        assert!(radius2 >= 0.0, "capped cone radius2 must be non-negative");
        assert!(height > 0.0, "capped cone height must be positive");
        Self {
            radius1,
            radius2,
            height,
        }
    }
}

impl Sdf3 for CappedCone {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        let h = self.height * 0.5;
        let q = [(point[0] * point[0] + point[1] * point[1]).sqrt(), point[2]];
        let k1 = [self.radius2, h];
        let k2 = [self.radius2 - self.radius1, 2.0 * h];

        let ca = [
            q[0] - q[0].min(if q[1] < 0.0 { self.radius1 } else { self.radius2 }),
            q[1].abs() - h,
        ];
        let k2_dot = k2[0] * k2[0] + k2[1] * k2[1];
        let h_proj = if k2_dot <= f64::EPSILON {
            0.0
        } else {
            clamp(((k1[0] - q[0]) * k2[0] + (k1[1] - q[1]) * k2[1]) / k2_dot, 0.0, 1.0)
        };
        let cb = [
            q[0] - k1[0] + k2[0] * h_proj,
            q[1] - k1[1] + k2[1] * h_proj,
        ];
        let s = if cb[0] < 0.0 && ca[1] < 0.0 {
            -1.0
        } else {
            1.0
        };

        let ca_dot = ca[0] * ca[0] + ca[1] * ca[1];
        let cb_dot = cb[0] * cb[0] + cb[1] * cb[1];
        s * ca_dot.min(cb_dot).sqrt()
    }
}

#[inline]
pub fn capped_cone(radius1: f64, radius2: f64, height: f64) -> CappedCone {
    CappedCone::new(radius1, radius2, height)
}

/// Cylinder with rounded edges.
#[derive(Debug, Clone, Copy)]
pub struct RoundedCylinder {
    radius: f64,
    height: f64,
    edge_radius: f64,
}

impl RoundedCylinder {
    pub fn new(radius: f64, height: f64, edge_radius: f64) -> Self {
        assert!(radius >= 0.0, "rounded cylinder radius must be non-negative");
        assert!(height >= 0.0, "rounded cylinder height must be non-negative");
        assert!(
            edge_radius >= 0.0,
            "rounded cylinder edge radius must be non-negative"
        );
        Self {
            radius,
            height,
            edge_radius,
        }
    }
}

impl Sdf3 for RoundedCylinder {
    #[inline]
    fn evaluate(&self, point: Point3) -> f64 {
        let core_radius = (self.radius - self.edge_radius).max(0.0);
        let core_half_height = (self.height * 0.5 - self.edge_radius).max(0.0);
        capped_cylinder(core_radius, core_half_height).evaluate(point) - self.edge_radius
    }
}

#[inline]
pub fn rounded_cylinder(radius: f64, height: f64, edge_radius: f64) -> RoundedCylinder {
    RoundedCylinder::new(radius, height, edge_radius)
}

#[cfg(test)]
mod tests {
    use super::{
        Sdf3, box3, capped_cone, capped_cylinder, capsule, cylinder, plane, rounded_box,
        rounded_cylinder, sphere, torus,
    };

    #[test]
    fn sphere_matches_analytical_points() {
        let sdf = sphere(1.0);

        assert!((sdf.evaluate([0.0, 0.0, 0.0]) + 1.0).abs() < 1e-12);
        assert!(sdf.evaluate([1.0, 0.0, 0.0]).abs() < 1e-12);
        assert!((sdf.evaluate([2.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn box_matches_axis_aligned_expectations() {
        let sdf = box3([1.0, 2.0, 3.0]);
        assert!((sdf.evaluate([0.0, 0.0, 0.0]) + 1.0).abs() < 1e-12);
        assert!(sdf.evaluate([1.0, 0.0, 0.0]).abs() < 1e-12);
        assert!((sdf.evaluate([3.0, 0.0, 0.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rounded_box_offsets_distance_by_radius() {
        let base = box3([1.0, 1.0, 1.0]);
        let rounded = rounded_box([1.0, 1.0, 1.0], 0.25);
        let p = [1.4, 0.0, 0.0];
        assert!((rounded.evaluate(p) - (base.evaluate(p) - 0.25)).abs() < 1e-12);
    }

    #[test]
    fn cylinder_and_capped_cylinder_match_for_equivalent_height() {
        let finite = cylinder(1.0, 2.0);
        let capped = capped_cylinder(1.0, 1.0);
        let p = [0.25, 0.25, 0.75];
        assert!((finite.evaluate(p) - capped.evaluate(p)).abs() < 1e-12);
        assert!(finite.evaluate([0.0, 0.0, 0.0]) < 0.0);
        assert!(finite.evaluate([1.0, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn torus_matches_ring_geometry() {
        let sdf = torus(2.0, 0.5);
        assert!((sdf.evaluate([2.0, 0.0, 0.0]) + 0.5).abs() < 1e-12);
        assert!(sdf.evaluate([2.5, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn plane_matches_dot_definition() {
        let sdf = plane([0.0, 0.0, 1.0], 2.5);
        assert!((sdf.evaluate([0.0, 0.0, 2.5])).abs() < 1e-12);
        assert!((sdf.evaluate([0.0, 0.0, 4.0]) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn capsule_uses_segment_distance_minus_radius() {
        let sdf = capsule([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 0.5);
        assert!((sdf.evaluate([0.0, 0.0, 0.0]) + 0.5).abs() < 1e-12);
        assert!(sdf.evaluate([0.5, 0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn capped_cone_reduces_to_cylinder_when_radii_match() {
        let sdf = capped_cone(1.0, 1.0, 2.0);
        assert!((sdf.evaluate([1.0, 0.0, 0.0])).abs() < 1e-12);
        assert!((sdf.evaluate([0.0, 0.0, 1.0])).abs() < 1e-12);
        assert!(sdf.evaluate([0.0, 0.0, 0.0]) < 0.0);
    }

    #[test]
    fn rounded_cylinder_with_zero_edge_matches_capped_cylinder() {
        let rounded = rounded_cylinder(1.0, 2.0, 0.0);
        let capped = capped_cylinder(1.0, 1.0);
        let p = [0.4, 0.3, 0.2];
        assert!((rounded.evaluate(p) - capped.evaluate(p)).abs() < 1e-12);
    }
}
