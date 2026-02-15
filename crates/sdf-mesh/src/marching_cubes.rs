use std::collections::HashMap;

use sdf_core::{Point3, Sdf3};

use crate::Mesh;
use crate::tables::{EDGE_TABLE, TRI_TABLE};

const CORNER_OFFSETS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

const EDGE_ENDPOINTS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

#[derive(Debug, Clone, Copy)]
pub struct MarchingCubesConfig {
    pub min: Point3,
    pub max: Point3,
    pub resolution: [usize; 3],
    pub iso_level: f64,
}

impl MarchingCubesConfig {
    pub fn new(min: Point3, max: Point3, resolution: [usize; 3], iso_level: f64) -> Self {
        Self {
            min,
            max,
            resolution,
            iso_level,
        }
    }

    fn spacing(&self) -> Point3 {
        let [nx, ny, nz] = self.resolution;
        [
            (self.max[0] - self.min[0]) / ((nx - 1) as f64),
            (self.max[1] - self.min[1]) / ((ny - 1) as f64),
            (self.max[2] - self.min[2]) / ((nz - 1) as f64),
        ]
    }
}

pub fn extract_mesh_from_sdf<S>(config: &MarchingCubesConfig, sdf: &S) -> Mesh
where
    S: Sdf3,
{
    extract_mesh_with(config, |point| sdf.evaluate(point))
}

pub fn extract_mesh_with<F>(config: &MarchingCubesConfig, mut sample: F) -> Mesh
where
    F: FnMut(Point3) -> f64,
{
    let [nx, ny, nz] = config.resolution;
    if nx < 2 || ny < 2 || nz < 2 {
        return Mesh::empty();
    }

    let spacing = config.spacing();
    let field = sample_grid(config, spacing, &mut sample);

    let mut mesh = Mesh::empty();
    let mut vertex_cache = HashMap::<(u64, u64, u64), u32>::new();
    let mut corner_values = [0.0_f64; 8];
    let mut corner_points = [[0.0_f64; 3]; 8];
    let mut edge_points = [[0.0_f64; 3]; 12];

    for z in 0..(nz - 1) {
        for y in 0..(ny - 1) {
            for x in 0..(nx - 1) {
                let mut case_index = 0usize;
                for corner_id in 0..8 {
                    let offset = CORNER_OFFSETS[corner_id];
                    let gx = x + offset[0];
                    let gy = y + offset[1];
                    let gz = z + offset[2];
                    let idx = grid_index(gx, gy, gz, nx, ny);
                    let value = field[idx];
                    corner_values[corner_id] = value;
                    corner_points[corner_id] = [
                        config.min[0] + (gx as f64) * spacing[0],
                        config.min[1] + (gy as f64) * spacing[1],
                        config.min[2] + (gz as f64) * spacing[2],
                    ];
                    if value < config.iso_level {
                        case_index |= 1 << corner_id;
                    }
                }

                if case_index == 0 || case_index == 255 {
                    continue;
                }

                let edge_mask = EDGE_TABLE[case_index];
                if edge_mask == 0 {
                    continue;
                }

                for edge_id in 0..12 {
                    if edge_mask & (1u16 << edge_id) == 0 {
                        continue;
                    }
                    let [a, b] = EDGE_ENDPOINTS[edge_id];
                    edge_points[edge_id] = interpolate_edge(
                        corner_points[a],
                        corner_points[b],
                        corner_values[a],
                        corner_values[b],
                        config.iso_level,
                    );
                }

                let row = TRI_TABLE[case_index];
                let mut tri_idx = 0usize;
                while tri_idx + 2 < 16 && row[tri_idx] != -1 {
                    let e0 = row[tri_idx] as usize;
                    let e1 = row[tri_idx + 1] as usize;
                    let e2 = row[tri_idx + 2] as usize;

                    let i0 = insert_vertex(&mut mesh, &mut vertex_cache, edge_points[e0]);
                    let i1 = insert_vertex(&mut mesh, &mut vertex_cache, edge_points[e1]);
                    let i2 = insert_vertex(&mut mesh, &mut vertex_cache, edge_points[e2]);
                    mesh.triangles.push([i0, i1, i2]);

                    tri_idx += 3;
                }
            }
        }
    }

    mesh
}

fn sample_grid<F>(config: &MarchingCubesConfig, spacing: Point3, sample: &mut F) -> Vec<f64>
where
    F: FnMut(Point3) -> f64,
{
    let [nx, ny, nz] = config.resolution;
    let mut field = vec![0.0_f64; nx * ny * nz];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let point = [
                    config.min[0] + (x as f64) * spacing[0],
                    config.min[1] + (y as f64) * spacing[1],
                    config.min[2] + (z as f64) * spacing[2],
                ];
                let idx = grid_index(x, y, z, nx, ny);
                field[idx] = sample(point);
            }
        }
    }

    field
}

#[inline]
fn grid_index(x: usize, y: usize, z: usize, nx: usize, ny: usize) -> usize {
    x + y * nx + z * nx * ny
}

#[inline]
fn interpolate_edge(p1: Point3, p2: Point3, v1: f64, v2: f64, iso: f64) -> Point3 {
    let dv = v2 - v1;
    let t = if dv.abs() <= f64::EPSILON {
        0.5
    } else {
        (iso - v1) / dv
    };
    [
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
        p1[2] + t * (p2[2] - p1[2]),
    ]
}

#[inline]
fn insert_vertex(
    mesh: &mut Mesh,
    cache: &mut HashMap<(u64, u64, u64), u32>,
    point: Point3,
) -> u32 {
    let key = (point[0].to_bits(), point[1].to_bits(), point[2].to_bits());
    if let Some(index) = cache.get(&key).copied() {
        return index;
    }
    let index = mesh.vertices.len() as u32;
    mesh.vertices.push(point);
    cache.insert(key, index);
    index
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::f64::consts::PI;

    use sdf_core::sphere;

    use super::{MarchingCubesConfig, extract_mesh_from_sdf, extract_mesh_with};

    #[test]
    fn empty_surface_if_field_is_positive_everywhere() {
        let config = MarchingCubesConfig::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [10, 10, 10], 0.0);
        let mesh = extract_mesh_with(&config, |_| 1.0);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.triangles.is_empty());
    }

    #[test]
    fn empty_surface_if_field_is_negative_everywhere() {
        let config = MarchingCubesConfig::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [10, 10, 10], 0.0);
        let mesh = extract_mesh_with(&config, |_| -1.0);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.triangles.is_empty());
    }

    #[test]
    fn sphere_mesh_matches_analytical_volume_and_area_approximately() {
        let config = MarchingCubesConfig::new([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [32, 32, 32], 0.0);
        let mesh = extract_mesh_from_sdf(&config, &sphere(1.0));
        assert!(!mesh.triangles.is_empty());

        let volume = mesh_volume(&mesh).abs();
        let area = mesh_area(&mesh);
        let exact_volume = 4.0 * PI / 3.0;
        let exact_area = 4.0 * PI;

        let vol_rel = (volume - exact_volume).abs() / exact_volume;
        let area_rel = (area - exact_area).abs() / exact_area;

        assert!(vol_rel < 0.1, "volume relative error too high: {vol_rel:.4}");
        assert!(area_rel < 0.12, "area relative error too high: {area_rel:.4}");
    }

    #[test]
    fn sphere_volume_error_converges_with_resolution() {
        let exact_volume = 4.0 * PI / 3.0;
        let config_low = MarchingCubesConfig::new([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [16, 16, 16], 0.0);
        let config_mid = MarchingCubesConfig::new([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [24, 24, 24], 0.0);
        let config_high = MarchingCubesConfig::new([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [32, 32, 32], 0.0);

        let e_low = (mesh_volume(&extract_mesh_from_sdf(&config_low, &sphere(1.0))).abs() - exact_volume).abs();
        let e_mid = (mesh_volume(&extract_mesh_from_sdf(&config_mid, &sphere(1.0))).abs() - exact_volume).abs();
        let e_high = (mesh_volume(&extract_mesh_from_sdf(&config_high, &sphere(1.0))).abs() - exact_volume).abs();

        assert!(e_high < e_mid, "expected high resolution error < mid ({e_high} !< {e_mid})");
        assert!(e_mid < e_low, "expected mid resolution error < low ({e_mid} !< {e_low})");
    }

    #[test]
    fn sphere_mesh_has_no_degenerate_triangles_and_is_watertight() {
        let config = MarchingCubesConfig::new([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [28, 28, 28], 0.0);
        let mesh = extract_mesh_from_sdf(&config, &sphere(1.0));
        assert!(!mesh.triangles.is_empty());

        for triangle in &mesh.triangles {
            let area = triangle_area(
                mesh.vertices[triangle[0] as usize],
                mesh.vertices[triangle[1] as usize],
                mesh.vertices[triangle[2] as usize],
            );
            assert!(area > 1e-10, "degenerate triangle area={area}");
        }

        let mut edge_counts = HashMap::<(u32, u32), usize>::new();
        for triangle in &mesh.triangles {
            let edges = [
                ordered_edge(triangle[0], triangle[1]),
                ordered_edge(triangle[1], triangle[2]),
                ordered_edge(triangle[2], triangle[0]),
            ];
            for edge in edges {
                *edge_counts.entry(edge).or_insert(0) += 1;
            }
        }

        for (edge, count) in edge_counts {
            assert_eq!(count, 2, "non-manifold edge {:?} has count {}", edge, count);
        }
    }

    fn ordered_edge(a: u32, b: u32) -> (u32, u32) {
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }

    fn mesh_area(mesh: &crate::Mesh) -> f64 {
        mesh.triangles
            .iter()
            .map(|tri| {
                triangle_area(
                    mesh.vertices[tri[0] as usize],
                    mesh.vertices[tri[1] as usize],
                    mesh.vertices[tri[2] as usize],
                )
            })
            .sum()
    }

    fn mesh_volume(mesh: &crate::Mesh) -> f64 {
        mesh.triangles
            .iter()
            .map(|tri| {
                let a = mesh.vertices[tri[0] as usize];
                let b = mesh.vertices[tri[1] as usize];
                let c = mesh.vertices[tri[2] as usize];
                dot(a, cross(b, c)) / 6.0
            })
            .sum()
    }

    fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let normal = cross(ab, ac);
        (dot(normal, normal)).sqrt() * 0.5
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}
