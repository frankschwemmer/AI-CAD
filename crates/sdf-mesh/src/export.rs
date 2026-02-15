use crate::Mesh;

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= f64::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[inline]
fn triangle_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalize(cross(ab, ac))
}

pub fn to_binary_stl(mesh: &Mesh, name: &str) -> Vec<u8> {
    let mut bytes = Vec::<u8>::with_capacity(84 + mesh.triangles.len() * 50);

    let mut header = [0u8; 80];
    let name_bytes = name.as_bytes();
    let header_len = name_bytes.len().min(80);
    header[..header_len].copy_from_slice(&name_bytes[..header_len]);
    bytes.extend_from_slice(&header);

    let tri_count = mesh.triangles.len() as u32;
    bytes.extend_from_slice(&tri_count.to_le_bytes());

    for tri in &mesh.triangles {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        let n = triangle_normal(a, b, c);

        push_f32_triplet(&mut bytes, n);
        push_f32_triplet(&mut bytes, a);
        push_f32_triplet(&mut bytes, b);
        push_f32_triplet(&mut bytes, c);
        bytes.extend_from_slice(&0u16.to_le_bytes());
    }

    bytes
}

pub fn to_ascii_stl(mesh: &Mesh, name: &str) -> String {
    let mut out = String::new();
    out.push_str("solid ");
    out.push_str(name);
    out.push('\n');

    for tri in &mesh.triangles {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        let n = triangle_normal(a, b, c);

        out.push_str(&format!("  facet normal {} {} {}\n", n[0], n[1], n[2]));
        out.push_str("    outer loop\n");
        out.push_str(&format!("      vertex {} {} {}\n", a[0], a[1], a[2]));
        out.push_str(&format!("      vertex {} {} {}\n", b[0], b[1], b[2]));
        out.push_str(&format!("      vertex {} {} {}\n", c[0], c[1], c[2]));
        out.push_str("    endloop\n");
        out.push_str("  endfacet\n");
    }

    out.push_str("endsolid ");
    out.push_str(name);
    out.push('\n');
    out
}

pub fn to_obj(mesh: &Mesh) -> String {
    let mut out = String::new();
    for vertex in &mesh.vertices {
        out.push_str(&format!("v {} {} {}\n", vertex[0], vertex[1], vertex[2]));
    }
    for triangle in &mesh.triangles {
        out.push_str(&format!(
            "f {} {} {}\n",
            triangle[0] + 1,
            triangle[1] + 1,
            triangle[2] + 1
        ));
    }
    out
}

#[inline]
fn push_f32_triplet(bytes: &mut Vec<u8>, value: [f64; 3]) {
    bytes.extend_from_slice(&(value[0] as f32).to_le_bytes());
    bytes.extend_from_slice(&(value[1] as f32).to_le_bytes());
    bytes.extend_from_slice(&(value[2] as f32).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use sdf_core::sphere;

    use crate::{MarchingCubesConfig, extract_mesh_from_sdf};

    use super::{to_ascii_stl, to_binary_stl, to_obj};

    fn simple_mesh() -> crate::Mesh {
        crate::Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        }
    }

    #[test]
    fn binary_stl_has_valid_size_and_triangle_count() {
        let mesh = simple_mesh();
        let bytes = to_binary_stl(&mesh, "test");
        assert_eq!(bytes.len(), 84 + 50);
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]);
        assert_eq!(count, 1);
    }

    #[test]
    fn binary_stl_export_is_deterministic() {
        let mesh = simple_mesh();
        let a = to_binary_stl(&mesh, "deterministic");
        let b = to_binary_stl(&mesh, "deterministic");
        assert_eq!(a, b);
    }

    #[test]
    fn ascii_stl_contains_required_tokens() {
        let mesh = simple_mesh();
        let stl = to_ascii_stl(&mesh, "tri");
        assert!(stl.starts_with("solid tri"));
        assert!(stl.contains("facet normal"));
        assert!(stl.contains("outer loop"));
        assert!(stl.contains("vertex 0 0 0"));
        assert!(stl.ends_with("endsolid tri\n"));
    }

    #[test]
    fn obj_contains_vertices_and_faces() {
        let mesh = simple_mesh();
        let obj = to_obj(&mesh);
        assert!(obj.contains("v 0 0 0"));
        assert!(obj.contains("v 1 0 0"));
        assert!(obj.contains("f 1 2 3"));
    }

    #[test]
    fn full_pipeline_marching_cubes_to_exports() {
        let config =
            MarchingCubesConfig::new([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [24, 24, 24], 0.0);
        let mesh = extract_mesh_from_sdf(&config, &sphere(1.0));
        assert!(!mesh.triangles.is_empty());

        let bin = to_binary_stl(&mesh, "sphere");
        let ascii = to_ascii_stl(&mesh, "sphere");
        let obj = to_obj(&mesh);

        let tri_count = u32::from_le_bytes([bin[80], bin[81], bin[82], bin[83]]) as usize;
        assert_eq!(tri_count, mesh.triangles.len());
        assert!(ascii.contains("facet normal"));
        assert!(obj.contains("\nf "));
    }
}
