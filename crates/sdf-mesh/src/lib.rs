pub mod export;
pub mod marching_cubes;
mod tables;

/// Triangle mesh container for marching-cubes output.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }
}

pub use marching_cubes::{MarchingCubesConfig, extract_mesh_from_sdf, extract_mesh_with};
pub use export::{to_ascii_stl, to_binary_stl, to_obj};

#[cfg(test)]
mod tests {
    use super::Mesh;

    #[test]
    fn empty_mesh_has_no_geometry() {
        let mesh = Mesh::empty();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.triangles.is_empty());
    }
}
