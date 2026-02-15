use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use sdf_core::{
    Sdf3, box3, capped_cone, capped_cylinder, capsule, cylinder, difference, elongate,
    intersection, inverse_bend_x, inverse_orient, inverse_rotate_z, inverse_translate,
    inverse_twist_z, mirror_point, negate, plane, repeat_point, rounded_box, rounded_cylinder,
    scale, shell, smooth_difference, smooth_intersection, smooth_union, sphere, torus, union,
};
use sdf_mesh::{MarchingCubesConfig, Mesh, extract_mesh_with, to_ascii_stl, to_binary_stl, to_obj};

type DynError = Box<dyn Error>;
type DynSdf = Box<dyn Sdf3>;
type Flags = HashMap<String, String>;

fn main() -> Result<(), DynError> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        print_usage();
        return Ok(());
    }

    match args[0].as_str() {
        "evaluate-sphere" => run_eval(&args[1..], |flags| {
            Ok(Box::new(sphere(required_f64(flags, "--radius")?)))
        }),
        "evaluate-box" => run_eval(&args[1..], |flags| {
            let half_extents = [
                required_f64(flags, "--hx")?,
                required_f64(flags, "--hy")?,
                required_f64(flags, "--hz")?,
            ];
            Ok(Box::new(box3(half_extents)))
        }),
        "evaluate-rounded-box" => run_eval(&args[1..], |flags| {
            let half_extents = [
                required_f64(flags, "--hx")?,
                required_f64(flags, "--hy")?,
                required_f64(flags, "--hz")?,
            ];
            Ok(Box::new(rounded_box(
                half_extents,
                required_f64(flags, "--radius")?,
            )))
        }),
        "evaluate-cylinder" => run_eval(&args[1..], |flags| {
            Ok(Box::new(cylinder(
                required_f64(flags, "--radius")?,
                required_f64(flags, "--height")?,
            )))
        }),
        "evaluate-capped-cylinder" => run_eval(&args[1..], |flags| {
            Ok(Box::new(capped_cylinder(
                required_f64(flags, "--radius")?,
                required_f64(flags, "--half-height")?,
            )))
        }),
        "evaluate-torus" => run_eval(&args[1..], |flags| {
            Ok(Box::new(torus(
                required_f64(flags, "--major-radius")?,
                required_f64(flags, "--minor-radius")?,
            )))
        }),
        "evaluate-plane" => run_eval(&args[1..], |flags| {
            Ok(Box::new(plane(
                [
                    required_f64(flags, "--nx")?,
                    required_f64(flags, "--ny")?,
                    required_f64(flags, "--nz")?,
                ],
                required_f64(flags, "--offset")?,
            )))
        }),
        "evaluate-capsule" => run_eval(&args[1..], |flags| {
            let a = [
                required_f64(flags, "--ax")?,
                required_f64(flags, "--ay")?,
                required_f64(flags, "--az")?,
            ];
            let b = [
                required_f64(flags, "--bx")?,
                required_f64(flags, "--by")?,
                required_f64(flags, "--bz")?,
            ];
            Ok(Box::new(capsule(a, b, required_f64(flags, "--radius")?)))
        }),
        "evaluate-capped-cone" => run_eval(&args[1..], |flags| {
            Ok(Box::new(capped_cone(
                required_f64(flags, "--radius1")?,
                required_f64(flags, "--radius2")?,
                required_f64(flags, "--height")?,
            )))
        }),
        "evaluate-rounded-cylinder" => run_eval(&args[1..], |flags| {
            Ok(Box::new(rounded_cylinder(
                required_f64(flags, "--radius")?,
                required_f64(flags, "--height")?,
                required_f64(flags, "--edge-radius")?,
            )))
        }),
        "evaluate-operation-scene" => run_evaluate_operation_scene(&args[1..]),
        "evaluate-transform-scene" => run_evaluate_transform_scene(&args[1..]),
        "mesh-metrics" => run_mesh_metrics(&args[1..]),
        "export-mesh" => run_export_mesh(&args[1..]),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn run_eval<F>(args: &[String], build_sdf: F) -> Result<(), DynError>
where
    F: Fn(&Flags) -> Result<DynSdf, DynError>,
{
    let flags = parse_flags(args)?;
    let points_file = required_str(&flags, "--points-file")?;
    let points = read_points(points_file)?;
    let sdf = build_sdf(&flags)?;

    for point in points {
        println!("{:.17}", sdf.evaluate(point));
    }
    Ok(())
}

fn run_evaluate_operation_scene(args: &[String]) -> Result<(), DynError> {
    let flags = parse_flags(args)?;
    let operation = required_str(&flags, "--operation")?;
    let points_file = required_str(&flags, "--points-file")?;
    let points = read_points(points_file)?;

    for point in points {
        println!("{:.17}", evaluate_operation_scene_point(operation, point, &flags)?);
    }

    Ok(())
}

fn evaluate_operation_scene_point(operation: &str, point: [f64; 3], flags: &Flags) -> Result<f64, DynError> {
    let a = sphere(1.0).evaluate(point);
    let shifted = [point[0] - 0.35, point[1] + 0.15, point[2]];
    let b = box3([0.8, 0.6, 0.9]).evaluate(shifted);

    let value = match operation {
        "union" => union(a, b),
        "intersection" => intersection(a, b),
        "difference" => difference(a, b),
        "smooth_union" => smooth_union(a, b, optional_f64(flags, "--k", 0.2)?),
        "smooth_intersection" => {
            smooth_intersection(a, b, optional_f64(flags, "--k", 0.2)?)
        }
        "smooth_difference" => smooth_difference(a, b, optional_f64(flags, "--k", 0.2)?),
        "negate" => negate(a),
        "shell" => shell(a, optional_f64(flags, "--thickness", 0.1)?),
        "elongate" => {
            let half_size = [
                optional_f64(flags, "--hx", 0.6)?,
                optional_f64(flags, "--hy", 0.2)?,
                optional_f64(flags, "--hz", 0.0)?,
            ];
            elongate(&sphere(0.6), point, half_size)
        }
        "repeat" => {
            let period = [
                optional_f64(flags, "--px", 1.5)?,
                optional_f64(flags, "--py", 1.0)?,
                optional_f64(flags, "--pz", 0.0)?,
            ];
            let repeated = repeat_point(point, period);
            sphere(0.55).evaluate(repeated)
        }
        _ => return Err(format!("unknown operation: {operation}").into()),
    };

    Ok(value)
}

fn run_evaluate_transform_scene(args: &[String]) -> Result<(), DynError> {
    let flags = parse_flags(args)?;
    let transform = required_str(&flags, "--transform")?;
    let points_file = required_str(&flags, "--points-file")?;
    let points = read_points(points_file)?;

    for point in points {
        println!("{:.17}", evaluate_transform_scene_point(transform, point, &flags)?);
    }

    Ok(())
}

fn evaluate_transform_scene_point(transform: &str, point: [f64; 3], flags: &Flags) -> Result<f64, DynError> {
    let base = sphere(1.0);

    let value = match transform {
        "translate" => {
            let offset = [
                optional_f64(flags, "--tx", 0.7)?,
                optional_f64(flags, "--ty", -0.4)?,
                optional_f64(flags, "--tz", 0.5)?,
            ];
            base.evaluate(inverse_translate(point, offset))
        }
        "rotate" => {
            let angle = optional_f64(flags, "--angle", 0.6)?;
            base.evaluate(inverse_rotate_z(point, angle))
        }
        "scale" => scale(&base, point, optional_f64(flags, "--factor", 1.5)?),
        "orient" => {
            let from = [0.0, 1.0, 0.0];
            let to = [
                optional_f64(flags, "--tx", 1.0)?,
                optional_f64(flags, "--ty", 0.0)?,
                optional_f64(flags, "--tz", 0.0)?,
            ];
            base.evaluate(inverse_orient(point, from, to))
        }
        "mirror" => {
            let normal = [
                optional_f64(flags, "--nx", 0.0)?,
                optional_f64(flags, "--ny", 1.0)?,
                optional_f64(flags, "--nz", 0.0)?,
            ];
            let offset = optional_f64(flags, "--offset", 0.0)?;
            base.evaluate(mirror_point(point, normal, offset))
        }
        "twist" => {
            let rate = optional_f64(flags, "--rate", 0.2)?;
            base.evaluate(inverse_twist_z(point, rate))
        }
        "bend" => {
            let rate = optional_f64(flags, "--rate", 0.15)?;
            base.evaluate(inverse_bend_x(point, rate))
        }
        _ => return Err(format!("unknown transform: {transform}").into()),
    };

    Ok(value)
}

fn run_mesh_metrics(args: &[String]) -> Result<(), DynError> {
    let flags = parse_flags(args)?;
    let mesh = build_scene_mesh(&flags)?;
    let volume = mesh_volume(&mesh).abs();
    let area = mesh_area(&mesh);

    println!("vertices {}", mesh.vertices.len());
    println!("triangles {}", mesh.triangles.len());
    println!("volume {:.17}", volume);
    println!("area {:.17}", area);
    Ok(())
}

fn run_export_mesh(args: &[String]) -> Result<(), DynError> {
    let flags = parse_flags(args)?;
    let mesh = build_scene_mesh(&flags)?;
    let output = required_str(&flags, "--output")?;
    let format = optional_str(&flags, "--format", "binary-stl");
    let name = optional_str(&flags, "--name", "mesh");

    match format {
        "binary-stl" => {
            let bytes = to_binary_stl(&mesh, name);
            fs::write(output, bytes)?;
        }
        "ascii-stl" => {
            let text = to_ascii_stl(&mesh, name);
            fs::write(output, text)?;
        }
        "obj" => {
            let text = to_obj(&mesh);
            fs::write(output, text)?;
        }
        _ => return Err(format!("unknown format: {format}").into()),
    }

    Ok(())
}

fn build_scene_mesh(flags: &Flags) -> Result<Mesh, DynError> {
    let scene = optional_str(flags, "--scene", "sphere");
    let resolution = optional_usize(flags, "--resolution", 64)?;
    let bounds = optional_f64(flags, "--bounds", 1.5)?;
    let config = MarchingCubesConfig::new(
        [-bounds, -bounds, -bounds],
        [bounds, bounds, bounds],
        [resolution, resolution, resolution],
        0.0,
    );

    let mesh = match scene {
        "sphere" => {
            let radius = optional_f64(flags, "--radius", 1.0)?;
            extract_mesh_with(&config, |point| sphere(radius).evaluate(point))
        }
        "union_sphere_box" => extract_mesh_with(&config, |point| {
            let a = sphere(1.0).evaluate(point);
            let shifted = [point[0] - 0.35, point[1] + 0.15, point[2]];
            let b = box3([0.8, 0.6, 0.9]).evaluate(shifted);
            union(a, b)
        }),
        _ => return Err(format!("unknown scene: {scene}").into()),
    };

    Ok(mesh)
}

fn mesh_area(mesh: &Mesh) -> f64 {
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

fn mesh_volume(mesh: &Mesh) -> f64 {
    mesh.triangles
        .iter()
        .map(|tri| {
            let a = mesh.vertices[tri[0] as usize];
            let b = mesh.vertices[tri[1] as usize];
            let c = mesh.vertices[tri[2] as usize];
            dot3(a, cross3(b, c)) / 6.0
        })
        .sum()
}

fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = cross3(ab, ac);
    (dot3(n, n)).sqrt() * 0.5
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn parse_flags(args: &[String]) -> Result<Flags, DynError> {
    if !args.len().is_multiple_of(2) {
        return Err("expected flag-value pairs".into());
    }

    let mut flags = HashMap::new();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if !flag.starts_with("--") {
            return Err(format!("expected flag at position {}", index + 1).into());
        }
        let value = args[index + 1].clone();
        if flags.insert(flag.to_string(), value).is_some() {
            return Err(format!("duplicate flag: {flag}").into());
        }
        index += 2;
    }
    Ok(flags)
}

fn required_str<'a>(flags: &'a Flags, key: &str) -> Result<&'a str, DynError> {
    flags
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing required {key}").into())
}

fn required_f64(flags: &Flags, key: &str) -> Result<f64, DynError> {
    required_str(flags, key)?
        .parse::<f64>()
        .map_err(|err| format!("invalid float for {key}: {err}").into())
}

fn optional_usize(flags: &Flags, key: &str, default: usize) -> Result<usize, DynError> {
    match flags.get(key) {
        Some(value) => value
            .parse::<usize>()
            .map_err(|err| format!("invalid usize for {key}: {err}").into()),
        None => Ok(default),
    }
}

fn optional_f64(flags: &Flags, key: &str, default: f64) -> Result<f64, DynError> {
    match flags.get(key) {
        Some(value) => value
            .parse::<f64>()
            .map_err(|err| format!("invalid float for {key}: {err}").into()),
        None => Ok(default),
    }
}

fn optional_str<'a>(flags: &'a Flags, key: &str, default: &'a str) -> &'a str {
    flags.get(key).map(String::as_str).unwrap_or(default)
}

fn read_points(path: impl AsRef<Path>) -> Result<Vec<[f64; 3]>, DynError> {
    let raw = fs::read_to_string(path)?;
    let mut points = Vec::new();

    for (line_no, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 3 {
            return Err(format!("line {}: expected exactly 3 floats", line_no + 1).into());
        }
        let x = parts[0].parse::<f64>()?;
        let y = parts[1].parse::<f64>()?;
        let z = parts[2].parse::<f64>()?;
        points.push([x, y, z]);
    }

    Ok(points)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  sdf-cli evaluate-sphere --radius <f64> --points-file <path>");
    eprintln!("  sdf-cli evaluate-box --hx <f64> --hy <f64> --hz <f64> --points-file <path>");
    eprintln!(
        "  sdf-cli evaluate-rounded-box --hx <f64> --hy <f64> --hz <f64> --radius <f64> --points-file <path>"
    );
    eprintln!("  sdf-cli evaluate-cylinder --radius <f64> --height <f64> --points-file <path>");
    eprintln!(
        "  sdf-cli evaluate-capped-cylinder --radius <f64> --half-height <f64> --points-file <path>"
    );
    eprintln!(
        "  sdf-cli evaluate-torus --major-radius <f64> --minor-radius <f64> --points-file <path>"
    );
    eprintln!(
        "  sdf-cli evaluate-plane --nx <f64> --ny <f64> --nz <f64> --offset <f64> --points-file <path>"
    );
    eprintln!(
        "  sdf-cli evaluate-capsule --ax <f64> --ay <f64> --az <f64> --bx <f64> --by <f64> --bz <f64> --radius <f64> --points-file <path>"
    );
    eprintln!(
        "  sdf-cli evaluate-capped-cone --radius1 <f64> --radius2 <f64> --height <f64> --points-file <path>"
    );
    eprintln!(
        "  sdf-cli evaluate-rounded-cylinder --radius <f64> --height <f64> --edge-radius <f64> --points-file <path>"
    );
    eprintln!(
        "  sdf-cli evaluate-operation-scene --operation <name> --points-file <path> [operation params]"
    );
    eprintln!(
        "  sdf-cli evaluate-transform-scene --transform <name> --points-file <path> [transform params]"
    );
    eprintln!(
        "  sdf-cli mesh-metrics --scene <sphere|union_sphere_box> [--radius <f64>] [--resolution <usize>] [--bounds <f64>]"
    );
    eprintln!(
        "  sdf-cli export-mesh --scene <sphere|union_sphere_box> --output <path> [--format <binary-stl|ascii-stl|obj>] [--name <str>] [--resolution <usize>] [--bounds <f64>]"
    );
}

#[cfg(test)]
mod tests {
    use super::{
        build_scene_mesh, evaluate_operation_scene_point, evaluate_transform_scene_point,
        parse_flags, read_points, required_f64,
    };

    #[test]
    fn parses_whitespace_points() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("sdf_cli_points_test.txt");
        std::fs::write(&path, "0 0 0\n1 0 0\n").expect("should write test points file");

        let points = read_points(&path).expect("should parse points");
        assert_eq!(points, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parses_flag_pairs() {
        let args = vec![
            "--radius".to_string(),
            "1.5".to_string(),
            "--points-file".to_string(),
            "points.txt".to_string(),
        ];
        let flags = parse_flags(&args).expect("should parse flag pairs");
        assert_eq!(flags.get("--radius").map(String::as_str), Some("1.5"));
        assert_eq!(
            flags.get("--points-file").map(String::as_str),
            Some("points.txt")
        );
    }

    #[test]
    fn parses_required_float() {
        let args = vec!["--radius".to_string(), "2.5".to_string()];
        let flags = parse_flags(&args).expect("flag parsing should succeed");
        let radius = required_f64(&flags, "--radius").expect("required float should parse");
        assert!((radius - 2.5).abs() < 1e-12);
    }

    #[test]
    fn evaluates_operation_scene_union() {
        let args = vec!["--operation".to_string(), "union".to_string()];
        let flags = parse_flags(&args).expect("flag parsing should succeed");
        let value = evaluate_operation_scene_point("union", [0.0, 0.0, 0.0], &flags)
            .expect("operation evaluation should succeed");
        assert!(value <= 0.0);
    }

    #[test]
    fn evaluates_transform_scene_translate() {
        let args = vec!["--transform".to_string(), "translate".to_string()];
        let flags = parse_flags(&args).expect("flag parsing should succeed");
        let value = evaluate_transform_scene_point("translate", [0.7, -0.4, 0.5], &flags)
            .expect("transform evaluation should succeed");
        assert!((value + 1.0).abs() < 1e-12);
    }

    #[test]
    fn builds_sphere_mesh_from_flags() {
        let args = vec![
            "--scene".to_string(),
            "sphere".to_string(),
            "--resolution".to_string(),
            "20".to_string(),
        ];
        let flags = parse_flags(&args).expect("flag parsing should succeed");
        let mesh = build_scene_mesh(&flags).expect("mesh build should succeed");
        assert!(!mesh.triangles.is_empty());
    }
}
