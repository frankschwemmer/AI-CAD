use crate::primitives::{Point3, Sdf3};

/// Evaluates an SDF on a batch of points.
pub fn evaluate_points<S>(sdf: &S, points: &[Point3]) -> Vec<f64>
where
    S: Sdf3,
{
    points.iter().map(|point| sdf.evaluate(*point)).collect()
}

#[cfg(test)]
mod tests {
    use crate::primitives::sphere;

    use super::evaluate_points;

    #[test]
    fn batch_evaluation_returns_expected_values() {
        let sdf = sphere(2.0);
        let points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let values = evaluate_points(&sdf, &points);

        assert_eq!(values.len(), 3);
        assert!((values[0] + 2.0).abs() < 1e-12);
        assert!(values[1].abs() < 1e-12);
        assert!((values[2] - 1.0).abs() < 1e-12);
    }
}
