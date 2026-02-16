use std::cell::RefCell;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use sdf_core::{
    Sdf3, capped_cone, capped_cylinder, capsule, cylinder, difference, intersection,
    inverse_bend_x, inverse_rotate_x, inverse_rotate_y, inverse_rotate_z, inverse_translate,
    inverse_twist_z, mirror_point, negate, plane, rounded_cylinder, shell, smooth_difference,
    smooth_intersection, smooth_union, torus, union,
};
use sdf_mesh::{MarchingCubesConfig, Mesh, extract_mesh_with};

#[derive(Debug, Clone, PartialEq)]
pub struct DslError {
    message: String,
    line: Option<usize>,
    column: Option<usize>,
}

impl DslError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            line: None,
            column: None,
        }
    }

    fn at(message: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            message: message.into(),
            line: Some(line),
            column: Some(column),
        }
    }
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.column) {
            (Some(line), Some(column)) => {
                write!(f, "{} at line {}, column {}", self.message, line, column)
            }
            _ => f.write_str(&self.message),
        }
    }
}

impl Error for DslError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    None,
    Mm,
    Deg,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumberLiteral {
    pub value: f64,
    pub unit: Unit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(NumberLiteral),
    Variable(String),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        lhs: Box<Expr>,
        op: BinaryOp,
        rhs: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParamDecl {
    pub name: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub name: String,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Assignment(Assignment),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub params: Vec<ParamDecl>,
    pub items: Vec<Item>,
}

impl Program {
    pub fn to_source(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.params.is_empty() {
            writeln!(f, "params {{")?;
            for decl in &self.params {
                writeln!(f, "  {} = {}", decl.name, decl.value)?;
            }
            writeln!(f, "}}")?;
            if !self.items.is_empty() {
                writeln!(f)?;
            }
        }

        for (index, item) in self.items.iter().enumerate() {
            match item {
                Item::Assignment(assignment) => {
                    write!(f, "{} = {}", assignment.name, assignment.expr)?;
                }
                Item::Expr(expr) => {
                    write!(f, "{expr}")?;
                }
            }
            if index + 1 < self.items.len() {
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(number) => write_number(f, number),
            Expr::Variable(name) => f.write_str(name),
            Expr::Unary { op, expr } => match op {
                UnaryOp::Neg => write!(f, "(-{expr})"),
            },
            Expr::Binary { lhs, op, rhs } => {
                let op_text = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                };
                write!(f, "({lhs} {op_text} {rhs})")
            }
            Expr::Call { name, args } => {
                write!(f, "{name}(")?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                f.write_str(")")
            }
        }
    }
}

fn write_number(f: &mut fmt::Formatter<'_>, number: &NumberLiteral) -> fmt::Result {
    if number.value.fract().abs() <= f64::EPSILON {
        write!(f, "{}", number.value as i64)?;
    } else {
        write!(f, "{}", number.value)?;
    }

    match number.unit {
        Unit::None => Ok(()),
        Unit::Mm => f.write_str("mm"),
        Unit::Deg => f.write_str("deg"),
    }
}

#[derive(Debug)]
pub struct Scene {
    params: BTreeMap<String, f64>,
    root: ShapeNode,
    compiled: RefCell<Option<Result<NumericNode, DslError>>>,
}

#[derive(Debug, Clone)]
pub struct CompiledScene {
    root: NumericNode,
}

impl CompiledScene {
    pub fn evaluate(&self, point: [f64; 3]) -> f64 {
        self.root.evaluate(point)
    }
}

impl Scene {
    pub fn evaluate(&self, point: [f64; 3]) -> Result<f64, DslError> {
        Ok(self.compiled_node()?.evaluate(point))
    }

    pub fn parameters(&self) -> &BTreeMap<String, f64> {
        &self.params
    }

    pub fn parameter(&self, name: &str) -> Option<f64> {
        self.params.get(name).copied()
    }

    pub fn parameter_names(&self) -> Vec<String> {
        self.params.keys().cloned().collect()
    }

    pub fn suggested_bounds(&self) -> ([f64; 3], [f64; 3]) {
        let bounds = self.estimated_bounds();
        (
            [-bounds, -bounds, -bounds],
            [bounds, bounds, bounds],
        )
    }

    pub fn set_param(&mut self, name: &str, value: f64) -> Result<(), DslError> {
        let old_value = self
            .params
            .get(name)
            .copied()
            .ok_or_else(|| DslError::new(format!("unknown parameter '{name}'")))?;
        self.params.insert(name.to_string(), value);
        *self.compiled.borrow_mut() = None;

        let compiled = self.root.materialize(&self.params);
        match compiled {
            Ok(node) => {
                *self.compiled.borrow_mut() = Some(Ok(node));
                Ok(())
            }
            Err(err) => {
                self.params.insert(name.to_string(), old_value);
                *self.compiled.borrow_mut() = None;
                Err(DslError::new(format!(
                    "invalid value for parameter '{name}': {err}"
                )))
            }
        }
    }

    pub fn compile_current(&self) -> Result<CompiledScene, DslError> {
        Ok(CompiledScene {
            root: self.compiled_node()?.clone(),
        })
    }

    pub fn evaluate_mesh(&self, resolution: usize) -> Result<Mesh, DslError> {
        let bounds = self.estimated_bounds();
        self.evaluate_mesh_with_bounds(
            resolution,
            [-bounds, -bounds, -bounds],
            [bounds, bounds, bounds],
        )
    }

    pub fn evaluate_mesh_with_bounds(
        &self,
        resolution: usize,
        min: [f64; 3],
        max: [f64; 3],
    ) -> Result<Mesh, DslError> {
        if resolution < 2 {
            return Err(DslError::new(
                "resolution must be at least 2 for marching cubes",
            ));
        }
        for axis in 0..3 {
            if max[axis] <= min[axis] {
                return Err(DslError::new(format!(
                    "invalid mesh bounds on axis {axis}: min must be less than max"
                )));
            }
        }

        let compiled = self.compile_current()?;
        let config = MarchingCubesConfig::new(min, max, [resolution, resolution, resolution], 0.0);
        Ok(extract_mesh_with(&config, |point| compiled.evaluate(point)))
    }

    fn ensure_compiled(&self) -> Result<(), DslError> {
        if self.compiled.borrow().is_none() {
            let compiled = self.root.materialize(&self.params);
            *self.compiled.borrow_mut() = Some(compiled);
        }

        let compiled = self.compiled.borrow();
        match compiled.as_ref() {
            Some(Ok(_)) => Ok(()),
            Some(Err(err)) => Err(err.clone()),
            None => Err(DslError::new("internal error: missing compiled scene")),
        }
    }

    fn compiled_node(&self) -> Result<std::cell::Ref<'_, NumericNode>, DslError> {
        self.ensure_compiled()?;
        let compiled = self.compiled.borrow();
        if let Some(Err(err)) = compiled.as_ref() {
            return Err(err.clone());
        }
        Ok(std::cell::Ref::map(compiled, |entry| match entry {
            Some(Ok(node)) => node,
            Some(Err(_)) => unreachable!("error handled above"),
            None => unreachable!("compiled cache populated by ensure_compiled"),
        }))
    }

    fn estimated_bounds(&self) -> f64 {
        let mut max_abs = 0.0f64;
        for value in self.params.values() {
            max_abs = max_abs.max(value.abs());
        }
        (max_abs * 1.5 + 5.0).clamp(10.0, 500.0)
    }
}

#[derive(Debug, Clone)]
enum ScalarExpr {
    Constant(f64),
    Parameter(String),
    Neg(Box<ScalarExpr>),
    Add(Box<ScalarExpr>, Box<ScalarExpr>),
    Sub(Box<ScalarExpr>, Box<ScalarExpr>),
    Mul(Box<ScalarExpr>, Box<ScalarExpr>),
    Div(Box<ScalarExpr>, Box<ScalarExpr>),
}

impl ScalarExpr {
    fn evaluate(&self, params: &BTreeMap<String, f64>) -> Result<f64, DslError> {
        match self {
            ScalarExpr::Constant(value) => Ok(*value),
            ScalarExpr::Parameter(name) => params
                .get(name)
                .copied()
                .ok_or_else(|| DslError::new(format!("unknown parameter '{name}'"))),
            ScalarExpr::Neg(expr) => Ok(-expr.evaluate(params)?),
            ScalarExpr::Add(a, b) => Ok(a.evaluate(params)? + b.evaluate(params)?),
            ScalarExpr::Sub(a, b) => Ok(a.evaluate(params)? - b.evaluate(params)?),
            ScalarExpr::Mul(a, b) => Ok(a.evaluate(params)? * b.evaluate(params)?),
            ScalarExpr::Div(a, b) => {
                let numerator = a.evaluate(params)?;
                let denominator = b.evaluate(params)?;
                if denominator.abs() <= f64::EPSILON {
                    return Err(DslError::new("division by zero"));
                }
                Ok(numerator / denominator)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum ShapeNode {
    Sphere {
        radius: ScalarExpr,
    },
    Box {
        width: ScalarExpr,
        depth: ScalarExpr,
        height: ScalarExpr,
    },
    RoundedBox {
        width: ScalarExpr,
        depth: ScalarExpr,
        height: ScalarExpr,
        radius: ScalarExpr,
    },
    Cylinder {
        radius: ScalarExpr,
        height: ScalarExpr,
    },
    CappedCylinder {
        radius: ScalarExpr,
        half_height: ScalarExpr,
    },
    Torus {
        major_radius: ScalarExpr,
        minor_radius: ScalarExpr,
    },
    Plane {
        nx: ScalarExpr,
        ny: ScalarExpr,
        nz: ScalarExpr,
        offset: ScalarExpr,
    },
    Capsule {
        ax: ScalarExpr,
        ay: ScalarExpr,
        az: ScalarExpr,
        bx: ScalarExpr,
        by: ScalarExpr,
        bz: ScalarExpr,
        radius: ScalarExpr,
    },
    CappedCone {
        radius1: ScalarExpr,
        radius2: ScalarExpr,
        height: ScalarExpr,
    },
    RoundedCylinder {
        radius: ScalarExpr,
        height: ScalarExpr,
        edge_radius: ScalarExpr,
    },
    Union {
        a: Box<ShapeNode>,
        b: Box<ShapeNode>,
    },
    Intersection {
        a: Box<ShapeNode>,
        b: Box<ShapeNode>,
    },
    Difference {
        a: Box<ShapeNode>,
        b: Box<ShapeNode>,
    },
    SmoothUnion {
        a: Box<ShapeNode>,
        b: Box<ShapeNode>,
        k: ScalarExpr,
    },
    SmoothIntersection {
        a: Box<ShapeNode>,
        b: Box<ShapeNode>,
        k: ScalarExpr,
    },
    SmoothDifference {
        a: Box<ShapeNode>,
        b: Box<ShapeNode>,
        k: ScalarExpr,
    },
    Negate {
        shape: Box<ShapeNode>,
    },
    Shell {
        shape: Box<ShapeNode>,
        thickness: ScalarExpr,
    },
    Translate {
        shape: Box<ShapeNode>,
        x: ScalarExpr,
        y: ScalarExpr,
        z: ScalarExpr,
    },
    RotateX {
        shape: Box<ShapeNode>,
        angle: ScalarExpr,
    },
    RotateY {
        shape: Box<ShapeNode>,
        angle: ScalarExpr,
    },
    RotateZ {
        shape: Box<ShapeNode>,
        angle: ScalarExpr,
    },
    Scale {
        shape: Box<ShapeNode>,
        factor: ScalarExpr,
    },
    Mirror {
        shape: Box<ShapeNode>,
        nx: ScalarExpr,
        ny: ScalarExpr,
        nz: ScalarExpr,
        offset: ScalarExpr,
    },
    TwistZ {
        shape: Box<ShapeNode>,
        rate: ScalarExpr,
    },
    BendX {
        shape: Box<ShapeNode>,
        rate: ScalarExpr,
    },
}

impl ShapeNode {
    fn materialize(&self, params: &BTreeMap<String, f64>) -> Result<NumericNode, DslError> {
        match self {
            ShapeNode::Sphere { radius } => {
                let radius = require_non_negative(radius.evaluate(params)?, "sphere radius")?;
                Ok(NumericNode::Sphere { radius })
            }
            ShapeNode::Box {
                width,
                depth,
                height,
            } => {
                let width = require_non_negative(width.evaluate(params)?, "box width")?;
                let depth = require_non_negative(depth.evaluate(params)?, "box depth")?;
                let height = require_non_negative(height.evaluate(params)?, "box height")?;
                Ok(NumericNode::Box {
                    half_extents: [width * 0.5, depth * 0.5, height * 0.5],
                })
            }
            ShapeNode::RoundedBox {
                width,
                depth,
                height,
                radius,
            } => {
                let width = require_non_negative(width.evaluate(params)?, "rounded_box width")?;
                let depth = require_non_negative(depth.evaluate(params)?, "rounded_box depth")?;
                let height = require_non_negative(height.evaluate(params)?, "rounded_box height")?;
                let radius = require_non_negative(radius.evaluate(params)?, "rounded_box radius")?;
                Ok(NumericNode::RoundedBox {
                    half_extents: [width * 0.5, depth * 0.5, height * 0.5],
                    radius,
                })
            }
            ShapeNode::Cylinder { radius, height } => {
                let radius = require_non_negative(radius.evaluate(params)?, "cylinder radius")?;
                let height = require_non_negative(height.evaluate(params)?, "cylinder height")?;
                Ok(NumericNode::Cylinder { radius, height })
            }
            ShapeNode::CappedCylinder {
                radius,
                half_height,
            } => {
                let radius =
                    require_non_negative(radius.evaluate(params)?, "capped_cylinder radius")?;
                let half_height = require_non_negative(
                    half_height.evaluate(params)?,
                    "capped_cylinder half_height",
                )?;
                Ok(NumericNode::CappedCylinder {
                    radius,
                    half_height,
                })
            }
            ShapeNode::Torus {
                major_radius,
                minor_radius,
            } => {
                let major_radius =
                    require_non_negative(major_radius.evaluate(params)?, "torus major_radius")?;
                let minor_radius =
                    require_non_negative(minor_radius.evaluate(params)?, "torus minor_radius")?;
                Ok(NumericNode::Torus {
                    major_radius,
                    minor_radius,
                })
            }
            ShapeNode::Plane { nx, ny, nz, offset } => Ok(NumericNode::Plane {
                normal: [
                    nx.evaluate(params)?,
                    ny.evaluate(params)?,
                    nz.evaluate(params)?,
                ],
                offset: offset.evaluate(params)?,
            }),
            ShapeNode::Capsule {
                ax,
                ay,
                az,
                bx,
                by,
                bz,
                radius,
            } => {
                let radius = require_non_negative(radius.evaluate(params)?, "capsule radius")?;
                Ok(NumericNode::Capsule {
                    a: [
                        ax.evaluate(params)?,
                        ay.evaluate(params)?,
                        az.evaluate(params)?,
                    ],
                    b: [
                        bx.evaluate(params)?,
                        by.evaluate(params)?,
                        bz.evaluate(params)?,
                    ],
                    radius,
                })
            }
            ShapeNode::CappedCone {
                radius1,
                radius2,
                height,
            } => {
                let radius1 =
                    require_non_negative(radius1.evaluate(params)?, "capped_cone radius1")?;
                let radius2 =
                    require_non_negative(radius2.evaluate(params)?, "capped_cone radius2")?;
                let height = require_positive(height.evaluate(params)?, "capped_cone height")?;
                Ok(NumericNode::CappedCone {
                    radius1,
                    radius2,
                    height,
                })
            }
            ShapeNode::RoundedCylinder {
                radius,
                height,
                edge_radius,
            } => {
                let radius =
                    require_non_negative(radius.evaluate(params)?, "rounded_cylinder radius")?;
                let height =
                    require_non_negative(height.evaluate(params)?, "rounded_cylinder height")?;
                let edge_radius = require_non_negative(
                    edge_radius.evaluate(params)?,
                    "rounded_cylinder edge_radius",
                )?;
                Ok(NumericNode::RoundedCylinder {
                    radius,
                    height,
                    edge_radius,
                })
            }
            ShapeNode::Union { a, b } => Ok(NumericNode::Union {
                a: Box::new(a.materialize(params)?),
                b: Box::new(b.materialize(params)?),
            }),
            ShapeNode::Intersection { a, b } => Ok(NumericNode::Intersection {
                a: Box::new(a.materialize(params)?),
                b: Box::new(b.materialize(params)?),
            }),
            ShapeNode::Difference { a, b } => Ok(NumericNode::Difference {
                a: Box::new(a.materialize(params)?),
                b: Box::new(b.materialize(params)?),
            }),
            ShapeNode::SmoothUnion { a, b, k } => Ok(NumericNode::SmoothUnion {
                a: Box::new(a.materialize(params)?),
                b: Box::new(b.materialize(params)?),
                k: k.evaluate(params)?,
            }),
            ShapeNode::SmoothIntersection { a, b, k } => Ok(NumericNode::SmoothIntersection {
                a: Box::new(a.materialize(params)?),
                b: Box::new(b.materialize(params)?),
                k: k.evaluate(params)?,
            }),
            ShapeNode::SmoothDifference { a, b, k } => Ok(NumericNode::SmoothDifference {
                a: Box::new(a.materialize(params)?),
                b: Box::new(b.materialize(params)?),
                k: k.evaluate(params)?,
            }),
            ShapeNode::Negate { shape } => Ok(NumericNode::Negate {
                shape: Box::new(shape.materialize(params)?),
            }),
            ShapeNode::Shell { shape, thickness } => {
                let thickness =
                    require_non_negative(thickness.evaluate(params)?, "shell thickness")?;
                Ok(NumericNode::Shell {
                    shape: Box::new(shape.materialize(params)?),
                    thickness,
                })
            }
            ShapeNode::Translate { shape, x, y, z } => Ok(NumericNode::Translate {
                shape: Box::new(shape.materialize(params)?),
                offset: [
                    x.evaluate(params)?,
                    y.evaluate(params)?,
                    z.evaluate(params)?,
                ],
            }),
            ShapeNode::RotateX { shape, angle } => Ok(NumericNode::RotateX {
                shape: Box::new(shape.materialize(params)?),
                angle: angle.evaluate(params)?,
            }),
            ShapeNode::RotateY { shape, angle } => Ok(NumericNode::RotateY {
                shape: Box::new(shape.materialize(params)?),
                angle: angle.evaluate(params)?,
            }),
            ShapeNode::RotateZ { shape, angle } => Ok(NumericNode::RotateZ {
                shape: Box::new(shape.materialize(params)?),
                angle: angle.evaluate(params)?,
            }),
            ShapeNode::Scale { shape, factor } => {
                let factor = require_positive(factor.evaluate(params)?, "scale factor")?;
                Ok(NumericNode::Scale {
                    shape: Box::new(shape.materialize(params)?),
                    factor,
                })
            }
            ShapeNode::Mirror {
                shape,
                nx,
                ny,
                nz,
                offset,
            } => Ok(NumericNode::Mirror {
                shape: Box::new(shape.materialize(params)?),
                normal: [
                    nx.evaluate(params)?,
                    ny.evaluate(params)?,
                    nz.evaluate(params)?,
                ],
                offset: offset.evaluate(params)?,
            }),
            ShapeNode::TwistZ { shape, rate } => Ok(NumericNode::TwistZ {
                shape: Box::new(shape.materialize(params)?),
                rate: rate.evaluate(params)?,
            }),
            ShapeNode::BendX { shape, rate } => Ok(NumericNode::BendX {
                shape: Box::new(shape.materialize(params)?),
                rate: rate.evaluate(params)?,
            }),
        }
    }
}

#[derive(Debug, Clone)]
enum NumericNode {
    Sphere {
        radius: f64,
    },
    Box {
        half_extents: [f64; 3],
    },
    RoundedBox {
        half_extents: [f64; 3],
        radius: f64,
    },
    Cylinder {
        radius: f64,
        height: f64,
    },
    CappedCylinder {
        radius: f64,
        half_height: f64,
    },
    Torus {
        major_radius: f64,
        minor_radius: f64,
    },
    Plane {
        normal: [f64; 3],
        offset: f64,
    },
    Capsule {
        a: [f64; 3],
        b: [f64; 3],
        radius: f64,
    },
    CappedCone {
        radius1: f64,
        radius2: f64,
        height: f64,
    },
    RoundedCylinder {
        radius: f64,
        height: f64,
        edge_radius: f64,
    },
    Union {
        a: Box<NumericNode>,
        b: Box<NumericNode>,
    },
    Intersection {
        a: Box<NumericNode>,
        b: Box<NumericNode>,
    },
    Difference {
        a: Box<NumericNode>,
        b: Box<NumericNode>,
    },
    SmoothUnion {
        a: Box<NumericNode>,
        b: Box<NumericNode>,
        k: f64,
    },
    SmoothIntersection {
        a: Box<NumericNode>,
        b: Box<NumericNode>,
        k: f64,
    },
    SmoothDifference {
        a: Box<NumericNode>,
        b: Box<NumericNode>,
        k: f64,
    },
    Negate {
        shape: Box<NumericNode>,
    },
    Shell {
        shape: Box<NumericNode>,
        thickness: f64,
    },
    Translate {
        shape: Box<NumericNode>,
        offset: [f64; 3],
    },
    RotateX {
        shape: Box<NumericNode>,
        angle: f64,
    },
    RotateY {
        shape: Box<NumericNode>,
        angle: f64,
    },
    RotateZ {
        shape: Box<NumericNode>,
        angle: f64,
    },
    Scale {
        shape: Box<NumericNode>,
        factor: f64,
    },
    Mirror {
        shape: Box<NumericNode>,
        normal: [f64; 3],
        offset: f64,
    },
    TwistZ {
        shape: Box<NumericNode>,
        rate: f64,
    },
    BendX {
        shape: Box<NumericNode>,
        rate: f64,
    },
}

impl NumericNode {
    fn evaluate(&self, point: [f64; 3]) -> f64 {
        match self {
            NumericNode::Sphere { radius } => sphere_distance(point, *radius),
            NumericNode::Box { half_extents } => box_distance(point, *half_extents),
            NumericNode::RoundedBox {
                half_extents,
                radius,
            } => box_distance(point, *half_extents) - radius,
            NumericNode::Cylinder { radius, height } => cylinder(*radius, *height).evaluate(point),
            NumericNode::CappedCylinder {
                radius,
                half_height,
            } => capped_cylinder(*radius, *half_height).evaluate(point),
            NumericNode::Torus {
                major_radius,
                minor_radius,
            } => torus(*major_radius, *minor_radius).evaluate(point),
            NumericNode::Plane { normal, offset } => plane(*normal, *offset).evaluate(point),
            NumericNode::Capsule { a, b, radius } => capsule(*a, *b, *radius).evaluate(point),
            NumericNode::CappedCone {
                radius1,
                radius2,
                height,
            } => capped_cone(*radius1, *radius2, *height).evaluate(point),
            NumericNode::RoundedCylinder {
                radius,
                height,
                edge_radius,
            } => rounded_cylinder(*radius, *height, *edge_radius).evaluate(point),
            NumericNode::Union { a, b } => union(a.evaluate(point), b.evaluate(point)),
            NumericNode::Intersection { a, b } => {
                intersection(a.evaluate(point), b.evaluate(point))
            }
            NumericNode::Difference { a, b } => difference(a.evaluate(point), b.evaluate(point)),
            NumericNode::SmoothUnion { a, b, k } => {
                smooth_union(a.evaluate(point), b.evaluate(point), *k)
            }
            NumericNode::SmoothIntersection { a, b, k } => {
                smooth_intersection(a.evaluate(point), b.evaluate(point), *k)
            }
            NumericNode::SmoothDifference { a, b, k } => {
                smooth_difference(a.evaluate(point), b.evaluate(point), *k)
            }
            NumericNode::Negate { shape } => negate(shape.evaluate(point)),
            NumericNode::Shell { shape, thickness } => shell(shape.evaluate(point), *thickness),
            NumericNode::Translate { shape, offset } => {
                shape.evaluate(inverse_translate(point, *offset))
            }
            NumericNode::RotateX { shape, angle } => {
                shape.evaluate(inverse_rotate_x(point, *angle))
            }
            NumericNode::RotateY { shape, angle } => {
                shape.evaluate(inverse_rotate_y(point, *angle))
            }
            NumericNode::RotateZ { shape, angle } => {
                shape.evaluate(inverse_rotate_z(point, *angle))
            }
            NumericNode::Scale { shape, factor } => {
                let local = [point[0] / factor, point[1] / factor, point[2] / factor];
                shape.evaluate(local) * factor
            }
            NumericNode::Mirror {
                shape,
                normal,
                offset,
            } => shape.evaluate(mirror_point(point, *normal, *offset)),
            NumericNode::TwistZ { shape, rate } => shape.evaluate(inverse_twist_z(point, *rate)),
            NumericNode::BendX { shape, rate } => shape.evaluate(inverse_bend_x(point, *rate)),
        }
    }
}

fn require_non_negative(value: f64, label: &str) -> Result<f64, DslError> {
    if value < 0.0 {
        Err(DslError::new(format!("{label} must be non-negative")))
    } else {
        Ok(value)
    }
}

fn require_positive(value: f64, label: &str) -> Result<f64, DslError> {
    if value <= 0.0 {
        Err(DslError::new(format!("{label} must be positive")))
    } else {
        Ok(value)
    }
}

fn sphere_distance(point: [f64; 3], radius: f64) -> f64 {
    (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt() - radius
}

fn box_distance(point: [f64; 3], half_extents: [f64; 3]) -> f64 {
    let q = [
        point[0].abs() - half_extents[0],
        point[1].abs() - half_extents[1],
        point[2].abs() - half_extents[2],
    ];
    let outside = length3([q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)]);
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    outside + inside
}

fn length3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub fn parse_program(source: &str) -> Result<Program, DslError> {
    let tokens = Lexer::new(source).tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

pub fn compile_dsl(source: &str) -> Result<Scene, DslError> {
    let program = parse_program(source)?;
    compile_program(&program)
}

pub fn compile_program(program: &Program) -> Result<Scene, DslError> {
    let params = resolve_params(program)?;

    let mut shape_bindings: BTreeMap<String, ShapeNode> = BTreeMap::new();
    let mut last_shape: Option<ShapeNode> = None;

    for item in &program.items {
        match item {
            Item::Assignment(assignment) => {
                let node = compile_shape_expr(&assignment.expr, &shape_bindings, &params)?;
                shape_bindings.insert(assignment.name.clone(), node.clone());
                last_shape = Some(node);
            }
            Item::Expr(expr) => {
                let node = compile_shape_expr(expr, &shape_bindings, &params)?;
                last_shape = Some(node);
            }
        }
    }

    let root = if let Some(result) = shape_bindings.get("result") {
        result.clone()
    } else if let Some(last_shape) = last_shape {
        last_shape
    } else {
        return Err(DslError::new(
            "program does not contain a shape expression to evaluate",
        ));
    };

    Ok(Scene {
        params,
        root,
        compiled: RefCell::new(None),
    })
}

fn resolve_params(program: &Program) -> Result<BTreeMap<String, f64>, DslError> {
    let mut values = BTreeMap::new();

    for decl in &program.params {
        if values.contains_key(&decl.name) {
            return Err(DslError::new(format!(
                "duplicate parameter '{}'",
                decl.name
            )));
        }

        let expr = compile_scalar_expr(&decl.value, &values)?;
        let value = expr.evaluate(&values)?;
        values.insert(decl.name.clone(), value);
    }

    Ok(values)
}

fn compile_shape_expr(
    expr: &Expr,
    shapes: &BTreeMap<String, ShapeNode>,
    params: &BTreeMap<String, f64>,
) -> Result<ShapeNode, DslError> {
    match expr {
        Expr::Variable(name) => shapes
            .get(name)
            .cloned()
            .ok_or_else(|| DslError::new(format!("unknown shape variable '{name}'"))),
        Expr::Call { name, args } => compile_shape_call(name, args, shapes, params),
        _ => Err(DslError::new("expected shape expression")),
    }
}

fn compile_shape_call(
    name: &str,
    args: &[Expr],
    shapes: &BTreeMap<String, ShapeNode>,
    params: &BTreeMap<String, f64>,
) -> Result<ShapeNode, DslError> {
    match name {
        "sphere" => {
            let names = ["radius"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "radius")?;
            Ok(ShapeNode::Sphere {
                radius: compile_scalar_expr(&args[0], params)?,
            })
        }
        "box" | "box3" => {
            let names = ["width", "depth", "height"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "width")?;
            reject_negative_literal(&args[1], "depth")?;
            reject_negative_literal(&args[2], "height")?;
            Ok(ShapeNode::Box {
                width: compile_scalar_expr(&args[0], params)?,
                depth: compile_scalar_expr(&args[1], params)?,
                height: compile_scalar_expr(&args[2], params)?,
            })
        }
        "rounded_box" => {
            let names = ["width", "depth", "height", "radius"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "width")?;
            reject_negative_literal(&args[1], "depth")?;
            reject_negative_literal(&args[2], "height")?;
            reject_negative_literal(&args[3], "radius")?;
            Ok(ShapeNode::RoundedBox {
                width: compile_scalar_expr(&args[0], params)?,
                depth: compile_scalar_expr(&args[1], params)?,
                height: compile_scalar_expr(&args[2], params)?,
                radius: compile_scalar_expr(&args[3], params)?,
            })
        }
        "cylinder" => {
            let names = ["radius", "height"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "radius")?;
            reject_negative_literal(&args[1], "height")?;
            Ok(ShapeNode::Cylinder {
                radius: compile_scalar_expr(&args[0], params)?,
                height: compile_scalar_expr(&args[1], params)?,
            })
        }
        "capped_cylinder" => {
            let names = ["radius", "half_height"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "radius")?;
            reject_negative_literal(&args[1], "half_height")?;
            Ok(ShapeNode::CappedCylinder {
                radius: compile_scalar_expr(&args[0], params)?,
                half_height: compile_scalar_expr(&args[1], params)?,
            })
        }
        "torus" => {
            let names = ["major_radius", "minor_radius"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "major_radius")?;
            reject_negative_literal(&args[1], "minor_radius")?;
            Ok(ShapeNode::Torus {
                major_radius: compile_scalar_expr(&args[0], params)?,
                minor_radius: compile_scalar_expr(&args[1], params)?,
            })
        }
        "plane" => {
            let names = ["nx", "ny", "nz", "offset"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::Plane {
                nx: compile_scalar_expr(&args[0], params)?,
                ny: compile_scalar_expr(&args[1], params)?,
                nz: compile_scalar_expr(&args[2], params)?,
                offset: compile_scalar_expr(&args[3], params)?,
            })
        }
        "capsule" => {
            let names = ["ax", "ay", "az", "bx", "by", "bz", "radius"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[6], "radius")?;
            Ok(ShapeNode::Capsule {
                ax: compile_scalar_expr(&args[0], params)?,
                ay: compile_scalar_expr(&args[1], params)?,
                az: compile_scalar_expr(&args[2], params)?,
                bx: compile_scalar_expr(&args[3], params)?,
                by: compile_scalar_expr(&args[4], params)?,
                bz: compile_scalar_expr(&args[5], params)?,
                radius: compile_scalar_expr(&args[6], params)?,
            })
        }
        "capped_cone" => {
            let names = ["radius1", "radius2", "height"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "radius1")?;
            reject_negative_literal(&args[1], "radius2")?;
            reject_non_positive_literal(&args[2], "height")?;
            Ok(ShapeNode::CappedCone {
                radius1: compile_scalar_expr(&args[0], params)?,
                radius2: compile_scalar_expr(&args[1], params)?,
                height: compile_scalar_expr(&args[2], params)?,
            })
        }
        "rounded_cylinder" => {
            let names = ["radius", "height", "edge_radius"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[0], "radius")?;
            reject_negative_literal(&args[1], "height")?;
            reject_negative_literal(&args[2], "edge_radius")?;
            Ok(ShapeNode::RoundedCylinder {
                radius: compile_scalar_expr(&args[0], params)?,
                height: compile_scalar_expr(&args[1], params)?,
                edge_radius: compile_scalar_expr(&args[2], params)?,
            })
        }
        "union" => {
            let names = ["a", "b"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::Union {
                a: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                b: Box::new(compile_shape_expr(&args[1], shapes, params)?),
            })
        }
        "intersection" => {
            let names = ["a", "b"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::Intersection {
                a: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                b: Box::new(compile_shape_expr(&args[1], shapes, params)?),
            })
        }
        "difference" => {
            let names = ["a", "b"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::Difference {
                a: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                b: Box::new(compile_shape_expr(&args[1], shapes, params)?),
            })
        }
        "smooth_union" => {
            let names = ["a", "b", "k"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::SmoothUnion {
                a: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                b: Box::new(compile_shape_expr(&args[1], shapes, params)?),
                k: compile_scalar_expr(&args[2], params)?,
            })
        }
        "smooth_intersection" => {
            let names = ["a", "b", "k"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::SmoothIntersection {
                a: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                b: Box::new(compile_shape_expr(&args[1], shapes, params)?),
                k: compile_scalar_expr(&args[2], params)?,
            })
        }
        "smooth_difference" => {
            let names = ["a", "b", "k"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::SmoothDifference {
                a: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                b: Box::new(compile_shape_expr(&args[1], shapes, params)?),
                k: compile_scalar_expr(&args[2], params)?,
            })
        }
        "negate" => {
            let names = ["shape"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::Negate {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
            })
        }
        "shell" => {
            let names = ["shape", "thickness"];
            expect_arity(name, args, &names)?;
            reject_negative_literal(&args[1], "thickness")?;
            Ok(ShapeNode::Shell {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                thickness: compile_scalar_expr(&args[1], params)?,
            })
        }
        "translate" => {
            let names = ["shape", "x", "y", "z"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::Translate {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                x: compile_scalar_expr(&args[1], params)?,
                y: compile_scalar_expr(&args[2], params)?,
                z: compile_scalar_expr(&args[3], params)?,
            })
        }
        "rotate" | "rotate_z" => {
            let names = ["shape", "angle"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::RotateZ {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                angle: compile_scalar_expr(&args[1], params)?,
            })
        }
        "rotate_x" => {
            let names = ["shape", "angle"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::RotateX {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                angle: compile_scalar_expr(&args[1], params)?,
            })
        }
        "rotate_y" => {
            let names = ["shape", "angle"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::RotateY {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                angle: compile_scalar_expr(&args[1], params)?,
            })
        }
        "scale" => {
            let names = ["shape", "factor"];
            expect_arity(name, args, &names)?;
            reject_non_positive_literal(&args[1], "factor")?;
            Ok(ShapeNode::Scale {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                factor: compile_scalar_expr(&args[1], params)?,
            })
        }
        "mirror" => {
            let names = ["shape", "nx", "ny", "nz", "offset"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::Mirror {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                nx: compile_scalar_expr(&args[1], params)?,
                ny: compile_scalar_expr(&args[2], params)?,
                nz: compile_scalar_expr(&args[3], params)?,
                offset: compile_scalar_expr(&args[4], params)?,
            })
        }
        "twist" | "twist_z" => {
            let names = ["shape", "rate"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::TwistZ {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                rate: compile_scalar_expr(&args[1], params)?,
            })
        }
        "bend" | "bend_x" => {
            let names = ["shape", "rate"];
            expect_arity(name, args, &names)?;
            Ok(ShapeNode::BendX {
                shape: Box::new(compile_shape_expr(&args[0], shapes, params)?),
                rate: compile_scalar_expr(&args[1], params)?,
            })
        }
        _ => {
            let mut message = format!("unknown primitive or operation '{name}'");
            if let Some(suggestion) = suggest_name(name, KNOWN_FUNCTIONS) {
                message.push_str(&format!(". Did you mean '{suggestion}'?"));
            }
            Err(DslError::new(message))
        }
    }
}

const KNOWN_FUNCTIONS: &[&str] = &[
    "sphere",
    "box",
    "box3",
    "rounded_box",
    "cylinder",
    "capped_cylinder",
    "torus",
    "plane",
    "capsule",
    "capped_cone",
    "rounded_cylinder",
    "union",
    "intersection",
    "difference",
    "smooth_union",
    "smooth_intersection",
    "smooth_difference",
    "negate",
    "shell",
    "translate",
    "rotate",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "scale",
    "mirror",
    "twist",
    "twist_z",
    "bend",
    "bend_x",
];

fn compile_scalar_expr(
    expr: &Expr,
    params: &BTreeMap<String, f64>,
) -> Result<ScalarExpr, DslError> {
    match expr {
        Expr::Number(number) => Ok(ScalarExpr::Constant(number_to_value(number))),
        Expr::Variable(name) => {
            if params.contains_key(name) {
                Ok(ScalarExpr::Parameter(name.clone()))
            } else {
                Err(DslError::new(format!("unknown parameter '{name}'")))
            }
        }
        Expr::Unary { op, expr } => match op {
            UnaryOp::Neg => Ok(ScalarExpr::Neg(Box::new(compile_scalar_expr(
                expr, params,
            )?))),
        },
        Expr::Binary { lhs, op, rhs } => {
            let lhs = Box::new(compile_scalar_expr(lhs, params)?);
            let rhs = Box::new(compile_scalar_expr(rhs, params)?);
            Ok(match op {
                BinaryOp::Add => ScalarExpr::Add(lhs, rhs),
                BinaryOp::Sub => ScalarExpr::Sub(lhs, rhs),
                BinaryOp::Mul => ScalarExpr::Mul(lhs, rhs),
                BinaryOp::Div => ScalarExpr::Div(lhs, rhs),
            })
        }
        Expr::Call { .. } => Err(DslError::new("expected scalar expression")),
    }
}

fn expect_arity(name: &str, args: &[Expr], expected: &[&str]) -> Result<(), DslError> {
    if args.len() < expected.len() {
        return Err(DslError::new(format!(
            "missing parameter '{}' for {}",
            expected[args.len()],
            name
        )));
    }
    if args.len() > expected.len() {
        return Err(DslError::new(format!(
            "too many arguments for {name}: expected {}, got {}",
            expected.len(),
            args.len()
        )));
    }
    Ok(())
}

fn reject_negative_literal(expr: &Expr, name: &str) -> Result<(), DslError> {
    if let Some(value) = literal_value(expr)
        && value < 0.0
    {
        return Err(DslError::new(format!("{name} must be non-negative")));
    }
    Ok(())
}

fn reject_non_positive_literal(expr: &Expr, name: &str) -> Result<(), DslError> {
    if let Some(value) = literal_value(expr)
        && value <= 0.0
    {
        return Err(DslError::new(format!("{name} must be positive")));
    }
    Ok(())
}

fn literal_value(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Number(number) => Some(number_to_value(number)),
        Expr::Unary {
            op: UnaryOp::Neg,
            expr,
        } => literal_value(expr).map(|v| -v),
        _ => None,
    }
}

fn number_to_value(number: &NumberLiteral) -> f64 {
    match number.unit {
        Unit::None | Unit::Mm => number.value,
        Unit::Deg => number.value.to_radians(),
    }
}

fn suggest_name<'a>(name: &str, candidates: &'a [&str]) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;

    for candidate in candidates {
        let distance = levenshtein(name, candidate);
        match best {
            Some((_, best_distance)) if distance >= best_distance => {}
            _ => best = Some((candidate, distance)),
        }
    }

    match best {
        Some((candidate, distance)) if distance <= 3 => Some(candidate),
        _ => None,
    }
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b_len = b.chars().count();
    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0usize; b_len + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let deletion = prev[j + 1] + 1;
            let insertion = curr[j] + 1;
            let substitution = prev[j] + usize::from(ca != cb);
            curr[j + 1] = deletion.min(insertion).min(substitution);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_len]
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Ident(String),
    Number(NumberLiteral),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Eq,
    Plus,
    Minus,
    Star,
    Slash,
    Pipe,
    Eof,
}

#[derive(Debug)]
struct Lexer<'a> {
    source: &'a str,
    index: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            index: 0,
            line: 1,
            column: 1,
        }
    }

    fn tokenize(mut self) -> Result<Vec<Token>, DslError> {
        let mut tokens = Vec::new();

        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.advance_char();
                continue;
            }

            if self.starts_with("//") {
                self.skip_line_comment();
                continue;
            }

            if self.starts_with("/*") {
                self.skip_block_comment()?;
                continue;
            }

            let line = self.line;
            let column = self.column;

            match ch {
                '(' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::LParen,
                        line,
                        column,
                    });
                }
                ')' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::RParen,
                        line,
                        column,
                    });
                }
                '{' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::LBrace,
                        line,
                        column,
                    });
                }
                '}' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::RBrace,
                        line,
                        column,
                    });
                }
                ',' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::Comma,
                        line,
                        column,
                    });
                }
                '=' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::Eq,
                        line,
                        column,
                    });
                }
                '+' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::Plus,
                        line,
                        column,
                    });
                }
                '-' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::Minus,
                        line,
                        column,
                    });
                }
                '*' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::Star,
                        line,
                        column,
                    });
                }
                '/' => {
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::Slash,
                        line,
                        column,
                    });
                }
                '|' if self.starts_with("|>") => {
                    self.advance_char();
                    self.advance_char();
                    tokens.push(Token {
                        kind: TokenKind::Pipe,
                        line,
                        column,
                    });
                }
                c if is_ident_start(c) => tokens.push(self.lex_identifier()?),
                c if c.is_ascii_digit()
                    || (c == '.'
                        && self
                            .peek_second_char()
                            .map(|next| next.is_ascii_digit())
                            .unwrap_or(false)) =>
                {
                    tokens.push(self.lex_number()?)
                }
                _ => {
                    return Err(DslError::at(
                        format!("unexpected character '{ch}'"),
                        self.line,
                        self.column,
                    ));
                }
            }
        }

        tokens.push(Token {
            kind: TokenKind::Eof,
            line: self.line,
            column: self.column,
        });

        Ok(tokens)
    }

    fn lex_identifier(&mut self) -> Result<Token, DslError> {
        let line = self.line;
        let column = self.column;
        let start = self.index;

        self.advance_char();
        while self.peek_char().map(is_ident_continue).unwrap_or(false) {
            self.advance_char();
        }

        let ident = self
            .source
            .get(start..self.index)
            .ok_or_else(|| DslError::new("invalid identifier span"))?
            .to_string();

        Ok(Token {
            kind: TokenKind::Ident(ident),
            line,
            column,
        })
    }

    fn lex_number(&mut self) -> Result<Token, DslError> {
        let line = self.line;
        let column = self.column;
        let start = self.index;

        let mut seen_digit = false;
        let mut seen_dot = false;

        if self.peek_char() == Some('.') {
            seen_dot = true;
            self.advance_char();
        }

        while self
            .peek_char()
            .map(|ch| ch.is_ascii_digit())
            .unwrap_or(false)
        {
            seen_digit = true;
            self.advance_char();
        }

        if self.peek_char() == Some('.') && !seen_dot {
            self.advance_char();
            while self
                .peek_char()
                .map(|ch| ch.is_ascii_digit())
                .unwrap_or(false)
            {
                seen_digit = true;
                self.advance_char();
            }
        }

        if let Some(exp) = self.peek_char()
            && (exp == 'e' || exp == 'E')
        {
            self.advance_char();
            if let Some(sign) = self.peek_char()
                && (sign == '+' || sign == '-')
            {
                self.advance_char();
            }

            let mut exp_digits = 0usize;
            while self
                .peek_char()
                .map(|ch| ch.is_ascii_digit())
                .unwrap_or(false)
            {
                exp_digits += 1;
                self.advance_char();
            }

            if exp_digits == 0 {
                return Err(DslError::at("invalid exponent in number", line, column));
            }
        }

        if !seen_digit {
            return Err(DslError::at("invalid number literal", line, column));
        }

        let text = self
            .source
            .get(start..self.index)
            .ok_or_else(|| DslError::new("invalid number span"))?;

        let value = text
            .parse::<f64>()
            .map_err(|err| DslError::at(format!("invalid number literal: {err}"), line, column))?;

        let unit = if self.starts_with("mm") {
            self.advance_char();
            self.advance_char();
            Unit::Mm
        } else if self.starts_with("deg") {
            self.advance_char();
            self.advance_char();
            self.advance_char();
            Unit::Deg
        } else {
            Unit::None
        };

        Ok(Token {
            kind: TokenKind::Number(NumberLiteral { value, unit }),
            line,
            column,
        })
    }

    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.peek_char() {
            self.advance_char();
            if ch == '\n' {
                break;
            }
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), DslError> {
        let start_line = self.line;
        let start_column = self.column;
        self.advance_char();
        self.advance_char();

        while self.index < self.source.len() {
            if self.starts_with("*/") {
                self.advance_char();
                self.advance_char();
                return Ok(());
            }
            self.advance_char();
        }

        Err(DslError::at(
            "unterminated block comment",
            start_line,
            start_column,
        ))
    }

    fn starts_with(&self, text: &str) -> bool {
        self.source[self.index..].starts_with(text)
    }

    fn peek_char(&self) -> Option<char> {
        self.source[self.index..].chars().next()
    }

    fn peek_second_char(&self) -> Option<char> {
        let mut chars = self.source[self.index..].chars();
        chars.next()?;
        chars.next()
    }

    fn advance_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.index += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

#[derive(Debug)]
struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, index: 0 }
    }

    fn parse_program(&mut self) -> Result<Program, DslError> {
        if self.check_kind(&TokenKind::Eof) {
            let token = self.peek();
            return Err(DslError::at("empty program", token.line, token.column));
        }

        let mut params = Vec::new();
        if self.check_ident("params") && self.peek_n_is(1, &TokenKind::LBrace) {
            self.advance();
            self.expect_kind(TokenKind::LBrace, "expected '{' after params")?;

            while !self.check_kind(&TokenKind::RBrace) {
                if self.check_kind(&TokenKind::Eof) {
                    let token = self.peek();
                    return Err(DslError::at(
                        "unterminated params block",
                        token.line,
                        token.column,
                    ));
                }

                let name = self.consume_ident("expected parameter name")?;
                self.expect_kind(TokenKind::Eq, "expected '=' in params block")?;
                let value = self.parse_expression()?;
                params.push(ParamDecl { name, value });
            }

            self.expect_kind(TokenKind::RBrace, "expected '}' to close params block")?;
        }

        let mut items = Vec::new();
        while !self.check_kind(&TokenKind::Eof) {
            if self.next_is_assignment() {
                let name = self.consume_ident("expected assignment target")?;
                self.expect_kind(TokenKind::Eq, "expected '=' in assignment")?;
                let expr = self.parse_expression()?;
                items.push(Item::Assignment(Assignment { name, expr }));
            } else {
                let expr = self.parse_expression()?;
                items.push(Item::Expr(expr));
            }
        }

        if params.is_empty() && items.is_empty() {
            let token = self.peek();
            return Err(DslError::at("empty program", token.line, token.column));
        }

        Ok(Program { params, items })
    }

    fn parse_expression(&mut self) -> Result<Expr, DslError> {
        self.parse_pipe()
    }

    fn parse_pipe(&mut self) -> Result<Expr, DslError> {
        let mut expr = self.parse_add_sub()?;
        while self.match_kind(&TokenKind::Pipe) {
            let name = self.consume_ident("expected function name after '|>'")?;
            self.expect_kind(TokenKind::LParen, "expected '(' after pipe stage name")?;

            let mut args = vec![expr];
            if !self.check_kind(&TokenKind::RParen) {
                loop {
                    args.push(self.parse_expression()?);
                    if !self.match_kind(&TokenKind::Comma) {
                        break;
                    }
                }
            }

            self.expect_kind(TokenKind::RParen, "expected ')' after pipe stage arguments")?;
            expr = Expr::Call { name, args };
        }

        Ok(expr)
    }

    fn parse_add_sub(&mut self) -> Result<Expr, DslError> {
        let mut expr = self.parse_mul_div()?;

        loop {
            if self.match_kind(&TokenKind::Plus) {
                let rhs = self.parse_mul_div()?;
                expr = Expr::Binary {
                    lhs: Box::new(expr),
                    op: BinaryOp::Add,
                    rhs: Box::new(rhs),
                };
            } else if self.match_kind(&TokenKind::Minus) {
                let rhs = self.parse_mul_div()?;
                expr = Expr::Binary {
                    lhs: Box::new(expr),
                    op: BinaryOp::Sub,
                    rhs: Box::new(rhs),
                };
            } else {
                return Ok(expr);
            }
        }
    }

    fn parse_mul_div(&mut self) -> Result<Expr, DslError> {
        let mut expr = self.parse_unary()?;

        loop {
            if self.match_kind(&TokenKind::Star) {
                let rhs = self.parse_unary()?;
                expr = Expr::Binary {
                    lhs: Box::new(expr),
                    op: BinaryOp::Mul,
                    rhs: Box::new(rhs),
                };
            } else if self.match_kind(&TokenKind::Slash) {
                let rhs = self.parse_unary()?;
                expr = Expr::Binary {
                    lhs: Box::new(expr),
                    op: BinaryOp::Div,
                    rhs: Box::new(rhs),
                };
            } else {
                return Ok(expr);
            }
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, DslError> {
        if self.match_kind(&TokenKind::Minus) {
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(expr),
            });
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, DslError> {
        let token = self.peek().clone();
        match &token.kind {
            TokenKind::Number(number) => {
                self.advance();
                Ok(Expr::Number(number.clone()))
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                if self.match_kind(&TokenKind::LParen) {
                    let mut args = Vec::new();
                    if !self.check_kind(&TokenKind::RParen) {
                        loop {
                            args.push(self.parse_expression()?);
                            if !self.match_kind(&TokenKind::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect_kind(TokenKind::RParen, "expected ')' after call arguments")?;
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Variable(name))
                }
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect_kind(TokenKind::RParen, "expected ')' to close group")?;
                Ok(expr)
            }
            _ => Err(DslError::at(
                "expected expression",
                token.line,
                token.column,
            )),
        }
    }

    fn next_is_assignment(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(_)) && self.peek_n_is(1, &TokenKind::Eq)
    }

    fn consume_ident(&mut self, message: &str) -> Result<String, DslError> {
        let token = self.peek().clone();
        match token.kind {
            TokenKind::Ident(name) => {
                self.advance();
                Ok(name)
            }
            _ => Err(DslError::at(message, token.line, token.column)),
        }
    }

    fn expect_kind(&mut self, expected: TokenKind, message: &str) -> Result<(), DslError> {
        if self.match_kind(&expected) {
            Ok(())
        } else {
            let token = self.peek();
            Err(DslError::at(message, token.line, token.column))
        }
    }

    fn check_ident(&self, text: &str) -> bool {
        matches!(
            self.peek_kind(),
            TokenKind::Ident(name) if name == text
        )
    }

    fn match_kind(&mut self, expected: &TokenKind) -> bool {
        if self.check_kind(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check_kind(&self, expected: &TokenKind) -> bool {
        same_variant(self.peek_kind(), expected)
    }

    fn peek_n_is(&self, n: usize, expected: &TokenKind) -> bool {
        self.tokens
            .get(self.index + n)
            .map(|token| same_variant(&token.kind, expected))
            .unwrap_or(false)
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.index]
    }

    fn advance(&mut self) {
        if self.index + 1 < self.tokens.len() {
            self.index += 1;
        }
    }
}

fn same_variant(a: &TokenKind, b: &TokenKind) -> bool {
    matches!(
        (a, b),
        (TokenKind::LParen, TokenKind::LParen)
            | (TokenKind::RParen, TokenKind::RParen)
            | (TokenKind::LBrace, TokenKind::LBrace)
            | (TokenKind::RBrace, TokenKind::RBrace)
            | (TokenKind::Comma, TokenKind::Comma)
            | (TokenKind::Eq, TokenKind::Eq)
            | (TokenKind::Plus, TokenKind::Plus)
            | (TokenKind::Minus, TokenKind::Minus)
            | (TokenKind::Star, TokenKind::Star)
            | (TokenKind::Slash, TokenKind::Slash)
            | (TokenKind::Pipe, TokenKind::Pipe)
            | (TokenKind::Eof, TokenKind::Eof)
            | (TokenKind::Ident(_), TokenKind::Ident(_))
            | (TokenKind::Number(_), TokenKind::Number(_))
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::f64::consts::PI;
    use std::hint::black_box;
    use std::time::Instant;

    use sdf_core::{Sdf3, box3, inverse_translate, sphere, union};
    use sdf_mesh::to_binary_stl;
    use sdf_mesh::{MarchingCubesConfig, Mesh, extract_mesh_with};

    use super::{BinaryOp, Expr, Item, NumberLiteral, Program, Unit, compile_dsl, parse_program};

    #[test]
    fn parses_minimal_sphere_program() {
        let program = parse_program("sphere(10mm)").expect("program should parse");
        assert!(program.params.is_empty());
        assert_eq!(program.items.len(), 1);

        match &program.items[0] {
            Item::Expr(Expr::Call { name, args }) => {
                assert_eq!(name, "sphere");
                assert_eq!(args.len(), 1);
                assert_eq!(
                    args[0],
                    Expr::Number(NumberLiteral {
                        value: 10.0,
                        unit: Unit::Mm,
                    })
                );
            }
            _ => panic!("expected sphere call expression"),
        }
    }

    #[test]
    fn parses_all_primitives() {
        let inputs = [
            "sphere(10mm)",
            "box(10mm, 20mm, 30mm)",
            "rounded_box(10mm, 20mm, 30mm, 2mm)",
            "cylinder(10mm, 30mm)",
            "capped_cylinder(10mm, 15mm)",
            "torus(20mm, 4mm)",
            "plane(0, 0, 1, 2mm)",
            "capsule(0, 0, -5mm, 0, 0, 5mm, 2mm)",
            "capped_cone(8mm, 4mm, 20mm)",
            "rounded_cylinder(10mm, 20mm, 2mm)",
        ];

        for input in inputs {
            compile_dsl(input).unwrap_or_else(|err| {
                panic!("expected primitive '{input}' to compile, got: {err}");
            });
        }
    }

    #[test]
    fn parses_operations() {
        let inputs = [
            "union(sphere(10mm), box(20mm, 20mm, 20mm))",
            "intersection(sphere(10mm), box(20mm, 20mm, 20mm))",
            "difference(sphere(10mm), box(20mm, 20mm, 20mm))",
            "smooth_union(sphere(10mm), box(20mm, 20mm, 20mm), 2mm)",
            "smooth_intersection(sphere(10mm), box(20mm, 20mm, 20mm), 2mm)",
            "smooth_difference(sphere(10mm), box(20mm, 20mm, 20mm), 2mm)",
        ];

        for input in inputs {
            compile_dsl(input).unwrap_or_else(|err| {
                panic!("expected operation '{input}' to compile, got: {err}");
            });
        }
    }

    #[test]
    fn parses_transforms() {
        let inputs = [
            "translate(sphere(10mm), 5mm, 0, 0)",
            "rotate(sphere(10mm), 15deg)",
            "rotate_x(sphere(10mm), 15deg)",
            "scale(sphere(10mm), 2)",
        ];

        for input in inputs {
            compile_dsl(input).unwrap_or_else(|err| {
                panic!("expected transform '{input}' to compile, got: {err}");
            });
        }
    }

    #[test]
    fn parses_pipe_operator_into_nested_calls() {
        let program = parse_program("sphere(5mm) |> translate(10, 0, 0)")
            .expect("pipe expression should parse");

        match &program.items[0] {
            Item::Expr(Expr::Call { name, args }) => {
                assert_eq!(name, "translate");
                assert_eq!(args.len(), 4);
                match &args[0] {
                    Expr::Call {
                        name: sphere_name,
                        args: sphere_args,
                    } => {
                        assert_eq!(sphere_name, "sphere");
                        assert_eq!(sphere_args.len(), 1);
                    }
                    _ => panic!("expected left side of pipe as first argument"),
                }
            }
            _ => panic!("expected call expression"),
        }
    }

    #[test]
    fn resolves_params_and_arithmetic() {
        let scene = compile_dsl(
            r#"
            params {
              width = 20mm
              radius = width / 2 + 5mm
            }
            sphere(radius)
            "#,
        )
        .expect("program should compile");

        let radius = scene
            .parameter("radius")
            .expect("radius parameter should exist");
        assert!((radius - 15.0).abs() < 1e-12);

        let value = scene
            .evaluate([0.0, 0.0, 0.0])
            .expect("evaluation should succeed");
        assert!((value + 15.0).abs() < 1e-12);
    }

    #[test]
    fn reports_line_and_column_for_malformed_input() {
        let err = parse_program("sphere(10mm").expect_err("parse should fail");
        let text = err.to_string();
        assert!(text.contains("line 1, column"), "error text: {text}");
    }

    #[test]
    fn ignores_line_and_block_comments() {
        let source = r#"
            // header comment
            /* block
               comment */
            sphere(10mm) // trailing comment
        "#;

        let scene = compile_dsl(source).expect("comments should be ignored");
        let value = scene
            .evaluate([0.0, 0.0, 0.0])
            .expect("evaluation should succeed");
        assert!((value + 10.0).abs() < 1e-12);
    }

    #[test]
    fn phone_stand_program_parses_and_meshes() {
        let scene = compile_dsl(phone_stand_source()).expect("phone stand should compile");
        let config = MarchingCubesConfig::new(
            [-60.0, -60.0, -40.0],
            [60.0, 60.0, 120.0],
            [56, 56, 56],
            0.0,
        );

        let mesh = extract_mesh_with(&config, |point| {
            scene
                .evaluate(point)
                .expect("phone stand evaluation should succeed")
        });

        assert!(!mesh.triangles.is_empty());
        let (watertight, _) = watertight_stats(&mesh);
        assert!(watertight);

        let volume = mesh_volume(&mesh).abs();
        assert!(
            volume > 10_000.0 && volume < 2_000_000.0,
            "unexpected phone stand volume: {volume}"
        );
    }

    #[test]
    fn round_trip_parse_serialize_parse_is_identical() {
        let source = r#"
            params {
              width = 40mm
              radius = width / 4 + 1mm
            }

            base = sphere(radius)
            result = difference(base |> translate(5mm, 0, 0), sphere(3mm))
        "#;

        let first = parse_program(source).expect("first parse should succeed");
        let serialized = first.to_source();
        let second = parse_program(&serialized).expect("second parse should succeed");
        assert_eq!(first, second);
    }

    #[test]
    fn empty_program_returns_error() {
        let err = parse_program("  \n // only comments").expect_err("expected empty error");
        assert!(err.to_string().contains("empty program"));
    }

    #[test]
    fn unknown_primitive_returns_suggestion() {
        let err = compile_dsl("sphre(10mm)").expect_err("compile should fail");
        let text = err.to_string();
        assert!(text.contains("unknown primitive or operation 'sphre'"));
        assert!(
            text.contains("Did you mean 'sphere'?"),
            "error text: {text}"
        );
    }

    #[test]
    fn missing_parameter_returns_named_error() {
        let err = compile_dsl("sphere()").expect_err("compile should fail");
        assert!(err.to_string().contains("missing parameter 'radius'"));
    }

    #[test]
    fn division_by_zero_fails_at_evaluation_time() {
        let scene = compile_dsl(
            r#"
            params {
              width = 10mm
            }
            sphere(width / 0)
            "#,
        )
        .expect("compile should succeed");

        let err = scene
            .evaluate([0.0, 0.0, 0.0])
            .expect_err("evaluation should fail");
        assert!(err.to_string().contains("division by zero"));
    }

    #[test]
    fn negative_radius_fails_at_compile_time() {
        let err = compile_dsl("sphere(-1mm)").expect_err("compile should fail");
        assert!(err.to_string().contains("radius must be non-negative"));
    }

    #[test]
    fn arithmetic_ast_is_parsed() {
        let program = parse_program("sphere(10 + 2 * 3)").expect("parse should succeed");
        match &program.items[0] {
            Item::Expr(Expr::Call { args, .. }) => match &args[0] {
                Expr::Binary { op, rhs, .. } => {
                    assert_eq!(*op, BinaryOp::Add);
                    assert!(matches!(
                        rhs.as_ref(),
                        Expr::Binary {
                            op: BinaryOp::Mul,
                            ..
                        }
                    ));
                }
                _ => panic!("expected binary expression"),
            },
            _ => panic!("expected call expression"),
        }
    }

    #[test]
    fn dsl_compile_matches_manual_scene_at_10000_points() {
        let source = r#"
            result = union(
              sphere(1mm),
              translate(box(1.6mm, 1.2mm, 1.8mm), 0.35mm, -0.15mm, 0)
            )
        "#;
        let scene = compile_dsl(source).expect("dsl should compile");

        for point in sample_points_10000() {
            let dsl_value = scene
                .evaluate(point)
                .expect("dsl evaluation should succeed");
            let manual_value = manual_union_scene(point);
            assert!(
                (dsl_value - manual_value).abs() < 1e-12,
                "point={point:?}, dsl={dsl_value}, manual={manual_value}"
            );
        }
    }

    #[test]
    fn compile_time_for_100_line_program_is_under_10ms() {
        let source = build_large_program(100);
        let start = Instant::now();
        let scene = compile_dsl(&source).expect("large program should compile");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        assert!(scene.evaluate([0.0, 0.0, 0.0]).is_ok());
        assert!(
            elapsed_ms < 10.0,
            "expected compile time under 10ms, got {elapsed_ms:.3}ms"
        );
    }

    #[test]
    fn evaluation_throughput_within_10_percent_of_manual() {
        let scene = compile_dsl("sphere(1mm)").expect("dsl should compile");
        let compiled = scene
            .compile_current()
            .expect("compiled scene should succeed");
        let manual = super::NumericNode::Sphere { radius: 1.0 };
        let points = sample_points_10000();
        let repeats = 25usize;

        // Warm up both paths to reduce one-time effects.
        let _ = points
            .iter()
            .map(|point| black_box(compiled.evaluate(black_box(*point))))
            .sum::<f64>();
        let _ = points
            .iter()
            .map(|point| black_box(manual.evaluate(black_box(*point))))
            .sum::<f64>();

        let dsl_start = Instant::now();
        let mut dsl_sum = 0.0f64;
        for _ in 0..repeats {
            dsl_sum += points
                .iter()
                .map(|point| black_box(compiled.evaluate(black_box(*point))))
                .sum::<f64>();
        }
        let dsl_time = dsl_start.elapsed().as_secs_f64();

        let manual_start = Instant::now();
        let mut manual_sum = 0.0f64;
        for _ in 0..repeats {
            manual_sum += points
                .iter()
                .map(|point| black_box(manual.evaluate(black_box(*point))))
                .sum::<f64>();
        }
        let manual_time = manual_start.elapsed().as_secs_f64();

        // Ensure work is not optimized away.
        assert!(dsl_sum.is_finite());
        assert!(manual_sum.is_finite());

        let slowdown = (dsl_time / manual_time) - 1.0;
        let max_slowdown = if cfg!(debug_assertions) { 2.00 } else { 0.10 };
        assert!(
            slowdown <= max_slowdown,
            "dsl throughput exceeded allowed slowdown: dsl={dsl_time:.6}s manual={manual_time:.6}s slowdown={slowdown:.3} limit={max_slowdown:.3}"
        );
    }

    #[test]
    fn compilation_is_deterministic() {
        let source = "smooth_union(sphere(1mm), translate(box(2mm, 2mm, 2mm), 0.1, 0.2, 0.3), 0.2)";
        let scene_a = compile_dsl(source).expect("first compile should succeed");
        let scene_b = compile_dsl(source).expect("second compile should succeed");

        for point in sample_points_10000() {
            let a = scene_a.evaluate(point).expect("scene_a evaluation");
            let b = scene_b.evaluate(point).expect("scene_b evaluation");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn full_pipeline_dsl_to_stl_volume_is_within_one_percent() {
        let scene = compile_dsl("sphere(10mm)").expect("dsl should compile");
        let config =
            MarchingCubesConfig::new([-12.0, -12.0, -12.0], [12.0, 12.0, 12.0], [96, 96, 96], 0.0);
        let mesh = extract_mesh_with(&config, |point| {
            scene
                .evaluate(point)
                .expect("sphere evaluation should succeed")
        });
        let stl = to_binary_stl(&mesh, "sphere");
        let volume = binary_stl_volume(&stl).abs();
        let expected = (4.0 / 3.0) * PI * 10.0_f64.powi(3);
        let rel = ((volume - expected) / expected).abs();

        assert!(
            rel < 0.01,
            "sphere STL volume outside 1%: got={volume:.6}, expected={expected:.6}, rel={rel:.4}"
        );
    }

    #[test]
    fn set_param_changes_sdf_value_for_sphere_radius() {
        let mut scene = compile_dsl(
            r#"
            params {
              radius = 10mm
            }
            sphere(radius)
            "#,
        )
        .expect("scene should compile");

        let initial = scene
            .evaluate([0.0, 0.0, 0.0])
            .expect("initial evaluation should succeed");
        assert!((initial + 10.0).abs() < 1e-12);

        scene
            .set_param("radius", 20.0)
            .expect("set_param should succeed");
        let updated = scene
            .evaluate([0.0, 0.0, 0.0])
            .expect("updated evaluation should succeed");
        assert!((updated + 20.0).abs() < 1e-12);
    }

    #[test]
    fn set_param_invalid_name_returns_error() {
        let mut scene = compile_dsl(
            r#"
            params {
              radius = 10mm
            }
            sphere(radius)
            "#,
        )
        .expect("scene should compile");

        let err = scene
            .set_param("not_a_param", 5.0)
            .expect_err("set_param should fail");
        assert!(err.to_string().contains("unknown parameter 'not_a_param'"));
    }

    #[test]
    fn set_param_rejects_out_of_range_value() {
        let mut scene = compile_dsl(
            r#"
            params {
              radius = 10mm
            }
            sphere(radius)
            "#,
        )
        .expect("scene should compile");

        let err = scene
            .set_param("radius", -1.0)
            .expect_err("set_param should fail");
        let text = err.to_string();
        assert!(text.contains("invalid value for parameter 'radius'"));
        assert!(text.contains("sphere radius must be non-negative"));

        let radius = scene
            .parameter("radius")
            .expect("radius should still exist");
        assert!((radius - 10.0).abs() < 1e-12);
    }

    #[test]
    fn phone_stand_five_param_changes_adjust_volume_proportionally() {
        let mut scene = compile_dsl(phone_stand_source()).expect("phone stand should compile");
        let base_mesh = scene
            .evaluate_mesh_with_bounds(56, [-60.0, -60.0, -40.0], [60.0, 60.0, 120.0])
            .expect("baseline mesh evaluation should succeed");
        let base_volume = mesh_volume(&base_mesh).abs();

        scene
            .set_param("width", 96.0)
            .expect("width update should succeed");
        scene
            .set_param("depth", 72.0)
            .expect("depth update should succeed");
        scene
            .set_param("height", 120.0)
            .expect("height update should succeed");
        scene
            .set_param("thickness", 3.6)
            .expect("thickness update should succeed");
        scene
            .set_param("slot_width", 14.4)
            .expect("slot width update should succeed");

        let updated_mesh = scene
            .evaluate_mesh_with_bounds(56, [-75.0, -75.0, -50.0], [75.0, 75.0, 150.0])
            .expect("updated mesh evaluation should succeed");
        let updated_volume = mesh_volume(&updated_mesh).abs();

        let measured_ratio = updated_volume / base_volume;
        let expected_ratio = (96.0 * 72.0 * 120.0) / (80.0 * 60.0 * 100.0);
        let relative_error = ((measured_ratio - expected_ratio) / expected_ratio).abs();

        assert!(
            relative_error < 0.35,
            "volume ratio should track scale changes: measured={measured_ratio:.3}, expected={expected_ratio:.3}, rel_error={relative_error:.3}"
        );
    }

    #[test]
    fn param_change_and_reevaluate_64_cubed_within_budget() {
        let mut scene = compile_dsl(phone_stand_source()).expect("phone stand should compile");
        let start = Instant::now();
        scene
            .set_param("width", 90.0)
            .expect("parameter update should succeed");
        let mesh = scene
            .evaluate_mesh_with_bounds(64, [-70.0, -70.0, -45.0], [70.0, 70.0, 135.0])
            .expect("mesh evaluation should succeed");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let limit_ms = if cfg!(debug_assertions) { 2500.0 } else { 200.0 };

        assert!(!mesh.triangles.is_empty());
        assert!(
            elapsed_ms < limit_ms,
            "parameter change + 64^3 re-evaluate exceeded budget: elapsed={elapsed_ms:.3}ms limit={limit_ms:.1}ms"
        );
    }

    #[test]
    fn setting_param_and_reverting_produces_bit_identical_mesh() {
        let mut scene = compile_dsl(
            r#"
            params {
              radius = 10mm
            }
            sphere(radius)
            "#,
        )
        .expect("scene should compile");

        let before = scene
            .evaluate_mesh_with_bounds(64, [-12.0, -12.0, -12.0], [12.0, 12.0, 12.0])
            .expect("initial mesh should evaluate");
        let before_stl = to_binary_stl(&before, "sphere");

        scene
            .set_param("radius", 12.0)
            .expect("intermediate parameter update should succeed");
        let _ = scene
            .evaluate_mesh_with_bounds(64, [-14.0, -14.0, -14.0], [14.0, 14.0, 14.0])
            .expect("intermediate mesh should evaluate");

        scene
            .set_param("radius", 10.0)
            .expect("revert parameter update should succeed");
        let after = scene
            .evaluate_mesh_with_bounds(64, [-12.0, -12.0, -12.0], [12.0, 12.0, 12.0])
            .expect("reverted mesh should evaluate");
        let after_stl = to_binary_stl(&after, "sphere");

        assert_eq!(before_stl, after_stl);
    }

    fn watertight_stats(mesh: &Mesh) -> (bool, usize) {
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();

        for tri in &mesh.triangles {
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let key = if edge.0 < edge.1 {
                    (edge.0, edge.1)
                } else {
                    (edge.1, edge.0)
                };
                *edges.entry(key).or_insert(0) += 1;
            }
        }

        let bad_edges = edges.values().filter(|count| **count != 2).count();
        (bad_edges == 0, bad_edges)
    }

    fn mesh_volume(mesh: &Mesh) -> f64 {
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

    fn phone_stand_source() -> &'static str {
        r#"
            // Phone stand with 15-degree tilt
            params {
              width = 80mm
              depth = 60mm
              height = 100mm
              tilt_angle = 15deg
              thickness = 3mm
              slot_width = 12mm
              fillet_radius = 2mm
            }

            base = rounded_box(width, depth, thickness, fillet_radius)

            back_wall = rounded_box(width, thickness, height, fillet_radius)
              |> translate(0, -depth/2 + thickness/2, height/2 - thickness/2)
              |> rotate_x(tilt_angle)

            slot = rounded_box(slot_width, thickness * 2, height * 0.6, 1mm)
              |> translate(0, -depth/2, height * 0.3)
              |> rotate_x(tilt_angle)

            result = smooth_union(base, back_wall, 3mm)
              |> difference(slot)
        "#
    }

    fn manual_union_scene(point: [f64; 3]) -> f64 {
        let a = sphere(1.0).evaluate(point);
        let local = inverse_translate(point, [0.35, -0.15, 0.0]);
        let b = box3([0.8, 0.6, 0.9]).evaluate(local);
        union(a, b)
    }

    fn sample_points_10000() -> Vec<[f64; 3]> {
        let mut points = Vec::with_capacity(10_000);
        let n = 10_000usize;
        for i in 0..n {
            let t = i as f64;
            let x = ((t * 0.754_877_666).sin() * 1.7).clamp(-2.0, 2.0);
            let y = ((t * 0.569_840_291).cos() * 1.7).clamp(-2.0, 2.0);
            let z = (((t * 0.438_579_123).sin() + (t * 0.233_117_9).cos()) * 0.85).clamp(-2.0, 2.0);
            points.push([x, y, z]);
        }
        points
    }

    fn build_large_program(lines: usize) -> String {
        let mut source = String::from("params {\n  p0 = 10mm\n");
        for i in 1..lines {
            source.push_str(&format!("  p{i} = p{} + 0.1mm\n", i - 1));
        }
        source.push_str("}\n\n");
        source.push_str(&format!("result = sphere(p{})\n", lines - 1));
        source
    }

    fn binary_stl_volume(bytes: &[u8]) -> f64 {
        assert!(
            bytes.len() >= 84,
            "binary STL must contain header and count"
        );
        let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        assert_eq!(bytes.len(), 84 + tri_count * 50);

        let mut offset = 84usize;
        let mut volume = 0.0f64;

        for _ in 0..tri_count {
            // Skip normal.
            offset += 12;
            let a = read_vec3(bytes, offset);
            let b = read_vec3(bytes, offset + 12);
            let c = read_vec3(bytes, offset + 24);
            volume += dot(a, cross(b, c)) / 6.0;
            // Skip vertices and attribute byte count.
            offset += 38;
        }

        volume
    }

    fn read_vec3(bytes: &[u8], offset: usize) -> [f64; 3] {
        [
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as f64,
            f32::from_le_bytes([
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]) as f64,
            f32::from_le_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
            ]) as f64,
        ]
    }

    #[allow(dead_code)]
    fn _assert_round_trip(program: &Program) {
        let text = program.to_source();
        let reparsed = parse_program(&text).expect("round-trip parse should succeed");
        assert_eq!(*program, reparsed);
    }
}
