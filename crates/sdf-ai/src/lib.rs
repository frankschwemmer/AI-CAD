use std::collections::BTreeMap;
use std::fmt;
use std::time::Instant;

use sdf_dsl::compile_dsl;
use sdf_mesh::Mesh;

pub const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an SDF CAD generator that outputs valid SDF DSL programs.

DSL grammar reference:
- Program format:
  params {
    name = expression
  }
  result = expression
- Numbers support units: mm, deg, or unitless scalars.
- Expressions support +, -, *, / and parentheses.
- Include a final shape expression or a `result = ...` assignment.

Primitive catalog:
- sphere(radius)
- box(width, depth, height)
- rounded_box(width, depth, height, radius)
- cylinder(radius, height)
- capped_cylinder(radius, half_height)
- torus(major_radius, minor_radius)
- capsule(ax, ay, az, bx, by, bz, radius)
- capped_cone(radius1, radius2, height)
- rounded_cylinder(radius, height, edge_radius)

Operation catalog:
- union(a, b)
- intersection(a, b)
- difference(a, b)
- smooth_union(a, b, k)
- smooth_intersection(a, b, k)
- smooth_difference(a, b, k)
- shell(shape, thickness)
- negate(shape)

Transform catalog:
- translate(shape, x, y, z)
- rotate(shape, angle) / rotate_x(shape, angle) / rotate_y(shape, angle)
- scale(shape, factor)
- mirror(shape, nx, ny, nz, offset)
- twist(shape, rate)
- bend(shape, rate)

Coordinate conventions:
- Y-up coordinate system.
- Dimensions are in millimeters unless explicitly unitless.

Common design patterns:
- Enclosure: difference(outer_shell, inner_shell)
- Bracket/stand: union of base and support volumes, then feature cutouts
- Container: shell(base_volume, wall_thickness)

Constraints:
- Output only valid SDF DSL.
- Always include a `params` block.
- Include short comments explaining key design decisions.
- Favor watertight, printable solids with positive volume.
"#;

pub const CANONICAL_PROMPTS: [&str; 20] = [
    "a sphere",
    "a 50mm cube",
    "a cylinder 30mm diameter, 80mm tall",
    "a phone stand",
    "a cable organizer with 4 slots",
    "a rounded rectangular enclosure for a Raspberry Pi",
    "a wall-mount hook",
    "a vase with a twisted pattern",
    "a pen holder with honeycomb pattern",
    "a soap dish with drainage holes",
    "a torus ring",
    "a capsule-style handle",
    "a tapered nozzle",
    "a rounded cylindrical foot",
    "a smooth blend between a sphere and a box",
    "an overlapping sphere-box intersection",
    "a sphere with a central hole",
    "a mirrored decorative fin",
    "a twisted column with a slight bend",
    "a scaled and rotated puck",
];

pub fn canonical_prompts() -> &'static [&'static str] {
    &CANONICAL_PROMPTS
}

pub fn default_system_prompt() -> &'static str {
    DEFAULT_SYSTEM_PROMPT
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_retries: usize,
    pub mesh_resolution: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            mesh_resolution: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationRequest<'a> {
    pub system_prompt: &'a str,
    pub user_prompt: &'a str,
    pub current_dsl: Option<&'a str>,
    pub validation_errors: &'a [String],
    pub attempt: usize,
}

pub trait LanguageModel {
    fn generate_dsl(&mut self, request: GenerationRequest<'_>) -> Result<String, String>;
}

#[derive(Debug, Clone)]
pub struct GenerationSuccess {
    pub dsl: String,
    pub attempts: usize,
    pub validation: ValidationResult,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationError {
    Model(String),
    ExhaustedRetries {
        attempts: usize,
        last_errors: Vec<String>,
    },
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenerationError::Model(message) => {
                write!(f, "model request failed: {message}")
            }
            GenerationError::ExhaustedRetries {
                attempts,
                last_errors,
            } => {
                write!(
                    f,
                    "failed to produce a valid DSL program after {attempts} attempt(s): {}",
                    last_errors.join("; ")
                )
            }
        }
    }
}

impl std::error::Error for GenerationError {}

pub struct DslGenerator<C: LanguageModel> {
    client: C,
    config: GenerationConfig,
    system_prompt: String,
}

impl<C: LanguageModel> DslGenerator<C> {
    pub fn new(client: C, config: GenerationConfig) -> Self {
        Self {
            client,
            config,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    pub fn into_client(self) -> C {
        self.client
    }

    pub fn generate(&mut self, user_prompt: &str) -> Result<GenerationSuccess, GenerationError> {
        self.generate_internal(user_prompt, None)
    }

    pub fn generate_modification(
        &mut self,
        current_dsl: &str,
        user_feedback: &str,
    ) -> Result<GenerationSuccess, GenerationError> {
        self.generate_internal(user_feedback, Some(current_dsl))
    }

    pub fn mesh_resolution(&self) -> usize {
        self.config.mesh_resolution
    }

    fn generate_internal(
        &mut self,
        user_prompt: &str,
        current_dsl: Option<&str>,
    ) -> Result<GenerationSuccess, GenerationError> {
        let max_attempts = self.config.max_retries.max(1);
        let mut validation_errors = Vec::new();

        for attempt in 1..=max_attempts {
            let candidate = self
                .client
                .generate_dsl(GenerationRequest {
                    system_prompt: &self.system_prompt,
                    user_prompt,
                    current_dsl,
                    validation_errors: &validation_errors,
                    attempt,
                })
                .map_err(GenerationError::Model)?;

            let validation = validate_dsl_output(&candidate, self.config.mesh_resolution);
            if validation.is_valid() {
                return Ok(GenerationSuccess {
                    dsl: candidate,
                    attempts: attempt,
                    validation,
                });
            }

            validation_errors = validation.errors.clone();
        }

        Err(GenerationError::ExhaustedRetries {
            attempts: max_attempts,
            last_errors: validation_errors,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelAnalysis {
    pub volume: f64,
    pub bbox_min: [f64; 3],
    pub bbox_max: [f64; 3],
    pub watertight: bool,
}

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user_input: String,
    pub dsl: String,
    pub attempts: usize,
    pub elapsed_ms: f64,
    pub analysis: ModelAnalysis,
}

#[derive(Debug)]
pub enum ConversationError {
    NoActiveModel,
    Generation(GenerationError),
    Analysis(String),
}

impl fmt::Display for ConversationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversationError::NoActiveModel => f.write_str("no active model in conversation"),
            ConversationError::Generation(err) => write!(f, "{err}"),
            ConversationError::Analysis(message) => write!(f, "analysis failed: {message}"),
        }
    }
}

impl std::error::Error for ConversationError {}

impl From<GenerationError> for ConversationError {
    fn from(value: GenerationError) -> Self {
        Self::Generation(value)
    }
}

pub struct ConversationEngine<C: LanguageModel> {
    generator: DslGenerator<C>,
    history: Vec<String>,
}

impl<C: LanguageModel> ConversationEngine<C> {
    pub fn new(generator: DslGenerator<C>) -> Self {
        Self {
            generator,
            history: Vec::new(),
        }
    }

    pub fn start(&mut self, prompt: &str) -> Result<ConversationTurn, ConversationError> {
        self.run_turn(prompt, None)
    }

    pub fn modify(&mut self, feedback: &str) -> Result<ConversationTurn, ConversationError> {
        let current = self
            .history
            .last()
            .cloned()
            .ok_or(ConversationError::NoActiveModel)?;
        self.run_turn(feedback, Some(current.as_str()))
    }

    pub fn current_dsl(&self) -> Option<&str> {
        self.history.last().map(String::as_str)
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn into_generator(self) -> DslGenerator<C> {
        self.generator
    }

    fn run_turn(
        &mut self,
        user_input: &str,
        current_dsl: Option<&str>,
    ) -> Result<ConversationTurn, ConversationError> {
        let start = Instant::now();
        let generated = match current_dsl {
            Some(base) => self.generator.generate_modification(base, user_input)?,
            None => self.generator.generate(user_input)?,
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let analysis = analyze_dsl(&generated.dsl, self.generator.mesh_resolution())
            .map_err(ConversationError::Analysis)?;
        self.history.push(generated.dsl.clone());

        Ok(ConversationTurn {
            user_input: user_input.to_string(),
            dsl: generated.dsl,
            attempts: generated.attempts,
            elapsed_ms,
            analysis,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Axis::X => "width",
            Axis::Y => "depth",
            Axis::Z => "height",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DimensionalConstraint {
    AxisExtent {
        axis: Axis,
        target_mm: f64,
        tolerance_mm: f64,
    },
    WallThickness {
        target_mm: f64,
        tolerance_mm: f64,
    },
    TiltFromVertical {
        target_deg: f64,
        tolerance_deg: f64,
    },
}

#[derive(Debug, Clone)]
pub struct ConstraintCheck {
    pub constraint: DimensionalConstraint,
    pub passed: bool,
    pub measured: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ConstraintReport {
    pub analysis: ModelAnalysis,
    pub checks: Vec<ConstraintCheck>,
    pub all_satisfied: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ConstraintAdjustmentConfig {
    pub max_adjustment_rounds: usize,
}

impl Default for ConstraintAdjustmentConfig {
    fn default() -> Self {
        Self {
            max_adjustment_rounds: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintConverged {
    pub dsl: String,
    pub generation_attempts: usize,
    pub adjustment_rounds: usize,
    pub report: ConstraintReport,
}

#[derive(Debug, Clone)]
pub enum ConstraintEnforcementError {
    Generation(GenerationError),
    Analysis(String),
    Unsatisfied {
        attempts: usize,
        last_report: ConstraintReport,
    },
}

impl fmt::Display for ConstraintEnforcementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintEnforcementError::Generation(err) => write!(f, "{err}"),
            ConstraintEnforcementError::Analysis(message) => {
                write!(f, "analysis failed: {message}")
            }
            ConstraintEnforcementError::Unsatisfied { attempts, .. } => write!(
                f,
                "constraints were not satisfied after {attempts} adjustment attempt(s)"
            ),
        }
    }
}

impl std::error::Error for ConstraintEnforcementError {}

impl From<GenerationError> for ConstraintEnforcementError {
    fn from(value: GenerationError) -> Self {
        Self::Generation(value)
    }
}

pub fn extract_dimensional_constraints(prompt: &str) -> Vec<DimensionalConstraint> {
    let mut constraints = Vec::new();
    let tokens = tokenize_prompt(prompt);

    if let Some(cube_idx) = tokens.iter().position(|token| token == "cube")
        && let Some(size) = parse_unit_before(&tokens, cube_idx, "mm")
    {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            constraints.push(DimensionalConstraint::AxisExtent {
                axis,
                target_mm: size,
                tolerance_mm: 1.0,
            });
        }
    }

    let cylinder_idx = tokens.iter().position(|token| token == "cylinder");
    let diameter_idx = tokens.iter().position(|token| token == "diameter");
    let tall_idx = tokens.iter().position(|token| token == "tall");
    if let (Some(cyl_i), Some(diam_i), Some(tall_i)) = (cylinder_idx, diameter_idx, tall_idx)
        && cyl_i < diam_i
        && diam_i < tall_i
        && let (Some(diameter), Some(height)) = (
            parse_unit_before(&tokens, diam_i, "mm"),
            parse_unit_before(&tokens, tall_i, "mm"),
        )
    {
        constraints.push(DimensionalConstraint::AxisExtent {
            axis: Axis::X,
            target_mm: diameter,
            tolerance_mm: 1.0,
        });
        constraints.push(DimensionalConstraint::AxisExtent {
            axis: Axis::Y,
            target_mm: diameter,
            tolerance_mm: 1.0,
        });
        constraints.push(DimensionalConstraint::AxisExtent {
            axis: Axis::Z,
            target_mm: height,
            tolerance_mm: 1.0,
        });
    }

    if let Some(wide_idx) = tokens.iter().position(|token| token == "wide")
        && let Some(width) = parse_unit_before(&tokens, wide_idx, "mm")
    {
        let exact = tokens[..wide_idx].iter().any(|token| token == "exactly");
        constraints.push(DimensionalConstraint::AxisExtent {
            axis: Axis::X,
            target_mm: width,
            tolerance_mm: if exact { 0.5 } else { 1.0 },
        });
    }

    if let Some(thickness_idx) = tokens.iter().position(|token| token == "thickness") {
        let has_wall = thickness_idx > 0 && tokens[thickness_idx - 1] == "wall";
        if has_wall && let Some(thickness) = parse_unit_after(&tokens, thickness_idx, "mm") {
            constraints.push(DimensionalConstraint::WallThickness {
                target_mm: thickness,
                tolerance_mm: 0.75,
            });
        }
    }

    if let Some(tilt_idx) = tokens.iter().position(|token| token == "tilt")
        && let Some(tilt_deg) = parse_degrees_before(&tokens, tilt_idx)
    {
        constraints.push(DimensionalConstraint::TiltFromVertical {
            target_deg: tilt_deg,
            tolerance_deg: 1.5,
        });
    }

    constraints
}

fn tokenize_prompt(prompt: &str) -> Vec<String> {
    let chars = prompt.chars().collect::<Vec<_>>();
    let mut normalized = String::with_capacity(chars.len());
    for (index, ch) in chars.iter().enumerate() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            continue;
        }
        if *ch == '.' {
            let prev_is_digit = index > 0 && chars[index - 1].is_ascii_digit();
            let next_is_digit = index + 1 < chars.len() && chars[index + 1].is_ascii_digit();
            if prev_is_digit && next_is_digit {
                normalized.push('.');
                continue;
            }
        }
        normalized.push(' ');
    }
    normalized
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>()
}

fn parse_unit_before(tokens: &[String], keyword_idx: usize, unit: &str) -> Option<f64> {
    if keyword_idx == 0 {
        return None;
    }
    parse_unit_token(tokens, keyword_idx - 1, unit).or_else(|| {
        if keyword_idx >= 2 && tokens[keyword_idx - 1] == unit {
            parse_number_token(&tokens[keyword_idx - 2])
        } else {
            None
        }
    })
}

fn parse_unit_after(tokens: &[String], keyword_idx: usize, unit: &str) -> Option<f64> {
    if keyword_idx + 1 >= tokens.len() {
        return None;
    }
    parse_unit_token(tokens, keyword_idx + 1, unit).or_else(|| {
        if keyword_idx + 2 < tokens.len() && tokens[keyword_idx + 2] == unit {
            parse_number_token(&tokens[keyword_idx + 1])
        } else {
            None
        }
    })
}

fn parse_unit_token(tokens: &[String], idx: usize, unit: &str) -> Option<f64> {
    let token = &tokens[idx];
    if token.ends_with(unit) && token.len() > unit.len() {
        parse_number_token(&token[..token.len() - unit.len()])
    } else {
        None
    }
}

fn parse_degrees_before(tokens: &[String], keyword_idx: usize) -> Option<f64> {
    if keyword_idx == 0 {
        return None;
    }

    let prev = &tokens[keyword_idx - 1];
    if prev.ends_with("deg") && prev.len() > 3 {
        return parse_number_token(&prev[..prev.len() - 3]);
    }
    if prev.ends_with("degree") && prev.len() > 6 {
        return parse_number_token(&prev[..prev.len() - 6]);
    }
    if (prev == "deg" || prev == "degree" || prev == "degrees") && keyword_idx >= 2 {
        return parse_number_token(&tokens[keyword_idx - 2]);
    }
    parse_number_token(prev)
}

fn parse_number_token(token: &str) -> Option<f64> {
    token.parse::<f64>().ok()
}

pub fn verify_constraints_for_dsl(
    dsl: &str,
    resolution: usize,
    constraints: &[DimensionalConstraint],
) -> Result<ConstraintReport, String> {
    let scene = compile_dsl(dsl).map_err(|err| format!("DSL parse/compile failed: {err}"))?;
    let mesh = scene
        .evaluate_mesh(resolution.max(2))
        .map_err(|err| format!("mesh evaluation failed: {err}"))?;
    if mesh.vertices.is_empty() || mesh.triangles.is_empty() {
        return Err("mesh must contain vertices and triangles".to_string());
    }

    let (bbox_min, bbox_max) = bounding_box(&mesh);
    let analysis = ModelAnalysis {
        volume: mesh_volume(&mesh).abs(),
        bbox_min,
        bbox_max,
        watertight: is_watertight(&mesh),
    };

    let checks = constraints
        .iter()
        .map(|constraint| check_constraint(constraint, &analysis, &mesh))
        .collect::<Vec<_>>();
    let all_satisfied = checks.iter().all(|check| check.passed);

    Ok(ConstraintReport {
        analysis,
        checks,
        all_satisfied,
    })
}

pub fn enforce_constraints_with_retries<C: LanguageModel>(
    generator: &mut DslGenerator<C>,
    user_prompt: &str,
    constraints: &[DimensionalConstraint],
    config: ConstraintAdjustmentConfig,
) -> Result<ConstraintConverged, ConstraintEnforcementError> {
    let mut generation_attempts = 0usize;
    let mut rounds = 0usize;

    let mut generated = generator.generate(user_prompt)?;
    generation_attempts += generated.attempts;

    loop {
        let report =
            verify_constraints_for_dsl(&generated.dsl, generator.mesh_resolution(), constraints)
                .map_err(ConstraintEnforcementError::Analysis)?;
        if report.all_satisfied {
            return Ok(ConstraintConverged {
                dsl: generated.dsl,
                generation_attempts,
                adjustment_rounds: rounds,
                report,
            });
        }

        if rounds >= config.max_adjustment_rounds {
            return Err(ConstraintEnforcementError::Unsatisfied {
                attempts: rounds + 1,
                last_report: report,
            });
        }

        let failed_constraints = report
            .checks
            .iter()
            .filter(|check| !check.passed)
            .map(|check| check.constraint.clone())
            .collect::<Vec<_>>();
        let feedback = build_adjustment_feedback(&failed_constraints);
        generated = generator.generate_modification(&generated.dsl, &feedback)?;
        generation_attempts += generated.attempts;
        rounds += 1;
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub parse_valid: bool,
    pub mesh_valid: bool,
    pub watertight: bool,
    pub positive_volume: bool,
    pub bounding_box_within_limits: bool,
    pub no_degenerate_triangles: bool,
    pub errors: Vec<String>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.parse_valid
            && self.mesh_valid
            && self.watertight
            && self.positive_volume
            && self.bounding_box_within_limits
            && self.no_degenerate_triangles
            && self.errors.is_empty()
    }
}

pub fn validate_dsl_output(dsl: &str, resolution: usize) -> ValidationResult {
    let mut errors = Vec::new();

    if !contains_params_block(dsl) {
        errors.push("missing required params block".to_string());
    }
    if !contains_comment(dsl) {
        errors.push("output should include a brief design comment".to_string());
    }

    let scene = match compile_dsl(dsl) {
        Ok(scene) => scene,
        Err(err) => {
            errors.push(format!("DSL parse/compile failed: {err}"));
            return ValidationResult {
                parse_valid: false,
                mesh_valid: false,
                watertight: false,
                positive_volume: false,
                bounding_box_within_limits: false,
                no_degenerate_triangles: false,
                errors,
            };
        }
    };

    let mesh = match scene.evaluate_mesh(resolution.max(2)) {
        Ok(mesh) => mesh,
        Err(err) => {
            errors.push(format!("mesh evaluation failed: {err}"));
            return ValidationResult {
                parse_valid: true,
                mesh_valid: false,
                watertight: false,
                positive_volume: false,
                bounding_box_within_limits: false,
                no_degenerate_triangles: false,
                errors,
            };
        }
    };

    if mesh.triangles.is_empty() || mesh.vertices.is_empty() {
        errors.push("mesh must contain vertices and triangles".to_string());
    }

    let watertight = is_watertight(&mesh);
    if !watertight {
        errors.push("mesh is not watertight".to_string());
    }

    let positive_volume = mesh_volume(&mesh).abs() > 1e-6;
    if !positive_volume {
        errors.push("mesh volume must be positive".to_string());
    }

    let bounding_box_within_limits = bounding_box_is_reasonable(&mesh);
    if !bounding_box_within_limits {
        errors.push("mesh bounding box must be within 1mm..500mm on each axis".to_string());
    }

    let no_degenerate_triangles = has_no_degenerate_triangles(&mesh);
    if !no_degenerate_triangles {
        errors.push("mesh contains degenerate triangles".to_string());
    }

    let mesh_valid =
        watertight && positive_volume && bounding_box_within_limits && no_degenerate_triangles;

    ValidationResult {
        parse_valid: true,
        mesh_valid,
        watertight,
        positive_volume,
        bounding_box_within_limits,
        no_degenerate_triangles,
        errors,
    }
}

#[derive(Debug, Clone)]
pub struct PromptEvaluation {
    pub prompt: String,
    pub success: bool,
    pub attempts: usize,
    pub parse_valid: bool,
    pub mesh_valid: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PromptSuiteReport {
    pub total: usize,
    pub parse_success_rate: f64,
    pub mesh_valid_rate: f64,
    pub retry_rate: f64,
    pub results: Vec<PromptEvaluation>,
}

impl PromptSuiteReport {
    pub fn meets_thresholds(&self, parse_min: f64, mesh_min: f64) -> bool {
        self.parse_success_rate >= parse_min && self.mesh_valid_rate >= mesh_min
    }
}

pub fn evaluate_prompt_suite<C: LanguageModel>(
    generator: &mut DslGenerator<C>,
    prompts: &[&str],
) -> PromptSuiteReport {
    let mut parse_ok = 0usize;
    let mut mesh_ok = 0usize;
    let mut retries = 0usize;
    let mut results = Vec::with_capacity(prompts.len());

    for prompt in prompts {
        match generator.generate(prompt) {
            Ok(success) => {
                if success.validation.parse_valid {
                    parse_ok += 1;
                }
                if success.validation.mesh_valid {
                    mesh_ok += 1;
                }
                if success.attempts > 1 {
                    retries += 1;
                }
                results.push(PromptEvaluation {
                    prompt: (*prompt).to_string(),
                    success: true,
                    attempts: success.attempts,
                    parse_valid: success.validation.parse_valid,
                    mesh_valid: success.validation.mesh_valid,
                    errors: success.validation.errors,
                });
            }
            Err(GenerationError::ExhaustedRetries {
                attempts,
                last_errors,
            }) => {
                if attempts > 1 {
                    retries += 1;
                }
                results.push(PromptEvaluation {
                    prompt: (*prompt).to_string(),
                    success: false,
                    attempts,
                    parse_valid: false,
                    mesh_valid: false,
                    errors: last_errors,
                });
            }
            Err(GenerationError::Model(message)) => {
                results.push(PromptEvaluation {
                    prompt: (*prompt).to_string(),
                    success: false,
                    attempts: 1,
                    parse_valid: false,
                    mesh_valid: false,
                    errors: vec![format!("model request failed: {message}")],
                });
            }
        }
    }

    let total = prompts.len();
    let denom = total.max(1) as f64;
    PromptSuiteReport {
        total,
        parse_success_rate: parse_ok as f64 / denom,
        mesh_valid_rate: mesh_ok as f64 / denom,
        retry_rate: retries as f64 / denom,
        results,
    }
}

pub fn analyze_dsl(dsl: &str, resolution: usize) -> Result<ModelAnalysis, String> {
    let scene = compile_dsl(dsl).map_err(|err| format!("DSL parse/compile failed: {err}"))?;
    let mesh = scene
        .evaluate_mesh(resolution.max(2))
        .map_err(|err| format!("mesh evaluation failed: {err}"))?;
    if mesh.vertices.is_empty() || mesh.triangles.is_empty() {
        return Err("mesh must contain vertices and triangles".to_string());
    }

    let (bbox_min, bbox_max) = bounding_box(&mesh);
    Ok(ModelAnalysis {
        volume: mesh_volume(&mesh).abs(),
        bbox_min,
        bbox_max,
        watertight: is_watertight(&mesh),
    })
}

fn check_constraint(
    constraint: &DimensionalConstraint,
    analysis: &ModelAnalysis,
    mesh: &Mesh,
) -> ConstraintCheck {
    match constraint {
        DimensionalConstraint::AxisExtent {
            axis,
            target_mm,
            tolerance_mm,
        } => {
            let measured = analysis.bbox_max[axis.index()] - analysis.bbox_min[axis.index()];
            let delta = (measured - target_mm).abs();
            ConstraintCheck {
                constraint: constraint.clone(),
                passed: delta <= *tolerance_mm,
                measured: Some(measured),
                detail: format!(
                    "{} measured={measured:.3}mm target={target_mm:.3}mm ±{tolerance_mm:.3}mm",
                    axis.label()
                ),
            }
        }
        DimensionalConstraint::WallThickness {
            target_mm,
            tolerance_mm,
        } => {
            let measured = estimate_wall_thickness(mesh);
            match measured {
                Some(value) => ConstraintCheck {
                    constraint: constraint.clone(),
                    passed: (value - target_mm).abs() <= *tolerance_mm,
                    measured: Some(value),
                    detail: format!(
                        "wall_thickness measured={value:.3}mm target={target_mm:.3}mm ±{tolerance_mm:.3}mm"
                    ),
                },
                None => ConstraintCheck {
                    constraint: constraint.clone(),
                    passed: false,
                    measured: None,
                    detail: "wall_thickness could not be estimated from mesh ray intersections"
                        .to_string(),
                },
            }
        }
        DimensionalConstraint::TiltFromVertical {
            target_deg,
            tolerance_deg,
        } => {
            let measured = estimate_tilt_from_vertical(mesh);
            ConstraintCheck {
                constraint: constraint.clone(),
                passed: (measured - target_deg).abs() <= *tolerance_deg,
                measured: Some(measured),
                detail: format!(
                    "tilt measured={measured:.3}deg target={target_deg:.3}deg ±{tolerance_deg:.3}deg"
                ),
            }
        }
    }
}

fn build_adjustment_feedback(constraints: &[DimensionalConstraint]) -> String {
    let mut lines =
        vec!["Adjust the current DSL to satisfy these dimensional constraints:".to_string()];
    for constraint in constraints {
        match constraint {
            DimensionalConstraint::AxisExtent {
                axis,
                target_mm,
                tolerance_mm,
            } => lines.push(format!(
                "- set {} to {target_mm:.3}mm ±{tolerance_mm:.3}mm",
                axis.label()
            )),
            DimensionalConstraint::WallThickness {
                target_mm,
                tolerance_mm,
            } => lines.push(format!(
                "- set wall thickness to {target_mm:.3}mm ±{tolerance_mm:.3}mm"
            )),
            DimensionalConstraint::TiltFromVertical {
                target_deg,
                tolerance_deg,
            } => lines.push(format!(
                "- set tilt from vertical to {target_deg:.3}deg ±{tolerance_deg:.3}deg"
            )),
        }
    }
    lines.join("\n")
}

fn estimate_wall_thickness(mesh: &Mesh) -> Option<f64> {
    let (min, max) = bounding_box(mesh);
    let center_y = (min[1] + max[1]) * 0.5;
    let center_z = (min[2] + max[2]) * 0.5;
    let origin = [min[0] - 1.0, center_y, center_z];
    let direction = [1.0, 0.0, 0.0];
    let mut intersections = ray_triangle_intersections(mesh, origin, direction);
    if intersections.len() < 2 {
        return None;
    }

    intersections.sort_by(|a, b| a.total_cmp(b));
    intersections.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    intersections
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|delta| *delta > 1e-5)
        .min_by(|a, b| a.total_cmp(b))
}

fn ray_triangle_intersections(mesh: &Mesh, origin: [f64; 3], direction: [f64; 3]) -> Vec<f64> {
    mesh.triangles
        .iter()
        .filter_map(|triangle| {
            let a = mesh.vertices[triangle[0] as usize];
            let b = mesh.vertices[triangle[1] as usize];
            let c = mesh.vertices[triangle[2] as usize];
            ray_intersects_triangle(origin, direction, a, b, c)
        })
        .collect()
}

fn ray_intersects_triangle(
    origin: [f64; 3],
    direction: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> Option<f64> {
    let edge1 = sub(v1, v0);
    let edge2 = sub(v2, v0);
    let pvec = cross(direction, edge2);
    let det = dot(edge1, pvec);
    if det.abs() <= 1e-9 {
        return None;
    }
    let inv_det = 1.0 / det;

    let tvec = sub(origin, v0);
    let u = dot(tvec, pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let qvec = cross(tvec, edge1);
    let v = dot(direction, qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = dot(edge2, qvec) * inv_det;
    if t > 1e-6 { Some(t) } else { None }
}

fn estimate_tilt_from_vertical(mesh: &Mesh) -> f64 {
    if mesh.vertices.is_empty() {
        return 0.0;
    }

    let mut centroid = [0.0; 3];
    for vertex in &mesh.vertices {
        centroid[0] += vertex[0];
        centroid[1] += vertex[1];
        centroid[2] += vertex[2];
    }
    let inv_count = 1.0 / mesh.vertices.len() as f64;
    centroid[0] *= inv_count;
    centroid[1] *= inv_count;
    centroid[2] *= inv_count;

    let mut cov = [[0.0; 3]; 3];
    for vertex in &mesh.vertices {
        let d = sub(*vertex, centroid);
        cov[0][0] += d[0] * d[0];
        cov[0][1] += d[0] * d[1];
        cov[0][2] += d[0] * d[2];
        cov[1][0] += d[1] * d[0];
        cov[1][1] += d[1] * d[1];
        cov[1][2] += d[1] * d[2];
        cov[2][0] += d[2] * d[0];
        cov[2][1] += d[2] * d[1];
        cov[2][2] += d[2] * d[2];
    }

    let mut axis = [0.0, 0.0, 1.0];
    for _ in 0..32 {
        let next = [
            cov[0][0] * axis[0] + cov[0][1] * axis[1] + cov[0][2] * axis[2],
            cov[1][0] * axis[0] + cov[1][1] * axis[1] + cov[1][2] * axis[2],
            cov[2][0] * axis[0] + cov[2][1] * axis[1] + cov[2][2] * axis[2],
        ];
        let len = length(next);
        if len <= 1e-12 {
            break;
        }
        axis = [next[0] / len, next[1] / len, next[2] / len];
    }

    let cos_theta = axis[2].abs().clamp(0.0, 1.0);
    cos_theta.acos().to_degrees()
}

fn contains_params_block(dsl: &str) -> bool {
    dsl.lines()
        .any(|line| line.trim_start().starts_with("params"))
}

fn contains_comment(dsl: &str) -> bool {
    dsl.contains("//") || dsl.contains("/*")
}

fn has_no_degenerate_triangles(mesh: &Mesh) -> bool {
    mesh.triangles.iter().all(|triangle| {
        let a = mesh.vertices[triangle[0] as usize];
        let b = mesh.vertices[triangle[1] as usize];
        let c = mesh.vertices[triangle[2] as usize];
        triangle_area(a, b, c) > 1e-10
    })
}

fn bounding_box_is_reasonable(mesh: &Mesh) -> bool {
    if mesh.vertices.is_empty() {
        return false;
    }

    let (min, max) = bounding_box(mesh);

    (0..3).all(|axis| {
        let extent = max[axis] - min[axis];
        (1.0..=500.0).contains(&extent)
    })
}

fn bounding_box(mesh: &Mesh) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (min, max)
}

fn is_watertight(mesh: &Mesh) -> bool {
    let mut edge_counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
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

    edge_counts.values().all(|count| *count == 2)
}

fn mesh_volume(mesh: &Mesh) -> f64 {
    mesh.triangles
        .iter()
        .map(|triangle| {
            let a = mesh.vertices[triangle[0] as usize];
            let b = mesh.vertices[triangle[1] as usize];
            let c = mesh.vertices[triangle[2] as usize];
            dot(a, cross(b, c)) / 6.0
        })
        .sum()
}

fn ordered_edge(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    0.5 * length(cross(sub(b, a), sub(c, a)))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn length(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};

    use super::{
        Axis, CANONICAL_PROMPTS, ConstraintAdjustmentConfig, ConversationEngine, DslGenerator,
        GenerationConfig, GenerationRequest, LanguageModel, canonical_prompts,
        default_system_prompt, enforce_constraints_with_retries, evaluate_prompt_suite,
        extract_dimensional_constraints, verify_constraints_for_dsl,
    };

    #[derive(Debug, Clone)]
    struct RequestLog {
        prompt: String,
        attempt: usize,
        current_dsl: Option<String>,
        validation_errors: Vec<String>,
    }

    #[derive(Default)]
    struct ScriptedModel {
        responses: BTreeMap<String, VecDeque<String>>,
        logs: Vec<RequestLog>,
    }

    impl ScriptedModel {
        fn with_response(mut self, prompt: &str, responses: Vec<&str>) -> Self {
            let queue = responses.into_iter().map(str::to_string).collect();
            self.responses.insert(prompt.to_string(), queue);
            self
        }
    }

    impl LanguageModel for ScriptedModel {
        fn generate_dsl(&mut self, request: GenerationRequest<'_>) -> Result<String, String> {
            self.logs.push(RequestLog {
                prompt: request.user_prompt.to_string(),
                attempt: request.attempt,
                current_dsl: request.current_dsl.map(str::to_string),
                validation_errors: request.validation_errors.to_vec(),
            });

            let queue = self.responses.get_mut(request.user_prompt).ok_or_else(|| {
                format!("no scripted response for prompt '{}'", request.user_prompt)
            })?;
            queue.pop_front().ok_or_else(|| {
                format!(
                    "no remaining scripted responses for '{}'",
                    request.user_prompt
                )
            })
        }
    }

    #[test]
    fn system_prompt_includes_required_components() {
        let prompt = default_system_prompt();
        assert!(prompt.contains("DSL grammar reference"));
        assert!(prompt.contains("Primitive catalog"));
        assert!(prompt.contains("Operation catalog"));
        assert!(prompt.contains("Coordinate conventions"));
        assert!(prompt.contains("Common design patterns"));
        assert!(prompt.contains("Constraints"));
    }

    #[test]
    fn canonical_prompt_suite_parses_and_meshes() {
        let model = canonical_scripted_model();
        let mut generator = DslGenerator::new(
            model,
            GenerationConfig {
                max_retries: 3,
                mesh_resolution: 24,
            },
        );

        let report = evaluate_prompt_suite(&mut generator, canonical_prompts());

        assert_eq!(report.total, CANONICAL_PROMPTS.len());
        assert!(
            (report.parse_success_rate - 1.0).abs() < 1e-12,
            "parse success rate mismatch: {:?}",
            report.results
        );
        assert!(
            (report.mesh_valid_rate - 1.0).abs() < 1e-12,
            "mesh valid rate mismatch: {:?}",
            report.results
        );
        assert!(
            (report.retry_rate - 0.0).abs() < 1e-12,
            "retry rate mismatch: {:?}",
            report.results
        );
        assert!(report.meets_thresholds(0.95, 0.90));

        for result in &report.results {
            assert!(
                result.success,
                "prompt '{}' failed: {:?}",
                result.prompt, result.errors
            );
            assert!(
                result.parse_valid,
                "prompt '{}' did not parse",
                result.prompt
            );
            assert!(
                result.mesh_valid,
                "prompt '{}' mesh invalid: {:?}",
                result.prompt, result.errors
            );
        }
    }

    #[test]
    fn integration_bookend_prompt_produces_valid_mesh() {
        let prompt = "make a bookend shaped like the letter L";
        let dsl = r#"// L-shaped bookend with thick orthogonal legs
params {
  upright_h = 90mm
  upright_t = 12mm
  base_w = 70mm
  base_t = 12mm
  depth = 45mm
}
upright = box(upright_t, depth, upright_h) |> translate((base_w - upright_t) / 2, 0, (upright_h - base_t) / 2)
base = box(base_w, depth, base_t)
result = union(base, upright)
"#;

        let model = ScriptedModel::default().with_response(prompt, vec![dsl]);
        let mut generator = DslGenerator::new(
            model,
            GenerationConfig {
                max_retries: 3,
                mesh_resolution: 28,
            },
        );

        let generated = generator
            .generate(prompt)
            .expect("bookend prompt should generate valid DSL");

        assert!(generated.validation.parse_valid);
        assert!(generated.validation.mesh_valid);
        assert!(generated.validation.watertight);
        assert!(generated.validation.positive_volume);
        assert!(generated.validation.bounding_box_within_limits);
        assert!(generated.validation.no_degenerate_triangles);
    }

    #[test]
    fn error_recovery_retries_with_validation_feedback() {
        let tricky_prompts = [
            "tricky prompt 1",
            "tricky prompt 2",
            "tricky prompt 3",
            "tricky prompt 4",
            "tricky prompt 5",
            "tricky prompt 6",
            "tricky prompt 7",
            "tricky prompt 8",
            "tricky prompt 9",
            "tricky prompt 10",
        ];

        let mut model = ScriptedModel::default();
        for (index, prompt) in tricky_prompts.iter().enumerate() {
            model = model.with_response(prompt, vec!["sphere(", retry_success_program(index)]);
        }

        let mut generator = DslGenerator::new(
            model,
            GenerationConfig {
                max_retries: 3,
                mesh_resolution: 20,
            },
        );

        for prompt in &tricky_prompts {
            let success = generator
                .generate(prompt)
                .expect("retry path should eventually produce valid DSL");
            assert_eq!(
                success.attempts, 2,
                "prompt '{prompt}' should succeed on second attempt"
            );
            assert!(success.validation.parse_valid);
            assert!(success.validation.mesh_valid);
        }

        let model = generator.into_client();
        for prompt in &tricky_prompts {
            let first = model
                .logs
                .iter()
                .find(|log| log.prompt == *prompt && log.attempt == 1)
                .expect("first attempt log should exist");
            assert!(first.validation_errors.is_empty());

            let second = model
                .logs
                .iter()
                .find(|log| log.prompt == *prompt && log.attempt == 2)
                .expect("second attempt log should exist");
            assert!(
                !second.validation_errors.is_empty(),
                "second attempt for '{prompt}' should include validation errors"
            );
        }
    }

    #[test]
    fn conversational_scripts_cover_required_behaviors() {
        // Script 1: dimensional changes.
        let mut script1 = make_engine(
            &[
                ("a 50mm cube", script1_turn1()),
                ("make it 80mm wide", script1_turn2()),
                ("and 30mm tall", script1_turn3()),
            ],
            26,
        );
        let s1_t1 = script1
            .start("a 50mm cube")
            .expect("script1 turn1 should work");
        let s1_t2 = script1
            .modify("make it 80mm wide")
            .expect("script1 turn2 should work");
        let s1_t3 = script1
            .modify("and 30mm tall")
            .expect("script1 turn3 should work");
        assert!(s1_t1.analysis.watertight);
        assert!(s1_t2.analysis.watertight);
        assert!(s1_t3.analysis.watertight);
        assert_approx(axis_extent(&s1_t2, 0), 80.0, 6.0);
        assert_approx(axis_extent(&s1_t3, 2), 30.0, 4.0);
        assert!(s1_t2.elapsed_ms < 3000.0);
        assert!(s1_t3.elapsed_ms < 3000.0);

        let script1_generator = script1.into_generator();
        let script1_model = script1_generator.into_client();
        let modify_logs: Vec<_> = script1_model
            .logs
            .iter()
            .filter(|log| log.prompt != "a 50mm cube")
            .collect();
        assert!(
            modify_logs
                .iter()
                .all(|log| log.current_dsl.as_deref().is_some()),
            "modification prompts must include current DSL context"
        );

        // Script 2: additive and subtractive operations.
        let mut script2 = make_engine(
            &[
                ("a cylinder 40mm diameter 60mm tall", script2_turn1()),
                ("hollow it out with 3mm walls", script2_turn2()),
                ("add a base plate", script2_turn3()),
            ],
            26,
        );
        let s2_t1 = script2
            .start("a cylinder 40mm diameter 60mm tall")
            .expect("script2 turn1 should work");
        let s2_t2 = script2
            .modify("hollow it out with 3mm walls")
            .expect("script2 turn2 should work");
        let s2_t3 = script2
            .modify("add a base plate")
            .expect("script2 turn3 should work");
        assert!(s2_t1.analysis.watertight);
        assert!(s2_t2.analysis.watertight);
        assert!(s2_t3.analysis.watertight);
        assert!(s2_t2.analysis.volume < s2_t1.analysis.volume);
        assert!(s2_t3.analysis.volume > s2_t2.analysis.volume);
        assert!(s2_t2.elapsed_ms < 3000.0);
        assert!(s2_t3.elapsed_ms < 3000.0);

        // Script 3: rounded edges then subtractive hole.
        let mut script3 = make_engine(
            &[
                ("a solid block 100x50x30mm", script3_turn1()),
                ("round all the edges", script3_turn2()),
                ("cut a 20mm hole through the center", script3_turn3()),
            ],
            24,
        );
        let s3_t1 = script3
            .start("a solid block 100x50x30mm")
            .expect("script3 turn1 should work");
        let s3_t2 = script3
            .modify("round all the edges")
            .expect("script3 turn2 should work");
        let s3_t3 = script3
            .modify("cut a 20mm hole through the center")
            .expect("script3 turn3 should work");
        assert!(s3_t1.analysis.watertight);
        assert!(s3_t2.analysis.watertight);
        assert!(s3_t3.analysis.watertight);
        assert!(s3_t3.analysis.volume < s3_t2.analysis.volume);
        assert!(s3_t2.elapsed_ms < 3000.0);
        assert!(s3_t3.elapsed_ms < 3000.0);

        // Script 4: style changes with twist.
        let mut script4 = make_engine(
            &[
                ("a simple vase", script4_turn1()),
                ("make it more organic looking", script4_turn2()),
                ("add a twist to it", script4_turn3()),
            ],
            26,
        );
        let s4_t1 = script4
            .start("a simple vase")
            .expect("script4 turn1 should work");
        let s4_t2 = script4
            .modify("make it more organic looking")
            .expect("script4 turn2 should work");
        let s4_t3 = script4
            .modify("add a twist to it")
            .expect("script4 turn3 should work");
        assert!(s4_t1.analysis.watertight);
        assert!(s4_t2.analysis.watertight);
        assert!(s4_t3.analysis.watertight);
        assert!(s4_t2.elapsed_ms < 3000.0);
        assert!(s4_t3.elapsed_ms < 3000.0);

        // Script 5: undo/revert to original.
        let mut script5 = make_engine(
            &[
                ("a bracket for a shelf", script5_turn1()),
                ("make it thicker", script5_turn2()),
                ("go back to the original", script5_turn3()),
            ],
            24,
        );
        let s5_t1 = script5
            .start("a bracket for a shelf")
            .expect("script5 turn1 should work");
        let s5_t2 = script5
            .modify("make it thicker")
            .expect("script5 turn2 should work");
        let s5_t3 = script5
            .modify("go back to the original")
            .expect("script5 turn3 should work");
        assert!(s5_t1.analysis.watertight);
        assert!(s5_t2.analysis.watertight);
        assert!(s5_t3.analysis.watertight);
        assert_eq!(normalize_dsl(&s5_t1.dsl), normalize_dsl(&s5_t3.dsl));
        assert!(s5_t2.elapsed_ms < 3000.0);
        assert!(s5_t3.elapsed_ms < 3000.0);
    }

    #[test]
    fn conversational_dimension_change_is_proportional_for_twice_width() {
        let mut engine = make_engine(
            &[
                ("a 40mm cube", twice_width_turn1()),
                ("make it twice as wide", twice_width_turn2()),
            ],
            26,
        );

        let t1 = engine.start("a 40mm cube").expect("turn1 should work");
        let t2 = engine
            .modify("make it twice as wide")
            .expect("turn2 should work");

        let ratio = axis_extent(&t2, 0) / axis_extent(&t1, 0);
        assert!(
            (ratio - 2.0).abs() <= 0.2,
            "expected width ratio near 2x (±10%), got {ratio:.3}"
        );
    }

    #[test]
    fn dimensional_constraints_validate_50mm_cube_bbox() {
        let prompt = "a 50mm cube";
        let constraints = extract_dimensional_constraints(prompt);
        assert_eq!(constraints.len(), 3);

        let report = verify_constraints_for_dsl(cube_50_dsl(), 48, &constraints)
            .expect("constraint verification should succeed");
        assert!(
            report.all_satisfied,
            "cube checks failed: {:?}",
            report.checks
        );
        assert_approx(
            report.analysis.bbox_max[0] - report.analysis.bbox_min[0],
            50.0,
            1.0,
        );
        assert_approx(
            report.analysis.bbox_max[1] - report.analysis.bbox_min[1],
            50.0,
            1.0,
        );
        assert_approx(
            report.analysis.bbox_max[2] - report.analysis.bbox_min[2],
            50.0,
            1.0,
        );
    }

    #[test]
    fn dimensional_constraints_validate_cylinder_dimensions() {
        let prompt = "cylinder 30mm diameter 80mm tall";
        let constraints = extract_dimensional_constraints(prompt);
        assert_eq!(constraints.len(), 3);

        let report = verify_constraints_for_dsl(cylinder_30x80_dsl(), 56, &constraints)
            .expect("constraint verification should succeed");
        assert!(
            report.all_satisfied,
            "cylinder checks failed: {:?}",
            report.checks
        );
        assert_approx(
            report.analysis.bbox_max[0] - report.analysis.bbox_min[0],
            30.0,
            1.0,
        );
        assert_approx(
            report.analysis.bbox_max[1] - report.analysis.bbox_min[1],
            30.0,
            1.0,
        );
        assert_approx(
            report.analysis.bbox_max[2] - report.analysis.bbox_min[2],
            80.0,
            1.0,
        );
    }

    #[test]
    fn dimensional_constraints_validate_wall_thickness_via_ray_cast() {
        let prompt = "hollow it out with wall thickness 3mm";
        let constraints = extract_dimensional_constraints(prompt);
        assert_eq!(constraints.len(), 1);

        let report = verify_constraints_for_dsl(shell_wall_3mm_dsl(), 64, &constraints)
            .expect("constraint verification should succeed");
        assert!(
            report.all_satisfied,
            "wall thickness checks failed: {:?}",
            report.checks
        );

        let check = report
            .checks
            .iter()
            .find(|check| {
                matches!(
                    check.constraint,
                    super::DimensionalConstraint::WallThickness { .. }
                )
            })
            .expect("wall thickness check should exist");
        let measured = check.measured.expect("wall thickness should be measurable");
        assert_approx(measured, 3.0, 0.8);
    }

    #[test]
    fn dimensional_constraints_validate_15_degree_tilt() {
        let prompt = "apply a 15 degree tilt";
        let constraints = extract_dimensional_constraints(prompt);
        assert_eq!(constraints.len(), 1);

        let report = verify_constraints_for_dsl(tilted_15deg_dsl(), 56, &constraints)
            .expect("constraint verification should succeed");
        assert!(
            report.all_satisfied,
            "tilt checks failed: {:?}",
            report.checks
        );

        let check = report
            .checks
            .iter()
            .find(|check| {
                matches!(
                    check.constraint,
                    super::DimensionalConstraint::TiltFromVertical { .. }
                )
            })
            .expect("tilt check should exist");
        let measured = check.measured.expect("tilt should be measurable");
        assert_approx(measured, 15.0, 1.5);
    }

    #[test]
    fn exact_width_constraint_auto_adjusts_and_retries() {
        let prompt = "make it exactly 42mm wide.";
        let constraints = extract_dimensional_constraints(prompt);
        assert_eq!(constraints.len(), 1, "constraints={constraints:?}");
        assert!(matches!(
            constraints[0],
            super::DimensionalConstraint::AxisExtent {
                axis: Axis::X,
                target_mm,
                tolerance_mm
            } if (target_mm - 42.0).abs() < 1e-9 && (tolerance_mm - 0.5).abs() < 1e-9
        ));

        let feedback = super::build_adjustment_feedback(&constraints);
        let model = ScriptedModel::default()
            .with_response(prompt, vec![width_40_dsl()])
            .with_response(&feedback, vec![width_42_dsl()]);
        let mut generator = DslGenerator::new(
            model,
            GenerationConfig {
                max_retries: 3,
                mesh_resolution: 128,
            },
        );

        let converged = enforce_constraints_with_retries(
            &mut generator,
            prompt,
            &constraints,
            ConstraintAdjustmentConfig {
                max_adjustment_rounds: 2,
            },
        )
        .expect("constraint adjustment should converge");

        assert_eq!(
            converged.adjustment_rounds, 1,
            "unexpected convergence details: rounds={}, checks={:?}, dsl={}",
            converged.adjustment_rounds, converged.report.checks, converged.dsl
        );
        assert!(converged.report.all_satisfied);
        let width = converged.report.analysis.bbox_max[0] - converged.report.analysis.bbox_min[0];
        assert_approx(width, 42.0, 0.5);

        let model = generator.into_client();
        let second = model
            .logs
            .iter()
            .find(|log| log.prompt == feedback)
            .expect("adjustment prompt call should be logged");
        assert!(
            second.current_dsl.is_some(),
            "adjustment retry should include current DSL context"
        );
    }

    #[test]
    fn property_constraints_are_extractable_and_verifiable() {
        let cases = [
            ("a 50mm cube", cube_50_dsl()),
            ("cylinder 30mm diameter 80mm tall", cylinder_30x80_dsl()),
            (
                "hollow it out with wall thickness 3mm",
                shell_wall_3mm_dsl(),
            ),
            ("apply a 15 degree tilt", tilted_15deg_dsl()),
            ("make it exactly 42mm wide", width_42_dsl()),
        ];

        for (prompt, dsl) in cases {
            let constraints = extract_dimensional_constraints(prompt);
            assert!(
                !constraints.is_empty(),
                "expected at least one constraint extracted from '{prompt}'"
            );
            let report = verify_constraints_for_dsl(dsl, 56, &constraints)
                .expect("constraint verification should succeed");
            assert!(
                report.all_satisfied,
                "constraints from '{prompt}' were not satisfied: {:?}",
                report.checks
            );
        }
    }

    fn canonical_scripted_model() -> ScriptedModel {
        let mut model = ScriptedModel::default();
        for (prompt, dsl) in canonical_programs() {
            model = model.with_response(prompt, vec![dsl]);
        }
        model
    }

    fn canonical_programs() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "a sphere",
                r#"// Centered sphere for baseline solid
params {
  radius = 20mm
}
sphere(radius)
"#,
            ),
            (
                "a 50mm cube",
                r#"// Exact 50mm cube primitive
params {
  size = 50mm
}
box(size, size, size)
"#,
            ),
            (
                "a cylinder 30mm diameter, 80mm tall",
                r#"// Vertical cylinder with specified diameter and height
params {
  radius = 15mm
  height = 80mm
}
cylinder(radius, height)
"#,
            ),
            (
                "a phone stand",
                r#"// Phone stand with base, back support, and cable slot
params {
  base_w = 90mm
  base_d = 70mm
  base_h = 8mm
  back_h = 75mm
  back_t = 10mm
  slot_w = 14mm
  slot_h = 6mm
}
base = rounded_box(base_w, base_d, base_h, 2mm)
back = rounded_box(base_w, back_t, back_h, 2mm) |> translate(0, (base_d - back_t) / 2, (back_h - base_h) / 2)
slot = box(slot_w, base_d + 2mm, slot_h) |> translate(0, 0, (slot_h - base_h) / 2)
result = union(base, back) |> difference(slot)
"#,
            ),
            (
                "a cable organizer with 4 slots",
                r#"// Block organizer with four cable cut channels
params {
  width = 80mm
  depth = 30mm
  height = 22mm
  slot_w = 8mm
  slot_d = 40mm
  slot_h = 14mm
}
body = rounded_box(width, depth, height, 2mm)
slot1 = box(slot_w, slot_d, slot_h) |> translate(-24mm, 0, 4mm)
slot2 = box(slot_w, slot_d, slot_h) |> translate(-8mm, 0, 4mm)
slot3 = box(slot_w, slot_d, slot_h) |> translate(8mm, 0, 4mm)
slot4 = box(slot_w, slot_d, slot_h) |> translate(24mm, 0, 4mm)
slots = union(union(slot1, slot2), union(slot3, slot4))
result = difference(body, slots)
"#,
            ),
            (
                "a rounded rectangular enclosure for a Raspberry Pi",
                r#"// Two-shell enclosure body with uniform wall thickness
params {
  outer_w = 96mm
  outer_d = 70mm
  outer_h = 36mm
  wall = 3mm
  fillet = 4mm
}
outer = rounded_box(outer_w, outer_d, outer_h, fillet)
inner = rounded_box(outer_w - (wall * 2), outer_d - (wall * 2), outer_h - (wall * 2), fillet - 1mm)
  |> translate(0, 0, wall)
result = difference(outer, inner)
"#,
            ),
            (
                "a wall-mount hook",
                r#"// Wall hook composed from a plate and curved arm
params {
  plate_w = 40mm
  plate_h = 70mm
  plate_t = 8mm
  hook_r = 7mm
  hook_len = 46mm
}
plate = box(plate_w, plate_t, plate_h)
arm = capped_cylinder(hook_r, hook_len / 2) |> rotate_x(90deg) |> translate(0, -7mm, 18mm)
tip = sphere(hook_r) |> translate(0, (hook_len / 2) - 7mm, 18mm)
result = union(plate, union(arm, tip))
"#,
            ),
            (
                "a vase with a twisted pattern",
                r#"// Twisted conical shell for decorative vase form
params {
  r1 = 26mm
  r2 = 12mm
  height = 120mm
  wall = 2.5mm
  twist_rate = 0.035
}
body = capped_cone(r1, r2, height)
result = shell(twist(body, twist_rate), wall)
"#,
            ),
            (
                "a pen holder with honeycomb pattern",
                r#"// Cylindrical holder with repeated side perforations
params {
  radius = 30mm
  height = 95mm
  wall = 3mm
  hole_r = 4mm
}
outer = cylinder(radius, height)
inner = cylinder(radius - wall, height - 4mm) |> translate(0, 0, 2mm)
side1 = cylinder(hole_r, height + 4mm) |> rotate_x(90deg) |> translate(18mm, 0, 0)
side2 = cylinder(hole_r, height + 4mm) |> rotate_x(90deg) |> translate(-18mm, 0, 0)
side3 = cylinder(hole_r, height + 4mm) |> rotate_x(90deg) |> translate(9mm, 15mm, 0)
side4 = cylinder(hole_r, height + 4mm) |> rotate_x(90deg) |> translate(-9mm, 15mm, 0)
side5 = cylinder(hole_r, height + 4mm) |> rotate_x(90deg) |> translate(9mm, -15mm, 0)
side6 = cylinder(hole_r, height + 4mm) |> rotate_x(90deg) |> translate(-9mm, -15mm, 0)
holes = union(union(side1, side2), union(union(side3, side4), union(side5, side6)))
result = difference(difference(outer, inner), holes)
"#,
            ),
            (
                "a soap dish with drainage holes",
                r#"// Rounded block with internal cavity and drainage tunnels
params {
  width = 100mm
  depth = 70mm
  height = 24mm
  wall = 3mm
  hole_r = 2.5mm
}
body = shell(rounded_box(width, depth, height, 6mm), wall)
hole1 = cylinder(hole_r, depth + 6mm) |> rotate_x(90deg) |> translate(-20mm, 0, -4mm)
hole2 = cylinder(hole_r, depth + 6mm) |> rotate_x(90deg) |> translate(0, 0, -4mm)
hole3 = cylinder(hole_r, depth + 6mm) |> rotate_x(90deg) |> translate(20mm, 0, -4mm)
holes = union(union(hole1, hole2), hole3)
result = difference(body, holes)
"#,
            ),
            (
                "a torus ring",
                r#"// Simple torus ring for primitive coverage
params {
  major = 28mm
  minor = 7mm
}
torus(major, minor)
"#,
            ),
            (
                "a capsule-style handle",
                r#"// Capsule profile suitable for a rounded handle
params {
  length = 60mm
  radius = 8mm
}
capsule(-length / 2, 0, 0, length / 2, 0, 0, radius)
"#,
            ),
            (
                "a tapered nozzle",
                r#"// Hollow tapered nozzle made from two cones
params {
  r1 = 14mm
  r2 = 6mm
  height = 48mm
  wall = 2mm
}
outer = capped_cone(r1, r2, height)
inner = capped_cone(r1 - wall, r2 - wall, height - 4mm) |> translate(0, 0, 2mm)
result = difference(outer, inner)
"#,
            ),
            (
                "a rounded cylindrical foot",
                r#"// Rounded cylinder for soft contact pad
params {
  radius = 16mm
  height = 18mm
  edge = 3mm
}
rounded_cylinder(radius, height, edge)
"#,
            ),
            (
                "a smooth blend between a sphere and a box",
                r#"// Smoothly blended sphere-box transition
params {
  sphere_r = 18mm
  box_size = 26mm
  blend = 4mm
}
shape_a = sphere(sphere_r)
shape_b = box(box_size, box_size, box_size) |> translate(12mm, 0, 0)
result = smooth_union(shape_a, shape_b, blend)
"#,
            ),
            (
                "an overlapping sphere-box intersection",
                r#"// Keep only overlap between sphere and translated box
params {
  sphere_r = 22mm
  box_size = 30mm
}
shape_a = sphere(sphere_r)
shape_b = box(box_size, box_size, box_size) |> translate(10mm, 0, 0)
result = intersection(shape_a, shape_b)
"#,
            ),
            (
                "a sphere with a central hole",
                r#"// Through-hole sphere using subtractive cylinder
params {
  sphere_r = 22mm
  hole_r = 6mm
}
base = sphere(sphere_r)
hole = cylinder(hole_r, sphere_r * 2 + 10mm) |> rotate_x(90deg)
result = difference(base, hole)
"#,
            ),
            (
                "a mirrored decorative fin",
                r#"// Build one fin then mirror it for bilateral symmetry
params {
  fin_w = 10mm
  fin_h = 48mm
  fin_d = 22mm
  tip_r = 6mm
}
half = union(
  box(fin_w, fin_d, fin_h),
  sphere(tip_r) |> translate(fin_w / 2, 0, fin_h / 2)
)
result = union(half, mirror(half, 1, 0, 0, 0))
"#,
            ),
            (
                "a twisted column with a slight bend",
                r#"// Organic column from twist and bend transforms
params {
  radius = 14mm
  height = 90mm
  twist_rate = 0.03
  bend_rate = 0.01
}
column = cylinder(radius, height)
result = bend(twist(column, twist_rate), bend_rate)
"#,
            ),
            (
                "a scaled and rotated puck",
                r#"// Scaled and tilted solid for placement preview
params {
  radius = 16mm
  scale_factor = 1.1
  tilt = 20deg
}
base = sphere(radius)
result = translate(scale(rotate_x(base, tilt), scale_factor), 4mm, -2mm, 1mm)
"#,
            ),
        ]
    }

    fn retry_success_program(index: usize) -> &'static str {
        const PROGRAMS: [&str; 10] = [
            r#"// Retry-safe program 1
params {
  radius = 10mm
}
sphere(radius)
"#,
            r#"// Retry-safe program 2
params {
  size = 18mm
}
box(size, size, size)
"#,
            r#"// Retry-safe program 3
params {
  r = 7mm
  h = 28mm
}
cylinder(r, h)
"#,
            r#"// Retry-safe program 4
params {
  r = 14mm
  t = 5mm
}
shell(sphere(r), t)
"#,
            r#"// Retry-safe program 5
params {
  a = 15mm
  b = 20mm
}
union(sphere(a), sphere(b) |> translate(10mm, 0, 0))
"#,
            r#"// Retry-safe program 6
params {
  r = 18mm
}
difference(sphere(r), cylinder(4mm, 60mm) |> rotate_x(90deg))
"#,
            r#"// Retry-safe program 7
params {
  r = 16mm
}
intersection(sphere(r), box(24mm, 24mm, 24mm))
"#,
            r#"// Retry-safe program 8
params {
  major = 24mm
  minor = 5mm
}
torus(major, minor)
"#,
            r#"// Retry-safe program 9
params {
  r = 8mm
  hh = 20mm
}
rounded_cylinder(r, hh * 2, 2mm)
"#,
            r#"// Retry-safe program 10
params {
  r = 12mm
  h = 50mm
}
bend(twist(cylinder(r, h), 0.02), 0.01)
"#,
        ];
        PROGRAMS[index]
    }

    fn make_engine(
        responses: &[(&'static str, &'static str)],
        mesh_resolution: usize,
    ) -> ConversationEngine<ScriptedModel> {
        let mut model = ScriptedModel::default();
        for (prompt, dsl) in responses {
            model = model.with_response(prompt, vec![dsl]);
        }
        ConversationEngine::new(DslGenerator::new(
            model,
            GenerationConfig {
                max_retries: 3,
                mesh_resolution,
            },
        ))
    }

    fn axis_extent(turn: &super::ConversationTurn, axis: usize) -> f64 {
        turn.analysis.bbox_max[axis] - turn.analysis.bbox_min[axis]
    }

    fn assert_approx(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected} ± {tolerance}, got {actual}"
        );
    }

    fn normalize_dsl(source: &str) -> String {
        source.lines().map(str::trim).collect::<Vec<_>>().join("\n")
    }

    fn script1_turn1() -> &'static str {
        r#"// Script1 turn1: 50mm cube
params {
  width = 50mm
  depth = 50mm
  height = 50mm
}
box(width, depth, height)
"#
    }

    fn script1_turn2() -> &'static str {
        r#"// Script1 turn2: widened to 80mm
params {
  width = 80mm
  depth = 50mm
  height = 50mm
}
box(width, depth, height)
"#
    }

    fn script1_turn3() -> &'static str {
        r#"// Script1 turn3: lowered to 30mm height
params {
  width = 80mm
  depth = 50mm
  height = 30mm
}
box(width, depth, height)
"#
    }

    fn script2_turn1() -> &'static str {
        r#"// Script2 turn1: solid cylinder
params {
  radius = 20mm
  height = 60mm
}
cylinder(radius, height)
"#
    }

    fn script2_turn2() -> &'static str {
        r#"// Script2 turn2: hollow cylinder walls
params {
  radius = 20mm
  height = 60mm
  wall = 3mm
}
outer = cylinder(radius, height)
inner = cylinder(radius - wall, height - (wall * 2))
result = difference(outer, inner)
"#
    }

    fn script2_turn3() -> &'static str {
        r#"// Script2 turn3: add a base plate
params {
  radius = 20mm
  height = 60mm
  wall = 3mm
}
outer = cylinder(radius, height)
inner = cylinder(radius - wall, height - (wall * 2))
shell = difference(outer, inner)
plate = box(60mm, 60mm, 6mm) |> translate(0, 0, -27mm)
result = union(shell, plate)
"#
    }

    fn script3_turn1() -> &'static str {
        r#"// Script3 turn1: solid rectangular block
params {
  width = 100mm
  depth = 50mm
  height = 30mm
}
box(width, depth, height)
"#
    }

    fn script3_turn2() -> &'static str {
        r#"// Script3 turn2: rounded block edges
params {
  width = 100mm
  depth = 50mm
  height = 30mm
  fillet = 4mm
}
rounded_box(width, depth, height, fillet)
"#
    }

    fn script3_turn3() -> &'static str {
        r#"// Script3 turn3: subtract central through-hole
params {
  width = 100mm
  depth = 50mm
  height = 30mm
  fillet = 4mm
  hole_r = 10mm
}
base = rounded_box(width, depth, height, fillet)
hole = cylinder(hole_r, 120mm) |> rotate_x(90deg)
result = difference(base, hole)
"#
    }

    fn script4_turn1() -> &'static str {
        r#"// Script4 turn1: simple vase shell
params {
  r1 = 26mm
  r2 = 14mm
  height = 110mm
  wall = 2.5mm
}
shell(capped_cone(r1, r2, height), wall)
"#
    }

    fn script4_turn2() -> &'static str {
        r#"// Script4 turn2: organic blend added
params {
  r1 = 26mm
  r2 = 14mm
  height = 110mm
  wall = 2.5mm
}
core = shell(capped_cone(r1, r2, height), wall)
bulge = sphere(16mm) |> translate(0, 0, 10mm)
result = smooth_union(core, bulge, 6mm)
"#
    }

    fn script4_turn3() -> &'static str {
        r#"// Script4 turn3: add twist transform
params {
  r1 = 26mm
  r2 = 14mm
  height = 110mm
  wall = 2.5mm
  twist_rate = 0.03
}
core = shell(capped_cone(r1, r2, height), wall)
bulge = sphere(16mm) |> translate(0, 0, 10mm)
organic = smooth_union(core, bulge, 6mm)
result = twist(organic, twist_rate)
"#
    }

    fn script5_turn1() -> &'static str {
        r#"// Script5 turn1: original shelf bracket
params {
  width = 60mm
  depth = 30mm
  thickness = 8mm
  upright_h = 50mm
}
base = box(width, depth, thickness)
upright = box(thickness, depth, upright_h) |> translate((width - thickness) / 2, 0, (upright_h - thickness) / 2)
result = union(base, upright)
"#
    }

    fn script5_turn2() -> &'static str {
        r#"// Script5 turn2: thicker shelf bracket
params {
  width = 60mm
  depth = 30mm
  thickness = 12mm
  upright_h = 50mm
}
base = box(width, depth, thickness)
upright = box(thickness, depth, upright_h) |> translate((width - thickness) / 2, 0, (upright_h - thickness) / 2)
result = union(base, upright)
"#
    }

    fn script5_turn3() -> &'static str {
        script5_turn1()
    }

    fn twice_width_turn1() -> &'static str {
        r#"// Dimensional proportionality baseline
params {
  width = 40mm
  depth = 40mm
  height = 40mm
}
box(width, depth, height)
"#
    }

    fn twice_width_turn2() -> &'static str {
        r#"// Dimensional proportionality target at 2x width
params {
  width = 80mm
  depth = 40mm
  height = 40mm
}
box(width, depth, height)
"#
    }

    fn cube_50_dsl() -> &'static str {
        r#"// Dimensional check: 50mm cube
params {
  size = 50mm
}
box(size, size, size)
"#
    }

    fn cylinder_30x80_dsl() -> &'static str {
        r#"// Dimensional check: 30mm diameter, 80mm tall cylinder
params {
  radius = 15mm
  height = 80mm
}
cylinder(radius, height)
"#
    }

    fn shell_wall_3mm_dsl() -> &'static str {
        r#"// Dimensional check: 3mm shell thickness
params {
  radius = 20mm
  wall = 1.5mm
}
shell(sphere(radius), wall)
"#
    }

    fn tilted_15deg_dsl() -> &'static str {
        r#"// Dimensional check: 15 degree tilt from vertical
params {
  radius = 10mm
  height = 80mm
  tilt = 15deg
}
rotate_x(cylinder(radius, height), tilt)
"#
    }

    fn width_40_dsl() -> &'static str {
        r#"// Initial model misses exact-width target
params {
  width = 36mm
  depth = 30mm
  height = 20mm
}
box(width, depth, height)
"#
    }

    fn width_42_dsl() -> &'static str {
        r#"// Adjusted model matches exact-width target
params {
  width = 42mm
  depth = 30mm
  height = 20mm
}
box(width, depth, height)
"#
    }
}
