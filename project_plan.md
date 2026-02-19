
# AI-Powered SDF CAD Engine — Project Plan

## Project Overview

**Product:** A conversational AI-driven CAD tool where users describe 3D objects in natural language, an LLM generates/modifies SDF (Signed Distance Function) programs, and a fast Rust engine evaluates them to meshes in real-time.

**Architecture:**
```
User (natural language) → LLM (generates SDF code) → Rust SDF Engine (evaluates to mesh) → Web Viewer (three.js)
                       ↑                                                                    |
                       └──────────────── User feedback (natural language) ←──────────────────┘
```

**Oracle Reference:** [fogleman/sdf](https://github.com/fogleman/sdf) — Pure Python SDF mesh generation library (~1,500 LOC, MIT license)

**Target Timeline:** 10–14 weeks for MVP with a small team (2–4 developers)

---

## Guiding Principles

1. **Every task produces a verifiable artifact.** No task is "done" without passing its defined tests.
2. **Oracle-first development.** The Python reference implementation is ground truth. Every Rust function must match its Python equivalent to within defined tolerances before proceeding.
3. **Bottom-up construction.** Build primitives → operations → pipeline → API → AI integration → frontend. Never skip ahead.
4. **Continuous benchmarking.** Every PR must include benchmark comparisons against the Python oracle. Regressions block merging.
5. **Test categories are explicit.** Each test is tagged as: `[UNIT]` (single function), `[INTEGRATION]` (multi-component), `[ORACLE]` (comparison to Python), `[PROPERTY]` (mathematical invariant), `[BENCHMARK]` (performance).

---

## Phase 1: Rust SDF Evaluation Engine (Weeks 1–4)

### Task 1.1: Project Scaffolding and Oracle Harness

**Goal:** Set up the Rust project, Python oracle runner, and cross-language test harness.

**Deliverables:**
- Rust workspace with `sdf-core`, `sdf-mesh`, `sdf-cli` crates
- Python virtualenv with fogleman/sdf installed
- Cross-language test harness: a script that evaluates both implementations on identical inputs and compares outputs
- CI pipeline (GitHub Actions) running all tests on every push

**Implementation Details:**
```
ai-sdf-cad/
├── Cargo.toml                  # Workspace root
├── crates/
│   ├── sdf-core/               # SDF primitives and operations
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── primitives.rs   # Sphere, box, cylinder, etc.
│   │   │   ├── operations.rs   # Union, intersection, blend, etc.
│   │   │   ├── transforms.rs   # Translate, rotate, scale, etc.
│   │   │   └── evaluate.rs     # Batch SDF evaluation on point grids
│   │   └── Cargo.toml
│   ├── sdf-mesh/               # Marching cubes mesh extraction
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── marching_cubes.rs
│   │   │   └── export.rs       # STL, OBJ export
│   │   └── Cargo.toml
│   ├── sdf-dsl/                # DSL parser for AI-generated SDF programs
│   │   └── ...
│   ├── sdf-server/             # HTTP API server
│   │   └── ...
│   └── sdf-cli/                # CLI tool for testing/benchmarking
│       └── ...
├── oracle/                     # Python oracle comparison tools
│   ├── requirements.txt        # fogleman/sdf, numpy, trimesh
│   ├── evaluate_oracle.py      # Evaluate Python SDF at given points
│   ├── generate_test_points.py # Generate deterministic test point sets
│   └── compare_results.py      # Cross-language numerical comparison
├── tests/
│   ├── golden/                 # Saved oracle outputs for regression
│   └── integration/            # Cross-crate integration tests
└── benches/                    # Criterion benchmarks
```

**Tests:**
- `[UNIT]` Rust project compiles with zero warnings (`cargo build`, `cargo clippy -- -D warnings`)
- `[UNIT]` Python oracle evaluates `sphere(1.0)` at origin and returns `-1.0`
- `[INTEGRATION]` Cross-language harness runs: generate 100 random points, evaluate sphere in both, compare with ε < 1e-12
- `[UNIT]` CI pipeline runs and reports pass/fail

**Acceptance Criteria:** Running `cargo test` and `python oracle/compare_results.py` both pass. CI is green.

---

### Task 1.2: Core SDF Primitives

**Goal:** Implement all geometric primitives from fogleman/sdf in Rust.

**Primitives to implement (each is an independent sub-task):**

| # | Primitive | Signature | Analytical Test |
|---|-----------|-----------|----------------|
| 1 | `sphere` | `(radius) → f64` | `sdf(origin) = -radius`, `sdf(surface) = 0`, `sdf(2*radius, 0, 0) = radius` |
| 2 | `box` | `(half_extents) → f64` | Interior point < 0, corner point = 0, exterior = distance to nearest face |
| 3 | `rounded_box` | `(half_extents, radius) → f64` | Equivalent to `box - radius` evaluated at `half_extents - radius` |
| 4 | `cylinder` | `(radius, height) → f64` | `sdf(axis_point_inside) < 0`, `sdf(surface) = 0` |
| 5 | `capped_cylinder` | `(radius, half_height) → f64` | Check both cap and barrel distances |
| 6 | `torus` | `(major_r, minor_r) → f64` | `sdf(major_r, 0, 0) = -minor_r`, ring geometry |
| 7 | `plane` | `(normal, offset) → f64` | `sdf = dot(p, normal) - offset` exactly |
| 8 | `capsule` | `(a, b, radius) → f64` | Line segment distance minus radius |
| 9 | `capped_cone` | `(r1, r2, height) → f64` | Check top/bottom cap radii |
| 10 | `rounded_cylinder` | `(radius, height, edge_r) → f64` | Edge rounding verification |

**Tests per primitive (template — apply to each):**

```
[ORACLE] 10,000 random points in [-3, 3]³:
         |rust_sdf(p) - python_sdf(p)| < 1e-12 for ALL points.
         Points are deterministic (seeded RNG, seed=42).

[PROPERTY] Sign consistency:
           For 1,000 points known to be inside (generated by rejection sampling):
           sdf(p) < 0 for all.
           For 1,000 points known to be outside: sdf(p) > 0 for all.

[PROPERTY] Lipschitz continuity:
           For 10,000 random point pairs (p, q):
           |sdf(p) - sdf(q)| ≤ |p - q| * (1 + ε), ε = 1e-6.
           (True SDF has Lipschitz constant 1.)

[PROPERTY] Surface point verification:
           Find points where |sdf(p)| < 1e-6 by bisection along rays.
           Verify these points lie on the expected geometric surface
           (e.g., distance from origin = radius for sphere).

[BENCHMARK] Evaluate 1M points: measure ns/point.
            Must be < 50ns/point for simple primitives (sphere, box).
            Log Python oracle time for comparison.
```

**Acceptance Criteria:** All 10 primitives pass all test categories. Zero oracle comparison failures at ε < 1e-12.

---

### Task 1.3: CSG Boolean Operations

**Goal:** Implement SDF composition operations.

**Operations to implement:**

| # | Operation | Formula | Notes |
|---|-----------|---------|-------|
| 1 | `union` | `min(a, b)` | Exact for true SDFs |
| 2 | `intersection` | `max(a, b)` | Exact for true SDFs |
| 3 | `difference` | `max(a, -b)` | Exact for true SDFs |
| 4 | `smooth_union` | smooth min with blending radius k | Not exact SDF; Lipschitz ≤ 1 still holds |
| 5 | `smooth_intersection` | smooth max with k | Dual of smooth_union |
| 6 | `smooth_difference` | smooth max(a, -b) with k | Dual |
| 7 | `negate` | `-sdf(p)` | Flips inside/outside |
| 8 | `shell` | `abs(sdf(p)) - thickness` | Hollow shell |
| 9 | `elongate` | Extend along axis | Modifies input point |
| 10 | `repeat` | Domain repetition | Modular arithmetic on input |

**Tests:**

```
[ORACLE] For each operation, compose two primitives (sphere + box),
         evaluate 10,000 random points, compare against Python oracle.
         Tolerance: ε < 1e-11 (slightly relaxed for smooth operations).

[PROPERTY] Boolean algebra identities (1,000 random points each):
           - union(A, A) ≡ A
           - intersection(A, A) ≡ A
           - difference(A, A) ≤ 0 everywhere (result is empty or surface)
           - union(A, B) ≡ union(B, A)  (commutativity)
           - intersection(A, B) ≡ intersection(B, A)
           - negate(negate(A)) ≡ A

[PROPERTY] Volume consistency (via Monte Carlo integration, 100K points):
           vol(A ∪ B) + vol(A ∩ B) = vol(A) + vol(B), tolerance < 1%.
           Volume estimated by counting points where sdf < 0.

[PROPERTY] Smooth operations converge to sharp:
           As k → 0, smooth_union(a, b, k) → union(a, b).
           Verify |smooth_union(a, b, 0.001) - min(a, b)| < 0.01
           at 10,000 points.

[INTEGRATION] Nested operations:
              difference(union(sphere, box), cylinder) —
              evaluate 10,000 points, compare to oracle.

[BENCHMARK] 1M points through union(sphere, box): < 100ns/point.
```

**Acceptance Criteria:** All operations pass oracle comparison. Boolean algebra identities hold. Volume consistency within 1%.

---

### Task 1.4: Spatial Transformations

**Goal:** Implement coordinate transformations that modify the SDF input point.

**Transforms to implement:**

| # | Transform | Method |
|---|-----------|--------|
| 1 | `translate` | `sdf(p - offset)` |
| 2 | `rotate` | `sdf(R⁻¹ · p)` — rotation matrix applied to point |
| 3 | `scale` | `sdf(p / s) * s` — uniform scale preserves SDF property |
| 4 | `orient` | Align object along arbitrary axis |
| 5 | `mirror` | Reflect across plane |
| 6 | `twist` | Rotate around axis as function of height |
| 7 | `bend` | Deform along axis |

**Tests:**

```
[ORACLE] Each transform applied to sphere: 10,000 points, ε < 1e-11.

[PROPERTY] translate(sphere, v): center moves to v.
           Verify sdf(v) = -radius.

[PROPERTY] rotate preserves distances:
           sdf_rotated(R·p) = sdf_original(p) for all p.

[PROPERTY] uniform scale(sphere, s):
           sdf_scaled(origin) = -radius * s.

[PROPERTY] mirror(sphere, plane): result has reflective symmetry.
           sdf(p) = sdf(reflect(p)) for 10,000 random points.

[PROPERTY] Composition: translate(rotate(X)) ≠ rotate(translate(X)) in general.
           Verify this holds (catches accidentally commuted transforms).

[BENCHMARK] Translated sphere, 1M points: overhead < 10ns/point vs bare sphere.
```

---

### Task 1.5: Marching Cubes Mesh Extraction

**Goal:** Implement marching cubes to convert the SDF field to a triangle mesh.

**Implementation Notes:**
- Standard marching cubes with 256-entry edge table and 256-entry triangle table
- Vertex interpolation along edges where SDF changes sign
- Output: vertex array `Vec<[f64; 3]>` + triangle index array `Vec<[u32; 3]>`

**Tests:**

```
[ORACLE] Generate mesh for unit sphere at resolution 64³.
         Compare vertex count within 5%.
         Compare mesh volume within 1% of analytical (4π/3).
         Compare mesh surface area within 2% of analytical (4π).

[ORACLE] Generate mesh for union(sphere, box) at resolution 64³.
         Compare mesh volume against Python oracle volume (via trimesh)
         within 1%.

[PROPERTY] Watertightness: Every edge in the mesh is shared by exactly
           2 triangles (manifold mesh). Verify using half-edge check.

[PROPERTY] Consistent winding: All triangle normals point outward
           (dot(normal, centroid_direction) > 0 for convex shapes).

[PROPERTY] Volume convergence: As resolution increases (32→64→128),
           mesh volume converges to analytical volume.
           Verify |vol(128) - analytical| < |vol(64) - analytical| < |vol(32) - analytical|.

[PROPERTY] No degenerate triangles: All triangles have area > ε (1e-10).

[UNIT] Edge cases:
       - SDF that is positive everywhere (empty mesh, 0 triangles)
       - SDF that is negative everywhere (no surface, 0 triangles)
       - SDF with sharp features (cube) produces reasonable mesh

[BENCHMARK] Sphere at resolution 128³:
            - Total time < 500ms (Python oracle takes 10-60s)
            - Memory usage < 500MB
```

---

### Task 1.6: Mesh Export (STL / OBJ / 3MF)

**Goal:** Export meshes to standard 3D printing and CAD interchange formats.

**Tests:**

```
[UNIT] STL binary export: output file is valid binary STL.
       Load with trimesh (Python), verify vertex/face counts match.

[UNIT] STL ASCII export: output is valid ASCII STL.
       Round-trip: export → load with trimesh → compare vertices.

[UNIT] OBJ export: valid Wavefront OBJ.
       Load with trimesh, verify mesh properties match.

[INTEGRATION] Full pipeline: SDF definition → evaluate → marching cubes → STL export.
              Load exported STL in trimesh, verify:
              - Volume within 1% of analytical
              - Watertight
              - No degenerate faces

[PROPERTY] Round-trip fidelity: export to STL → load → re-export → binary compare.
           Files must be identical (deterministic export).

[BENCHMARK] Export 100K-triangle mesh to binary STL: < 50ms.
```

---

### Task 1.7: Performance Optimization and SIMD

**Goal:** Optimize the SDF evaluation pipeline for throughput.

**Optimizations:**
- Batch evaluation: evaluate SDF at N points simultaneously using SIMD
- Spatial subdivision: skip marching cubes cells where all 8 corners have same sign
- Adaptive resolution: finer grid near surface, coarser far away (octree)
- Parallelism: rayon for parallel grid evaluation

**Tests:**

```
[BENCHMARK] Baseline vs optimized, all using sphere at 128³ resolution:
            - Sequential baseline: measure time
            - SIMD batch evaluation: ≥ 4x speedup over sequential
            - Parallel (rayon): ≥ 3x speedup on 4-core machine
            - Combined: ≥ 10x speedup over sequential baseline

[ORACLE] Optimized output must be BIT-IDENTICAL to non-optimized output.
         Compare vertex arrays, triangle arrays — zero tolerance.
         (Optimizations must not change results, only speed.)

[BENCHMARK] Full pipeline (evaluate + marching cubes + export) comparison:
            Rust total time vs Python total time.
            Target: ≥ 100x speedup for sphere at 128³.
            Log and track across runs.

[PROPERTY] Memory profile: peak RSS < 1GB for 256³ resolution.
```

**Acceptance Criteria:** ≥ 100x speedup over Python oracle on the full pipeline. Output is bit-identical to unoptimized Rust.

---

## Phase 2: SDF Domain-Specific Language (Weeks 4–6)

### Task 2.1: SDF DSL Design

**Goal:** Design a textual language for describing SDF scenes that an LLM can generate.

**Design Requirements:**
- Simple enough for an LLM to generate reliably (minimal syntax)
- Expressive enough to describe complex compositions
- Parameters are named and have units (mm)
- Variables and arithmetic expressions for parametric designs
- Comments for AI self-documentation

**Example DSL:**

```
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
```

**Tests:**

```
[UNIT] Parse minimal program: "sphere(10mm)" → AST with Sphere node, radius=10.0.

[UNIT] Parse all primitives: one test per primitive type.

[UNIT] Parse operations: union, intersection, difference, smooth variants.

[UNIT] Parse transforms: translate, rotate, scale with correct parameters.

[UNIT] Parse pipe operator: "sphere(5mm) |> translate(10, 0, 0)"
       → TranslatedSphere AST.

[UNIT] Parse params block: variables resolve to correct numeric values.

[UNIT] Parse arithmetic: "width/2 + 5mm" evaluates correctly.

[UNIT] Error handling: malformed input produces clear error message
       with line number and column.

[UNIT] Comments: "//" line comments and "/* */" block comments ignored.

[INTEGRATION] Parse the phone stand example above → evaluate → mesh.
              Mesh is watertight and has reasonable volume.

[PROPERTY] Round-trip: parse → serialize → parse → compare ASTs.
           Must be identical.

[UNIT] Edge cases:
       - Empty program → error with message
       - Unknown primitive name → error with suggestion
       - Missing parameter → error naming the parameter
       - Division by zero → error at evaluation time
       - Negative radius → error at parse time
```

---

### Task 2.2: DSL-to-SDF Compiler

**Goal:** Compile the DSL AST into an optimized SDF evaluation tree.

**Tests:**

```
[INTEGRATION] DSL string → compile → evaluate at 10,000 points.
              Compare against manually-constructed SDF tree
              (built directly in Rust) at same points. ε < 1e-12.

[BENCHMARK] Compile time for 100-line DSL program: < 10ms.

[BENCHMARK] Evaluation throughput of compiled DSL matches
            hand-built Rust SDF tree within 10%.

[PROPERTY] Deterministic compilation:
           same DSL string → same evaluation results every time.

[INTEGRATION] Full pipeline from DSL string to exported STL:
              "sphere(10mm)" → STL file.
              Load STL, verify volume = 4π(10)³/3 mm³ within 1%.
```

---

### Task 2.3: Parametric Modification API

**Goal:** Support modifying DSL parameters without re-parsing, enabling rapid AI iteration.

**API:**
```rust
let mut scene = compile_dsl(source)?;
scene.set_param("width", 100.0)?;     // Change a parameter
scene.set_param("tilt_angle", 20.0)?;  // Change another
let mesh = scene.evaluate(resolution)?; // Re-evaluate with new params
```

**Tests:**

```
[UNIT] set_param("radius", 20.0) on sphere(10mm):
       sdf(origin) changes from -10 to -20.

[UNIT] set_param with invalid name → descriptive error.

[UNIT] set_param with out-of-range value (negative radius) → error.

[INTEGRATION] Modify 5 parameters on phone stand DSL,
              re-evaluate, verify mesh volume changes proportionally.

[BENCHMARK] Parameter change + re-evaluate at 64³: < 200ms.
            (This is the AI iteration loop — must be fast.)

[PROPERTY] Changing a parameter and changing it back
           produces bit-identical mesh to original.
```

---

## Phase 3: HTTP API Server (Weeks 6–8)

### Task 3.1: REST API

**Goal:** HTTP server that accepts DSL programs and returns meshes.

**Endpoints:**

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/evaluate` | `{ "dsl": "...", "resolution": 64 }` | `{ "mesh": { "vertices": [...], "triangles": [...] }, "stats": { "time_ms": ..., "triangle_count": ... } }` |
| POST | `/evaluate/stl` | `{ "dsl": "...", "resolution": 64 }` | Binary STL file |
| POST | `/modify` | `{ "dsl": "...", "params": { "width": 100 } }` | Same as /evaluate but with param overrides |
| POST | `/validate` | `{ "dsl": "..." }` | `{ "valid": true, "errors": [], "params": [...], "bounding_box": {...} }` |
| GET | `/health` | — | `{ "status": "ok" }` |

**Tests:**

```
[UNIT] POST /evaluate with "sphere(10mm)" returns valid JSON
       with vertices and triangles arrays.

[UNIT] POST /evaluate/stl returns valid binary STL
       (check magic number and triangle count header).

[UNIT] POST /validate with valid DSL returns valid=true
       and lists all extractable parameters.

[UNIT] POST /validate with invalid DSL returns valid=false
       with descriptive error messages.

[UNIT] POST /modify changes parameter and returns different mesh.

[UNIT] POST /evaluate with empty body → 400 with error message.

[UNIT] POST /evaluate with DSL that would produce > 10M triangles → 413
       with message about resolution limit.

[INTEGRATION] Full round-trip:
              POST /evaluate → receive mesh JSON → verify volume of sphere.

[BENCHMARK] POST /evaluate with sphere at 64³: < 500ms total
            (including HTTP overhead, serialization).

[BENCHMARK] 10 concurrent requests: all complete within 5s.
            Server does not crash or deadlock.

[UNIT] CORS headers present (required for browser frontend).
```

---

### Task 3.2: WebSocket API for Real-Time Preview

**Goal:** WebSocket endpoint for streaming mesh updates during parameter tweaking.

**Protocol:**
```
Client → Server: { "type": "set_dsl", "dsl": "..." }
Server → Client: { "type": "mesh", "vertices": [...], "triangles": [...], "time_ms": ... }

Client → Server: { "type": "set_param", "name": "width", "value": 100 }
Server → Client: { "type": "mesh", ... }  // Updated mesh within 200ms
```

**Tests:**

```
[UNIT] Connect to WebSocket, send set_dsl, receive mesh response.

[UNIT] Send set_param, receive updated mesh.

[UNIT] Send rapid set_param (10 updates in 100ms):
       server debounces and responds with final state, not 10 meshes.

[UNIT] Send invalid DSL → receive error message, connection stays open.

[UNIT] Disconnect and reconnect — server handles gracefully.

[BENCHMARK] Latency from set_param to mesh response: < 200ms at 64³.
            This is the critical real-time interaction metric.
```

---

## Phase 4: AI Integration (Weeks 8–10)

### Task 4.1: LLM Prompt Engineering for SDF Generation

**Goal:** Design and test system prompts that reliably produce valid SDF DSL from natural language descriptions.

**System Prompt Components:**
1. DSL grammar reference (complete syntax specification)
2. Primitive catalog with descriptions and typical use cases
3. Operation catalog with visual descriptions of effects
4. Coordinate system conventions (Y-up, mm units)
5. Common patterns (enclosure, bracket, stand, container)
6. Constraints: output must be valid DSL, params block required, comments explaining design choices

**Tests:**

```
[UNIT] 20 canonical prompts → DSL → validate → all must parse successfully.
       Prompts cover range of complexity:
       1. "a sphere" → valid DSL
       2. "a 50mm cube" → valid DSL with correct dimensions
       3. "a cylinder 30mm diameter, 80mm tall" → valid DSL
       4. "a phone stand" → valid DSL that evaluates to watertight mesh
       5. "a cable organizer with 4 slots" → valid, watertight
       6. "a rounded rectangular enclosure for a Raspberry Pi" → valid
       7. "a wall-mount hook" → valid
       8. "a vase with a twisted pattern" → valid (uses twist)
       9. "a pen holder with honeycomb pattern" → valid
       10. "a soap dish with drainage holes" → valid
       (+ 10 more covering all primitives and operations)

[PROPERTY] Every generated DSL, when evaluated, produces:
           - Watertight mesh
           - Positive volume
           - Bounding box within reasonable range (1mm–500mm per axis)
           - No degenerate triangles

[INTEGRATION] Full loop:
              "make a bookend shaped like the letter L" →
              LLM generates DSL → engine evaluates → mesh exported →
              verify watertight, volume > 0, bounding box reasonable.

[UNIT] Error recovery: if LLM output fails validation,
       send error message back to LLM with the validation errors.
       LLM must produce valid DSL on retry (test 10 deliberately tricky prompts).
       Allow up to 3 retries.
```

**Evaluation Harness:**
```python
# Run nightly against test prompt suite
# Track: parse success rate, mesh validity rate, retry rate
# Alert if parse success drops below 95% or mesh validity below 90%
```

---

### Task 4.2: Conversational Modification Loop

**Goal:** Enable multi-turn conversations where the AI modifies existing SDF programs based on feedback.

**Architecture:**
```
Turn 1: User: "make a phone stand"
        AI: generates initial DSL
        Engine: evaluates to mesh, renders preview

Turn 2: User: "make the base wider"
        AI: receives current DSL + user feedback
        AI: modifies DSL (changes width parameter or geometry)
        Engine: re-evaluates

Turn 3: User: "add a cable slot in the back"
        AI: adds difference operation with slot geometry
        Engine: re-evaluates
```

**Tests:**

```
[INTEGRATION] 5 multi-turn conversation test scripts:

  Script 1 - Dimensional changes:
    T1: "a 50mm cube" → cube DSL
    T2: "make it 80mm wide" → modified DSL, verify bounding box X ≈ 80mm
    T3: "and 30mm tall" → verify bounding box Z ≈ 30mm

  Script 2 - Adding features:
    T1: "a cylinder 40mm diameter 60mm tall" → cylinder DSL
    T2: "hollow it out with 3mm walls" → verify shell operation, volume decreases
    T3: "add a base plate" → verify volume increases

  Script 3 - Material operations:
    T1: "a solid block 100x50x30mm" → box DSL
    T2: "round all the edges" → verify smooth operations added
    T3: "cut a 20mm hole through the center" → verify difference, volume decreases

  Script 4 - Style changes:
    T1: "a simple vase" → vase DSL
    T2: "make it more organic looking" → verify smooth operations
    T3: "add a twist to it" → verify twist transform

  Script 5 - Undo/revert:
    T1: "a bracket for a shelf" → DSL_v1
    T2: "make it thicker" → DSL_v2
    T3: "go back to the original" → DSL_v3, verify DSL_v3 ≈ DSL_v1

[PROPERTY] Every intermediate DSL produces a watertight mesh.

[PROPERTY] Dimensional changes are proportional:
           "make it twice as wide" → bounding box X doubles (±10%).

[PROPERTY] Additive operations increase volume:
           "add a base plate" → vol(after) > vol(before).

[PROPERTY] Subtractive operations decrease volume:
           "cut a hole" → vol(after) < vol(before).

[BENCHMARK] Each modification turn: < 3 seconds total
            (LLM generation + validation + evaluation + mesh response).
```

---

### Task 4.3: Dimensional Constraint Validation

**Goal:** Automatically verify that AI-generated models meet user-specified dimensional requirements.

**Tests:**

```
[UNIT] "50mm cube" → bounding box is 50×50×50mm ± 1mm.

[UNIT] "cylinder 30mm diameter 80mm tall" →
       bounding box X,Y ≈ 30mm, Z ≈ 80mm, ± 1mm.

[UNIT] "wall thickness 3mm" → shell operation with correct offset.
       Verify by ray casting: measure distance between inner and outer surfaces.

[UNIT] "15 degree tilt" → measure angle from vertical axis.

[INTEGRATION] User says "make it exactly 42mm wide."
              Verify bounding box X = 42mm ± 0.5mm.
              If not, system automatically adjusts and retries.

[PROPERTY] All dimensional constraints from user prompt are
           extractable and verifiable against the output mesh.
```

---

## Phase 5: Web Frontend (Weeks 9–12)

### Task 5.1: Three.js 3D Viewer

**Goal:** Browser-based 3D mesh viewer with orbit controls, wireframe toggle, and measurement tools.

**Tests:**

```
[UNIT] Viewer loads and renders a test mesh (sphere, 1000 triangles)
       without console errors.

[UNIT] Orbit controls: camera responds to mouse drag.

[UNIT] Wireframe toggle: switches between solid and wireframe rendering.

[UNIT] Lighting: mesh has visible shading (not flat black or white).

[INTEGRATION] Receive mesh from WebSocket → render in viewer.
              Visual regression test: screenshot comparison against
              reference rendering (SSIM > 0.95).

[BENCHMARK] Render 100K-triangle mesh at 60fps.
            Measure frame time with requestAnimationFrame.

[UNIT] Empty mesh (0 triangles) → viewer shows empty scene, no crash.
```

---

### Task 5.2: Chat Interface

**Goal:** Chat UI for natural language interaction with the AI.

**Tests:**

```
[UNIT] Type message, press enter → message appears in chat history.

[UNIT] AI response appears after sending message.

[UNIT] Loading indicator shown while AI is generating.

[UNIT] Error message displayed when AI/server fails.

[UNIT] Chat history scrolls to latest message.

[UNIT] Code block rendering: DSL code shown with syntax highlighting.

[INTEGRATION] Full loop: type "make a sphere" → see mesh appear in 3D viewer.

[UNIT] Mobile responsive: chat and viewer usable at 375px width.
```

---

### Task 5.3: Parameter Adjustment UI

**Goal:** Slider/input controls for tweaking DSL parameters in real time.

**Tests:**

```
[UNIT] Parameters extracted from DSL appear as labeled sliders.

[UNIT] Moving slider sends set_param WebSocket message.

[UNIT] Mesh updates within 200ms of slider change.

[UNIT] Slider range auto-detected from parameter type
       (radius: 1–100mm, angle: 0–360deg).

[UNIT] Manual numeric input in text field works alongside slider.

[INTEGRATION] Adjust width slider from 50 to 100 →
              mesh bounding box X changes proportionally.
              Verify by reading back mesh data from viewer.

[UNIT] Reset button restores original parameter values.
```

---

### Task 5.4: Export and Download

**Goal:** Download generated models in standard formats.

**Tests:**

```
[UNIT] "Download STL" button triggers file download.

[UNIT] Downloaded STL is valid (load with trimesh programmatically).

[UNIT] "Download OBJ" produces valid OBJ file.

[UNIT] "Download 3MF" produces valid 3MF archive.

[UNIT] Filename includes model description: "phone-stand.stl".

[INTEGRATION] Generate model → download STL → load in PrusaSlicer CLI →
              verify it slices without errors.
              (Automated: PrusaSlicer has CLI mode for validation.)
```

---

## Phase 6: Quality, Polish, and Launch Prep (Weeks 12–14)

### Task 6.1: End-to-End Test Suite

**Goal:** Comprehensive tests covering the entire user journey.

**Tests:**

```
[E2E] 10 complete user journeys, each fully automated:

Journey 1: "Phone Stand"
  - User: "design a phone stand for an iPhone 15"
  - Verify: mesh generated, watertight, width ≈ 80mm
  - User: "make the tilt angle steeper"
  - Verify: tilt increases, mesh still watertight
  - User: "add a cable slot"
  - Verify: volume decreases (hole cut), still watertight
  - Export STL → validate with trimesh

Journey 2: "Desk Organizer"
  - User: "pen holder with 4 compartments"
  - Verify: watertight, reasonable dimensions
  - User: "make it rounded"
  - Verify: smooth operations applied
  - Export → validate

Journey 3: "Mechanical Bracket"
  - User: "L-bracket with mounting holes, 5mm thick aluminum"
  - Verify: thickness ≈ 5mm (measured by ray casting)
  - User: "add reinforcing ribs"
  - Verify: volume increases
  - Export → validate

... (7 more covering: vase, hook, enclosure, spacer, knob, clip, funnel)

[PROPERTY] ALL 10 journeys produce watertight meshes at every step.

[PROPERTY] ALL export files are valid and loadable by trimesh.

[BENCHMARK] No single turn takes > 5 seconds (LLM + evaluation + render).
```

---

### Task 6.2: Error Handling and Edge Cases

**Tests:**

```
[UNIT] User sends empty message → polite "could you describe what you'd like?" response.

[UNIT] User sends non-CAD request ("what's the weather?") →
       polite redirect to CAD functionality.

[UNIT] LLM generates invalid DSL → automatic retry (up to 3 times).
       If all retries fail → user sees friendly error message.

[UNIT] DSL produces non-manifold mesh → system detects and
       attempts automatic repair (adjust resolution, slight parameter tweak).

[UNIT] Very large model (>1M triangles) → warning shown,
       offer to reduce resolution.

[UNIT] Server crash recovery: if evaluation panics,
       server recovers and accepts next request.

[UNIT] WebSocket disconnection → frontend auto-reconnects.

[UNIT] Concurrent users (simulate 10) → all get correct responses,
       no cross-contamination of models.
```

---

### Task 6.3: Performance Budget Enforcement

**Goal:** Automated performance tracking that blocks deployment if budgets are exceeded.

**Performance Budgets:**

| Metric | Budget | Measurement |
|--------|--------|-------------|
| Simple primitive (sphere) evaluation, 64³ | < 50ms | Criterion bench |
| Complex scene (10 operations) evaluation, 64³ | < 200ms | Criterion bench |
| Marching cubes mesh extraction, 64³ | < 100ms | Criterion bench |
| Full pipeline (DSL → mesh JSON), 64³ | < 500ms | HTTP API test |
| Parameter modification → updated mesh | < 200ms | WebSocket test |
| STL export, 100K triangles | < 50ms | Criterion bench |
| Frontend render, 100K triangles | 60fps | Browser perf test |
| Memory, 128³ evaluation | < 500MB | RSS measurement |
| Python oracle speedup | ≥ 100x | Cross-language bench |

**Tests:**

```
[BENCHMARK] All performance budgets checked in CI.
            Any budget exceeded → CI fails → merge blocked.

[BENCHMARK] Historical tracking: performance numbers logged
            per commit. Dashboard shows trends.

[BENCHMARK] Comparison against Python oracle logged per commit.
            Speedup ratio must be ≥ 100x.
```

---

## Appendix A: Cross-Language Oracle Test Protocol

This protocol is used for every Rust function that has a Python equivalent in fogleman/sdf.

### Point Generation
```python
# generate_test_points.py
import numpy as np

def generate_test_points(n=10000, bounds=3.0, seed=42):
    """Deterministic test point generation."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-bounds, bounds, size=(n, 3))
    # Also include structured points:
    # - Origin
    # - Axis-aligned points at various distances
    # - Points on known surfaces
    structured = np.array([
        [0, 0, 0],           # Origin
        [1, 0, 0], [0, 1, 0], [0, 0, 1],  # Unit axes
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [0.5, 0.5, 0.5],     # Interior diagonal
        [2, 2, 2],           # Exterior diagonal
    ])
    return np.vstack([structured, points])
```

### Comparison Protocol
```python
# compare_results.py
import numpy as np

def compare(rust_values, python_values, tolerance=1e-12, name=""):
    """Compare SDF values from Rust and Python implementations."""
    diff = np.abs(rust_values - python_values)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    failures = np.sum(diff > tolerance)

    print(f"[{name}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
          f"failures={failures}/{len(diff)}")

    assert failures == 0, (
        f"{name}: {failures} points exceed tolerance {tolerance}. "
        f"Max diff: {max_diff:.2e} at point {points[np.argmax(diff)]}"
    )
```

### Golden File Management
- Oracle outputs are saved as `.npy` files in `tests/golden/`
- Golden files are regenerated when the oracle version changes
- CI verifies Rust output matches golden files (avoids needing Python in CI for fast runs)
- Full oracle comparison runs nightly

---

## Appendix B: Recommended Tool Configuration

### Claude Code / Codex Task Prompting Template

When delegating tasks to an AI coding agent, use this template:

```
TASK: [Task ID, e.g., "1.2.3 — Implement sphere primitive"]

CONTEXT:
- Working in crate: sdf-core
- File: src/primitives.rs
- Oracle reference: fogleman/sdf, file sdf/d3.py, function sphere()

SPECIFICATION:
[Exact mathematical formula or pseudocode]

CONSTRAINTS:
- Must match Python oracle to ε < 1e-12 at all test points
- Must pass cargo clippy with zero warnings
- Must include doc comments with mathematical definition
- Must include #[inline] attribute for evaluation function
- No unsafe code unless benchmarks prove necessity

TESTS TO WRITE:
- [List specific tests from the task]

ACCEPTANCE:
- cargo test passes
- cargo clippy passes
- oracle comparison script passes
- criterion benchmark shows < 50ns/point

DO NOT:
- Skip tests
- Use approximate formulas when exact formulas exist
- Add dependencies without justification
- Modify code outside the specified file without asking
```

### CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Clippy
        run: cargo clippy -- -D warnings
      - name: Tests
        run: cargo test --all
      - name: Golden file comparison
        run: cargo test --test golden_oracle
      - name: Benchmarks (no regression)
        run: cargo bench --bench sdf_bench -- --save-baseline current

  oracle-comparison:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -r oracle/requirements.txt
      - run: python oracle/full_comparison.py
```

---

## Appendix C: Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM generates syntactically valid but geometrically nonsensical models | Medium | Bounding box sanity checks, volume range validation, automatic retry with error context |
| Marching cubes produces non-manifold meshes for complex scenes | High | Mesh repair pass (edge collapse, hole filling), fallback to higher resolution |
| SDF DSL isn't expressive enough for user requests | High | Start with fogleman/sdf's full primitive set; add new primitives based on user request analysis |
| Performance regression introduced silently | Medium | CI benchmark gates, historical tracking, budget enforcement |
| Floating-point divergence between Rust and Python | Low | Use identical algorithms (not "equivalent"), test at high precision, document any intentional deviations |
| LLM API latency dominates interaction loop | Medium | Stream responses, begin mesh evaluation on partial DSL when possible, cache common patterns |
| 3D printing slicers reject exported meshes | High | Validate all exports with PrusaSlicer CLI in CI, maintain watertightness as hard invariant |

---

## Appendix D: Milestone Summary

| Milestone | Week | Deliverable | Key Metric |
|-----------|------|-------------|------------|
| M1: Engine Core | 4 | All primitives + operations + marching cubes | ≥ 100x speedup, all oracle tests pass |
| M2: DSL + API | 6 | Parseable DSL, HTTP + WebSocket API | < 200ms parameter update |
| M3: AI Integration | 10 | LLM generates valid DSL from natural language | > 95% parse success rate |
| M4: Frontend MVP | 12 | Chat + 3D viewer + export | Full user journey works |
| M5: Launch Ready | 14 | E2E tests pass, performance budgets met | 10 journeys pass, all budgets met |
