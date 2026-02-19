# Frontend Viewer (Phase 5.1)

This directory contains the Three.js mesh viewer from Phase 5.1.

## Run

```bash
cd /Users/homeuse/Code/06\ AI\ CAD/frontend
npm run serve
```

Open `http://127.0.0.1:4173`.

## Features implemented

- Three.js viewer with OrbitControls.
- Wireframe toggle.
- Mesh measurement (point-to-point click distance + bounding box readout).
- WebSocket integration with `/ws` (set_dsl -> mesh render).
- Empty mesh handling (no crash).
- 100k-triangle benchmark (`Run 100k Triangle Benchmark`).
- Self-check runner (`Run Viewer Checks`) including baseline visual SSIM check.

## Unit tests

```bash
cd /Users/homeuse/Code/06\ AI\ CAD/frontend
npm test
```

Covers:

- sphere fixture (~1000 triangles)
- benchmark mesh fixture (100k+ triangles)
- empty mesh behavior
- SSIM utility behavior

## WebSocket quick check

1. Start backend server:

```bash
cd /Users/homeuse/Code/06\ AI\ CAD
cargo run -p sdf-server
```

2. In the viewer, click `Connect`, then `Send set_dsl`.
3. Mesh updates from server should render immediately.
