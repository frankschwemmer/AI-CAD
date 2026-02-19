export function makeUvSphereMesh(radius = 20, latSegments = 20, lonSegments = 25) {
  const vertices = [];
  const triangles = [];

  for (let y = 0; y <= latSegments; y += 1) {
    const v = y / latSegments;
    const phi = v * Math.PI;
    for (let x = 0; x <= lonSegments; x += 1) {
      const u = x / lonSegments;
      const theta = u * Math.PI * 2;
      const px = radius * Math.sin(phi) * Math.cos(theta);
      const py = radius * Math.cos(phi);
      const pz = radius * Math.sin(phi) * Math.sin(theta);
      vertices.push([px, py, pz]);
    }
  }

  const row = lonSegments + 1;
  for (let y = 0; y < latSegments; y += 1) {
    for (let x = 0; x < lonSegments; x += 1) {
      const a = y * row + x;
      const b = a + 1;
      const c = a + row;
      const d = c + 1;
      triangles.push([a, c, b]);
      triangles.push([b, c, d]);
    }
  }

  return { vertices, triangles };
}

export function makeGridMesh(cols = 225, rows = 225, scale = 200) {
  const vertices = [];
  const triangles = [];

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const u = x / (cols - 1);
      const v = y / (rows - 1);
      const px = (u - 0.5) * scale;
      const py = (v - 0.5) * scale;
      const pz = Math.sin(u * Math.PI * 6) * Math.cos(v * Math.PI * 6) * 4;
      vertices.push([px, py, pz]);
    }
  }

  for (let y = 0; y < rows - 1; y += 1) {
    for (let x = 0; x < cols - 1; x += 1) {
      const i = y * cols + x;
      const right = i + 1;
      const down = i + cols;
      const downRight = down + 1;
      triangles.push([i, down, right]);
      triangles.push([right, down, downRight]);
    }
  }

  return { vertices, triangles };
}

export function validateMesh(mesh) {
  const result = {
    valid: true,
    message: 'ok'
  };

  if (!mesh || !Array.isArray(mesh.vertices) || !Array.isArray(mesh.triangles)) {
    return { valid: false, message: 'mesh requires vertices and triangles arrays' };
  }

  for (const tri of mesh.triangles) {
    if (!Array.isArray(tri) || tri.length !== 3) {
      return { valid: false, message: 'triangle must contain 3 indices' };
    }
    if (tri.some((idx) => idx < 0 || idx >= mesh.vertices.length)) {
      return { valid: false, message: 'triangle index out of range' };
    }
  }

  return result;
}

export function flattenMesh(mesh) {
  const positions = new Float32Array(mesh.vertices.length * 3);
  for (let i = 0; i < mesh.vertices.length; i += 1) {
    positions[i * 3] = mesh.vertices[i][0];
    positions[i * 3 + 1] = mesh.vertices[i][1];
    positions[i * 3 + 2] = mesh.vertices[i][2];
  }

  const indices = new Uint32Array(mesh.triangles.length * 3);
  for (let i = 0; i < mesh.triangles.length; i += 1) {
    indices[i * 3] = mesh.triangles[i][0];
    indices[i * 3 + 1] = mesh.triangles[i][1];
    indices[i * 3 + 2] = mesh.triangles[i][2];
  }

  return { positions, indices };
}

export function boundingBox(vertices) {
  if (!vertices.length) {
    return {
      min: [0, 0, 0],
      max: [0, 0, 0],
      size: [0, 0, 0]
    };
  }

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const [x, y, z] of vertices) {
    min[0] = Math.min(min[0], x);
    min[1] = Math.min(min[1], y);
    min[2] = Math.min(min[2], z);
    max[0] = Math.max(max[0], x);
    max[1] = Math.max(max[1], y);
    max[2] = Math.max(max[2], z);
  }

  return {
    min,
    max,
    size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]]
  };
}

export function computeSsim(lumaA, lumaB) {
  if (lumaA.length !== lumaB.length || lumaA.length === 0) {
    return 0;
  }

  const n = lumaA.length;
  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < n; i += 1) {
    meanA += lumaA[i];
    meanB += lumaB[i];
  }
  meanA /= n;
  meanB /= n;

  let varA = 0;
  let varB = 0;
  let covar = 0;
  for (let i = 0; i < n; i += 1) {
    const da = lumaA[i] - meanA;
    const db = lumaB[i] - meanB;
    varA += da * da;
    varB += db * db;
    covar += da * db;
  }

  const denom = Math.max(1, n - 1);
  varA /= denom;
  varB /= denom;
  covar /= denom;

  const c1 = (0.01 * 255) ** 2;
  const c2 = (0.03 * 255) ** 2;
  const num = (2 * meanA * meanB + c1) * (2 * covar + c2);
  const den = (meanA ** 2 + meanB ** 2 + c1) * (varA + varB + c2);
  if (den === 0) {
    return 0;
  }
  return num / den;
}

export function rgbaToLuma(rgba) {
  const out = new Float64Array(rgba.length / 4);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    out[p] = rgba[i] * 0.2126 + rgba[i + 1] * 0.7152 + rgba[i + 2] * 0.0722;
  }
  return out;
}
