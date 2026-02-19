import test from 'node:test';
import assert from 'node:assert/strict';

import {
  boundingBox,
  computeSsim,
  flattenMesh,
  makeGridMesh,
  makeUvSphereMesh,
  rgbaToLuma,
  validateMesh
} from '../../src/mesh-utils.js';

test('uv sphere mesh is near 1000 triangles for 20x25 segments', () => {
  const mesh = makeUvSphereMesh(20, 20, 25);
  assert.equal(mesh.triangles.length, 1000);
  assert.equal(mesh.vertices.length, (20 + 1) * (25 + 1));

  const check = validateMesh(mesh);
  assert.equal(check.valid, true);
});

test('grid mesh benchmark fixture has >=100k triangles', () => {
  const mesh = makeGridMesh(225, 225, 200);
  assert.equal(mesh.triangles.length, 100352);
  assert.equal(mesh.vertices.length, 225 * 225);

  const flat = flattenMesh(mesh);
  assert.equal(flat.indices.length, mesh.triangles.length * 3);
  assert.equal(flat.positions.length, mesh.vertices.length * 3);
});

test('empty mesh validates and produces zero bounding box', () => {
  const mesh = { vertices: [], triangles: [] };
  const check = validateMesh(mesh);
  assert.equal(check.valid, true);

  const box = boundingBox(mesh.vertices);
  assert.deepEqual(box.size, [0, 0, 0]);
});

test('ssim behaves as expected on identical and perturbed data', () => {
  const rgbaA = new Uint8Array([20, 30, 50, 255, 200, 180, 170, 255]);
  const rgbaB = new Uint8Array([20, 30, 50, 255, 200, 180, 170, 255]);
  const rgbaC = new Uint8Array([220, 220, 220, 255, 5, 10, 15, 255]);

  const lumaA = rgbaToLuma(rgbaA);
  const lumaB = rgbaToLuma(rgbaB);
  const lumaC = rgbaToLuma(rgbaC);

  assert.ok(computeSsim(lumaA, lumaB) > 0.99);
  assert.ok(computeSsim(lumaA, lumaC) < 0.9);
});

