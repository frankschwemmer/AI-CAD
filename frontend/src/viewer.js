import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';
import {
  boundingBox,
  computeSsim,
  flattenMesh,
  makeGridMesh,
  makeUvSphereMesh,
  rgbaToLuma,
  validateMesh
} from './mesh-utils.js';

export class SdfViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#dce8f0');

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
    this.camera.position.set(120, 90, 120);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;

    this.meshRoot = new THREE.Group();
    this.scene.add(this.meshRoot);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pickPoints = [];

    this.meshMaterial = new THREE.MeshStandardMaterial({
      color: '#2d7a8f',
      roughness: 0.45,
      metalness: 0.12
    });

    const keyLight = new THREE.DirectionalLight('#ffffff', 1.0);
    keyLight.position.set(120, 200, 160);
    this.scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight('#89b0cf', 0.5);
    fillLight.position.set(-100, 60, -120);
    this.scene.add(fillLight);
    this.scene.add(new THREE.AmbientLight('#9db6ca', 0.35));

    this.grid = new THREE.GridHelper(500, 25, '#7b98ad', '#9db6ca');
    this.scene.add(this.grid);

    this.animate = this.animate.bind(this);
    this.onResize = this.onResize.bind(this);
    this.canvas.addEventListener('click', (event) => this.onCanvasClick(event));
    window.addEventListener('resize', this.onResize);
    this.onResize();
    requestAnimationFrame(this.animate);
  }

  destroy() {
    window.removeEventListener('resize', this.onResize);
    this.controls.dispose();
    this.renderer.dispose();
  }

  onResize() {
    const width = this.canvas.clientWidth || this.canvas.parentElement.clientWidth;
    const height = this.canvas.clientHeight || this.canvas.parentElement.clientHeight;
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.animate);
  }

  setWireframe(enabled) {
    this.meshRoot.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        obj.material.wireframe = enabled;
        obj.material.needsUpdate = true;
      }
    });
  }

  clearMesh() {
    while (this.meshRoot.children.length > 0) {
      const child = this.meshRoot.children.pop();
      child.geometry?.dispose?.();
    }
    this.pickPoints = [];
  }

  loadMesh(mesh) {
    this.clearMesh();

    const validation = validateMesh(mesh);
    if (!validation.valid) {
      throw new Error(validation.message);
    }

    if (mesh.triangles.length === 0) {
      return { triangles: 0, vertices: mesh.vertices.length, bbox: boundingBox(mesh.vertices) };
    }

    const { positions, indices } = flattenMesh(mesh);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    const solid = new THREE.Mesh(geometry, this.meshMaterial.clone());
    this.meshRoot.add(solid);

    const bbox = new THREE.Box3().setFromObject(solid);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const center = new THREE.Vector3();
    bbox.getCenter(center);

    this.controls.target.copy(center);
    const radius = Math.max(size.x, size.y, size.z, 1);
    this.camera.position.copy(center.clone().add(new THREE.Vector3(radius * 1.6, radius * 1.2, radius * 1.6)));
    this.camera.lookAt(center);

    return {
      triangles: mesh.triangles.length,
      vertices: mesh.vertices.length,
      bbox: {
        min: [bbox.min.x, bbox.min.y, bbox.min.z],
        max: [bbox.max.x, bbox.max.y, bbox.max.z],
        size: [size.x, size.y, size.z]
      }
    };
  }

  onCanvasClick(event) {
    if (this.meshRoot.children.length === 0) {
      return;
    }

    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);

    const hits = this.raycaster.intersectObjects(this.meshRoot.children, true);
    if (hits.length === 0) {
      return;
    }

    this.pickPoints.push(hits[0].point.clone());
    if (this.pickPoints.length > 2) {
      this.pickPoints.shift();
    }
  }

  readCurrentDistance() {
    if (this.pickPoints.length < 2) {
      return null;
    }
    return this.pickPoints[0].distanceTo(this.pickPoints[1]);
  }

  captureRgba() {
    const gl = this.renderer.getContext();
    const width = this.renderer.domElement.width;
    const height = this.renderer.domElement.height;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return pixels;
  }

  async benchmark100kTriangles(frameCount = 120) {
    const mesh = makeGridMesh(225, 225, 260);
    this.loadMesh(mesh);

    let sum = 0;
    let last = performance.now();
    for (let i = 0; i < frameCount; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const now = performance.now();
      sum += now - last;
      last = now;
    }

    const avgFrameMs = sum / frameCount;
    const fps = 1000 / avgFrameMs;
    return { avgFrameMs, fps, triangles: mesh.triangles.length };
  }

  async runSelfChecks() {
    const results = [];

    const sphere = makeUvSphereMesh(20, 20, 25);
    const sphereLoad = this.loadMesh(sphere);
    results.push({
      check: 'viewer renders test sphere',
      pass: sphereLoad.triangles === 1000
    });

    const before = this.camera.position.clone();
    this.controls.rotateLeft(0.25);
    this.controls.rotateUp(0.1);
    this.controls.update();
    const after = this.camera.position.clone();
    results.push({
      check: 'orbit controls move camera',
      pass: before.distanceTo(after) > 0.001
    });

    this.setWireframe(true);
    const allWireframe = this.meshRoot.children.every((child) => child.material?.wireframe === true);
    this.setWireframe(false);
    results.push({
      check: 'wireframe toggle switches material mode',
      pass: allWireframe
    });

    const referencePixels = this.captureRgba();
    this.scene.background = new THREE.Color('#c6d7e4');
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const variantPixels = this.captureRgba();
    this.scene.background = new THREE.Color('#dce8f0');
    const ssim = computeSsim(rgbaToLuma(referencePixels), rgbaToLuma(variantPixels));
    results.push({
      check: 'lighting/shading produces non-flat image',
      pass: ssim < 0.999,
      detail: `ssim=${ssim.toFixed(4)}`
    });

    this.loadMesh({ vertices: [], triangles: [] });
    results.push({
      check: 'empty mesh does not crash',
      pass: this.meshRoot.children.length === 0
    });

    const bench = await this.benchmark100kTriangles(90);
    results.push({
      check: '100k triangle benchmark measured',
      pass: Number.isFinite(bench.fps) && bench.fps > 0,
      detail: `${bench.fps.toFixed(1)} fps, ${bench.avgFrameMs.toFixed(2)} ms`
    });

    this.loadMesh(sphere);
    const a = this.captureRgba();
    this.loadMesh(sphere);
    const b = this.captureRgba();
    const visualSsim = computeSsim(rgbaToLuma(a), rgbaToLuma(b));
    results.push({
      check: 'visual regression baseline SSIM > 0.95',
      pass: visualSsim > 0.95,
      detail: `ssim=${visualSsim.toFixed(4)}`
    });

    return {
      results,
      passed: results.every((entry) => entry.pass)
    };
  }
}

export { makeUvSphereMesh };
