import { makeUvSphereMesh } from './mesh-utils.js';
import { SdfViewer } from './viewer.js';

const canvas = document.getElementById('stage');
const viewer = new SdfViewer(canvas);

const nodes = {
  wireframe: document.getElementById('wireframe'),
  loadSphere: document.getElementById('load-sphere'),
  loadEmpty: document.getElementById('load-empty'),
  measure: document.getElementById('measure-output'),
  bbox: document.getElementById('bbox-output'),
  wsUrl: document.getElementById('ws-url'),
  wsConnect: document.getElementById('ws-connect'),
  wsSend: document.getElementById('ws-send'),
  wsStatus: document.getElementById('ws-status'),
  dsl: document.getElementById('dsl'),
  resolution: document.getElementById('resolution'),
  bench: document.getElementById('bench-output'),
  benchRun: document.getElementById('run-benchmark'),
  runTests: document.getElementById('run-self-tests'),
  testOutput: document.getElementById('test-output')
};

let socket = null;

function setStatus(text) {
  nodes.wsStatus.textContent = `WS: ${text}`;
}

function updateBboxDisplay(bounds) {
  nodes.bbox.textContent = `BBox: ${bounds.size.map((n) => n.toFixed(2)).join(' x ')} mm`;
}

function loadMeshAndReport(mesh) {
  const info = viewer.loadMesh(mesh);
  updateBboxDisplay(info.bbox);
  return info;
}

nodes.loadSphere.addEventListener('click', () => {
  const mesh = makeUvSphereMesh(20, 20, 25);
  const info = loadMeshAndReport(mesh);
  nodes.measure.textContent = `Distance: -- (tris=${info.triangles})`;
});

nodes.loadEmpty.addEventListener('click', () => {
  const info = loadMeshAndReport({ vertices: [], triangles: [] });
  nodes.measure.textContent = `Distance: -- (tris=${info.triangles})`;
});

nodes.wireframe.addEventListener('change', () => {
  viewer.setWireframe(nodes.wireframe.checked);
});

function connectWebSocket() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }

  socket = new WebSocket(nodes.wsUrl.value.trim());
  setStatus('connecting');

  socket.addEventListener('open', () => {
    setStatus('connected');
  });

  socket.addEventListener('close', () => {
    setStatus('disconnected');
  });

  socket.addEventListener('error', () => {
    setStatus('error');
  });

  socket.addEventListener('message', (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === 'mesh') {
        const info = loadMeshAndReport({
          vertices: payload.vertices,
          triangles: payload.triangles
        });
        nodes.measure.textContent = `Distance: -- (tris=${info.triangles}, time=${Number(payload.time_ms).toFixed(2)}ms)`;
      } else if (payload.type === 'error') {
        setStatus(`server error - ${payload.message}`);
      }
    } catch (err) {
      setStatus(`message parse error - ${err.message}`);
    }
  });
}

nodes.wsConnect.addEventListener('click', connectWebSocket);

nodes.wsSend.addEventListener('click', () => {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    setStatus('connect first');
    return;
  }

  const resolution = Number.parseInt(nodes.resolution.value, 10);
  const payload = {
    type: 'set_dsl',
    dsl: nodes.dsl.value,
    resolution
  };
  socket.send(JSON.stringify(payload));
});

nodes.benchRun.addEventListener('click', async () => {
  nodes.bench.textContent = 'Running benchmark...';
  const result = await viewer.benchmark100kTriangles(120);
  nodes.bench.textContent = `Frame: ${result.avgFrameMs.toFixed(2)} ms | FPS: ${result.fps.toFixed(1)} | triangles=${result.triangles}`;
});

nodes.runTests.addEventListener('click', async () => {
  nodes.testOutput.textContent = 'Running checks...';
  const result = await viewer.runSelfChecks();
  const lines = result.results.map((entry) => {
    const status = entry.pass ? 'PASS' : 'FAIL';
    return `[${status}] ${entry.check}${entry.detail ? ` (${entry.detail})` : ''}`;
  });
  lines.push(result.passed ? '\nAll viewer checks passed.' : '\nOne or more checks failed.');
  nodes.testOutput.textContent = lines.join('\n');
  window.__viewerTestResults = result;
});

canvas.addEventListener('click', () => {
  const distance = viewer.readCurrentDistance();
  nodes.measure.textContent = distance == null ? 'Distance: --' : `Distance: ${distance.toFixed(2)} mm`;
});

const initial = makeUvSphereMesh(20, 20, 25);
loadMeshAndReport(initial);
setStatus('disconnected');

window.__viewer = viewer;
