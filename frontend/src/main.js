import { makeUvSphereMesh } from './mesh-utils.js';
import { SdfViewer } from './viewer.js';

const canvas = document.getElementById('stage');
const viewer = new SdfViewer(canvas);

const nodes = {
  wireframe: document.getElementById('wireframe'),
  measure: document.getElementById('measure-output'),
  bbox: document.getElementById('bbox-output'),
  wsUrl: document.getElementById('ws-url'),
  wsStatus: document.getElementById('ws-status'),
  dsl: document.getElementById('dsl'),
  resolution: document.getElementById('resolution'),
  bench: document.getElementById('bench-output'),
  benchRun: document.getElementById('run-benchmark'),
  
  // New chat nodes
  chatHistory: document.getElementById('chat-history'),
  chatInput: document.getElementById('chat-input'),
  chatSend: document.getElementById('chat-send')
};

let socket = null;

function setStatus(text, stateClass) {
  nodes.wsStatus.textContent = text.toUpperCase();
  nodes.wsStatus.className = `status-badge ${stateClass}`;
}

function updateBboxDisplay(bounds) {
  nodes.bbox.textContent = `BBox: ${bounds.size.map((n) => n.toFixed(1)).join(' x ')} mm`;
}

function loadMeshAndReport(mesh) {
  const info = viewer.loadMesh(mesh);
  updateBboxDisplay(info.bbox);
  return info;
}

nodes.wireframe.addEventListener('change', () => {
  viewer.setWireframe(nodes.wireframe.checked);
});

canvas.addEventListener('click', () => {
  const distance = viewer.readCurrentDistance();
  nodes.measure.textContent = distance == null ? 'Dist: --' : `Dist: ${distance.toFixed(2)}mm`;
});

nodes.benchRun.addEventListener('click', async () => {
  nodes.bench.textContent = 'Benchmarking...';
  const result = await viewer.benchmark100kTriangles(120);
  nodes.bench.textContent = `${result.avgFrameMs.toFixed(1)}ms | ${result.fps.toFixed(0)}fps`;
});

// Auto-connect WS
function connectWebSocket() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }

  socket = new WebSocket(nodes.wsUrl.value.trim());
  setStatus('connecting', 'connecting');

  socket.addEventListener('open', () => {
    setStatus('connected', 'connected');
  });

  socket.addEventListener('close', () => {
    setStatus('disconnected', 'disconnected');
    setTimeout(connectWebSocket, 3000); // auto reconnect
  });

  socket.addEventListener('error', () => {
    setStatus('error', 'error');
  });

  socket.addEventListener('message', (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === 'mesh') {
        const info = loadMeshAndReport({
          vertices: payload.vertices,
          triangles: payload.triangles
        });
        finishAssistantMessage(`Generated mesh in ${Number(payload.time_ms).toFixed(1)}ms with ${info.triangles} triangles.`);
      } else if (payload.type === 'error') {
        finishAssistantMessage(`Error: ${payload.message}`);
      }
    } catch (err) {
      console.error(err);
    }
  });
}

// --- SIMULATED AI CHAT LOGIC ---

function addUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.innerHTML = `
    <div class="avatar">U</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  nodes.chatHistory.appendChild(msg);
  scrollToBottom();
}

let currentAssistantBubble = null;

function addAssistantTyping() {
  const msg = document.createElement('div');
  msg.className = 'message assistant';
  msg.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble typing">
      <div class="loading-dots"><div></div><div></div><div></div></div>
    </div>
  `;
  nodes.chatHistory.appendChild(msg);
  currentAssistantBubble = msg.querySelector('.bubble');
  scrollToBottom();
}

function finishAssistantMessage(text) {
  if (currentAssistantBubble) {
    currentAssistantBubble.classList.remove('typing');
    currentAssistantBubble.innerHTML = escapeHtml(text).replace(/\n/g, '<br/>');
    currentAssistantBubble = null;
    scrollToBottom();
  } else {
    // If no typing bubble exists, create a new one
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.innerHTML = `
      <div class="avatar">AI</div>
      <div class="bubble">${escapeHtml(text).replace(/\n/g, '<br/>')}</div>
    `;
    nodes.chatHistory.appendChild(msg);
    scrollToBottom();
  }
}

function scrollToBottom() {
  nodes.chatHistory.scrollTop = nodes.chatHistory.scrollHeight;
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }

function parseUserPrompt(prompt) {
  const text = prompt.toLowerCase();
  
  // Very naive pattern matching for demo purposes
  
  if (text.includes("box") || text.includes("cube")) {
    let size = 50; // default
    const match = text.match(/(\\d+)mm/);
    if (match) size = parseInt(match[1]);
    
    return `// AI Generated Box
params {
  size = ${size}mm
}
result = box(size/2, size/2, size/2)`;

  } else if (text.includes("cylinder")) {
    return `// AI Generated Cylinder
params {
  radius = 20mm
  height = 80mm
}
result = cylinder(radius, height)`;

  } else if (text.includes("capsule")) {
    return `// AI Generated Capsule
params {
  r = 15mm
  len = 60mm
}
result = capsule(0, 0, 0, 0, len, 0, r)`;

  } else if (text.includes("torus")) {
    return `// AI Generated Torus
params {
  major = 40mm
  minor = 10mm
}
result = torus(major, minor)`;
  
  } else if (text.includes("union") || text.includes("combine")) {
    return `// AI Generated Union
params {
  r = 25mm
  width = 40mm
}
a = sphere(r)
b = box(width/2, width/2, width/2)
result = smooth_union(a, translate(b, 20, 0, 0), 0.2)`;
  } else {
    // Default fallback
    let size = 30; // default
    const match = text.match(/(\\d+)mm/);
    if (match) size = parseInt(match[1]);
    
    return `// AI Generated Sphere
params {
  radius = ${size}mm
}
result = sphere(radius)`;
  }
}

function handleInput() {
  const prompt = nodes.chatInput.value.trim();
  if (!prompt) return;
  
  nodes.chatInput.value = '';
  addUserMessage(prompt);
  addAssistantTyping();
  
  // Simulate AI latency
  setTimeout(() => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      finishAssistantMessage("Error: Not connected to the backend server.");
      return;
    }
    
    const dsl = parseUserPrompt(prompt);
    
    const payload = {
      type: 'set_dsl',
      dsl: dsl,
      resolution: 64
    };
    socket.send(JSON.stringify(payload));
    
  }, 600);
}

nodes.chatSend.addEventListener('click', handleInput);
nodes.chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleInput();
  }
});

// Initialization
const initial = makeUvSphereMesh(20, 20, 25);
loadMeshAndReport(initial);
connectWebSocket();

window.__viewer = viewer;
