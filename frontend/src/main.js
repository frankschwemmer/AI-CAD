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
        if (payload.message.startsWith('AI Retry')) {
          appendAssistantMessage(payload.message);
        } else {
          finishAssistantMessage(`Error: ${payload.message}`);
        }
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

function appendAssistantMessage(text) {
  if (!currentAssistantBubble) {
    addAssistantTyping();
  }

  // Remove typing indicator if present
  const typing = currentAssistantBubble.querySelector('.loading-dots');
  if (typing) {
    typing.remove();
    currentAssistantBubble.classList.remove('typing');
  }

  const block = document.createElement('div');
  block.style.marginTop = '8px';
  block.style.fontSize = '0.85em';
  block.style.color = 'var(--text-muted)';
  block.innerHTML = escapeHtml(text).replace(/\n/g, '<br/>');
  currentAssistantBubble.appendChild(block);

  // Re-add typing indicator at the bottom
  const newTyping = document.createElement('div');
  newTyping.className = 'loading-dots';
  newTyping.style.marginTop = '8px';
  newTyping.innerHTML = '<div></div><div></div><div></div>';
  currentAssistantBubble.appendChild(newTyping);

  scrollToBottom();
}

function finishAssistantMessage(text) {
  if (currentAssistantBubble) {
    const typing = currentAssistantBubble.querySelector('.loading-dots');
    if (typing) typing.remove();

    currentAssistantBubble.classList.remove('typing');

    const block = document.createElement('div');
    block.style.marginTop = currentAssistantBubble.children.length > 0 ? '8px' : '0';
    block.style.color = 'var(--text)';
    block.innerHTML = escapeHtml(text).replace(/\n/g, '<br/>');
    currentAssistantBubble.appendChild(block);

    currentAssistantBubble = null;
    scrollToBottom();
  } else {
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

function handleInput() {
  const prompt = nodes.chatInput.value.trim();
  if (!prompt) return;

  nodes.chatInput.value = '';
  addUserMessage(prompt);
  addAssistantTyping();

  if (!socket || socket.readyState !== WebSocket.OPEN) {
    finishAssistantMessage("Error: Not connected to the backend server.");
    return;
  }

  const payload = {
    type: 'generate',
    prompt: prompt,
    resolution: 64
  };
  socket.send(JSON.stringify(payload));
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
