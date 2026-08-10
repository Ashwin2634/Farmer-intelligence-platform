/* ─────────────────────────────────────────────────────────────────────────────
   AlexxaFarms — Frontend Logic
   Handles Tabs, Drag & Drop, Validation, and FastAPI Endpoints Integration
───────────────────────────────────────────────────────────────────────────── */

const API_BASE = 'http://localhost:8000';

// ── DOM References ─────────────────────────────────────────────────────────
const apiStatusEl   = document.getElementById('api-status');

// Tabs
const tabBtns       = document.querySelectorAll('.tab-btn');
const tabPanels     = document.querySelectorAll('.tab-panel');

// Disease Detection
const dropzone      = document.getElementById('dropzone');
const fileInput     = document.getElementById('file-input');
const browseBtn     = document.getElementById('browse-btn');
const previewBox    = document.getElementById('preview-box');
const previewImg    = document.getElementById('preview-img');
const fileMeta      = document.getElementById('file-meta');
const fileNameEl    = document.getElementById('file-name');
const clearBtn      = document.getElementById('clear-btn');
const detectBtn     = document.getElementById('detect-btn');
const detectBtnText = document.getElementById('detect-btn-text');
const detectSpinner = document.getElementById('detect-spinner');
const diseaseEmpty  = document.getElementById('disease-empty');
const diseaseResult = document.getElementById('disease-result');

// Crop Recommendation
const cropForm       = document.getElementById('crop-form');
const categoryPills  = document.querySelectorAll('.category-pills .pill');
const categoryInput  = document.getElementById('category');
const recommendBtn   = document.getElementById('recommend-btn');
const recommendText  = document.getElementById('recommend-btn-text');
const recommendSpin  = document.getElementById('recommend-spinner');
const cropEmpty      = document.getElementById('crop-empty');
const cropResult     = document.getElementById('crop-result');

// Toast
const toastContainer = document.getElementById('toast-container');

// State
let selectedFile = null;

// ── TAB SWITCHING ─────────────────────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const targetTab = btn.dataset.tab;
    
    tabBtns.forEach(b => b.classList.remove('active'));
    tabPanels.forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    document.getElementById(`panel-${targetTab}`).classList.add('active');
  });
});

// ── TOAST NOTIFICATIONS ───────────────────────────────────────────────────
function showToast(message, duration = 3500) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, duration);
}

// ── API HEALTH CHECK ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      apiStatusEl.className = 'status-pill online';
      apiStatusEl.innerHTML = '<span class="status-dot"></span> API Connected';
    } else {
      throw new Error();
    }
  } catch {
    apiStatusEl.className = 'status-pill offline';
    apiStatusEl.innerHTML = '<span class="status-dot"></span> API Disconnected';
  }
}
checkHealth();
setInterval(checkHealth, 20000);

// ── DISEASE DETECTION LOGIC ───────────────────────────────────────────────

// File Selection & Drag-and-Drop
['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    handleFile(file);
  } else {
    showToast('Please select a valid image file');
  }
});

dropzone.addEventListener('click', e => {
  if (e.target === clearBtn) return;
  if (!selectedFile) fileInput.click();
});

browseBtn.addEventListener('click', e => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
  fileInput.value = '';
});

function handleFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewBox.classList.remove('hidden');
  fileMeta.classList.remove('hidden');
  fileNameEl.textContent = file.name;
  detectBtn.disabled = false;

  diseaseResult.classList.add('hidden');
  diseaseEmpty.classList.remove('hidden');
}

clearBtn.addEventListener('click', e => {
  e.stopPropagation();
  resetDiseaseForm();
});

function resetDiseaseForm() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewBox.classList.add('hidden');
  fileMeta.classList.add('hidden');
  detectBtn.disabled = true;
  diseaseResult.classList.add('hidden');
  diseaseEmpty.classList.remove('hidden');
}

detectBtn.addEventListener('click', runDiseaseDetection);

async function runDiseaseDetection() {
  if (!selectedFile) return;

  detectBtn.disabled = true;
  detectBtnText.textContent = 'Analyzing Leaf...';
  detectSpinner.classList.remove('hidden');

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const res = await fetch(`${API_BASE}/disease-detection/detect`, {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Analysis failed');

    renderDiseaseResult(data);
    showToast('Diagnosis report generated successfully');
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    detectBtn.disabled = false;
    detectBtnText.textContent = 'Run Disease Diagnosis';
    detectSpinner.classList.add('hidden');
  }
}

function renderDiseaseResult(data) {
  const isHealthy = data.status === 'healthy';
  const confidence = data.confidence ? (data.confidence * 100).toFixed(1) : null;
  const detections = data.detections || [];
  const cropName = data.crop || (detections.length > 0 ? detections[0].crop : null);

  let html = `
    <div class="result-header-card ${isHealthy ? 'healthy' : 'diseased'}">
      <div class="status-icon">${isHealthy ? '🌿' : '⚠️'}</div>
      <div>
        <div class="title">${isHealthy ? 'Healthy Plant Leaf' : escapeHtml(data.diagnosis || 'Disease Detected')}</div>
        <div class="sub">${isHealthy ? 'No disease lesions observed' : `${detections.length} symptom area(s) detected`}</div>
      </div>
    </div>

    <div class="stats-grid">
      <div class="stat-box">
        <div class="lbl">Diagnosis Status</div>
        <div class="val">${isHealthy ? 'Healthy' : 'Diseased'}</div>
      </div>
      ${!isHealthy && cropName ? `
      <div class="stat-box">
        <div class="lbl">Crop</div>
        <div class="val">${escapeHtml(cropName)}</div>
      </div>
      ` : ''}
      <div class="stat-box">
        <div class="lbl">Inference Time</div>
        <div class="val">${data.inference_time_ms ? data.inference_time_ms.toFixed(1) + ' ms' : 'N/A'}</div>
      </div>
      <div class="stat-box">
        <div class="lbl">Model</div>
        <div class="val">${escapeHtml(data.model || 'YOLO11')}</div>
      </div>
      <div class="stat-box">
        <div class="lbl">Confidence</div>
        <div class="val">${confidence ? confidence + '%' : '100%'}</div>
      </div>
    </div>
  `;

  if (detections.length > 0) {
    html += `<div style="margin-top:0.5rem"><span class="input-label">Detected Disease Lesions</span>`;
    detections.forEach((d) => {
      const pct = (d.confidence * 100).toFixed(1);
      html += `
        <div class="detection-card" style="margin-top:6px">
          <div>
            <div class="name">${escapeHtml(d.crop || '')} - ${escapeHtml(d.disease || '')}</div>
          </div>
          <span class="badge">${pct}% Conf.</span>
        </div>
      `;
    });
    html += `</div>`;
  }

  diseaseResult.innerHTML = html;
  diseaseResult.classList.remove('hidden');
  diseaseEmpty.classList.add('hidden');
}

// ── CROP RECOMMENDATION LOGIC ─────────────────────────────────────────────

categoryPills.forEach(pill => {
  pill.addEventListener('click', () => {
    categoryPills.forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    categoryInput.value = pill.dataset.value;
  });
});

cropForm.addEventListener('submit', async e => {
  e.preventDefault();

  const n = parseFloat(document.getElementById('nitrogen').value);
  const p = parseFloat(document.getElementById('phosphorus').value);
  const k = parseFloat(document.getElementById('potassium').value);
  const temp = parseFloat(document.getElementById('temperature').value);
  const hum = parseFloat(document.getElementById('humidity').value);
  const phVal = parseFloat(document.getElementById('ph').value);

  if ([n, p, k, temp, hum, phVal].some(val => isNaN(val))) {
    showToast('Please fill all soil and climate parameters.');
    return;
  }

  recommendBtn.disabled = true;
  recommendText.textContent = 'Calculating...';
  recommendSpin.classList.remove('hidden');

  try {
    const payload = {
      nitrogen: n,
      phosphorus: p,
      potassium: k,
      temperature: temp,
      humidity: hum,
      ph: phVal
    };

    if (categoryInput.value) {
      payload.category = categoryInput.value;
    }

    const res = await fetch(`${API_BASE}/crop-recommendation/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Prediction failed');

    renderCropResult(data.recommendations || []);
    showToast('Crop recommendations generated!');
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    recommendBtn.disabled = false;
    recommendText.textContent = 'Predict Optimal Crops';
    recommendSpin.classList.add('hidden');
  }
});

function renderCropResult(recs) {
  if (recs.length === 0) {
    cropResult.innerHTML = '<p class="sub-text" style="text-align:center;padding:2rem">No crops found matching these criteria.</p>';
    cropResult.classList.remove('hidden');
    cropEmpty.classList.add('hidden');
    return;
  }

  let html = '';
  recs.forEach((rec, idx) => {
    const rankClass = idx === 0 ? 'rank-1' : '';
    html += `
      <div class="crop-item ${rankClass}">
        <div class="crop-rank">#${idx + 1}</div>
        <div class="crop-details">
          <div class="crop-name">${escapeHtml(rec.crop.replace(/_/g, ' '))}</div>
          <span class="crop-tag">${escapeHtml(rec.category)}</span>
        </div>
        <div class="crop-confidence">
          <div class="pct">${rec.confidence.toFixed(1)}%</div>
          <div class="lbl">Suitability</div>
        </div>
      </div>
    `;
  });

  cropResult.innerHTML = html;
  cropResult.classList.remove('hidden');
  cropEmpty.classList.add('hidden');
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = String(str || '');
  return d.innerHTML;
}
