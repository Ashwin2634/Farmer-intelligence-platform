/* ─────────────────────────────────────────────────────────────────────────────
   AlexxaFarms — Frontend Logic
   Handles Tabs, Drag & Drop, Validation, and FastAPI Endpoints Integration
───────────────────────────────────────────────────────────────────────────── */

const API_BASE = 'http://192.168.1.3:8000';

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
const previewVideo  = document.getElementById('preview-video');
const fileIcon      = document.getElementById('file-icon');
const videoOptions  = document.getElementById('video-options');
const numFramesSel  = document.getElementById('num-frames');
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
let previewUrl = null;

const VIDEO_EXTS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp', '.m4v'];
const MAX_VIDEO_MB = 100;

function isImageFile(file) {
  return !!file.type && file.type.startsWith('image/');
}
function isVideoFile(file) {
  const name = (file.name || '').toLowerCase();
  return (!!file.type && file.type.startsWith('video/')) || VIDEO_EXTS.some(ext => name.endsWith(ext));
}
function prettyClass(name) {
  return String(name || '').replace(/___/g, ' - ').replace(/_/g, ' ');
}

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
  if (file && (isImageFile(file) || isVideoFile(file))) {
    handleFile(file);
  } else {
    showToast('Please select a valid image or video file');
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
  const video = isVideoFile(file);

  if (!video && !isImageFile(file)) {
    showToast('Please select a valid image or video file');
    return;
  }
  if (video && file.size > MAX_VIDEO_MB * 1024 * 1024) {
    showToast(`Video is too large. Maximum size is ${MAX_VIDEO_MB} MB.`);
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);

  if (video) {
    previewImg.classList.add('hidden');
    previewImg.removeAttribute('src');
    previewVideo.src = previewUrl;
    previewVideo.classList.remove('hidden');
  } else {
    previewVideo.pause();
    previewVideo.classList.add('hidden');
    previewVideo.removeAttribute('src');
    previewImg.src = previewUrl;
    previewImg.classList.remove('hidden');
  }

  previewBox.classList.remove('hidden');
  fileMeta.classList.remove('hidden');
  fileIcon.textContent = video ? '🎥' : '📷';
  fileNameEl.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  videoOptions.classList.toggle('hidden', !video);
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
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  previewImg.removeAttribute('src');
  previewImg.classList.remove('hidden');
  previewVideo.pause();
  previewVideo.removeAttribute('src');
  previewVideo.load();
  previewVideo.classList.add('hidden');
  videoOptions.classList.add('hidden');
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
  const videoMode = isVideoFile(selectedFile);
  detectBtnText.textContent = videoMode ? 'Analyzing Video (this may take a bit)...' : 'Analyzing Leaf...';
  detectSpinner.classList.remove('hidden');

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    let url = `${API_BASE}/disease-detection/detect`;
    if (videoMode) url += `?num_frames=${encodeURIComponent(numFramesSel.value)}`;

    const res = await fetch(url, {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Analysis failed');
    console.log(data);
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

// function renderDiseaseResult(data) {
//   const isHealthy = data.status === 'healthy';
//   const confidence = data.confidence ? (data.confidence * 100).toFixed(1) : null;
//   const detections = data.detections || [];
//   const cropName = data.crop || (detections.length > 0 ? detections[0].crop : null);

//   let html = `
//     <div class="result-header-card ${isHealthy ? 'healthy' : 'diseased'}">
//       <div class="status-icon">${isHealthy ? '🌿' : '⚠️'}</div>
//       <div>
//         <div class="title">${isHealthy ? 'Healthy Plant Leaf' : escapeHtml(data.diagnosis || 'Disease Detected')}</div>
//         <div class="sub">${isHealthy ? 'No disease lesions observed' : `${detections.length} symptom area(s) detected`}</div>
//       </div>
//     </div>

//     <div class="stats-grid">
//       <div class="stat-box">
//         <div class="lbl">Diagnosis Status</div>
//         <div class="val">${isHealthy ? 'Healthy' : 'Diseased'}</div>
//       </div>
//       ${!isHealthy && cropName ? `
//       <div class="stat-box">
//         <div class="lbl">Crop</div>
//         <div class="val">${escapeHtml(cropName)}</div>
//       </div>
//       ` : ''}
//       <div class="stat-box">
//         <div class="lbl">Inference Time</div>
//         <div class="val">${data.inference_time_ms ? data.inference_time_ms.toFixed(1) + ' ms' : 'N/A'}</div>
//       </div>
//       <div class="stat-box">
//         <div class="lbl">Model</div>
//         <div class="val">${escapeHtml(data.model || 'YOLO11')}</div>
//       </div>
//       <div class="stat-box">
//         <div class="lbl">Confidence</div>
//         <div class="val">${confidence ? confidence + '%' : '100%'}</div>
//       </div>
//     </div>
//   `;

//   if (detections.length > 0) {
//     html += `<div style="margin-top:0.5rem"><span class="input-label">Detected Disease Lesions</span>`;
//     detections.forEach((d) => {
//       const pct = (d.confidence * 100).toFixed(1);
//       html += `
//         <div class="detection-card" style="margin-top:6px">
//           <div>
//             <div class="name">${escapeHtml(d.crop || '')} - ${escapeHtml(d.disease || '')}</div>
//           </div>
//           <span class="badge">${pct}% Conf.</span>
//         </div>
//       `;
//     });
//     html += `</div>`;
//   }

//   diseaseResult.innerHTML = html;
//   diseaseResult.classList.remove('hidden');
//   diseaseEmpty.classList.add('hidden');
// }


function renderDiseaseResult(data) {
  const isHealthy = data.status === 'healthy';
  const confidence = data.confidence ? (data.confidence * 100).toFixed(1) : null;
  const detections = data.detections || [];
  const cropName = data.crop || (detections.length > 0 ? detections[0].crop : null);
  const isVideo = data.media_type === 'video';

  // Prefer the clean summary if backend sends it, otherwise fall back to full treatment object
  const treatment = data.treatment_summary || data.treatment || null;

  let html = `
    <div class="result-header-card ${isHealthy ? 'healthy' : 'diseased'}">
      <div class="status-icon">${isHealthy ? '🌿' : '⚠️'}</div>
      <div>
        <div class="title">${isHealthy ? (isVideo ? 'Healthy Plant' : 'Healthy Plant Leaf') : escapeHtml(data.diagnosis || 'Disease Detected')}</div>
        <div class="sub">${isHealthy ? (isVideo ? 'No disease observed across the sampled frames' : 'No disease lesions observed') : (detections.length > 0 ? `${detections.length} symptom area(s) detected` : (isVideo ? 'Disease identified across the sampled video frames' : 'Disease identified by classification model'))}</div>
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
        <div class="val">${escapeHtml(data.model || 'EfficientNetV2-M')}</div>
      </div>
      <div class="stat-box">
        <div class="lbl">Confidence</div>
        <div class="val">${confidence ? confidence + '%' : '—'}</div>
      </div>
    </div>
  `;

  // ====================== VIDEO ANALYSIS ======================
  if (isVideo && data.analysis) {
    const a = data.analysis;
    const v = data.video || {};
    const agreementPct = Math.round((a.agreement || 0) * 100);

    html += `
      <div class="stats-grid" style="margin-top:0.625rem">
        <div class="stat-box">
          <div class="lbl">Frames Analyzed</div>
          <div class="val">${a.frames_analyzed}</div>
        </div>
        <div class="stat-box">
          <div class="lbl">Frame Agreement</div>
          <div class="val">${agreementPct}%</div>
        </div>
        <div class="stat-box">
          <div class="lbl">Video Length</div>
          <div class="val">${v.duration_s ? v.duration_s + ' s' : 'N/A'}</div>
        </div>
        <div class="stat-box">
          <div class="lbl">Total Processing</div>
          <div class="val">${data.processing_time_ms ? (data.processing_time_ms / 1000).toFixed(1) + ' s' : 'N/A'}</div>
        </div>
      </div>
    `;

    if (data.warning) {
      html += `<div class="warning-box">⚠️ ${escapeHtml(data.warning)}</div>`;
    }
    if (agreementPct < 50) {
      html += `<div class="warning-box">Frames disagree with each other, so this result is less reliable. Try recording again closer to the leaf, steady, in good light.</div>`;
    }

    const issues = a.detected_issues || [];
    if (issues.length > 0) {
      html += `<div style="margin-top:1rem"><span class="input-label">Diseases Seen In Video</span>`;
      issues.forEach(issue => {
        html += `
          <div class="detection-card" style="margin-top:6px">
            <div>
              <div class="name">${escapeHtml(prettyClass(issue.class_name))}</div>
              <div class="sub-text">${Math.round(issue.frame_ratio * 100)}% of frames · avg confidence ${(issue.mean_confidence * 100).toFixed(1)}%</div>
            </div>
            <span class="badge">${issue.frames} frame(s)</span>
          </div>
        `;
      });
      html += `</div>`;
    }

    const dist = a.class_distribution || [];
    if (dist.length > 0) {
      html += `<div style="margin-top:1rem"><span class="input-label">Frame-by-Frame Breakdown</span>`;
      dist.forEach(d => {
        const pct = Math.round(d.ratio * 100);
        html += `
          <div class="dist-row">
            <div class="dist-label">${escapeHtml(prettyClass(d.class_name))}</div>
            <div class="dist-bar"><div class="dist-fill" style="width:${pct}%"></div></div>
            <div class="dist-pct">${pct}%</div>
          </div>
        `;
      });
      html += `</div>`;
    }
  }

  // Old YOLO detections (kept for compatibility)
  if (detections.length > 0) {
    html += `<div style="margin-top:1rem"><span class="input-label">Detected Disease Lesions</span>`;
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

  // ====================== TREATMENT SECTION ======================
  if (treatment && !isHealthy) {
    html += `
      <div class="treatment-section" style="margin-top:1.5rem">
        <div class="section-title" style="font-size:1.1rem; font-weight:600; margin-bottom:0.75rem; color:#1e3a5f;">
          💊 Recommended Treatment
        </div>
    `;

    // Pathogen
    if (treatment.pathogen) {
      html += `
        <div class="treatment-card" style="margin-bottom:12px">
          <div class="lbl">Pathogen</div>
          <div class="val">${escapeHtml(treatment.pathogen)}</div>
        </div>
      `;
    }

    // Organic Treatment
    if (treatment.organic_treatment && treatment.organic_treatment.length > 0) {
      html += `
        <div class="treatment-card" style="margin-bottom:12px">
          <div class="lbl">🌿 Organic Treatment</div>
          <ul style="margin:6px 0 0 18px; padding:0;">
            ${treatment.organic_treatment.map(item => `<li style="margin-bottom:4px">${escapeHtml(item)}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    // Chemical Treatment
    if (treatment.chemical_treatment && treatment.chemical_treatment.length > 0) {
      html += `
        <div class="treatment-card" style="margin-bottom:12px">
          <div class="lbl">🧪 Chemical Treatment</div>
          <ul style="margin:6px 0 0 18px; padding:0;">
            ${treatment.chemical_treatment.map(item => `<li style="margin-bottom:4px">${escapeHtml(item)}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    // Prevention
    if (treatment.prevention && treatment.prevention.length > 0) {
      html += `
        <div class="treatment-card" style="margin-bottom:12px">
          <div class="lbl">🛡️ Prevention</div>
          <ul style="margin:6px 0 0 18px; padding:0;">
            ${treatment.prevention.map(item => `<li style="margin-bottom:4px">${escapeHtml(item)}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    // Polyhouse Specific
    if (treatment.polyhouse_specific && treatment.polyhouse_specific.length > 0) {
      html += `
        <div class="treatment-card" style="margin-bottom:12px">
          <div class="lbl">🏡 Polyhouse Specific Advice</div>
          <ul style="margin:6px 0 0 18px; padding:0;">
            ${treatment.polyhouse_specific.map(item => `<li style="margin-bottom:4px">${escapeHtml(item)}</li>`).join('')}
          </ul>
        </div>
      `;
    }

    // Severity Levels (optional – shown in compact form)
    if (treatment.severity_levels) {
      html += `
        <div class="treatment-card">
          <div class="lbl">📊 Severity Guide</div>
          <div style="margin-top:8px; font-size:0.9rem;">
      `;

      ['mild', 'moderate', 'severe'].forEach(level => {
        const info = treatment.severity_levels[level];
        if (info) {
          html += `
            <div style="margin-bottom:10px; padding:8px; background:#f8fafc; border-radius:6px;">
              <strong style="text-transform:capitalize; color:#1e40af;">${level}</strong><br>
              <span style="color:#475569;">${escapeHtml(info.criteria || '')}</span><br>
              <span style="color:#166534;"><strong>Action:</strong> ${escapeHtml(info.action || '')}</span>
            </div>
          `;
        }
      });

      html += `</div></div>`;
    }

    html += `</div>`; // close treatment-section
  } else if (!isHealthy) {
    // No treatment found in knowledge base
    html += `
      <div class="treatment-section" style="margin-top:1.5rem; padding:12px; background:#fff7ed; border-radius:8px; border:1px solid #fdba74;">
        <div style="color:#9a3412; font-size:0.95rem;">
          ℹ️ No specific treatment information found in the knowledge base for this disease.
        </div>
      </div>
    `;
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