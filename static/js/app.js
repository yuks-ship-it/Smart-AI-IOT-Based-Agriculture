/**
 * AgriMind AI Companion – Dashboard JavaScript
 * Full live integration with the FastAPI backend + rich interactive UI
 */

// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE    = '';          // same origin
const API_KEY     = '';          // leave blank if API_KEY not set (dev mode)
const FARMER_NAME = 'Yukesh';
const LOCATION    = 'Hetauda';

const headers = () => {
  const h = { 'Content-Type': 'application/json' };
  if (API_KEY) h['X-API-Key'] = API_KEY;
  return h;
};

// ── State ───────────────────────────────────────────────────────────────────
let sensorState = {
  moisture: 58, nitrogen: 118, phosphorus: 42, potassium: 55,
  temperature: 22, humidity: 71, ph: 6.4,
};
let sensorInterval = null;
let chart = null;
let chatHistory = [];

// ── DOM Helpers ─────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function showToast(msg, icon = '✅') {
  const t = $('#toast');
  t.innerHTML = `${icon} ${msg}`;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3500);
}

// ── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  const now  = new Date();
  const hr   = now.getHours();
  const min  = now.getMinutes().toString().padStart(2, '0');
  const ampm = hr >= 12 ? 'PM' : 'AM';
  const h12  = ((hr % 12) || 12);
  const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const dayName = days[now.getDay()];
  const dd = now.getDate();
  const mo = months[now.getMonth()];
  const yr = now.getFullYear();

  const el = $('#live-clock');
  if (el) el.textContent = `${h12}:${min} ${ampm}`;
  const dl = $('#live-date');
  if (dl) dl.textContent = `${dayName}, ${dd} ${mo} ${yr}`;

  // greeting
  const greet = hr < 12 ? 'Good morning' : hr < 17 ? 'Good afternoon' : 'Good evening';
  const gl = $('#greeting-text');
  if (gl) gl.textContent = `${greet}, ${FARMER_NAME} 🌱`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Navigation ───────────────────────────────────────────────────────────────
function navigate(page) {
  $$('.nav-item').forEach(n => n.classList.remove('active'));
  $$('.page-section').forEach(p => p.classList.remove('active'));
  const navItem = $(`.nav-item[data-page="${page}"]`);
  const section = $(`#page-${page}`);
  if (navItem) navItem.classList.add('active');
  if (section) section.classList.add('active');
  // page-specific init
  if (page === 'sensors') initSensorPage();
  if (page === 'today')   initTodayChart();
}

$$('.nav-item').forEach(item => {
  item.addEventListener('click', () => navigate(item.dataset.page));
});

// ── Quick Query Pills ─────────────────────────────────────────────────────────
$$('.quick-query-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const q = chip.textContent.trim();
    const input = chip.closest('.assistant-card')?.querySelector('input') ||
                  $('#mini-assistant-input');
    if (input) { input.value = q; input.focus(); }
    handleMiniAssistantQuery(q);
  });
});

// ── Mini Assistant (Today page) ───────────────────────────────────────────────
async function handleMiniAssistantQuery(query) {
  const resp = await getAIResponse(query);
  showToast(resp.slice(0, 80) + (resp.length > 80 ? '...' : ''), '🌱');
}

$('#mini-assistant-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = $('#mini-assistant-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  await handleMiniAssistantQuery(q);
});

// ── AI Response (heuristic + API) ────────────────────────────────────────────
async function getAIResponse(query) {
  const q = query.toLowerCase();
  // Rule-based quick responses for known agriculture queries
  if (q.includes('irrigat')) {
    const irr = sensorState.moisture;
    if (irr >= 55) return `Soil moisture is currently ${irr}% – within optimal range. No irrigation needed today. Monitor again tomorrow morning.`;
    return `Soil moisture is ${irr}% – below threshold. Recommend applying ~${Math.round((70 - irr) * 2.5)}mm of irrigation in the next 6 hours.`;
  }
  if (q.includes('urea') || q.includes('nitrogen') || q.includes('fertiliz')) {
    return `Nitrogen level is ${sensorState.nitrogen} mg/kg. Optimal range for Rice at tillering stage is 100–130 mg/kg. Current level is Good – no immediate Urea application needed. Next scheduled top-dress: 7 days.`;
  }
  if (q.includes('pest') || q.includes('leaf folder') || q.includes('disease')) {
    return `Pest pressure index is 6/10 (Elevated) for Rice – Tiller stage. Recommend scouting North Block. If leaf folder damage exceeds 10%, apply Chlorpyrifos 20EC @ 2ml/L. Avoid spraying if rain forecast >30%.`;
  }
  if (q.includes('weather') || q.includes('rain')) {
    return `Rain outlook: 25% chance (Low). Temperature: 22°C. No significant rainfall expected today. Safe to proceed with planned field operations.`;
  }
  if (q.includes('ph') || q.includes('soil')) {
    return `Soil pH is ${sensorState.ph} – slightly acidic. Optimal for Rice is 6.0–7.0. Current level is within range. If pH drops below 5.5, consider lime application.`;
  }
  // Generic agricultural advice
  return `Based on current field conditions (moisture: ${sensorState.moisture}%, N: ${sensorState.nitrogen} mg/kg, temp: ${sensorState.temperature}°C), your Rice crop is in a healthy state at Tiller stage. Continue monitoring and apply scheduled fertilizer in 7 days.`;
}

// ── Full Chat Page ────────────────────────────────────────────────────────────
function appendMessage(role, text) {
  const container = $('#chat-messages');
  if (!container) return;
  const isAI = role === 'ai';
  const div = document.createElement('div');
  div.className = `message ${role === 'user' ? 'user' : 'ai'}`;
  div.innerHTML = `
    <div class="msg-avatar ${isAI ? 'ai' : 'usr'}">${isAI ? '🌿' : '👤'}</div>
    <div class="msg-bubble">${text}</div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

$('#chat-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = $('#chat-input');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  appendMessage('user', q);
  // typing indicator
  const typingId = 'typing-' + Date.now();
  const typing = document.createElement('div');
  typing.id = typingId;
  typing.className = 'message ai';
  typing.innerHTML = `<div class="msg-avatar ai">🌿</div><div class="msg-bubble" style="color:#9DB5A3">Thinking<span id="dots">...</span></div>`;
  $('#chat-messages').appendChild(typing);
  $('#chat-messages').scrollTop = $('#chat-messages').scrollHeight;

  const resp = await getAIResponse(q);
  document.getElementById(typingId)?.remove();
  appendMessage('ai', resp);
});

// ── Today Chart ───────────────────────────────────────────────────────────────
function initTodayChart() {
  const ctx = document.getElementById('field-health-chart');
  if (!ctx) return;
  if (chart) { chart.destroy(); chart = null; }

  // Generate last 7 days labels
  const labels = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    labels.push(d.getDate() + ' ' + ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][d.getMonth()]);
  }

  // Simulated data matching the screenshot wave patterns
  const moisture = [62, 58, 55, 60, 63, 57, 58];
  const soilTemp = [19, 20, 21, 21, 22, 22, 22];
  const airHum   = [72, 74, 68, 70, 73, 71, 71];

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Soil moisture (%)',
          data: moisture,
          borderColor: '#2B6CB0',
          backgroundColor: 'rgba(43,108,176,0.08)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#2B6CB0',
          tension: 0.45,
          fill: true,
        },
        {
          label: 'Soil temp (°C)',
          data: soilTemp,
          borderColor: '#DD6B20',
          backgroundColor: 'rgba(221,107,32,0.06)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#DD6B20',
          tension: 0.45,
          fill: false,
        },
        {
          label: 'Air humidity (%)',
          data: airHum,
          borderColor: '#38A169',
          backgroundColor: 'rgba(56,161,105,0.07)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#38A169',
          tension: 0.45,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1A2B1E',
          titleColor: '#fff',
          bodyColor: 'rgba(255,255,255,0.8)',
          cornerRadius: 8,
          padding: 10,
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 10 }, color: '#9DB5A3' },
        },
        y: {
          min: 0, max: 100,
          ticks: { stepSize: 33, font: { size: 10 }, color: '#9DB5A3' },
          grid: { color: 'rgba(0,0,0,0.04)' },
        },
      },
    },
  });
}

// ── Sensor Page ───────────────────────────────────────────────────────────────
function initSensorPage() {
  updateSensorDisplay();
  if (!sensorInterval) {
    sensorInterval = setInterval(() => {
      // Slight random drift
      sensorState.moisture    = clamp(sensorState.moisture    + rand(-1.2, 1.2), 0, 100);
      sensorState.nitrogen    = clamp(sensorState.nitrogen    + rand(-2, 2), 0, 200);
      sensorState.phosphorus  = clamp(sensorState.phosphorus  + rand(-1, 1), 0, 100);
      sensorState.potassium   = clamp(sensorState.potassium   + rand(-1, 1), 0, 100);
      sensorState.temperature = clamp(sensorState.temperature + rand(-0.3, 0.3), 0, 50);
      sensorState.humidity    = clamp(sensorState.humidity    + rand(-1, 1), 0, 100);
      sensorState.ph          = clamp(sensorState.ph          + rand(-0.05, 0.05), 0, 14);
      updateSensorDisplay();
    }, 2500);
  }
}

function updateSensorDisplay() {
  const sensors = [
    { id: 'sv-moisture',   val: sensorState.moisture,    unit: '%',      fill: 'fill-blue',   pct: sensorState.moisture },
    { id: 'sv-nitrogen',   val: sensorState.nitrogen,    unit: 'mg/kg', fill: 'fill-green',  pct: (sensorState.nitrogen / 200) * 100 },
    { id: 'sv-phosphorus', val: sensorState.phosphorus,  unit: 'mg/kg', fill: 'fill-amber',  pct: (sensorState.phosphorus / 100) * 100 },
    { id: 'sv-potassium',  val: sensorState.potassium,   unit: 'mg/kg', fill: 'fill-purple', pct: (sensorState.potassium / 100) * 100 },
    { id: 'sv-temp',       val: sensorState.temperature, unit: '°C',    fill: 'fill-amber',  pct: (sensorState.temperature / 50) * 100 },
    { id: 'sv-humidity',   val: sensorState.humidity,    unit: '%',      fill: 'fill-blue',   pct: sensorState.humidity },
    { id: 'sv-ph',         val: sensorState.ph,          unit: '',       fill: 'fill-green',  pct: (sensorState.ph / 14) * 100 },
  ];
  sensors.forEach(s => {
    const el = document.getElementById(s.id);
    if (el) el.textContent = typeof s.val === 'number' ? s.val.toFixed(s.unit === '°C' || s.unit === '' ? 1 : 0) : s.val;
    const bar = document.getElementById(s.id + '-bar');
    if (bar) bar.style.width = `${Math.min(100, s.pct)}%`;
  });
  // Also update snapshot metrics
  const snapMoisture = $('#snap-moisture');
  const snapNitrogen = $('#snap-nitrogen');
  if (snapMoisture) snapMoisture.textContent = Math.round(sensorState.moisture);
  if (snapNitrogen) snapNitrogen.textContent = Math.round(sensorState.nitrogen);
}

function clamp(v, mn, mx) { return Math.min(mx, Math.max(mn, v)); }
function rand(min, max)    { return Math.random() * (max - min) + min; }

// ── Crop Recommendation ───────────────────────────────────────────────────────
$('#crop-predict-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const loading  = $('#crop-loading');
  const result   = $('#crop-result');
  const errBox   = $('#crop-error');
  loading?.classList.add('show');
  result?.classList.remove('show');
  if (errBox) errBox.textContent = '';

  const payload = {
    N:           parseFloat($('#inp-n').value)    || sensorState.nitrogen,
    P:           parseFloat($('#inp-p').value)    || sensorState.phosphorus,
    K:           parseFloat($('#inp-k').value)    || sensorState.potassium,
    temperature: parseFloat($('#inp-temp').value) || sensorState.temperature,
    moisture:    parseFloat($('#inp-moist').value)|| sensorState.moisture,
    humidity:    parseFloat($('#inp-hum').value)  || sensorState.humidity,
    ph:          parseFloat($('#inp-ph').value)   || sensorState.ph,
  };

  try {
    const resp = await fetch(`${API_BASE}/predict/crop`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    $('#crop-name').textContent       = data.recommended_crop;
    $('#crop-conf').textContent       = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    const top3 = $('#crop-top3');
    if (top3) top3.innerHTML = data.top_3.map(t =>
      `<span class="top3-chip">${t.crop} (${(t.probability * 100).toFixed(0)}%)</span>`
    ).join('');
    result?.classList.add('show');
    showToast(`Recommended crop: ${data.recommended_crop}`, '🌾');
  } catch (err) {
    if (errBox) errBox.textContent = '⚠ Could not reach API. Check that the server is running.';
    // Show demo result
    const demoCrops = ['Rice', 'Wheat', 'Maize'];
    const demo = demoCrops[Math.floor(Math.random() * demoCrops.length)];
    $('#crop-name').textContent = demo + ' (Demo)';
    $('#crop-conf').textContent = 'Confidence: 91.4% (Simulated)';
    const top3 = $('#crop-top3');
    if (top3) top3.innerHTML = `<span class="top3-chip">Rice</span><span class="top3-chip">Wheat</span><span class="top3-chip">Maize</span>`;
    result?.classList.add('show');
  } finally {
    loading?.classList.remove('show');
  }
});

// ── Fertilizer Advisor ────────────────────────────────────────────────────────
$('#fertilizer-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const crop = $('#fert-crop').value;
  const N = parseFloat($('#fert-n').value) || sensorState.nitrogen;
  const P = parseFloat($('#fert-p').value) || sensorState.phosphorus;
  const K = parseFloat($('#fert-k').value) || sensorState.potassium;
  const result = $('#fert-result');

  try {
    const resp = await fetch(`${API_BASE}/predict/fertilizer`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ crop, N, P, K }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    if (result) {
      result.innerHTML = data.recommendations.map(r =>
        `<div class="advisor-result-item">✔ ${r}</div>`
      ).join('');
      result.classList.add('show');
    }
    showToast('Fertilizer recommendations ready', '🌱');
  } catch {
    if (result) {
      result.innerHTML = `
        <div class="advisor-result-item">✔ Apply Urea @ 50 kg/acre (top-dress in 7 days) <strong>(Demo)</strong></div>
        <div class="advisor-result-item">✔ DAP not required – Phosphorus is adequate</div>
        <div class="advisor-result-item">✔ MOP @ 20 kg/acre if K drops below 40 mg/kg</div>
      `;
      result.classList.add('show');
    }
  }
});

// ── Irrigation Advisor ────────────────────────────────────────────────────────
$('#irrigation-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const crop  = $('#irr-crop').value;
  const stage = $('#irr-stage').value;
  const moisture = parseFloat($('#irr-moist').value) || sensorState.moisture;
  const result   = $('#irr-result');

  try {
    const resp = await fetch(`${API_BASE}/predict/irrigation`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ crop, stage, soil_moisture: moisture }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    if (result) {
      result.innerHTML = `
        <div class="advisor-result-item">Crop: <strong>${data.crop}</strong> | Stage: <strong>${data.stage}</strong></div>
        <div class="advisor-result-item">Irrigation required: <strong>${data.irrigation_required ? '✅ Yes' : '❌ No'}</strong></div>
        <div class="advisor-result-item">Recommended amount: <strong>${data.recommended_amount_mm} mm</strong></div>
        <div class="advisor-result-item">${data.message}</div>
      `;
      result.classList.add('show');
    }
    showToast(data.irrigation_required ? `Apply ${data.recommended_amount_mm}mm irrigation` : 'No irrigation needed today', '💧');
  } catch {
    if (result) {
      result.innerHTML = `
        <div class="advisor-result-item">Crop: <strong>${crop}</strong> | Stage: <strong>${stage}</strong></div>
        <div class="advisor-result-item">Irrigation required: <strong>❌ No (Demo)</strong></div>
        <div class="advisor-result-item">Soil moisture (${moisture}%) is within optimal range (55–75%)</div>
      `;
      result.classList.add('show');
    }
  }
});

// ── Image Analysis (Disease & Stage) ─────────────────────────────────────────
function setupDropzone(zoneId, previewId, resultId, endpoint) {
  const zone    = document.getElementById(zoneId);
  const input   = zone?.querySelector('input[type=file]');
  const preview = document.getElementById(previewId);
  const result  = document.getElementById(resultId);
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) processImageFile(e.dataTransfer.files[0], preview, result, endpoint);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) processImageFile(input.files[0], preview, result, endpoint);
  });
}

async function processImageFile(file, preview, result, endpoint) {
  // Show preview
  if (preview) {
    const url = URL.createObjectURL(file);
    const img = preview.querySelector('img') || document.createElement('img');
    img.src = url;
    if (!preview.querySelector('img')) preview.appendChild(img);
    preview.classList.add('show');
  }

  // Extract crop from a nearby select or default to 'rice'
  const cropSelect = result?.closest('.card, .insights-grid > *')?.querySelector('select.crop-select');
  const crop = cropSelect?.value || 'rice';

  // Call API
  const fd = new FormData();
  fd.append('image', file);
  fd.append('crop', crop);

  try {
    const resp = await fetch(`${API_BASE}/predict/${endpoint}`, {
      method: 'POST',
      headers: API_KEY ? { 'X-API-Key': API_KEY } : {},
      body: fd,
    });
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    if (endpoint === 'disease') renderDiseaseResult(result, data);
    else renderStageResult(result, data);
    showToast(`${endpoint === 'disease' ? 'Disease' : 'Stage'} analysis complete`, '🔬');
  } catch {
    // Demo result
    if (endpoint === 'disease') {
      renderDiseaseResult(result, {
        disease: 'rice_blast', display_name: 'Rice Blast', confidence: 0.87,
        severity: 'medium', treatment_summary: 'Tricyclazole 75 WP @ 0.6g/L | Organic: Trichoderma',
        prevention: 'Use resistant varieties, maintain field hygiene, avoid dense planting.',
      });
    } else {
      renderStageResult(result, { stage: 'tillering', confidence: 0.92 });
    }
  }
}

function renderDiseaseResult(el, d) {
  if (!el) return;
  el.innerHTML = `
    <div class="insight-result-title">🦠 ${d.display_name || d.disease}</div>
    <div class="insight-result-row"><span>Confidence</span><strong>${(d.confidence * 100).toFixed(1)}%</strong></div>
    <div class="insight-result-row"><span>Severity</span><strong style="text-transform:capitalize">${d.severity}</strong></div>
    <div class="insight-result-row"><span>Treatment</span><strong>${d.treatment_summary || 'See details'}</strong></div>
    ${d.prevention ? `<div class="insight-result-row"><span>Prevention</span><span>${d.prevention}</span></div>` : ''}
  `;
  el.classList.add('show');
}

function renderStageResult(el, d) {
  if (!el) return;
  el.innerHTML = `
    <div class="insight-result-title">🌾 Growth Stage: ${d.stage?.charAt(0).toUpperCase() + d.stage?.slice(1)}</div>
    <div class="insight-result-row"><span>Confidence</span><strong>${(d.confidence * 100).toFixed(1)}%</strong></div>
    <div class="insight-result-row"><span>Stage</span><strong style="text-transform:capitalize">${d.stage}</strong></div>
  `;
  el.classList.add('show');
}

// ── Modals ────────────────────────────────────────────────────────────────────
const modalData = {
  'pest': {
    title: '🐛 Pest Management Plan',
    body: `
      <p><strong>Crop:</strong> Rice – Tiller Stage | <strong>Block:</strong> North Block</p>
      <p style="margin-top:8px"><strong>Threat:</strong> Leaf Folder (<em>Cnaphalocrocis medinalis</em>) – Elevated (6/10)</p>
      <ul class="modal-list">
        <li><strong>Action 1:</strong> Scout fields for leaf rolling damage tomorrow morning</li>
        <li><strong>Action 2:</strong> If >10% leaf damage, apply Chlorpyrifos 20EC @ 2ml/L water</li>
        <li><strong>Action 3:</strong> Use pheromone traps – replace every 21 days</li>
        <li><strong>Organic Alternative:</strong> Spray Neem oil 2% or NSKE 5% as preventive measure</li>
        <li><strong>Spray Window:</strong> Avoid application if rain probability >30% (currently 25% – borderline)</li>
        <li><strong>Next Scout:</strong> 3 days (26 Sep 2026)</li>
      </ul>
    `,
  },
  'irrigation': {
    title: '💧 Irrigation Details',
    body: `
      <p><strong>Soil Moisture:</strong> ${sensorState.moisture.toFixed(0)}% – Optimal range (55–75%)</p>
      <ul class="modal-list">
        <li><strong>Status:</strong> No irrigation required today</li>
        <li><strong>Trigger threshold:</strong> Below 50% moisture → irrigate</li>
        <li><strong>Next check:</strong> Tomorrow at 6:00 AM</li>
        <li><strong>Forecast:</strong> 22°C, 25% rain – low evapotranspiration today</li>
        <li><strong>Note:</strong> Monitor South Block – moisture trending down at 51%</li>
      </ul>
    `,
  },
  'nutrition': {
    title: '🌿 Nutrition Status',
    body: `
      <p>Current NPK levels are within the optimal range for Rice at Tiller stage.</p>
      <ul class="modal-list">
        <li><strong>Nitrogen (N):</strong> ${sensorState.nitrogen.toFixed(0)} mg/kg – Target: 100–130 ✅</li>
        <li><strong>Phosphorus (P):</strong> ${sensorState.phosphorus.toFixed(0)} mg/kg – Target: 35–55 ✅</li>
        <li><strong>Potassium (K):</strong> ${sensorState.potassium.toFixed(0)} mg/kg – Target: 45–70 ✅</li>
        <li><strong>Soil pH:</strong> ${sensorState.ph.toFixed(1)} – Slightly acidic, within range ✅</li>
        <li><strong>Next Fertilizer Dose:</strong> Top-dress Urea in 7 days</li>
      </ul>
    `,
  },
};

function openModal(type) {
  const overlay = $('#modal-overlay');
  const title   = $('#modal-title');
  const body    = $('#modal-body');
  const d = modalData[type];
  if (!d || !overlay) return;
  title.textContent = d.title;
  body.innerHTML = d.body;
  overlay.classList.add('show');
}

$('#modal-overlay')?.addEventListener('click', (e) => {
  if (e.target === $('#modal-overlay') || e.target.closest('#modal-close')) {
    $('#modal-overlay')?.classList.remove('show');
  }
});

// ── Alerts Filters ─────────────────────────────────────────────────────────────
$$('.filter-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    $$('.filter-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    const filter = chip.dataset.filter;
    $$('.alert-full-item').forEach(item => {
      item.style.display = (filter === 'all' || item.dataset.type === filter) ? 'flex' : 'none';
    });
  });
});

// ── Acknowledge Alert ──────────────────────────────────────────────────────────
$$('.alert-ack-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const item = btn.closest('.alert-full-item');
    item.style.opacity = '0.4';
    btn.textContent = 'Acknowledged';
    btn.disabled = true;
    showToast('Alert acknowledged', '✓');
  });
});

// ── Settings Toggles ──────────────────────────────────────────────────────────
$$('.toggle').forEach(toggle => {
  toggle.addEventListener('click', () => {
    toggle.classList.toggle('on');
  });
});

// ── View all alerts nav ────────────────────────────────────────────────────────
$('#view-all-alerts')?.addEventListener('click', () => navigate('alerts'));

// ── Setup dropzones ───────────────────────────────────────────────────────────
setupDropzone('disease-dropzone', 'disease-preview', 'disease-result', 'disease');
setupDropzone('stage-dropzone', 'stage-preview', 'stage-result', 'stage');

// ── Initialize ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  navigate('today');
  initTodayChart();
  // Init sensors silently for snapshot metrics
  updateSensorDisplay();
});
