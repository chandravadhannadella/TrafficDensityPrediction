// ============================================
// LEGACY APP.JS - Old UI Handler
// Now delegated to ui-controller.js for new UI
// ============================================
// This file is kept for backwards compatibility and polling logic

// Check if old UI elements exist, if not the new UI is being used
const isOldUI = document.getElementById('video-mode-panel') !== null;
if (!isOldUI) {
  console.log('New UI detected - using ui-controller.js for form handling');
  // Exit early - let ui-controller.js handle everything
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      // New UI is active, no need to initialize old handlers
    });
  }
} else {
  // Old UI initialization continues below...
}

const videoUploadForm = document.getElementById('video-upload-form');
const imageUploadForm = document.getElementById('image-upload-form');
const videoInput = document.getElementById('video-input');
const imageInput = document.getElementById('image-input');
const videoDropZone = document.getElementById('video-drop-zone');
const imageDropZone = document.getElementById('image-drop-zone');
const videoAnalyzeBtn = document.getElementById('video-analyze-btn');
const imageAnalyzeBtn = document.getElementById('image-analyze-btn');
const videoRemoveBtn = document.getElementById('video-remove-btn');
const imageRemoveBtn = document.getElementById('image-remove-btn');
const videoFileMeta = document.getElementById('video-file-meta');
const imageFileMeta = document.getElementById('image-file-meta');
const selectedFileName = document.getElementById('selected-file-name');
const selectedFileSize = document.getElementById('selected-file-size');
const selectedFileFormat = document.getElementById('selected-file-format');
const imageSelectedFileName = document.getElementById('image-selected-file-name');
const imageSelectedFileSize = document.getElementById('image-selected-file-size');
const imageSelectedFileFormat = document.getElementById('image-selected-file-format');
const videoUploadMessage = document.getElementById('video-upload-message');
const imageUploadMessage = document.getElementById('image-upload-message');
const statusPanel = document.getElementById('status-panel');
const progressBar = document.getElementById('progress-bar');
const progressPercent = document.getElementById('progress-percent');
const resultsSection = document.getElementById('results-section');
const imageResultsSection = document.getElementById('image-results-section');
const resultCard = document.getElementById('result-card');
const densityPill = document.getElementById('density-pill');
const overallConfidence = document.getElementById('overall-confidence');
const previewSection = document.getElementById('preview-section');
const previewTitle = document.getElementById('preview-title');
const videoPreview = document.getElementById('video-preview');
const previewFile = document.getElementById('preview-file');
const previewDuration = document.getElementById('preview-duration');
const previewResolution = document.getElementById('preview-resolution');
const imagePreviewSection = document.getElementById('image-preview-section');
const imagePreview = document.getElementById('image-preview');
const imagePreviewFile = document.getElementById('image-preview-file');
const imagePreviewFormat = document.getElementById('image-preview-format');
const imagePreviewDimensions = document.getElementById('image-preview-dimensions');
const imagePreviewSize = document.getElementById('image-preview-size');
const summaryText = document.getElementById('summary-text');
const imageSummaryText = document.getElementById('image-summary-text');
const imageDensityPill = document.getElementById('image-density-pill');
const imageConfidence = document.getElementById('image-confidence');
const imageConfidenceBar = document.getElementById('image-confidence-bar');
const imageResultCaption = document.getElementById('image-result-caption');
const imageDetectionList = document.getElementById('image-detection-list');
const imageTotalVehicles = document.getElementById('image-total-vehicles');
const imageModelVersion = document.getElementById('image-model-version');
const imageResultFile = document.getElementById('image-result-file');
const imageProxyNote = document.getElementById('image-proxy-note');
const windowTableBody = document.getElementById('window-table-body');
const chartWrap = document.getElementById('chart-wrap');
const trafficIndicator = document.getElementById('traffic-indicator');
const modeButtons = document.querySelectorAll('.mode-btn') || [];
const videoModePanel = document.getElementById('video-mode-panel');
const imageModePanel = document.getElementById('image-mode-panel');

const appState = {
  mode: 'video',
  video: {
    selectedFile: null,
    previewUrl: null,
    currentJobId: null,
    pollTimer: null,
  },
  image: {
    selectedFile: null,
    previewUrl: null,
    currentJobId: null,
    pollTimer: null,
  },
};

function setMessage(text, type = 'neutral', mode = appState.mode) {
  const target = mode === 'video' ? videoUploadMessage : imageUploadMessage;
  if (!target) return;
  target.className = `message-box message-${type}`;
  target.textContent = text;
}

function setStatus(text, tone = 'ready') {
  const pipClass = tone === 'processing' ? 'processing' : tone === 'error' ? 'error' : 'ready';
  statusPanel.innerHTML = `<div class="status-row"><span class="status-pip ${pipClass}"></span>${text}</div>`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function setBusyState(mode, isBusy) {
  if (mode === 'video') {
    videoAnalyzeBtn.disabled = isBusy || !appState.video.selectedFile;
    videoAnalyzeBtn.textContent = isBusy ? 'Analyzing...' : 'Analyze Traffic';
    videoRemoveBtn.disabled = isBusy;
  } else {
    imageAnalyzeBtn.disabled = isBusy || !appState.image.selectedFile;
    imageAnalyzeBtn.textContent = isBusy ? 'Analyzing...' : 'Analyze Image';
    imageRemoveBtn.disabled = isBusy;
  }
}

function setProcessingSteps(mode, completedCount = 0) {
  const steps = [...document.querySelectorAll('.step')];
  const labels = mode === 'video'
    ? ['Video uploaded', 'Vehicle detection', 'Traffic analysis', 'Final prediction']
    : ['Uploading image', 'Detecting vehicles', 'Analyzing traffic', 'Generating prediction'];

  steps.forEach((step, index) => {
    const label = labels[index];
    step.dataset.stepsVideo = 'Video uploaded';
    step.dataset.stepsImage = 'Uploading image';
    const mark = index < completedCount ? '✓' : '○';
    const stateClass = index < completedCount ? 'done' : 'pending';
    step.className = `step ${stateClass}`;
    step.innerHTML = `<span class="step-mark">${mark}</span><span>${label}</span>`;
  });
}

function setTrafficColor(label) {
  const normalized = String(label || '').toUpperCase();
  if (normalized === 'LOW') return 'low';
  if (normalized === 'MEDIUM') return 'medium';
  if (normalized === 'HIGH') return 'high';
  return 'low';
}

function formatDensityLabel(label) {
  const value = String(label || 'LOW').toUpperCase();
  return value === 'HIGH' ? 'HIGH' : value === 'MEDIUM' ? 'MEDIUM' : 'LOW';
}

function updateStepStatus(mode, completedStepName) {
  const labels = mode === 'video'
    ? ['Video uploaded', 'Vehicle detection', 'Traffic analysis', 'Final prediction']
    : ['Uploading image', 'Detecting vehicles', 'Analyzing traffic', 'Generating prediction'];

  const steps = [...document.querySelectorAll('.step')];
  steps.forEach((step, index) => {
    const match = labels[index] === completedStepName;
    if (match) {
      step.className = 'step done';
      step.innerHTML = `<span class="step-mark">✓</span><span>${labels[index]}</span>`;
    } else if (index < labels.indexOf(completedStepName) || completedStepName === null) {
      step.className = 'step done';
      step.innerHTML = `<span class="step-mark">✓</span><span>${labels[index]}</span>`;
    } else {
      step.className = 'step pending';
      step.innerHTML = `<span class="step-mark">○</span><span>${labels[index]}</span>`;
    }
  });
}

function clearTimers() {
  if (appState.video.pollTimer) {
    window.clearTimeout(appState.video.pollTimer);
    appState.video.pollTimer = null;
  }
  if (appState.image.pollTimer) {
    window.clearTimeout(appState.image.pollTimer);
    appState.image.pollTimer = null;
  }
}

function setMode(mode) {
  appState.mode = mode;
  clearTimers();
  modeButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.mode === mode);
  });

  videoModePanel.classList.toggle('hidden', mode !== 'video');
  imageModePanel.classList.toggle('hidden', mode !== 'image');
  resultsSection.classList.toggle('hidden', mode !== 'video');
  imageResultsSection.classList.toggle('hidden', mode !== 'image');
  previewSection.classList.toggle('hidden', mode !== 'video' || !appState.video.selectedFile);
  imagePreviewSection.classList.toggle('hidden', mode !== 'image' || !appState.image.selectedFile);

  if (mode === 'video') {
    setStatus('Waiting for a video upload');
    setProcessingSteps('video', 0);
    setMessage('Supported formats: MP4 • AVI • MOV • MKV', 'neutral', 'video');
  } else {
    setStatus('Waiting for an image upload');
    setProcessingSteps('image', 0);
    setMessage('Supported formats: JPG • JPEG • PNG • WEBP • BMP', 'neutral', 'image');
  }
}

function buildChart(windows) {
  if (!Array.isArray(windows) || windows.length === 0) {
    chartWrap.innerHTML = '<p class="summary-text">No window data available.</p>';
    return;
  }

  const width = 780;
  const height = 220;
  const margin = { top: 18, right: 12, bottom: 32, left: 32 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const bars = windows.map((window, index) => {
    const x = margin.left + (index / Math.max(windows.length, 1)) * chartWidth + 12;
    const barWidth = Math.max((chartWidth / windows.length) * 0.62, 18);
    const probs = window.probabilities || {};
    const low = Number(probs.LOW || 0);
    const medium = Number(probs.MEDIUM || 0);
    const high = Number(probs.HIGH || 0);

    return `
      <g>
        <rect x="${x}" y="${margin.top + chartHeight - low * chartHeight}" width="${barWidth * 0.32}" height="${low * chartHeight}" fill="#34d399" opacity="0.9" rx="4"/>
        <rect x="${x + barWidth * 0.35}" y="${margin.top + chartHeight - medium * chartHeight}" width="${barWidth * 0.32}" height="${medium * chartHeight}" fill="#fbbf24" opacity="0.9" rx="4"/>
        <rect x="${x + barWidth * 0.7}" y="${margin.top + chartHeight - high * chartHeight}" width="${barWidth * 0.3}" height="${high * chartHeight}" fill="#f87171" opacity="0.95" rx="4"/>
      </g>
    `;
  }).join('');

  const xAxisTicks = windows.map((window, index) => {
    const x = margin.left + (index / Math.max(windows.length, 1)) * chartWidth + 12 + 10;
    return `<text x="${x}" y="${height - 8}" class="axis-label">${index + 1}</text>`;
  }).join('');

  const yGrid = [0, 0.25, 0.5, 0.75, 1].map((mark) => {
    const y = margin.top + chartHeight - mark * chartHeight;
    return `
      <line x1="${margin.left}" x2="${width - margin.right}" y1="${y}" y2="${y}" stroke="rgba(148,163,184,0.15)" />
      <text x="4" y="${y + 4}" class="axis-label">${mark * 100}%</text>
    `;
  }).join('');

  chartWrap.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="Traffic density chart">
      ${yGrid}
      ${bars}
      ${xAxisTicks}
    </svg>
    <div class="legend">
      <span class="legend-item"><span class="legend-swatch" style="background:#34d399"></span>Low</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#fbbf24"></span>Medium</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#f87171"></span>High</span>
    </div>
  `;
}

function renderSummary(payload) {
  const windows = Array.isArray(payload.window_predictions) ? payload.window_predictions : [];
  const aggregateCount = windows.reduce((total, row) => total + (Number(row.confidence || 0) || 0), 0);
  const averageConfidence = windows.length ? aggregateCount / windows.length : 0;
  const vehicleTotals = windows.reduce((acc, row) => {
    acc.cars += Number(row.avg_car_count || 0);
    acc.buses += Number(row.avg_bus_count || 0);
    acc.trucks += Number(row.avg_truck_count || 0);
    acc.motorcycles += Number(row.avg_motorcycle_count || 0);
    acc.bicycles += Number(row.avg_bicycle_count || 0);
    return acc;
  }, { cars: 0, buses: 0, trucks: 0, motorcycles: 0, bicycles: 0 });

  document.getElementById('stat-windows').textContent = String(payload.windows_processed || windows.length || 0);
  document.getElementById('stat-confidence').textContent = windows.length ? `${(averageConfidence * 100).toFixed(1)}%` : '—';
  document.getElementById('stat-dominant').textContent = formatDensityLabel(payload.overall_traffic_density || 'LOW');
  document.getElementById('stat-vehicle-total').textContent = Object.values(vehicleTotals).reduce((sum, value) => sum + value, 0).toFixed(1);

  const densityState = formatDensityLabel(payload.overall_traffic_density || 'LOW');
  densityPill.className = `density-pill ${setTrafficColor(densityState)}`;
  densityPill.textContent = densityState;
  overallConfidence.textContent = payload.overall_confidence ? `${(Number(payload.overall_confidence) * 100).toFixed(1)}%` : '0%';

  summaryText.textContent = `Traffic was predominantly ${densityState.toUpperCase()} across the analyzed video.`;

  const indicatorLevels = ['LOW', 'MEDIUM', 'HIGH'];
  [...trafficIndicator.children].forEach((node, index) => {
    node.classList.toggle('is-active', indicatorLevels[index] === densityState.toUpperCase());
  });

  const windowRows = windows.map((window) => {
    const label = formatDensityLabel(window.predicted_traffic_density || 'LOW');
    const probs = window.probabilities || {};
    return `
      <tr>
        <td>${Number(window.window_start).toFixed(0)}–${Number(window.window_end).toFixed(0)}s</td>
        <td><span class="badge ${setTrafficColor(label)}">${label}</span></td>
        <td>${Number(window.confidence || 0).toFixed(2)}</td>
        <td>${Number(probs.LOW || 0).toFixed(2)}</td>
        <td>${Number(probs.MEDIUM || 0).toFixed(2)}</td>
        <td>${Number(probs.HIGH || 0).toFixed(2)}</td>
      </tr>
    `;
  }).join('');
  windowTableBody.innerHTML = windowRows || '<tr><td colspan="6">No window data available.</td></tr>';

  buildChart(windows);
  resultCard.classList.remove('hidden');
  resultsSection.classList.remove('hidden');
}

function resetVideoState() {
  appState.video.selectedFile = null;
  appState.video.previewUrl = null;
  appState.video.currentJobId = null;
  if (appState.video.pollTimer) {
    window.clearTimeout(appState.video.pollTimer);
    appState.video.pollTimer = null;
  }
  videoInput.value = '';
  videoFileMeta.classList.add('hidden');
  videoRemoveBtn.classList.add('hidden');
  videoAnalyzeBtn.disabled = true;
  setMessage('Supported formats: MP4 • AVI • MOV • MKV', 'neutral', 'video');
  previewSection.classList.add('hidden');
  if (videoPreview.src) {
    videoPreview.pause();
    videoPreview.removeAttribute('src');
    videoPreview.load();
  }
  setStatus('Waiting for a video upload');
  setProcessingSteps('video', 0);
  resultsSection.classList.add('hidden');
  resultCard.classList.add('hidden');
  progressBar.style.width = '0%';
  progressPercent.textContent = '0%';
}

function handleVideoSelection(file) {
  if (!file) return;
  const supported = ['.mp4', '.avi', '.mov', '.mkv'];
  const suffix = `.${file.name.split('.').pop().toLowerCase()}`;

  if (!supported.includes(suffix)) {
    setMessage('Unable to analyze this video. Please use MP4, AVI, MOV, or MKV.', 'error', 'video');
    return;
  }

  appState.video.selectedFile = file;
  selectedFileName.textContent = file.name;
  selectedFileSize.textContent = formatBytes(file.size);
  selectedFileFormat.textContent = suffix.replace('.', '').toUpperCase();
  videoFileMeta.classList.remove('hidden');
  videoRemoveBtn.classList.remove('hidden');
  videoAnalyzeBtn.disabled = false;
  setMessage('Video selected and ready for analysis.', 'success', 'video');

  if (appState.video.previewUrl) {
    URL.revokeObjectURL(appState.video.previewUrl);
  }
  appState.video.previewUrl = URL.createObjectURL(file);
  previewSection.classList.remove('hidden');
  previewTitle.textContent = file.name;
  previewFile.textContent = file.name;
  previewDuration.textContent = 'Duration unavailable';
  previewResolution.textContent = 'Resolution unavailable';
  videoPreview.src = appState.video.previewUrl;

  videoPreview.onloadedmetadata = () => {
    const duration = videoPreview.duration;
    if (Number.isFinite(duration)) {
      previewDuration.textContent = `${duration.toFixed(0)} sec`;
    }
    if (videoPreview.videoWidth && videoPreview.videoHeight) {
      previewResolution.textContent = `${videoPreview.videoWidth} × ${videoPreview.videoHeight}`;
    }
  };
}

function resetImageState() {
  appState.image.selectedFile = null;
  appState.image.previewUrl = null;
  appState.image.currentJobId = null;
  if (appState.image.pollTimer) {
    window.clearTimeout(appState.image.pollTimer);
    appState.image.pollTimer = null;
  }
  imageInput.value = '';
  imageFileMeta.classList.add('hidden');
  imageRemoveBtn.classList.add('hidden');
  imageAnalyzeBtn.disabled = true;
  setMessage('Supported formats: JPG • JPEG • PNG • WEBP • BMP', 'neutral', 'image');
  imagePreviewSection.classList.add('hidden');
  imageResultsSection.classList.add('hidden');
  if (imagePreview.src) {
    imagePreview.removeAttribute('src');
  }
  setStatus('Waiting for an image upload');
  setProcessingSteps('image', 0);
  progressBar.style.width = '0%';
  progressPercent.textContent = '0%';
  imageSummaryText.textContent = 'No image analysis result yet.';
}

function handleImageSelection(file) {
  if (!file) return;
  const supported = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];
  const suffix = `.${file.name.split('.').pop().toLowerCase()}`;

  if (!supported.includes(suffix)) {
    setMessage('Unable to analyze this image. Please use JPG, JPEG, PNG, WEBP, or BMP.', 'error', 'image');
    return;
  }

  appState.image.selectedFile = file;
  imageSelectedFileName.textContent = file.name;
  imageSelectedFileSize.textContent = formatBytes(file.size);
  imageSelectedFileFormat.textContent = suffix.replace('.', '').toUpperCase();
  imageFileMeta.classList.remove('hidden');
  imageRemoveBtn.classList.remove('hidden');
  imageAnalyzeBtn.disabled = false;
  setMessage('Image selected and ready for analysis.', 'success', 'image');

  if (appState.image.previewUrl) {
    URL.revokeObjectURL(appState.image.previewUrl);
  }
  appState.image.previewUrl = URL.createObjectURL(file);
  imagePreview.src = appState.image.previewUrl;
  imagePreviewSection.classList.remove('hidden');
  imagePreviewFile.textContent = file.name;
  imagePreviewFormat.textContent = suffix.replace('.', '').toUpperCase();
  imagePreviewSize.textContent = formatBytes(file.size);
  imagePreviewDimensions.textContent = 'Dimensions unavailable';

  const imageLoader = new Image();
  imageLoader.onload = () => {
    imagePreviewDimensions.textContent = `${imageLoader.naturalWidth} × ${imageLoader.naturalHeight}`;
  };
  imageLoader.onerror = () => {
    imagePreviewDimensions.textContent = 'Dimensions unavailable';
  };
  imageLoader.src = appState.image.previewUrl;
}

function renderImageResult(payload) {
  const density = formatDensityLabel(payload.overall_traffic_density || 'LOW');
  const confidence = Number(payload.overall_confidence || 0) * 100;
  const entries = Object.entries(payload.counts || {}).filter(([key]) => ['car', 'bus', 'truck', 'motorcycle', 'bicycle'].includes(key));
  const total = Number(payload.total_vehicles ?? entries.reduce((sum, [, value]) => sum + Number(value || 0), 0));

  imageDensityPill.className = `density-pill ${setTrafficColor(density)}`;
  imageDensityPill.textContent = density;
  imageConfidence.textContent = `${confidence.toFixed(1)}%`;
  imageConfidenceBar.style.width = `${Math.min(Math.max(confidence, 0), 100)}%`;
  imageResultCaption.textContent = `${density} traffic density across ${total} detected vehicle${total === 1 ? '' : 's'}.`;
  imageModelVersion.textContent = payload.model_version || 'V4-IMAGE';
  imageResultFile.textContent = payload.image || '—';
  imageProxyNote.textContent = payload.proxy_label_note || '';

  if (entries.length) {
    const formatter = {
      car: 'Cars',
      bus: 'Buses',
      truck: 'Trucks',
      motorcycle: 'Motorcycles',
      bicycle: 'Bicycles',
    };
    imageDetectionList.innerHTML = entries.map(([key, value]) => `
      <div class="detection-item">
        <span>${formatter[key] || key}</span>
        <strong>${Number(value || 0)}</strong>
      </div>
    `).join('');
  } else {
    imageDetectionList.innerHTML = '<div class="detection-item"><span>No vehicles detected</span><strong>0</strong></div>';
  }

  imageTotalVehicles.textContent = String(total || 0);
  imageSummaryText.textContent = `This image suggests ${density.toLowerCase()} traffic density with ${confidence.toFixed(1)}% confidence based on ${total || 0} detected vehicles.`;
  imageResultsSection.classList.remove('hidden');
}

function startVideoPolling(jobId) {
  if (!jobId) return;
  appState.video.currentJobId = jobId;

  fetch(`/api/status/${jobId}`)
    .then((response) => response.json())
    .then((payload) => {
      const percent = Math.min(Number(payload.progress || 0), 100);
      progressBar.style.width = `${percent}%`;
      progressPercent.textContent = `${Math.round(percent)}%`;

      if (payload.status === 'processing') {
        setBusyState('video', true);
        setStatus(`Analyzing traffic... ${Math.round(percent)}% complete`, 'processing');
        updateStepStatus('video', 'Traffic analysis');
        appState.video.pollTimer = window.setTimeout(() => startVideoPolling(jobId), 1200);
        return;
      }

      if (payload.status === 'completed') {
        setBusyState('video', false);
        setStatus('Analysis complete', 'ready');
        fetch(`/api/results/${jobId}`)
          .then((response) => response.json())
          .then((result) => {
            if (result && result.success) {
              renderSummary(result);
              setMessage('✓ Analysis complete', 'success', 'video');
              updateStepStatus('video', 'Final prediction');
            } else {
              setStatus('Unable to load results', 'error');
              setMessage('Unable to analyze this video. Please try another file.', 'error', 'video');
            }
          })
          .catch(() => {
            setStatus('Results unavailable', 'error');
            setMessage('Unable to analyze this video. Please try another file.', 'error', 'video');
          });
        return;
      }

      if (payload.status === 'failed') {
        setBusyState('video', false);
        setStatus('Analysis could not be completed', 'error');
        setMessage('Please try again or upload another file.', 'error', 'video');
        return;
      }

      setBusyState('video', true);
      setStatus('Waiting for processing to begin', 'processing');
      appState.video.pollTimer = window.setTimeout(() => startVideoPolling(jobId), 1200);
    })
    .catch(() => {
      setBusyState('video', false);
      setStatus('API request failed', 'error');
      setMessage('Network issue while processing the video. Please try again.', 'error', 'video');
    });
}

function startImagePolling(jobId) {
  if (!jobId) return;
  appState.image.currentJobId = jobId;

  fetch(`/api/status/${jobId}`)
    .then((response) => response.json())
    .then((payload) => {
      const percent = Math.min(Number(payload.progress || 0), 100);
      progressBar.style.width = `${percent}%`;
      progressPercent.textContent = `${Math.round(percent)}%`;

      if (payload.status === 'processing') {
        setBusyState('image', true);
        setStatus(`Analyzing image... ${Math.round(percent)}% complete`, 'processing');
        updateStepStatus('image', 'Analyzing traffic');
        appState.image.pollTimer = window.setTimeout(() => startImagePolling(jobId), 1200);
        return;
      }

      if (payload.status === 'completed') {
        setBusyState('image', false);
        setStatus('Image analysis complete', 'ready');
        fetch(`/api/results/${jobId}`)
          .then((response) => response.json())
          .then((result) => {
            if (result && result.success) {
              renderImageResult(result);
              setMessage('✓ Image analysis complete', 'success', 'image');
              updateStepStatus('image', 'Generating prediction');
            } else {
              setStatus('Unable to load image result', 'error');
              setMessage('Unable to analyze this image. Please try another file.', 'error', 'image');
            }
          })
          .catch(() => {
            setStatus('Image results unavailable', 'error');
            setMessage('Unable to analyze this image. Please try another file.', 'error', 'image');
          });
        return;
      }

      if (payload.status === 'failed') {
        setBusyState('image', false);
        setStatus('Analysis could not be completed', 'error');
        setMessage('Please try again or upload another file.', 'error', 'image');
        return;
      }

      setBusyState('image', true);
      setStatus('Waiting for processing to begin', 'processing');
      appState.image.pollTimer = window.setTimeout(() => startImagePolling(jobId), 1200);
    })
    .catch(() => {
      setBusyState('image', false);
      setStatus('API request failed', 'error');
      setMessage('Network issue while processing the image. Please try again.', 'error', 'image');
    });
}

videoInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (file) handleVideoSelection(file);
});

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (file) handleImageSelection(file);
});

videoDropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  videoDropZone.classList.add('dragging');
});

videoDropZone.addEventListener('dragleave', () => {
  videoDropZone.classList.remove('dragging');
});

videoDropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  videoDropZone.classList.remove('dragging');
  const file = event.dataTransfer.files[0];
  if (file) handleVideoSelection(file);
});

imageDropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  imageDropZone.classList.add('dragging');
});

imageDropZone.addEventListener('dragleave', () => {
  imageDropZone.classList.remove('dragging');
});

imageDropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  imageDropZone.classList.remove('dragging');
  const file = event.dataTransfer.files[0];
  if (file) handleImageSelection(file);
});

videoRemoveBtn.addEventListener('click', resetVideoState);
imageRemoveBtn.addEventListener('click', resetImageState);

videoUploadForm.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!appState.video.selectedFile) {
    setMessage('Please select a valid traffic video before analyzing.', 'error', 'video');
    return;
  }

  clearTimers();
  setBusyState('video', true);
  setStatus('Uploading video', 'processing');
  setProcessingSteps('video', 1);

  const formData = new FormData();
  formData.append('video', appState.video.selectedFile);

  fetch('/api/upload', { method: 'POST', body: formData })
    .then((response) => response.json())
    .then((payload) => {
      if (!payload.success) {
        setBusyState('video', false);
        setStatus('Upload failed', 'error');
        setMessage(payload.error || 'Upload failed. Please try another file.', 'error', 'video');
        return;
      }

      setMessage('Video uploaded successfully.', 'success', 'video');
      setStatus('Video uploaded and queued for analysis', 'ready');
      setProcessingSteps('video', 1);
      updateStepStatus('video', 'Video uploaded');

      fetch(`/api/process/${payload.job_id}`, { method: 'POST' })
        .then((response) => response.json())
        .then((processPayload) => {
          if (!processPayload.success) {
            setBusyState('video', false);
            setStatus(processPayload.error || 'Could not start analysis', 'error');
            setMessage('Unable to analyze this video. Please try another file.', 'error', 'video');
            return;
          }
          startVideoPolling(payload.job_id);
        })
        .catch(() => {
          setBusyState('video', false);
          setStatus('Processing request failed', 'error');
          setMessage('Unable to analyze this video. Please try another file.', 'error', 'video');
        });
    })
    .catch(() => {
      setBusyState('video', false);
      setStatus('Upload request failed', 'error');
      setMessage('Network issue while uploading the video. Please try again.', 'error', 'video');
    });
});

imageUploadForm.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!appState.image.selectedFile) {
    setMessage('Please select a valid traffic image before analyzing.', 'error', 'image');
    return;
  }

  clearTimers();
  setBusyState('image', true);
  setStatus('Uploading image', 'processing');
  setProcessingSteps('image', 1);

  const formData = new FormData();
  formData.append('image', appState.image.selectedFile);

  fetch('/api/upload-image', { method: 'POST', body: formData })
    .then((response) => response.json())
    .then((payload) => {
      if (!payload.success) {
        setBusyState('image', false);
        setStatus('Upload failed', 'error');
        setMessage(payload.error || 'Upload failed. Please try another file.', 'error', 'image');
        return;
      }

      setMessage('Image uploaded successfully.', 'success', 'image');
      setStatus('Image uploaded and queued for analysis', 'ready');
      setProcessingSteps('image', 1);
      updateStepStatus('image', 'Uploading image');

      fetch(`/api/process-image/${payload.job_id}`, { method: 'POST' })
        .then((response) => response.json())
        .then((processPayload) => {
          if (!processPayload.success) {
            setBusyState('image', false);
            setStatus(processPayload.error || 'Could not start image analysis', 'error');
            setMessage('Unable to analyze this image. Please try another file.', 'error', 'image');
            return;
          }
          startImagePolling(payload.job_id);
        })
        .catch(() => {
          setBusyState('image', false);
          setStatus('Image processing request failed', 'error');
          setMessage('Unable to analyze this image. Please try another file.', 'error', 'image');
        });
    })
    .catch(() => {
      setBusyState('image', false);
      setStatus('Image upload request failed', 'error');
      setMessage('Network issue while uploading the image. Please try again.', 'error', 'image');
    });
});

modeButtons.forEach((button) => {
  button.addEventListener('click', () => setMode(button.dataset.mode));
});

setMode('video');
setProcessingSteps('video', 0);
setBusyState('video', false);
setBusyState('image', false);
resetVideoState();
resetImageState();

