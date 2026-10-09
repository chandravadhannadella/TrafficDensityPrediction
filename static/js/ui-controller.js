// New UI Controller for the Redesigned Frontend
document.addEventListener('DOMContentLoaded', () => {
  initializeNavigation();
  initializeModeSwitching();
  initializeFormHandlers();
  initializeCardButtons();
});

// State to track if results exist
const appState = {
  hasImageResults: false,
  hasVideoResults: false,
};

// Navigation Management
function initializeNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const sections = document.querySelectorAll('.section');

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const section = item.getAttribute('data-section');
      showSection(section);
      updateNavigation(section);
    });
  });

  function updateNavigation(activeSection) {
    navItems.forEach(item => {
      if (item.getAttribute('data-section') === activeSection) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });
  }

  function showSection(sectionId) {
    sections.forEach(section => {
      if (section.id === sectionId) {
        section.classList.remove('hidden');
        updatePageTitle(sectionId);
        
        // Handle empty states for results sections
        if (sectionId === 'image-results') {
          handleImageResultsView();
        } else if (sectionId === 'video-results') {
          handleVideoResultsView();
        }
      } else {
        section.classList.add('hidden');
      }
    });
  }

  function handleImageResultsView() {
    const emptyState = document.getElementById('image-results-empty');
    const resultsContent = document.getElementById('image-results-content');
    
    if (appState.hasImageResults) {
      emptyState.classList.add('hidden');
      resultsContent.classList.remove('hidden');
    } else {
      emptyState.classList.remove('hidden');
      resultsContent.classList.add('hidden');
    }
  }

  function handleVideoResultsView() {
    const emptyState = document.getElementById('video-results-empty');
    const resultsContent = document.getElementById('video-results-content');
    
    if (appState.hasVideoResults) {
      emptyState.classList.add('hidden');
      resultsContent.classList.remove('hidden');
    } else {
      emptyState.classList.remove('hidden');
      resultsContent.classList.add('hidden');
    }
  }

  function updatePageTitle(sectionId) {
    const titles = {
      'dashboard': { title: 'Dashboard', subtitle: 'Traffic density intelligence' },
      'analyze': { title: 'Analysis Workspace', subtitle: 'Choose an input source and start analysis' },
      'image-results': { title: 'Image Analysis Result', subtitle: 'Traffic density assessment' },
      'video-results': { title: 'Video Analysis Results', subtitle: 'Traffic timeline analysis' },
    };

    const config = titles[sectionId] || titles['dashboard'];
    document.getElementById('page-title').textContent = config.title;
    document.getElementById('page-subtitle').textContent = config.subtitle;
  }
}

// Mode Switching (Image/Video)
function initializeModeSwitching() {
  const tabButtons = document.querySelectorAll('.tab-button');
  const modeContents = document.querySelectorAll('.mode-content');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mode');
      
      // Update tabs
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Update content
      modeContents.forEach(content => {
        if (content.id === `${mode}-mode`) {
          content.classList.add('active');
          content.classList.remove('hidden');
        } else {
          content.classList.remove('active');
          content.classList.add('hidden');
        }
      });
    });
  });
}

// Dashboard Card Buttons
function initializeCardButtons() {
  const cardButtons = document.querySelectorAll('[data-action]');

  cardButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.getAttribute('data-action');
      
      if (action === 'goto-analyze-image') {
        navigateToAnalyze('image');
      } else if (action === 'goto-analyze-video') {
        navigateToAnalyze('video');
      }
    });
  });

  function navigateToAnalyze(mode) {
    // Show analyze section
    document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
    document.getElementById('analyze').classList.remove('hidden');
    
    // Switch to correct mode
    const tabButtons = document.querySelectorAll('.tab-button');
    const modeContents = document.querySelectorAll('.mode-content');
    
    tabButtons.forEach(btn => {
      if (btn.getAttribute('data-mode') === mode) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    modeContents.forEach(content => {
      if (content.id === `${mode}-mode`) {
        content.classList.add('active');
        content.classList.remove('hidden');
      } else {
        content.classList.remove('active');
        content.classList.add('hidden');
      }
    });

    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
      if (item.getAttribute('data-section') === 'analyze') {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });
  }
}

// Form Handlers - Bridge to existing backend
function initializeFormHandlers() {
  setupImageForm();
  setupVideoForm();
}

function setupImageForm() {
  const form = document.getElementById('image-upload-form');
  const input = document.getElementById('image-input');
  const dropZone = document.getElementById('image-drop-zone');
  const fileMeta = document.getElementById('image-file-meta');
  const removeBtn = document.getElementById('image-remove-btn');
  const analyzeBtn = document.getElementById('image-analyze-btn');
  const preview = document.getElementById('image-preview');
  const previewSection = document.getElementById('image-preview-section');

  // File selection
  input.addEventListener('change', (e) => {
    handleImageFileSelect(e.target.files[0]);
  });

  // Drag and drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragging');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragging');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) {
      handleImageFileSelect(e.dataTransfer.files[0]);
      input.files = e.dataTransfer.files;
    }
  });

  // Remove button
  removeBtn.addEventListener('click', () => {
    input.value = '';
    fileMeta.classList.add('hidden');
    removeBtn.classList.add('hidden');
    analyzeBtn.disabled = true;
    previewSection.classList.add('hidden');
  });

  // Analyze button
  analyzeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    if (input.files[0]) {
      submitImageForm();
    }
  });

  function handleImageFileSelect(file) {
    if (!file) return;

    // Update file meta
    document.getElementById('image-selected-name').textContent = file.name;
    document.getElementById('image-selected-size').textContent = formatFileSize(file.size);
    document.getElementById('image-selected-type').textContent = file.type || 'Image file';
    fileMeta.classList.remove('hidden');

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      document.getElementById('image-preview-size').textContent = `${formatFileSize(file.size)} • ${file.name}`;
      previewSection.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Enable buttons
    removeBtn.classList.remove('hidden');
    analyzeBtn.disabled = false;
  }

  function submitImageForm() {
    showProcessing();
    const formData = new FormData();
    formData.append('image', input.files[0]);

    // Step 1: Upload image
    fetch('/api/upload-image', {
      method: 'POST',
      body: formData
    })
    .then(r => r.json())
    .then(uploadData => {
      if (!uploadData.success) {
        throw new Error(uploadData.error || 'Upload failed');
      }
      updateProcessingStep(2);
      // Step 2: Start processing
      return fetch(`/api/process-image/${uploadData.job_id}`, { method: 'POST' })
        .then(r => r.json())
        .then(processData => {
          if (!processData.success) {
            throw new Error(processData.error || 'Could not start analysis');
          }
          // Step 3: Poll for completion
          return pollImageStatus(uploadData.job_id);
        });
    })
    .then(resultData => {
      hideProcessing();
      displayImageResult(resultData);
    })
    .catch(err => {
      hideProcessing();
      showError('Image analysis failed. Please try again.');
      console.error(err);
    });
  }

  function pollImageStatus(jobId) {
    return new Promise((resolve, reject) => {
      const poll = () => {
        fetch(`/api/status/${jobId}`)
          .then(r => r.json())
          .then(payload => {
            const percent = Math.min(Number(payload.progress || 0), 100);
            // Update progress in modal if needed
            if (payload.status === 'processing') {
              updateProcessingStep(3);
              setTimeout(poll, 1200);
              return;
            }
            if (payload.status === 'completed') {
              updateProcessingStep(4);
              // Step 4: Get results
              fetch(`/api/results/${jobId}`)
                .then(r => r.json())
                .then(result => {
                  if (result && result.success) {
                    resolve(result);
                  } else {
                    reject(new Error('Unable to load results'));
                  }
                })
                .catch(reject);
              return;
            }
            if (payload.status === 'failed') {
              reject(new Error(payload.error || 'Analysis failed'));
              return;
            }
            setTimeout(poll, 1200);
          })
          .catch(reject);
      };
      poll();
    });
  }
}

function setupVideoForm() {
  const form = document.getElementById('video-upload-form');
  const input = document.getElementById('video-input');
  const dropZone = document.getElementById('video-drop-zone');
  const fileMeta = document.getElementById('video-file-meta');
  const removeBtn = document.getElementById('video-remove-btn');
  const analyzeBtn = document.getElementById('video-analyze-btn');
  const preview = document.getElementById('video-preview');
  const previewSection = document.getElementById('video-preview-section');

  // File selection
  input.addEventListener('change', (e) => {
    handleVideoFileSelect(e.target.files[0]);
  });

  // Drag and drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragging');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragging');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) {
      handleVideoFileSelect(e.dataTransfer.files[0]);
      input.files = e.dataTransfer.files;
    }
  });

  // Remove button
  removeBtn.addEventListener('click', () => {
    input.value = '';
    fileMeta.classList.add('hidden');
    removeBtn.classList.add('hidden');
    analyzeBtn.disabled = true;
    previewSection.classList.add('hidden');
  });

  // Analyze button
  analyzeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    if (input.files[0]) {
      submitVideoForm();
    }
  });

  function handleVideoFileSelect(file) {
    if (!file) return;

    // Update file meta
    document.getElementById('video-selected-name').textContent = file.name;
    document.getElementById('video-selected-size').textContent = formatFileSize(file.size);
    document.getElementById('video-selected-type').textContent = file.type || 'Video file';
    fileMeta.classList.remove('hidden');

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      document.getElementById('video-preview-size').textContent = `${formatFileSize(file.size)} • ${file.name}`;
      previewSection.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Enable buttons
    removeBtn.classList.remove('hidden');
    analyzeBtn.disabled = false;
  }

  function submitVideoForm() {
    showProcessing();
    const formData = new FormData();
    formData.append('video', input.files[0]);

    // Step 1: Upload video
    fetch('/api/upload', {
      method: 'POST',
      body: formData
    })
    .then(r => r.json())
    .then(uploadData => {
      if (!uploadData.success) {
        throw new Error(uploadData.error || 'Upload failed');
      }
      updateProcessingStep(2);
      // Step 2: Start processing
      return fetch(`/api/process/${uploadData.job_id}`, { method: 'POST' })
        .then(r => r.json())
        .then(processData => {
          if (!processData.success) {
            throw new Error(processData.error || 'Could not start analysis');
          }
          // Step 3: Poll for completion
          return pollVideoStatus(uploadData.job_id);
        });
    })
    .then(resultData => {
      hideProcessing();
      displayVideoResult(resultData);
    })
    .catch(err => {
      hideProcessing();
      showError('Video analysis failed. Please try again.');
      console.error(err);
    });
  }

  function pollVideoStatus(jobId) {
    return new Promise((resolve, reject) => {
      const poll = () => {
        fetch(`/api/status/${jobId}`)
          .then(r => r.json())
          .then(payload => {
            const percent = Math.min(Number(payload.progress || 0), 100);
            // Update progress in modal if needed
            if (payload.status === 'processing') {
              updateProcessingStep(3);
              setTimeout(poll, 1200);
              return;
            }
            if (payload.status === 'completed') {
              updateProcessingStep(4);
              // Step 4: Get results
              fetch(`/api/results/${jobId}`)
                .then(r => r.json())
                .then(result => {
                  if (result && result.success) {
                    resolve(result);
                  } else {
                    reject(new Error('Unable to load results'));
                  }
                })
                .catch(reject);
              return;
            }
            if (payload.status === 'failed') {
              reject(new Error(payload.error || 'Analysis failed'));
              return;
            }
            setTimeout(poll, 1200);
          })
          .catch(reject);
      };
      poll();
    });
  }
}

// Results Display
function displayImageResult(data) {
  const filename = data.filename || 'Analysis Result';
  const confidence = (data.overall_confidence * 100).toFixed(1);
  const density = data.overall_traffic_density || 'LOW';

  // Update result section
  document.getElementById('image-result-filename').textContent = filename;
  document.getElementById('result-image').src = `/data/images/processed/${filename}` || '/data/images/raw/' + filename;
  document.getElementById('image-result-confidence').textContent = `${confidence}%`;
  document.getElementById('image-result-density').textContent = density;
  document.getElementById('image-result-density').className = `density-display ${density.toLowerCase()}`;
  document.getElementById('image-confidence-fill').style.width = `${parseFloat(confidence)}%`;

  // Vehicle breakdown
  const detectionList = document.getElementById('image-detection-list');
  detectionList.innerHTML = '';
  if (data.counts) {
    const formatter = {
      car: 'Cars',
      bus: 'Buses',
      truck: 'Trucks',
      motorcycle: 'Motorcycles',
      bicycle: 'Bicycles',
    };
    Object.entries(data.counts).forEach(([type, count]) => {
      const item = document.createElement('div');
      item.className = 'vehicle-item';
      item.innerHTML = `<span>${formatter[type] || type}</span><strong>${count}</strong>`;
      detectionList.appendChild(item);
    });
  }

  document.getElementById('image-total-vehicles').textContent = data.total_vehicles || 0;
  document.getElementById('image-result-summary').textContent = `${data.total_vehicles} vehicles detected. Traffic density classified as ${density}.`;

  // Set state and show results content
  appState.hasImageResults = true;
  const emptyState = document.getElementById('image-results-empty');
  const resultsContent = document.getElementById('image-results-content');
  emptyState.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  // Navigate to results
  document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
  document.getElementById('image-results').classList.remove('hidden');
  document.querySelectorAll('.nav-item').forEach(item => {
    if (item.getAttribute('data-section') === 'image-results') {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });
  document.getElementById('page-title').textContent = 'Image Analysis Result';
  document.getElementById('page-subtitle').textContent = filename;
}

function displayVideoResult(data) {
  const filename = data.filename || 'Analysis Result';
  const previewVideoFilename = data.preview_video_filename || data.annotated_video_filename || filename;
  const windows = data.window_predictions || [];

  // Update metrics
  document.getElementById('metric-windows').textContent = data.windows_processed || windows.length;
  document.getElementById('metric-confidence').textContent = `${((data.overall_confidence || 0) * 100).toFixed(1)}%`;
  document.getElementById('metric-dominant').textContent = data.overall_traffic_density || 'LOW';
  document.getElementById('metric-vehicles').textContent = data.total_vehicles || 0;

  // Update timeline
  const timeline = document.getElementById('traffic-timeline');
  timeline.innerHTML = '';
  windows.forEach((w, i) => {
    const segment = document.createElement('div');
    const trafficLevel = w.predicted_traffic_density || 'LOW';
    segment.className = `timeline-segment ${trafficLevel.toLowerCase()}`;
    segment.textContent = `${Math.round(w.window_start || i * 10)}s`;
    timeline.appendChild(segment);
  });

  // Update table
  const tableBody = document.getElementById('window-table-body');
  tableBody.innerHTML = '';
  windows.forEach((w, i) => {
    const trafficLevel = w.predicted_traffic_density || 'LOW';
    const probs = w.probabilities || {};
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${Math.round(w.window_start || i * 10)}-${Math.round(w.window_end || (i + 1) * 10)}s</td>
      <td><span class="badge ${trafficLevel.toLowerCase()}">${trafficLevel}</span></td>
      <td>${((w.confidence || 0) * 100).toFixed(1)}%</td>
      <td>${(probs.LOW || 0) * 100}%</td>
      <td>${(probs.MEDIUM || 0) * 100}%</td>
      <td>${(probs.HIGH || 0) * 100}%</td>
    `;
    tableBody.appendChild(row);
  });

  // Distribution
  const low = data.low_windows || windows.filter(w => w.predicted_traffic_density === 'LOW').length;
  const medium = data.medium_windows || windows.filter(w => w.predicted_traffic_density === 'MEDIUM').length;
  const high = data.high_windows || windows.filter(w => w.predicted_traffic_density === 'HIGH').length;
  
  document.getElementById('dist-low').textContent = low;
  document.getElementById('dist-medium').textContent = medium;
  document.getElementById('dist-high').textContent = high;
  document.getElementById('video-result-summary').textContent = `Predominantly ${data.overall_traffic_density} traffic detected. Video contains ${data.windows_processed || windows.length} analysis windows.`;

  // Video player - use preview video filename (browser-compatible)
  document.getElementById('result-video').src = `/data/videos/inference/${previewVideoFilename}` || `/uploads/videos/${filename}`;
  document.getElementById('video-result-filename').textContent = filename;

  // Set state and show results content
  appState.hasVideoResults = true;
  const emptyState = document.getElementById('video-results-empty');
  const resultsContent = document.getElementById('video-results-content');
  emptyState.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  // Navigate to results
  document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
  document.getElementById('video-results').classList.remove('hidden');
  document.querySelectorAll('.nav-item').forEach(item => {
    if (item.getAttribute('data-section') === 'video-results') {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });
  document.getElementById('page-title').textContent = 'Video Analysis Results';
  document.getElementById('page-subtitle').textContent = filename;
}

// UI Helpers
function showProcessing() {
  const modal = document.getElementById('processing-modal');
  modal.classList.remove('hidden');
  updateProcessingStep(1);
}

function hideProcessing() {
  const modal = document.getElementById('processing-modal');
  modal.classList.add('hidden');
}

function updateProcessingStep(step) {
  const steps = document.querySelectorAll('.step-item');
  steps.forEach((s, i) => {
    const stepNum = i + 1;
    if (stepNum < step) {
      s.classList.add('completed');
      s.classList.remove('active');
    } else if (stepNum === step) {
      s.classList.add('active');
      s.classList.remove('completed');
    } else {
      s.classList.remove('active', 'completed');
    }
  });
}

function showError(message) {
  const modal = document.getElementById('processing-modal');
  modal.classList.add('hidden');
  alert(message);
}

function formatFileSize(bytes) {
  if (!bytes) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit++;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
