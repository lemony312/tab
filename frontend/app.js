/**
 * YouTube Tab Extractor - Frontend JavaScript
 * Handles form submission, job polling, and UI updates.
 */

// DOM Elements
const form = document.getElementById('extract-form');
const urlInput = document.getElementById('url-input');
const thresholdInput = document.getElementById('threshold');
const thresholdValue = document.getElementById('threshold-value');
const intervalInput = document.getElementById('interval');
const introSkipInput = document.getElementById('intro-skip');
const tabAwareInput = document.getElementById('tab-aware');
const detectionModeInput = document.getElementById('detection-mode');
const submitBtn = document.getElementById('submit-btn');
const btnText = submitBtn.querySelector('.btn-text');
const btnLoading = submitBtn.querySelector('.btn-loading');

const statusSection = document.getElementById('status-section');
const videoTitle = document.getElementById('video-title');
const statusBadge = document.getElementById('status-badge');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const statusMessage = document.getElementById('status-message');
const statsDiv = document.getElementById('stats');
const framesExtracted = document.getElementById('frames-extracted');
const tabsDetected = document.getElementById('tabs-detected');
const downloadSection = document.getElementById('download-section');
const downloadBtn = document.getElementById('download-btn');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');

// State
let currentJobId = null;
let pollInterval = null;

// Update threshold display value
thresholdInput.addEventListener('input', () => {
    thresholdValue.textContent = thresholdInput.value;
});

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const url = urlInput.value.trim();
    if (!url) return;
    
    // Reset UI
    setLoading(true);
    hideError();
    hideDownload();
    
    try {
        const response = await fetch('/extract', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                url: url,
                threshold: parseFloat(thresholdInput.value),
                min_interval: parseFloat(intervalInput.value),
                tab_aware: tabAwareInput.checked,
                intro_skip: parseFloat(introSkipInput.value),
                detection_mode: detectionModeInput.value,
            }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start extraction');
        }
        
        const data = await response.json();
        currentJobId = data.id;
        
        // Show status section and start polling
        showStatus();
        startPolling();
        
    } catch (error) {
        showError(error.message);
        setLoading(false);
    }
});

/**
 * Start polling for job status updates.
 */
function startPolling() {
    // Clear any existing interval
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    // Poll immediately, then every second
    pollStatus();
    pollInterval = setInterval(pollStatus, 1000);
}

/**
 * Stop polling for job status.
 */
function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

/**
 * Poll the server for job status.
 */
async function pollStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`/status/${currentJobId}`);
        
        if (!response.ok) {
            throw new Error('Failed to get status');
        }
        
        const data = await response.json();
        updateStatus(data);
        
        // Stop polling when job is complete or failed
        if (data.status === 'completed' || data.status === 'failed') {
            stopPolling();
            setLoading(false);
        }
        
    } catch (error) {
        console.error('Polling error:', error);
        // Don't stop polling on transient errors
    }
}

/**
 * Update the UI with job status.
 */
function updateStatus(data) {
    // Update title
    if (data.video_title) {
        videoTitle.textContent = data.video_title;
    }
    
    // Update badge
    statusBadge.textContent = formatStatus(data.status);
    statusBadge.className = `badge ${data.status}`;
    
    // Update progress
    progressFill.style.width = `${data.progress}%`;
    progressText.textContent = `${data.progress}%`;
    
    // Update message
    statusMessage.textContent = data.message || '';
    
    // Update stats
    if (data.frames_extracted > 0 || data.tabs_detected > 0) {
        framesExtracted.textContent = data.frames_extracted;
        tabsDetected.textContent = data.tabs_detected;
        statsDiv.hidden = false;
    }
    
    // Show download button when complete
    if (data.status === 'completed' && data.download_url) {
        downloadBtn.href = data.download_url;
        downloadSection.hidden = false;
    }
    
    // Show error when failed
    if (data.status === 'failed') {
        showError(data.error || 'An error occurred');
    }
}

/**
 * Format status for display.
 */
function formatStatus(status) {
    const statusMap = {
        'pending': 'Pending',
        'downloading': 'Downloading',
        'processing': 'Processing',
        'detecting': 'Detecting Tabs',
        'generating_pdf': 'Generating PDF',
        'completed': 'Completed',
        'failed': 'Failed',
    };
    return statusMap[status] || status;
}

/**
 * Set loading state on the submit button.
 */
function setLoading(loading) {
    submitBtn.disabled = loading;
    btnText.hidden = loading;
    btnLoading.hidden = !loading;
}

/**
 * Show the status section.
 */
function showStatus() {
    statusSection.hidden = false;
    videoTitle.textContent = 'Processing...';
    statusBadge.textContent = 'Pending';
    statusBadge.className = 'badge pending';
    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    statusMessage.textContent = 'Starting...';
    statsDiv.hidden = true;
    downloadSection.hidden = true;
    errorSection.hidden = true;
}

/**
 * Show an error message.
 */
function showError(message) {
    errorMessage.textContent = message;
    errorSection.hidden = false;
}

/**
 * Hide the error section.
 */
function hideError() {
    errorSection.hidden = true;
}

/**
 * Hide the download section.
 */
function hideDownload() {
    downloadSection.hidden = true;
}
