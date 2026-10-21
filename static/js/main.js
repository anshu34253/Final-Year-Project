/* ==========================================================================
   ViralShort AI - Main Application JavaScript
   Navbar scroll blur, Upload Handlers, Modal Players & UI Interactivity
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initNavbarScroll();
  initMobileMenu();
  initUploadForm();
  initTabSwitchers();
  initCopyButtons();
});

/* 1. Navbar Blur & Scrolled state */
function initNavbarScroll() {
  const navbar = document.querySelector('.navbar-custom');
  if (!navbar) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 20) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
}

/* 2. Mobile Menu Toggle */
function initMobileMenu() {
  const toggleBtn = document.querySelector('.mobile-toggle-btn');
  const navMenu = document.querySelector('.nav-menu');
  if (!toggleBtn || !navMenu) return;

  toggleBtn.addEventListener('click', () => {
    navMenu.classList.toggle('open');
    const icon = toggleBtn.querySelector('i');
    if (icon) {
      icon.classList.toggle('bi-list');
      icon.classList.toggle('bi-x-lg');
    }
  });
}

/* 3. Upload Page & Hero Section Upload Handlers */
function initUploadForm() {
  // Hero Section URL Form
  const heroUrlForm = document.getElementById('hero-url-form');
  if (heroUrlForm) {
    heroUrlForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const input = document.getElementById('hero-yt-input');
      if (input && input.value.trim() !== '') {
        handleYoutubeDownload(input.value.trim(), true);
      }
    });
  }
  // Hero Section Drag & Drop Zone
  const heroDropZone = document.getElementById('hero-drop-zone');
  const heroFileInput = document.getElementById('hero-file-input');
  if (heroDropZone && heroFileInput) {
    heroDropZone.addEventListener('click', () => heroFileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      heroDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      heroDropZone.addEventListener(eventName, () => heroDropZone.classList.add('drag-active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      heroDropZone.addEventListener(eventName, () => heroDropZone.classList.remove('drag-active'), false);
    });

    heroDropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt.files.length > 0) {
        handleUploadSubmit(dt.files[0].name);
      }
    });

    heroFileInput.addEventListener('change', () => {
      if (heroFileInput.files.length > 0) {
        handleUploadSubmit(heroFileInput.files[0].name);
      }
    });
  }

  // Upload Page Form Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-upload-input');
  const urlForm = document.getElementById('url-upload-form');

  if (dropZone && fileInput) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-active'), false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        fileInput.files = files;
        handleUploadSubmit(files[0].name);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleUploadSubmit(fileInput.files[0].name);
      }
    });
  }

  // Remove button listeners
  const removeBtn = document.getElementById('remove-url-btn');
  const urlInputEl = document.getElementById('youtube-url-input');
  if (removeBtn && urlInputEl) {
    removeBtn.addEventListener('click', () => {
      urlInputEl.value = '';
      removeBtn.style.display = 'none';
      hideVideoPreview(false);
      const feedback = document.getElementById('download-feedback');
      if (feedback) feedback.style.display = 'none';
    });
  }

  const heroRemoveBtn = document.getElementById('hero-remove-url-btn');
  const heroYtInputEl = document.getElementById('hero-yt-input');
  if (heroRemoveBtn && heroYtInputEl) {
    heroRemoveBtn.addEventListener('click', () => {
      heroYtInputEl.value = '';
      heroRemoveBtn.style.display = 'none';
      hideVideoPreview(true);
      const feedback = document.getElementById('hero-download-feedback');
      if (feedback) feedback.style.display = 'none';
    });
  }

  // URL Input Listener for Video Preview
  if (urlInputEl) {
    const checkUrlAndPreview = () => {
      const url = urlInputEl.value.trim();
      if (removeBtn) removeBtn.style.display = url !== '' ? 'inline-block' : 'none';
      if (isValidYoutubeUrl(url)) {
        fetchVideoPreview(url, false);
      } else {
        hideVideoPreview(false);
      }
    };

    ['input', 'change', 'paste', 'blur'].forEach(evt => {
      urlInputEl.addEventListener(evt, debounce(checkUrlAndPreview, 200));
    });

    if (urlInputEl.value.trim() !== '') {
      checkUrlAndPreview();
    }
  }

  if (heroYtInputEl) {
    const checkHeroUrlAndPreview = () => {
      const url = heroYtInputEl.value.trim();
      if (heroRemoveBtn) heroRemoveBtn.style.display = url !== '' ? 'inline-block' : 'none';
      if (isValidYoutubeUrl(url)) {
        fetchVideoPreview(url, true);
      } else {
        hideVideoPreview(true);
      }
    };

    ['input', 'change', 'paste', 'blur'].forEach(evt => {
      heroYtInputEl.addEventListener(evt, debounce(checkHeroUrlAndPreview, 200));
    });

    if (heroYtInputEl.value.trim() !== '') {
      checkHeroUrlAndPreview();
    }
  }

  // Preview Card Download Button Listener
  const previewDownloadBtn = document.getElementById('preview-download-btn');
  if (previewDownloadBtn) {
    previewDownloadBtn.addEventListener('click', () => {
      const urlInput = document.getElementById('youtube-url-input');
      if (urlInput && urlInput.value.trim() !== '') {
        handleYoutubeDownload(urlInput.value.trim(), false);
      }
    });
  }
}

let previewDebounceTimer = null;
function debounce(func, delay) {
  return function(...args) {
    clearTimeout(previewDebounceTimer);
    previewDebounceTimer = setTimeout(() => func.apply(this, args), delay);
  };
}

function hideVideoPreview(isHero = false) {
  const prefix = isHero ? 'hero-' : '';
  const previewCard = document.getElementById(prefix + 'video-preview-card');
  const previewLoading = document.getElementById(prefix + 'preview-loading');
  if (previewCard) previewCard.style.display = 'none';
  if (previewLoading) previewLoading.style.display = 'none';
}

function fetchVideoPreview(youtubeUrl, isHero = false) {
  const prefix = isHero ? 'hero-' : '';
  const previewCard = document.getElementById(prefix + 'video-preview-card');
  const previewLoading = document.getElementById(prefix + 'preview-loading');
  const thumbnailImg = document.getElementById(prefix + 'preview-thumbnail');
  const titleEl = document.getElementById(prefix + 'preview-title');
  const channelEl = document.getElementById(prefix + 'preview-channel');
  const durationEl = document.getElementById(prefix + 'preview-duration');
  const removeBtn = document.getElementById(isHero ? 'hero-remove-url-btn' : 'remove-url-btn');

  if (removeBtn) removeBtn.style.display = 'inline-block';
  if (previewLoading) previewLoading.style.display = 'block';
  if (previewCard) previewCard.style.display = 'none';

  fetch('/api/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ youtube_url: youtubeUrl })
  })
  .then(res => res.json())
  .then(data => {
    if (previewLoading) previewLoading.style.display = 'none';
    if (data.status === 'success' && data.metadata) {
      const meta = data.metadata;
      if (thumbnailImg) thumbnailImg.src = meta.thumbnail || '';
      if (titleEl) titleEl.textContent = meta.title || 'Video Title';
      if (channelEl) channelEl.innerHTML = `<i class="bi bi-person-circle me-1"></i> ${meta.channel || 'Channel'}`;
      if (durationEl) durationEl.textContent = meta.duration || '00:00';
      
      if (previewCard) previewCard.style.display = 'block';
    }
  })
  .catch(err => {
    console.error('Preview fetch error:', err);
    if (previewLoading) previewLoading.style.display = 'none';
  });
}

/* Validate YouTube URL client side */
function isValidYoutubeUrl(url) {
  if (!url) return false;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)\/(watch\?v=|embed\/|v\/|shorts\/|e\/)?([^&=%\?\{\}\s]{11})([\?&].*)?$/i;
  return youtubeRegex.test(url);
}

/* Handles YouTube Download including loading state, success message and error validation */
function handleYoutubeDownload(youtubeUrl, isHero = false) {
  const prefix = isHero ? 'hero-' : '';
  const feedbackContainer = document.getElementById(prefix + 'download-feedback');
  const loadingContainer = document.getElementById(prefix + 'download-loading');
  const successContainer = document.getElementById(prefix + 'download-success');
  const errorContainer = document.getElementById(prefix + 'download-error');
  const downloadedPathEl = document.getElementById(prefix + 'downloaded-path');
  const errorTextEl = document.getElementById(prefix + 'error-text');
  
  const urlInput = document.getElementById(isHero ? 'hero-yt-input' : 'youtube-url-input');
  const downloadBtn = document.getElementById(isHero ? 'hero-download-btn' : 'download-btn');
  const previewDownloadBtn = document.getElementById(prefix + 'preview-download-btn');

  // Client-side validation: URL format
  if (!isValidYoutubeUrl(youtubeUrl)) {
    showToast('Invalid YouTube URL format.', 'error');
    if (feedbackContainer && errorContainer && errorTextEl) {
      feedbackContainer.style.display = 'block';
      errorContainer.style.display = 'block';
      loadingContainer.style.display = 'none';
      successContainer.style.display = 'none';
      errorTextEl.textContent = 'Invalid YouTube URL format. Please paste a valid YouTube video link.';
    }
    return;
  }

  // Update UI elements to show loading state
  if (urlInput) urlInput.disabled = true;
  if (downloadBtn) downloadBtn.disabled = true;
  if (previewDownloadBtn) previewDownloadBtn.disabled = true;
  
  if (feedbackContainer) feedbackContainer.style.display = 'block';
  if (loadingContainer) loadingContainer.style.display = 'flex';
  if (successContainer) successContainer.style.display = 'none';
  if (errorContainer) errorContainer.style.display = 'none';

  showToast('Initializing download pipeline...', 'info');

  const numReelsEl = document.querySelector('select[name="num_reels"]');
  const captionsEl = document.querySelector('select[name="captions"]');
  const reelLengthEl = document.querySelector('select[name="reel_length"]');
  const aspectRatioEl = document.querySelector('select[name="aspect_ratio"]');

  const payload = {
    upload_type: 'youtube',
    youtube_url: youtubeUrl,
    num_reels: numReelsEl ? numReelsEl.value : 'auto',
    captions: captionsEl ? captionsEl.value : 'on',
    reel_length: reelLengthEl ? reelLengthEl.value : '30-60',
    aspect_ratio: aspectRatioEl ? aspectRatioEl.value : '9:16'
  };

  fetch('/api/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(async (res) => {
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.message || 'Processing service returned an error.');
    }
    return data;
  })
  .then(data => {
    if (data.redirect_url) {
      showToast('Redirecting to AI processing engine...', 'success');
      window.location.href = data.redirect_url;
    } else {
      showToast('Processing initialized!', 'success');
    }
  })
  .catch(err => {
    if (urlInput) urlInput.disabled = false;
    if (downloadBtn) downloadBtn.disabled = false;
    if (previewDownloadBtn) previewDownloadBtn.disabled = false;
    
    if (loadingContainer) loadingContainer.style.display = 'none';
    if (errorContainer) errorContainer.style.display = 'block';
    if (errorTextEl) errorTextEl.textContent = err.message || 'An error occurred during submission.';
    
    showToast('Processing initialization failed.', 'error');
    console.error('Download Failure Details:', err);
  });
}


/* Submit Upload and trigger API redirection to Processing view */
function handleUploadSubmit(sourceName) {
  showToast(`Submitting source video...`, 'info');

  fetch('/api/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ upload_type: 'youtube', youtube_url: sourceName })
  })
  .then(res => res.json())
  .then(data => {
    if (data.redirect_url) {
      window.location.href = data.redirect_url;
    }
  })
  .catch(err => {
    console.error('Upload Error:', err);
  });
}

/* 4. Tab Switcher Utility & Global Hero Handlers */
function switchInputTab(tab) {
  const ytBtn = document.getElementById('tab-yt-btn');
  const fileBtn = document.getElementById('tab-file-btn');
  const ytContent = document.getElementById('tab-yt-content');
  const fileContent = document.getElementById('tab-file-content');

  if (tab === 'yt') {
    if (ytBtn) ytBtn.classList.add('active');
    if (fileBtn) fileBtn.classList.remove('active');
    if (ytContent) ytContent.style.display = 'block';
    if (fileContent) fileContent.style.display = 'none';
  } else {
    if (fileBtn) fileBtn.classList.add('active');
    if (ytBtn) ytBtn.classList.remove('active');
    if (fileContent) fileContent.style.display = 'block';
    if (ytContent) ytContent.style.display = 'none';
  }
}

function handleFileSelected(input) {
  if (input && input.files && input.files.length > 0) {
    const file = input.files[0];
    const infoContainer = document.getElementById('selected-file-info');
    const fileNameEl = document.getElementById('selected-file-name');
    if (infoContainer && fileNameEl) {
      fileNameEl.textContent = file.name + ' (' + (file.size / (1024 * 1024)).toFixed(1) + ' MB)';
      infoContainer.style.display = 'block';
    }
  }
}

function initTabSwitchers() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      const parentGroup = btn.closest('.tab-container');
      if (!parentGroup) return;

      parentGroup.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      parentGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const activeContent = parentGroup.querySelector(`#tab-${targetTab}`);
      if (activeContent) activeContent.classList.add('active');
    });
  });
}

/* 5. Copy Link Toast Notification */
function initCopyButtons() {
  document.querySelectorAll('.btn-copy-link').forEach(btn => {
    btn.addEventListener('click', () => {
      const copyText = btn.getAttribute('data-copy') || window.location.href;
      navigator.clipboard.writeText(copyText).then(() => {
        showToast('Link copied to clipboard!', 'success');
      });
    });
  });
}

/* Toast System */
function showToast(message, type = 'info') {
  let toastContainer = document.querySelector('.toast-container-custom');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container-custom';
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  toast.className = `toast-item toast-${type}`;
  toast.innerHTML = `<i class="bi bi-check-circle-fill me-2"></i> ${message}`;

  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('show');
  }, 10);

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
