/**
 * static/js/editor.js
 * -------------------
 * Complete Browser-Based Video Editor Controller for VideoHighlightAI.
 * Inspired by CapCut and OpusClip.
 *
 * Implements:
 * - Real-time client-side video preview with CSS filter adjustments.
 * - Interactive multi-track timeline with draggable playhead & trim handles.
 * - Split at playhead functionality.
 * - Undo / Redo history management with keyboard shortcuts.
 * - Aspect ratio masking (9:16, 16:9, 1:1, 4:5) & manual/smart crop positioning.
 * - Synchronized live subtitle & text/emoji overlay rendering.
 * - Audio controls, background music upload & mixing settings.
 * - Project persistence, reset, and background FFmpeg export status polling.
 */

(function () {
  'use strict';

  // Global Context injected by editor.html template
  const JOB_ID = window.EDITOR_CONFIG.jobId;
  const CLIP_ID = window.EDITOR_CONFIG.clipId;
  const MEDIA_URL = window.EDITOR_CONFIG.mediaUrl;
  const INITIAL_STATE = window.EDITOR_CONFIG.initialProject;

  // Working Project State
  let project = JSON.parse(JSON.stringify(INITIAL_STATE));
  let hasUnsavedChanges = false;

  // Undo / Redo History Stack
  const history = [];
  let historyIndex = -1;
  const MAX_HISTORY = 40;

  // DOM Elements
  const video = document.getElementById('stageVideo');
  const stageViewport = document.getElementById('stageViewport');
  const stageSubtitleOverlay = document.getElementById('stageSubtitleOverlay');
  const subtitleTextBox = document.getElementById('subtitleTextBox');
  const stageOverlaysContainer = document.getElementById('stageOverlaysContainer');

  // Timeline DOM
  const timelineScroll = document.getElementById('timelineScroll');
  const timelineRuler = document.getElementById('timelineRuler');
  const timelinePlayhead = document.getElementById('timelinePlayhead');
  const videoClipTrack = document.getElementById('videoClipTrack');
  const trimHandleLeft = document.getElementById('trimHandleLeft');
  const trimHandleRight = document.getElementById('trimHandleRight');
  const subTrackLane = document.getElementById('subTrackLane');
  const overlayTrackLane = document.getElementById('overlayTrackLane');

  // Playback DOM
  const playPauseBtn = document.getElementById('playPauseBtn');
  const timeDisplay = document.getElementById('timeDisplay');
  const volumeSlider = document.getElementById('volumeSlider');
  const muteBtn = document.getElementById('muteBtn');
  const speedSelect = document.getElementById('speedSelect');

  // Header DOM
  const undoBtn = document.getElementById('undoBtn');
  const redoBtn = document.getElementById('redoBtn');
  const saveBtn = document.getElementById('saveBtn');
  const unsavedBadge = document.getElementById('unsavedBadge');

  // Timeline Scale Settings (pixels per second)
  let zoomPixelsPerSec = 20; // 20px per second by default

  // =========================================================================
  // 1. INITIALIZATION & HISTORY
  // =========================================================================

  function init() {
    setupEventListeners();
    pushHistory('Initial state', true);
    applyAspectPreset(project.crop.aspect_ratio || '9:16');
    applyVisualAdjustments();
    populateUI();
    renderTimelineRuler();
    updateTimelineClipPosition();
    renderTimelineSubtitles();
    renderTimelineOverlays();

    // Set video src
    if (MEDIA_URL) {
      video.src = MEDIA_URL;
      video.load();
    }

    // Set initial volume & speed
    video.volume = project.audio.original_volume || 1.0;
    video.playbackRate = project.speed || 1.0;
  }

  function pushHistory(actionName, isInitial = false) {
    if (!isInitial) {
      // Discard future states if branching
      if (historyIndex < history.length - 1) {
        history.splice(historyIndex + 1);
      }
    }

    history.push({
      action: actionName,
      state: JSON.parse(JSON.stringify(project))
    });

    if (history.length > MAX_HISTORY) {
      history.shift();
    } else {
      historyIndex++;
    }

    if (!isInitial) {
      setUnsaved(true);
    }
    updateHistoryButtons();
  }

  function undo() {
    if (historyIndex > 0) {
      historyIndex--;
      project = JSON.parse(JSON.stringify(history[historyIndex].state));
      applyAllProjectState();
      updateHistoryButtons();
      setUnsaved(true);
    }
  }

  function redo() {
    if (historyIndex < history.length - 1) {
      historyIndex++;
      project = JSON.parse(JSON.stringify(history[historyIndex].state));
      applyAllProjectState();
      updateHistoryButtons();
      setUnsaved(true);
    }
  }

  function updateHistoryButtons() {
    undoBtn.disabled = historyIndex <= 0;
    redoBtn.disabled = historyIndex >= history.length - 1;
  }

  function setUnsaved(status) {
    hasUnsavedChanges = status;
    unsavedBadge.style.display = status ? 'inline-flex' : 'none';
  }

  function applyAllProjectState() {
    applyAspectPreset(project.crop.aspect_ratio);
    applyVisualAdjustments();
    populateUI();
    updateTimelineClipPosition();
    renderTimelineSubtitles();
    renderTimelineOverlays();
  }

  // =========================================================================
  // 2. VIDEO PLAYBACK & PREVIEW
  // =========================================================================

  function togglePlayPause() {
    if (video.paused || video.ended) {
      // If outside trim bounds, seek to start of trim
      const s = project.trim.start || 0;
      const e = project.trim.end || video.duration;
      if (video.currentTime < s || video.currentTime >= e) {
        video.currentTime = s;
      }
      video.play();
    } else {
      video.pause();
    }
  }

  function stepTime(seconds) {
    const s = project.trim.start || 0;
    const e = project.trim.end || (video.duration || 30);
    let target = video.currentTime + seconds;
    target = Math.max(s, Math.min(e, target));
    video.currentTime = target;
  }

  function onVideoTimeUpdate() {
    const t = video.currentTime;
    const s = project.trim.start || 0;
    const e = project.trim.end || (video.duration || 30);

    // Update playhead position on timeline
    const playheadPx = t * zoomPixelsPerSec;
    timelinePlayhead.style.left = `${playheadPx}px`;

    // Loop within trim range during playback if enabled
    if (!video.paused && t >= e) {
      video.currentTime = s;
    }

    // Update formatted time display
    const fmt = (sec) => {
      const m = Math.floor(sec / 60);
      const ss = Math.floor(sec % 60);
      const cs = Math.floor((sec % 1) * 10);
      return `${m.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}.${cs}`;
    };
    timeDisplay.innerText = `${fmt(t)} / ${fmt(e - s)}`;

    // Render Live Subtitles on Stage
    updateLiveSubtitles(t);

    // Render Live Overlays on Stage
    updateLiveOverlays(t);
  }

  function updateLiveSubtitles(currentTime) {
    const subsCfg = project.subtitles || {};
    if (!subsCfg.enabled) {
      stageSubtitleOverlay.style.display = 'none';
      return;
    }

    const segments = subsCfg.segments || [];
    const activeSeg = segments.find(seg => currentTime >= seg.start && currentTime <= seg.end);

    if (activeSeg && activeSeg.text) {
      stageSubtitleOverlay.style.display = 'block';
      subtitleTextBox.innerText = activeSeg.text;

      // Apply styling
      subtitleTextBox.style.fontFamily = subsCfg.font_family || 'Arial';
      subtitleTextBox.style.fontSize = `${subsCfg.font_size ? subsCfg.font_size * 0.45 : 20}px`;
      subtitleTextBox.style.color = subsCfg.color || '#FFFFFF';
      subtitleTextBox.style.fontWeight = subsCfg.bold ? '800' : '400';
      subtitleTextBox.style.backgroundColor = subsCfg.bg_color || 'transparent';

      // Position class
      stageSubtitleOverlay.className = `stage-subtitle-overlay pos-${subsCfg.position || 'bottom'}`;
    } else {
      stageSubtitleOverlay.style.display = 'none';
    }
  }

  function updateLiveOverlays(currentTime) {
    const overlays = project.overlays || [];
    stageOverlaysContainer.innerHTML = '';

    overlays.forEach(ov => {
      if (currentTime >= ov.start_sec && currentTime <= ov.end_sec) {
        const div = document.createElement('div');
        div.className = 'stage-overlay-item';
        div.style.left = `${ov.x_pct}%`;
        div.style.top = `${ov.y_pct}%`;
        div.style.fontSize = `${(ov.font_size || 36) * 0.5}px`;
        div.style.color = ov.color || '#FFD700';
        div.innerText = ov.content;
        div.setAttribute('data-overlay-id', ov.id);
        stageOverlaysContainer.appendChild(div);
      }
    });
  }

  // =========================================================================
  // 3. VISUAL ADJUSTMENTS & ASPECT RATIO
  // =========================================================================

  function applyAspectPreset(aspect) {
    project.crop.aspect_ratio = aspect;
    stageViewport.className = `stage-viewport aspect-${aspect.replace(':', '-')}`;

    // Highlight button in properties panel
    document.querySelectorAll('.aspect-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-aspect') === aspect);
    });
  }

  function applyVisualAdjustments() {
    const adj = project.adjustments || {};
    const b = 1 + (adj.brightness || 0); // CSS brightness 1.0 is default
    const c = adj.contrast !== undefined ? adj.contrast : 1.0;
    const s = adj.saturation !== undefined ? adj.saturation : 1.0;
    const preset = adj.filter_preset || 'normal';

    let filterStr = `brightness(${b}) contrast(${c}) saturate(${s})`;
    if (preset === 'bw') {
      filterStr += ' grayscale(100%)';
    } else if (preset === 'warm') {
      filterStr += ' sepia(20%)';
    } else if (preset === 'cool') {
      filterStr += ' hue-rotate(15deg)';
    } else if (preset === 'vivid') {
      filterStr = `brightness(${b}) contrast(${c * 1.15}) saturate(${s * 1.35})`;
    } else if (preset === 'dramatic') {
      filterStr = `brightness(${b}) contrast(${c * 1.3}) saturate(${s * 0.85})`;
    }

    video.style.filter = filterStr;
  }

  // =========================================================================
  // 4. TIMELINE CONTROLLER & SPLIT AT PLAYHEAD
  // =========================================================================

  function renderTimelineRuler() {
    const dur = project.source_duration || (video.duration || 60);
    const totalPx = Math.max(1200, dur * zoomPixelsPerSec + 200);
    timelineScroll.querySelector('.timeline-content-wrapper').style.width = `${totalPx}px`;

    timelineRuler.innerHTML = '';
    const step = zoomPixelsPerSec < 15 ? 5 : (zoomPixelsPerSec < 35 ? 2 : 1);

    for (let sec = 0; sec <= dur; sec += step) {
      const tick = document.createElement('div');
      tick.className = 'ruler-tick';
      tick.style.left = `${sec * zoomPixelsPerSec}px`;

      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      tick.innerText = `${m}:${s.toString().padStart(2, '0')}`;
      timelineRuler.appendChild(tick);
    }
  }

  function updateTimelineClipPosition() {
    const s = project.trim.start || 0;
    const e = project.trim.end || (project.source_duration || 30);

    const leftPx = s * zoomPixelsPerSec;
    const widthPx = Math.max(20, (e - s) * zoomPixelsPerSec);

    videoClipTrack.style.left = `${leftPx}px`;
    videoClipTrack.style.width = `${widthPx}px`;

    // Update numeric inputs in panel
    const trimStartIn = document.getElementById('trimStartInput');
    const trimEndIn = document.getElementById('trimEndInput');
    if (trimStartIn) trimStartIn.value = s.toFixed(1);
    if (trimEndIn) trimEndIn.value = e.toFixed(1);

    const durBadge = document.getElementById('clipDurationBadge');
    if (durBadge) durBadge.innerText = `${(e - s).toFixed(1)}s`;
  }

  function renderTimelineSubtitles() {
    subTrackLane.innerHTML = '';
    const segs = project.subtitles?.segments || [];
    segs.forEach(seg => {
      const block = document.createElement('div');
      block.className = 'timeline-sub-block';
      const leftPx = seg.start * zoomPixelsPerSec;
      const widthPx = Math.max(12, (seg.end - seg.start) * zoomPixelsPerSec);
      block.style.left = `${leftPx}px`;
      block.style.width = `${widthPx}px`;
      block.innerText = seg.text;
      block.title = `${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s: ${seg.text}`;
      subTrackLane.appendChild(block);
    });
  }

  function renderTimelineOverlays() {
    overlayTrackLane.innerHTML = '';
    const overlays = project.overlays || [];
    overlays.forEach(ov => {
      const block = document.createElement('div');
      block.className = 'timeline-overlay-block';
      const leftPx = ov.start_sec * zoomPixelsPerSec;
      const widthPx = Math.max(12, (ov.end_sec - ov.start_sec) * zoomPixelsPerSec);
      block.style.left = `${leftPx}px`;
      block.style.width = `${widthPx}px`;
      block.innerText = `${ov.type === 'emoji' ? '😀' : 'T:'} ${ov.content}`;
      overlayTrackLane.appendChild(block);
    });
  }

  // Split Clip at Current Playhead
  function splitClipAtPlayhead() {
    const t = video.currentTime;
    const s = project.trim.start || 0;
    const e = project.trim.end || (video.duration || 30);

    if (t > s + 1.0 && t < e - 1.0) {
      // Split current active range into two segments, keeping the segment user is on
      project.trim.end = parseFloat(t.toFixed(1));
      project.trim.duration = parseFloat((project.trim.end - project.trim.start).toFixed(1));
      pushHistory('Split clip at playhead');
      updateTimelineClipPosition();
      alert(`Clip trimmed up to split point (${t.toFixed(1)}s).`);
    } else {
      alert('Playhead must be at least 1.0s away from start and end points to split.');
    }
  }

  // Dragging handles logic
  let isDraggingLeftHandle = false;
  let isDraggingRightHandle = false;
  let isScrubbingTimeline = false;

  function onTimelineMouseDown(e) {
    if (e.target === trimHandleLeft) {
      isDraggingLeftHandle = true;
      e.stopPropagation();
    } else if (e.target === trimHandleRight) {
      isDraggingRightHandle = true;
      e.stopPropagation();
    } else {
      isScrubbingTimeline = true;
      seekFromMouseEvent(e);
    }
  }

  function onTimelineMouseMove(e) {
    if (!isDraggingLeftHandle && !isDraggingRightHandle && !isScrubbingTimeline) return;

    const rect = timelineScroll.querySelector('.timeline-content-wrapper').getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const targetSec = Math.max(0, clickX / zoomPixelsPerSec);

    if (isDraggingLeftHandle) {
      const endSec = project.trim.end || 30;
      if (targetSec < endSec - 1.0) {
        project.trim.start = parseFloat(targetSec.toFixed(1));
        project.trim.duration = parseFloat((endSec - project.trim.start).toFixed(1));
        updateTimelineClipPosition();
      }
    } else if (isDraggingRightHandle) {
      const startSec = project.trim.start || 0;
      if (targetSec > startSec + 1.0) {
        project.trim.end = parseFloat(targetSec.toFixed(1));
        project.trim.duration = parseFloat((project.trim.end - startSec).toFixed(1));
        updateTimelineClipPosition();
      }
    } else if (isScrubbingTimeline) {
      seekFromMouseEvent(e);
    }
  }

  function onTimelineMouseUp() {
    if (isDraggingLeftHandle || isDraggingRightHandle) {
      pushHistory('Trim adjusted');
    }
    isDraggingLeftHandle = false;
    isDraggingRightHandle = false;
    isScrubbingTimeline = false;
  }

  function seekFromMouseEvent(e) {
    const rect = timelineScroll.querySelector('.timeline-content-wrapper').getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const targetSec = Math.max(0, clickX / zoomPixelsPerSec);
    video.currentTime = targetSec;
  }

  // =========================================================================
  // 5. SIDEBAR TOOL TABS & PROPERTIES BINDINGS
  // =========================================================================

  function setupToolTabs() {
    const tabs = document.querySelectorAll('.tool-tab-btn');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const toolName = tab.getAttribute('data-tool');
        showPropertyPanel(toolName);
      });
    });
  }

  function showPropertyPanel(toolName) {
    document.querySelectorAll('.tool-prop-pane').forEach(p => p.style.display = 'none');
    const target = document.getElementById(`pane_${toolName}`);
    if (target) {
      target.style.display = 'block';
    }
    const titleMap = {
      trim: 'Trim & Split',
      crop: 'Crop & Aspect Ratio',
      subtitles: 'Subtitle Editor',
      overlays: 'Text & Emoji Overlays',
      audio: 'Audio & Music Controls',
      speed: 'Output Playback Speed',
      adjustments: 'Visual Adjustments & Filters'
    };
    document.getElementById('currentPanelTitle').innerText = titleMap[toolName] || 'Properties';
  }

  function populateUI() {
    // Trim
    document.getElementById('trimStartInput').value = (project.trim.start || 0).toFixed(1);
    document.getElementById('trimEndInput').value = (project.trim.end || 30).toFixed(1);

    // Crop
    const cropModeBtns = document.querySelectorAll('.crop-mode-btn');
    cropModeBtns.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === project.crop.mode);
    });
    document.getElementById('manualXSlider').value = project.crop.manual_x || 0.5;
    document.getElementById('manualXContainer').style.display = (project.crop.mode === 'manual') ? 'block' : 'none';

    // Subtitles
    document.getElementById('subtitlesToggle').checked = project.subtitles.enabled;
    document.getElementById('subStyleSelect').value = project.subtitles.style || 'bold';
    document.getElementById('subPosSelect').value = project.subtitles.position || 'bottom';
    document.getElementById('subFontSelect').value = project.subtitles.font_family || 'Arial';
    document.getElementById('subFontSize').value = project.subtitles.font_size || 48;
    document.getElementById('subColorInput').value = project.subtitles.color || '#FFFFFF';
    document.getElementById('subBoldToggle').checked = project.subtitles.bold !== false;
    document.getElementById('subEmojisToggle').checked = project.subtitles.emojis_enabled === true;
    renderSubtitlesList();

    // Overlays
    renderOverlaysList();

    // Audio
    document.getElementById('origAudioToggle').checked = project.audio.original_enabled !== false;
    document.getElementById('origVolumeSlider').value = project.audio.original_volume !== undefined ? project.audio.original_volume : 1.0;
    document.getElementById('bgmVolumeSlider').value = project.audio.music_volume !== undefined ? project.audio.music_volume : 0.3;
    document.getElementById('fadeInSlider').value = project.audio.fade_in || 0.0;
    document.getElementById('fadeOutSlider').value = project.audio.fade_out || 0.0;

    // Speed
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.classList.toggle('active', parseFloat(btn.getAttribute('data-speed')) === (project.speed || 1.0));
    });

    // Adjustments
    document.getElementById('brightSlider').value = project.adjustments.brightness || 0.0;
    document.getElementById('contrastSlider').value = project.adjustments.contrast !== undefined ? project.adjustments.contrast : 1.0;
    document.getElementById('saturateSlider').value = project.adjustments.saturation !== undefined ? project.adjustments.saturation : 1.0;
    document.querySelectorAll('.filter-preset-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-preset') === (project.adjustments.filter_preset || 'normal'));
    });
  }

  function renderSubtitlesList() {
    const list = document.getElementById('subtitlesItemsList');
    list.innerHTML = '';
    const segments = project.subtitles.segments || [];

    if (segments.length === 0) {
      list.innerHTML = '<div class="text-muted text-xs text-center py-3">No subtitles added. Click "Add Subtitle" to create one.</div>';
      return;
    }

    segments.forEach((seg, idx) => {
      const card = document.createElement('div');
      card.className = 'subtitle-item-card';
      card.innerHTML = `
        <div class="d-flex align-items-center justify-content-between text-xs">
          <span class="font-bold text-white">#${idx + 1} (${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s)</span>
          <div class="d-flex gap-1">
            <button class="btn btn-sm btn-link text-info p-0 text-xs" title="Seek" onclick="window.seekToTime(${seg.start})"><i class="bi bi-play-circle"></i></button>
            <button class="btn btn-sm btn-link text-danger p-0 text-xs" title="Delete" onclick="window.deleteSubtitle('${seg.id}')"><i class="bi bi-trash"></i></button>
          </div>
        </div>
        <input type="text" class="editor-input py-1 text-xs" value="${seg.text.replace(/"/g, '&quot;')}" onchange="window.updateSubtitleText('${seg.id}', this.value)">
      `;
      list.appendChild(card);
    });
  }

  function renderOverlaysList() {
    const list = document.getElementById('overlaysItemsList');
    list.innerHTML = '';
    const overlays = project.overlays || [];

    if (overlays.length === 0) {
      list.innerHTML = '<div class="text-muted text-xs text-center py-3">No text or emoji overlays. Click buttons above to add.</div>';
      return;
    }

    overlays.forEach((ov, idx) => {
      const card = document.createElement('div');
      card.className = 'subtitle-item-card';
      card.innerHTML = `
        <div class="d-flex align-items-center justify-content-between text-xs">
          <span class="font-bold text-white">${ov.type === 'emoji' ? '😀 Emoji' : '📝 Text'} (${ov.start_sec.toFixed(1)}s - ${ov.end_sec.toFixed(1)}s)</span>
          <button class="btn btn-sm btn-link text-danger p-0 text-xs" title="Delete" onclick="window.deleteOverlay('${ov.id}')"><i class="bi bi-trash"></i></button>
        </div>
        <div class="d-flex gap-2">
          <input type="text" class="editor-input py-1 text-xs flex-grow-1" value="${ov.content.replace(/"/g, '&quot;')}" onchange="window.updateOverlayContent('${ov.id}', this.value)">
          <input type="color" class="form-control form-control-color p-0 bg-transparent border-0" style="width: 28px; height: 28px;" value="${ov.color || '#FFD700'}" onchange="window.updateOverlayColor('${ov.id}', this.value)">
        </div>
      `;
      list.appendChild(card);
    });
  }

  // =========================================================================
  // 6. GLOBAL METHODS EXPOSED FOR HTML CONTROLS
  // =========================================================================

  window.seekToTime = function (sec) {
    video.currentTime = Math.max(0, sec);
  };

  window.deleteSubtitle = function (subId) {
    project.subtitles.segments = project.subtitles.segments.filter(s => s.id !== subId);
    pushHistory('Deleted subtitle');
    renderSubtitlesList();
    renderTimelineSubtitles();
  };

  window.updateSubtitleText = function (subId, newText) {
    const s = project.subtitles.segments.find(x => x.id === subId);
    if (s) {
      s.text = newText.trim();
      pushHistory('Updated subtitle text');
      renderTimelineSubtitles();
    }
  };

  window.addSubtitle = function () {
    const cur = video.currentTime;
    const newSeg = {
      id: `sub_${Math.random().toString(36).substr(2, 6)}`,
      start: parseFloat(cur.toFixed(1)),
      end: parseFloat((cur + 2.5).toFixed(1)),
      text: 'New Caption'
    };
    project.subtitles.segments.push(newSeg);
    project.subtitles.segments.sort((a, b) => a.start - b.start);
    pushHistory('Added subtitle');
    renderSubtitlesList();
    renderTimelineSubtitles();
  };

  window.deleteOverlay = function (ovId) {
    project.overlays = project.overlays.filter(o => o.id !== ovId);
    pushHistory('Deleted overlay');
    renderOverlaysList();
    renderTimelineOverlays();
  };

  window.updateOverlayContent = function (ovId, text) {
    const ov = project.overlays.find(x => x.id === ovId);
    if (ov) {
      ov.content = text.trim();
      pushHistory('Updated overlay content');
      renderTimelineOverlays();
    }
  };

  window.updateOverlayColor = function (ovId, color) {
    const ov = project.overlays.find(x => x.id === ovId);
    if (ov) {
      ov.color = color;
      pushHistory('Updated overlay color');
    }
  };

  window.addTextOverlay = function () {
    const cur = video.currentTime;
    const newOv = {
      id: `ov_${Math.random().toString(36).substr(2, 6)}`,
      type: 'text',
      content: 'ENTER HEADLINE',
      font_size: 42,
      color: '#FFD700',
      x_pct: 50,
      y_pct: 22,
      start_sec: parseFloat(cur.toFixed(1)),
      end_sec: parseFloat((cur + 5.0).toFixed(1))
    };
    project.overlays.push(newOv);
    pushHistory('Added text overlay');
    renderOverlaysList();
    renderTimelineOverlays();
  };

  window.addEmojiOverlay = function (emojiChar) {
    const cur = video.currentTime;
    const newOv = {
      id: `ov_${Math.random().toString(36).substr(2, 6)}`,
      type: 'emoji',
      content: emojiChar,
      font_size: 54,
      color: '#FFFFFF',
      x_pct: 50,
      y_pct: 50,
      start_sec: parseFloat(cur.toFixed(1)),
      end_sec: parseFloat((cur + 4.0).toFixed(1))
    };
    project.overlays.push(newOv);
    pushHistory(`Added emoji ${emojiChar}`);
    renderOverlaysList();
    renderTimelineOverlays();
  };

  // =========================================================================
  // 7. EVENT LISTENERS
  // =========================================================================

  function setupEventListeners() {
    // Playback events
    playPauseBtn.addEventListener('click', togglePlayPause);
    video.addEventListener('click', togglePlayPause);
    video.addEventListener('play', () => playPauseBtn.innerHTML = '<i class="bi bi-pause-fill fs-5"></i>');
    video.addEventListener('pause', () => playPauseBtn.innerHTML = '<i class="bi bi-play-fill fs-5"></i>');
    video.addEventListener('timeupdate', onVideoTimeUpdate);

    // Timeline Scrubbing & Dragging
    timelineScroll.addEventListener('mousedown', onTimelineMouseDown);
    window.addEventListener('mousemove', onTimelineMouseMove);
    window.addEventListener('mouseup', onTimelineMouseUp);

    // Zoom Buttons
    document.getElementById('zoomInBtn').addEventListener('click', () => {
      zoomPixelsPerSec = Math.min(60, zoomPixelsPerSec + 8);
      renderTimelineRuler();
      updateTimelineClipPosition();
      renderTimelineSubtitles();
      renderTimelineOverlays();
    });
    document.getElementById('zoomOutBtn').addEventListener('click', () => {
      zoomPixelsPerSec = Math.max(10, zoomPixelsPerSec - 8);
      renderTimelineRuler();
      updateTimelineClipPosition();
      renderTimelineSubtitles();
      renderTimelineOverlays();
    });

    // Split at playhead button
    document.getElementById('splitBtn').addEventListener('click', splitClipAtPlayhead);

    // Step buttons
    document.getElementById('stepBackBtn').addEventListener('click', () => stepTime(-0.1));
    document.getElementById('stepFwdBtn').addEventListener('click', () => stepTime(0.1));

    // Volume & Mute
    volumeSlider.addEventListener('input', (e) => {
      video.volume = parseFloat(e.target.value);
      video.muted = (video.volume === 0);
      muteBtn.innerHTML = video.muted ? '<i class="bi bi-volume-mute-fill"></i>' : '<i class="bi bi-volume-up-fill"></i>';
    });
    muteBtn.addEventListener('click', () => {
      video.muted = !video.muted;
      muteBtn.innerHTML = video.muted ? '<i class="bi bi-volume-mute-fill"></i>' : '<i class="bi bi-volume-up-fill"></i>';
    });

    // Fullscreen
    document.getElementById('fullscreenBtn').addEventListener('click', () => {
      if (!document.fullscreenElement) {
        stageViewport.requestFullscreen().catch(err => alert(err.message));
      } else {
        document.exitFullscreen();
      }
    });

    // Header buttons
    undoBtn.addEventListener('click', undo);
    redoBtn.addEventListener('click', redo);
    saveBtn.addEventListener('click', saveProjectToServer);
    document.getElementById('resetBtn').addEventListener('click', resetProject);

    // Hotkeys
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayPause();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        stepTime(-0.1);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        stepTime(0.1);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        saveProjectToServer();
      }
    });

    // Tool Tabs
    setupToolTabs();

    // Aspect Ratio Buttons
    document.querySelectorAll('.aspect-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const a = btn.getAttribute('data-aspect');
        applyAspectPreset(a);
        pushHistory(`Aspect ratio: ${a}`);
      });
    });

    // Crop Mode Buttons
    document.querySelectorAll('.crop-mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.crop-mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.getAttribute('data-mode');
        project.crop.mode = mode;
        document.getElementById('manualXContainer').style.display = (mode === 'manual') ? 'block' : 'none';
        pushHistory(`Crop mode: ${mode}`);
      });
    });

    document.getElementById('manualXSlider').addEventListener('input', (e) => {
      project.crop.manual_x = parseFloat(e.target.value);
      setUnsaved(true);
    });

    // Numeric Trim Inputs
    document.getElementById('trimStartInput').addEventListener('change', (e) => {
      let s = parseFloat(e.target.value) || 0;
      let end = project.trim.end || 30;
      if (s >= end) s = Math.max(0, end - 1.0);
      project.trim.start = s;
      project.trim.duration = end - s;
      pushHistory('Trim start updated');
      updateTimelineClipPosition();
    });

    document.getElementById('trimEndInput').addEventListener('change', (e) => {
      let end = parseFloat(e.target.value) || 30;
      let s = project.trim.start || 0;
      if (end <= s) end = s + 1.0;
      project.trim.end = end;
      project.trim.duration = end - s;
      pushHistory('Trim end updated');
      updateTimelineClipPosition();
    });

    // Subtitles Options
    document.getElementById('subtitlesToggle').addEventListener('change', (e) => {
      project.subtitles.enabled = e.target.checked;
      pushHistory('Subtitles toggled');
      updateLiveSubtitles(video.currentTime);
    });
    document.getElementById('subStyleSelect').addEventListener('change', (e) => {
      project.subtitles.style = e.target.value;
      pushHistory('Subtitle style changed');
    });
    document.getElementById('subPosSelect').addEventListener('change', (e) => {
      project.subtitles.position = e.target.value;
      pushHistory('Subtitle position changed');
      updateLiveSubtitles(video.currentTime);
    });
    document.getElementById('subFontSelect').addEventListener('change', (e) => {
      project.subtitles.font_family = e.target.value;
      pushHistory('Subtitle font changed');
      updateLiveSubtitles(video.currentTime);
    });
    document.getElementById('subFontSize').addEventListener('input', (e) => {
      project.subtitles.font_size = parseInt(e.target.value);
      updateLiveSubtitles(video.currentTime);
      setUnsaved(true);
    });
    document.getElementById('subColorInput').addEventListener('input', (e) => {
      project.subtitles.color = e.target.value;
      updateLiveSubtitles(video.currentTime);
      setUnsaved(true);
    });
    document.getElementById('subBoldToggle').addEventListener('change', (e) => {
      project.subtitles.bold = e.target.checked;
      updateLiveSubtitles(video.currentTime);
      setUnsaved(true);
    });
    document.getElementById('subEmojisToggle').addEventListener('change', (e) => {
      project.subtitles.emojis_enabled = e.target.checked;
      pushHistory('Contextual emojis toggled');
    });

    // Audio Settings
    document.getElementById('origAudioToggle').addEventListener('change', (e) => {
      project.audio.original_enabled = e.target.checked;
      video.muted = !e.target.checked;
      pushHistory('Original audio toggle');
    });
    document.getElementById('origVolumeSlider').addEventListener('input', (e) => {
      project.audio.original_volume = parseFloat(e.target.value);
      video.volume = project.audio.original_volume;
      setUnsaved(true);
    });
    document.getElementById('bgmVolumeSlider').addEventListener('input', (e) => {
      project.audio.music_volume = parseFloat(e.target.value);
      setUnsaved(true);
    });
    document.getElementById('fadeInSlider').addEventListener('input', (e) => {
      project.audio.fade_in = parseFloat(e.target.value);
      setUnsaved(true);
    });
    document.getElementById('fadeOutSlider').addEventListener('input', (e) => {
      project.audio.fade_out = parseFloat(e.target.value);
      setUnsaved(true);
    });

    // Background Audio File Upload
    document.getElementById('bgmFileInput').addEventListener('change', uploadAudioFile);

    // Speed Buttons
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const spd = parseFloat(btn.getAttribute('data-speed'));
        project.speed = spd;
        video.playbackRate = spd;
        pushHistory(`Speed: ${spd}x`);
      });
    });

    // Visual Adjustments Sliders
    ['brightSlider', 'contrastSlider', 'saturateSlider'].forEach(id => {
      document.getElementById(id).addEventListener('input', () => {
        project.adjustments.brightness = parseFloat(document.getElementById('brightSlider').value);
        project.adjustments.contrast = parseFloat(document.getElementById('contrastSlider').value);
        project.adjustments.saturation = parseFloat(document.getElementById('saturateSlider').value);
        applyVisualAdjustments();
        setUnsaved(true);
      });
    });

    // Filter Preset Buttons
    document.querySelectorAll('.filter-preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const p = btn.getAttribute('data-preset');
        project.adjustments.filter_preset = p;
        applyVisualAdjustments();
        pushHistory(`Filter: ${p}`);
      });
    });

    // Render Final Video Button
    document.getElementById('renderFinalBtn').addEventListener('click', openExportModalAndRender);

    // Window Unload Warning
    window.addEventListener('beforeunload', (e) => {
      if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
      }
    });
  }

  // =========================================================================
  // 8. SERVER PERSISTENCE & AUDIO UPLOAD
  // =========================================================================

  async function saveProjectToServer() {
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Saving...';

    try {
      const resp = await fetch(`/api/editor/${JOB_ID}/${CLIP_ID}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(project)
      });
      const data = await resp.json();
      if (!resp.ok || data.status !== 'success') {
        throw new Error(data.message || 'Failed to save project');
      }
      setUnsaved(false);
      saveBtn.innerHTML = '<i class="bi bi-check-lg me-1"></i> Saved!';
      setTimeout(() => {
        saveBtn.innerHTML = '<i class="bi bi-cloud-arrow-up me-1"></i> Save Project';
        saveBtn.disabled = false;
      }, 1500);
    } catch (err) {
      alert(`Save error: ${err.message}`);
      saveBtn.innerHTML = '<i class="bi bi-cloud-arrow-up me-1"></i> Save Project';
      saveBtn.disabled = false;
    }
  }

  async function resetProject() {
    if (!confirm('Are you sure you want to reset all edits to the original clip state?')) {
      return;
    }
    try {
      const resp = await fetch(`/api/editor/${JOB_ID}/${CLIP_ID}/reset`, {
        method: 'POST'
      });
      const data = await resp.json();
      if (!resp.ok || data.status !== 'success') {
        throw new Error(data.message || 'Failed to reset project');
      }
      project = data.project;
      applyAllProjectState();
      pushHistory('Reset to original', true);
      setUnsaved(false);
      alert('Edits successfully reset to original clip.');
    } catch (err) {
      alert(`Reset error: ${err.message}`);
    }
  }

  async function uploadAudioFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    const label = document.getElementById('bgmFileLabel');
    label.innerText = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append('audio_file', file);

    try {
      const resp = await fetch(`/api/editor/${JOB_ID}/${CLIP_ID}/upload_audio`, {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();
      if (!resp.ok || data.status !== 'success') {
        throw new Error(data.message || 'Upload failed');
      }
      project.audio.music_filename = data.filename;
      label.innerText = `✓ ${file.name}`;
      pushHistory('Uploaded background music');
    } catch (err) {
      alert(`Audio upload failed: ${err.message}`);
      label.innerText = 'Choose Audio File (.mp3, .wav)';
    }
  }

  // =========================================================================
  // 9. EXPORT & RENDER POLLING
  // =========================================================================

  let renderPollInterval = null;

  async function openExportModalAndRender() {
    // Ensure current state is saved first
    await saveProjectToServer();

    const modal = document.getElementById('exportModal');
    const progressSection = document.getElementById('exportProgressSection');
    const completeSection = document.getElementById('exportCompleteSection');
    const progressBar = document.getElementById('exportProgressBar');
    const progressStatus = document.getElementById('exportProgressStatus');
    const progressPercent = document.getElementById('exportProgressPercent');

    modal.style.display = 'flex';
    progressSection.style.display = 'block';
    completeSection.style.display = 'none';

    progressBar.style.width = '10%';
    progressStatus.innerText = 'Submitting render job...';
    progressPercent.innerText = '10%';

    try {
      const resp = await fetch(`/api/editor/${JOB_ID}/${CLIP_ID}/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(project)
      });
      const data = await resp.json();
      if (!resp.ok || (data.status !== 'queued' && data.status !== 'already_running')) {
        throw new Error(data.message || 'Failed to start render');
      }

      // Start polling
      if (renderPollInterval) clearInterval(renderPollInterval);
      renderPollInterval = setInterval(pollRenderStatus, 1000);
    } catch (err) {
      progressStatus.innerText = `Error: ${err.message}`;
      progressBar.style.background = '#ef4444';
    }
  }

  async function pollRenderStatus() {
    try {
      const resp = await fetch(`/api/editor/${JOB_ID}/${CLIP_ID}/render_status`);
      const data = await resp.json();
      if (!resp.ok) return;

      const progressBar = document.getElementById('exportProgressBar');
      const progressStatus = document.getElementById('exportProgressStatus');
      const progressPercent = document.getElementById('exportProgressPercent');

      if (data.status === 'rendering' || data.status === 'queued') {
        const p = data.progress || 35;
        progressBar.style.width = `${p}%`;
        progressPercent.innerText = `${p}%`;
        progressStatus.innerText = data.stage || 'Rendering video...';
      } else if (data.status === 'complete') {
        clearInterval(renderPollInterval);
        progressBar.style.width = '100%';
        progressPercent.innerText = '100%';
        progressStatus.innerText = '✓ Video Rendered Successfully!';

        setTimeout(() => {
          showExportComplete(data);
        }, 800);
      } else if (data.status === 'error') {
        clearInterval(renderPollInterval);
        progressStatus.innerText = `Render Failed: ${data.error || 'Unknown error'}`;
        progressBar.style.background = '#ef4444';
      }
    } catch (e) {
      console.warn('Status poll warning:', e);
    }
  }

  function showExportComplete(statusData) {
    document.getElementById('exportProgressSection').style.display = 'none';
    const completeSection = document.getElementById('exportCompleteSection');
    completeSection.style.display = 'block';

    const finalPlayer = document.getElementById('exportFinalPlayer');
    const finalUrl = `/media/${JOB_ID}/${statusData.output_filename}?t=${Date.now()}`;
    finalPlayer.src = finalUrl;
    finalPlayer.load();

    document.getElementById('exportFileSize').innerText = `${statusData.file_size_mb || 0} MB`;
    document.getElementById('exportDownloadBtn').href = `/download/${JOB_ID}/${statusData.output_filename}`;
  }

  window.closeExportModal = function () {
    if (renderPollInterval) clearInterval(renderPollInterval);
    document.getElementById('exportModal').style.display = 'none';
    const finalPlayer = document.getElementById('exportFinalPlayer');
    if (finalPlayer) finalPlayer.pause();
  };

  // Run initial setup when DOM is ready
  document.addEventListener('DOMContentLoaded', init);

})();
