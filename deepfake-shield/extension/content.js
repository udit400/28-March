const VIDEO_SELECTOR = "video";
const API_BASE_URL = "http://127.0.0.1:8000";

const trackedVideos = new WeakSet();
const trackedVideoUis = new WeakMap();
const observer = new MutationObserver(() => scanForVideos());

scanForVideos();
observer.observe(document.documentElement, {
  childList: true,
  subtree: true
});

setInterval(() => {
  refreshTrackedVideoUis();
}, 500);

window.addEventListener("load", () => {
  scanForVideos();
  refreshTrackedVideoUis();
});

function scanForVideos() {
  const videos = document.querySelectorAll(VIDEO_SELECTOR);
  videos.forEach((video) => {
    if (trackedVideos.has(video)) {
      return;
    }
    trackedVideos.add(video);
    mountShield(video);
  });
}

function mountShield(video) {
  const host = ensureHost(video);
  const panel = document.createElement("div");
  panel.className = "shield-panel";

  const button = document.createElement("button");
  button.className = "deepfake-shield-btn";
  button.type = "button";
  button.textContent = "SCAN MEDIA";

  const badge = document.createElement("div");
  badge.className = "shield-result-box shield-result-idle";
  badge.textContent = "Ready to analyze current frame";

  const overlay = document.createElement("div");
  overlay.className = "deepfake-shield-overlay";

  panel.appendChild(button);
  panel.appendChild(badge);
  host.appendChild(panel);
  host.appendChild(overlay);

  trackedVideoUis.set(video, { panel, badge, overlay, button });
  updateVideoUi(video);

  button.addEventListener("click", async (event) => {
    event.preventDefault();
    await analyzeVideo(video, button, badge, overlay);
  });

  ["loadeddata", "play", "pause", "emptied", "resize", "canplay"].forEach((eventName) => {
    video.addEventListener(eventName, () => updateVideoUi(video));
  });
}

function refreshTrackedVideoUis() {
  const videos = document.querySelectorAll(VIDEO_SELECTOR);
  videos.forEach((video) => updateVideoUi(video));
}

function updateVideoUi(video) {
  const ui = trackedVideoUis.get(video);
  if (!ui) {
    return;
  }

  const visible = isVideoVisible(video);
  ui.panel.style.display = visible ? "flex" : "none";
  ui.overlay.style.display = visible ? "block" : "none";

  if (!visible) {
    return;
  }

  if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
    setState(ui.button, ui.badge, ui.overlay, {
      buttonLabel: "VIDEO NOT READY",
      badgeLabel: "Wait until the video frame is available.",
      tone: "error"
    });
    return;
  }

  if (ui.button.dataset.state === "error" && ui.button.textContent === "VIDEO NOT READY") {
    setState(ui.button, ui.badge, ui.overlay, {
      buttonLabel: "SCAN MEDIA",
      badgeLabel: "Ready to analyze current frame",
      tone: "idle"
    });
  }
}

function isVideoVisible(video) {
  if (!video.isConnected) {
    return false;
  }

  const style = window.getComputedStyle(video);
  if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) {
    return false;
  }

  const rect = video.getBoundingClientRect();
  return rect.width > 80 && rect.height > 60;
}

function ensureHost(video) {
  const parent = video.parentElement;
  if (!parent) {
    throw new Error("Video element has no parent container.");
  }

  const style = window.getComputedStyle(parent);
  if (style.position === "static") {
    parent.style.position = "relative";
  }

  return parent;
}

async function analyzeVideo(video, button, badge, overlay) {
  if (!video.videoWidth || !video.videoHeight) {
    setState(button, badge, overlay, {
      buttonLabel: "VIDEO NOT READY",
      badgeLabel: "Wait until the video frame is available.",
      tone: "error"
    });
    return;
  }

  setState(button, badge, overlay, {
    buttonLabel: "ANALYZING...",
    badgeLabel: "Capturing frame and sending it to the local API.",
    tone: "loading"
  });

  try {
    const imageDataUrl = captureFrame(video);
    const result = await requestAnalysis(imageDataUrl);
    const tone = result.is_deepfake ? "alert" : "safe";
    const buttonLabel = result.is_deepfake
      ? `DEEPFAKE ${result.fake_score}%`
      : `AUTHENTIC ${result.fake_score}%`;
    const badgeLabel = `${result.message} Engine: ${String(result.engine).toUpperCase()}`;

    setState(button, badge, overlay, { buttonLabel, badgeLabel, tone });
  } catch (error) {
    console.error("Synthetic Media Shield error", error);
    setState(button, badge, overlay, {
      buttonLabel: "SERVER OFFLINE",
      badgeLabel: error.message || "Unable to reach the backend.",
      tone: "error"
    });
  }
}

function captureFrame(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas capture is not available on this page.");
  }

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  try {
    return canvas.toDataURL("image/jpeg", 0.92);
  } catch (error) {
    throw new Error("This site blocks frame capture. Use the local demo page at http://127.0.0.1:8000/demo for a reliable scan.");
  }
}

function requestAnalysis(imageDataUrl) {
  const runtime = globalThis.chrome?.runtime;
  if (!runtime?.sendMessage) {
    return requestAnalysisDirect(imageDataUrl);
  }

  return new Promise((resolve, reject) => {
    runtime.sendMessage(
      {
        type: "shield-analyze-frame",
        imageDataUrl
      },
      (response) => {
        const runtimeError = runtime.lastError;
        if (runtimeError) {
          requestAnalysisDirect(imageDataUrl).then(resolve).catch(reject);
          return;
        }

        if (!response?.ok) {
          requestAnalysisDirect(imageDataUrl).then(resolve).catch(reject);
          return;
        }

        resolve(response.data);
      }
    );
  });
}

async function requestAnalysisDirect(imageDataUrl) {
  const blob = dataUrlToBlob(imageDataUrl);
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const fallback = await response.text();
    throw new Error(fallback || `Backend error: ${response.status}`);
  }

  return response.json();
}

function dataUrlToBlob(dataUrl) {
  const [header, encoded] = dataUrl.split(",");
  const mimeMatch = header.match(/data:(.*?);base64/);
  const mimeType = mimeMatch ? mimeMatch[1] : "image/jpeg";
  const binary = atob(encoded);
  const length = binary.length;
  const bytes = new Uint8Array(length);

  for (let index = 0; index < length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return new Blob([bytes], { type: mimeType });
}

function setState(button, badge, overlay, state) {
  button.textContent = state.buttonLabel;
  badge.textContent = state.badgeLabel;

  button.dataset.state = state.tone;
  badge.className = `shield-result-box shield-result-${state.tone}`;
  overlay.dataset.state = state.tone;
}