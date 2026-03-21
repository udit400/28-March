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

  if (["loading", "safe", "alert"].includes(ui.button.dataset.state)) {
    return;
  }

  if (ui.button.textContent !== "SCAN MEDIA" || ui.button.dataset.state !== "idle") {
    setState(ui.button, ui.badge, ui.overlay, {
      buttonLabel: "SCAN MEDIA",
      badgeLabel: video.videoWidth && video.videoHeight && video.readyState >= 2
        ? "Ready to analyze current frame"
        : "Click to analyze. Fallback mode will be used if the frame is not ready.",
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
  const storage = await getStoredSession();
  const hasPremiumAccess = Boolean(storage.userEmail && storage.hasSub);

  setState(button, badge, overlay, {
    buttonLabel: "ANALYZING...",
    badgeLabel: "Capturing media snapshot and extracting forensic metadata.",
    tone: "loading"
  });

  try {
    const imageDataUrl = captureFrame(video);
    const result = await requestAnalysis(imageDataUrl);
    const tone = result.is_deepfake ? "alert" : "safe";
    const buttonLabel = result.is_deepfake
      ? `DEEPFAKE ${result.fake_score}%`
      : `AUTHENTIC ${result.fake_score}%`;
    const meta = result.metadata || {};
    const badgeLines = [
      `Status: ${result.message}`,
      `Platform: ${meta.software_platform || "Unknown"}`,
      `Date: ${meta.creation_date || "Unknown"}`,
      `Provenance: ${meta.c2pa_signature || "Unknown"}`,
      `Source: ${meta.is_original || "Unknown"}`
    ];

    if (!hasPremiumAccess) {
      badgeLines.unshift("Mode: Demo access enabled");
    }

    const badgeLabel = badgeLines.join("\n");

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
  const width = video.videoWidth || Math.max(640, Math.round(video.clientWidth || 640));
  const height = video.videoHeight || Math.max(360, Math.round(video.clientHeight || 360));
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas capture is not available on this page.");
  }

  const frameReady = video.videoWidth && video.videoHeight && video.readyState >= 2;
  if (frameReady) {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
  } else {
    drawFallbackSnapshot(context, canvas, video);
  }

  try {
    return canvas.toDataURL("image/jpeg", 0.92);
  } catch (error) {
    throw new Error("This site blocks frame capture. Use the local demo page at http://127.0.0.1:8000/demo for a reliable scan.");
  }
}

function drawFallbackSnapshot(context, canvas, video) {
  const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, "#07101f");
  gradient.addColorStop(1, "#12324a");
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.fillStyle = "rgba(110, 245, 219, 0.16)";
  context.fillRect(24, 24, canvas.width - 48, canvas.height - 48);

  context.fillStyle = "#b4fff6";
  context.font = "bold 28px Segoe UI";
  context.fillText("Synthetic Media Shield", 40, 72);

  context.fillStyle = "#dce9ff";
  context.font = "18px Segoe UI";
  context.fillText("Fallback analysis snapshot", 40, 118);
  context.fillText(`Page: ${window.location.hostname}`, 40, 154);
  context.fillText(`Time: ${new Date().toLocaleTimeString()}`, 40, 190);

  const status = video.readyState >= 1
    ? "Video element visible but frame not ready at click time"
    : "No decoded frame available, using resilient demo capture";
  wrapCanvasText(context, status, 40, 234, canvas.width - 80, 28);
}

function wrapCanvasText(context, text, x, y, maxWidth, lineHeight) {
  const words = text.split(" ");
  let line = "";

  for (const word of words) {
    const testLine = line ? `${line} ${word}` : word;
    if (context.measureText(testLine).width > maxWidth && line) {
      context.fillText(line, x, y);
      line = word;
      y += lineHeight;
    } else {
      line = testLine;
    }
  }

  if (line) {
    context.fillText(line, x, y);
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

function getStoredSession() {
  const storageApi = globalThis.chrome?.storage?.local;
  if (!storageApi) {
    return Promise.resolve({ userEmail: null, hasSub: false, authToken: null });
  }

  return new Promise((resolve) => {
    storageApi.get(["userEmail", "hasSub", "authToken"], (data) => {
      resolve(data || {});
    });
  });
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