const API_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await restoreView();
});

function bindEvents() {
  document.getElementById("mode-login").addEventListener("click", () => setAuthMode("login"));
  document.getElementById("mode-signup").addEventListener("click", () => setAuthMode("signup"));
  document.getElementById("btn-login-otp").addEventListener("click", () => requestOtp("login"));
  document.getElementById("btn-signup-otp").addEventListener("click", () => requestOtp("signup"));
  document.getElementById("btn-verify-otp").addEventListener("click", verifyOtp);
  document.getElementById("btn-verify-pay").addEventListener("click", verifyPayment);
  document.getElementById("btn-logout").addEventListener("click", logout);
  document.getElementById("btn-back-login").addEventListener("click", resetToLogin);
  document.getElementById("btn-switch-account").addEventListener("click", resetToLogin);
  document.getElementById("btn-select-file").addEventListener("click", () => {
    document.getElementById("full-file-upload").click();
  });
  document.getElementById("full-file-upload").addEventListener("change", handleFullFileSelection);
}

async function restoreView() {
  const data = await storageGet(["userEmail", "hasSub", "authStage", "profileName", "authMode"]);
  updateAccountLabels(data.profileName || "", data.userEmail || "");
  if (data.userEmail && data.authStage === "verified") {
    showView("view-dashboard");
    updateDashboardState(Boolean(data.hasSub), data.profileName || "", data.userEmail || "");
    return;
  }
  showView("view-auth");
  setAuthMode(data.authMode === "signup" ? "signup" : "login");
  if (data.authStage === "otp-sent" && data.userEmail) {
    showOtpPanel(true);
    updateAccountLabels(data.profileName || "", data.userEmail || "");
    return;
  }
  showOtpPanel(false);
}

function showView(viewId) {
  ["view-auth", "view-dashboard"].forEach((id) => {
    document.getElementById(id).classList.toggle("active", id === viewId);
  });
}

function setAuthMode(mode) {
  document.getElementById("mode-login").classList.toggle("active", mode === "login");
  document.getElementById("mode-signup").classList.toggle("active", mode === "signup");
  document.getElementById("login-form").classList.toggle("active", mode === "login");
  document.getElementById("signup-form").classList.toggle("active", mode === "signup");
}

function showOtpPanel(visible) {
  document.getElementById("otp-panel").classList.toggle("active", visible);
}

function setStatus(message) {
  document.getElementById("status").textContent = message || "";
}

function formatAccountLabel(profileName, email) {
  if (!email) {
    return "No account selected yet.";
  }
  return profileName ? `${profileName} • ${email}` : `Account: ${email}`;
}

function updateAccountLabels(profileName, email) {
  const fallback = formatAccountLabel(profileName, email);
  document.getElementById("otp-email-chip").textContent = fallback;
  document.getElementById("pay-email-chip").textContent = fallback;
  document.getElementById("dashboard-email-chip").textContent = fallback;
}

function updateDashboardState(hasSub, profileName, email) {
  const badge = document.getElementById("subscription-badge");
  const title = document.getElementById("dashboard-title");
  const copy = document.getElementById("dashboard-copy");
  const upgradeCard = document.getElementById("upgrade-card");
  const lockedNote = document.getElementById("locked-note");
  const premiumPanel = document.getElementById("premium-panel");

  updateAccountLabels(profileName, email);
  if (hasSub) {
    badge.textContent = "Premium active";
    badge.classList.add("premium");
    title.textContent = profileName ? `Welcome, ${profileName}` : "Welcome back";
    copy.textContent = "Your premium shield is active. Run deep forensic uploads directly from the main page.";
    upgradeCard.classList.add("hidden");
    lockedNote.classList.add("hidden");
    premiumPanel.classList.remove("hidden");
    return;
  }

  badge.textContent = "Subscription inactive";
  badge.classList.remove("premium");
  title.textContent = profileName ? `Welcome, ${profileName}` : "Main page";
  copy.textContent = "You are logged in successfully. Upgrade here whenever you want premium forensic uploads.";
  upgradeCard.classList.remove("hidden");
  lockedNote.classList.remove("hidden");
  premiumPanel.classList.add("hidden");
}

function buildProfileName(mode) {
  if (mode !== "signup") {
    return "";
  }

  const firstName = document.getElementById("signup-first-name").value.trim();
  const lastName = document.getElementById("signup-last-name").value.trim();
  return [firstName, lastName].filter(Boolean).join(" ");
}

function getAuthEmail(mode) {
  const inputId = mode === "signup" ? "signup-email-input" : "login-email-input";
  return document.getElementById(inputId).value.trim().toLowerCase();
}

async function resetToLogin() {
  await chrome.storage.local.remove(["userEmail", "hasSub", "authToken", "authStage", "profileName", "authMode"]);
  document.getElementById("login-email-input").value = "";
  document.getElementById("signup-first-name").value = "";
  document.getElementById("signup-last-name").value = "";
  document.getElementById("signup-email-input").value = "";
  document.getElementById("otp-input").value = "";
  document.getElementById("utr-input").value = "";
  updateAccountLabels("", "");
  showView("view-auth");
  setAuthMode("login");
  showOtpPanel(false);
  setStatus("");
}

function setBusy(buttonId, busy) {
  const button = document.getElementById(buttonId);
  if (!button) {
    return;
  }
  button.disabled = busy;
  button.style.opacity = busy ? "0.7" : "1";
}

function isGmailAddress(value) {
  return /^[^\s@]+@gmail\.com$/i.test(String(value || "").trim());
}

async function requestOtp(mode) {
  const email = getAuthEmail(mode);
  const profileName = buildProfileName(mode);
  if (!email) {
    setStatus("Enter your Gmail address first.");
    return;
  }
  if (!isGmailAddress(email)) {
    setStatus("Enter a valid Gmail address ending in @gmail.com.");
    return;
  }
  if (mode === "signup" && !profileName) {
    setStatus("Enter at least your first name to create the account.");
    return;
  }

  const triggerButtonId = mode === "signup" ? "btn-signup-otp" : "btn-login-otp";
  setBusy(triggerButtonId, true);
  setStatus("Sending OTP...");
  try {
    const response = await fetch(`${API_URL}/auth/request-otp`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email })
    });
    const payload = await safeJson(response);

    if (!response.ok) {
      setStatus(payload.detail || "Failed to send OTP.");
      return;
    }

    await storageSet({
      userEmail: email,
      hasSub: false,
      authStage: "otp-sent",
      profileName,
      authMode: mode
    });
    updateAccountLabels(profileName, email);
    document.getElementById("otp-input").value = "";
    showView("view-auth");
    setAuthMode(mode);
    showOtpPanel(true);
    const maskedEmail = payload.masked_email || email;
    const expiryMinutes = Math.max(1, Math.round((payload.expires_in_seconds || 600) / 60));
    if (payload.delivery_mode === "local_preview" && payload.otp_preview) {
      setStatus(`Verification code issued for this environment: ${payload.otp_preview}. It expires in ${expiryMinutes} minute(s).`);
    } else {
      setStatus(`OTP sent to ${maskedEmail}. Check Gmail. The code expires in ${expiryMinutes} minute(s).`);
    }
  } catch (error) {
    setStatus(error.message || "Could not contact the backend.");
  } finally {
    setBusy(triggerButtonId, false);
  }
}

async function verifyOtp() {
  const otp = document.getElementById("otp-input").value.trim();
  const { userEmail, profileName } = await storageGet(["userEmail", "profileName"]);
  if (!userEmail || !otp) {
    setStatus("Enter the OTP sent to your email.");
    return;
  }

  setBusy("btn-verify-otp", true);
  setStatus("Verifying OTP...");
  try {
    const response = await fetch(`${API_URL}/auth/verify-otp`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: userEmail, otp })
    });
    const payload = await safeJson(response);

    if (!response.ok) {
      setStatus(payload.detail || "Invalid OTP.");
      return;
    }

    await storageSet({
      userEmail: payload.email,
      hasSub: Boolean(payload.has_subscription),
      authStage: "verified",
      profileName,
      authToken: payload.token || null
    });
    showView("view-dashboard");
    updateDashboardState(Boolean(payload.has_subscription), profileName || "", payload.email);
    setStatus(payload.has_subscription ? "Login successful. Premium is already active." : "Login successful. You are now on the main page.");
  } catch (error) {
    setStatus(error.message || "Could not verify OTP.");
  } finally {
    setBusy("btn-verify-otp", false);
  }
}

async function verifyPayment() {
  const utrNumber = document.getElementById("utr-input").value.trim();
  const { userEmail, profileName } = await storageGet(["userEmail", "profileName"]);
  if (!userEmail || !utrNumber) {
    setStatus(userEmail ? "Enter your UTR/reference number." : "Session expired. Start again with your email.");
    if (!userEmail) {
      resetToLogin();
    }
    return;
  }

  setBusy("btn-verify-pay", true);
  setStatus("Verifying payment...");
  try {
    const response = await fetch(`${API_URL}/payment/verify-upi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: userEmail, utr_number: utrNumber })
    });
    const payload = await safeJson(response);

    if (!response.ok) {
      setStatus(payload.detail || "Invalid UTR number.");
      return;
    }

    await storageSet({
      userEmail,
      hasSub: true,
      authStage: "verified",
      profileName,
      authToken: payload.token || null
    });
    showView("view-dashboard");
    updateDashboardState(true, profileName || "", userEmail);
    setStatus(`Premium activated until ${new Date(payload.subscription_end).toLocaleString()}`);
  } catch (error) {
    setStatus(error.message || "Could not verify payment.");
  } finally {
    setBusy("btn-verify-pay", false);
  }
}

async function logout() {
  await resetToLogin();
  setStatus("Logged out.");
}

function storageGet(keys) {
  return new Promise((resolve) => {
    chrome.storage.local.get(keys, (result) => resolve(result || {}));
  });
}

function storageSet(values) {
  return new Promise((resolve) => {
    chrome.storage.local.set(values, resolve);
  });
}

async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function clampPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return Math.max(0, Math.min(100, numeric));
}

function formatPercent(value) {
  const numeric = clampPercent(value);
  return numeric === null ? "N/A" : `${Math.round(numeric)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getMediaLabel(mediaType) {
  switch (String(mediaType || "").toLowerCase()) {
    case "video":
      return "video";
    case "audio":
      return "audio sample";
    case "image":
      return "image";
    default:
      return "media file";
  }
}

function buildLegendMetrics(data) {
  const overall = clampPercent(data.overall_threat_score) ?? 0;
  const video = clampPercent(data.visual_deepfake_score) ?? 0;
  const audio = clampPercent(data.audio_clone_score) ?? 0;
  const facial = data.biological_liveness == null ? 0 : Math.max(0, 100 - (clampPercent(data.biological_liveness) ?? 0));
  const original = Math.max(0, 100 - overall);
  return [
    { key: "video", label: "Video fakeness", value: video, color: "#4a89ff" },
    { key: "audio", label: "Audio fakeness", value: audio, color: "#a24dff" },
    { key: "facial", label: "Facial inconsistency", value: facial, color: "#3850ff" },
    { key: "original", label: "Original", value: original, color: "#4a4a4a" }
  ];
}

function getForensicVerdict(data) {
  const score = clampPercent(data.overall_threat_score) ?? 0;
  const mediaLabel = getMediaLabel(data.media_type);
  if (score >= 66) {
    return `We believe that the ${mediaLabel} is AI Generated`;
  }
  if (score >= 35) {
    return `We believe that the ${mediaLabel} is Dubious (Mixed)`;
  }
  return `We believe that the ${mediaLabel} is Authentic (Human Creation)`;
}

function getForensicSummary(data) {
  const score = clampPercent(data.overall_threat_score) ?? 0;
  const mediaLabel = getMediaLabel(data.media_type);
  if (score >= 66) {
    return `The ${mediaLabel} shows strong signals of AI-generated or manipulated content`;
  }
  if (score >= 35) {
    return `The ${mediaLabel} may have a mix of human-created components along with AI-generated components`;
  }
  if (mediaLabel === "video") {
    return "Both the video frames and the audio have a very low score of fakeness";
  }
  return `The ${mediaLabel} has a very low score of fakeness`;
}

function buildForensicResultsHtml(data) {
  const overall = clampPercent(data.overall_threat_score) ?? 0;
  const gaugeColor = overall >= 66 ? "#ff5b83" : overall >= 35 ? "#d86a3d" : "#5b86ff";
  const legendMetrics = buildLegendMetrics(data);
  const suspiciousShare = overall;
  const originalShare = Math.max(0, 100 - overall);
  const mediaTitle = getMediaLabel(data.media_type).replace(/^./, (char) => char.toUpperCase());

  return `
    <section class="forensic-report">
      <div class="forensic-report-head">
        <div class="forensic-report-kicker">Classification</div>
      </div>
      <div class="forensic-report-layout">
        <div class="forensic-gauge-panel">
          <div class="forensic-gauge-title">AI Generated Probability</div>
          <div class="forensic-gauge" style="--forensic-score:${overall};--gauge-color:${gaugeColor};">
            <div class="forensic-gauge-inner">${Math.round(overall)}%</div>
          </div>
          <div class="forensic-verdict">${escapeHtml(getForensicVerdict(data))}</div>
        </div>

        <div class="forensic-section">
          <h4 class="forensic-section-title">Probability Breakdown</h4>
          <p class="forensic-section-copy">The probabilities of the audio and video fakeness classifiers</p>
          <div class="forensic-breakdown-bar">
            <div class="forensic-breakdown-segment" style="width:${suspiciousShare}%;background:#4a89ff;">${suspiciousShare >= 12 ? `${Math.round(suspiciousShare)}%` : ""}</div>
            <div class="forensic-breakdown-segment original" style="width:${originalShare}%;">${originalShare >= 12 ? `${Math.round(originalShare)}%` : ""}</div>
          </div>
          <div class="forensic-legend">
            ${legendMetrics.map((metric) => `
              <div class="forensic-legend-item">
                <span class="forensic-legend-swatch" style="background:${metric.color};"></span>
                <span>${escapeHtml(metric.label)}</span>
              </div>
            `).join("")}
          </div>
        </div>

        <div class="forensic-section">
          <h4 class="forensic-section-title">${escapeHtml(mediaTitle)} Summary</h4>
          <div class="forensic-summary-copy">${escapeHtml(getForensicSummary(data))}</div>
        </div>
      </div>
    </section>
  `;
}

async function handleFullFileSelection(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }

  const statusEl = document.getElementById("file-scan-status");
  const resultsEl = document.getElementById("forensic-results");
  statusEl.textContent = "Extracting temporal and audio data. Please wait.";
  resultsEl.classList.add("hidden");
  resultsEl.textContent = "";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_URL}/analyze-full-media`, {
      method: "POST",
      body: formData
    });
    const data = await safeJson(response);

    if (!response.ok) {
      statusEl.textContent = `Error: ${data.detail || "Could not analyze the selected file."}`;
      return;
    }

    statusEl.textContent = "";
    resultsEl.classList.remove("hidden");

    resultsEl.innerHTML = buildForensicResultsHtml(data);
  } catch (error) {
    statusEl.textContent = error.message || "Server error. Ensure backend is running.";
  }
}