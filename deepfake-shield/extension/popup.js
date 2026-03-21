const API_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await restoreView();
});

function bindEvents() {
  document.getElementById("btn-get-otp").addEventListener("click", requestOtp);
  document.getElementById("btn-verify-otp").addEventListener("click", verifyOtp);
  document.getElementById("btn-verify-pay").addEventListener("click", verifyPayment);
  document.getElementById("btn-logout").addEventListener("click", logout);
  document.getElementById("btn-back-login").addEventListener("click", () => {
    resetToLogin();
  });
  document.getElementById("btn-reset-pay").addEventListener("click", resetToLogin);
  document.getElementById("btn-demo-utr").addEventListener("click", () => {
    document.getElementById("utr-input").value = "123456789012";
    setStatus("Demo UTR inserted. Click Activate subscription.");
  });
}

async function restoreView() {
  const data = await storageGet(["userEmail", "hasSub"]);
  updateAccountLabels(data.userEmail || "");
  if (data.userEmail && data.hasSub) {
    showView("view-dashboard");
    return;
  }
  if (data.userEmail) {
    showView("view-pay");
    return;
  }
  showView("view-login");
}

function showView(viewId) {
  ["view-login", "view-otp", "view-pay", "view-dashboard"].forEach((id) => {
    document.getElementById(id).classList.toggle("active", id === viewId);
  });
}

function setStatus(message) {
  document.getElementById("status").textContent = message || "";
}

function updateAccountLabels(email) {
  const fallback = email ? `Account: ${email}` : "No account selected yet.";
  document.getElementById("otp-email-chip").textContent = fallback;
  document.getElementById("pay-email-chip").textContent = fallback;
  document.getElementById("dashboard-email-chip").textContent = fallback;
}

async function resetToLogin() {
  await chrome.storage.local.remove(["userEmail", "hasSub", "authToken"]);
  document.getElementById("email-input").value = "";
  document.getElementById("otp-input").value = "";
  document.getElementById("utr-input").value = "";
  updateAccountLabels("");
  showView("view-login");
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

async function requestOtp() {
  const email = document.getElementById("email-input").value.trim().toLowerCase();
  if (!email) {
    setStatus("Enter your email first.");
    return;
  }

  setBusy("btn-get-otp", true);
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

    await storageSet({ userEmail: email, hasSub: false });
    updateAccountLabels(email);
    showView("view-otp");
    if (payload.delivery_mode === "dev-preview" && payload.otp_preview) {
      document.getElementById("otp-input").value = payload.otp_preview;
      setStatus(`SMTP is not configured. Demo OTP inserted automatically: ${payload.otp_preview}`);
      return;
    }
    setStatus("OTP sent to your email.");
  } catch (error) {
    setStatus(error.message || "Could not contact the backend.");
  } finally {
    setBusy("btn-get-otp", false);
  }
}

async function verifyOtp() {
  const otp = document.getElementById("otp-input").value.trim();
  const { userEmail } = await storageGet(["userEmail"]);
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
      authToken: payload.token || null
    });
    updateAccountLabels(payload.email);
    showView(payload.has_subscription ? "view-dashboard" : "view-pay");
    setStatus(payload.has_subscription ? "Premium already active." : "Login successful. Activate premium to unlock scanning.");
  } catch (error) {
    setStatus(error.message || "Could not verify OTP.");
  } finally {
    setBusy("btn-verify-otp", false);
  }
}

async function verifyPayment() {
  const utrNumber = document.getElementById("utr-input").value.trim();
  const { userEmail } = await storageGet(["userEmail"]);
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

    await storageSet({ userEmail, hasSub: true, authToken: payload.token || null });
    updateAccountLabels(userEmail);
    showView("view-dashboard");
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