const API_BASE_URL = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "shield-analyze-frame" || !message.imageDataUrl) {
    return false;
  }

  analyzeFrame(message.imageDataUrl)
    .then((data) => sendResponse({ ok: true, data }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));

  return true;
});

async function analyzeFrame(imageDataUrl) {
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