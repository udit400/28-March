# Synthetic Media Shield

Synthetic Media Shield is a local demo stack for scanning visible video frames with a Chrome extension and a FastAPI backend.

## Project layout

```text
deepfake-shield/
├── backend/
│   ├── main.py
│   ├── model.pt
│   └── requirements.txt
├── extension/
│   ├── background.js
│   ├── content.js
│   ├── manifest.json
│   └── styles.css
└── README.md
```

## Backend behavior

- If `REALITY_DEFENDER_API_KEY` is configured, the API uses the Reality Defender SDK as the primary detection engine for higher-quality cloud-based media analysis.
- Cloud results now use all returned detector model scores as a consensus instead of trusting only one top-level value.
- If `backend/model.pt` exists and loads correctly, the API uses the PyTorch model.
- If the model file is missing or fails to load, the API falls back to a deterministic mock engine so the UI still works.
- `GET /` now serves the local product landing page.
- `GET /api/status` returns the machine-readable runtime summary.
- `GET /health` returns engine status for a quick smoke test.
- SQLite-backed OTP login and premium subscription state are stored in `backend/shield_saas.db`.

## SaaS flow

- Open the extension popup from the browser toolbar.
- Request an OTP using a Gmail address.
- The backend now sends a real OTP through Gmail SMTP and requires valid Gmail SMTP credentials.
- After OTP verification, activate premium with a UTR/reference number to unlock scans.
- Scan results now include forensic metadata such as EXIF presence, software tags, device hints, and AI provenance markers.
- Premium users also get a popup-based forensic sandbox for full video, full audio, and image uploads.

## Full forensic sandbox

- Use the extension popup after premium activation.
- In the dashboard, select a full `.mp4`, `.mov`, `.avi`, `.mp3`, `.wav`, `.jpg`, `.jpeg`, or `.png` file.
- The backend performs file-type aware analysis:
	- Images: metadata plus visual deepfake scan
	- Audio: full audio clone risk simulation
	- Video: sampled temporal frame analysis, multi-frame score consensus, biological liveness estimation, and audio track extraction when available
- The response includes an overall threat score and a forensic summary.

## SMTP configuration

- Set `SMS_SHIELD_SMTP_EMAIL` to your Gmail address before starting the backend.
- Set `SMS_SHIELD_SMTP_PASSWORD` to a Gmail App Password. Do not use your normal Gmail password.
- Default Gmail SMTP settings are already wired: `SMS_SHIELD_SMTP_SERVER=smtp.gmail.com` and `SMS_SHIELD_SMTP_PORT=587`.
- Optional variables: `SMS_SHIELD_JWT_SECRET`, `SMS_SHIELD_OTP_SECRET`, `SMS_SHIELD_OTP_EXPIRY_MINUTES`, `SMS_SHIELD_OTP_RESEND_COOLDOWN_SECONDS`, and `SMS_SHIELD_OTP_MAX_ATTEMPTS`.
- OTP login accepts only `@gmail.com` inboxes and no longer exposes a local preview code.

## Reality Defender integration

- Install dependencies with `pip install -r requirements.txt` to include the Reality Defender Python SDK.
- Set `REALITY_DEFENDER_API_KEY` to enable the cloud detection engine.
- Optional tuning variables:
	- `REALITY_DEFENDER_MAX_ATTEMPTS`
	- `REALITY_DEFENDER_POLLING_INTERVAL_MS`
	- `REALITY_DEFENDER_SEQUENCE_FRAME_LIMIT`
- When the API key is missing or the SDK cannot initialize, the backend falls back to the local PyTorch or mock detector automatically.
- Supported social links can be routed directly through Reality Defender for better link analysis quality.

## Local startup

1. Open a terminal in `deepfake-shield/backend`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the API with `uvicorn main:app --reload`.
4. Open `chrome://extensions`, enable Developer mode, and load the `deepfake-shield/extension` folder.

## Quick launch scripts

- Run `deepfake-shield/start-backend.ps1` to start the FastAPI service from the project virtual environment.
- Run `deepfake-shield/launch-extension.ps1` to open Chrome, Edge, or Brave with the unpacked extension preloaded into a dedicated browser profile and land on the local demo page.

## Reliable demo flow

- Open `http://127.0.0.1:8000/demo` in the browser that has the extension loaded.
- Load a local video file into the page.
- Click the extension's `SCAN MEDIA` button over the video.
- This demo path avoids the browser security restrictions that often block canvas capture on sites like YouTube, TikTok, and X.

## Online link flow

- Paste a direct public media URL into the demo page.
- For direct MP4 or WebM links, click `Load link`, then use the extension overlay to scan the current frame.
- For direct JPG or PNG links, click `Analyze image URL` and the backend will fetch and score the image itself.
- Remote URL analysis is restricted to public `http(s)` hosts and capped to small files for safety.

## Optional model dependencies

- The demo runs immediately with the mock engine and does not require PyTorch.
- If you want to load a real model later, install `backend/requirements-model.txt` in a Python version supported by PyTorch, then place `model.pt` in `backend/`.

## Real model swap

Drop your trained file at `backend/model.pt` or set the `DEEPFAKE_MODEL_PATH` environment variable before starting the API.